"""Five-epoch, two-fold V7 CHO-GO pilot.

Nothing is written while training.  Only after every requested fold completes
successfully are compact results written to /content and copied once to Drive.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CFG, apply_runtime_profile
from data_pipeline import (ColdStartIndexDataset, ColdStartStationDataset,
    build_or_load_hourly_cube, fit_train_only_scaler, haversine_matrix,
    load_static, make_meta_crossfit_folds, standardize_static)
from model_v7 import ContextHeldOutGradientOperator
from train_formal import (DeviceFeatureBuilder, amp_dtype_for, make_grad_scaler,
    regression_metrics, seed_all)
from v5_training import event_metrics
from v7_training import V7BatchAdapter, fixed_hash_groups, forward_v7

REVISION = "v7_chogo_context_held_out_gradient_operator_1"


def _target_index(static, value):
    hit=np.flatnonzero((static.siteid.astype(str).to_numpy()==str(value)) | (static.sitename.astype(str).to_numpy()==str(value)))
    if len(hit)!=1: raise ValueError(f"outer target {value!r} matched {len(hit)} stations")
    return int(hit[0])


def _raw_batch(dataset, indices):
    return {"target_idx":torch.as_tensor(dataset.row_targets[indices],dtype=torch.long),
            "time_idx":torch.as_tensor(dataset.row_times[indices],dtype=torch.long)}


def _station_weights(dataset, station_count, device):
    counts=np.bincount(dataset.row_targets,minlength=station_count).astype("float64")
    active=counts>0; weights=np.zeros(station_count,dtype="float32")
    weights[active]=len(dataset)/(active.sum()*counts[active])
    return torch.as_tensor(weights,device=device)


def _metrics(y,p,s,static):
    overall=regression_metrics(y,p); rows=[]
    for station in np.unique(s):
        keep=s==station; row={"station_index":int(station),"siteid":str(static.loc[station,"siteid"]),
            "sitename":str(static.loc[station,"sitename"]),"n":int(keep.sum()),**regression_metrics(y[keep],p[keep])}
        rows.append(row)
    frame=pd.DataFrame(rows)
    overall["macro_rmse"]=float(frame.rmse.mean());overall["macro_r2"]=float(frame.r2.mean())
    overall["mean_abs_bias"]=float(frame.bias.abs().mean());overall.update(event_metrics(y,p))
    return overall,frame


@torch.inference_mode()
def validate(model,dataset,builder,adapter,static,timestamps,batch_size,device,description):
    model.eval();dtype=amp_dtype_for(device);ys=[];ps=[];ss=[];tt=[];supported=[]
    for start in tqdm(range(0,len(dataset),batch_size),desc=description,unit="batch",dynamic_ncols=True,leave=False):
        idx=np.arange(start,min(start+batch_size,len(dataset)))
        batch=adapter.prepare(builder(_raw_batch(dataset,idx)))
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None): pred,aux=forward_v7(model,batch)
        if not torch.isfinite(pred).all(): raise RuntimeError("V7 validation non-finite prediction")
        ys.append(batch["label"].float().cpu().numpy());ps.append(pred.float().cpu().numpy())
        ss.append(batch["target_idx"].cpu().numpy());tt.append(batch["time_idx"].cpu().numpy())
        supported.append(aux["donor_supported"].cpu().numpy())
    y,p,s,ti,sup=map(np.concatenate,(ys,ps,ss,tt,supported))
    metrics,station=_metrics(y,p,s,static);metrics["donor_supported_coverage"]=float(sup.mean())
    predictions=pd.DataFrame({"station_index":s.astype(int),"timestamp":pd.DatetimeIndex(timestamps[ti]),
        "y_true":y.astype("float32"),"y_pred":p.astype("float32"),"donor_supported":sup.astype(bool)})
    return metrics,station,predictions


def _sanity(model,batch):
    model.train();model.zero_grad(set_to_none=True);pred,aux=forward_v7(model,batch)
    loss=(pred.float()-batch["label"].float()).square().mean();loss.backward()
    if not torch.isfinite(pred).all() or not torch.isfinite(loss): raise RuntimeError("V7 sanity non-finite")
    required=("static.","point.","temporal.","graph.","query.")
    for prefix in required:
        gradients=[p.grad for n,p in model.named_parameters() if n.startswith(prefix)]
        if not gradients or all(g is None or float(g.abs().sum())==0 for g in gradients):
            raise RuntimeError(f"V7 sanity missing gradient: {prefix}")
        if any(g is not None and not torch.isfinite(g).all() for g in gradients): raise RuntimeError(f"V7 sanity nonfinite gradient: {prefix}")
    if not torch.all(aux["candidate_weights"].sum((1,2))[aux["donor_supported"]].sub(1).abs()<1e-5):
        raise RuntimeError("V7 candidate weights do not sum to one")
    return {"loss":float(loss.detach().cpu()),"parameters":sum(p.numel() for p in model.parameters()),
            "candidate_count_min":int(aux["candidate_count"].min().item())}


def run_fold(fold,train_idx,val_idx,cube,timestamps,static,cols,distance,device,epochs,batch_size,seed):
    scaler=fit_train_only_scaler(cube,timestamps,static,cols,train_idx); scaled=standardize_static(static,cols,scaler)
    train_ds=ColdStartStationDataset(train_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,static,scaled,distance,scaler)
    val_ds=ColdStartStationDataset(val_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,static,scaled,distance,scaler)
    hidden=np.setdiff1d(np.arange(len(static)),train_idx)
    max_time=max(int(train_ds.row_times.max()),int(val_ds.row_times.max()))
    builder=DeviceFeatureBuilder(train_idx,cube,max_time,timestamps,static,scaled,distance,scaler,hidden,device)
    adapter=V7BatchAdapter(static,timestamps,distance,scaler,device)
    model=ContextHeldOutGradientOperator(len(cols),adapter.pm_mean,adapter.pm_std,dropout=.1).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4,fused=device.type=="cuda")
    grad_scaler=make_grad_scaler(False);dtype=amp_dtype_for(device);weights=_station_weights(train_ds,len(static),device)
    rng=np.random.default_rng(seed+fold*1009);probe=rng.choice(len(train_ds),size=min(2,len(train_ds)),replace=False)
    sanity_batch=adapter.prepare(builder(_raw_batch(train_ds,probe)));sanity=_sanity(model,sanity_batch);optimizer.zero_grad(set_to_none=True)
    print(json.dumps({"fold":fold,"train_samples":len(train_ds),"validation_samples":len(val_ds),
        "model_parameters":sanity["parameters"],"sanity":sanity,
        "feature_precompute_seconds":builder.precompute_seconds,"feature_tables_mb":builder.precomputed_table_mb},ensure_ascii=False),flush=True)
    steps=int(os.environ.get("V7_STEPS_PER_EPOCH","400"));history=[];best=float("inf");best_payload=None
    best_station=None;best_predictions=None;started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();torch.cuda.reset_peak_memory_stats(device);epoch_start=time.perf_counter();loss_sum=0.0;seen=0
        bar=tqdm(range(steps),desc=f"V7 fold {fold} epoch {epoch}/{epochs}",unit="batch",dynamic_ncols=True)
        for _ in bar:
            idx=rng.choice(len(train_ds),size=batch_size,replace=False)
            batch=adapter.prepare(builder(_raw_batch(train_ds,idx)))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None):
                pred,_=forward_v7(model,batch)
                sample_w=weights[batch["target_idx"]]
                loss=torch.mean(sample_w*(pred.float()-batch["label"].float()).square())
            if not torch.isfinite(loss): raise RuntimeError("V7 non-finite training loss")
            grad_scaler.scale(loss).backward();grad_scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            grad_scaler.step(optimizer);grad_scaler.update()
            n=len(idx);loss_sum+=float(loss.detach().cpu())*n;seen+=n
            bar.set_postfix(loss=f"{loss_sum/seen:.3f}")
        validation_batch=int(os.environ.get("V7_VALIDATION_BATCH_SIZE",str(batch_size)))
        metrics,station,predictions=validate(model,val_ds,builder,adapter,static,timestamps,validation_batch,device,
            f"V7 fold {fold} validation {epoch}")
        row={"epoch":epoch,"train_loss":loss_sum/seen,**metrics,"runtime_seconds":time.perf_counter()-epoch_start,
            "gpu_peak_vram_mb":torch.cuda.max_memory_allocated(device)/1024**2}
        history.append(row);print(f"fold {fold} epoch {epoch}: macroRMSE={metrics['macro_rmse']:.4f} R2={metrics['r2']:.4f} absBias={metrics['mean_abs_bias']:.4f}",flush=True)
        if metrics["macro_rmse"]<best:
            best=metrics["macro_rmse"];best_station=station.copy();best_predictions=predictions.copy()
            best_payload={"revision":REVISION,"fold":fold,"epoch":epoch,"metrics":dict(metrics),
                "model":copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}),
                "train_indices":train_idx.copy(),"validation_indices":val_idx.copy(),"sanity":sanity}
    return {"fold":fold,"best_epoch":int(best_payload["epoch"]),"best_metrics":best_payload["metrics"],
        "runtime_seconds":time.perf_counter()-started,"history":pd.DataFrame(history),"station":best_station,
        "predictions":best_predictions,"checkpoint":best_payload,"sanity":sanity,
        "precompute_seconds":builder.precompute_seconds,"precomputed_mb":builder.precomputed_table_mb}


def _write_completed(results,root,settings):
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True,exist_ok=True);compact=[]
    for result in results:
        out=root/f"fold_{result['fold']:02d}";out.mkdir(parents=True,exist_ok=True)
        result["history"].to_csv(out/"training_history.csv",index=False,encoding="utf-8-sig")
        result["station"].to_csv(out/"best_validation_station_metrics.csv",index=False,encoding="utf-8-sig")
        result["predictions"].to_csv(out/"best_validation_predictions.csv.gz",index=False,compression="gzip")
        torch.save(result["checkpoint"],out/"best_checkpoint.pt")
        compact.append({k:v for k,v in result.items() if k not in {"history","station","predictions","checkpoint"}})
    comparison=pd.DataFrame([{"fold":r["fold"],"method":"v7_chogo","best_epoch":r["best_epoch"],
        **r["best_metrics"],"runtime_seconds":r["runtime_seconds"]} for r in results])
    comparison.to_csv(root/"pilot_comparison.csv",index=False,encoding="utf-8-sig")
    aggregate={"mean_macro_rmse":float(comparison.macro_rmse.mean()),
        "mean_pooled_r2":float(comparison.r2.mean()),
        "mean_absolute_bias":float(comparison.mean_abs_bias.mean()),
        "mean_donor_supported_coverage":float(comparison.donor_supported_coverage.mean()),
        "total_runtime_seconds":float(comparison.runtime_seconds.sum())}
    summary={"status":"done","revision":REVISION,"settings":settings,"results":compact,
        "aggregate":aggregate,
        "files_kept":["pilot_summary.json","pilot_comparison.csv","per-fold best checkpoint/history/station metrics/compressed predictions"]}
    (root/"pilot_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# V7 CHO-GO pilot summary","","Status: completed",f"Revision: {REVISION}","","```",
        comparison.round(4).to_string(index=False),"```","","Aggregate:","",
        *[f"- {key}: {value:.4f}" for key,value in aggregate.items()],"",
        "This is an exploratory two-fold pilot. It does not establish outer-station performance."]
    (root/"pilot_summary.md").write_text("\n".join(lines),encoding="utf-8")
    return summary


def _sync_once(root):
    destination=Path(os.environ.get("V7_DRIVE_OUTPUT","/content/drive/MyDrive/DL_TCN_V7_CHOGO_PILOT"))
    if not Path("/content/drive/MyDrive").is_dir():
        print("Drive not mounted; completed compact output remains at",root,flush=True);return None
    temporary=destination.with_name(destination.name+"_complete_tmp")
    if temporary.exists(): shutil.rmtree(temporary)
    shutil.copytree(root,temporary)
    if destination.exists(): shutil.rmtree(destination)
    temporary.replace(destination)
    print("Completed output copied once to",destination,flush=True);return str(destination)


def main():
    os.environ.setdefault("DL_TCN_COMPILE_MODE","off");runtime=apply_runtime_profile(CFG)
    if CFG.device.type!="cuda": raise RuntimeError("V7 pilot requires a Colab CUDA GPU")
    torch.set_num_threads(min(12,os.cpu_count() or 1));torch.backends.cuda.matmul.allow_tf32=True
    epochs=int(os.environ.get("V7_EPOCHS","5"));batch_size=int(os.environ.get("V7_BATCH_SIZE","16"))
    folds=tuple(int(x) for x in os.environ.get("V7_FOLDS","0,3").split(","))
    root=Path(os.environ.get("V7_OUTPUT_ROOT","/content/DL_TCN_V7_CHOGO_PILOT"))
    if "/drive/" in str(root): raise RuntimeError("V7_OUTPUT_ROOT must be /content; Drive copy occurs only after completion")
    static,clusters,cols=load_static();outer=_target_index(static,CFG.target_site)
    splits=make_meta_crossfit_folds(clusters,outer);cube,timestamps=build_or_load_hourly_cube(static)
    distance=haversine_matrix(static.longitude,static.latitude)
    group_counts=np.bincount(fixed_hash_groups(static.siteid),minlength=4).tolist()
    settings={"revision":REVISION,"outer":{"index":outer,"siteid":str(static.loc[outer,"siteid"]),"sitename":str(static.loc[outer,"sitename"])},
        "folds":list(folds),"epochs":epochs,"batch_size":batch_size,"steps_per_epoch":int(os.environ.get("V7_STEPS_PER_EPOCH","400")),
        "train_period":[CFG.train_start,CFG.train_end],"history_hours":24,"dynamic_items":list(CFG.dynamic_items),
        "hash_group_counts":group_counts,"runtime":runtime,"write_policy":"nothing during training; one compact write+Drive sync after all folds finish"}
    print(json.dumps(settings,ensure_ascii=False,indent=2),flush=True);results=[]
    for fold in folds:
        train_idx,val_idx=splits[fold]
        print(f"START fold {fold}: train={len(train_idx)} validation={len(val_idx)} donors=59/60",flush=True)
        results.append(run_fold(fold,train_idx,val_idx,cube,timestamps,static,cols,distance,CFG.device,epochs,batch_size,CFG.seed))
        gc.collect();torch.cuda.empty_cache()
    settings["drive_output"]=os.environ.get("V7_DRIVE_OUTPUT","/content/drive/MyDrive/DL_TCN_V7_CHOGO_PILOT")
    summary=_write_completed(results,root,settings);summary["drive_output"]=_sync_once(root)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()

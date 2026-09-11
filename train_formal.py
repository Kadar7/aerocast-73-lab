from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    ColdStartIndexDataset,
    VectorizedFormalCollator,
    build_or_load_hourly_cube,
    choose_split,
    collate_variable_donors,
    fit_train_only_scaler,
    haversine_matrix,
    load_static,
    standardize_static,
    time_features,
)
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset, smoke_forward


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_target(static: pd.DataFrame, name: str) -> int:
    hit=np.flatnonzero(
        (static.siteid.astype(str).to_numpy()==str(name))
        | (static.sitename.astype(str).to_numpy()==str(name))
    )
    if len(hit)!=1:
        raise ValueError(f"target {name!r} 對到{len(hit)}站")
    return int(hit[0])


def make_loader(dataset, shuffle: bool, batch_size: int | None = None) -> DataLoader:
    kwargs={
        "dataset":dataset,
        "batch_size":batch_size or CFG.batch_size,
        "shuffle":shuffle,
        "num_workers":CFG.formal_num_workers,
        "pin_memory":CFG.device.type=="cuda",
        "collate_fn":collate_variable_donors,
    }
    if CFG.formal_num_workers>0:
        kwargs.update(persistent_workers=True,prefetch_factor=CFG.prefetch_factor)
    return DataLoader(**kwargs)


def make_vectorized_loader(source_dataset,donor_pool,cube,timestamps,static,static_scaled,distance,scaler,shuffle):
    index_dataset=ColdStartIndexDataset(source_dataset)
    collator=VectorizedFormalCollator(donor_pool,cube,timestamps,static,static_scaled,distance,scaler)
    return DataLoader(index_dataset,batch_size=CFG.batch_size,shuffle=shuffle,num_workers=0,
                      pin_memory=CFG.device.type=="cuda",collate_fn=collator)


def collate_indices(batch):
    return {"target_idx":torch.tensor([row[0] for row in batch],dtype=torch.long),
            "time_idx":torch.tensor([row[1] for row in batch],dtype=torch.long)}


def make_index_loader(source_dataset,shuffle):
    kwargs={
        "dataset":ColdStartIndexDataset(source_dataset),"batch_size":CFG.batch_size,
        "shuffle":shuffle,"num_workers":CFG.formal_num_workers,
        "pin_memory":True,"collate_fn":collate_indices,
    }
    if CFG.formal_num_workers>0:
        kwargs.update(persistent_workers=True,prefetch_factor=CFG.prefetch_factor)
    return DataLoader(**kwargs)


class DeviceFeatureBuilder:
    """Gather reviewed features from compact, once-per-run CUDA tables."""
    def __init__(self,donor_pool,cube,max_time_index,timestamps,static,static_scaled,distance,scaler,outer_index,device):
        precompute_started=time.perf_counter()
        cube_np=np.array(cube[:max_time_index+1],dtype="float32",copy=True)
        # Preserve target labels before hiding outer dynamic values. The formal
        # 60/12 trainer never requests outer labels; evaluate_outer.py does so
        # only after model selection is locked.
        pm25_index=CFG.aq_cube_items.index("PM2.5")
        labels_np=np.array(cube_np[:,:,pm25_index],dtype="float32",copy=True)
        # Accept either one ordinary LOSO target or a complete held-out group.
        # Every excluded station is hidden before GPU feature tables are built.
        hidden=np.atleast_1d(np.asarray(outer_index,dtype=int))
        cube_np[:,hidden,:]=np.nan
        cube_tensor=torch.from_numpy(cube_np).to(device)
        del cube_np
        donor_pool=np.asarray(donor_pool,dtype="int64")
        self.static_scaled=torch.as_tensor(static_scaled,dtype=torch.float32,device=device)
        self.dynamic_mean=torch.as_tensor(scaler.dynamic_mean,dtype=torch.float32,device=device)
        self.dynamic_std=torch.as_tensor(scaler.dynamic_std,dtype=torch.float32,device=device)
        raw_indices=torch.tensor([CFG.aq_cube_items.index(x) for x in CFG.raw_dynamic_items],dtype=torch.long,device=device)
        self.raw_channel_indices=torch.arange(len(CFG.raw_dynamic_items),dtype=torch.long,device=device)
        self.speed_index=CFG.aq_cube_items.index("WIND_SPEED"); self.direction_index=CFG.aq_cube_items.index("WIND_DIREC")
        self.pm25_index=pm25_index; self.device=device
        lon=static.longitude.to_numpy(float); lat=static.latitude.to_numpy(float)
        from data_pipeline import bearing_degrees
        bearings=np.stack([bearing_degrees(lon,lat,lon[target],lat[target]) for target in range(len(static))]).astype("float32")
        max_donors=len(donor_pool); donor_table=np.full((len(static),max_donors),-1,dtype="int64")
        donor_counts=np.zeros(len(static),dtype="int64"); geometry_table=np.zeros((len(static),max_donors,3),dtype="float32")
        donor_static_table=np.zeros((len(static),max_donors,static_scaled.shape[1]),dtype="float32")
        for target in range(len(static)):
            donors=donor_pool[donor_pool!=target]; count=len(donors); donor_counts[target]=count; donor_table[target,:count]=donors
            bearing=bearings[target,donors]; sin_b=np.sin(np.deg2rad(bearing)).astype("float32"); cos_b=np.cos(np.deg2rad(bearing)).astype("float32")
            geometry_table[target,:count]=np.stack([np.log1p(distance[target,donors]/1000.0),sin_b,cos_b],axis=-1).astype("float32")
            donor_static_table[target,:count]=static_scaled[donors]
        self.donor_table=torch.from_numpy(donor_table).to(device)
        self.donor_counts=torch.from_numpy(donor_counts).to(device)
        self.geometry_table=torch.from_numpy(geometry_table).to(device)
        self.donor_static_table=torch.from_numpy(donor_static_table).to(device)
        self.history_offsets=torch.arange(CFG.history_hours,device=device)-CFG.history_hours+1
        self.zero=torch.zeros((),device=device,dtype=torch.float32)
        tf=np.stack([time_features(ts) for ts in timestamps[:max_time_index+1]]).astype("float32")
        self.time_feature_table=torch.from_numpy(tf).to(device)

        # Labels and nine raw channels are target-independent. Normalize once.
        self.labels=torch.from_numpy(labels_np).to(device)
        del labels_np
        raw=cube_tensor.index_select(2,raw_indices)
        self.raw_mask=torch.isfinite(raw)
        self.raw_values=(raw-self.dynamic_mean[:len(CFG.raw_dynamic_items)])/self.dynamic_std[:len(CFG.raw_dynamic_items)]
        self.raw_values=torch.where(self.raw_mask,self.raw_values,self.zero)
        del raw

        # Target-relative wind is pair-specific but fixed for the whole run.
        # Store hourly pair tables only; 24h sample windows remain on-demand.
        n_time,n_station,_=cube_tensor.shape
        self.wind_values=torch.empty((n_time,n_station,n_station,2),dtype=torch.float32,device=device)
        self.wind_mask=torch.empty((n_time,n_station,n_station,2),dtype=torch.bool,device=device)
        speed=cube_tensor[:,:,self.speed_index]
        wfrom=cube_tensor[:,:,self.direction_index]
        bearing_all=torch.from_numpy(bearings).to(device)
        wind_mean=self.dynamic_mean[len(CFG.raw_dynamic_items):]
        wind_std=self.dynamic_std[len(CFG.raw_dynamic_items):]
        chunk=max(1,int(CFG.gpu_precompute_chunk_hours))
        for start in range(0,n_time,chunk):
            stop=min(start+chunk,n_time)
            sp=speed[start:stop]
            wd=wfrom[start:stop]
            valid=torch.isfinite(sp)&torch.isfinite(wd)
            delta=torch.deg2rad(torch.remainder(wd[:,None,:]+180.0,360.0)-bearing_all[None,:,:])
            pair=torch.stack([sp[:,None,:]*torch.cos(delta),sp[:,None,:]*torch.sin(delta)],dim=-1)
            pair=(pair-wind_mean)/wind_std
            pair_valid=valid[:,None,:,None].expand(-1,n_station,-1,2)
            self.wind_values[start:stop]=torch.where(pair_valid,pair,self.zero)
            self.wind_mask[start:stop]=pair_valid
            del delta,pair,pair_valid
        del speed,wfrom,bearing_all,cube_tensor
        if device.type=="cuda": torch.cuda.synchronize(device)
        self.precompute_seconds=time.perf_counter()-precompute_started
        self.precomputed_table_mb=(
            self.raw_values.numel()*self.raw_values.element_size()
            +self.raw_mask.numel()*self.raw_mask.element_size()
            +self.wind_values.numel()*self.wind_values.element_size()
            +self.wind_mask.numel()*self.wind_mask.element_size()
            +self.labels.numel()*self.labels.element_size()
        )/1024**2

    def __call__(self,index_batch):
        targets=index_batch["target_idx"].to(self.device,non_blocking=True)
        current=index_batch["time_idx"].to(self.device,non_blocking=True)
        counts=self.donor_counts[targets]
        if not torch.all(counts==counts[0]): raise RuntimeError("同一batch donor數不一致")
        donor_count=int(counts[0].item()); donors=self.donor_table[targets,:donor_count]
        history=current[:,None]+self.history_offsets[None,:]
        if int(history.min().item())<0: raise RuntimeError("AQ cube起點不足24h history")
        raw=self.raw_values[history[:,:,None,None],donors[:,None,:,None],self.raw_channel_indices[None,None,None,:]]
        raw_mask=self.raw_mask[history[:,:,None,None],donors[:,None,:,None],self.raw_channel_indices[None,None,None,:]]
        wind=self.wind_values[history[:,:,None],targets[:,None,None],donors[:,None,:]]
        wind_mask=self.wind_mask[history[:,:,None],targets[:,None,None],donors[:,None,:]]
        values=torch.cat([raw,wind],dim=-1).permute(0,2,1,3).contiguous()
        mask=torch.cat([raw_mask,wind_mask],dim=-1).permute(0,2,1,3).to(torch.float32).contiguous()
        all_missing=mask.sum(dim=(2,3))==0
        geometry=self.geometry_table[targets,:donor_count]
        labels=self.labels[current,targets]
        if not torch.isfinite(labels).all(): raise RuntimeError("index batch包含缺失target label")
        return {
            "values":values,"mask":mask,"donor_static":self.donor_static_table[targets,:donor_count],"geometry":geometry,
            "donor_padding_mask":torch.zeros(donors.shape,dtype=torch.bool,device=self.device),
            "donor_all_missing":all_missing,"target_static":self.static_scaled[targets],
            "time_features":self.time_feature_table[current],"label":labels,
            "target_idx":targets,"time_idx":current,"donor_indices":donors,
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key:(value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) else value)
        for key,value in batch.items()
    }


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda",enabled=enabled,init_scale=CFG.amp_init_scale,growth_interval=CFG.amp_growth_interval)
    except (AttributeError,TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled,init_scale=CFG.amp_init_scale,growth_interval=CFG.amp_growth_interval)


def amp_dtype_for(device: torch.device):
    if device.type!="cuda" or not CFG.use_amp:
        return None
    if CFG.prefer_bf16 and torch.cuda.get_device_capability(0)[0]>=8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_optimizer(model,device):
    kwargs={"lr":CFG.learning_rate,"weight_decay":CFG.weight_decay}
    if device.type=="cuda":
        try:
            return torch.optim.AdamW(model.parameters(),fused=True,**kwargs),True
        except (TypeError,RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(),**kwargs),False


def regression_metrics(y_true: np.ndarray,y_pred: np.ndarray) -> dict[str,float]:
    y_true=np.asarray(y_true,dtype="float64"); y_pred=np.asarray(y_pred,dtype="float64")
    err=y_pred-y_true
    mae=float(np.mean(np.abs(err)))
    rmse=float(np.sqrt(np.mean(np.square(err))))
    bias=float(np.mean(err))
    denom=float(np.sum(np.square(y_true-y_true.mean())))
    r2=float(1.0-np.sum(np.square(err))/denom) if denom>0 else float("nan")
    return {"mae":mae,"rmse":rmse,"r2":r2,"bias":bias}


def make_station_sample_weights(dataset,station_count: int,device: torch.device) -> torch.Tensor:
    """Weights make the expected epoch objective the mean of station MSEs."""
    counts=np.bincount(np.asarray(dataset.row_targets,dtype="int64"),minlength=station_count)
    active=np.flatnonzero(counts>0)
    if active.size==0:
        raise RuntimeError("training dataset沒有任何target samples")
    weights=np.zeros(station_count,dtype="float32")
    weights[active]=len(dataset)/(active.size*counts[active])
    sample_mean=float(weights[np.asarray(dataset.row_targets,dtype="int64")].mean())
    if not np.isclose(sample_mean,1.0,rtol=1e-6,atol=1e-6):
        raise RuntimeError(f"station-balanced weights未正規化: mean={sample_mean}")
    return torch.as_tensor(weights,dtype=torch.float32,device=device)


def train_epoch(model,loader,optimizer,grad_scaler,device,epoch: int,feature_builder=None,station_sample_weights=None) -> float:
    model.train(); amp_dtype=amp_dtype_for(device); amp_enabled=amp_dtype is not None
    loss_sum=0.0; sample_count=0; started=time.perf_counter()
    for batch_number,batch in enumerate(loader,start=1):
        batch=feature_builder(batch) if feature_builder is not None else move_batch(batch,device)
        if batch_number==1 and device.type=="cuda":
            if batch["values"].device.type!="cuda" or next(model.parameters()).device.type!="cuda":
                raise RuntimeError("CUDA available但model或batch沒有移到GPU")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):
            pred,_=model(
                batch["values"],batch["mask"],batch["donor_static"],batch["geometry"],
                batch["donor_padding_mask"],batch["target_static"],batch["time_features"],
            )
            squared_error=torch.square(pred-batch["label"])
            if station_sample_weights is None:
                raise RuntimeError("station-balanced loss缺少training station weights")
            sample_weights=station_sample_weights[batch["target_idx"]]
            loss=torch.mean(squared_error*sample_weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"epoch {epoch} batch {batch_number}: loss非finite")
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CFG.gradient_clip_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"epoch {epoch} batch {batch_number}: gradient norm非finite")
        grad_scaler.step(optimizer); grad_scaler.update()
        n=int(batch["label"].numel()); loss_sum+=float(loss.detach().cpu())*n; sample_count+=n
        if CFG.progress_every_batches>0 and batch_number%CFG.progress_every_batches==0:
            elapsed=time.perf_counter()-started
            rate=sample_count/max(elapsed,1e-9)
            print(f"  epoch {epoch} train {batch_number:,}/{len(loader):,} batches | {sample_count:,} samples | {rate:,.1f} samples/s",flush=True)
    if sample_count!=len(loader.dataset):
        raise RuntimeError(f"training coverage錯誤: {sample_count} != {len(loader.dataset)}")
    return loss_sum/sample_count


@torch.no_grad()
def validate_epoch(model,loader,device,static,timestamps,epoch: int,feature_builder=None):
    model.eval(); amp_dtype=amp_dtype_for(device); amp_enabled=amp_dtype is not None
    truths=[]; predictions=[]; station_indices=[]; time_indices=[]; seen=0; started=time.perf_counter()
    for batch_number,batch in enumerate(loader,start=1):
        batch=feature_builder(batch) if feature_builder is not None else move_batch(batch,device)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):
            pred,_=model(
                batch["values"],batch["mask"],batch["donor_static"],batch["geometry"],
                batch["donor_padding_mask"],batch["target_static"],batch["time_features"],
            )
        if not torch.isfinite(pred).all():
            raise RuntimeError(f"epoch {epoch} validation prediction非finite")
        truths.append(batch["label"].float().cpu().numpy())
        predictions.append(pred.float().cpu().numpy())
        station_indices.append(batch["target_idx"].cpu().numpy())
        time_indices.append(batch["time_idx"].cpu().numpy())
        seen+=int(batch["label"].numel())
        if CFG.progress_every_batches>0 and batch_number%CFG.progress_every_batches==0:
            elapsed=time.perf_counter()-started
            print(f"  epoch {epoch} valid {batch_number:,}/{len(loader):,} batches | {seen:,} samples | {seen/max(elapsed,1e-9):,.1f} samples/s",flush=True)
    if seen!=len(loader.dataset):
        raise RuntimeError(f"validation coverage錯誤: {seen} != {len(loader.dataset)}")
    y=np.concatenate(truths); p=np.concatenate(predictions); s=np.concatenate(station_indices); ti=np.concatenate(time_indices)
    overall=regression_metrics(y,p); station_rows=[]
    for station in sorted(np.unique(s)):
        keep=s==station; metrics=regression_metrics(y[keep],p[keep])
        station_rows.append({
            "station_index":int(station),"siteid":str(static.loc[station,"siteid"]),
            "sitename":str(static.loc[station,"sitename"]),"n":int(keep.sum()),**metrics,
        })
    overall["macro_station_rmse"]=float(np.mean([row["rmse"] for row in station_rows]))
    prediction_frame=pd.DataFrame({
        "station_index":s.astype(int),
        "timestamp":pd.DatetimeIndex(timestamps[ti]),
        "y_true":y.astype("float32"),
        "y_pred":p.astype("float32"),
    })
    return overall,pd.DataFrame(station_rows),prediction_frame


def cpu_state_dict(model) -> dict:
    return {name:tensor.detach().cpu() for name,tensor in model.state_dict().items()}


def serializable_config() -> dict:
    result={}
    for key,value in asdict(CFG).items():
        if isinstance(value,Path): result[key]=str(value)
        elif isinstance(value,torch.device): result[key]=str(value)
        else: result[key]=value
    return result


def main() -> None:
    seed_all(CFG.seed)
    device=CFG.device
    runtime_profile=apply_runtime_profile(CFG)
    torch.set_num_threads(min(os.cpu_count() or 1,12))
    if device.type=="cuda":
        if CFG.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.allow_tf32=True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark=True
    output_dir=CFG.formal_output_root/CFG.formal_output_dirname
    output_dir.mkdir(parents=True,exist_ok=True)
    if not CFG.aq_data_dir.is_dir(): raise FileNotFoundError(f"AQ資料夾不存在: {CFG.aq_data_dir}")

    static,clusters,static_cols=load_static()
    outer=resolve_target(static,CFG.target_site)
    train_idx,val_idx=choose_split(clusters,outer)
    if outer in train_idx or outer in val_idx: raise RuntimeError("outer target leakage")
    cube,timestamps=build_or_load_hourly_cube(static)
    distance=haversine_matrix(static.longitude,static.latitude)
    scaler=fit_train_only_scaler(cube,timestamps,static,static_cols,train_idx)
    static_scaled=standardize_static(static,static_cols,scaler)
    train_ds=ColdStartStationDataset(train_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,static,static_scaled,distance,scaler)
    val_ds=ColdStartStationDataset(val_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,static,static_scaled,distance,scaler)
    if any(len(train_idx[train_idx!=target])!=59 for target in train_idx): raise RuntimeError("training donor protocol不是59")
    if any(len(train_idx)!=60 for _ in val_idx): raise RuntimeError("validation donor protocol不是60")

    model=TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    parameter_count=sum(parameter.numel() for parameter in model.parameters())
    gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (CPU training)"
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU name: {gpu_name}")
    print(f"runtime profile: {json.dumps(runtime_profile,ensure_ascii=False)}")
    print(f"train sample count: {len(train_ds):,}")
    print(f"validation sample count: {len(val_ds):,}")
    print(f"model parameters: {parameter_count:,}")
    print(f"device: {device} | batch size: {CFG.batch_size} | workers: {CFG.formal_num_workers} | max epochs: {CFG.max_epochs} | patience: {CFG.early_stopping_patience}",flush=True)

    # Mandatory pre-training gate; it does not construct/read an outer dataset.
    sanity={"train":audit_dataset(train_ds),"validation":audit_dataset(val_ds)}
    smoke=smoke_forward(model,make_loader(train_ds,False,CFG.smoke_batch_size),device)
    model.zero_grad(set_to_none=True)
    if sanity["train"]["sampled_donor_counts"]!=[59] or sanity["validation"]["sampled_donor_counts"]!=[60]:
        raise RuntimeError("sanity donor count未通過")
    (output_dir/"pretraining_sanity.json").write_text(json.dumps({**sanity,"model":smoke},ensure_ascii=False,indent=2),encoding="utf-8")
    print("PRE-TRAINING SANITY PASSED",flush=True)

    feature_builder=None
    if device.type=="cuda":
        max_time_index=int(max(train_ds.row_times.max(),val_ds.row_times.max()))
        feature_builder=DeviceFeatureBuilder(train_idx,cube,max_time_index,timestamps,static,static_scaled,distance,scaler,outer,device)
        runtime_profile["feature_precompute_seconds"]=feature_builder.precompute_seconds
        runtime_profile["precomputed_feature_tables_mb"]=feature_builder.precomputed_table_mb
        print(f"GPU feature precompute: {feature_builder.precompute_seconds:.1f}s | tables={feature_builder.precomputed_table_mb:.1f}MiB",flush=True)
        train_loader=make_index_loader(train_ds,True); val_loader=make_index_loader(val_ds,False)
    else:
        train_loader=make_vectorized_loader(train_ds,train_idx,cube,timestamps,static,static_scaled,distance,scaler,True)
        val_loader=make_vectorized_loader(val_ds,train_idx,cube,timestamps,static,static_scaled,distance,scaler,False)
    optimizer,fused_optimizer=make_optimizer(model,device)
    amp_dtype=amp_dtype_for(device)
    grad_scaler=make_grad_scaler(amp_dtype==torch.float16)
    print(f"AMP dtype: {str(amp_dtype).replace('torch.','') if amp_dtype else 'disabled'} | fused AdamW: {fused_optimizer} | TF32: {CFG.enable_tf32 and device.type=='cuda'}",flush=True)
    station_sample_weights=make_station_sample_weights(train_ds,len(static),device)
    print(f"loss: {CFG.loss_name} | balanced training stations: {(station_sample_weights>0).sum().item()}",flush=True)
    best_macro=math.inf; best_epoch=0; best_metrics=None; epochs_without_improvement=0
    history=[]; all_station_metrics=[]; total_start=time.perf_counter(); overall_peak_vram=0.0
    checkpoint_path=output_dir/"best_checkpoint.pt"
    prediction_path=output_dir/"validation_predictions.csv"
    station_metric_path=output_dir/"best_validation_station_metrics.csv"
    all_station_metric_path=output_dir/"validation_station_metrics_all_epochs.csv"
    history_path=output_dir/"training_history.csv"

    for epoch in range(1,CFG.max_epochs+1):
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        epoch_start=time.perf_counter()
        train_loss=train_epoch(model,train_loader,optimizer,grad_scaler,device,epoch,feature_builder,station_sample_weights)
        metrics,station_metrics,predictions=validate_epoch(model,val_loader,device,static,timestamps,epoch,feature_builder)
        epoch_seconds=time.perf_counter()-epoch_start
        peak_vram=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else 0.0
        overall_peak_vram=max(overall_peak_vram,peak_vram)
        improved=metrics["macro_station_rmse"]<best_macro
        row={"epoch":epoch,"train_loss":train_loss,**{f"validation_{k}":v for k,v in metrics.items()},"epoch_runtime_seconds":epoch_seconds,"gpu_peak_vram_mb":peak_vram,"is_best":improved}
        history.append(row); pd.DataFrame(history).to_csv(history_path,index=False,encoding="utf-8-sig")
        print(
            f"epoch {epoch}: train_loss={train_loss:.6f} | val MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} R2={metrics['r2']:.4f} Bias={metrics['bias']:.4f} "
            f"MacroRMSE={metrics['macro_station_rmse']:.4f} | {epoch_seconds:.1f}s | peak={peak_vram:.1f}MiB",
            flush=True,
        )
        if improved:
            best_macro=metrics["macro_station_rmse"]; best_epoch=epoch; best_metrics=metrics
            epochs_without_improvement=0
            checkpoint={
                "epoch":epoch,"model_state_dict":cpu_state_dict(model),"optimizer_state_dict":optimizer.state_dict(),
                "validation_metrics":metrics,"train_indices":train_idx,"validation_indices":val_idx,
                "outer_index_excluded":outer,"static_columns":static_cols,
                "scaler":{
                    "dynamic_mean":scaler.dynamic_mean,"dynamic_std":scaler.dynamic_std,
                    "static_mean":scaler.static_mean,"static_std":scaler.static_std,"static_median":scaler.static_median,
                },
                "config":serializable_config(),"torch_version":torch.__version__,
            }
            torch.save(checkpoint,checkpoint_path)
            station_metrics.to_csv(station_metric_path,index=False,encoding="utf-8-sig")
            predictions.to_csv(prediction_path,index=False,encoding="utf-8-sig",date_format="%Y-%m-%d %H:%M:%S")
        else:
            epochs_without_improvement+=1
        station_epoch=station_metrics.copy()
        station_epoch.insert(0,"epoch",epoch)
        station_epoch["is_new_best"]=improved
        all_station_metrics.append(station_epoch)
        station_history=pd.concat(all_station_metrics,ignore_index=True)
        station_history["is_best"]=station_history["epoch"].eq(best_epoch)
        station_history.to_csv(all_station_metric_path,index=False,encoding="utf-8-sig")
        if epochs_without_improvement>=CFG.early_stopping_patience:
            print(f"EARLY STOPPING: {CFG.early_stopping_patience} epochs without MacroRMSE improvement",flush=True)
            break

    total_seconds=time.perf_counter()-total_start
    summary={
        "best_epoch":best_epoch,"best_validation_macro_rmse":best_macro,
        "best_validation_rmse":best_metrics["rmse"],"best_validation_r2":best_metrics["r2"],
        "best_validation_bias":best_metrics["bias"],"training_time_seconds":total_seconds,
        "peak_vram_mb":overall_peak_vram,"checkpoint":str(checkpoint_path),
        "epochs_completed":len(history),"device":str(device),"gpu_name":gpu_name,
        "runtime_profile":runtime_profile,"fused_adamw":fused_optimizer,
        "all_epoch_station_metrics":str(all_station_metric_path),
    }
    (output_dir/"training_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()

"""One-pass, read-only V4 diagnosis. No optimizer, checkpoint writes or outer evaluation."""
from pathlib import Path
from types import SimpleNamespace
import copy
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from diagnose_v4_background import decompose


def shuffled_dynamic(batch, permutation, pm_only=False):
    """Shuffle complete windows within the SAME target; retain query time/static."""
    if not torch.equal(batch['target_idx'], batch['target_idx'][permutation]):
        raise ValueError('Permutation crossed target stations')
    if not torch.equal(batch['donor_indices'], batch['donor_indices'][permutation]):
        raise ValueError('Permutation changed donor identities')
    out=dict(batch)
    if pm_only:
        out['values']=batch['values'].clone(); out['mask']=batch['mask'].clone()
        out['values'][...,7]=batch['values'][permutation,...,7]
        out['mask'][...,7]=batch['mask'][permutation,...,7]
    else:
        for key in ('values','mask','physical_wind'):
            out[key]=batch[key][permutation]
    return out


def prediction_rows(y,p,s,ti,static,background,fold,epoch,split):
    rows=[]
    for station in np.unique(s):
        keep=s==station
        if len(np.unique(ti[keep])) != int(keep.sum()):
            raise ValueError('Duplicate station timestamps')
        result=decompose(y[keep],p[keep],float(background[station]))
        variance=float(np.mean((y[keep]-y[keep].mean())**2))
        rows.append(dict(fold=fold,epoch=epoch,split=split,station_index=int(station),
                         siteid=str(static.loc[station,'siteid']),sitename=static.loc[station,'sitename'],
                         n=int(keep.sum()),r2=1-result['rmse']**2/variance if variance else np.nan,
                         **result))
    return rows


@torch.inference_mode()
def run(root='/content/DL_TCN_V4_VALIDATION_PILOT', samples_per_station=128):
    from config import CFG,apply_runtime_profile
    from data_pipeline import (load_static,standardize_static,haversine_matrix,
                               ColdStartStationDataset,fit_train_only_scaler)
    from train_formal import (DeviceFeatureBuilder,make_index_loader,make_vectorized_loader,
                              amp_dtype_for,make_loader,move_batch)
    from model_v2 import TCNTargetCrossAttentionV2
    from v2_training import build_v2_context,_forward
    from sanity_check import audit_dataset,assert_tcn_causal
    original=copy.deepcopy(vars(CFG))
    started=time.perf_counter(); rows=[]; probes=[]; audits=[]
    try:
        root=Path(root); cache=CFG.output_dir/'_cache'
        # Never call the cache builder: missing inputs must fail without writing.
        meta=json.loads((cache/'aq_hourly_meta.json').read_text(encoding='utf-8'))
        cube=np.load(cache/'aq_hourly_cube.npy',mmap_mode='r',allow_pickle=False)
        static,_,cols=load_static()
        if meta['siteids']!=static.siteid.astype(str).tolist() or meta['items']!=list(CFG.aq_cube_items) or list(cube.shape)!=meta['shape']:
            raise ValueError('Cache schema/station order mismatch')
        if len(cols)!=49: raise ValueError('Expected 49 static features')
        times=pd.date_range(meta['start'],periods=len(cube),freq='h')
        distance=haversine_matrix(static.longitude,static.latitude)
        device=CFG.device; torch.set_num_threads(min(12,__import__('os').cpu_count() or 1))
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        for fold in (0,3):
            folder=root/f'fold_{fold:02d}'
            history=pd.read_csv(folder/'validation_station_metrics_all_epochs.csv')
            epoch=int(history.groupby('epoch').rmse.mean().idxmin())
            ck=torch.load(folder/'epoch_checkpoints'/f'epoch_{epoch:03d}.pt',map_location='cpu',weights_only=False)
            if ck.get('v2_config',{}).get('training_revision')!='full_donor_nested_background_bounded_relation_v4':
                raise ValueError('Not a V4 checkpoint')
            if ck['static_columns']!=cols: raise ValueError('Static column mismatch')
            # Restore scientific/model settings, not obsolete machine filesystem paths.
            for key in ('train_start','train_end','history_hours','raw_dynamic_items','derived_dynamic_items',
                        'aq_cube_items','tcn_hidden','tcn_kernel_size','tcn_dilations','dropout',
                        'attention_dim','attention_heads','final_hidden','use_amp','prefer_bf16',
                        'batch_size','compile_mode','enable_tf32'):
                value=ck['config'][key]
                if isinstance(getattr(CFG,key),tuple): value=tuple(value)
                setattr(CFG,key,value)
            saved_batch=CFG.batch_size
            apply_runtime_profile(CFG); CFG.batch_size=saved_batch; CFG.formal_num_workers=0
            if device.type=='cuda':
                torch.backends.cuda.matmul.allow_tf32=CFG.enable_tf32
                torch.backends.cudnn.allow_tf32=CFG.enable_tf32
                torch.backends.cudnn.benchmark=True
                torch.set_float32_matmul_precision('high' if CFG.enable_tf32 else 'highest')
            if CFG.history_hours!=24 or CFG.n_dynamic_channels!=11: raise ValueError('History/channels mismatch')
            source=np.asarray(ck['train_indices'],int); val=np.asarray(ck['validation_indices'],int)
            excluded=np.asarray(ck['outer_indices_excluded'],int)
            if len(set(np.concatenate([source,val,excluded])))!=len(static) or len(source)+len(val)+len(excluded)!=len(static):
                raise ValueError('Split overlaps or omits stations')
            if times[0]>pd.Timestamp(CFG.train_start)-pd.Timedelta(hours=23) or times[-1]<pd.Timestamp(CFG.train_end):
                raise ValueError('Insufficient cached history')
            scaler=SimpleNamespace(**ck['scaler'])
            print(f'fold {fold}, epoch {epoch}: source-only scaler/background audit',flush=True)
            recomputed=fit_train_only_scaler(cube,times,static,cols,source)
            for name in ck['scaler']:
                np.testing.assert_allclose(getattr(scaler,name),getattr(recomputed,name),rtol=1e-5,atol=1e-6,err_msg=name)
            scaled=standardize_static(static,cols,scaler)
            adapter,bgaudit=build_v2_context(cube,times,scaled,distance,source,scaler,static,device)
            for key in ('ridge_lambda','static_weight','source_loo_rmse','relation_scales'):
                np.testing.assert_allclose(bgaudit[key],ck['background'][key],rtol=1e-5,atol=1e-5,err_msg=key)
            datasets={name:ColdStartStationDataset(targets,source,CFG.train_start,CFG.train_end,
                       cube,times,static,scaled,distance,scaler) for name,targets in [('training',source),('validation',val)]}
            model=TCNTargetCrossAttentionV2(len(cols)).to(device)
            model.load_state_dict(ck['model_state_dict'],strict=True); model.eval()
            assert_tcn_causal(model,device)
            # Replay the original execution mode, not an unconditionally eager model.
            if device.type=='cuda' and CFG.compile_mode!='off':
                model=torch.compile(model,mode=CFG.compile_mode,fullgraph=False,dynamic=False)
            print(f'  replay batch={CFG.batch_size}, compile={CFG.compile_mode}, AMP={amp_dtype_for(device)}',flush=True)
            builder=None
            if device.type=='cuda':
                builder=DeviceFeatureBuilder(source,cube,max(int(d.row_times.max()) for d in datasets.values()),
                                              times,static,scaled,distance,scaler,excluded,device)
            dtype=amp_dtype_for(device)
            def predict(batch):
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None):
                    pred,aux=_forward(model,batch)
                if not torch.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
                torch.testing.assert_close(pred.float(),batch['target_background_raw']+aux['residual'].float(),rtol=1e-5,atol=1e-5)
                return pred.float().cpu().numpy()
            for split,ds in datasets.items():
                audit=audit_dataset(ds,max_samples=12)
                expected=len(source)-(split=='training')
                if audit['sampled_donor_counts']!=[expected] or len(ds)!=ds.truth_rows:
                    raise ValueError('Donor count/coverage mismatch')
                audits.append(dict(fold=fold,split=split,n=len(ds),donors=expected,scaler_match=True,background_match=True,causal=True))
                loader=make_index_loader(ds,False) if builder else make_vectorized_loader(ds,source,cube,times,static,scaled,distance,scaler,False)
                ys=[]; ps=[]; ss=[]; ts=[]
                print(f'fold {fold}: {split} full inference, {len(ds):,} samples (no training)',flush=True)
                for number,raw in enumerate(loader,1):
                    batch=adapter.prepare(builder(raw) if builder else move_batch(raw,device))
                    pred=predict(batch)
                    ys.append(batch['label'].cpu().numpy()); ps.append(pred)
                    ss.append(batch['target_idx'].cpu().numpy()); ts.append(batch['time_idx'].cpu().numpy())
                    if number%100==0: print(f'  {number}/{len(loader)} batches',flush=True)
                y,p,s,ti=map(np.concatenate,(ys,ps,ss,ts))
                if len(y)!=len(ds): raise ValueError('Incomplete inference')
                if split=='validation':
                    with np.load(folder/'epoch_predictions'/f'epoch_{epoch:03d}.npz') as saved:
                        order=np.lexsort((ti,s)); old=np.lexsort((saved['timestamp_ns'],saved['station_index']))
                        np.testing.assert_array_equal(s[order],saved['station_index'][old])
                        np.testing.assert_array_equal(times[ti[order]].asi8,saved['timestamp_ns'][old])
                        np.testing.assert_array_equal(y[order],saved['y_true'][old])
                        diff=p[order]-saved['y_pred'][old]
                        delta=float(np.max(np.abs(diff)))
                        audits[-1].update(prediction_max_delta=delta,
                            prediction_rms_delta=float(np.sqrt(np.mean(diff**2))),
                            prediction_mean_delta=float(diff.mean()),
                            prediction_p99_delta=float(np.quantile(np.abs(diff),.99)),
                            prediction_replay_pass=delta<=0.1)
                        print(f'  prediction replay: {audits[-1]}',flush=True)
                        if delta>0.1:
                            # Keep useful diagnostic output, but do not silently accept mismatches.
                            print('WARNING: replay mismatch remains. Results are provisional; do not infer a training cause.',flush=True)
                background=adapter.predicted.cpu().numpy()
                rows.extend(prediction_rows(y,p,s,ti,static,background,fold,epoch,split))
                # Small, fixed random sample per station. Shift whole donor windows together.
                rng=np.random.default_rng(20260912+fold)
                for station in ds.targets:
                    indices=np.flatnonzero(ds.row_targets==station)
                    chosen=rng.choice(indices,min(samples_per_station,len(indices)),replace=False)
                    if len(chosen)<2: continue
                    subset=torch.utils.data.Subset(ds,chosen.tolist())
                    raw=next(iter(make_loader(subset,False,len(chosen))))
                    batch=adapter.prepare(move_batch(raw,device))
                    baseline=predict(batch); truth=batch['label'].cpu().numpy()
                    perm=torch.roll(torch.arange(len(chosen),device=device),1)
                    for pm_only in (True,False):
                        changed=predict(shuffled_dynamic(batch,perm,pm_only))
                        probes.append(dict(fold=fold,split=split,sitename=static.loc[station,'sitename'],
                            intervention='PM2.5 time shuffle' if pm_only else 'all dynamic time shuffle',n=len(chosen),
                            baseline_rmse=float(np.sqrt(np.mean((baseline-truth)**2))),
                            shuffled_rmse=float(np.sqrt(np.mean((changed-truth)**2))),
                            prediction_change_rms=float(np.sqrt(np.mean((changed-baseline)**2)))))
                del loader
            del builder,model,adapter,datasets
            gc.collect()
            if device.type=='cuda': torch.cuda.empty_cache()
        result={'stations':pd.DataFrame(rows),'probes':pd.DataFrame(probes),'audit':pd.DataFrame(audits),
                'seconds':time.perf_counter()-started,'files_written':0,
                'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0.}
        print(f"Completed: {result['seconds']:.1f}s; peak VRAM {result['peak_vram_mb']:.0f}MB; files written=0",flush=True)
        return result
    finally:
        for key,value in original.items(): setattr(CFG,key,value)


def show(result):
    import ipywidgets as w
    from IPython.display import display,clear_output
    frame=result['stations']; probes=result['probes'].copy()
    probes['rmse_increase']=probes.shuffled_rmse-probes.baseline_rmse
    print('Training 是樣本內診斷，不是泛化成績。時間打亂是敏感度對照，不是正式可部署結果。')
    display(result['audit'])
    audit=result['audit']
    if 'prediction_replay_pass' in audit and (audit.prediction_replay_pass.dropna()==False).any():
        print('STOP interpretation: checkpoint replay still differs. Show the audit above first; the remaining tables are provisional, not a verified reproduction.')
    summary=frame.groupby(['fold','split']).agg(stations=('siteid','size'),macro_rmse=('rmse','mean'),
              mean_abs_bias=('final_bias',lambda x:x.abs().mean()),
              worsened_bias_stations=('abs_bias_reduction',lambda x:int((x<0).sum())),
              mean_bias_reduction=('abs_bias_reduction','mean'))
    display(summary.round(3))
    display(probes.groupby(['split','intervention'])[['rmse_increase','prediction_change_rms']].mean().round(3))
    print('Training好、validation差：支持換站失準；兩邊都差：優先查擬合。打亂後變差代表利用時間訊號，不代表使用方式已正確。')
    choice=w.Dropdown(options=[('全部',None)]+[(name,name) for name in sorted(frame.sitename.unique())],description='測站：')
    output=w.Output()
    def render(change=None):
        with output:
            clear_output(wait=True)
            display((frame if choice.value is None else frame[frame.sitename==choice.value]).round(3).reset_index(drop=True))
            if choice.value is not None: display(probes[probes.sitename==choice.value].round(3))
    choice.observe(render,names='value'); display(choice,output); render()
    return choice

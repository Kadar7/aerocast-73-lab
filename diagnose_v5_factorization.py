"""Read-only V5 decomposition: background -> slow level -> final prediction."""
from __future__ import annotations
from pathlib import Path
import gc
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd
import torch


def _metrics(y,p):
    y=np.asarray(y,dtype='float64');p=np.asarray(p,dtype='float64');error=p-y
    denominator=float(np.sum((y-y.mean())**2))
    return {'rmse':float(np.sqrt(np.mean(error**2))),
            'mae':float(np.mean(np.abs(error))),
            'bias':float(np.mean(error)),
            'r2':float(1-np.sum(error**2)/denominator) if denominator else float('nan')}


def station_decomposition(y,background,slow_level,final,fast_anomaly):
    b=_metrics(y,background);s=_metrics(y,slow_level);f=_metrics(y,final)
    return {
        'truth_mean':float(np.mean(y)),
        'background_mean':float(np.mean(background)),
        'slow_level_mean':float(np.mean(slow_level)),
        'final_mean':float(np.mean(final)),
        'mean_slow_correction':float(np.mean(slow_level-background)),
        'mean_fast_anomaly':float(np.mean(fast_anomaly)),
        **{f'background_{key}':value for key,value in b.items()},
        **{f'slow_{key}':value for key,value in s.items()},
        **{f'final_{key}':value for key,value in f.items()},
        'slow_abs_bias_reduction_vs_background':abs(b['bias'])-abs(s['bias']),
        'final_abs_bias_reduction_vs_slow':abs(s['bias'])-abs(f['bias']),
        'slow_rmse_reduction_vs_background':b['rmse']-s['rmse'],
        'final_rmse_reduction_vs_slow':s['rmse']-f['rmse'],
        'fast_worsened_slow_bias':abs(f['bias'])>abs(s['bias']),
    }


def summarize(stations):
    rows=[]
    for (fold,method),part in stations.groupby(['fold','method'],sort=True):
        slow_bias_gain=float(part.slow_abs_bias_reduction_vs_background.mean())
        fast_bias_gain=float(part.final_abs_bias_reduction_vs_slow.mean())
        if slow_bias_gain<=0:
            branch='slow_mean_abs_bias_not_improved'
        elif fast_bias_gain<0:
            branch='fast_mean_abs_bias_increased_after_slow'
        else:
            branch='both_stage_mean_abs_bias_improved'
        rows.append({
            'fold':fold,'method':method,'stations':len(part),
            'background_macro_rmse':part.background_rmse.mean(),
            'slow_macro_rmse':part.slow_rmse.mean(),
            'final_macro_rmse':part.final_rmse.mean(),
            'background_mean_abs_bias':part.background_bias.abs().mean(),
            'slow_mean_abs_bias':part.slow_bias.abs().mean(),
            'final_mean_abs_bias':part.final_bias.abs().mean(),
            'slow_improved_background_bias_stations':int((part.slow_abs_bias_reduction_vs_background>0).sum()),
            'fast_improved_slow_bias_stations':int((part.final_abs_bias_reduction_vs_slow>0).sum()),
            'fast_worsened_slow_bias_stations':int(part.fast_worsened_slow_bias.sum()),
            'mean_slow_abs_bias_reduction':slow_bias_gain,
            'mean_fast_abs_bias_reduction':fast_bias_gain,
            'diagnostic_branch':branch,
        })
    return pd.DataFrame(rows)


@torch.inference_mode()
def run(root='/content/DL_TCN_V5_FACTOR_MLDG_PILOT'):
    from tqdm.auto import tqdm
    from config import CFG,apply_runtime_profile
    from data_pipeline import ColdStartStationDataset,haversine_matrix,load_static
    from model_v5 import FactorizedResidualKriging
    from run_v5_factorized_mldg_pilot import full_validation_context,target_means,raw_index_batch
    from train_formal import DeviceFeatureBuilder,amp_dtype_for
    from v5_training import V5BatchAdapter,forward_v5

    root=Path(root);started=time.perf_counter();rows=[]
    if not (root/'settings.json').exists(): raise FileNotFoundError(f'Missing {root / "settings.json"}')
    settings=json.loads((root/'settings.json').read_text(encoding='utf-8'))
    settings_hash=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    expected={'train_period':[CFG.train_start,CFG.train_end],'history_hours':CFG.history_hours,
              'dynamic_items':list(CFG.dynamic_items)}
    for key,value in expected.items():
        if settings.get(key)!=value: raise RuntimeError(f'Current config differs from saved V5 setting: {key}')
    architecture={'tcn_hidden':CFG.tcn_hidden,'tcn_kernel_size':CFG.tcn_kernel_size,
        'tcn_dilations':list(CFG.tcn_dilations),'attention_dim':CFG.attention_dim,
        'attention_heads':CFG.attention_heads,'final_hidden':CFG.final_hidden,'static_dim':49}
    if settings.get('architecture')!=architecture: raise RuntimeError('Current architecture differs from saved V5 settings')
    apply_runtime_profile(CFG);CFG.formal_num_workers=0
    if CFG.device.type=='cuda':
        torch.set_num_threads(min(12,os.cpu_count() or 1));torch.cuda.reset_peak_memory_stats()
    static,_,cols=load_static();cache=CFG.output_dir/'_cache';meta_path=cache/'aq_hourly_meta.json';cube_path=cache/'aq_hourly_cube.npy'
    if not meta_path.exists() or not cube_path.exists():
        raise FileNotFoundError('Read-only diagnosis requires the existing AQ cube cache; it will not rebuild/write it')
    fingerprint=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    fingerprints=settings.get('data_fingerprint',{})
    actual={'cube_meta':fingerprint(meta_path),'static':fingerprint(CFG.static_path),'clusters':fingerprint(CFG.cluster_path)}
    if fingerprints!=actual: raise RuntimeError('Current AQ/static/cluster data differ from saved V5 fingerprints')
    meta=json.loads(meta_path.read_text(encoding='utf-8'));cube=np.load(cube_path,mmap_mode='r',allow_pickle=False)
    timestamps=pd.date_range(meta['start'],periods=len(cube),freq='h')
    if meta['siteids']!=static.siteid.astype(str).tolist() or meta['items']!=list(CFG.aq_cube_items) or list(cube.shape)!=meta['shape']:
        raise RuntimeError('AQ cube schema/station order mismatch')
    distance=haversine_matrix(static.longitude,static.latitude);dtype=amp_dtype_for(CFG.device)
    for fold in settings['folds']:
        for method in settings['methods']:
            checkpoint=root/f'fold_{int(fold):02d}'/method/'best_checkpoint.pt'
            if not checkpoint.exists(): raise FileNotFoundError(f'Missing {checkpoint}')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved.get('settings_hash')!=settings_hash or int(saved.get('fold',-1))!=int(fold) or saved.get('method')!=method:
                raise RuntimeError(f'Checkpoint identity/settings mismatch: {checkpoint}')
            train_idx=np.asarray(saved['train_indices'],int);val_idx=np.asarray(saved['validation_indices'],int)
            if set(train_idx)&set(val_idx): raise RuntimeError('Training/validation station overlap')
            means=target_means(cube,timestamps,train_idx)
            context=full_validation_context(train_idx,cube,timestamps,static,cols,distance)
            dataset=ColdStartStationDataset(val_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,
                                             static,context.static_scaled,distance,context.scaler)
            hidden=np.setdiff1d(np.arange(len(static)),train_idx)
            builder=DeviceFeatureBuilder(train_idx,cube,int(dataset.row_times.max()),timestamps,static,
                context.static_scaled,distance,context.scaler,hidden,CFG.device)
            adapter=V5BatchAdapter(context.fit,context.scaler,static,CFG.device,context.relation_scales,means)
            model=FactorizedResidualKriging(len(cols)).to(CFG.device)
            model.load_state_dict(saved['model'],strict=True);model.eval()
            collected={key:[] for key in ('y','station','time','background','slow','final','fast')}
            step=CFG.batch_size
            progress=tqdm(range(0,len(dataset),step),desc=f'decompose {method} fold {fold}',dynamic_ncols=True)
            for start in progress:
                indices=np.arange(start,min(start+step,len(dataset)))
                batch=adapter.prepare(builder(raw_index_batch(dataset,indices)),need_target_mean=False)
                with torch.autocast(device_type=CFG.device.type,dtype=dtype,enabled=dtype is not None):
                    prediction,aux=forward_v5(model,batch)
                background=batch['target_background_raw'].float()
                slow_level=background+aux['slow_correction'].float()
                torch.testing.assert_close(prediction.float(),slow_level+aux['anomaly'].float(),rtol=1e-5,atol=1e-5)
                tensors={'y':batch['label'],'station':batch['target_idx'],'time':batch['time_idx'],
                         'background':background,
                         'slow':slow_level,'final':prediction.float(),'fast':aux['anomaly'].float()}
                for key,value in tensors.items(): collected[key].append(value.cpu().numpy())
            values={key:np.concatenate(value) for key,value in collected.items()}
            if len(values['y'])!=len(dataset) or len(dataset)!=dataset.truth_rows or not all(
                    np.isfinite(values[key]).all() for key in ('y','background','slow','final','fast')):
                raise RuntimeError('Incomplete or nonfinite V5 decomposition')
            for station in np.unique(values['station']):
                keep=values['station']==station
                if len(np.unique(values['time'][keep]))!=int(keep.sum()):
                    raise RuntimeError(f'Duplicate timestamps for station {station}')
                result=station_decomposition(values['y'][keep],values['background'][keep],
                    values['slow'][keep],values['final'][keep],values['fast'][keep])
                rows.append({'fold':int(fold),'method':method,'best_epoch':int(saved['epoch']),
                    'station_index':int(station),'siteid':str(static.loc[station,'siteid']),
                    'sitename':static.loc[station,'sitename'],'n':int(keep.sum()),**result})
            del model,builder,adapter,dataset,context
            gc.collect()
            if CFG.device.type=='cuda': torch.cuda.empty_cache()
    stations=pd.DataFrame(rows);summary=summarize(stations)
    result={'summary':summary,'stations':stations,'seconds':time.perf_counter()-started,
            'files_written':0,'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2 if CFG.device.type=='cuda' else 0.0}
    print(f"Completed in {result['seconds']:.1f}s; files written=0",flush=True)
    return result


def show(result):
    import ipywidgets as widgets
    from IPython.display import display,clear_output
    display(result['summary'].round(4))
    frame=result['stations'];options=[]
    for row in frame[['station_index','siteid','sitename']].drop_duplicates().sort_values('station_index').itertuples():
        options.append((f'{row.sitename} (siteid={row.siteid})',int(row.station_index)))
    choice=widgets.Dropdown(options=options,description='測站：',layout=widgets.Layout(width='450px'))
    output=widgets.Output()
    def render(change=None):
        with output:
            clear_output(wait=True)
            columns=['fold','method','best_epoch','sitename','truth_mean','background_mean','slow_level_mean','final_mean',
                     'background_bias','slow_bias','final_bias','background_rmse','slow_rmse','final_rmse',
                     'mean_slow_correction','mean_fast_anomaly','fast_worsened_slow_bias']
            display(frame[frame.station_index==choice.value][columns].round(4).reset_index(drop=True))
    choice.observe(render,names='value');display(choice,output);render();return choice


if __name__=='__main__':
    result=run();print(result['summary'].to_string(index=False))

"""Matched fold-0/3 pilot: V5-style control versus V6 anchored fast branch."""
from __future__ import annotations
from pathlib import Path
import gc,hashlib,json,math,os,time
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from config import CFG,apply_runtime_profile
from data_pipeline import ColdStartStationDataset,build_or_load_hourly_cube,haversine_matrix,load_static
from group_selector_protocol import make_outer_groups,make_selector_inner_folds
from model_v6 import AnchoredFactorizedResidualKriging
from run_v5_factorized_mldg_pilot import (build_contexts,full_validation_context,raw_index_batch,
                                          selected_rows,target_means)
from train_crossfit_snapshots import atomic_csv_save,atomic_text_save,atomic_torch_save,rng_payload,restore_rng
from train_formal import DeviceFeatureBuilder,amp_dtype_for,cpu_state_dict,make_optimizer,regression_metrics,seed_all
from v5_training import event_metrics,make_station_weights
from v6_training import V6BatchAdapter,forward_v6,ordinary_two_domain_step

REVISION='source_climatology_anchored_fast_v6'
SLOW_WEIGHT=1.0


@torch.inference_mode()
def validate(model,context,val_idx,cube,timestamps,static,distance,means,device,description):
    dataset=ColdStartStationDataset(val_idx,context.source,CFG.train_start,CFG.train_end,cube,timestamps,
        static,context.static_scaled,distance,context.scaler)
    hidden=np.setdiff1d(np.arange(len(static)),context.source)
    builder=DeviceFeatureBuilder(context.source,cube,int(dataset.row_times.max()),timestamps,static,
        context.static_scaled,distance,context.scaler,hidden,device)
    adapter=V6BatchAdapter(context.fit,context.scaler,static,device,context.relation_scales,means)
    collected={key:[] for key in ('y','p','station')};model.eval();dtype=amp_dtype_for(device)
    for start in tqdm(range(0,len(dataset),CFG.batch_size),desc=description,unit='batch',leave=False,dynamic_ncols=True):
        idx=np.arange(start,min(start+CFG.batch_size,len(dataset)))
        batch=adapter.prepare(builder(raw_index_batch(dataset,idx)),need_target_mean=False)
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None): prediction,_=forward_v6(model,batch)
        collected['y'].append(batch['label'].cpu().numpy());collected['p'].append(prediction.float().cpu().numpy())
        collected['station'].append(batch['target_idx'].cpu().numpy())
    y,p,station=(np.concatenate(collected[key]) for key in ('y','p','station'))
    if len(y)!=len(dataset) or len(dataset)!=dataset.truth_rows or not np.isfinite(p).all():
        raise RuntimeError('Validation coverage/finite check failed')
    overall=regression_metrics(y,p);rows=[]
    for target in np.unique(station):
        keep=station==target;metrics=regression_metrics(y[keep],p[keep]);background=float(context.fit.predicted[target])
        background_bias=background-float(y[keep].mean())
        rows.append({'station_index':int(target),'siteid':str(static.loc[target,'siteid']),
            'sitename':static.loc[target,'sitename'],'n':int(keep.sum()),'background':background,
            'background_bias':background_bias,'dl_worsened_background_bias':abs(metrics['bias'])>abs(background_bias),**metrics})
    overall['macro_rmse']=float(np.mean([row['rmse'] for row in rows]))
    overall['macro_r2']=float(np.mean([row['r2'] for row in rows]))
    overall['mean_abs_bias']=float(np.mean([abs(row['bias']) for row in rows]))
    overall['worsened_background_stations']=int(sum(row['dl_worsened_background_bias'] for row in rows))
    overall.update(event_metrics(y,p))
    del builder,adapter,dataset;gc.collect();torch.cuda.empty_cache()
    return overall,pd.DataFrame(rows)


def train_line(method,anchored,fold,train_idx,val_idx,contexts,val_context,means,cube,timestamps,
               static,cols,distance,root,device,settings_hash,line_number):
    out=root/f'fold_{fold:02d}'/method;out.mkdir(parents=True,exist_ok=True);summary_path=out/'summary.json'
    if summary_path.exists():
        result=json.loads(summary_path.read_text(encoding='utf-8'))
        if (result.get('settings_hash')!=settings_hash or result.get('method')!=method
                or int(result.get('fold',-1))!=fold or bool(result.get('anchored'))!=anchored):
            raise RuntimeError(f'Incompatible completed run: {out}')
        print(f'[{line_number}/4] {method} fold {fold}: already complete',flush=True);return result
    print(f'\n[{line_number}/4] START {method} fold {fold} | anchored={anchored}',flush=True)
    seed_all(CFG.seed+fold*1009);torch.cuda.reset_peak_memory_stats()
    model=AnchoredFactorizedResidualKriging(len(cols),anchored=anchored).to(device);optimizer,_=make_optimizer(model,device)
    history=[];best=float('inf');best_epoch=0;best_metrics={};start_epoch=1
    resume=out/'active_resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        if (state.get('settings_hash')!=settings_hash or state.get('method')!=method
                or int(state.get('fold',-1))!=fold or bool(state.get('anchored'))!=anchored):
            raise RuntimeError(f'Incompatible resume: {resume}')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        history=state['history'];best=state['best'];best_epoch=state['best_epoch'];best_metrics=state['best_metrics']
        start_epoch=int(state['epoch'])+1;print(f'  resume from epoch {start_epoch}',flush=True)
    expected=sum(len(ColdStartStationDataset([station],train_idx,CFG.train_start,CFG.train_end,cube,timestamps,
        static,val_context.static_scaled,distance,val_context.scaler)) for station in train_idx)
    for epoch in range(start_epoch,CFG.max_epochs+1):
        epoch_started=time.perf_counter();totals=[];covered=0;plans=[]
        order=[contexts[(epoch-1+i)%len(contexts)] for i in range(len(contexts))]
        for context in order:
            shard=(context.context_id+epoch-1)%5
            support=selected_rows(context.support_ds,shard);query=selected_rows(context.query_ds,shard)
            covered+=len(support)+len(query);rng=np.random.default_rng(CFG.seed+fold*100000+epoch*101+context.context_id)
            rng.shuffle(support);rng.shuffle(query);steps=max(1,math.ceil((len(support)+len(query))/CFG.batch_size))
            plans.append((context,support,query,np.array_split(support,steps),np.array_split(query,steps)))
        bar=tqdm(total=sum(len(plan[3]) for plan in plans),desc=f'[{line_number}/4] {method} f{fold} epoch {epoch}/{CFG.max_epochs}',
                 unit='batch',dynamic_ncols=True)
        for context,support,query,s_chunks,q_chunks in plans:
            bar.set_postfix_str(f'context {context.context_id+1}/5')
            s_weights=make_station_weights(context.support_ds.row_targets[support],len(static),device)
            q_weights=make_station_weights(context.query_ds.row_targets[query],len(static),device)
            hidden=np.setdiff1d(np.arange(len(static)),context.source)
            max_time=max(int(context.support_ds.row_times.max()),int(context.query_ds.row_times.max()))
            builder=DeviceFeatureBuilder(context.source,cube,max_time,timestamps,static,context.static_scaled,
                distance,context.scaler,hidden,device)
            adapter=V6BatchAdapter(context.fit,context.scaler,static,device,context.relation_scales,means)
            for sidx,qidx in zip(s_chunks,q_chunks):
                sb=adapter.prepare(builder(raw_index_batch(context.support_ds,sidx)))
                qb=adapter.prepare(builder(raw_index_batch(context.query_ds,qidx)))
                totals.append(ordinary_two_domain_step(model,sb,qb,optimizer,s_weights,q_weights,
                    SLOW_WEIGHT,amp_dtype_for(device)));bar.update(1)
            del builder,adapter;gc.collect();torch.cuda.empty_cache()
        bar.close()
        if covered!=expected: raise RuntimeError(f'Epoch coverage {covered} != {expected}')
        metrics,station=validate(model,val_context,val_idx,cube,timestamps,static,distance,means,device,
                                  f'validate {method} f{fold} e{epoch}')
        row={'epoch':epoch,'train_support_loss':float(torch.stack([x['support_loss'] for x in totals]).mean().cpu()),
             'train_query_loss':float(torch.stack([x['query_loss'] for x in totals]).mean().cpu()),**metrics,
             'runtime_seconds':time.perf_counter()-epoch_started,
             'gpu_peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2}
        history.append(row);atomic_csv_save(pd.DataFrame(history),out/'training_history.csv')
        if metrics['macro_rmse']<best:
            best=metrics['macro_rmse'];best_epoch=epoch;best_metrics=dict(metrics)
            atomic_torch_save({'revision':REVISION,'method':method,'anchored':anchored,'fold':fold,'epoch':epoch,
                'model':cpu_state_dict(model),'train_indices':train_idx,'validation_indices':val_idx,
                'metrics':metrics,'settings_hash':settings_hash},out/'best_checkpoint.pt')
            atomic_csv_save(station,out/'best_validation_station_metrics.csv')
        atomic_torch_save({'revision':REVISION,'method':method,'anchored':anchored,'fold':fold,'epoch':epoch,'model':cpu_state_dict(model),
            'optimizer':optimizer.state_dict(),'rng':rng_payload(),'history':history,'best':best,'best_epoch':best_epoch,
            'best_metrics':best_metrics,'settings_hash':settings_hash},resume)
        print(f'  epoch {epoch}: macroRMSE={metrics["macro_rmse"]:.4f} R2={metrics["r2"]:.4f} '
              f'absBias={metrics["mean_abs_bias"]:.4f} recall={metrics["event_recall"]:.4f} '
              f'F1={metrics["event_f1"]:.4f} time={row["runtime_seconds"]:.1f}s '
              f'VRAM={row["gpu_peak_vram_mb"]:.0f}MB',flush=True)
    result={'revision':REVISION,'method':method,'anchored':anchored,'fold':fold,'best_epoch':best_epoch,
        'best_macro_rmse':best,'best_metrics':best_metrics,'runtime_seconds':float(sum(x['runtime_seconds'] for x in history)),
        'peak_vram_mb':float(max(x['gpu_peak_vram_mb'] for x in history)),'settings_hash':settings_hash,
        'files_kept':['training_history.csv','best_checkpoint.pt','best_validation_station_metrics.csv','summary.json']}
    atomic_text_save(summary_path,json.dumps(result,ensure_ascii=False,indent=2));resume.unlink(missing_ok=True)
    del model,optimizer;gc.collect();torch.cuda.empty_cache();return result


def show_results(root='/content/DL_TCN_V6_ANCHORED_PILOT'):
    import ipywidgets as widgets
    from IPython.display import display,clear_output
    root=Path(root);comparison=pd.read_csv(root/'pilot_comparison.csv');display(comparison.round(4))
    choice=widgets.Dropdown(options=[(f'{row.method} / fold {row.fold}',(row.method,int(row.fold)))
        for row in comparison[['method','fold']].itertuples(index=False)],description='結果：',layout=widgets.Layout(width='450px'))
    output=widgets.Output()
    def render(change=None):
        with output:
            clear_output(wait=True);method,fold=choice.value
            history=pd.read_csv(root/f'fold_{fold:02d}'/method/'training_history.csv')
            station=pd.read_csv(root/f'fold_{fold:02d}'/method/'best_validation_station_metrics.csv')
            display(history.round(4));display(station.sort_values('rmse').round(4).reset_index(drop=True))
    choice.observe(render,names='value');display(choice,output);render();return choice


def main():
    os.environ.setdefault('DL_TCN_MAX_EPOCHS','15');CFG.max_epochs=int(os.environ['DL_TCN_MAX_EPOCHS'])
    runtime=apply_runtime_profile(CFG);CFG.compile_mode='off';CFG.formal_num_workers=0
    if CFG.device.type!='cuda' or amp_dtype_for(CFG.device) is not torch.bfloat16:
        raise RuntimeError('V6 pilot requires BF16-capable A100/L4 GPU')
    torch.set_num_threads(min(12,os.cpu_count() or 1));torch.backends.cuda.matmul.allow_tf32=True
    root=Path(os.environ.get('DL_TCN_V6_PILOT_ROOT','/content/DL_TCN_V6_ANCHORED_PILOT'))
    if '/content/drive/' in str(root): raise RuntimeError('Pilot output must stay in /content, not Drive')
    root.mkdir(parents=True,exist_ok=True);static,clusters,cols=load_static();excluded=make_outer_groups(clusters)[0]
    splits=make_selector_inner_folds(clusters,excluded);cube,timestamps=build_or_load_hourly_cube(static)
    distance=haversine_matrix(static.longitude,static.latitude)
    files=[Path(__file__).with_name(name) for name in ('run_v6_anchored_pilot.py','model_v6.py','v6_training.py',
        'run_v5_factorized_mldg_pilot.py','v5_training.py','model.py','model_v2.py','v2_training.py',
        'background_crossfit.py','data_pipeline.py','train_formal.py',
        'group_selector_protocol.py','config.py')]
    code_hash=hashlib.sha256(b''.join(path.read_bytes() for path in files)).hexdigest()
    fingerprint=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cache_meta=CFG.output_dir/'_cache'/'aq_hourly_meta.json'
    settings={'revision':REVISION,'code_hash':code_hash,'seed':CFG.seed,'folds':[0,3],
        'methods':{'matched_control':False,'anchored_fast':True},'epochs':CFG.max_epochs,'batch_size':CFG.batch_size,
        'lr':CFG.learning_rate,'weight_decay':CFG.weight_decay,'dropout':CFG.dropout,
        'gradient_clip':CFG.gradient_clip_norm,'slow_weight':SLOW_WEIGHT,
        'architecture':{'tcn_hidden':CFG.tcn_hidden,'tcn_kernel_size':CFG.tcn_kernel_size,
            'tcn_dilations':list(CFG.tcn_dilations),'attention_dim':CFG.attention_dim,
            'attention_heads':CFG.attention_heads,'final_hidden':CFG.final_hidden,'static_dim':len(cols)},
        'train_period':[CFG.train_start,CFG.train_end],'history_hours':CFG.history_hours,
        'dynamic_items':list(CFG.dynamic_items),'runtime':{'gpu':runtime.get('gpu_name'),'amp':'bfloat16'},
        'data_fingerprint':{'cube_meta':fingerprint(cache_meta),'static':fingerprint(CFG.static_path),
                            'clusters':fingerprint(CFG.cluster_path)},
        'splits':{str(f):{'train':splits[f][0].tolist(),'validation':splits[f][1].tolist()} for f in (0,3)}}
    settings_hash=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest();marker=root/'settings.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))!=settings:
        raise RuntimeError('V6 settings/code changed; use a fresh output root')
    atomic_text_save(marker,json.dumps(settings,ensure_ascii=False,indent=2));summaries=[];line=0
    for fold in (0,3):
        train_idx,val_idx=splits[fold];means=target_means(cube,timestamps,train_idx)
        print(f'\nFold {fold}: building 5 strict source-only contexts',flush=True)
        contexts=build_contexts(train_idx,clusters,cube,timestamps,static,cols,distance,means)
        val_context=full_validation_context(train_idx,cube,timestamps,static,cols,distance)
        for method,anchored in settings['methods'].items():
            line+=1;summaries.append(train_line(method,anchored,fold,train_idx,val_idx,contexts,val_context,means,
                cube,timestamps,static,cols,distance,root,CFG.device,settings_hash,line))
    comparison=pd.DataFrame([{'fold':x['fold'],'method':x['method'],'best_epoch':x['best_epoch'],
        **x['best_metrics'],'runtime_seconds':x['runtime_seconds'],'peak_vram_mb':x['peak_vram_mb']} for x in summaries])
    atomic_csv_save(comparison,root/'pilot_comparison.csv')
    atomic_text_save(root/'pilot_summary.json',json.dumps(summaries,ensure_ascii=False,indent=2))
    print(f'\nDONE: {root}',flush=True);print(comparison.to_string(index=False),flush=True)


if __name__=='__main__':main()

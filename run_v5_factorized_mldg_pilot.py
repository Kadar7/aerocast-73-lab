"""V5 two-fold pilot: matched episodic control versus first-order MLDG."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import gc,hashlib,json,math,os,random,time
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from config import CFG,apply_runtime_profile
from data_pipeline import (ColdStartStationDataset,build_or_load_hourly_cube,
    fit_train_only_scaler,haversine_matrix,load_static,standardize_static)
from group_selector_protocol import make_outer_groups,make_selector_inner_folds
from model_v5 import FactorizedResidualKriging
from train_crossfit_snapshots import atomic_csv_save,atomic_text_save,atomic_torch_save,rng_payload,restore_rng
from train_formal import (DeviceFeatureBuilder,amp_dtype_for,cpu_state_dict,make_optimizer,
                          regression_metrics,seed_all)
from v2_training import fit_background,relation_scales
from v5_training import (V5BatchAdapter,balanced_partitions,first_order_mldg_step,
                         forward_v5,make_station_weights,event_metrics)

REVISION='factorized_residual_kriging_fo_mldg_v5'
BETA=1.0
SLOW_WEIGHT=1.0


@dataclass
class Context:
    context_id:int; source:np.ndarray; query:np.ndarray; scaler:object
    static_scaled:np.ndarray; fit:object; relation_scales:np.ndarray
    support_ds:object; query_ds:object


def target_means(cube,timestamps,stations):
    ti=np.flatnonzero((timestamps>=pd.Timestamp(CFG.train_start))&(timestamps<=pd.Timestamp(CFG.train_end)))
    pm=CFG.aq_cube_items.index('PM2.5'); result=np.full(cube.shape[1],np.nan,dtype='float32')
    result[stations]=np.nanmean(np.asarray(cube[np.ix_(ti,stations,[pm])])[...,0],axis=0)
    if not np.isfinite(result[stations]).all(): raise RuntimeError('Training station has no PM2.5 mean')
    return result


def build_contexts(train_idx,clusters,cube,timestamps,static,cols,distance,means):
    groups=balanced_partitions(train_idx,clusters,5,CFG.seed+505)
    contexts=[]
    for cid,query in enumerate(groups):
        print(f'  context {cid+1}/5: fitting strict source-only preprocessing',flush=True)
        source=np.setdiff1d(train_idx,query)
        scaler=fit_train_only_scaler(cube,timestamps,static,cols,source)
        scaled=standardize_static(static,cols,scaler)
        fit=fit_background(cube,timestamps,scaled,distance,source)
        scales=relation_scales(scaled,distance,source,scaler)
        support=ColdStartStationDataset(source,source,CFG.train_start,CFG.train_end,cube,timestamps,static,scaled,distance,scaler)
        qds=ColdStartStationDataset(query,source,CFG.train_start,CFG.train_end,cube,timestamps,static,scaled,distance,scaler)
        contexts.append(Context(cid,source,query,scaler,scaled,fit,scales,support,qds))
    return contexts


def selected_rows(dataset,shard):
    return np.flatnonzero(np.asarray(dataset.row_times)%5==shard)


def raw_index_batch(dataset,indices):
    return {'target_idx':torch.as_tensor(dataset.row_targets[indices],dtype=torch.long),
            'time_idx':torch.as_tensor(dataset.row_times[indices],dtype=torch.long)}


@torch.inference_mode()
def validate(model,context,val_idx,cube,timestamps,static,distance,means,excluded,device,epoch):
    ds=ColdStartStationDataset(val_idx,context.source,CFG.train_start,CFG.train_end,cube,timestamps,
                               static,context.static_scaled,distance,context.scaler)
    hidden=np.setdiff1d(np.arange(len(static)),context.source)
    builder=DeviceFeatureBuilder(context.source,cube,int(ds.row_times.max()),timestamps,static,
        context.static_scaled,distance,context.scaler,hidden,device)
    adapter=V5BatchAdapter(context.fit,context.scaler,static,device,context.relation_scales,means)
    ys=[];ps=[];ss=[]; step=CFG.batch_size
    model.eval(); dtype=amp_dtype_for(device)
    for start in range(0,len(ds),step):
        idx=np.arange(start,min(start+step,len(ds)))
        batch=adapter.prepare(builder(raw_index_batch(ds,idx)),need_target_mean=False)
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=dtype is not None): pred,_=forward_v5(model,batch)
        ys.append(batch['label'].cpu().numpy()); ps.append(pred.float().cpu().numpy()); ss.append(batch['target_idx'].cpu().numpy())
    y,p,s=map(np.concatenate,(ys,ps,ss))
    if len(y)!=len(ds) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise RuntimeError('Validation coverage or finite prediction check failed')
    overall=regression_metrics(y,p); rows=[]
    for station in np.unique(s):
        keep=s==station; station_metrics=regression_metrics(y[keep],p[keep])
        background=float(context.fit.predicted[station]); background_bias=background-float(y[keep].mean())
        rows.append({'station_index':int(station),'siteid':str(static.loc[station,'siteid']),
            'sitename':static.loc[station,'sitename'],'n':int(keep.sum()),'background':background,
            'background_bias':background_bias,'dl_worsened_background_bias':abs(station_metrics['bias'])>abs(background_bias),
            **station_metrics})
    overall['macro_rmse']=float(np.mean([r['rmse'] for r in rows])); overall['macro_r2']=float(np.mean([r['r2'] for r in rows]))
    overall['mean_abs_bias']=float(np.mean([abs(r['bias']) for r in rows])); overall.update(event_metrics(y,p))
    overall['worsened_background_stations']=int(sum(r['dl_worsened_background_bias'] for r in rows))
    del builder,adapter,ds; gc.collect(); torch.cuda.empty_cache()
    return overall,pd.DataFrame(rows)


def full_validation_context(train_idx,cube,timestamps,static,cols,distance):
    scaler=fit_train_only_scaler(cube,timestamps,static,cols,train_idx); scaled=standardize_static(static,cols,scaler)
    fit=fit_background(cube,timestamps,scaled,distance,train_idx); scales=relation_scales(scaled,distance,train_idx,scaler)
    return Context(-1,np.asarray(train_idx),np.array([],int),scaler,scaled,fit,scales,None,None)


def train_method(name,alpha,fold,train_idx,val_idx,excluded,contexts,val_context,means,
                 cube,timestamps,static,cols,distance,root,device,settings_hash):
    out=root/f'fold_{fold:02d}'/name; out.mkdir(parents=True,exist_ok=True)
    complete=out/'summary.json'
    if complete.exists():
        saved=json.loads(complete.read_text())
        if saved.get('settings_hash')!=settings_hash: raise RuntimeError(f'{out}: incompatible completed run')
        print(f'{name} fold {fold}: complete, skip',flush=True); return saved
    seed_all(CFG.seed+fold*1009)
    model=FactorizedResidualKriging(len(cols)).to(device); optimizer,_=make_optimizer(model,device)
    history=[]; start_epoch=1; best=float('inf'); best_epoch=0; best_metrics={};preflight={}
    resume=out/'active_resume.pt'
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        if state.get('settings_hash')!=settings_hash or state.get('method')!=name or state.get('fold')!=fold:
            raise RuntimeError(f'{resume}: incompatible resume state')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng']); history=state['history']
        start_epoch=int(state['epoch'])+1; best=float(state['best']); best_epoch=int(state['best_epoch'])
        best_metrics=state.get('best_metrics',{});preflight=state.get('preflight',{})
        print(f'{name} fold {fold}: resume epoch {start_epoch}',flush=True)
    expected=sum(len(ColdStartStationDataset([s],train_idx,CFG.train_start,CFG.train_end,cube,timestamps,
        static,val_context.static_scaled,distance,val_context.scaler)) for s in train_idx)
    started=time.perf_counter(); preflight_seconds=0.0
    # A real dry run: no model/optimizer mutation, and RNG is restored before training.
    if alpha>0 and start_epoch==1:
        preflight_started=time.perf_counter()
        context=contexts[0]; shard=0; si=selected_rows(context.support_ds,shard);qi=selected_rows(context.query_ds,shard)
        probe_rng=np.random.default_rng(CFG.seed+fold*100000+101);probe_rng.shuffle(si);probe_rng.shuffle(qi)
        steps=max(20,math.ceil((len(si)+len(qi))/CFG.batch_size));s_chunks=np.array_split(si,steps);q_chunks=np.array_split(qi,steps)
        sw=make_station_weights(context.support_ds.row_targets[si],len(static),device)
        qw=make_station_weights(context.query_ds.row_targets[qi],len(static),device)
        hidden=np.setdiff1d(np.arange(len(static)),context.source)
        builder=DeviceFeatureBuilder(context.source,cube,max(int(context.support_ds.row_times.max()),int(context.query_ds.row_times.max())),
            timestamps,static,context.static_scaled,distance,context.scaler,hidden,device)
        adapter=V5BatchAdapter(context.fit,context.scaler,static,device,context.relation_scales,means)
        saved_rng=rng_payload();checks=[]
        for sidx,qidx in list(zip(s_chunks,q_chunks))[:20]:
            sb=adapter.prepare(builder(raw_index_batch(context.support_ds,sidx)))
            qb=adapter.prepare(builder(raw_index_batch(context.query_ds,qidx)))
            checks.append(first_order_mldg_step(model,sb,qb,optimizer,sw,qw,alpha,
                beta=BETA,slow_weight=SLOW_WEIGHT,amp_dtype=amp_dtype_for(device),
                diagnostic=True,do_update=False))
        restore_rng(saved_rng);optimizer.zero_grad(set_to_none=True)
        rel=np.array([x['relative_inner_step'] for x in checks]);ratio=np.array([x['query_loss_ratio'] for x in checks])
        preflight={'batches':len(checks),'median_relative_inner_step':float(np.median(rel)),
                   'max_query_loss_ratio':float(np.max(ratio)),
                   'mean_gradient_cosine':float(np.mean([x['gradient_cosine'] for x in checks]))}
        preflight_seconds=time.perf_counter()-preflight_started
        preflight['runtime_seconds']=preflight_seconds
        print(f'{name} fold {fold} preflight: {preflight}',flush=True)
        if not np.isfinite(rel).all() or not np.isfinite(ratio).all() or np.median(rel)<1e-6 or np.median(rel)>1e-2 or np.max(ratio)>=5:
            raise RuntimeError(f'MLDG numeric gate failed: {preflight}')
        del builder,adapter;gc.collect();torch.cuda.empty_cache()
    for epoch in range(start_epoch,CFG.max_epochs+1):
        epoch_start=time.perf_counter(); totals=[]; covered=0
        # Context-blocked, rotating order: avoids rebuilding five huge GPU tables
        # every five steps while preserving exact per-epoch station/time exposure.
        context_order=[contexts[(epoch-1+i)%len(contexts)] for i in range(len(contexts))]
        plans=[]
        for context in context_order:
            shard=(context.context_id+epoch-1)%5
            si=selected_rows(context.support_ds,shard); qi=selected_rows(context.query_ds,shard)
            covered+=len(si)+len(qi)
            rng=np.random.default_rng(CFG.seed+fold*100000+epoch*101+context.context_id)
            rng.shuffle(si); rng.shuffle(qi)
            steps=max(1,math.ceil((len(si)+len(qi))/CFG.batch_size))
            s_chunks=np.array_split(si,steps); q_chunks=np.array_split(qi,steps)
            plans.append((context,si,qi,s_chunks,q_chunks))
        progress=tqdm(total=sum(len(x[3]) for x in plans),desc=f'{name} fold {fold} epoch {epoch}',
                      unit='batch',dynamic_ncols=True)
        for context,si,qi,s_chunks,q_chunks in plans:
            s_weights=make_station_weights(context.support_ds.row_targets[si],len(static),device)
            q_weights=make_station_weights(context.query_ds.row_targets[qi],len(static),device)
            hidden=np.setdiff1d(np.arange(len(static)),context.source)
            max_time=max(int(context.support_ds.row_times.max()),int(context.query_ds.row_times.max()))
            builder=DeviceFeatureBuilder(context.source,cube,max_time,timestamps,static,context.static_scaled,
                                         distance,context.scaler,hidden,device)
            adapter=V5BatchAdapter(context.fit,context.scaler,static,device,context.relation_scales,means)
            for step,(sidx,qidx) in enumerate(zip(s_chunks,q_chunks)):
                sb=adapter.prepare(builder(raw_index_batch(context.support_ds,sidx)))
                qb=adapter.prepare(builder(raw_index_batch(context.query_ds,qidx)))
                stat=first_order_mldg_step(model,sb,qb,optimizer,s_weights,q_weights,alpha,
                                           beta=BETA,slow_weight=SLOW_WEIGHT,
                                           amp_dtype=amp_dtype_for(device))
                totals.append(stat)
                progress.update(1)
            del builder,adapter; gc.collect(); torch.cuda.empty_cache()
        progress.close()
        if covered!=expected: raise RuntimeError(f'epoch coverage {covered} != {expected}')
        metrics,station=validate(model,val_context,val_idx,cube,timestamps,static,distance,means,excluded,device,epoch)
        row={'epoch':epoch,'alpha':alpha,'train_support_loss':float(torch.stack([x['support_loss'] for x in totals]).mean().cpu()),
             'train_query_loss':float(torch.stack([x['query_loss'] for x in totals]).mean().cpu()),
             **metrics,'runtime_seconds':time.perf_counter()-epoch_start}
        history.append(row); atomic_csv_save(pd.DataFrame(history),out/'training_history.csv')
        if metrics['macro_rmse']<best:
            best=metrics['macro_rmse'];best_epoch=epoch;best_metrics=dict(metrics)
            atomic_torch_save({'revision':REVISION,'fold':fold,'method':name,'epoch':epoch,'alpha':alpha,
                'model':cpu_state_dict(model),'train_indices':train_idx,'validation_indices':val_idx,
                'metrics':metrics,'settings_hash':settings_hash},out/'best_checkpoint.pt')
            atomic_csv_save(station,out/'best_validation_station_metrics.csv')
        atomic_torch_save({'epoch':epoch,'model':cpu_state_dict(model),'optimizer':optimizer.state_dict(),
            'rng':rng_payload(),'history':history,'best':best,'best_epoch':best_epoch,
            'best_metrics':best_metrics,'preflight':preflight,'revision':REVISION,'method':name,
            'fold':fold,'settings_hash':settings_hash},resume)
        print(f'{name} fold {fold} epoch {epoch}: macroRMSE={metrics["macro_rmse"]:.4f}, '
              f'pooledR2={metrics["r2"]:.4f}, absBias={metrics["mean_abs_bias"]:.4f}, '
              f'{row["runtime_seconds"]:.1f}s',flush=True)
    accumulated_runtime=float(sum(x['runtime_seconds'] for x in history))+float(preflight.get('runtime_seconds',0.0))
    summary={'revision':REVISION,'method':name,'fold':fold,'alpha':alpha,'best_epoch':best_epoch,
             'best_macro_rmse':best,'best_metrics':best_metrics,'preflight':preflight,
             'settings_hash':settings_hash,'runtime_seconds':accumulated_runtime,
             'last_process_segment_seconds':time.perf_counter()-started,
             'schedule':'five context-blocks per epoch; rotating order; disjoint time-index modulo-5 shards',
             'files_kept':
             ['training_history.csv','best_checkpoint.pt','best_validation_station_metrics.csv','summary.json']}
    atomic_text_save(complete,json.dumps(summary,ensure_ascii=False,indent=2)); resume.unlink(missing_ok=True)
    del model,optimizer;gc.collect();torch.cuda.empty_cache();return summary


def show_results(root='/content/DL_TCN_V5_FACTOR_MLDG_PILOT'):
    import ipywidgets as widgets
    from IPython.display import display,clear_output
    root=Path(root); files=sorted(root.glob('fold_*/*/training_history.csv'))
    data=pd.concat([pd.read_csv(f).assign(fold=int(f.parents[1].name[-2:]),method=f.parent.name) for f in files],ignore_index=True)
    options=[(f'{r.method} / fold {r.fold}',(r.method,int(r.fold))) for r in data[['method','fold']].drop_duplicates().itertuples()]
    choice=widgets.Dropdown(options=options,description='結果：',layout=widgets.Layout(width='500px'));out=widgets.Output()
    def render(change=None):
        with out:
            clear_output(wait=True); method,fold=choice.value; part=data[(data.method==method)&(data.fold==fold)]
            display(part.round(4)); best=part.loc[part.macro_rmse.idxmin()]
            display(pd.DataFrame([best]).round(4))
    choice.observe(render,names='value');display(choice,out);render();return choice


def main():
    os.environ.setdefault('DL_TCN_MAX_EPOCHS','15'); CFG.max_epochs=int(os.environ['DL_TCN_MAX_EPOCHS'])
    runtime=apply_runtime_profile(CFG); CFG.compile_mode='off'; CFG.formal_num_workers=0; runtime['compile_mode']='off'
    if CFG.device.type!='cuda': raise RuntimeError('V5 pilot requires Colab GPU')
    actual_amp=amp_dtype_for(CFG.device)
    if actual_amp is not torch.bfloat16:
        raise RuntimeError('V5 FO-MLDG pilot requires BF16-capable CUDA (A100/L4); FP16 is disabled because this meta-gradient path has no GradScaler')
    torch.set_num_threads(min(12,os.cpu_count() or 1)); torch.backends.cuda.matmul.allow_tf32=True
    root=Path(os.environ.get('DL_TCN_V5_PILOT_ROOT','/content/DL_TCN_V5_FACTOR_MLDG_PILOT'))
    if '/content/drive/' in str(root): raise RuntimeError('Pilot root must stay in /content, not Drive')
    root.mkdir(parents=True,exist_ok=True)
    static,clusters,cols=load_static(); excluded=make_outer_groups(clusters)[0]
    splits=make_selector_inner_folds(clusters,excluded); cube,timestamps=build_or_load_hourly_cube(static)
    distance=haversine_matrix(static.longitude,static.latitude)
    alpha=float(os.environ.get('DL_TCN_V5_INNER_ALPHA','1e-4'))
    if not np.isfinite(alpha) or alpha<=0: raise ValueError('DL_TCN_V5_INNER_ALPHA must be finite and > 0')
    cache_meta=CFG.output_dir/'_cache'/'aq_hourly_meta.json'
    fingerprint=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    code_files=[Path(__file__).with_name(name) for name in (
        'run_v5_factorized_mldg_pilot.py','model_v5.py','v5_training.py','model.py','model_v2.py',
        'v2_training.py','background_crossfit.py','data_pipeline.py','train_formal.py',
        'group_selector_protocol.py','config.py')]
    code_hash=hashlib.sha256(b''.join(path.read_bytes() for path in code_files)).hexdigest()
    settings={'revision':REVISION,'code_hash':code_hash,'seed':CFG.seed,
        'folds':[0,3],'methods':{'episodic_control':0.0,'fo_mldg':alpha},
        'epochs':CFG.max_epochs,'lr':CFG.learning_rate,'weight_decay':CFG.weight_decay,
        'dropout':CFG.dropout,'gradient_clip':CFG.gradient_clip_norm,'batch_size':CFG.batch_size,
        'loss':{'beta':BETA,'slow_weight':SLOW_WEIGHT,'name':CFG.loss_name},
        'architecture':{'tcn_hidden':CFG.tcn_hidden,'tcn_kernel_size':CFG.tcn_kernel_size,
            'tcn_dilations':list(CFG.tcn_dilations),'attention_dim':CFG.attention_dim,
            'attention_heads':CFG.attention_heads,'final_hidden':CFG.final_hidden,'static_dim':len(cols)},
        'precision':{'use_amp':CFG.use_amp,'prefer_bf16':CFG.prefer_bf16,
                     'actual_amp_dtype':str(actual_amp),'gpu_name':runtime.get('gpu_name'),
                     'compute_capability':runtime.get('compute_capability')},
        'train_period':[CFG.train_start,CFG.train_end],'history_hours':CFG.history_hours,
        'dynamic_items':list(CFG.dynamic_items),'outer_group':excluded.tolist(),
        'splits':{str(f):{'train':splits[f][0].tolist(),'validation':splits[f][1].tolist()} for f in (0,3)},
        'data_fingerprint':{'cube_meta':fingerprint(cache_meta),'static':fingerprint(CFG.static_path),
                            'clusters':fingerprint(CFG.cluster_path)}}
    settings_hash=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    marker=root/'settings.json'
    if marker.exists() and json.loads(marker.read_text())!=settings: raise RuntimeError('V5 settings changed; use a fresh root')
    atomic_text_save(marker,json.dumps(settings,ensure_ascii=False,indent=2))
    summaries=[]
    for fold in (0,3):
        train_idx,val_idx=splits[fold];means=target_means(cube,timestamps,train_idx)
        print(f'Build fold {fold} five strict source contexts',flush=True)
        contexts=build_contexts(train_idx,clusters,cube,timestamps,static,cols,distance,means)
        val_context=full_validation_context(train_idx,cube,timestamps,static,cols,distance)
        for name,a in [('episodic_control',0.0),('fo_mldg',alpha)]:
            summaries.append(train_method(name,a,fold,train_idx,val_idx,excluded,contexts,val_context,means,
                cube,timestamps,static,cols,distance,root,CFG.device,settings_hash))
    atomic_text_save(root/'pilot_summary.json',json.dumps(summaries,ensure_ascii=False,indent=2))
    comparison=pd.DataFrame([{'fold':x['fold'],'method':x['method'],'best_epoch':x['best_epoch'],
                              **x['best_metrics'],'runtime_seconds':x['runtime_seconds']} for x in summaries])
    atomic_csv_save(comparison,root/'pilot_comparison.csv')
    print(json.dumps({'status':'done','root':str(root),'summaries':summaries},ensure_ascii=False,indent=2))


if __name__=='__main__': main()

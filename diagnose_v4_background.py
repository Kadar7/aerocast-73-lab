"""Read-only validation decomposition. No training, inference, or output files."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import torch


def decompose(y, prediction, background):
    y=np.asarray(y,dtype=float); prediction=np.asarray(prediction,dtype=float)
    if y.shape != prediction.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all() or not np.isfinite(background):
        raise ValueError('Invalid prediction/truth/background')
    mean=float(y.mean()); bias=float((prediction-y).mean())
    bg_bias=float(background-mean)
    centered=float(np.sqrt(np.mean(((prediction-y)-bias)**2)))
    rmse=float(np.sqrt(np.mean((prediction-y)**2)))
    assert np.isclose(rmse**2,centered**2+bias**2,rtol=1e-8,atol=1e-8)
    return dict(truth_mean=mean,background=float(background),background_bias=bg_bias,
                mean_dl_correction=float(prediction.mean()-background),final_bias=bias,
                abs_bias_reduction=abs(bg_bias)-abs(bias),rmse=rmse,
                centered_rmse_diagnostic= centered,
                bias_mse_fraction=bias**2/rmse**2 if rmse else 0.)


def build_diagnostics(root='/content/DL_TCN_V4_VALIDATION_PILOT'):
    from config import CFG
    from data_pipeline import load_static,standardize_static,haversine_matrix
    from background_crossfit import select_background,ridge_predict,idw_predictions
    root=Path(root)
    # Read an existing cube only. Never call build_or_load_hourly_cube (may write).
    cache=CFG.output_dir/'_cache'
    for path in (cache/'aq_hourly_cube.npy',cache/'aq_hourly_meta.json'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing existing cache: {path}. This diagnostic will not rebuild or train.')
    static,_,columns=load_static()
    meta=json.loads((cache/'aq_hourly_meta.json').read_text(encoding='utf-8'))
    if meta['siteids']!=static.siteid.astype(str).tolist() or meta['items']!=list(CFG.aq_cube_items):
        raise ValueError('Cache station order/items mismatch')
    cube=np.load(cache/'aq_hourly_cube.npy',mmap_mode='r',allow_pickle=False)
    if list(cube.shape)!=meta['shape']:
        raise ValueError('Cache shape mismatch')
    timestamps=pd.date_range(meta['start'],periods=len(cube),freq='h')
    distances=haversine_matrix(static.longitude,static.latitude)
    rows=[]
    for fold in (0,3):
        folder=root/f'fold_{fold:02d}'
        history=pd.read_csv(folder/'validation_station_metrics_all_epochs.csv',dtype={'siteid':str})
        checkpoints=sorted((folder/'epoch_checkpoints').glob('epoch_*.pt'))
        if not checkpoints:
            raise FileNotFoundError(f'{folder}: need one original checkpoint for its saved scaler (no model execution).')
        # Trusted, locally generated training checkpoint; never load untrusted .pt.
        checkpoint=torch.load(checkpoints[0],map_location='cpu',weights_only=False)
        if checkpoint.get('v2_config',{}).get('training_revision')!='full_donor_nested_background_bounded_relation_v4':
            raise ValueError('Expected original V4 checkpoint')
        if checkpoint['static_columns']!=columns:
            raise ValueError('Static column order changed')
        source=np.asarray(checkpoint['train_indices'],dtype=int)
        validation=np.asarray(checkpoint['validation_indices'],dtype=int)
        if np.intersect1d(source,validation).size:
            raise ValueError('Training/validation overlap')
        cfg=checkpoint['config']
        start=pd.Timestamp(cfg['train_start']); end=pd.Timestamp(cfg['train_end'])
        if timestamps[0]>start or timestamps[-1]<end:
            raise ValueError('Cached cube does not cover training dates')
        ti=np.flatnonzero((timestamps>=start)&(timestamps<=end))
        pm=meta['items'].index('PM2.5')
        # Only SOURCE training labels fit this background; validation means below are diagnostic only.
        means=np.nanmean(np.asarray(cube[np.ix_(ti,source,[pm])])[...,0],axis=0)
        x=standardize_static(static,columns,SimpleNamespace(**checkpoint['scaler'])).astype(float)
        _,penalty,alpha=select_background(x[source],means,distances[np.ix_(source,source)])
        backgrounds=alpha*ridge_predict(x[source],means,x[validation],penalty)+(1-alpha)*idw_predictions(means,distances[np.ix_(validation,source)])
        saved=checkpoint['background']
        if not np.isclose(penalty,saved['ridge_lambda']) or not np.isclose(alpha,saved['static_weight'],atol=1e-5):
            raise ValueError('Reconstructed background settings differ from checkpoint; source data may have changed')
        lookup=dict(zip(validation,backgrounds.astype('float32')))
        summary_epoch=int(history.groupby('epoch').rmse.mean().idxmin())
        for epoch in sorted(history.epoch.unique()):
            path=folder/'epoch_predictions'/f'epoch_{epoch:03d}.npz'
            if not path.exists():
                raise FileNotFoundError(f'Missing saved predictions: {path}; no retraining will be started')
            with np.load(path,allow_pickle=False) as data:
                stations=data['station_index']; truth=data['y_true']; pred=data['y_pred']
                ts=pd.to_datetime(data['timestamp_ns'],unit='ns')
                if set(np.unique(stations))!=set(validation) or ((ts<start)|(ts>end)).any():
                    raise ValueError('Prediction station/time mismatch')
                for station in validation:
                    keep=stations==station
                    if pd.Index(ts[keep]).has_duplicates:
                        raise ValueError('Duplicate station timestamp')
                    metrics=decompose(truth[keep],pred[keep],lookup[station])
                    recorded=history[(history.epoch==epoch)&(history.station_index==station)]
                    if len(recorded)!=1 or int(recorded.iloc[0]['n'])!=int(keep.sum()) or not np.isclose(metrics['rmse'],recorded.iloc[0].rmse,atol=1e-5):
                        raise ValueError('Predictions disagree with recorded station metrics')
                    rows.append(dict(fold=fold,epoch=int(epoch),siteid=str(static.loc[station,'siteid']),
                                     sitename=static.loc[station,'sitename'],n=int(keep.sum()),
                                     summary_epoch=summary_epoch,**metrics))
        print(f'fold {fold}: read {len(history)} station-epoch records; background reconstructed from sources only')
    return pd.DataFrame(rows)


def show_diagnostics(frame):
    import ipywidgets as widgets
    from IPython.display import display,clear_output
    options=[('各fold摘要（原macro RMSE最低那次，僅供診斷）',0)]+[(f'epoch {e}',int(e)) for e in sorted(frame.epoch.unique())]
    selector=widgets.Dropdown(options=options,description='顯示：',layout=widgets.Layout(width='620px'))
    output=widgets.Output()
    names={'fold':'fold','epoch':'epoch','sitename':'測站','truth_mean':'實際平均','background':'背景估計',
           'background_bias':'背景偏差','mean_dl_correction':'DL平均修正','final_bias':'最後偏差',
           'abs_bias_reduction':'偏差縮小量（正值較好）','rmse':'RMSE',
           'centered_rmse_diagnostic':'扣平均偏差後RMSE（僅診斷）','bias_mse_fraction':'平均偏差占MSE比例'}
    def render(change=None):
        with output:
            clear_output(wait=True)
            selected=frame[frame.epoch==frame.summary_epoch] if selector.value==0 else frame[frame.epoch==selector.value]
            display(selected[list(names)].rename(columns=names).round(3).reset_index(drop=True))
    selector.observe(render,names='value')
    print('正Bias＝高估。扣平均偏差後RMSE使用validation真值，只能診斷，不是可部署的校正成績。')
    display(selector,output); render()
    return selector

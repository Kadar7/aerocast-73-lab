from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import CFG, Config

VALUE_COLUMNS = [f"monitorvalue{h:02d}" for h in range(24)]


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try: return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError: pass
    raise ValueError(f"無法辨識CSV編碼: {path}")


def load_static(cfg: Config = CFG):
    static = pd.read_csv(cfg.static_path, dtype={"siteid": str}, encoding="utf-8-sig")
    clusters = pd.read_csv(cfg.cluster_path, dtype={"siteid": str}, encoding="utf-8-sig")
    static.columns = [str(c).lstrip("\ufeff") for c in static.columns]
    clusters.columns = [str(c).lstrip("\ufeff") for c in clusters.columns]
    if len(static) != 73 or static.siteid.nunique() != 73: raise ValueError("static應為73站")
    if set(static.siteid) != set(clusters.siteid): raise ValueError("static/cluster siteid不一致")
    clusters = clusters.set_index("siteid").loc[static.siteid]["kmeans_cluster"].to_numpy(int)
    static_cols = [c for c in static.columns if c not in ("siteid", "sitename")]
    if len(static_cols) != 49: raise ValueError(f"static應為49欄，實際{len(static_cols)}")
    for c in ("longitude", "latitude"):
        static[c] = pd.to_numeric(static[c], errors="coerce")
        if static[c].isna().any(): raise ValueError(f"{c}有缺值")
    return static.reset_index(drop=True), clusters, static_cols


def choose_split(clusters: np.ndarray, heldout: int, cfg: Config = CFG):
    """Exactly reproduce the Reduced cluster-aware 60/12 split."""
    rng = np.random.default_rng(cfg.seed)
    all_idx = np.arange(len(clusters)); known = all_idx[all_idx != heldout]
    cluster_values = sorted(np.unique(clusters))
    by = {c: all_idx[(clusters == c) & (all_idx != heldout)] for c in cluster_values}
    if any(len(v) < 2 for v in by.values()):
        raise ValueError("每個cluster扣除outer target後至少需2站，才能讓train/validation都保留該群")
    raw = {c: cfg.validation_stations * len(by[c]) / len(known) for c in cluster_values}
    q = {c: min(len(by[c])-1, max(1, math.floor(raw[c]))) for c in cluster_values}
    while sum(q.values()) < cfg.validation_stations:
        eligible = [c for c in cluster_values if q[c] < len(by[c])-1]
        c = max(eligible, key=lambda x: (raw[x]-q[x], len(by[x])))
        q[c] += 1
    while sum(q.values()) > cfg.validation_stations:
        eligible = [c for c in cluster_values if q[c] > 1]
        c = min(eligible, key=lambda x: (raw[x]-q[x], -len(by[x])))
        q[c] -= 1
    val=[]
    for c in cluster_values:
        val.extend(rng.choice(by[c], size=q[c], replace=False).tolist())
    val=np.array(sorted(val),int); train=np.setdiff1d(known,val)
    if (len(train),len(val)) != (60,12): raise AssertionError(f"split應60/12，實際{len(train)}/{len(val)}")
    if heldout in train or heldout in val: raise AssertionError("outer target leakage")
    for c in cluster_values:
        if not np.any(clusters[train] == c) or not np.any(clusters[val] == c):
            raise AssertionError(f"cluster {c} 未同時出現在training與validation")
    return train,val


def make_meta_crossfit_folds(clusters: np.ndarray, heldout: int, cfg: Config = CFG):
    """Create six disjoint 60/12 folds over the 72 known stations.

    Fold 0 is exactly the existing Reduced cluster-aware split.  Its 60
    training stations are then partitioned into five balanced validation
    folds.  Consequently every known station is an unseen validation target
    exactly once, while the outer station is excluded from every fold.
    """
    clusters=np.asarray(clusters,dtype=int)
    known=np.setdiff1d(np.arange(len(clusters),dtype=int),np.array([heldout],dtype=int))
    base_train,base_validation=choose_split(clusters,heldout,cfg)
    rng=np.random.default_rng(cfg.seed+20260901)
    remaining_folds=[[] for _ in range(5)]
    # Assign each cluster separately, always to a currently smallest fold.
    # This spreads rare clusters whenever possible and keeps every fold at 12.
    for cluster in sorted(np.unique(clusters[base_train])):
        members=base_train[clusters[base_train]==cluster].copy()
        rng.shuffle(members)
        cluster_counts=np.zeros(5,dtype=int)
        for station in members:
            sizes=np.array([len(x) for x in remaining_folds],dtype=int)
            candidates=np.flatnonzero(
                (sizes==sizes.min()) & (cluster_counts==cluster_counts.min())
            )
            if candidates.size==0:
                candidates=np.flatnonzero(sizes==sizes.min())
            fold=int(candidates[0])
            remaining_folds[fold].append(int(station))
            cluster_counts[fold]+=1
    folds=[]
    for fold_id,validation in enumerate([base_validation.tolist(),*remaining_folds]):
        validation=np.asarray(sorted(validation),dtype=int)
        training=np.setdiff1d(known,validation)
        if len(training)!=60 or len(validation)!=12:
            raise AssertionError(
                f"crossfit fold {fold_id}應為60/12，實際{len(training)}/{len(validation)}"
            )
        if heldout in training or heldout in validation:
            raise AssertionError(f"crossfit fold {fold_id}混入outer target")
        folds.append((training,validation))
    validation_union=np.concatenate([validation for _,validation in folds])
    if len(np.unique(validation_union))!=72 or set(validation_union)!=set(known):
        raise AssertionError("crossfit validation folds沒有讓72個known stations各出現一次")
    if not np.array_equal(folds[0][0],base_train) or not np.array_equal(folds[0][1],base_validation):
        raise AssertionError("crossfit fold 0沒有保留原Reduced 60/12 split")
    return folds


def haversine_matrix(lon,lat):
    lon=np.radians(np.asarray(lon,float)); lat=np.radians(np.asarray(lat,float))
    dlon=lon[:,None]-lon[None,:]; dlat=lat[:,None]-lat[None,:]
    a=np.sin(dlat/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
    return 2*6371000*np.arcsin(np.sqrt(np.clip(a,0,1)))


def bearing_degrees(donor_lon, donor_lat, target_lon, target_lat):
    lon1=np.radians(np.asarray(donor_lon,float)); lat1=np.radians(np.asarray(donor_lat,float))
    lon2=math.radians(float(target_lon)); lat2=math.radians(float(target_lat)); dlon=lon2-lon1
    x=np.sin(dlon)*math.cos(lat2)
    y=np.cos(lat1)*math.sin(lat2)-np.sin(lat1)*math.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x,y))+360)%360


def discover_aq_files(aq_dir: Path):
    out=sorted(aq_dir.rglob("*.csv"))
    if not out: raise FileNotFoundError(f"找不到AQ CSV: {aq_dir}")
    return out


def build_or_load_hourly_cube(static: pd.DataFrame, cfg: Config = CFG):
    cache_dir=cfg.output_dir/"_cache"; cache_dir.mkdir(parents=True,exist_ok=True)
    cube_path=cache_dir/"aq_hourly_cube.npy"; meta_path=cache_dir/"aq_hourly_meta.json"
    cube_start=pd.Timestamp(cfg.train_start)-pd.Timedelta(hours=cfg.history_hours-1)
    cube_end=pd.Timestamp(cfg.test_end); timestamps=pd.date_range(cube_start,cube_end,freq="h")
    shape=(len(timestamps),len(static),len(cfg.aq_cube_items))
    files=discover_aq_files(cfg.aq_data_dir)
    source_files=[{"path":str(p.resolve()),"size":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns} for p in files]
    expected={"cache_schema_version":cfg.cache_schema_version,"siteids":static.siteid.astype(str).tolist(),"items":list(cfg.aq_cube_items),"shape":list(shape),"start":str(cube_start),"end":str(cube_end),"history_hours":cfg.history_hours,"source_files":source_files}
    if cube_path.exists() and meta_path.exists():
        try:
            meta=json.loads(meta_path.read_text(encoding="utf-8")); cube=np.load(cube_path,mmap_mode="r")
            if tuple(cube.shape)==shape and all(meta.get(k)==v for k,v in expected.items()):
                return cube,timestamps
        except Exception: pass
    siteids=static.siteid.astype(str).tolist(); smap={s:i for i,s in enumerate(siteids)}; imap={s:i for i,s in enumerate(cfg.aq_cube_items)}
    cube=np.lib.format.open_memmap(cube_path,mode="w+",dtype="float32",shape=shape); cube[:]=np.nan
    required={"siteid","itemengname","monitordate",*VALUE_COLUMNS}
    contributing_files=0; read_errors=[]
    for path in files:
        try:
            h=read_csv_flexible(path,nrows=0); cols={str(c).lstrip("\ufeff").strip().lower() for c in h.columns}
            if not required.issubset(cols): continue
            f=read_csv_flexible(path,dtype=str,low_memory=False); f.columns=[str(c).lstrip("\ufeff").strip().lower() for c in f.columns]
        except Exception as exc:
            read_errors.append({"path":str(path),"error":repr(exc)})
            continue
        f["siteid"]=f.siteid.astype(str).str.strip(); f["itemengname"]=f.itemengname.astype(str).str.strip()
        dates=pd.to_datetime(f.monitordate,format="mixed",errors="coerce")
        keep=dates.notna() & dates.between(cube_start.normalize(),cube_end.normalize()) & f.siteid.isin(smap) & f.itemengname.isin(imap)
        if not keep.any(): continue
        contributing_files += 1
        f=f.loc[keep].copy(); dates=dates.loc[keep]
        dup=f.duplicated(["siteid","itemengname","monitordate"],keep="last")
        f=f.loc[~dup]; dates=dates.loc[~dup]
        num=f[VALUE_COLUMNS].fillna("").apply(lambda c:c.str.strip()).apply(pd.to_numeric,errors="coerce").to_numpy("float32",copy=True)
        num[~np.isfinite(num)] = np.nan
        offsets=((dates-cube_start)/pd.Timedelta(hours=1)).astype(int).to_numpy(); sp=f.siteid.map(smap).to_numpy(int); ip=f.itemengname.map(imap).to_numpy(int)
        for hour in range(24):
            ok=(offsets+hour>=0)&(offsets+hour<len(timestamps)); cube[offsets[ok]+hour,sp[ok],ip[ok]]=num[ok,hour]
    if contributing_files == 0:
        raise RuntimeError("AQ資料夾中沒有任何CSV貢獻到指定時段/站點/測項")
    cube.flush(); meta={**expected,"contributing_files":contributing_files,"read_errors":read_errors}
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8"); del cube
    return np.load(cube_path,mmap_mode="r"),timestamps

@dataclass
class TrainOnlyScaler:
    dynamic_mean: np.ndarray; dynamic_std: np.ndarray
    static_mean: np.ndarray; static_std: np.ndarray; static_median: np.ndarray


def fit_train_only_scaler(cube,timestamps,static,static_cols,train_idx,cfg:Config=CFG):
    tidx=np.flatnonzero((timestamps>=pd.Timestamp(cfg.train_start))&(timestamps<=pd.Timestamp(cfg.train_end)))
    ridx=[cfg.aq_cube_items.index(x) for x in cfg.raw_dynamic_items]
    raw=np.asarray(cube[np.ix_(tidx,train_idx,ridx)],dtype="float32").reshape(-1,len(ridx))
    mean=np.nanmean(raw,axis=0); std=np.nanstd(raw,axis=0); std[~np.isfinite(std)|(std<1e-6)]=1
    # Exact target-relative wind statistics over ordered training target/donor pairs only.
    speed=np.asarray(cube[np.ix_(tidx,train_idx,[cfg.aq_cube_items.index("WIND_SPEED")])],dtype="float32")[...,0]
    wfrom=np.asarray(cube[np.ix_(tidx,train_idx,[cfg.aq_cube_items.index("WIND_DIREC")])],dtype="float32")[...,0]
    wind_count=np.zeros(2,dtype="int64"); wind_sum=np.zeros(2,dtype="float64"); wind_sumsq=np.zeros(2,dtype="float64")
    lon=static.longitude.to_numpy(float); lat=static.latitude.to_numpy(float)
    for target_pos,target in enumerate(train_idx):
        donor_pos=np.flatnonzero(train_idx != target)
        bearings=bearing_degrees(lon[train_idx[donor_pos]],lat[train_idx[donor_pos]],lon[target],lat[target])
        sp=speed[:,donor_pos]; wd=wfrom[:,donor_pos]
        valid=np.isfinite(sp)&np.isfinite(wd)
        delta=np.deg2rad(((wd+180.0)%360.0)-bearings[None,:])
        pair=np.stack([sp*np.cos(delta),sp*np.sin(delta)],axis=-1)
        for j in range(2):
            v=pair[...,j][valid].astype("float64",copy=False)
            wind_count[j]+=v.size; wind_sum[j]+=v.sum(); wind_sumsq[j]+=np.square(v).sum()
    if np.any(wind_count == 0): raise ValueError("60 training stations無可用的target-relative wind統計")
    wind_mean=wind_sum/wind_count
    wind_std=np.sqrt(np.maximum(wind_sumsq/wind_count-np.square(wind_mean),0.0))
    wind_std[~np.isfinite(wind_std)|(wind_std<1e-6)]=1.0
    dmean=np.concatenate([mean.astype("float32"),wind_mean.astype("float32")]); dstd=np.concatenate([std.astype("float32"),wind_std.astype("float32")])
    sx=static[static_cols].apply(pd.to_numeric,errors="coerce").to_numpy("float32"); tr=sx[train_idx]
    med=np.nanmedian(tr,axis=0)
    if np.any(~np.isfinite(med)): raise ValueError("training static有整欄缺失")
    tr=np.where(np.isfinite(tr),tr,med); sm=tr.mean(axis=0).astype("float32"); ss=tr.std(axis=0).astype("float32"); ss[ss<1e-6]=1
    return TrainOnlyScaler(dmean,dstd,sm,ss,med.astype("float32"))


def standardize_static(static,static_cols,scaler):
    x=static[static_cols].apply(pd.to_numeric,errors="coerce").to_numpy("float32"); x=np.where(np.isfinite(x),x,scaler.static_median)
    return ((x-scaler.static_mean)/scaler.static_std).astype("float32")


def build_dynamic_history(cube,current_idx,donors,target_idx,static,scaler,cfg:Config=CFG):
    start=current_idx-cfg.history_hours+1; tidx=np.arange(start,current_idx+1)
    ridx=[cfg.aq_cube_items.index(x) for x in cfg.raw_dynamic_items]
    raw=np.asarray(cube[np.ix_(tidx,donors,ridx)],dtype="float32").transpose(1,0,2)
    speed=np.asarray(cube[np.ix_(tidx,donors,[cfg.aq_cube_items.index("WIND_SPEED")])],dtype="float32")[...,0].T
    wfrom=np.asarray(cube[np.ix_(tidx,donors,[cfg.aq_cube_items.index("WIND_DIREC")])],dtype="float32")[...,0].T
    b=bearing_degrees(static.loc[donors,"longitude"],static.loc[donors,"latitude"],static.loc[target_idx,"longitude"],static.loc[target_idx,"latitude"]).astype("float32")
    toward=(wfrom+180)%360; delta=np.deg2rad(toward-b[:,None]); along=speed*np.cos(delta); cross=speed*np.sin(delta)
    values=np.concatenate([raw,np.stack([along,cross],axis=-1).astype("float32")],axis=-1)
    mask=np.isfinite(values).astype("float32")
    values=(values-scaler.dynamic_mean[None,None,:])/scaler.dynamic_std[None,None,:]
    values=np.where(mask>0,values,0.0).astype("float32")
    all_missing=(mask.sum(axis=(1,2))==0)
    return values,mask,all_missing


def geometry_features(donors,target_idx,static,distance):
    d=distance[target_idx,donors]/1000; b=bearing_degrees(static.loc[donors,"longitude"],static.loc[donors,"latitude"],static.loc[target_idx,"longitude"],static.loc[target_idx,"latitude"])
    return np.column_stack([np.log1p(d),np.sin(np.deg2rad(b)),np.cos(np.deg2rad(b))]).astype("float32")


def time_features(ts):
    h=2*np.pi*ts.hour/24; d=2*np.pi*(ts.dayofyear-1)/365.2425
    return np.array([np.sin(h),np.cos(h),np.sin(d),np.cos(d)],dtype="float32")

class ColdStartStationDataset(Dataset):
    def __init__(self,targets,donor_pool,period_start,period_end,cube,timestamps,static,static_scaled,distance,scaler,cfg:Config=CFG):
        self.targets=np.asarray(targets,int); self.donor_pool=np.asarray(donor_pool,int); self.cube=cube; self.timestamps=timestamps; self.static=static; self.static_scaled=static_scaled; self.distance=distance; self.scaler=scaler; self.cfg=cfg
        time_idx=np.flatnonzero((timestamps>=pd.Timestamp(period_start))&(timestamps<=pd.Timestamp(period_end))); pidx=cfg.aq_cube_items.index("PM2.5")
        self.truth_rows=0; self.history_eligible_truth_rows=0; row_targets=[]; row_times=[]
        for target in self.targets:
            y=np.asarray(cube[time_idx,target,pidx],dtype="float32"); valid=np.isfinite(y); self.truth_rows+=int(valid.sum())
            eligible=valid & (time_idx-cfg.history_hours+1>=0); self.history_eligible_truth_rows+=int(eligible.sum())
            chosen=time_idx[eligible]
            row_targets.append(np.full(chosen.size,target,dtype="int16")); row_times.append(chosen.astype("int32"))
        self.row_targets=np.concatenate(row_targets) if row_targets else np.empty(0,dtype="int16")
        self.row_times=np.concatenate(row_times) if row_times else np.empty(0,dtype="int32")
    def __len__(self): return self.row_times.size
    def __getitem__(self,i):
        target=int(self.row_targets[i]); tidx=int(self.row_times[i]); donors=self.donor_pool[self.donor_pool!=target]
        values,mask,all_missing=build_dynamic_history(self.cube,tidx,donors,target,self.static,self.scaler,self.cfg)
        y=float(self.cube[tidx,target,self.cfg.aq_cube_items.index("PM2.5")]); ts=self.timestamps[tidx]
        # A real donor remains attendable even when its whole history is missing:
        # its zero value/mask TCN token, static features and geometry are still valid.
        return {"values":torch.from_numpy(values),"mask":torch.from_numpy(mask),"donor_static":torch.from_numpy(self.static_scaled[donors]),"geometry":torch.from_numpy(geometry_features(donors,target,self.static,self.distance)),"donor_padding_mask":torch.zeros(len(donors),dtype=torch.bool),"donor_all_missing":torch.from_numpy(all_missing.astype(bool)),"target_static":torch.from_numpy(self.static_scaled[target]),"time_features":torch.from_numpy(time_features(ts)),"label":torch.tensor(y,dtype=torch.float32),"target_idx":torch.tensor(target),"time_idx":torch.tensor(tidx),"donor_indices":torch.from_numpy(donors.astype("int64"))}


class ColdStartIndexDataset(Dataset):
    """Compact row-index view used by the vectorized formal-training collator."""
    def __init__(self, source: ColdStartStationDataset):
        self.row_targets=source.row_targets
        self.row_times=source.row_times
    def __len__(self): return self.row_times.size
    def __getitem__(self,i): return int(self.row_targets[i]),int(self.row_times[i])


class VectorizedFormalCollator:
    """Build an entire batch at once without changing any feature semantics."""
    def __init__(self,donor_pool,cube,timestamps,static,static_scaled,distance,scaler,cfg:Config=CFG):
        self.donor_pool=np.asarray(donor_pool,int); self.cube=cube; self.timestamps=timestamps
        self.static=static; self.static_scaled=static_scaled; self.distance=distance; self.scaler=scaler; self.cfg=cfg
        self.raw_indices=np.asarray([cfg.aq_cube_items.index(x) for x in cfg.raw_dynamic_items],int)
        self.speed_index=cfg.aq_cube_items.index("WIND_SPEED"); self.direction_index=cfg.aq_cube_items.index("WIND_DIREC")
        self.pm25_index=cfg.aq_cube_items.index("PM2.5")
        lon=static.longitude.to_numpy(float); lat=static.latitude.to_numpy(float)
        self.bearings=np.stack([bearing_degrees(lon,lat,lon[target],lat[target]) for target in range(len(static))]).astype("float32")

    def __call__(self,batch):
        targets=np.asarray([row[0] for row in batch],dtype="int64")
        current=np.asarray([row[1] for row in batch],dtype="int64")
        donor_rows=[self.donor_pool[self.donor_pool!=target] for target in targets]
        donor_count={len(row) for row in donor_rows}
        if len(donor_count)!=1: raise RuntimeError("同一loader batch的donor數不一致")
        donors=np.stack(donor_rows).astype("int64",copy=False)
        history=current[:,None]-self.cfg.history_hours+1+np.arange(self.cfg.history_hours)[None,:]

        raw=np.asarray(self.cube[history[:,:,None,None],donors[:,None,:,None],self.raw_indices[None,None,None,:]],dtype="float32")
        speed=np.asarray(self.cube[history[:,:,None],donors[:,None,:],self.speed_index],dtype="float32")
        wfrom=np.asarray(self.cube[history[:,:,None],donors[:,None,:],self.direction_index],dtype="float32")
        bearing=self.bearings[targets[:,None],donors]
        delta=np.deg2rad(((wfrom+180.0)%360.0)-bearing[:,None,:])
        along=speed*np.cos(delta); cross=speed*np.sin(delta)
        values=np.concatenate([raw,np.stack([along,cross],axis=-1).astype("float32")],axis=-1).transpose(0,2,1,3)
        mask=np.isfinite(values).astype("float32")
        values=(values-self.scaler.dynamic_mean[None,None,None,:])/self.scaler.dynamic_std[None,None,None,:]
        values=np.ascontiguousarray(np.where(mask>0,values,0.0).astype("float32"))
        mask=np.ascontiguousarray(mask); all_missing=(mask.sum(axis=(2,3))==0)

        distance_km=self.distance[targets[:,None],donors]/1000.0
        geometry=np.stack([np.log1p(distance_km),np.sin(np.deg2rad(bearing)),np.cos(np.deg2rad(bearing))],axis=-1).astype("float32")
        tf=np.stack([time_features(self.timestamps[t]) for t in current]).astype("float32")
        labels=np.asarray(self.cube[current,targets,self.pm25_index],dtype="float32")
        if not np.isfinite(labels).all(): raise RuntimeError("index dataset包含缺失target label")
        return {
            "values":torch.from_numpy(values),"mask":torch.from_numpy(mask),
            "donor_static":torch.from_numpy(np.ascontiguousarray(self.static_scaled[donors])),
            "geometry":torch.from_numpy(np.ascontiguousarray(geometry)),
            "donor_padding_mask":torch.zeros(donors.shape,dtype=torch.bool),
            "donor_all_missing":torch.from_numpy(all_missing),
            "target_static":torch.from_numpy(np.ascontiguousarray(self.static_scaled[targets])),
            "time_features":torch.from_numpy(tf),"label":torch.from_numpy(labels),
            "target_idx":torch.from_numpy(targets),"time_idx":torch.from_numpy(current),
            "donor_indices":torch.from_numpy(donors),
        }


def collate_variable_donors(batch):
    B=len(batch); maxd=max(x["values"].shape[0] for x in batch); T=batch[0]["values"].shape[1]; C=batch[0]["values"].shape[2]; S=batch[0]["donor_static"].shape[1]
    values=torch.zeros(B,maxd,T,C); mask=torch.zeros_like(values); ds=torch.zeros(B,maxd,S); geo=torch.zeros(B,maxd,3); pad=torch.ones(B,maxd,dtype=torch.bool); all_missing=torch.ones(B,maxd,dtype=torch.bool); di=torch.full((B,maxd),-1,dtype=torch.long)
    for b,x in enumerate(batch):
        d=x["values"].shape[0]; values[b,:d]=x["values"]; mask[b,:d]=x["mask"]; ds[b,:d]=x["donor_static"]; geo[b,:d]=x["geometry"]; pad[b,:d]=x["donor_padding_mask"]; all_missing[b,:d]=x["donor_all_missing"]; di[b,:d]=x["donor_indices"]
    return {"values":values,"mask":mask,"donor_static":ds,"geometry":geo,"donor_padding_mask":pad,"donor_all_missing":all_missing,"target_static":torch.stack([x["target_static"] for x in batch]),"time_features":torch.stack([x["time_features"] for x in batch]),"label":torch.stack([x["label"] for x in batch]),"target_idx":torch.stack([x["target_idx"] for x in batch]),"time_idx":torch.stack([x["time_idx"] for x in batch]),"donor_indices":di}

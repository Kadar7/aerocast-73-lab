"""Data adaptation and metrics for V7 CHO-GO."""
from __future__ import annotations

import hashlib
import math
import numpy as np
import pandas as pd
import torch

from config import CFG
from data_pipeline import bearing_degrees


def fixed_hash_groups(siteids) -> np.ndarray:
    return np.asarray([int(hashlib.sha256(str(x).encode()).hexdigest(), 16) % 4 for x in siteids], dtype=np.int64)


def calendar6(timestamps) -> np.ndarray:
    out=[]
    for ts in pd.DatetimeIndex(timestamps):
        h=2*math.pi*ts.hour/24; d=2*math.pi*(ts.dayofyear-1)/365.25; w=2*math.pi*ts.dayofweek/7
        out.append([math.sin(h),math.cos(h),math.sin(d),math.cos(d),math.sin(w),math.cos(w)])
    return np.asarray(out,dtype="float32")


class V7BatchAdapter:
    def __init__(self, static, timestamps, distance, scaler, device):
        self.device=device
        self.groups=torch.as_tensor(fixed_hash_groups(static.siteid),device=device)
        self.calendar=torch.as_tensor(calendar6(timestamps),device=device)
        self.wind_mean=torch.as_tensor(scaler.dynamic_mean[-2:],device=device)
        self.wind_std=torch.as_tensor(scaler.dynamic_std[-2:],device=device)
        self.pm_mean=float(scaler.dynamic_mean[CFG.raw_dynamic_items.index("PM2.5")])
        self.pm_std=float(scaler.dynamic_std[CFG.raw_dynamic_items.index("PM2.5")])
        lon=static.longitude.to_numpy(float);lat=static.latitude.to_numpy(float)
        bearings=np.stack([bearing_degrees(lon,lat,lon[q],lat[q]) for q in range(len(static))]).astype("float32")
        sinb=np.sin(np.deg2rad(bearings));cosb=np.cos(np.deg2rad(bearings))
        geom=np.stack([np.log1p(distance/1000),sinb,cosb],-1).astype("float32")
        self.geometry=torch.as_tensor(geom,device=device)
        self.distance=torch.as_tensor(distance/1000,dtype=torch.float32,device=device)
        self.history_offsets=torch.arange(-23,1,device=device)
        pair=np.asarray(distance,dtype=float)/1000
        off=pair[np.isfinite(pair)&(pair>0)]
        self.rho=torch.tensor(float(np.median(off)),device=device)

    @staticmethod
    def _bands(pair, valid):
        # pair/valid [B,Q,S,T,2]
        pieces=[]
        for lo,hi in ((23,24),(20,23),(12,20),(0,12)):
            v=valid[...,lo:hi,:]
            x=pair[...,lo:hi,:]
            n=v.sum(-2)
            mean=(x*v).sum(-2)/n.clamp_min(1)
            coverage=n.to(x.dtype)/(hi-lo)
            pieces.extend([mean,coverage])
        return torch.cat(pieces,-1)

    def prepare(self,batch):
        batch=dict(batch); donors=batch["donor_indices"]; targets=batch["target_idx"]
        b,d=donors.shape; safe=donors.clamp_min(0)
        history=batch["time_idx"][:,None]+self.history_offsets[None]
        cal=self.calendar[history]
        pmz=batch["values"][:,:,-1,CFG.raw_dynamic_items.index("PM2.5")]
        pmm=batch["mask"][:,:,-1,CFG.raw_dynamic_items.index("PM2.5")]>0
        pmraw=pmz*self.pm_std+self.pm_mean

        # Recover geographic wind vector from target-relative along/cross, then
        # reproject it for every donor->donor query without reading target data.
        wind=batch["values"][...,9:11]*self.wind_std+self.wind_mean
        wmask=batch["mask"][...,9:11]>0
        target_geom=self.geometry[targets[:,None],safe]
        sb=target_geom[...,1];cb=target_geom[...,2]
        along,cross=wind[...,0],wind[...,1]
        east=along*sb[...,None]+cross*cb[...,None]
        north=along*cb[...,None]-cross*sb[...,None]
        base_valid=wmask.all(-1)

        dd=self.geometry[safe[:,:,None],safe[:,None,:]]  # [B,query,source,3]
        diagonal=torch.eye(d,device=self.device,dtype=torch.bool)[None]
        dd=torch.where(diagonal[...,None],torch.zeros_like(dd),dd)
        sbd=dd[...,1].transpose(1,2); cbd=dd[...,2].transpose(1,2)  # [B,source,query]
        al=east[:,:,:,None]*sbd[:,:,None,:]+north[:,:,:,None]*cbd[:,:,None,:]
        cr=east[:,:,:,None]*cbd[:,:,None,:]-north[:,:,:,None]*sbd[:,:,None,:]
        pair=torch.stack([al,cr],-1).permute(0,3,1,2,4) # [B,query,source,T,2]
        valid=base_valid[:,:,:,None,None].expand(-1,-1,-1,d,2).permute(0,3,1,2,4)
        dd_edge=torch.cat([dd,self._bands(pair,valid)],-1)
        dd_edge=torch.where(diagonal[...,None],torch.zeros_like(dd_edge),dd_edge)

        target_pair=wind.permute(0,1,2,3)[:,None]
        target_valid=wmask[:,None]
        target_edge=torch.cat([target_geom,self._bands(target_pair,target_valid).squeeze(1)],-1)
        batch.update({
            "calendar_history":cal,
            "edge_dd":dd_edge,
            "edge_target":target_edge,
            "distance_dd":self.distance[safe[:,:,None],safe[:,None,:]],
            "distance_target":self.distance[targets[:,None],safe],
            "donor_groups":self.groups[safe],
            "current_pm_raw":pmraw,
            "current_pm_mask":pmm & (donors>=0),
            "geometry_weight_scale":self.rho,
            "pm_std":self.pm_std,
        })
        return batch


def forward_v7(model,b):
    return model(b["values"],b["mask"],b["donor_static"],b["target_static"],b["calendar_history"],
        b["edge_dd"],b["edge_target"],b["donor_groups"],b["donor_padding_mask"],
        b["current_pm_raw"],b["current_pm_mask"],b["distance_dd"],b["distance_target"],b["geometry_weight_scale"])

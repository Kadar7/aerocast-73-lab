import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from config import CFG
from data_pipeline import *
from train_formal import DeviceFeatureBuilder,resolve_target


def compare(source,indices,builder):
    old=collate_variable_donors([source[int(i)] for i in indices])
    new=builder({"target_idx":torch.tensor(source.row_targets[indices].astype("int64")),"time_idx":torch.tensor(source.row_times[indices].astype("int64"))})
    maximum={}
    for key in old:
        candidate=new[key].cpu()
        if old[key].dtype.is_floating_point:
            delta=float(torch.max(torch.abs(old[key]-candidate)))
            maximum[key]=delta
            if not torch.allclose(old[key],candidate,rtol=2e-5,atol=2e-5,equal_nan=True): raise RuntimeError(f"{key} mismatch {delta}")
        elif not torch.equal(old[key],candidate): raise RuntimeError(f"{key} mismatch")
    print({"rows":len(indices),"donors":new["values"].shape[1],"equivalent":True,"max_float_delta":maximum})


def main():
    static,clusters,cols=load_static(); outer=resolve_target(static,CFG.target_site); train_idx,val_idx=choose_split(clusters,outer)
    cube,timestamps=build_or_load_hourly_cube(static); distance=haversine_matrix(static.longitude,static.latitude)
    scaler=fit_train_only_scaler(cube,timestamps,static,cols,train_idx); ss=standardize_static(static,cols,scaler)
    train=ColdStartStationDataset(train_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,static,ss,distance,scaler)
    valid=ColdStartStationDataset(val_idx,train_idx,CFG.train_start,CFG.train_end,cube,timestamps,static,ss,distance,scaler)
    builder=DeviceFeatureBuilder(train_idx,cube,int(max(train.row_times.max(),valid.row_times.max())),timestamps,static,ss,distance,scaler,outer,CFG.device)
    rng=np.random.default_rng(91); compare(train,rng.choice(len(train),32,replace=False),builder); compare(valid,rng.choice(len(valid),32,replace=False),builder)


if __name__=="__main__": main()

import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import CFG
from data_pipeline import (
    ColdStartStationDataset,
    build_or_load_hourly_cube,
    choose_split,
    fit_train_only_scaler,
    haversine_matrix,
    load_static,
    standardize_static,
)
from evaluate_outer import audit_checkpoint_protocol, audit_locked_selection
from train_formal import DeviceFeatureBuilder, resolve_target


def main():
    static,clusters,static_cols=load_static()
    outer=resolve_target(static,CFG.target_site)
    train_idx,val_idx=choose_split(clusters,outer)
    checkpoint={
        "train_indices":train_idx,
        "validation_indices":val_idx,
        "outer_index_excluded":outer,
        "static_columns":static_cols,
        "epoch":9,
        "config":{
            "learning_rate":5e-4,"weight_decay":1e-4,"dropout":0.20,
            "gradient_clip_norm":5.0,"batch_size":512,"loss_name":"station_balanced_mse",
        },
    }
    audit_locked_selection(checkpoint)
    audited_train,audited_val=audit_checkpoint_protocol(checkpoint,static,clusters,outer)
    cube,timestamps=build_or_load_hourly_cube(static)
    distance=haversine_matrix(static.longitude,static.latitude)
    scaler=fit_train_only_scaler(cube,timestamps,static,static_cols,audited_train)
    static_scaled=standardize_static(static,static_cols,scaler)
    outer_ds=ColdStartStationDataset(
        [outer],audited_train,CFG.test_start,CFG.test_end,
        cube,timestamps,static,static_scaled,distance,scaler,
    )
    sample=outer_ds[0]
    donors=sample["donor_indices"].numpy()
    assert len(audited_train)==60 and len(audited_val)==12
    assert len(donors)==60
    assert np.array_equal(np.sort(donors),np.sort(audited_train))
    assert outer not in donors and np.intersect1d(donors,audited_val).size==0
    assert np.isfinite(float(sample["label"]))

    builder=DeviceFeatureBuilder(
        audited_train,cube,int(outer_ds.row_times[0]),timestamps,static,
        static_scaled,distance,scaler,outer,CFG.device,
    )
    built=builder({
        "target_idx":torch.tensor([outer],dtype=torch.long),
        "time_idx":torch.tensor([int(outer_ds.row_times[0])],dtype=torch.long),
    })
    assert built["values"].shape[1:]==(60,CFG.history_hours,CFG.n_dynamic_channels)
    assert torch.isfinite(built["label"]).all()
    assert np.isclose(float(built["label"].cpu()),float(sample["label"]),rtol=0,atol=0)
    assert np.array_equal(np.sort(built["donor_indices"][0].cpu().numpy()),np.sort(audited_train))
    print({
        "protocol":"60 train / 12 validation / 1 outer",
        "outer":str(static.loc[outer,"sitename"]),
        "outer_donors":60,
        "validation_donors_in_outer":0,
        "outer_rows":len(outer_ds),
        "truth_rows":outer_ds.truth_rows,
        "gpu_label_preserved":True,
        "first_batch_shape":list(built["values"].shape),
    })


if __name__=="__main__":
    main()

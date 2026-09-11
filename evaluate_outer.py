from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    TrainOnlyScaler,
    build_or_load_hourly_cube,
    choose_split,
    haversine_matrix,
    load_static,
    standardize_static,
)
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset
from train_formal import (
    DeviceFeatureBuilder,
    make_index_loader,
    make_vectorized_loader,
    resolve_target,
    seed_all,
    validate_epoch,
)


def checkpoint_path_from_environment() -> Path:
    raw=os.environ.get("DL_TCN_SELECTION_CHECKPOINT","").strip()
    if not raw:
        raise RuntimeError(
            "請先設定DL_TCN_SELECTION_CHECKPOINT，指向dropout=0.20、weight_decay=1e-4的best_checkpoint.pt"
        )
    path=Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"selection checkpoint不存在: {path}")
    return path


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:
        return torch.load(path,map_location="cpu")


def scaler_from_checkpoint(checkpoint: dict) -> TrainOnlyScaler:
    saved=checkpoint.get("scaler")
    required=("dynamic_mean","dynamic_std","static_mean","static_std","static_median")
    if not isinstance(saved,dict) or any(key not in saved for key in required):
        raise RuntimeError("checkpoint缺少完整train-only scaler")
    return TrainOnlyScaler(**{
        key:np.asarray(saved[key],dtype="float32") for key in required
    })


def audit_locked_selection(checkpoint: dict) -> None:
    saved=checkpoint.get("config",{})
    expected={
        "learning_rate":5e-4,
        "weight_decay":1e-4,
        "dropout":0.20,
        "gradient_clip_norm":5.0,
        "batch_size":512,
    }
    mismatches=[]
    for key,value in expected.items():
        actual=saved.get(key)
        if actual is None or not np.isclose(float(actual),float(value),rtol=0,atol=1e-12):
            mismatches.append(f"{key}: checkpoint={actual!r}, expected={value!r}")
    if saved.get("loss_name")!="station_balanced_mse":
        mismatches.append(f"loss_name: checkpoint={saved.get('loss_name')!r}, expected='station_balanced_mse'")
    expected_epoch=int(os.environ.get("DL_TCN_EXPECTED_SELECTION_EPOCH","9"))
    if int(checkpoint.get("epoch",-1))!=expected_epoch:
        mismatches.append(f"epoch: checkpoint={checkpoint.get('epoch')!r}, expected={expected_epoch}")
    if mismatches:
        raise RuntimeError("不是已鎖定的dropout-only最佳checkpoint:\n"+"\n".join(mismatches))


def audit_checkpoint_protocol(checkpoint,static,clusters,outer):
    train_idx=np.asarray(checkpoint.get("train_indices"),dtype="int64")
    val_idx=np.asarray(checkpoint.get("validation_indices"),dtype="int64")
    saved_outer=int(checkpoint.get("outer_index_excluded",-1))
    expected_train,expected_val=choose_split(clusters,outer)
    if saved_outer!=outer:
        raise RuntimeError(f"checkpoint outer={saved_outer}，目前設定outer={outer}")
    if not np.array_equal(train_idx,expected_train) or not np.array_equal(val_idx,expected_val):
        raise RuntimeError("checkpoint 60/12 split與目前Reduced cluster-aware split不一致")
    if len(train_idx)!=60 or len(val_idx)!=12:
        raise RuntimeError(f"checkpoint不是60/12: {len(train_idx)}/{len(val_idx)}")
    if outer in train_idx or outer in val_idx:
        raise RuntimeError("outer target出現在checkpoint train/validation")
    if len(np.union1d(train_idx,val_idx))!=72:
        raise RuntimeError("60/12沒有覆蓋完整72個known stations")
    if len(checkpoint.get("static_columns",[]))!=49:
        raise RuntimeError("checkpoint static features不是49欄")
    return train_idx,val_idx


def main() -> None:
    seed_all(CFG.seed)
    device=CFG.device
    runtime_profile=apply_runtime_profile(CFG)
    if device.type=="cuda":
        if CFG.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32=True
            torch.backends.cudnn.allow_tf32=True
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark=True

    selection_path=checkpoint_path_from_environment()
    checkpoint=load_checkpoint(selection_path)
    audit_locked_selection(checkpoint)
    static,clusters,static_cols=load_static()
    outer=resolve_target(static,CFG.target_site)
    train_idx,val_idx=audit_checkpoint_protocol(checkpoint,static,clusters,outer)
    if list(checkpoint["static_columns"])!=list(static_cols):
        raise RuntimeError("checkpoint static欄位名稱/順序與目前資料不一致")
    scaler=scaler_from_checkpoint(checkpoint)

    cube,timestamps=build_or_load_hourly_cube(static)
    distance=haversine_matrix(static.longitude,static.latitude)
    static_scaled=standardize_static(static,static_cols,scaler)
    outer_ds=ColdStartStationDataset(
        [outer],train_idx,CFG.test_start,CFG.test_end,
        cube,timestamps,static,static_scaled,distance,scaler,
    )
    if len(train_idx)!=60 or outer in train_idx:
        raise RuntimeError("outer donor pool必須是排除outer的60 training stations")
    if len(outer_ds)==0:
        raise RuntimeError("outer test沒有可評估PM2.5 truth")
    first=outer_ds[0]
    if len(first["donor_indices"])!=60:
        raise RuntimeError("outer sample donor數不是60")
    if not np.array_equal(np.sort(first["donor_indices"].numpy()),np.sort(train_idx)):
        raise RuntimeError("outer donors不是checkpoint的60 training stations")
    if outer in first["donor_indices"].numpy() or np.intersect1d(first["donor_indices"].numpy(),val_idx).size:
        raise RuntimeError("outer donor pool混入outer或12 validation stations")

    model=TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    model.eval()
    if int(checkpoint.get("epoch",-1))<=0:
        raise RuntimeError("checkpoint沒有有效best epoch")

    output_dir=CFG.formal_output_root/CFG.outer_output_dirname
    output_dir.mkdir(parents=True,exist_ok=True)
    sanity=audit_dataset(outer_ds,max_samples=min(64,len(outer_ds)))
    if sanity["sampled_donor_counts"]!=[60]:
        raise RuntimeError("outer sanity donor count不是60")
    protocol={
        "total_stations":73,
        "training_stations":60,
        "validation_stations":12,
        "outer_test_stations":1,
        "outer_donors":60,
        "outer_index":outer,
        "outer_siteid":str(static.loc[outer,"siteid"]),
        "outer_sitename":str(static.loc[outer,"sitename"]),
        "selection_checkpoint":str(selection_path),
        "selected_epoch":int(checkpoint["epoch"]),
        "test_start":CFG.test_start,
        "test_end":CFG.test_end,
        "outer_used_for_model_selection":False,
        "validation_stations_used_as_outer_donors":False,
    }
    (output_dir/"outer_protocol_sanity.json").write_text(
        json.dumps({"protocol":protocol,"dataset":sanity},ensure_ascii=False,indent=2),encoding="utf-8"
    )

    feature_builder=None
    if device.type=="cuda":
        max_time_index=int(outer_ds.row_times.max())
        feature_builder=DeviceFeatureBuilder(
            train_idx,cube,max_time_index,timestamps,static,static_scaled,
            distance,scaler,outer,device,
        )
        loader=make_index_loader(outer_ds,False)
        runtime_profile["feature_precompute_seconds"]=feature_builder.precompute_seconds
        runtime_profile["precomputed_feature_tables_mb"]=feature_builder.precomputed_table_mb
    else:
        loader=make_vectorized_loader(
            outer_ds,train_idx,cube,timestamps,static,static_scaled,distance,scaler,False
        )

    print(f"torch version: {torch.__version__}",flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}",flush=True)
    print(f"GPU name: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}",flush=True)
    print(f"selection checkpoint epoch: {checkpoint['epoch']}",flush=True)
    print(f"outer target: {protocol['outer_sitename']} (siteid={protocol['outer_siteid']})",flush=True)
    print(f"outer sample count: {len(outer_ds):,} / truth timestamps: {outer_ds.truth_rows:,}",flush=True)
    print("outer donors: 60 training stations; validation donors: 0",flush=True)

    started=time.perf_counter()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    metrics,station_metrics,predictions=validate_epoch(
        model,loader,device,static,timestamps,int(checkpoint["epoch"]),feature_builder
    )
    elapsed=time.perf_counter()-started
    peak_vram=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else 0.0
    coverage=float(len(outer_ds)/max(outer_ds.truth_rows,1))
    metrics.update({
        "truth_timestamp_denominator":int(outer_ds.truth_rows),
        "predicted_timestamps":int(len(outer_ds)),
        "prediction_to_truth_coverage":coverage,
        "evaluation_seconds":elapsed,
        "gpu_peak_vram_mb":peak_vram,
    })
    predictions.insert(1,"siteid",protocol["outer_siteid"])
    predictions.insert(2,"sitename",protocol["outer_sitename"])
    predictions.to_csv(output_dir/"outer_predictions.csv",index=False,encoding="utf-8-sig",date_format="%Y-%m-%d %H:%M:%S")
    station_metrics.to_csv(output_dir/"outer_station_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{**protocol,**metrics}]).to_csv(output_dir/"outer_metrics.csv",index=False,encoding="utf-8-sig")
    summary={"protocol":protocol,"metrics":metrics,"runtime_profile":runtime_profile,"output_dir":str(output_dir)}
    (output_dir/"outer_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()

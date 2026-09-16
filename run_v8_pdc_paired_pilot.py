"""Two-fold paired V8-PDC pilot; never launches an outer or full-73 run."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    build_or_load_hourly_cube,
    fit_train_only_scaler,
    haversine_matrix,
    load_static,
    make_meta_crossfit_folds,
    standardize_static,
)
from model_v8_pdc import PooledDonorCorrection
from train_formal import DeviceFeatureBuilder, amp_dtype_for, make_grad_scaler, regression_metrics, seed_all
from v5_training import event_metrics
from v7_training import calendar6
from v8_sampling import BalancedStationTimeSchedule


REVISION = "v8_pdc_paired_pilot_3"
FOLDS = (0, 3)
ARMS = ("control", "treatment")
MIN_EPOCHS = 8
MAX_EPOCHS = 20
PATIENCE = 5
TRAIN_BATCH_SIZE = 256  # fixed by 4 stations x 64 times
VALIDATION_BATCH_SIZE = 512
# The measured A100 worst case for four complete arms is about 57 minutes.
# Keep a real safety margin without reducing epochs, steps or model capacity.
PREFLIGHT_LIMIT_SECONDS = 75 * 60
LOCAL_CHECKPOINT_EVERY_STEPS = 200
IMPROVEMENT_EPSILON = 0.01
SCRIPT_DIR = Path(__file__).resolve().parent
FROZEN_V8_PATH = SCRIPT_DIR / "resources" / "v8_fold03_locked_baseline.csv"
FROZEN_V8_SHA256 = "ffe8020465178d9405f854df31cc49a14fd1e9ef53ac7a20c328f5cfd56bda46"
FROZEN_V8_EPOCHS = {0: 3, 3: 9}
FROZEN_V8_OUTER = {"siteid": "17", "sitename": "桃園"}
FROZEN_V8_DATES = {
    "train_start": "2024-07-01 00:00:00",
    "train_end": "2025-06-30 23:00:00",
}
SPARSE_DEFINITION = "within-fold validation stations whose nearest training-donor distance is >= fold 75th percentile"
PHASES = ("training", "validation_pending", "completed")
FINGERPRINT_CODE_FILES = (
    "config.py", "data_pipeline.py", "model_v8.py", "model_v8_pdc.py",
    "run_v8_pdc_paired_pilot.py", "train_formal.py", "v7_training.py", "v8_sampling.py",
)


def atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_torch(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def sync_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value) -> str:
    array = np.ascontiguousarray(value)
    if np.issubdtype(array.dtype, np.floating) and np.isnan(array).any():
        array = array.copy()
        array[np.isnan(array)] = np.nan  # canonical NumPy NaN payload
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_protocol_payload(static, static_cols, cube, timestamps, outer, splits) -> dict:
    period = (timestamps >= pd.Timestamp(CFG.train_start)) & (timestamps <= pd.Timestamp(CFG.train_end))
    cube_period = np.asarray(cube[period], dtype="float32")
    static_canonical = static[["siteid", "sitename", *static_cols]].copy()
    static_canonical["siteid"] = static_canonical.siteid.astype(str)
    static_csv = static_canonical.to_csv(
        index=False, lineterminator="\n", na_rep="<NA>", float_format="%.17g"
    ).encode("utf-8")
    code_hashes = {name: sha256_file(SCRIPT_DIR / name) for name in FINGERPRINT_CODE_FILES}
    split_payload = {
        str(fold): {
            "train_indices": [int(x) for x in splits[fold][0]],
            "train_siteids": [str(static.loc[int(x), "siteid"]) for x in splits[fold][0]],
            "validation_indices": [int(x) for x in splits[fold][1]],
            "validation_siteids": [str(static.loc[int(x), "siteid"]) for x in splits[fold][1]],
        }
        for fold in FOLDS
    }
    hparams = {
        "folds": list(FOLDS), "arms": list(ARMS), "min_epochs": MIN_EPOCHS,
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "improvement_epsilon": IMPROVEMENT_EPSILON,
        "learning_rate": 5e-4, "weight_decay": 1e-4, "gradient_clip": 1.0,
        "station_rounds_per_epoch": 128, "stations_per_batch": 4,
        "times_per_station": 64, "validation_batch_size": VALIDATION_BATCH_SIZE,
        "loss": "station-balanced standardized MSE only",
        "checkpoint_metric": "macro station RMSE",
        "seed": CFG.seed, "amp": CFG.use_amp, "prefer_bf16": CFG.prefer_bf16,
        "compile_mode": "off", "lr_schedule": "0.5-epoch warmup then cosine to 0.1x",
        "model": {"static_dim": 49, "dynamic_channels": 11, "history_hours": 24,
                  "field_heads": 8, "field_value_dim": 16, "hidden_dim": 96,
                  "dropout": 0.10, "pm_channel_index": 7,
                  "distance_prior_logit": "-2*log1p(distance_km)",
                  "candidate_correction_last_layer_zero_init": True,
                  "weight_residual_last_layer_zero_init": True},
        "raw_dynamic_items": list(CFG.raw_dynamic_items),
        "derived_dynamic_items": list(CFG.derived_dynamic_items),
    }
    payload = {
        "revision": REVISION,
        "outer": {"index": int(outer), "siteid": str(static.loc[outer, "siteid"]),
                  "sitename": str(static.loc[outer, "sitename"])},
        "dates": {"train_start": CFG.train_start, "train_end": CFG.train_end},
        "static_columns": list(static_cols),
        "static_table_sha256": hashlib.sha256(static_csv).hexdigest(),
        "training_cube_sha256": sha256_array(cube_period),
        "training_timestamps_sha256": sha256_array(timestamps[period].view("int64")),
        "splits": split_payload,
        "code_sha256": code_hashes,
        "hyperparameters": hparams,
        "frozen_v8_sha256": FROZEN_V8_SHA256,
    }
    return payload


def validate_protocol_fingerprint(payload: dict, expected: str, source: str) -> None:
    actual = payload.get("protocol_fingerprint")
    if actual != expected:
        raise RuntimeError(f"protocol fingerprint mismatch in {source}: {actual!r} != {expected!r}")


def validate_existing_manifest(path: Path, expected: str) -> None:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_protocol_fingerprint(payload, expected, str(path))


def clear_abort_only_root(root: Path | None) -> bool:
    """Remove stale preflight-only metadata, never trained artefacts.

    Revision 2 could abort after benchmarking because its 45-minute ceiling
    was lower than the measured 57-minute worst case.  That abort changes no
    model state, but its old protocol manifest would block the corrected run.
    """
    if root is None or not root.exists():
        return False
    gate = root / "paired_pilot_gate.json"
    if not gate.exists():
        return False
    try:
        payload = json.loads(gate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "ABORTED_BEFORE_TRAINING":
        return False
    protected = ("result.json", "active_resume.pt", "best_model.pt")
    if any(any(root.rglob(name)) for name in protected):
        raise RuntimeError(f"Refusing to clear aborted root containing trained artefacts: {root}")
    for name in (
        "protocol_manifest.json", "preflight_cost.json", "paired_pilot_gate.json",
        "preflight_io.tmp.pt",
    ):
        (root / name).unlink(missing_ok=True)
    print(f"CLEARED stale abort-only metadata: {root}", flush=True)
    return True


def phase_action(phase: str, epoch: int, bad_epochs: int) -> str:
    if phase not in PHASES:
        raise RuntimeError(f"invalid resume phase: {phase!r}")
    if phase == "completed" or (phase == "training" and epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE):
        return "finalize"
    if phase == "validation_pending":
        return "validate"
    return "train"


def checkpoint_phase_after_batch(schedule_epoch: int, batch_epoch: int) -> str:
    """Close the train→validate boundary even when a periodic save lands on it."""
    return "validation_pending" if int(schedule_epoch) > int(batch_epoch) else "training"


def select_resume_payload(local_path: Path, drive_path: Path | None, expected_fingerprint: str,
                          fold: int, arm: str, map_location="cpu"):
    candidates = []
    for path in (local_path, drive_path):
        if path is None or not path.exists():
            continue
        payload = torch.load(path, map_location=map_location, weights_only=False)
        validate_protocol_fingerprint(payload, expected_fingerprint, str(path))
        if payload.get("revision") != REVISION or payload.get("fold") != fold or payload.get("arm") != arm:
            raise RuntimeError(f"resume identity mismatch: {path}")
        phase = payload.get("phase")
        if phase not in PHASES:
            raise RuntimeError(f"resume missing/invalid phase: {path}")
        rank = PHASES.index(phase)
        # A post-validation ``training`` checkpoint has one more history row
        # than the preceding ``validation_pending`` checkpoint at the same
        # global step, so validated-history length must outrank phase.
        candidates.append((int(payload.get("global_step", -1)), len(payload.get("history", [])), rank, path, payload))
    if not candidates:
        return None, None
    _, _, _, path, payload = max(candidates, key=lambda row: row[:3])
    return path, payload


def negative_station_ids(station_frame: pd.DataFrame) -> set[int]:
    return set(station_frame.loc[station_frame.r2 < 0, "station_index"].astype(int))


def no_new_negative_station_ids(current: set[int], frozen: set[int]) -> bool:
    return current.issubset(frozen)


def standardized_mse(prediction: torch.Tensor, label: torch.Tensor, pm_std: float) -> torch.Tensor:
    if not np.isfinite(pm_std) or pm_std <= 0:
        raise ValueError("pm_std must be finite and positive")
    return torch.square((prediction.float() - label.float()) / float(pm_std)).mean()


def project_complete_cost(*, train_step_seconds: float, validation_batch_seconds: float,
                          checkpoint_local_seconds: float, checkpoint_drive_seconds: float,
                          setup_seconds: float, benchmark_seconds: float,
                          train_steps_per_epoch: int, validation_batches: int,
                          runs: int = 4, epochs: int = MAX_EPOCHS) -> dict:
    periodic_local = math.ceil(train_steps_per_epoch / LOCAL_CHECKPOINT_EVERY_STEPS)
    train_steps = runs * epochs * train_steps_per_epoch
    # Every epoch validates, and every arm performs one final best-checkpoint validation.
    validation_passes = runs * (epochs + 1)
    validation_batch_count = validation_passes * validation_batches
    # Per epoch: periodic resumes, validation_pending active, post-validation
    # active, and in the worst case a new best checkpoint.  Add one
    # checkpoint-equivalent final-artifact write per run to cover compact CSV,
    # JSON and NPZ output.
    local_writes = runs * epochs * (periodic_local + 3) + runs
    drive_writes = runs * epochs * 3 + runs
    training = train_steps * train_step_seconds
    validation = validation_batch_count * validation_batch_seconds
    checkpoint = local_writes * checkpoint_local_seconds + drive_writes * checkpoint_drive_seconds
    projected = training + validation + checkpoint + setup_seconds + benchmark_seconds + 60.0
    return {
        "runs": runs, "epochs_per_run_worst_case": epochs,
        "train_steps": train_steps, "validation_passes": validation_passes,
        "validation_batches": validation_batch_count,
        "periodic_local_writes_per_epoch": periodic_local,
        "local_checkpoint_writes": local_writes, "drive_checkpoint_writes": drive_writes,
        "projected_training_seconds": training,
        "projected_validation_seconds": validation,
        "projected_checkpoint_io_seconds": checkpoint,
        "full_setup_seconds": setup_seconds,
        "benchmark_seconds": benchmark_seconds,
        "projected_complete_pilot_seconds": projected,
        "preflight_limit_seconds": PREFLIGHT_LIMIT_SECONDS,
        "within_preflight_limit": projected <= PREFLIGHT_LIMIT_SECONDS,
    }


def target_index(static: pd.DataFrame, value: str) -> int:
    found = np.flatnonzero(
        (static.siteid.astype(str).to_numpy() == str(value))
        | (static.sitename.astype(str).to_numpy() == str(value))
    )
    if len(found) != 1:
        raise ValueError(f"outer target {value!r} matched {len(found)} stations")
    return int(found[0])


def raw_batch(targets, times) -> dict[str, torch.Tensor]:
    return {
        "target_idx": torch.as_tensor(targets, dtype=torch.long),
        "time_idx": torch.as_tensor(times, dtype=torch.long),
    }


class V8PDCAdapter:
    """Add only tensors absent from the reviewed V8 DeviceFeatureBuilder."""

    def __init__(self, timestamps, scaler, device: torch.device) -> None:
        self.device = device
        self.calendar = torch.as_tensor(calendar6(timestamps), device=device)
        self.offsets = torch.arange(-23, 1, device=device)
        pm_index = CFG.raw_dynamic_items.index("PM2.5")
        self.pm_index = pm_index
        self.pm_mean = float(scaler.dynamic_mean[pm_index])
        self.pm_std = float(scaler.dynamic_std[pm_index])
        if not np.isfinite(self.pm_mean) or not np.isfinite(self.pm_std) or self.pm_std <= 0:
            raise ValueError("invalid train-only PM2.5 statistics")

    def prepare(self, batch: dict) -> dict:
        batch = dict(batch)
        history = batch["time_idx"][:, None] + self.offsets[None]
        batch["calendar"] = self.calendar[history]
        pm_z = batch["values"][:, :, -1, self.pm_index]
        observed = batch["mask"][:, :, -1, self.pm_index] > 0
        physical = pm_z * self.pm_std + self.pm_mean
        batch["current_pm_physical"] = torch.where(
            observed, physical, torch.full_like(physical, torch.nan)
        )
        return batch


def forward_model(model: PooledDonorCorrection, batch: dict, adapter: V8PDCAdapter, arm: str):
    return model(
        batch["values"], batch["mask"], batch["donor_static"], batch["geometry"],
        batch["donor_padding_mask"], batch["target_static"], batch["calendar"],
        batch["current_pm_physical"], adapter.pm_mean, adapter.pm_std,
        output_mode=arm,
    )


def valid_times_by_station(dataset: ColdStartStationDataset) -> dict[int, np.ndarray]:
    return {
        int(station): dataset.row_times[dataset.row_targets == station].astype("int64")
        for station in dataset.targets
    }


def station_metrics(y, prediction, station_ids, static) -> pd.DataFrame:
    rows = []
    for station in np.unique(station_ids):
        keep = station_ids == station
        rows.append({
            "station_index": int(station),
            "siteid": str(static.loc[station, "siteid"]),
            "sitename": str(static.loc[station, "sitename"]),
            "n": int(keep.sum()),
            **regression_metrics(y[keep], prediction[keep]),
        })
    return pd.DataFrame(rows)


def load_frozen_v8_baseline(static, splits, distance, outer) -> dict:
    actual_hash = sha256_file(FROZEN_V8_PATH)
    if actual_hash != FROZEN_V8_SHA256:
        raise RuntimeError(f"frozen V8 baseline hash changed: {actual_hash}")
    if {"siteid": str(static.loc[outer, "siteid"]), "sitename": str(static.loc[outer, "sitename"])} != FROZEN_V8_OUTER:
        raise RuntimeError("frozen V8 baseline is valid only for the locked 桃園 outer target")
    if {"train_start": CFG.train_start, "train_end": CFG.train_end} != FROZEN_V8_DATES:
        raise RuntimeError("frozen V8 baseline year does not match the current protocol")
    if len([column for column in static.columns if column not in ("siteid", "sitename")]) != 49:
        raise RuntimeError("frozen V8 baseline requires exactly static49")
    frame = pd.read_csv(FROZEN_V8_PATH, dtype={"siteid": str})
    required = {"fold", "epoch", "station_index", "siteid", "sitename", "n", "mae", "rmse", "r2", "bias"}
    if not required.issubset(frame.columns):
        raise RuntimeError("frozen V8 baseline columns changed")
    result = {
        "version": 1, "path": str(FROZEN_V8_PATH), "sha256": actual_hash,
        **FROZEN_V8_DATES,
        "outer": dict(FROZEN_V8_OUTER), "sparse_definition": SPARSE_DEFINITION,
        "folds": {},
    }
    all_macro = []
    for fold in FOLDS:
        train_idx, val_idx = splits[fold]
        part = frame.loc[frame.fold == fold].copy()
        if set(part.station_index.astype(int)) != set(map(int, val_idx)):
            raise RuntimeError(f"frozen V8 fold {fold} station IDs do not match current split")
        if set(part.epoch.astype(int)) != {FROZEN_V8_EPOCHS[fold]}:
            raise RuntimeError(f"frozen V8 fold {fold} epoch changed")
        for row in part.itertuples():
            station = int(row.station_index)
            if str(row.siteid) != str(static.loc[station, "siteid"]) or str(row.sitename) != str(static.loc[station, "sitename"]):
                raise RuntimeError(f"frozen V8 station identity mismatch at index {station}")
        nearest = np.asarray([distance[int(station), train_idx].min() for station in part.station_index])
        cutoff = float(np.quantile(nearest, 0.75))
        sparse_ids = set(part.loc[nearest >= cutoff, "station_index"].astype(int))
        negative_ids = negative_station_ids(part)
        fold_payload = {
            "epoch": FROZEN_V8_EPOCHS[fold],
            "validation_station_ids": sorted(map(int, val_idx)),
            "macro_rmse": float(part.rmse.mean()),
            "macro_r2": float(part.r2.mean()),
            "mean_abs_bias": float(part.bias.abs().mean()),
            "negative_r2_station_ids": sorted(negative_ids),
            "sparse_station_ids": sorted(sparse_ids),
            "sparse_cutoff_m": cutoff,
            "sparse_macro_rmse": float(part.loc[part.station_index.isin(sparse_ids), "rmse"].mean()),
        }
        result["folds"][str(fold)] = fold_payload
        all_macro.append(fold_payload["macro_rmse"])
    result["mean_macro_rmse"] = float(np.mean(all_macro))
    return result


@torch.inference_mode()
def validate(model, arm, dataset, builder, adapter, static, train_indices, distance, device):
    model.eval()
    dtype = amp_dtype_for(device)
    ys, ps, stations, times = [], [], [], []
    anchors, refs, corrections = [], [], []
    for start in tqdm(
        range(0, len(dataset), VALIDATION_BATCH_SIZE),
        desc=f"{arm} full validation", unit="batch", dynamic_ncols=True, leave=False,
    ):
        stop = min(start + VALIDATION_BATCH_SIZE, len(dataset))
        idx = np.arange(start, stop)
        batch = adapter.prepare(builder(raw_batch(dataset.row_targets[idx], dataset.row_times[idx])))
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            prediction, aux = forward_model(model, batch, adapter, arm)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("non-finite V8-PDC validation prediction")
        ys.append(batch["label"].float().cpu().numpy())
        ps.append(prediction.float().cpu().numpy())
        stations.append(batch["target_idx"].cpu().numpy())
        times.append(batch["time_idx"].cpu().numpy())
        anchors.append(aux["anchor_count"].cpu().numpy())
        refs.append(aux["reference_component"].float().cpu().numpy())
        corrections.append(aux["correction_component"].float().cpu().numpy())
    y = np.concatenate(ys); prediction = np.concatenate(ps)
    station_ids = np.concatenate(stations); time_ids = np.concatenate(times)
    anchor_count = np.concatenate(anchors); reference = np.concatenate(refs)
    correction = np.concatenate(corrections)
    per_station = station_metrics(y, prediction, station_ids, static)
    overall = regression_metrics(y, prediction)
    overall.update(event_metrics(y, prediction))
    nearest = np.asarray([distance[int(s), train_indices].min() for s in per_station.station_index])
    sparse = nearest >= np.quantile(nearest, 0.75)
    sparse_station_ids = sorted(per_station.loc[sparse, "station_index"].astype(int).tolist())
    overall.update({
        "macro_rmse": float(per_station.rmse.mean()),
        "macro_r2": float(per_station.r2.mean()),
        "mean_abs_bias": float(per_station.bias.abs().mean()),
        "sparse_quartile_macro_rmse": float(per_station.loc[sparse, "rmse"].mean()),
        "sparse_station_ids": sparse_station_ids,
        "negative_r2_stations": int((per_station.r2 < 0).sum()),
        "negative_r2_station_ids": sorted(negative_station_ids(per_station)),
        "anchorless_rate": float(np.mean(anchor_count == 0)),
        "anchored_coverage": float(np.mean(anchor_count > 0)),
        "mean_anchor_count": float(np.mean(anchor_count)),
        "mean_reference_component": float(np.mean(reference)),
        "mean_correction_component": float(np.mean(correction)),
    })
    prediction_payload = {
        "y": y.astype("float32"),
        "prediction": prediction.astype("float32"),
        "station_index": station_ids.astype("int16"),
        "time_index": time_ids.astype("int32"),
        "anchor_count": anchor_count.astype("int16"),
    }
    return overall, per_station, prediction_payload


def optimizer_for(model, device):
    kwargs = {"lr": 5e-4, "weight_decay": 1e-4}
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs), False


def lr_lambda(step: int, total_steps: int) -> float:
    warmup = max(1, math.ceil(0.5 * total_steps / MAX_EPOCHS))
    if step < warmup:
        return max((step + 1) / warmup, 1e-3)
    progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([
            torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in state["cuda"]
        ])


def checkpoint_payload(fold, arm, model, optimizer, scheduler, grad_scaler, schedule,
                       history, station_history, best_macro, best_epoch, bad_epochs,
                       epoch_loss_sum, epoch_seen, global_step, phase, protocol_fingerprint,
                       stop_confirmed=False) -> dict:
    if phase not in PHASES:
        raise ValueError(f"invalid checkpoint phase: {phase}")
    return {
        "revision": REVISION, "protocol_fingerprint": protocol_fingerprint,
        "fold": fold, "arm": arm, "phase": phase, "stop_confirmed": bool(stop_confirmed),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "grad_scaler": grad_scaler.state_dict(),
        "schedule": schedule.state_dict(), "history": history,
        "station_history": station_history, "best_macro": best_macro,
        "best_epoch": best_epoch, "bad_epochs": bad_epochs,
        "epoch_loss_sum": epoch_loss_sum, "epoch_seen": epoch_seen,
        "global_step": global_step, "rng": rng_state(),
    }


def save_resume(payload, local_path: Path, drive_path: Path | None, *, sync_drive: bool) -> None:
    atomic_torch(payload, local_path)
    if sync_drive and drive_path is not None:
        sync_file(local_path, drive_path)


def sanity(model, batch, adapter, arm) -> dict:
    model.train(); model.zero_grad(set_to_none=True)
    prediction, aux = forward_model(model, batch, adapter, arm)
    loss = standardized_mse(prediction, batch["label"], adapter.pm_std)
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
        raise RuntimeError("V8-PDC sanity produced non-finite values")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("V8-PDC sanity produced non-finite gradients")
    anchored = aux["has_anchor"]
    if anchored.any() and not torch.allclose(
        aux["weight_sum"][anchored], torch.ones_like(aux["weight_sum"][anchored]), atol=1e-5
    ):
        raise RuntimeError("V8-PDC candidate weights do not sum to one")
    if (~anchored).any() and not torch.equal(
        prediction[~anchored], torch.full_like(prediction[~anchored], adapter.pm_mean)
    ):
        raise RuntimeError("V8-PDC no-anchor fallback is not exact")
    return {"loss": float(loss.detach()), "parameters": sum(p.numel() for p in model.parameters())}


def benchmark(train_ds, val_ds, builder, adapter, device, drive_root, setup_seconds):
    benchmark_started = time.perf_counter()
    seed = CFG.seed + 99001
    seed_all(seed)
    schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=seed)
    model = PooledDonorCorrection(dropout=0.10).to(device)
    optimizer, fused = optimizer_for(model, device)
    dtype = amp_dtype_for(device); grad_scaler = make_grad_scaler(dtype == torch.float16)
    train_times = []
    for iteration in range(110):
        item = schedule.next_batch(); started = time.perf_counter()
        batch = adapter.prepare(builder(raw_batch(item.station_indices, item.time_indices)))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            prediction, _ = forward_model(model, batch, adapter, "treatment")
            loss = standardized_mse(prediction, batch["label"], adapter.pm_std)
        grad_scaler.scale(loss).backward(); grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        grad_scaler.step(optimizer); grad_scaler.update()
        if device.type == "cuda": torch.cuda.synchronize(device)
        if iteration >= 10: train_times.append(time.perf_counter() - started)
    eval_times = []; eval_y = []; eval_p = []; eval_s = []; model.eval()
    with torch.inference_mode():
        for iteration in range(110):
            start = (iteration * VALIDATION_BATCH_SIZE) % len(val_ds)
            idx = np.arange(start, min(start + VALIDATION_BATCH_SIZE, len(val_ds)))
            started = time.perf_counter()
            batch = adapter.prepare(builder(raw_batch(val_ds.row_targets[idx], val_ds.row_times[idx])))
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                prediction, _ = forward_model(model, batch, adapter, "treatment")
            if device.type == "cuda": torch.cuda.synchronize(device)
            if iteration >= 10:
                eval_y.append(batch["label"].float().cpu().numpy())
                eval_p.append(prediction.float().cpu().numpy())
                eval_s.append(batch["target_idx"].cpu().numpy())
                eval_times.append(time.perf_counter() - started)
    post_started = time.perf_counter()
    measured_y = np.concatenate(eval_y); measured_p = np.concatenate(eval_p)
    measured_s = np.concatenate(eval_s)
    regression_metrics(measured_y, measured_p)
    event_metrics(measured_y, measured_p)
    for station in np.unique(measured_s):
        keep = measured_s == station
        regression_metrics(measured_y[keep], measured_p[keep])
    compressed = io.BytesIO()
    np.savez_compressed(compressed, y=measured_y.astype("float32"),
                        prediction=measured_p.astype("float32"),
                        station_index=measured_s.astype("int16"))
    validation_post_seconds = time.perf_counter() - post_started

    io_payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "marker": REVISION}
    io_local = Path("/content/v8_pdc_preflight_io.pt")
    io_started = time.perf_counter(); atomic_torch(io_payload, io_local)
    local_io_seconds = time.perf_counter() - io_started
    drive_io_seconds = 0.0
    if drive_root is not None:
        io_drive = drive_root / "preflight_io.tmp.pt"
        io_started = time.perf_counter(); sync_file(io_local, io_drive)
        drive_io_seconds = time.perf_counter() - io_started
        io_drive.unlink(missing_ok=True)
    io_local.unlink(missing_ok=True)

    train_p95 = float(np.percentile(train_times, 95))
    eval_p95 = float(np.percentile(eval_times, 95) + validation_post_seconds / len(eval_times))
    validation_batches = math.ceil(len(val_ds) / VALIDATION_BATCH_SIZE)
    benchmark_elapsed = time.perf_counter() - benchmark_started
    cost = project_complete_cost(
        train_step_seconds=train_p95,
        validation_batch_seconds=eval_p95,
        checkpoint_local_seconds=local_io_seconds,
        checkpoint_drive_seconds=drive_io_seconds,
        # Includes initial cube/static loading and conservatively charges it
        # once per fold, although the real command builds the cube only once.
        setup_seconds=len(FOLDS) * setup_seconds,
        benchmark_seconds=benchmark_elapsed,
        train_steps_per_epoch=len(schedule),
        validation_batches=validation_batches,
    )
    result = {
        "warmup_batches": 10, "measured_batches": 100,
        "train_batch_p95_seconds": train_p95,
        "validation_batch_p95_seconds": eval_p95,
        "validation_postprocess_seconds_for_100_batches": validation_post_seconds,
        "validation_timing_includes": "GPU forward, CPU copy, pooled/per-station metrics, npz compression",
        "steps_per_epoch": len(schedule),
        "validation_batches_per_full_pass": validation_batches,
        "local_checkpoint_seconds": local_io_seconds,
        "drive_checkpoint_seconds": drive_io_seconds,
        "fused_adamw": fused,
        "setup_seconds_including_cube_static": setup_seconds,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        **cost,
    }
    del model, optimizer, grad_scaler
    torch.cuda.empty_cache()
    return result


def train_arm(fold, arm, train_idx, val_idx, train_ds, val_ds, builder, adapter,
              static, distance, device, local_root, drive_root, global_started,
              protocol_fingerprint):
    arm_seed = CFG.seed + fold * 1009
    seed_all(arm_seed)
    schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=arm_seed)
    model = PooledDonorCorrection(dropout=0.10, output_mode=arm).to(device)
    optimizer, fused = optimizer_for(model, device)
    total_steps = MAX_EPOCHS * len(schedule)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, total_steps)
    )
    dtype = amp_dtype_for(device); grad_scaler = make_grad_scaler(dtype == torch.float16)
    arm_dir = local_root / f"fold_{fold:02d}" / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    drive_arm = drive_root / f"fold_{fold:02d}" / arm if drive_root else None
    local_active = arm_dir / "active_resume.pt"
    drive_active = drive_arm / "active_resume.pt" if drive_arm else None
    local_best = arm_dir / "best_model.pt"
    drive_best = drive_arm / "best_model.pt" if drive_arm else None
    history: list[dict] = []; station_history: list[dict] = []
    best_macro = float("inf"); best_epoch = 0; bad_epochs = 0
    epoch_loss_sum = 0.0; epoch_seen = 0; global_step = 0
    phase = "training"; stop_confirmed = False

    resume, payload = select_resume_payload(
        local_active, drive_active, protocol_fingerprint, fold, arm, map_location=device
    )
    if payload is not None:
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"]); grad_scaler.load_state_dict(payload["grad_scaler"])
        schedule.load_state_dict(payload["schedule"]); history = payload["history"]
        station_history = payload["station_history"]; best_macro = float(payload["best_macro"])
        best_epoch = int(payload["best_epoch"]); bad_epochs = int(payload["bad_epochs"])
        epoch_loss_sum = float(payload["epoch_loss_sum"]); epoch_seen = int(payload["epoch_seen"])
        global_step = int(payload["global_step"]); phase = payload["phase"]
        stop_confirmed = bool(payload.get("stop_confirmed", False)); restore_rng(payload["rng"])
        print(f"RESUME {resume} fold={fold} arm={arm} phase={phase} epoch={schedule.epoch} step={global_step}", flush=True)
    else:
        probe = schedule.next_batch()
        batch = adapter.prepare(builder(raw_batch(probe.station_indices, probe.time_indices)))
        check = sanity(model, batch, adapter, arm)
        schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=arm_seed)
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"fold": fold, "arm": arm, "sanity": check}, ensure_ascii=False), flush=True)

    run_started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    epoch_started = time.perf_counter()
    while True:
        action = phase_action(phase, schedule.epoch, bad_epochs)
        if action == "finalize":
            phase = "completed"
            break
        if action == "train":
            if schedule.epoch >= MAX_EPOCHS:
                phase = "completed"
                break
            epoch_number = schedule.epoch + 1; epoch_started = time.perf_counter(); model.train()
            for item in schedule.iter_epoch():
                batch = adapter.prepare(builder(raw_batch(item.station_indices, item.time_indices)))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                    prediction, _ = forward_model(model, batch, adapter, arm)
                    # Equal samples per station make this the station-balanced
                    # standardized MSE; it is the only optimized objective.
                    loss = standardized_mse(prediction, batch["label"], adapter.pm_std)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite V8-PDC loss")
                grad_scaler.scale(loss).backward(); grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                grad_scaler.step(optimizer); grad_scaler.update(); scheduler.step()
                count = len(item.station_indices)
                epoch_loss_sum += float(loss.detach()) * count; epoch_seen += count; global_step += 1
                if global_step % 100 == 0:
                    progress = {
                        "revision": REVISION, "protocol_fingerprint": protocol_fingerprint,
                        "fold": fold, "arm": arm, "phase": phase,
                        "epoch": epoch_number, "batch_in_epoch": item.batch_index + 1,
                        "batches_per_epoch": len(schedule), "global_step": global_step,
                        "running_train_standardized_mse": epoch_loss_sum / max(epoch_seen, 1),
                        "elapsed_total_seconds": time.perf_counter() - global_started,
                        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                    }
                    atomic_json(progress, local_root / "progress.json")
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
                if global_step % LOCAL_CHECKPOINT_EVERY_STEPS == 0:
                    # ``next_batch`` advances ``schedule.epoch`` on the final
                    # batch.  If that batch also hits the periodic checkpoint
                    # boundary, mark validation pending immediately; otherwise
                    # a disconnect in the tiny gap below could skip validation.
                    periodic_phase = checkpoint_phase_after_batch(schedule.epoch, item.epoch)
                    save_resume(checkpoint_payload(
                        fold, arm, model, optimizer, scheduler, grad_scaler, schedule,
                        history, station_history, best_macro, best_epoch, bad_epochs,
                        epoch_loss_sum, epoch_seen, global_step, periodic_phase, protocol_fingerprint,
                        stop_confirmed,
                    ), local_active, drive_active, sync_drive=periodic_phase == "validation_pending")
            # This checkpoint is intentionally written before validation.  A
            # disconnect after the last train batch therefore cannot skip the
            # mandatory full validation or start the next epoch.
            phase = "validation_pending"
            save_resume(checkpoint_payload(
                fold, arm, model, optimizer, scheduler, grad_scaler, schedule,
                history, station_history, best_macro, best_epoch, bad_epochs,
                epoch_loss_sum, epoch_seen, global_step, phase, protocol_fingerprint,
                stop_confirmed,
            ), local_active, drive_active, sync_drive=True)
            continue

        # validation_pending: schedule.epoch is the epoch whose final training
        # batch was already committed.
        epoch_number = int(schedule.epoch)
        if epoch_number <= 0 or epoch_seen <= 0:
            raise RuntimeError("validation_pending checkpoint has no completed epoch samples")
        train_standardized_mse = epoch_loss_sum / epoch_seen
        metrics, station, prediction = validate(
            model, arm, val_ds, builder, adapter, static, train_idx, distance, device
        )
        epoch_row = {
            "fold": fold, "arm": arm, "epoch": epoch_number,
            "train_standardized_mse": train_standardized_mse,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_runtime_seconds": time.perf_counter() - epoch_started, **metrics,
        }
        history.append(epoch_row)
        station_history.extend({"fold": fold, "arm": arm, "epoch": epoch_number, **row}
                               for row in station.to_dict("records"))
        improved = metrics["macro_rmse"] < best_macro - IMPROVEMENT_EPSILON
        if improved:
            best_macro = float(metrics["macro_rmse"]); best_epoch = epoch_number; bad_epochs = 0
            best_payload = {
                "revision": REVISION, "protocol_fingerprint": protocol_fingerprint,
                "fold": fold, "arm": arm, "epoch": epoch_number,
                "model": copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()}),
                "metrics": metrics, "train_indices": np.asarray(train_idx),
                "validation_indices": np.asarray(val_idx),
            }
            atomic_torch(best_payload, local_best)
            if drive_best is not None: sync_file(local_best, drive_best)
            np.savez_compressed(arm_dir / "best_predictions.npz", **prediction)
        else:
            bad_epochs += 1
        epoch_loss_sum = 0.0; epoch_seen = 0
        stop_confirmed = epoch_number >= MIN_EPOCHS and bad_epochs >= PATIENCE
        phase = "completed" if stop_confirmed or epoch_number >= MAX_EPOCHS else "training"
        save_resume(checkpoint_payload(
            fold, arm, model, optimizer, scheduler, grad_scaler, schedule,
            history, station_history, best_macro, best_epoch, bad_epochs,
            epoch_loss_sum, epoch_seen, global_step, phase, protocol_fingerprint,
            stop_confirmed,
        ), local_active, drive_active, sync_drive=True)
        pd.DataFrame(history).to_csv(arm_dir / "training_history.csv", index=False, encoding="utf-8-sig")
        atomic_json({"protocol_fingerprint": protocol_fingerprint, "phase": phase,
                     "latest": epoch_row, "best_epoch": best_epoch,
                     "best_macro_rmse": best_macro, "bad_epochs": bad_epochs},
                    arm_dir / "progress_epoch.json")

    reached_max = schedule.epoch >= MAX_EPOCHS
    recent = history[-3:]
    recent_slope = float(np.polyfit(
        [row["epoch"] for row in recent],
        [row["macro_rmse"] for row in recent],
        1,
    )[0]) if len(recent) == 3 else float("nan")
    unconverged = bool(
        reached_max
        and (best_epoch >= MAX_EPOCHS - 1 or recent_slope <= -IMPROVEMENT_EPSILON)
    )
    if not local_best.exists() and drive_best is not None and drive_best.exists():
        sync_file(drive_best, local_best)
    best_payload = torch.load(local_best, map_location=device, weights_only=False)
    validate_protocol_fingerprint(best_payload, protocol_fingerprint, str(local_best))
    model.load_state_dict(best_payload["model"])
    final_metrics, best_station, best_prediction = validate(
        model, arm, val_ds, builder, adapter, static, train_idx, distance, device
    )
    pd.DataFrame(history).to_csv(arm_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(station_history).to_csv(arm_dir / "station_metrics_history.csv", index=False, encoding="utf-8-sig")
    best_station.to_csv(arm_dir / "best_station_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(arm_dir / "best_predictions.npz", **best_prediction)
    result = {
        "revision": REVISION, "protocol_fingerprint": protocol_fingerprint,
        "fold": fold, "arm": arm, "phase": "completed",
        "best_epoch": int(best_payload["epoch"]), "epochs_completed": int(schedule.epoch),
        "early_stop_confirmed": stop_confirmed, "unconverged_at_max20": unconverged,
        "recent_three_epoch_macro_rmse_slope": recent_slope,
        "metrics": final_metrics, "runtime_seconds": time.perf_counter() - run_started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "fused_adamw": fused,
    }
    atomic_json(result, arm_dir / "result.json")
    if drive_arm is not None:
        for name in ("result.json", "training_history.csv", "station_metrics_history.csv",
                     "best_station_metrics.csv", "best_predictions.npz", "best_model.pt"):
            sync_file(arm_dir / name, drive_arm / name)
        drive_active.unlink(missing_ok=True)
    local_active.unlink(missing_ok=True)
    return result


def paired_gate(results, local_root, drive_root, frozen_v8, protocol_fingerprint):
    for index, result in enumerate(results):
        validate_protocol_fingerprint(result, protocol_fingerprint, f"gate result[{index}]")
    indexed = {(row["fold"], row["arm"]): row for row in results}
    checks = []
    for fold in FOLDS:
        control = indexed[(fold, "control")]; treatment = indexed[(fold, "treatment")]
        frozen = frozen_v8["folds"][str(fold)]
        treatment_negative = set(map(int, treatment["metrics"]["negative_r2_station_ids"]))
        frozen_negative = set(map(int, frozen["negative_r2_station_ids"]))
        treatment_sparse = set(map(int, treatment["metrics"]["sparse_station_ids"]))
        frozen_sparse = set(map(int, frozen["sparse_station_ids"]))
        checks.extend([
            {"fold": fold, "criterion": "both_arms_converged",
             "pass": not control["unconverged_at_max20"] and not treatment["unconverged_at_max20"]},
            {"fold": fold, "criterion": "treatment_macro_rmse_not_worse",
             "pass": treatment["metrics"]["macro_rmse"] <= control["metrics"]["macro_rmse"]},
            {"fold": fold, "criterion": "boundary_rmse_improves_at_least_8pct_vs_frozen_v8",
             "pass": treatment_sparse == frozen_sparse
                     and treatment["metrics"]["sparse_quartile_macro_rmse"]
                     <= 0.92 * frozen["sparse_macro_rmse"],
             "treatment_sparse_station_ids": sorted(treatment_sparse),
             "frozen_sparse_station_ids": frozen["sparse_station_ids"]},
            {"fold": fold, "criterion": "no_new_negative_r2_station_ids_vs_frozen_v8",
             "pass": no_new_negative_station_ids(treatment_negative, frozen_negative),
             "treatment_negative_station_ids": sorted(treatment_negative),
             "frozen_negative_station_ids": sorted(frozen_negative),
             "new_negative_station_ids": sorted(treatment_negative - frozen_negative)},
            {"fold": fold, "criterion": "anchored_coverage_at_least_99pct",
             "pass": treatment["metrics"]["anchored_coverage"] >= 0.99},
        ])
    control_mean = float(np.mean([indexed[(fold, "control")]["metrics"]["macro_rmse"] for fold in FOLDS]))
    treatment_mean = float(np.mean([indexed[(fold, "treatment")]["metrics"]["macro_rmse"] for fold in FOLDS]))
    checks.append({"fold": "both", "criterion": "mean_macro_rmse_at_least_5pct_below_frozen_v8",
                   "pass": treatment_mean <= 0.95 * frozen_v8["mean_macro_rmse"]})
    status = "GO_FOR_SEPARATE_REVIEW" if all(item["pass"] for item in checks) else "NO_GO"
    summary = {
        "revision": REVISION, "protocol_fingerprint": protocol_fingerprint,
        "status": status, "checks": checks,
        "control_mean_macro_rmse": control_mean,
        "treatment_mean_macro_rmse": treatment_mean,
        "relative_macro_rmse_change": treatment_mean / control_mean - 1.0,
        "frozen_v8": frozen_v8,
        "results": results, "outer_truth_used": False, "full73_launched": False,
        "note": "This pilot can never launch full73; even GO requires a separate reviewed command.",
    }
    local_gate = local_root / "paired_pilot_gate.json"
    drive_gate = drive_root / "paired_pilot_gate.json" if drive_root is not None else None
    validate_existing_manifest(local_gate, protocol_fingerprint)
    if drive_gate is not None:
        validate_existing_manifest(drive_gate, protocol_fingerprint)
    atomic_json(summary, local_gate)
    if drive_root is not None:
        sync_file(local_gate, drive_gate)
    return summary


def main() -> None:
    global_started = time.perf_counter()
    os.environ["DL_TCN_COMPILE_MODE"] = "off"
    runtime = apply_runtime_profile(CFG)
    if CFG.device.type != "cuda":
        raise RuntimeError("V8-PDC paired pilot requires Colab CUDA")
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_all(CFG.seed)
    local_root = Path(os.environ.get("V8_PDC_OUTPUT_ROOT", "/content/DL_TCN_V8_PDC_PAIRED_PILOT"))
    if "/drive/" in str(local_root):
        raise RuntimeError("V8_PDC_OUTPUT_ROOT must remain under /content")
    drive_base = Path("/content/drive/MyDrive")
    drive_root = Path(os.environ.get(
        "V8_PDC_DRIVE_OUTPUT", str(drive_base / "DL_TCN_V8_PDC_PAIRED_PILOT")
    )) if drive_base.is_dir() else None
    local_root.mkdir(parents=True, exist_ok=True)
    # Safe migration from revision 2: only its metadata-only aborted run is
    # cleared. Any checkpoint or completed result makes this refuse deletion.
    clear_abort_only_root(local_root)
    clear_abort_only_root(drive_root)

    complete_setup_started = time.perf_counter()
    static, clusters, static_cols = load_static()
    outer = target_index(static, CFG.target_site)
    splits = make_meta_crossfit_folds(clusters, outer)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    frozen_v8 = load_frozen_v8_baseline(static, splits, distance, outer)
    protocol_payload = build_protocol_payload(static, static_cols, cube, timestamps, outer, splits)
    protocol_fingerprint = canonical_hash(protocol_payload)
    manifest = {
        "revision": REVISION, "protocol_fingerprint": protocol_fingerprint,
        "protocol_payload": protocol_payload,
        "purpose": "paired two-fold output-parameterization pilot",
        "folds": list(FOLDS), "arms": list(ARMS), "outer_index": outer,
        "outer_siteid": str(static.loc[outer, "siteid"]),
        "outer_sitename": str(static.loc[outer, "sitename"]),
        "train_start": CFG.train_start, "train_end": CFG.train_end,
        "train_stations": 60, "validation_stations": 12,
        "training_donors": 59, "validation_donors": 60,
        "history_hours": 24, "dynamic_channels": 11, "static_columns": static_cols,
        "exact_v8_difference": (
            "V8 already uses a signed-affine current-PM skip plus one global post-pooling delta; "
            "PDC instead learns per-donor deltas before nonnegative pooling, and the paired control "
            "removes only the explicit current-PM term from the final equation"
        ),
        "sampler": "4 stations x 64 times x 128 full-station rounds",
        "steps_per_epoch": 1920, "loss": "station-balanced standardized MSE only",
        "checkpoint_metric": "validation macro station RMSE",
        "epoch_policy": {"min": MIN_EPOCHS, "max": MAX_EPOCHS, "patience": PATIENCE,
                         "material_improvement_rmse": IMPROVEMENT_EPSILON,
                         "unconverged": "at max20, best is epoch19/20 or recent three-epoch slope <= -0.01"},
        "frozen_v8_gate": frozen_v8,
        "runtime": runtime, "full73_enabled": False,
    }
    local_manifest = local_root / "protocol_manifest.json"
    drive_manifest = drive_root / "protocol_manifest.json" if drive_root is not None else None
    validate_existing_manifest(local_manifest, protocol_fingerprint)
    if drive_manifest is not None:
        validate_existing_manifest(drive_manifest, protocol_fingerprint)
    atomic_json(manifest, local_manifest)
    if drive_root is not None:
        sync_file(local_manifest, drive_manifest)

    first_train, first_val = splits[FOLDS[0]]
    scaler = fit_train_only_scaler(cube, timestamps, static, static_cols, first_train)
    scaled = standardize_static(static, static_cols, scaler)
    first_train_ds = ColdStartStationDataset(first_train, first_train, CFG.train_start, CFG.train_end,
        cube, timestamps, static, scaled, distance, scaler)
    first_val_ds = ColdStartStationDataset(first_val, first_train, CFG.train_start, CFG.train_end,
        cube, timestamps, static, scaled, distance, scaler)
    if len(first_train) != 60 or len(first_val) != 12:
        raise RuntimeError("pilot split is not 60/12")
    hidden = np.setdiff1d(np.arange(len(static)), first_train)
    builder = DeviceFeatureBuilder(first_train, cube,
        max(int(first_train_ds.row_times.max()), int(first_val_ds.row_times.max())),
        timestamps, static, scaled, distance, scaler, hidden, CFG.device)
    adapter = V8PDCAdapter(timestamps, scaler, CFG.device)
    setup_seconds = time.perf_counter() - complete_setup_started
    preflight = benchmark(first_train_ds, first_val_ds, builder, adapter, CFG.device,
                          drive_root, setup_seconds)
    preflight["revision"] = REVISION
    preflight["protocol_fingerprint"] = protocol_fingerprint
    atomic_json(preflight, local_root / "preflight_cost.json")
    if drive_root is not None:
        sync_file(local_root / "preflight_cost.json", drive_root / "preflight_cost.json")
    print(json.dumps({"manifest": manifest, "preflight": preflight}, ensure_ascii=False, indent=2), flush=True)
    if not preflight["within_preflight_limit"]:
        abort = {
            "status": "ABORTED_BEFORE_TRAINING", "revision": REVISION,
            "protocol_fingerprint": protocol_fingerprint,
            "reason": (
                "complete paired pilot worst-case projection exceeds "
                f"{PREFLIGHT_LIMIT_SECONDS / 60:.0f} minutes"
            ),
            "preflight": preflight, "full73_launched": False,
        }
        local_gate = local_root / "paired_pilot_gate.json"
        drive_gate = drive_root / "paired_pilot_gate.json" if drive_root is not None else None
        validate_existing_manifest(local_gate, protocol_fingerprint)
        if drive_gate is not None:
            validate_existing_manifest(drive_gate, protocol_fingerprint)
        atomic_json(abort, local_gate)
        if drive_root is not None:
            sync_file(local_gate, drive_gate)
        print(json.dumps(abort, ensure_ascii=False, indent=2), flush=True)
        return
    del builder, adapter, first_train_ds, first_val_ds
    torch.cuda.empty_cache()

    results = []
    for fold in FOLDS:
        train_idx, val_idx = splits[fold]
        scaler = fit_train_only_scaler(cube, timestamps, static, static_cols, train_idx)
        scaled = standardize_static(static, static_cols, scaler)
        train_ds = ColdStartStationDataset(train_idx, train_idx, CFG.train_start, CFG.train_end,
            cube, timestamps, static, scaled, distance, scaler)
        val_ds = ColdStartStationDataset(val_idx, train_idx, CFG.train_start, CFG.train_end,
            cube, timestamps, static, scaled, distance, scaler)
        if any(len(train_idx[train_idx != target]) != 59 for target in train_idx):
            raise RuntimeError("training donor protocol is not 59")
        hidden = np.setdiff1d(np.arange(len(static)), train_idx)
        builder = DeviceFeatureBuilder(train_idx, cube,
            max(int(train_ds.row_times.max()), int(val_ds.row_times.max())),
            timestamps, static, scaled, distance, scaler, hidden, CFG.device)
        adapter = V8PDCAdapter(timestamps, scaler, CFG.device)
        for arm in ARMS:
            local_completed = local_root / f"fold_{fold:02d}" / arm / "result.json"
            drive_completed = drive_root / f"fold_{fold:02d}" / arm / "result.json" if drive_root else None
            completed_payloads = []
            for completed in (local_completed, drive_completed):
                if completed is None or not completed.exists():
                    continue
                completed_payload = json.loads(completed.read_text(encoding="utf-8"))
                validate_protocol_fingerprint(completed_payload, protocol_fingerprint, str(completed))
                if completed_payload.get("fold") != fold or completed_payload.get("arm") != arm:
                    raise RuntimeError(f"completed result identity mismatch: {completed}")
                completed_payloads.append(completed_payload)
            if completed_payloads:
                if len(completed_payloads) == 2 and canonical_hash(completed_payloads[0]) != canonical_hash(completed_payloads[1]):
                    raise RuntimeError(f"local/Drive completed results disagree for fold={fold} arm={arm}")
                results.append(completed_payloads[0])
                print(f"SKIP completed fold={fold} arm={arm}", flush=True)
                continue
            results.append(train_arm(
                fold, arm, train_idx, val_idx, train_ds, val_ds, builder, adapter,
                static, distance, CFG.device, local_root, drive_root, global_started,
                protocol_fingerprint,
            ))
        del builder, adapter, train_ds, val_ds
        torch.cuda.empty_cache()

    summary = paired_gate(results, local_root, drive_root, frozen_v8, protocol_fingerprint)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

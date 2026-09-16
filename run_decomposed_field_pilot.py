"""Two-fold, budget-gated pilot for the decomposed field nowcaster.

This runner never launches a 73-station experiment.  It benchmarks the exact
training and validation path first, then trains folds 0 and 3 only.  Essential
resume state is mirrored to Drive; compact final artefacts replace it after a
fold completes.
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset, build_or_load_hourly_cube, fit_train_only_scaler,
    haversine_matrix, load_static, make_meta_crossfit_folds, standardize_static,
)
from model_decomposed_field import DecomposedFieldNowcaster, decomposed_loss
from run_v8_sasfo_pilot import (
    V8Adapter, amp_dtype_for, atomic_json, atomic_torch, make_grad_scaler,
    raw_batch, regression_metrics, restore_rng, rng_state, seed_all, sync_file,
    target_index, valid_times_by_station,
)
from train_formal import DeviceFeatureBuilder
from v5_training import event_metrics
from v8_sampling import BalancedStationTimeSchedule


REVISION = "decomposed_field_background_offset_anomaly_1"
FOLDS = (0, 3)
INITIAL_EPOCHS = 15
CONVERGENCE_CHECK_EPOCH = 18
MAX_EPOCHS = 20
VALIDATION_BATCH_SIZE = 512
CHECKPOINT_EVERY_STEPS = 100
MATERIAL_IMPROVEMENT_FRACTION = 0.0025
LEVEL_WEIGHT = 0.35
ANOMALY_CENTER_WEIGHT = 0.10
PILOT_BUDGET_SECONDS = 2.0 * 60 * 60
AVAILABLE_A100_SECONDS = 30.0 * 60 * 60
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_PATH = SCRIPT_DIR / "resources" / "v8_fold03_locked_baseline.csv"


@contextmanager
def visible_stage(name, heartbeat_seconds=20.0):
    """Keep slow setup work visible in notebook output."""
    started = time.perf_counter()
    stopped = threading.Event()
    print(f"[START] {name}", flush=True)

    def heartbeat():
        while not stopped.wait(heartbeat_seconds):
            elapsed = time.perf_counter() - started
            print(f"[WORKING] {name} | elapsed={elapsed:.0f}s", flush=True)

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        elapsed = time.perf_counter() - started
        print(f"[FAILED] {name} | elapsed={elapsed:.1f}s", flush=True)
        raise
    finally:
        stopped.set(); worker.join(timeout=1.0)
    elapsed = time.perf_counter() - started
    print(f"[DONE] {name} | elapsed={elapsed:.1f}s", flush=True)


def forward_model(model, batch, adapter, *, need_weights=False):
    return model(
        batch["values"], batch["mask"], batch["donor_static"], batch["geometry"],
        batch["donor_padding_mask"], batch["target_static"], batch["calendar"],
        batch["current_pm_physical"], adapter.pm_mean, need_weights=need_weights,
    )


def optimizer_for(model, device):
    kwargs = {"lr": 5e-4, "weight_decay": 1e-4}
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs), False


def lr_lambda(step, total_steps):
    warmup = max(1, math.ceil(0.5 * total_steps / MAX_EPOCHS))
    if step < warmup:
        return max((step + 1) / warmup, 1e-3)
    progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def station_metrics(y, prediction, station_ids, static):
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


@torch.inference_mode()
def validate(model, dataset, builder, adapter, static, device, description):
    model.eval(); dtype = amp_dtype_for(device)
    collected = {key: [] for key in (
        "y", "prediction", "station", "time", "field", "offset", "anomaly", "anchor",
    )}
    for start in tqdm(
        range(0, len(dataset), VALIDATION_BATCH_SIZE), desc=description,
        unit="batch", dynamic_ncols=True,
    ):
        idx = np.arange(start, min(start + VALIDATION_BATCH_SIZE, len(dataset)))
        batch = adapter.prepare(builder(raw_batch(dataset.row_targets[idx], dataset.row_times[idx])))
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            prediction, aux = forward_model(model, batch, adapter)
        tensors = {
            "y": batch["label"], "prediction": prediction,
            "station": batch["target_idx"], "time": batch["time_idx"],
            "field": aux["spatial_field"], "offset": aux["static_offset"],
            "anomaly": aux["anomaly"], "anchor": aux["has_anchor"],
        }
        for key, value in tensors.items():
            collected[key].append(value.float().cpu().numpy())
    values = {key: np.concatenate(parts) for key, parts in collected.items()}
    if not all(np.isfinite(values[key]).all() for key in ("y", "prediction", "field", "offset", "anomaly")):
        raise RuntimeError("validation produced non-finite values")
    per_station = station_metrics(values["y"], values["prediction"], values["station"], static)
    overall = regression_metrics(values["y"], values["prediction"])
    overall.update(event_metrics(values["y"], values["prediction"]))
    overall.update({
        "macro_rmse": float(per_station.rmse.mean()),
        "macro_r2": float(per_station.r2.mean()),
        "mean_abs_bias": float(per_station.bias.abs().mean()),
        "anchorless_rate": float((values["anchor"] < 0.5).mean()),
        "mean_abs_static_offset": float(np.abs(values["offset"]).mean()),
        "mean_abs_anomaly": float(np.abs(values["anomaly"]).mean()),
    })
    prediction_output = {
        "y": values["y"].astype("float32"),
        "prediction": values["prediction"].astype("float32"),
        "station_index": values["station"].astype("int16"),
        "time_index": values["time"].astype("int32"),
        "spatial_field": values["field"].astype("float32"),
        "static_offset": values["offset"].astype("float32"),
        "anomaly": values["anomaly"].astype("float32"),
    }
    return overall, per_station, prediction_output


def sanity(model, batch, adapter):
    model.train(); model.zero_grad(set_to_none=True)
    prediction, aux = forward_model(model, batch, adapter)
    loss, parts = decomposed_loss(
        prediction, aux, batch["label"], batch["target_idx"], adapter.pm_std,
        level_weight=LEVEL_WEIGHT, anomaly_center_weight=ANOMALY_CENTER_WEIGHT,
    )
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
        raise RuntimeError("sanity produced non-finite values")
    for prefix in ("field_gate", "offset_head", "dynamic_encoder", "temporal", "anomaly_head"):
        gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"finite-gradient check failed: {prefix}")
    if not bool(aux["field_inside_donor_range"].all()):
        raise RuntimeError("convex spatial field escaped donor PM range")
    return {
        "loss": float(loss), "parts": {k: float(v) for k, v in parts.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def benchmark(train_ds, val_ds, builder, adapter, device, seed):
    seed_all(seed)
    schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=seed)
    model = DecomposedFieldNowcaster(dropout=.10).to(device)
    optimizer, fused = optimizer_for(model, device)
    dtype = amp_dtype_for(device); scaler = make_grad_scaler(dtype == torch.float16)
    train_times = []
    train_progress = tqdm(range(60), desc="preflight train benchmark", unit="batch", dynamic_ncols=True)
    for iteration in train_progress:
        started = time.perf_counter(); item = schedule.next_batch()
        batch = adapter.prepare(builder(raw_batch(item.station_indices, item.time_indices)))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            prediction, aux = forward_model(model, batch, adapter)
            loss, _ = decomposed_loss(
                prediction, aux, batch["label"], batch["target_idx"], adapter.pm_std,
                level_weight=LEVEL_WEIGHT, anomaly_center_weight=ANOMALY_CENTER_WEIGHT,
            )
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        if device.type == "cuda": torch.cuda.synchronize(device)
        if iteration >= 10: train_times.append(time.perf_counter() - started)
    eval_times = []
    model.eval()
    with torch.inference_mode():
        eval_progress = tqdm(range(35), desc="preflight validation benchmark", unit="batch", dynamic_ncols=True)
        for iteration in eval_progress:
            start = (iteration * VALIDATION_BATCH_SIZE) % len(val_ds)
            idx = np.arange(start, min(start + VALIDATION_BATCH_SIZE, len(val_ds)))
            started = time.perf_counter()
            batch = adapter.prepare(builder(raw_batch(val_ds.row_targets[idx], val_ds.row_times[idx])))
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                forward_model(model, batch, adapter)
            if device.type == "cuda": torch.cuda.synchronize(device)
            if iteration >= 5: eval_times.append(time.perf_counter() - started)
    result = {
        "warmup_train_batches": 10, "timed_train_batches": 50,
        "timed_validation_batches": 30, "batch_size": 256,
        "steps_per_epoch": len(schedule), "samples_per_station_per_epoch": 8192,
        "train_batch_p50_seconds": float(np.median(train_times)),
        "train_batch_p95_seconds": float(np.percentile(train_times, 95)),
        "validation_batch_p95_seconds": float(np.percentile(eval_times, 95)),
        "fused_adamw": fused,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    del model, optimizer, scaler
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def checkpoint_payload(fold, model, optimizer, scheduler, schedule, history,
                       station_history, global_step, planned_stop_epoch):
    return {
        "revision": REVISION, "fold": fold, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "schedule": schedule.state_dict(), "history": history,
        "station_history": station_history, "global_step": global_step,
        "planned_stop_epoch": planned_stop_epoch, "rng": rng_state(),
    }


def save_resume(payload, local_path, drive_path):
    atomic_torch(payload, local_path)
    if drive_path is not None:
        sync_file(local_path, drive_path)


def train_fold(fold, train_idx, val_idx, cube, timestamps, static, static_cols,
               distance, device, local_root, drive_root):
    fold_started = time.perf_counter()
    print(f"\n===== FOLD {fold} START | train=60 validation=12 | max_epoch=20 =====", flush=True)
    with visible_stage(f"fold {fold}: fit train-only scalers"):
        scaler = fit_train_only_scaler(cube, timestamps, static, static_cols, train_idx)
        scaled = standardize_static(static, static_cols, scaler)
    with visible_stage(f"fold {fold}: build compact train/validation row indices"):
        train_ds = ColdStartStationDataset(
            train_idx, train_idx, CFG.train_start, CFG.train_end, cube, timestamps,
            static, scaled, distance, scaler,
        )
        val_ds = ColdStartStationDataset(
            val_idx, train_idx, CFG.train_start, CFG.train_end, cube, timestamps,
            static, scaled, distance, scaler,
        )
    print(f"[ROWS] fold={fold} train={len(train_ds):,} validation={len(val_ds):,}", flush=True)
    hidden = np.setdiff1d(np.arange(len(static)), train_idx)
    with visible_stage(f"fold {fold}: precompute GPU feature tables"):
        builder = DeviceFeatureBuilder(
            train_idx, cube, max(int(train_ds.row_times.max()), int(val_ds.row_times.max())),
            timestamps, static, scaled, distance, scaler, hidden, device,
        )
    print(f"[GPU TABLES] fold={fold} seconds={builder.precompute_seconds:.1f} MiB={builder.precomputed_table_mb:.1f}", flush=True)
    adapter = V8Adapter(timestamps, scaler, device)
    schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=CFG.seed + fold * 1009)
    model = DecomposedFieldNowcaster(dropout=.10).to(device)
    optimizer, fused = optimizer_for(model, device)
    total_steps = MAX_EPOCHS * len(schedule)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, total_steps))
    dtype = amp_dtype_for(device); grad_scaler = make_grad_scaler(dtype == torch.float16)

    fold_dir = local_root / f"fold_{fold:02d}"; fold_dir.mkdir(parents=True, exist_ok=True)
    local_resume = fold_dir / "active_resume.pt"
    drive_fold = drive_root / fold_dir.name if drive_root else None
    drive_resume = drive_fold / "active_resume.pt" if drive_fold else None
    local_best = fold_dir / "best_model.pt"
    drive_best = drive_fold / "best_model.pt" if drive_fold else None
    history, station_history, global_step = [], [], 0
    planned_stop_epoch = CONVERGENCE_CHECK_EPOCH
    resume = drive_resume if drive_resume is not None and drive_resume.exists() else local_resume
    if resume.exists():
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload.get("revision") != REVISION or payload.get("fold") != fold:
            raise RuntimeError(f"resume protocol mismatch: {resume}")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"]); schedule.load_state_dict(payload["schedule"])
        history = payload["history"]; station_history = payload.get("station_history", [])
        global_step = int(payload["global_step"])
        planned_stop_epoch = int(payload.get("planned_stop_epoch", CONVERGENCE_CHECK_EPOCH))
        restore_rng(payload["rng"])
        print(f"RESUME fold={fold} next_epoch={schedule.epoch + 1} step={global_step}", flush=True)
    else:
        probe = schedule.next_batch()
        batch = adapter.prepare(builder(raw_batch(probe.station_indices, probe.time_indices)))
        print(json.dumps({"fold": fold, "sanity": sanity(model, batch, adapter)}, ensure_ascii=False), flush=True)
        schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=CFG.seed + fold * 1009)
        optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    best_macro = min((row["macro_rmse"] for row in history), default=float("inf"))
    while schedule.epoch < planned_stop_epoch:
        epoch = schedule.epoch + 1; epoch_started = time.perf_counter()
        loss_sums = {"total": 0.0, "main_mse": 0.0, "level_mse": 0.0, "anomaly_center_mse": 0.0}
        seen = 0; model.train()
        completed_in_epoch = schedule.station_round * schedule.batches_per_station_round + schedule.batch_in_round
        remaining_batches = len(schedule) - completed_in_epoch
        print(
            f"[EPOCH START] fold={fold} epoch={epoch}/{planned_stop_epoch} "
            f"remaining_batches={remaining_batches:,}", flush=True,
        )
        progress = tqdm(
            schedule.iter_epoch(), total=remaining_batches,
            desc=f"fold {fold} epoch {epoch}", unit="batch", dynamic_ncols=True,
        )
        for item in progress:
            batch = adapter.prepare(builder(raw_batch(item.station_indices, item.time_indices)))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                prediction, aux = forward_model(model, batch, adapter)
                loss, parts = decomposed_loss(
                    prediction, aux, batch["label"], batch["target_idx"], adapter.pm_std,
                    level_weight=LEVEL_WEIGHT, anomaly_center_weight=ANOMALY_CENTER_WEIGHT,
                )
            if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
            grad_scaler.scale(loss).backward(); grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            grad_scaler.step(optimizer); grad_scaler.update(); scheduler.step()
            n = int(batch["label"].numel()); seen += n; global_step += 1
            loss_sums["total"] += float(loss.detach()) * n
            for key, value in parts.items(): loss_sums[key] += float(value) * n
            if global_step % 25 == 0:
                progress.set_postfix(loss=f"{loss_sums['total']/seen:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2g}")
            if global_step % CHECKPOINT_EVERY_STEPS == 0:
                save_resume(checkpoint_payload(
                    fold, model, optimizer, scheduler, schedule, history,
                    station_history, global_step, planned_stop_epoch,
                ), local_resume, drive_resume)
        metrics, per_station, _ = validate(
            model, val_ds, builder, adapter, static, device, f"fold {fold} epoch {epoch} validation",
        )
        nearest = np.asarray([distance[int(s), train_idx].min() for s in per_station.station_index])
        sparse = nearest >= np.quantile(nearest, .75)
        metrics["sparse_quartile_macro_rmse"] = float(per_station.loc[sparse, "rmse"].mean())
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / max(seen, 1) for key, value in loss_sums.items()},
            "runtime_seconds": time.perf_counter() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"], **metrics,
        }
        history.append(row)
        station_history.extend({"epoch": epoch, **record} for record in per_station.to_dict("records"))
        print(json.dumps({"fold": fold, **row}, ensure_ascii=False), flush=True)
        print(
            f"[EPOCH DONE] fold={fold} epoch={epoch} "
            f"macro_rmse={metrics['macro_rmse']:.4f} r2={metrics['r2']:.4f} "
            f"bias={metrics['bias']:.4f} seconds={row['runtime_seconds']:.1f}",
            flush=True,
        )
        if metrics["macro_rmse"] < best_macro:
            best_macro = metrics["macro_rmse"]
            best = {
                "revision": REVISION, "fold": fold, "epoch": epoch,
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "metrics": metrics, "train_indices": train_idx, "validation_indices": val_idx,
            }
            atomic_torch(best, local_best)
            if drive_best is not None: sync_file(local_best, drive_best)
        if epoch == CONVERGENCE_CHECK_EPOCH:
            before = min(x["macro_rmse"] for x in history if x["epoch"] <= INITIAL_EPOCHS)
            recent = min(x["macro_rmse"] for x in history if INITIAL_EPOCHS < x["epoch"] <= CONVERGENCE_CHECK_EPOCH)
            planned_stop_epoch = MAX_EPOCHS if (before - recent) / before >= MATERIAL_IMPROVEMENT_FRACTION else CONVERGENCE_CHECK_EPOCH
            print(json.dumps({"fold": fold, "convergence_check": 18, "planned_stop_epoch": planned_stop_epoch}, ensure_ascii=False), flush=True)
        save_resume(checkpoint_payload(
            fold, model, optimizer, scheduler, schedule, history,
            station_history, global_step, planned_stop_epoch,
        ), local_resume, drive_resume)

    if not local_best.exists() and drive_best is not None and drive_best.exists():
        sync_file(drive_best, local_best)
    best = torch.load(local_best, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    metrics, per_station, prediction = validate(
        model, val_ds, builder, adapter, static, device, f"fold {fold} final validation",
    )
    nearest = np.asarray([distance[int(s), train_idx].min() for s in per_station.station_index])
    sparse = nearest >= np.quantile(nearest, .75)
    metrics["sparse_quartile_macro_rmse"] = float(per_station.loc[sparse, "rmse"].mean())
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(station_history).to_csv(fold_dir / "station_metrics_history.csv", index=False, encoding="utf-8-sig")
    per_station.to_csv(fold_dir / "best_station_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(fold_dir / "best_predictions.npz", **prediction)
    result = {
        "fold": fold, "best_epoch": int(best["epoch"]), "epochs_completed": int(schedule.epoch),
        "metrics": metrics, "runtime_seconds": time.perf_counter() - fold_started,
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
        "fused_adamw": fused,
    }
    atomic_json(result, fold_dir / "fold_result.json")
    print(
        f"===== FOLD {fold} DONE | best_epoch={result['best_epoch']} "
        f"macro_rmse={metrics['macro_rmse']:.4f} runtime={result['runtime_seconds']/60:.1f}min =====",
        flush=True,
    )
    # A completed fold keeps only research-essential compact artefacts on Drive.
    if drive_fold is not None:
        drive_fold.mkdir(parents=True, exist_ok=True)
        for name in (
            "best_model.pt", "training_history.csv", "station_metrics_history.csv",
            "best_station_metrics.csv", "best_predictions.npz", "fold_result.json",
        ):
            sync_file(fold_dir / name, drive_fold / name)
        if drive_resume.exists(): drive_resume.unlink()
    if local_resume.exists(): local_resume.unlink()
    return result


def evaluate_gate(results, local_root, drive_root):
    baseline = pd.read_csv(BASELINE_PATH, dtype={"siteid": str})
    comparisons = []
    checks = []
    for result in results:
        fold = int(result["fold"])
        current = pd.read_csv(local_root / f"fold_{fold:02d}" / "best_station_metrics.csv", dtype={"siteid": str})
        old = baseline.loc[baseline.fold == fold].copy()
        merged = old.merge(current, on="station_index", suffixes=("_v8", "_new"), validate="one_to_one")
        if len(merged) != 12: raise RuntimeError(f"fold {fold} baseline station mismatch")
        merged.insert(0, "fold", fold)
        merged["rmse_change_fraction"] = merged.rmse_new / merged.rmse_v8 - 1.0
        comparisons.append(merged)
        old_macro = float(old.rmse.mean()); old_bias = float(old.bias.abs().mean())
        old_sparse = {0: 6.0567177520099955, 3: 5.051381758251864}[fold]
        now = result["metrics"]
        checks.extend([
            {"fold": fold, "criterion": "macro_rmse_improves_5pct", "pass": now["macro_rmse"] <= .95 * old_macro,
             "new": now["macro_rmse"], "v8": old_macro},
            {"fold": fold, "criterion": "mean_abs_bias_not_worse", "pass": now["mean_abs_bias"] <= old_bias,
             "new": now["mean_abs_bias"], "v8": old_bias},
            {"fold": fold, "criterion": "sparse_quartile_not_worse", "pass": now["sparse_quartile_macro_rmse"] <= old_sparse,
             "new": now["sparse_quartile_macro_rmse"], "v8": old_sparse},
            {"fold": fold, "criterion": "no_station_worsens_over_10pct", "pass": bool((merged.rmse_change_fraction <= .10).all()),
             "worst_change_fraction": float(merged.rmse_change_fraction.max())},
        ])
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(local_root / "comparison_to_v8.csv", index=False, encoding="utf-8-sig")
    status = "GO" if all(item["pass"] for item in checks) else "NO_GO"
    summary = {
        "status": status, "revision": REVISION, "checks": checks, "results": results,
        "full73_authorized": False,
        "note": "GO only permits a separate decision; this runner never starts 73-station training.",
    }
    atomic_json(summary, local_root / "pilot_summary.json")
    if drive_root is not None:
        sync_file(local_root / "comparison_to_v8.csv", drive_root / "comparison_to_v8.csv")
        sync_file(local_root / "pilot_summary.json", drive_root / "pilot_summary.json")
    return summary


def main():
    seed_all(CFG.seed); runtime = apply_runtime_profile(CFG)
    if CFG.device.type != "cuda": raise RuntimeError("pilot requires a CUDA Colab runtime")
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    local_root = Path(os.environ.get("DECOMPOSED_FIELD_OUTPUT", "/content/DL_DECOMPOSED_FIELD_PILOT"))
    if "/drive/" in str(local_root): raise RuntimeError("working output must stay under /content")
    drive_base = Path("/content/drive/MyDrive")
    drive_root = Path(os.environ.get(
        "DECOMPOSED_FIELD_DRIVE_OUTPUT", str(drive_base / "DL_DECOMPOSED_FIELD_PILOT")
    )) if drive_base.is_dir() else None
    local_root.mkdir(parents=True, exist_ok=True)
    if drive_root is not None: drive_root.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "status": "STARTING", "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "plan": "preflight -> fold0 -> fold3 -> locked V8 comparison",
        "epochs": "15 recorded; convergence check at 18; hard cap 20",
        "drive_output": str(drive_root) if drive_root is not None else None,
    }, ensure_ascii=False, indent=2), flush=True)
    with visible_stage("load static features and construct fixed 60/12 folds"):
        static, clusters, cols = load_static(); outer = target_index(static, CFG.target_site)
        splits = make_meta_crossfit_folds(clusters, outer)
        distance = haversine_matrix(static.longitude, static.latitude)
    with visible_stage("load or build hourly AQ cube"):
        cube, timestamps = build_or_load_hourly_cube(static)

    # Exact-path preflight on fold 0 before any training.
    train_idx, val_idx = splits[0]
    with visible_stage("preflight: fit fold-0 train-only scalers"):
        scaler = fit_train_only_scaler(cube, timestamps, static, cols, train_idx)
        scaled = standardize_static(static, cols, scaler)
    with visible_stage("preflight: build compact row indices"):
        train_ds = ColdStartStationDataset(train_idx, train_idx, CFG.train_start, CFG.train_end, cube, timestamps, static, scaled, distance, scaler)
        val_ds = ColdStartStationDataset(val_idx, train_idx, CFG.train_start, CFG.train_end, cube, timestamps, static, scaled, distance, scaler)
    print(f"[ROWS] preflight train={len(train_ds):,} validation={len(val_ds):,}", flush=True)
    hidden = np.setdiff1d(np.arange(len(static)), train_idx)
    with visible_stage("preflight: precompute GPU feature tables"):
        builder = DeviceFeatureBuilder(train_idx, cube, max(int(train_ds.row_times.max()), int(val_ds.row_times.max())), timestamps, static, scaled, distance, scaler, hidden, CFG.device)
    adapter = V8Adapter(timestamps, scaler, CFG.device)
    benchmark_started = time.perf_counter()
    with visible_stage("preflight: exact forward/backward and validation benchmark"):
        measured = benchmark(train_ds, val_ds, builder, adapter, CFG.device, CFG.seed + 9100)
    benchmark_seconds = time.perf_counter() - benchmark_started
    steps_per_epoch = int(measured["steps_per_epoch"])
    validation_batches = math.ceil(len(val_ds) / VALIDATION_BATCH_SIZE)
    fold20 = MAX_EPOCHS * (
        steps_per_epoch * measured["train_batch_p95_seconds"]
        + validation_batches * measured["validation_batch_p95_seconds"]
    ) + builder.precompute_seconds + 60.0
    two_fold20 = benchmark_seconds + 2 * fold20
    profile = {
        "revision": REVISION, "runtime": runtime, "folds": list(FOLDS),
        "protocol": {"train_stations": 60, "validation_stations": 12,
                     "training_donors": 59, "validation_donors": 60,
                     "history_hours": 24, "dynamic_channels": 11,
                     "static_features": 49, "target_dynamic_inputs": 0},
        "epoch_policy": {"record_at": 15, "check_at": 18, "maximum": 20},
        "loss": {"main": 1.0, "level": LEVEL_WEIGHT, "anomaly_center": ANOMALY_CENTER_WEIGHT},
        "benchmark": measured,
        "cost_projection": {"benchmark_seconds": benchmark_seconds,
                            "one_fold_worst20_seconds": fold20,
                            "two_fold_worst20_seconds": two_fold20,
                            "two_fold_worst20_a100_hours": two_fold20 / 3600,
                            "available_a100_hours": AVAILABLE_A100_SECONDS / 3600},
    }
    atomic_json(profile, local_root / "preflight.json")
    if drive_root is not None: sync_file(local_root / "preflight.json", drive_root / "preflight.json")
    print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)
    print(
        f"[PREFLIGHT COST] two-fold worst20={two_fold20/60:.1f}min "
        f"| limit={PILOT_BUDGET_SECONDS/60:.0f}min",
        flush=True,
    )
    del builder, adapter, train_ds, val_ds
    torch.cuda.empty_cache()
    if two_fold20 > PILOT_BUDGET_SECONDS or two_fold20 > AVAILABLE_A100_SECONDS:
        aborted = {"status": "ABORTED_BEFORE_TRAINING", "reason": "measured exact-path cost exceeds locked pilot budget", **profile}
        atomic_json(aborted, local_root / "pilot_summary.json")
        if drive_root is not None: sync_file(local_root / "pilot_summary.json", drive_root / "pilot_summary.json")
        print(json.dumps(aborted, ensure_ascii=False, indent=2)); return

    results = []
    for fold in FOLDS:
        print(f"[QUEUE] next fold={fold} | completed={len(results)}/{len(FOLDS)}", flush=True)
        completed = drive_root / f"fold_{fold:02d}" / "fold_result.json" if drive_root is not None else local_root / f"fold_{fold:02d}" / "fold_result.json"
        if completed.exists():
            local_fold = local_root / f"fold_{fold:02d}"
            if drive_root is not None and not local_fold.exists(): shutil.copytree(completed.parent, local_fold)
            results.append(json.loads(completed.read_text(encoding="utf-8")))
            print(f"SKIP completed fold {fold}", flush=True); continue
        train_idx, val_idx = splits[fold]
        results.append(train_fold(
            fold, train_idx, val_idx, cube, timestamps, static, cols,
            distance, CFG.device, local_root, drive_root,
        ))
    summary = evaluate_gate(results, local_root, drive_root)
    print(f"[ALL DONE] status={summary['status']} | output={local_root}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

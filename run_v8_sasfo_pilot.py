"""Budget-gated two-fold V8 SASFO pilot (strict unseen-station protocol)."""
from __future__ import annotations

import copy
import hashlib
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
from data_pipeline import (ColdStartStationDataset, build_or_load_hourly_cube,
    fit_train_only_scaler, haversine_matrix, load_static,
    make_meta_crossfit_folds, standardize_static)
from model_v8 import SASFO
from train_formal import (DeviceFeatureBuilder, amp_dtype_for, make_grad_scaler,
    regression_metrics, seed_all)
from v5_training import event_metrics
from v7_training import calendar6
from v8_sampling import BalancedStationTimeSchedule

REVISION = "v8_sasfo_convergence_budget_2"
FOLDS = (0, 3)
INITIAL_EPOCHS = 15
CONVERGENCE_CHECK_EPOCH = 18
MAX_EPOCHS = 20
BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 512
PILOT_PREFLIGHT_LIMIT_SECONDS = 4 * 60 * 60
TOTAL_A100_BUDGET_SECONDS = 60 * 60 * 60
CHECKPOINT_EVERY_STEPS = 100
MATERIAL_IMPROVEMENT_FRACTION = 0.0025


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_torch(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def sync_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def target_index(static, value):
    found = np.flatnonzero(
        (static.siteid.astype(str).to_numpy() == str(value))
        | (static.sitename.astype(str).to_numpy() == str(value))
    )
    if len(found) != 1:
        raise ValueError(f"outer target {value!r} matched {len(found)} stations")
    return int(found[0])


def raw_batch(targets, times):
    return {
        "target_idx": torch.as_tensor(targets, dtype=torch.long),
        "time_idx": torch.as_tensor(times, dtype=torch.long),
    }


class V8Adapter:
    def __init__(self, timestamps, scaler, device):
        self.device = device
        self.calendar = torch.as_tensor(calendar6(timestamps), device=device)
        self.offsets = torch.arange(-23, 1, device=device)
        pm = CFG.raw_dynamic_items.index("PM2.5")
        self.pm_mean = float(scaler.dynamic_mean[pm])
        self.pm_std = float(scaler.dynamic_std[pm])

    def prepare(self, batch):
        batch = dict(batch)
        history = batch["time_idx"][:, None] + self.offsets[None]
        batch["calendar"] = self.calendar[history]
        pmz = batch["values"][:, :, -1, CFG.raw_dynamic_items.index("PM2.5")]
        pmm = batch["mask"][:, :, -1, CFG.raw_dynamic_items.index("PM2.5")] > 0
        raw = pmz * self.pm_std + self.pm_mean
        batch["current_pm_physical"] = torch.where(pmm, raw, torch.full_like(raw, torch.nan))
        return batch


def forward_model(model, batch, adapter):
    return model(
        batch["values"], batch["mask"], batch["donor_static"], batch["geometry"],
        batch["donor_padding_mask"], batch["target_static"], batch["calendar"],
        batch["current_pm_physical"], adapter.pm_mean, adapter.pm_std,
    )


def valid_times_by_station(dataset):
    answer = {}
    for station in dataset.targets:
        answer[int(station)] = dataset.row_times[dataset.row_targets == station].astype("int64")
    return answer


def station_metrics(y, prediction, station_ids, static):
    rows = []
    for station in np.unique(station_ids):
        keep = station_ids == station
        rows.append({
            "station_index": int(station), "siteid": str(static.loc[station, "siteid"]),
            "sitename": str(static.loc[station, "sitename"]), "n": int(keep.sum()),
            **regression_metrics(y[keep], prediction[keep]),
        })
    return pd.DataFrame(rows)


@torch.inference_mode()
def validate(model, dataset, builder, adapter, static, device):
    model.eval(); dtype = amp_dtype_for(device)
    ys, ps, ss, tt, anchors = [], [], [], [], []
    for start in tqdm(range(0, len(dataset), VALIDATION_BATCH_SIZE),
                      desc="V8 full validation", unit="batch", dynamic_ncols=True):
        stop = min(start + VALIDATION_BATCH_SIZE, len(dataset))
        idx = np.arange(start, stop)
        batch = adapter.prepare(builder(raw_batch(dataset.row_targets[idx], dataset.row_times[idx])))
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            pred, aux = forward_model(model, batch, adapter)
        if not torch.isfinite(pred).all():
            raise RuntimeError("non-finite V8 validation prediction")
        ys.append(batch["label"].float().cpu().numpy())
        ps.append(pred.float().cpu().numpy())
        ss.append(batch["target_idx"].cpu().numpy())
        tt.append(batch["time_idx"].cpu().numpy())
        anchors.append(aux["has_anchor"].cpu().numpy())
    y, pred, stations, times, has_anchor = map(np.concatenate, (ys, ps, ss, tt, anchors))
    per_station = station_metrics(y, pred, stations, static)
    overall = regression_metrics(y, pred)
    overall.update(event_metrics(y, pred))
    overall.update({
        "macro_rmse": float(per_station.rmse.mean()),
        "macro_r2": float(per_station.r2.mean()),
        "mean_abs_bias": float(per_station.bias.abs().mean()),
        "anchorless_rate": float((~has_anchor).mean()),
    })
    return overall, per_station, {"y": y.astype("float32"), "prediction": pred.astype("float32"),
        "station_index": stations.astype("int16"), "time_index": times.astype("int32")}


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


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu"))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8, device="cpu") for x in state["cuda"]])


def checkpoint_payload(fold, model, optimizer, scheduler, schedule, history, station_history,
                       loss_sum, seen, global_step, planned_stop_epoch):
    return {"revision": REVISION, "fold": fold, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "schedule": schedule.state_dict(), "history": history, "station_history": station_history,
        "loss_sum": loss_sum, "seen": seen, "global_step": global_step,
        "planned_stop_epoch": planned_stop_epoch, "rng": rng_state()}


def save_checkpoint(payload, local_path, drive_path):
    atomic_torch(payload, local_path)
    if drive_path is not None:
        sync_file(local_path, drive_path)


def sanity(model, batch, adapter):
    model.train(); model.zero_grad(set_to_none=True)
    prediction, aux = forward_model(model, batch, adapter)
    loss = ((prediction.float() - batch["label"].float()) / adapter.pm_std).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
        raise RuntimeError("V8 sanity produced non-finite values")
    for prefix in ("dynamic_encoder", "field_logits", "temporal", "static_encoder", "correction_head"):
        gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"V8 sanity gradient failed: {prefix}")
    anchored = aux["has_anchor"]
    if anchored.any():
        if not torch.allclose(aux["skip_weight_sum"][anchored], torch.ones_like(aux["skip_weight_sum"][anchored]), atol=1e-5):
            raise RuntimeError("signed skip weights do not sum to one")
        if bool((aux["skip_l1"][anchored] > 3.00001).any()):
            raise RuntimeError("signed skip L1 bound exceeded")
    return {"loss": float(loss), "parameters": sum(p.numel() for p in model.parameters())}


def benchmark(train_ds, val_ds, train_builder, eval_builder, adapter, device,
              train_donors, eval_donors, seed):
    seed_all(seed)
    schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=seed)
    model = SASFO(dropout=.10).to(device); optimizer, fused = optimizer_for(model, device)
    dtype = amp_dtype_for(device); scaler = make_grad_scaler(dtype == torch.float16)
    train_times = []
    for iteration in range(110):
        started = time.perf_counter(); item = schedule.next_batch()
        batch = adapter.prepare(train_builder(raw_batch(item.station_indices, item.time_indices)))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            pred, _ = forward_model(model, batch, adapter)
            loss = ((pred.float() - batch["label"].float()) / adapter.pm_std).square().mean()
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        if device.type == "cuda": torch.cuda.synchronize(device)
        if iteration >= 10: train_times.append(time.perf_counter() - started)
    eval_times = []
    model.eval()
    with torch.inference_mode():
        for iteration in range(110):
            start = (iteration * VALIDATION_BATCH_SIZE) % len(val_ds)
            idx = np.arange(start, min(start + VALIDATION_BATCH_SIZE, len(val_ds)))
            started = time.perf_counter()
            batch = adapter.prepare(eval_builder(raw_batch(val_ds.row_targets[idx], val_ds.row_times[idx])))
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                forward_model(model, batch, adapter)
            if device.type == "cuda": torch.cuda.synchronize(device)
            if iteration >= 10: eval_times.append(time.perf_counter() - started)
    result = {"warmup_batches": 10, "timed_batches": 100, "train_donors": train_donors,
        "eval_donors": eval_donors, "batch_size": BATCH_SIZE,
        "steps_per_epoch": len(schedule), "samples_per_station_per_epoch": 8192,
        "train_batch_p50_seconds": float(np.median(train_times)),
        "train_batch_p95_seconds": float(np.percentile(train_times, 95)),
        "eval_batch_p95_seconds": float(np.percentile(eval_times, 95)),
        "fused_adamw": fused, "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2}
    del model, optimizer, scaler; torch.cuda.empty_cache()
    return result


def load_baseline(fold, val_idx, val_ds, train_idx, distance, cube):
    root_value = os.environ.get("V8_MATCHED_BASELINE_ROOT", "").strip()
    if not root_value:
        return None
    root = Path(root_value)
    manifest_path = root / "protocol_manifest.json"
    path = root / f"fold_{fold:02d}" / "station_metrics_history.csv"
    if not manifest_path.exists() or not path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"train_stations": 60, "validation_stations": 12,
        "training_donors": 59, "validation_donors": 60,
        "history_hours": 24, "dynamic_channels": 11,
        "train_start": CFG.train_start, "train_end": CFG.train_end,
        "updates_per_epoch": 1920, "samples_per_station_per_epoch": 8192,
        "checkpoint_selection": "minimum_macro_station_rmse"}
    mismatches = {key: (manifest.get(key), value) for key, value in required.items()
                  if manifest.get(key) != value}
    selected_epoch = manifest.get("selected_epoch_by_fold", {}).get(str(fold))
    if mismatches or selected_epoch is None:
        return None
    frame = pd.read_csv(path, dtype={"siteid": str})
    selected = frame.loc[frame.epoch == int(selected_epoch)].copy()
    expected = set(map(int, val_idx)); actual = set(map(int, selected.station_index))
    if expected != actual:
        raise RuntimeError(f"V6 baseline fold {fold} station mismatch: expected={sorted(expected)} actual={sorted(actual)}")
    counts = pd.Series(val_ds.row_targets).value_counts().to_dict()
    if any(int(row.n) != int(counts[int(row.station_index)]) for row in selected.itertuples()):
        raise RuntimeError(f"V6 baseline fold {fold} target-label counts no longer match")
    nearest = np.asarray([distance[int(station), train_idx].min() for station in selected.station_index])
    sparse = nearest >= np.quantile(nearest, .75)
    pidx = CFG.aq_cube_items.index("PM2.5")
    y = np.asarray(cube[val_ds.row_times, val_ds.row_targets, pidx], dtype="float64")
    sse = float(np.sum(selected.n.to_numpy() * selected.rmse.to_numpy() ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"fold": fold, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": int(selected_epoch), "protocol_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "updates_per_epoch": manifest.get("updates_per_epoch"),
        "samples_per_station_per_epoch": manifest.get("samples_per_station_per_epoch"),
        "stopping_rule": manifest.get("stopping_rule"),
        "checkpoint_selection": manifest.get("checkpoint_selection"),
        "macro_rmse": float(selected.rmse.mean()),
        "mean_abs_bias": float(selected.bias.abs().mean()),
        "sparse_quartile_macro_rmse": float(selected.loc[sparse, "rmse"].mean()),
        "pooled_r2_exact": float(1 - sse / sst), "baseline_sse": sse,
        "truth_n": int(y.size), "truth_sum": float(y.sum()), "truth_sum_squares": float(np.square(y).sum()),
        "station_count": len(selected)}


def train_fold(fold, train_idx, val_idx, cube, timestamps, static, cols, distance,
               device, local_root, drive_root, global_started):
    fold_started = time.perf_counter()
    scaler = fit_train_only_scaler(cube, timestamps, static, cols, train_idx)
    scaled = standardize_static(static, cols, scaler)
    train_ds = ColdStartStationDataset(train_idx, train_idx, CFG.train_start, CFG.train_end,
        cube, timestamps, static, scaled, distance, scaler)
    val_ds = ColdStartStationDataset(val_idx, train_idx, CFG.train_start, CFG.train_end,
        cube, timestamps, static, scaled, distance, scaler)
    hidden = np.setdiff1d(np.arange(len(static)), train_idx)
    builder = DeviceFeatureBuilder(train_idx, cube, max(int(train_ds.row_times.max()), int(val_ds.row_times.max())),
        timestamps, static, scaled, distance, scaler, hidden, device)
    adapter = V8Adapter(timestamps, scaler, device)
    schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=CFG.seed + fold * 1009)
    model = SASFO(dropout=.10).to(device); optimizer, fused = optimizer_for(model, device)
    total_steps = MAX_EPOCHS * len(schedule)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, total_steps))
    dtype = amp_dtype_for(device); grad_scaler = make_grad_scaler(dtype == torch.float16)
    local_ckpt = local_root / "active_last.pt"
    drive_ckpt = drive_root / "active_last.pt" if drive_root else None
    fold_dir = local_root / f"fold_{fold:02d}"; fold_dir.mkdir(parents=True, exist_ok=True)
    local_best = fold_dir / "best_model.pt"
    drive_best = drive_root / fold_dir.name / "best_model.pt" if drive_root else None
    history, station_history, loss_sum, seen, global_step = [], [], 0.0, 0, 0
    planned_stop_epoch = CONVERGENCE_CHECK_EPOCH
    resume = drive_ckpt if drive_ckpt and drive_ckpt.exists() else local_ckpt
    if resume.exists():
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload.get("revision") != REVISION or payload.get("fold") != fold:
            raise RuntimeError(f"checkpoint protocol mismatch: {resume}")
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"]); schedule.load_state_dict(payload["schedule"])
        history = payload["history"]; station_history = payload.get("station_history", [])
        loss_sum, seen, global_step = payload["loss_sum"], payload["seen"], payload["global_step"]
        planned_stop_epoch = int(payload.get("planned_stop_epoch", CONVERGENCE_CHECK_EPOCH))
        restore_rng(payload["rng"]); print(f"RESUME fold={fold} epoch={schedule.epoch+1} step={global_step}", flush=True)
    else:
        probe = schedule.next_batch()
        batch = adapter.prepare(builder(raw_batch(probe.station_indices, probe.time_indices)))
        check = sanity(model, batch, adapter); schedule = BalancedStationTimeSchedule(valid_times_by_station(train_ds), seed=CFG.seed + fold * 1009)
        optimizer.zero_grad(set_to_none=True); print(json.dumps({"sanity": check}, ensure_ascii=False), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    best_macro = min((row["macro_rmse"] for row in history), default=float("inf"))
    while schedule.epoch < planned_stop_epoch:
        epoch_number = schedule.epoch + 1; epoch_started = time.perf_counter()
        model.train()
        for item in schedule.iter_epoch():
            batch = adapter.prepare(builder(raw_batch(item.station_indices, item.time_indices)))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                prediction, _ = forward_model(model, batch, adapter)
                loss = ((prediction.float() - batch["label"].float()) / adapter.pm_std).square().mean()
            if not torch.isfinite(loss): raise RuntimeError("non-finite V8 loss")
            grad_scaler.scale(loss).backward(); grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            grad_scaler.step(optimizer); grad_scaler.update(); scheduler.step()
            n = len(item.station_indices); loss_sum += float(loss.detach()) * n; seen += n; global_step += 1
            if global_step % CHECKPOINT_EVERY_STEPS == 0:
                elapsed = time.perf_counter() - global_started
                progress = {"revision": REVISION, "fold": fold, "epoch": epoch_number,
                    "step": global_step, "steps_total_fold_max20": total_steps, "loss": loss_sum / seen,
                    "samples_per_second": seen / max(time.perf_counter() - epoch_started, 1e-6),
                    "planned_stop_epoch": planned_stop_epoch,
                    "elapsed_total_seconds": elapsed,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2}
                atomic_json(progress, local_root / "progress.json")
                save_checkpoint(checkpoint_payload(fold, model, optimizer, scheduler, schedule,
                    history, station_history, loss_sum, seen, global_step, planned_stop_epoch), local_ckpt, drive_ckpt)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
        metrics, station, _ = validate(model, val_ds, builder, adapter, static, device)
        nearest = np.asarray([distance[int(s), train_idx].min() for s in station.station_index])
        sparse = nearest >= np.quantile(nearest, .75)
        metrics["sparse_quartile_macro_rmse"] = float(station.loc[sparse, "rmse"].mean())
        epoch_row = {"epoch": epoch_number, "train_loss": loss_sum / max(seen, 1),
            "runtime_seconds": time.perf_counter() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"], **metrics}
        history.append(epoch_row)
        for row in station.to_dict("records"):
            station_history.append({"epoch": epoch_number, **row})
        if metrics["macro_rmse"] < best_macro:
            best_macro = metrics["macro_rmse"]
            best_payload = {"revision": REVISION, "fold": fold, "epoch": epoch_number,
                "model": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                "metrics": metrics, "train_indices": train_idx, "validation_indices": val_idx}
            atomic_torch(best_payload, local_best)
            if drive_best: sync_file(local_best, drive_best)
        if epoch_number == INITIAL_EPOCHS:
            print(json.dumps({"fold": fold, "checkpoint": "epoch15_recorded",
                "best_epoch_so_far": int(min(history, key=lambda x: x["macro_rmse"])["epoch"])}, ensure_ascii=False), flush=True)
        if epoch_number == CONVERGENCE_CHECK_EPOCH:
            before = min(row["macro_rmse"] for row in history if row["epoch"] <= INITIAL_EPOCHS)
            probe_best = min(row["macro_rmse"] for row in history if INITIAL_EPOCHS < row["epoch"] <= CONVERGENCE_CHECK_EPOCH)
            improvement = (before - probe_best) / before
            still_improving = improvement >= MATERIAL_IMPROVEMENT_FRACTION
            planned_stop_epoch = MAX_EPOCHS if still_improving else CONVERGENCE_CHECK_EPOCH
            print(json.dumps({"fold": fold, "convergence_check_epoch": 18,
                "best_through_15": before, "best_16_to_18": probe_best,
                "relative_improvement": improvement, "still_improving": still_improving,
                "planned_stop_epoch": planned_stop_epoch}, ensure_ascii=False), flush=True)
        loss_sum, seen = 0.0, 0
        save_checkpoint(checkpoint_payload(fold, model, optimizer, scheduler, schedule,
            history, station_history, loss_sum, seen, global_step, planned_stop_epoch), local_ckpt, drive_ckpt)
    if not local_ckpt.exists() and drive_ckpt and drive_ckpt.exists(): sync_file(drive_ckpt, local_ckpt)
    sync_file(local_ckpt, fold_dir / "last_resume.pt")
    if not local_best.exists() and drive_best and drive_best.exists(): sync_file(drive_best, local_best)
    best_payload = torch.load(local_best, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"])
    metrics, station, prediction = validate(model, val_ds, builder, adapter, static, device)
    baseline = load_baseline(fold, val_idx, val_ds, train_idx, distance, cube)
    nearest = np.asarray([distance[int(s), train_idx].min() for s in station.station_index])
    sparse = nearest >= np.quantile(nearest, .75)
    metrics["sparse_quartile_macro_rmse"] = float(station.loc[sparse, "rmse"].mean())
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(station_history).to_csv(fold_dir / "station_metrics_history.csv", index=False, encoding="utf-8-sig")
    station.to_csv(fold_dir / "best_station_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(fold_dir / "best_predictions.npz", **prediction)
    converged_by_18 = planned_stop_epoch == CONVERGENCE_CHECK_EPOCH
    capped_while_improving = planned_stop_epoch == MAX_EPOCHS and history[-1]["epoch"] == MAX_EPOCHS \
        and history[-1]["macro_rmse"] <= min(row["macro_rmse"] for row in history[:-1])
    result = {"fold": fold, "best_epoch": int(best_payload["epoch"]), "epochs_completed": int(schedule.epoch),
        "converged_by_epoch18": converged_by_18, "training_cap_reached_while_improving": capped_while_improving,
        "continuation_available_from_epoch": int(schedule.epoch + 1), "metrics": metrics, "baseline": baseline,
        "runtime_seconds": time.perf_counter() - fold_started, "fused_adamw": fused}
    atomic_json(result, fold_dir / "fold_result.json")
    if drive_root:
        destination = drive_root / fold_dir.name
        temporary = destination.with_name(destination.name + "_tmp")
        if temporary.exists(): shutil.rmtree(temporary)
        shutil.copytree(fold_dir, temporary)
        if destination.exists(): shutil.rmtree(destination)
        temporary.replace(destination)
        if drive_ckpt.exists(): drive_ckpt.unlink()
    if local_ckpt.exists(): local_ckpt.unlink()
    return result


def final_gate(results, local_root, drive_root):
    if any(result.get("baseline") is None for result in results):
        summary = {"status": "BASELINE_UNAVAILABLE", "revision": REVISION,
            "selection_used_outer_truth": False, "results": results,
            "full73_authorized": False,
            "note": "V8 completed, but no protocol-matched V6 60/12 artifact was supplied; no GO claim is made."}
        atomic_json(summary, local_root / "pilot_gate.json")
        if drive_root: sync_file(local_root / "pilot_gate.json", drive_root / "pilot_gate.json")
        return summary
    checks = []
    for result in results:
        current, baseline = result["metrics"], result["baseline"]
        checks.extend([
            {"fold": result["fold"], "criterion": "macro_rmse_5pct", "pass": current["macro_rmse"] <= .95 * baseline["macro_rmse"]},
            {"fold": result["fold"], "criterion": "mean_abs_bias", "pass": current["mean_abs_bias"] <= baseline["mean_abs_bias"]},
            {"fold": result["fold"], "criterion": "sparse_quartile_rmse", "pass": current["sparse_quartile_macro_rmse"] <= baseline["sparse_quartile_macro_rmse"]},
        ])
    v8_sse = baseline_sse = truth_sum = truth_sum_squares = 0.0
    truth_n = 0
    for result in results:
        fold = result["fold"]
        pred = np.load(local_root / f"fold_{fold:02d}" / "best_predictions.npz")
        y, p = pred["y"].astype(float), pred["prediction"].astype(float)
        v8_sse += float(np.square(p-y).sum())
        baseline_sse += float(result["baseline"]["baseline_sse"])
        truth_sum += float(y.sum()); truth_sum_squares += float(np.square(y).sum()); truth_n += int(y.size)
    v8_sst = truth_sum_squares - truth_sum * truth_sum / truth_n
    v8_r2 = 1-v8_sse/v8_sst; baseline_r2 = 1-baseline_sse/v8_sst
    checks.append({"fold": "both", "criterion": "pooled_r2_plus_0.04", "pass": v8_r2 >= baseline_r2 + .04})
    status = "GO" if all(row["pass"] for row in checks) else "NO_GO"
    summary = {"status": status, "revision": REVISION,
        "selection_used_outer_truth": False, "checks": checks,
        "v8_pooled_r2_exact": v8_r2, "v6_baseline_pooled_r2_exact": baseline_r2,
        "results": results, "full73_authorized": False,
        "note": "Full73 is never launched by this pilot; GO only permits a separate explicit decision."}
    atomic_json(summary, local_root / "pilot_gate.json")
    if drive_root: sync_file(local_root / "pilot_gate.json", drive_root / "pilot_gate.json")
    return summary


def main():
    global_started = time.perf_counter(); os.environ["DL_TCN_COMPILE_MODE"] = "off"
    runtime = apply_runtime_profile(CFG)
    if CFG.device.type != "cuda": raise RuntimeError("V8 pilot requires Colab CUDA")
    torch.set_num_threads(min(12, os.cpu_count() or 1)); torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True; seed_all(CFG.seed)
    local_root = Path(os.environ.get("V8_OUTPUT_ROOT", "/content/DL_TCN_V8_SASFO_PILOT"))
    if "/drive/" in str(local_root): raise RuntimeError("V8_OUTPUT_ROOT must remain under /content")
    drive_base = Path("/content/drive/MyDrive")
    drive_root = Path(os.environ.get("V8_DRIVE_OUTPUT", str(drive_base / "DL_TCN_V8_SASFO_PILOT"))) if drive_base.is_dir() else None
    local_root.mkdir(parents=True, exist_ok=True)
    static, clusters, cols = load_static(); outer = target_index(static, CFG.target_site)
    splits = make_meta_crossfit_folds(clusters, outer)
    cube, timestamps = build_or_load_hourly_cube(static); distance = haversine_matrix(static.longitude, static.latitude)
    first_train, first_val = splits[FOLDS[0]]
    first_scaler = fit_train_only_scaler(cube, timestamps, static, cols, first_train)
    first_scaled = standardize_static(static, cols, first_scaler)
    first_train_ds = ColdStartStationDataset(first_train, first_train, CFG.train_start, CFG.train_end, cube, timestamps, static, first_scaled, distance, first_scaler)
    first_val_ds = ColdStartStationDataset(first_val, first_train, CFG.train_start, CFG.train_end, cube, timestamps, static, first_scaled, distance, first_scaler)
    hidden = np.setdiff1d(np.arange(len(static)), first_train)
    benchmark_started = time.perf_counter()
    builder = DeviceFeatureBuilder(first_train, cube, max(int(first_train_ds.row_times.max()), int(first_val_ds.row_times.max())),
        timestamps, static, first_scaled, distance, first_scaler, hidden, CFG.device)
    adapter = V8Adapter(timestamps, first_scaler, CFG.device)
    pilot_estimate = benchmark(first_train_ds, first_val_ds, builder, builder, adapter, CFG.device,
        train_donors=59, eval_donors=60, seed=CFG.seed + 8000)
    pilot_precompute_seconds = builder.precompute_seconds
    del builder, adapter, first_train_ds, first_val_ds; torch.cuda.empty_cache()

    known = np.setdiff1d(np.arange(len(static), dtype=int), np.array([outer]))
    refit_scaler = fit_train_only_scaler(cube, timestamps, static, cols, known)
    refit_scaled = standardize_static(static, cols, refit_scaler)
    refit_train_ds = ColdStartStationDataset(known, known, CFG.train_start, CFG.train_end,
        cube, timestamps, static, refit_scaled, distance, refit_scaler)
    shape72_eval_ds = ColdStartStationDataset(first_val, np.arange(len(static)), CFG.train_start, CFG.train_end,
        cube, timestamps, static, refit_scaled, distance, refit_scaler)
    refit_builder = DeviceFeatureBuilder(known, cube, int(refit_train_ds.row_times.max()), timestamps,
        static, refit_scaled, distance, refit_scaler, np.array([outer]), CFG.device)
    shape72_builder = DeviceFeatureBuilder(np.arange(len(static)), cube, int(shape72_eval_ds.row_times.max()), timestamps,
        static, refit_scaled, distance, refit_scaler, np.array([outer]), CFG.device)
    refit_adapter = V8Adapter(timestamps, refit_scaler, CFG.device)
    refit_estimate = benchmark(refit_train_ds, shape72_eval_ds, refit_builder, shape72_builder,
        refit_adapter, CFG.device, train_donors=71, eval_donors=72, seed=CFG.seed + 8100)
    refit_precompute_seconds = refit_builder.precompute_seconds
    benchmark_elapsed = time.perf_counter() - benchmark_started

    pilot_train = len(FOLDS) * MAX_EPOCHS * 1920 * pilot_estimate["train_batch_p95_seconds"]
    pilot_validation = len(FOLDS) * MAX_EPOCHS * math.ceil(len(shape72_eval_ds) / VALIDATION_BATCH_SIZE) \
        * pilot_estimate["eval_batch_p95_seconds"]
    pilot_overhead = len(FOLDS) * (pilot_precompute_seconds + 30.0)
    pilot_projected = benchmark_elapsed + pilot_train + pilot_validation + pilot_overhead
    sixfold_projected = 6 * (MAX_EPOCHS * 1920 * pilot_estimate["train_batch_p95_seconds"]
        + MAX_EPOCHS * math.ceil(len(shape72_eval_ds) / VALIDATION_BATCH_SIZE)
        * pilot_estimate["eval_batch_p95_seconds"] + pilot_precompute_seconds + 30.0)
    outer_inference_batches = math.ceil((366 * 24) / VALIDATION_BATCH_SIZE)
    refit73_projected = 73 * (MAX_EPOCHS * 2304 * refit_estimate["train_batch_p95_seconds"]
        + outer_inference_batches * refit_estimate["eval_batch_p95_seconds"]
        + refit_precompute_seconds + 30.0)
    core_projected = pilot_projected + sixfold_projected + refit73_projected
    recovery_reserve = 0.20 * core_projected
    analysis_margin = 5 * 60 * 60
    total_projected = core_projected + recovery_reserve + analysis_margin
    cost = {"benchmark_elapsed_seconds": benchmark_elapsed,
        "two_fold_pilot_worst20_seconds": pilot_projected,
        "six_fold_selection_worst20_seconds": sixfold_projected,
        "seventy_three_refits_worst20_seconds": refit73_projected,
        "recovery_reserve_20pct_seconds": recovery_reserve,
        "analysis_margin_seconds": analysis_margin,
        "total_projected_seconds": total_projected,
        "total_projected_a100_hours": total_projected / 3600,
        "available_a100_hours": TOTAL_A100_BUDGET_SECONDS / 3600,
        "fits_60_hour_budget": total_projected <= TOTAL_A100_BUDGET_SECONDS}
    profile = {"revision": REVISION, "runtime": runtime, "outer": {"index": outer,
        "siteid": str(static.loc[outer,"siteid"]), "sitename": str(static.loc[outer,"sitename"])},
        "folds": list(FOLDS), "epoch_policy": {"record_at": 15, "convergence_check_at": 18,
            "continue_if_still_improving_to": 20, "later_resume_from_21_supported": True},
        "batch_size": BATCH_SIZE,
        "station_batch": "4 stations x 64 times", "rounds_per_epoch": 128,
        "train_steps_per_fold_max20": 60 // 4 * 128 * MAX_EPOCHS,
        "two_fold_forward_backward_steps_max20": 2 * (60 // 4) * 128 * MAX_EPOCHS,
        "pilot_59_60_benchmark": pilot_estimate, "refit_71_72_benchmark": refit_estimate,
        "cost_projection": cost}
    atomic_json(profile, local_root / "runtime_profile.json")
    if drive_root: sync_file(local_root / "runtime_profile.json", drive_root / "runtime_profile.json")
    print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)
    if pilot_projected > PILOT_PREFLIGHT_LIMIT_SECONDS or not cost["fits_60_hour_budget"]:
        reasons = []
        if pilot_projected > PILOT_PREFLIGHT_LIMIT_SECONDS: reasons.append("two-fold max20 pilot projects above 4 A100 hours")
        if not cost["fits_60_hour_budget"]: reasons.append("complete plan plus recovery reserve projects above 60 A100 hours")
        abort = {"status": "ABORTED_BEFORE_TRAINING", "reason": reasons,
            "action": "optimize implementation and re-benchmark without reducing epochs or samples", **profile}
        atomic_json(abort, local_root / "pilot_gate.json")
        if drive_root: sync_file(local_root / "pilot_gate.json", drive_root / "pilot_gate.json")
        print(json.dumps(abort, ensure_ascii=False, indent=2)); return
    del refit_builder, shape72_builder, refit_adapter, refit_train_ds, shape72_eval_ds
    torch.cuda.empty_cache()
    results = []
    for fold in FOLDS:
        completed = (drive_root / f"fold_{fold:02d}" / "fold_result.json") if drive_root else (local_root / f"fold_{fold:02d}" / "fold_result.json")
        if completed.exists():
            if drive_root:
                local_fold = local_root / f"fold_{fold:02d}"
                if not local_fold.exists(): shutil.copytree(completed.parent, local_fold)
            results.append(json.loads(completed.read_text(encoding="utf-8"))); print(f"SKIP completed fold {fold}", flush=True); continue
        train_idx, val_idx = splits[fold]
        results.append(train_fold(fold, train_idx, val_idx, cube, timestamps, static, cols,
            distance, CFG.device, local_root, drive_root, global_started))
    summary = final_gate(results, local_root, drive_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

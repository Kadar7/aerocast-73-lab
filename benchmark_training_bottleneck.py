from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

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
from model import TCNTargetCrossAttention
from train_formal import (
    DeviceFeatureBuilder,
    amp_dtype_for,
    make_grad_scaler,
    make_index_loader,
    make_optimizer,
    make_station_sample_weights,
    resolve_target,
    seed_all,
)


def main() -> None:
    seed_all(CFG.seed)
    profile = apply_runtime_profile(CFG)
    if CFG.device.type != "cuda":
        raise RuntimeError("此benchmark需要CUDA；目前沒有GPU")
    device = CFG.device
    torch.backends.cuda.matmul.allow_tf32 = CFG.enable_tf32
    torch.backends.cudnn.allow_tf32 = CFG.enable_tf32
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    warmup_batches = int(os.environ.get("DL_TCN_PROFILE_WARMUP_BATCHES", "5"))
    measured_batches = int(os.environ.get("DL_TCN_PROFILE_BATCHES", "50"))
    compile_mode = os.environ.get("DL_TCN_COMPILE_MODE", "off").strip().lower()
    if compile_mode not in {"off", "default", "reduce-overhead"}:
        raise ValueError("DL_TCN_COMPILE_MODE只能是off/default/reduce-overhead")
    output = Path(os.environ.get("DL_TCN_PROFILE_OUTPUT", "/content/training_bottleneck_profile"))
    output.mkdir(parents=True, exist_ok=True)

    static, clusters, static_cols = load_static()
    outer = resolve_target(static, CFG.target_site)
    train_idx, _ = make_meta_crossfit_folds(clusters, outer)[0]
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    scaler = fit_train_only_scaler(cube, timestamps, static, static_cols, train_idx)
    static_scaled = standardize_static(static, static_cols, scaler)
    train_ds = ColdStartStationDataset(
        train_idx, train_idx, CFG.train_start, CFG.train_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    builder = DeviceFeatureBuilder(
        train_idx, cube, int(train_ds.row_times.max()), timestamps,
        static, static_scaled, distance, scaler, outer, device,
    )
    loader = make_index_loader(train_ds, True)
    base_model = TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    model = base_model
    if compile_mode != "off":
        if not hasattr(torch, "compile"):
            raise RuntimeError("目前PyTorch沒有torch.compile")
        model = torch.compile(
            base_model,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        )
    optimizer, fused = make_optimizer(base_model, device)
    amp_dtype = amp_dtype_for(device)
    grad_scaler = make_grad_scaler(amp_dtype == torch.float16)
    station_weights = make_station_sample_weights(train_ds, len(static), device)

    timings = {"loader_wait_ms": [], "feature_build_ms": [], "forward_loss_ms": [], "backward_step_ms": []}
    total_samples = 0
    previous_end = time.perf_counter()
    first_batch_wall_seconds = None
    limit = warmup_batches + measured_batches
    for batch_number, index_batch in enumerate(loader, start=1):
        batch_wall_started = time.perf_counter()
        loader_wait_ms = (batch_wall_started - previous_end) * 1000.0
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        batch = builder(index_batch)
        events[1].record()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            prediction, _ = model(
                batch["values"], batch["mask"], batch["donor_static"], batch["geometry"],
                batch["donor_padding_mask"], batch["target_static"], batch["time_features"],
            )
            weights = station_weights[batch["target_idx"]]
            loss = torch.mean(torch.square(prediction - batch["label"]) * weights)
        events[2].record()
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        events[3].record()
        torch.cuda.synchronize(device)
        if batch_number == 1:
            first_batch_wall_seconds = time.perf_counter() - batch_wall_started
        if batch_number > warmup_batches:
            timings["loader_wait_ms"].append(loader_wait_ms)
            timings["feature_build_ms"].append(events[0].elapsed_time(events[1]))
            timings["forward_loss_ms"].append(events[1].elapsed_time(events[2]))
            timings["backward_step_ms"].append(events[2].elapsed_time(events[3]))
            total_samples += int(batch["label"].numel())
        previous_end = time.perf_counter()
        if batch_number >= limit:
            break

    means = {key: float(np.mean(value)) for key, value in timings.items()}
    measured_total_ms = sum(means.values())
    percentages = {
        key.replace("_ms", "_percent"): 100.0 * value / measured_total_ms
        for key, value in means.items()
    }
    samples_per_second = total_samples / max(measured_total_ms * measured_batches / 1000.0, 1e-9)
    result = {
        "runtime_profile": profile,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": CFG.batch_size,
        "workers": CFG.formal_num_workers,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "train_samples": len(train_ds),
        "donors_per_training_sample": 59,
        "history_hours": CFG.history_hours,
        "dynamic_channels": CFG.n_dynamic_channels,
        "precompute_seconds": builder.precompute_seconds,
        "precomputed_tables_mb": builder.precomputed_table_mb,
        "fused_adamw": fused,
        "compile_mode": compile_mode,
        "first_batch_wall_seconds_including_lazy_compile": first_batch_wall_seconds,
        **means,
        **percentages,
        "measured_samples_per_second": samples_per_second,
        "estimated_train_epoch_seconds": len(train_ds) / max(samples_per_second, 1e-9),
        "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    (output / "training_bottleneck_profile.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()

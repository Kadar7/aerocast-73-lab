"""One-batch V2 integration check. This never starts formal training."""
from __future__ import annotations

import os

from data_pipeline import (
    ColdStartStationDataset, build_or_load_hourly_cube, choose_split,
    fit_train_only_scaler, haversine_matrix, load_static, standardize_static,
)
from model_v2 import TCNTargetCrossAttentionV2
from train_formal import DeviceFeatureBuilder, make_index_loader, make_loader, resolve_target
from v2_training import build_v2_context, smoke_v2
from config import CFG


def main() -> None:
    static, clusters, columns = load_static()
    outer = resolve_target(static, CFG.target_site)
    train, _ = choose_split(clusters, outer)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    scaler = fit_train_only_scaler(cube, timestamps, static, columns, train)
    scaled = standardize_static(static, columns, scaler)
    dataset = ColdStartStationDataset(
        train[:1], train, CFG.train_start, CFG.train_end, cube, timestamps,
        static, scaled, distance, scaler,
    )
    adapter, background = build_v2_context(
        cube, timestamps, scaled, distance, train, scaler, static, CFG.device,
    )
    if CFG.device.type == "cuda" and os.environ.get("DL_TCN_REAL_SMOKE_DEVICE_PIPELINE") == "1":
        builder = DeviceFeatureBuilder(
            train, cube, int(dataset.row_times.max()), timestamps, static, scaled,
            distance, scaler, outer, CFG.device,
        )
        batch = builder(next(iter(make_index_loader(dataset, False))))
    else:
        batch = next(iter(make_loader(dataset, False, batch_size=2)))
    model = TCNTargetCrossAttentionV2(len(columns)).to(CFG.device)
    report = smoke_v2(model, batch, CFG.device, adapter)
    print({"status": "ok", "background": background, "smoke": report})


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from config import CFG
from model_v2 import TCNTargetCrossAttentionV2
from v2_training import (
    BackgroundFit, V2BatchAdapter, _forward, _v2_loss, fit_background, smoke_v2,
)


def main() -> None:
    rng = np.random.default_rng(7)
    stations, batch_size, donors = 73, 4, 5
    static = pd.DataFrame({
        "longitude": np.linspace(120.0, 122.0, stations),
        "latitude": np.linspace(22.0, 25.0, stations),
    })
    fit = BackgroundFit(
        predicted=np.linspace(8, 22, stations).astype("float32"),
        observed=np.linspace(9, 21, stations).astype("float32"),
        center=15.0, scale=3.0, ridge_lambda=10.0, static_weight=0.5,
        loo_rmse=2.0,
    )
    class Scaler:
        dynamic_mean = np.zeros(11, dtype="float32")
        dynamic_std = np.ones(11, dtype="float32")
    adapter = V2BatchAdapter(fit, Scaler(), static, torch.device("cpu"))
    target_idx = torch.tensor([0, 1, 2, 3])
    donor_indices = torch.tensor([[x for x in range(1, donors+1)] for _ in range(batch_size)])
    values = torch.tensor(rng.normal(size=(batch_size, donors, 24, 11)), dtype=torch.float32)
    mask = torch.ones_like(values)
    geometry = torch.zeros(batch_size, donors, 3)
    geometry[..., 0] = 1.0
    geometry[..., 1] = 0.6
    geometry[..., 2] = 0.8
    raw_batch = {
        "values": values, "mask": mask,
        "donor_static": torch.tensor(rng.normal(size=(batch_size, donors, 49)), dtype=torch.float32),
        "geometry": geometry,
        "donor_padding_mask": torch.zeros(batch_size, donors, dtype=torch.bool),
        "donor_all_missing": torch.zeros(batch_size, donors, dtype=torch.bool),
        "target_static": torch.tensor(rng.normal(size=(batch_size, 49)), dtype=torch.float32),
        "time_features": torch.tensor(rng.normal(size=(batch_size, 4)), dtype=torch.float32),
        "label": torch.tensor([10.0, 20.0, 35.0, 50.0]),
        "target_idx": target_idx, "time_idx": torch.tensor([50, 51, 52, 53]),
        "donor_indices": donor_indices,
    }
    model = TCNTargetCrossAttentionV2(49)
    report = smoke_v2(model, raw_batch, torch.device("cpu"), adapter)
    prepared = adapter.prepare(raw_batch)
    virtual, valid = adapter.virtual_batch(prepared)
    assert virtual is not None and len(valid) == batch_size and valid.all()
    assert virtual["values"].shape == values.shape
    assert torch.isfinite(virtual["geometry"]).all()
    assert torch.all(prepared["values"][prepared["mask"] == 0] == 0)
    model.zero_grad(set_to_none=True)
    v_prediction, v_auxiliary = _forward(model, virtual)
    v_loss = _v2_loss(
        v_prediction, v_auxiliary, virtual["label"], valid,
        torch.tensor(2.0),
    )
    v_loss.backward()
    assert torch.isfinite(v_loss)

    # Changing a held-out station's PM2.5 must not change any fitted background.
    timestamps = pd.date_range("2024-07-01", periods=48, freq="h")
    cube = np.full((48, stations, len(CFG.aq_cube_items)), np.nan, dtype="float32")
    source = np.arange(12)
    cube[:, source, CFG.aq_cube_items.index("PM2.5")] = rng.uniform(5, 30, size=(48, len(source)))
    static_scaled = rng.normal(size=(stations, 49)).astype("float32")
    coords = np.arange(stations, dtype=float)
    distance = np.abs(coords[:, None]-coords[None, :]) * 10_000.0
    first_fit = fit_background(cube, timestamps, static_scaled, distance, source)
    cube[:, 30, CFG.aq_cube_items.index("PM2.5")] = 1_000_000.0
    second_fit = fit_background(cube, timestamps, static_scaled, distance, source)
    assert np.array_equal(first_fit.predicted, second_fit.predicted)
    print({"status": "ok", **report})


if __name__ == "__main__":
    main()

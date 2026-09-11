from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    build_or_load_hourly_cube,
    haversine_matrix,
    load_static,
    standardize_static,
)
from evaluate_outer import load_checkpoint, scaler_from_checkpoint
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset
from train_formal import (
    DeviceFeatureBuilder,
    make_index_loader,
    make_vectorized_loader,
    regression_metrics,
    resolve_target,
    validate_epoch,
)


EPOCHS = tuple(range(1, 16))
TOP_K_CANDIDATES = (2, 3, 5, 8, 15)
RIDGE_CANDIDATES = (0.0, 0.01, 0.1, 1.0, 10.0)


def aligned_predictions(
    model, paths, loader, device, static, timestamps, feature_builder
) -> tuple[pd.DataFrame, np.ndarray]:
    base = None
    columns = []
    for epoch, path in zip(EPOCHS, paths):
        checkpoint = load_checkpoint(path)
        if int(checkpoint["epoch"]) != epoch:
            raise RuntimeError(f"epoch {epoch} checkpoint metadata錯誤")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        _, _, predictions = validate_epoch(
            model, loader, device, static, timestamps, epoch, feature_builder
        )
        current = predictions[["station_index", "timestamp", "y_true"]].copy()
        if base is None:
            base = current
        else:
            if not np.array_equal(base.station_index.to_numpy(), current.station_index.to_numpy()):
                raise RuntimeError("snapshot station alignment不同")
            if not np.array_equal(
                pd.to_datetime(base.timestamp).astype("int64").to_numpy(),
                pd.to_datetime(current.timestamp).astype("int64").to_numpy(),
            ):
                raise RuntimeError("snapshot timestamp alignment不同")
            if not np.allclose(base.y_true, current.y_true, rtol=0, atol=1e-6):
                raise RuntimeError("snapshot truth alignment不同")
        columns.append(predictions.y_pred.to_numpy("float64"))
        print(f"snapshot inference {epoch}/15", flush=True)
    assert base is not None
    return base, np.stack(columns, axis=1)


def station_rmse_by_epoch(base: pd.DataFrame, matrix: np.ndarray, stations: np.ndarray) -> np.ndarray:
    result = np.empty((len(stations), matrix.shape[1]), dtype=float)
    truth = base.y_true.to_numpy("float64")
    station_values = base.station_index.to_numpy(int)
    for row, station in enumerate(stations):
        mask = station_values == station
        result[row] = np.sqrt(np.mean((matrix[mask] - truth[mask, None]) ** 2, axis=0))
    return result


def fit_convex_weights(
    base: pd.DataFrame,
    matrix: np.ndarray,
    stations: np.ndarray,
    ridge: float,
) -> np.ndarray:
    station_values = base.station_index.to_numpy(int)
    truth = base.y_true.to_numpy("float64")
    q = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    c = np.zeros(matrix.shape[1], dtype=float)
    y2 = 0.0
    for station in stations:
        mask = station_values == station
        xs = matrix[mask]
        ys = truth[mask]
        q += xs.T @ xs / len(ys)
        c += xs.T @ ys / len(ys)
        y2 += float(ys @ ys / len(ys))
    q /= len(stations)
    c /= len(stations)
    y2 /= len(stations)
    scale = max(y2, 1.0)
    uniform = np.full(matrix.shape[1], 1.0 / matrix.shape[1])

    def objective(weight):
        residual = float(weight @ q @ weight - 2.0 * c @ weight + y2)
        penalty = float(ridge * scale * np.sum((weight - uniform) ** 2))
        return residual + penalty

    def gradient(weight):
        return 2.0 * q @ weight - 2.0 * c + 2.0 * ridge * scale * (weight - uniform)

    result = minimize(
        objective,
        uniform,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * matrix.shape[1],
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0), "jac": lambda w: np.ones_like(w)},
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"convex ensemble最佳化失敗: {result.message}")
    weight = np.clip(result.x, 0.0, None)
    return weight / weight.sum()


def station_rmse(base: pd.DataFrame, prediction: np.ndarray, station: int) -> float:
    mask = base.station_index.to_numpy(int) == station
    truth = base.y_true.to_numpy("float64")[mask]
    return float(np.sqrt(np.mean((prediction[mask] - truth) ** 2)))


def choose_weights_from_validation(base: pd.DataFrame, matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    stations = np.unique(base.station_index.to_numpy(int))
    if len(stations) != 12:
        raise RuntimeError(f"validation必須是12站，實際{len(stations)}")
    rmse_curve = station_rmse_by_epoch(base, matrix, stations)
    candidates = []

    for top_k in TOP_K_CANDIDATES:
        held_rmse = []
        for held_position, held_station in enumerate(stations):
            train_positions = np.arange(len(stations)) != held_position
            ranked = np.argsort(rmse_curve[train_positions].mean(axis=0), kind="stable")
            selected = ranked[:top_k]
            prediction = matrix[:, selected].mean(axis=1)
            held_rmse.append(station_rmse(base, prediction, int(held_station)))
        candidates.append({
            "family": "uniform_top_k",
            "parameter": float(top_k),
            "nested_validation_macro_rmse": float(np.mean(held_rmse)),
        })

    for ridge in RIDGE_CANDIDATES:
        held_rmse = []
        for held_station in stations:
            train_stations = stations[stations != held_station]
            weight = fit_convex_weights(base, matrix, train_stations, ridge)
            held_rmse.append(station_rmse(base, matrix @ weight, int(held_station)))
        candidates.append({
            "family": "convex_ridge",
            "parameter": float(ridge),
            "nested_validation_macro_rmse": float(np.mean(held_rmse)),
        })

    selected_method = min(candidates, key=lambda row: row["nested_validation_macro_rmse"])
    if selected_method["family"] == "uniform_top_k":
        top_k = int(selected_method["parameter"])
        ranked = np.argsort(rmse_curve.mean(axis=0), kind="stable")
        weights = np.zeros(len(EPOCHS), dtype=float)
        weights[ranked[:top_k]] = 1.0 / top_k
    else:
        weights = fit_convex_weights(
            base, matrix, stations, float(selected_method["parameter"])
        )
    audit = {
        "selection_used_outer_truth": False,
        "validation_stations": 12,
        "candidate_scores": sorted(candidates, key=lambda row: row["nested_validation_macro_rmse"]),
        "selected_method": selected_method,
        "weights": [
            {"epoch": epoch, "weight": float(weight)}
            for epoch, weight in zip(EPOCHS, weights)
            if weight > 1e-8
        ],
    }
    return weights, audit


def main() -> None:
    apply_runtime_profile(CFG)
    device = CFG.device
    root = Path(os.environ.get(
        "DL_TCN_CROSSFIT_ROOT",
        str(CFG.formal_output_root / "crossfit_target_conditioned_snapshots"),
    ))
    paths = [root / "fold_00" / "epoch_checkpoints" / f"epoch_{epoch:03d}.pt" for epoch in EPOCHS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("fold_00缺少checkpoint:\n" + "\n".join(missing))

    static, _, static_cols = load_static()
    outer = resolve_target(static, CFG.target_site)
    first = load_checkpoint(paths[0])
    train_idx = np.asarray(first["train_indices"], dtype=int)
    validation_idx = np.asarray(first["validation_indices"], dtype=int)
    if len(train_idx) != 60 or len(validation_idx) != 12:
        raise RuntimeError("checkpoint不是60 train / 12 validation")
    if outer in train_idx or outer in validation_idx:
        raise RuntimeError("outer target混入train/validation")

    scaler = scaler_from_checkpoint(first)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    static_scaled = standardize_static(static, static_cols, scaler)
    validation_ds = ColdStartStationDataset(
        validation_idx, train_idx, CFG.train_start, CFG.train_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    outer_ds = ColdStartStationDataset(
        [outer], train_idx, CFG.test_start, CFG.test_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    if audit_dataset(validation_ds, max_samples=32)["sampled_donor_counts"] != [60]:
        raise RuntimeError("validation donors不是60")
    if audit_dataset(outer_ds, max_samples=32)["sampled_donor_counts"] != [60]:
        raise RuntimeError("outer donors不是60")

    validation_builder = outer_builder = None
    if device.type == "cuda":
        max_time = int(max(validation_ds.row_times.max(), outer_ds.row_times.max()))
        validation_builder = DeviceFeatureBuilder(
            train_idx, cube, max_time, timestamps, static, static_scaled,
            distance, scaler, outer, device,
        )
        outer_builder = validation_builder
        validation_loader = make_index_loader(validation_ds, False)
        outer_loader = make_index_loader(outer_ds, False)
    else:
        validation_loader = make_vectorized_loader(
            validation_ds, train_idx, cube, timestamps, static,
            static_scaled, distance, scaler, False,
        )
        outer_loader = make_vectorized_loader(
            outer_ds, train_idx, cube, timestamps, static,
            static_scaled, distance, scaler, False,
        )

    model = TCNTargetCrossAttention(static_dim=len(static_cols)).to(device)
    print("Validation inference; outer truth has not been read.", flush=True)
    validation_base, validation_matrix = aligned_predictions(
        model, paths, validation_loader, device, static, timestamps, validation_builder
    )
    weights, audit = choose_weights_from_validation(validation_base, validation_matrix)
    print("\nLOCKED VALIDATION-ONLY ENSEMBLE BEFORE OUTER TRUTH")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)

    print("\nOuter inference after weights are locked.", flush=True)
    outer_base, outer_matrix = aligned_predictions(
        model, paths, outer_loader, device, static, timestamps, outer_builder
    )
    prediction = outer_matrix @ weights
    metrics = regression_metrics(outer_base.y_true.to_numpy(float), prediction)
    result = {
        "target_station_index": int(outer),
        "siteid": str(static.loc[outer, "siteid"]),
        "sitename": str(static.loc[outer, "sitename"]),
        "selection_used_outer_truth": False,
        "donors": 60,
        "predicted_timestamps": int(len(outer_base)),
        "metrics": metrics,
        "selected_method": audit["selected_method"],
        "weights": audit["weights"],
        "files_written": 0,
    }
    print("\nOUTER ENSEMBLE RESULT")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

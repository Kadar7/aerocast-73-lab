"""V7: station-balanced, set-aggregated LightGBM for strict unseen-station LOSO.

The outer station is excluded from labels, donors, scaling and early stopping.
Only essential station metrics and resume state are persisted. Feature matrices
are temporary local files and are removed after each completed station.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(values, **_kwargs):
        return values

from config import CFG
from data_pipeline import (
    bearing_degrees,
    build_or_load_hourly_cube,
    choose_split,
    haversine_matrix,
    load_static,
)


TRAIN_START = pd.Timestamp("2024-07-01 00:00:00")
TRAIN_END = pd.Timestamp("2025-06-30 23:00:00")
TEST_START = pd.Timestamp("2025-07-01 00:00:00")
TEST_END = pd.Timestamp("2026-06-30 23:00:00")
LAGS = (0, 1, 3, 6, 12, 24, 48, 72)
RAW_ITEMS = ("SO2", "CO", "O3", "PM10", "NO", "NO2", "AMB_TEMP", "PM2.5", "RH")
DYNAMIC_ITEMS = RAW_ITEMS + ("WIND_ALONG", "WIND_CROSS")
DISPERSION_ITEMS = ("O3", "PM10", "NO2", "PM2.5")
NEAREST_ITEMS = ("PM10", "NO2", "PM2.5")
GEO_BANDWIDTHS_KM = (25.0, 75.0, 200.0)
HIGH_LINE = 35.0
SEED = 42


def output_root() -> Path:
    requested = os.environ.get("AEROCAST_V7_OUTPUT")
    if requested:
        return Path(requested)
    drive = Path("/content/drive/MyDrive")
    if drive.is_dir():
        return drive / "AeroCast_V7_essential"
    if Path("/content").is_dir():
        return Path("/content/AeroCast_V7_essential")
    return CFG.output_dir / "AeroCast_V7_essential"


OUT = output_root()
WORK = Path(os.environ.get("AEROCAST_V7_WORK", "/content/AeroCast_V7_work" if Path("/content").is_dir() else str(CFG.output_dir / "v7_lgbm_work")))


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_remove_work(path: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    root = WORK.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to delete non-station work path: {resolved}")
    shutil.rmtree(resolved)


def prepare_cube():
    # The LGBM experiment itself starts in 2024-07. The earlier cube start only
    # guarantees lag-72 availability and permits sharing the cache with IDW.
    original_start = CFG.train_start
    original_history = CFG.history_hours
    try:
        CFG.train_start = "2023-07-01 00:00:00"
        CFG.history_hours = max(original_history, max(LAGS) + 1)
        static, clusters, static_cols = load_static(CFG)
        cube, timestamps = build_or_load_hourly_cube(static, CFG)
    finally:
        CFG.train_start = original_start
        CFG.history_hours = original_history
    return static, clusters, static_cols, cube, timestamps


def time_indices(timestamps: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return np.flatnonzero((timestamps >= start) & (timestamps <= end))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if weights.ndim == 1:
        w = weights[None, :, None]
    else:
        w = weights[:, :, None]
    denominator = np.sum(valid * w, axis=1)
    numerator = np.sum(np.where(valid, values, 0.0) * w, axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def weighted_std(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = weighted_mean(values, weights)
    valid = np.isfinite(values)
    w = weights[None, :, None]
    denominator = np.sum(valid * w, axis=1)
    numerator = np.sum(np.where(valid, (values - mean[:, None, :]) ** 2, 0.0) * w, axis=1)
    variance = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return np.sqrt(np.maximum(variance, 0.0))


def static_context(static: pd.DataFrame, static_cols: list[str], fit_pool: np.ndarray):
    raw = static[static_cols].apply(pd.to_numeric, errors="coerce").to_numpy("float32")
    median = np.nanmedian(raw[fit_pool], axis=0)
    filled = np.where(np.isfinite(raw), raw, median)
    mean = filled[fit_pool].mean(axis=0)
    std = filled[fit_pool].std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = (filled - mean) / std
    delta = scaled[:, None, :] - scaled[None, :, :]
    static_distance = np.sqrt(np.mean(delta * delta, axis=2))
    return filled.astype("float32"), scaled.astype("float32"), static_distance.astype("float32")


def feature_names(static_cols: list[str]) -> list[str]:
    names: list[str] = []
    for lag in LAGS:
        for bandwidth in GEO_BANDWIDTHS_KM:
            names.extend(f"geo{int(bandwidth)}_mean_{item}_lag{lag}" for item in DYNAMIC_ITEMS)
        names.extend(f"static_mean_{item}_lag{lag}" for item in DYNAMIC_ITEMS)
        names.extend(f"hybrid_mean_{item}_lag{lag}" for item in DYNAMIC_ITEMS)
        names.extend(f"availability_{item}_lag{lag}" for item in DYNAMIC_ITEMS)
        names.extend(f"geo75_std_{item}_lag{lag}" for item in DISPERSION_ITEMS)
        names.extend(f"upwind_mean_{item}_lag{lag}" for item in RAW_ITEMS)
        for rank in range(1, 4):
            names.extend(f"near{rank}_{item}_lag{lag}" for item in NEAREST_ITEMS)
    names.extend(f"target_{column}" for column in static_cols)
    names.extend([
        "nearest_km", "mean3_km", "mean5_km", "donors_within_25km",
        "donors_within_50km", "donors_within_100km", "static_nearest",
        "static_mean3", "static_mean5", "donor_count",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    ])
    return names


def dynamic_values(cube: np.ndarray, indices: np.ndarray, donors: np.ndarray, target: int, static: pd.DataFrame):
    raw_idx = [CFG.aq_cube_items.index(item) for item in RAW_ITEMS]
    raw = np.asarray(cube[np.ix_(indices, donors, raw_idx)], dtype="float32")
    speed = np.asarray(cube[np.ix_(indices, donors, [CFG.aq_cube_items.index("WIND_SPEED")])][..., 0], dtype="float32")
    direction_from = np.asarray(cube[np.ix_(indices, donors, [CFG.aq_cube_items.index("WIND_DIREC")])][..., 0], dtype="float32")
    bearing = bearing_degrees(
        static.loc[donors, "longitude"], static.loc[donors, "latitude"],
        static.loc[target, "longitude"], static.loc[target, "latitude"],
    )
    delta = np.deg2rad(((direction_from + 180.0) % 360.0) - bearing[None, :])
    valid_wind = np.isfinite(speed) & np.isfinite(direction_from)
    along = np.where(valid_wind, speed * np.cos(delta), np.nan).astype("float32")
    cross = np.where(valid_wind, speed * np.sin(delta), np.nan).astype("float32")
    return np.concatenate([raw, along[..., None], cross[..., None]], axis=2), along


def build_target_block(
    cube: np.ndarray,
    timestamps: pd.DatetimeIndex,
    target: int,
    donor_pool: np.ndarray,
    requested_times: np.ndarray,
    static: pd.DataFrame,
    static_cols: list[str],
    static_raw: np.ndarray,
    static_scaled: np.ndarray,
    static_distance: np.ndarray,
    distance_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    donors = donor_pool[donor_pool != target]
    pm_idx = CFG.aq_cube_items.index("PM2.5")
    truth = np.asarray(cube[requested_times, target, pm_idx], dtype="float32")
    times = requested_times[np.isfinite(truth) & (requested_times >= max(LAGS))]
    y = np.asarray(cube[times, target, pm_idx], dtype="float32")
    distance_km = distance_m[target, donors] / 1000.0
    order = np.argsort(distance_km, kind="stable")
    nearest = donors[order[:3]]
    sorted_km = distance_km[order]

    geo_weights = [np.exp(-distance_km / bandwidth).astype("float32") for bandwidth in GEO_BANDWIDTHS_KM]
    sw = np.exp(-static_distance[target, donors]).astype("float32")
    hw = np.exp(-distance_km / 75.0 - static_distance[target, donors]).astype("float32")
    blocks: list[np.ndarray] = []
    dispersion_indices = np.array([DYNAMIC_ITEMS.index(item) for item in DISPERSION_ITEMS])
    nearest_indices = np.array([RAW_ITEMS.index(item) for item in NEAREST_ITEMS])

    for lag in LAGS:
        values, along = dynamic_values(cube, times - lag, donors, target, static)
        for weights in geo_weights:
            blocks.append(weighted_mean(values, weights))
        blocks.append(weighted_mean(values, sw))
        blocks.append(weighted_mean(values, hw))
        blocks.append(np.mean(np.isfinite(values), axis=1, dtype="float32"))
        blocks.append(weighted_std(values[:, :, dispersion_indices], geo_weights[1]))
        upwind = geo_weights[1][None, :] * (0.1 + np.maximum(np.nan_to_num(along, nan=0.0), 0.0))
        blocks.append(weighted_mean(values[:, :, : len(RAW_ITEMS)], upwind))
        nearest_values = np.asarray(cube[np.ix_(times - lag, nearest, [CFG.aq_cube_items.index(item) for item in NEAREST_ITEMS])], dtype="float32")
        blocks.append(nearest_values.reshape(len(times), -1))

    target_static = np.broadcast_to(static_scaled[target], (len(times), len(static_cols)))
    blocks.append(target_static)
    support = np.array([
        sorted_km[0], sorted_km[:3].mean(), sorted_km[:5].mean(),
        np.sum(distance_km <= 25), np.sum(distance_km <= 50), np.sum(distance_km <= 100),
        np.sort(static_distance[target, donors])[0],
        np.sort(static_distance[target, donors])[:3].mean(),
        np.sort(static_distance[target, donors])[:5].mean(), len(donors),
    ], dtype="float32")
    blocks.append(np.broadcast_to(support, (len(times), len(support))))
    ts = timestamps[times]
    hour = 2 * np.pi * ts.hour.to_numpy() / 24.0
    doy = 2 * np.pi * (ts.dayofyear.to_numpy() - 1) / 365.2425
    dow = 2 * np.pi * ts.dayofweek.to_numpy() / 7.0
    blocks.append(np.column_stack([np.sin(hour), np.cos(hour), np.sin(doy), np.cos(doy), np.sin(dow), np.cos(dow)]).astype("float32"))
    x = np.concatenate(blocks, axis=1).astype("float32", copy=False)
    return x, y, times


def build_matrix(
    path: Path,
    targets: np.ndarray,
    donor_pool: np.ndarray,
    requested_times: np.ndarray,
    cube: np.ndarray,
    timestamps: pd.DatetimeIndex,
    static: pd.DataFrame,
    static_cols: list[str],
    fit_pool: np.ndarray,
    distance_m: np.ndarray,
    names: list[str],
):
    static_raw, static_scaled, static_distance = static_context(static, static_cols, fit_pool)
    pm_idx = CFG.aq_cube_items.index("PM2.5")
    target_times = []
    for target in targets:
        truth = np.asarray(cube[requested_times, target, pm_idx], dtype="float32")
        target_times.append(requested_times[np.isfinite(truth) & (requested_times >= max(LAGS))])
    counts = np.array([len(value) for value in target_times], dtype=int)
    total = int(counts.sum())
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(total, len(names)))
    y = np.empty(total, dtype="float32")
    station_group = np.empty(total, dtype="int16")
    cursor = 0
    for group, target in enumerate(tqdm(targets, desc=f"features {path.stem}", unit="station", leave=False)):
        block, label, _ = build_target_block(
            cube, timestamps, int(target), donor_pool, target_times[group], static, static_cols,
            static_raw, static_scaled, static_distance, distance_m,
        )
        end = cursor + len(label)
        x[cursor:end] = block
        y[cursor:end] = label
        station_group[cursor:end] = group
        cursor = end
        del block
    x.flush()
    sample_weights = np.empty(total, dtype="float32")
    for group, count in enumerate(counts):
        sample_weights[station_group == group] = total / (len(counts) * count)
    return np.load(path, mmap_mode="r"), y, sample_weights, station_group, counts


def macro_rmse_metric(groups: np.ndarray, group_count: int):
    def evaluate(prediction: np.ndarray, dataset: lgb.Dataset):
        error2 = (prediction - dataset.get_label()) ** 2
        score = np.mean([np.sqrt(np.mean(error2[groups == group])) for group in range(group_count)])
        return "macro_station_rmse", float(score), False
    return evaluate


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - y
    sst = np.sum((y - y.mean()) ** 2)
    actual_high = y >= HIGH_LINE
    predicted_high = prediction >= HIGH_LINE
    tp = int(np.sum(actual_high & predicted_high))
    fp = int(np.sum(~actual_high & predicted_high))
    fn = int(np.sum(actual_high & ~predicted_high))
    return {
        "n": int(len(y)), "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "r2": float(1 - np.sum(error * error) / sst), "bias": float(error.mean()),
        "event_threshold": HIGH_LINE, "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
    }


def lgb_params() -> dict:
    return {
        "objective": "regression", "metric": "None", "learning_rate": 0.03,
        "num_leaves": 31, "min_data_in_leaf": 160, "max_bin": 63,
        "feature_fraction": 0.80, "bagging_fraction": 0.85, "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 3.0, "feature_pre_filter": False,
        "force_col_wise": True, "num_threads": max(1, min(os.cpu_count() or 1, 12)),
        "seed": SEED, "feature_fraction_seed": SEED, "bagging_seed": SEED,
        "verbosity": -1,
    }


def train_one_outer(outer: int, static, clusters, static_cols, cube, timestamps, distance_m, names):
    siteid = str(static.loc[outer, "siteid"])
    station_work = WORK / f"site_{siteid}"
    station_work.mkdir(parents=True, exist_ok=True)
    train, validation = choose_split(clusters, outer, CFG)
    known = np.setdiff1d(np.arange(len(static)), np.array([outer]))
    tuning_times = time_indices(timestamps, TRAIN_START, TRAIN_END)[::3]
    full_times = time_indices(timestamps, TRAIN_START, TRAIN_END)
    test_times = time_indices(timestamps, TEST_START, TEST_END)

    started = time.perf_counter()
    x_train, y_train, w_train, _, _ = build_matrix(
        station_work / "tune_train.npy", train, train, tuning_times, cube, timestamps,
        static, static_cols, train, distance_m, names,
    )
    x_valid, y_valid, w_valid, valid_groups, valid_counts = build_matrix(
        station_work / "tune_valid.npy", validation, train, tuning_times, cube, timestamps,
        static, static_cols, train, distance_m, names,
    )
    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, feature_name=names, free_raw_data=False)
    valid_set = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, feature_name=names, reference=train_set, free_raw_data=False)
    model = lgb.train(
        lgb_params(), train_set, num_boost_round=900, valid_sets=[valid_set],
        feval=macro_rmse_metric(valid_groups, len(valid_counts)),
        callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False), lgb.log_evaluation(50)],
    )
    best_iteration = int(model.best_iteration or model.current_iteration())
    valid_prediction = model.predict(x_valid, num_iteration=best_iteration)
    validation_metrics = regression_metrics(y_valid, valid_prediction)
    del model, train_set, valid_set, x_train, x_valid, y_train, y_valid, w_train, w_valid, valid_prediction
    gc.collect()

    x_final, y_final, w_final, _, _ = build_matrix(
        station_work / "final_train.npy", known, known, full_times, cube, timestamps,
        static, static_cols, known, distance_m, names,
    )
    final_set = lgb.Dataset(x_final, label=y_final, weight=w_final, feature_name=names, free_raw_data=False)
    final_model = lgb.train(lgb_params(), final_set, num_boost_round=best_iteration, callbacks=[lgb.log_evaluation(0)])
    del final_set, x_final, y_final, w_final
    gc.collect()

    static_raw, static_scaled, static_distance = static_context(static, static_cols, known)
    x_test, y_test, _ = build_target_block(
        cube, timestamps, outer, known, test_times, static, static_cols,
        static_raw, static_scaled, static_distance, distance_m,
    )
    prediction = final_model.predict(x_test, num_iteration=best_iteration)
    result = {
        "station_index": int(outer), "siteid": siteid, "sitename": str(static.loc[outer, "sitename"]),
        "best_iteration": best_iteration, "training_targets": 72, "outer_donors": 72,
        "validation": validation_metrics, "test": regression_metrics(y_test, prediction),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    del final_model, x_test, y_test, prediction
    gc.collect()
    safe_remove_work(station_work)
    return result


def parse_stations(value: str | None, static: pd.DataFrame) -> list[int]:
    if not value:
        return list(range(len(static)))
    requested = {token.strip() for token in value.split(",") if token.strip()}
    lookup = {str(row.siteid): int(index) for index, row in static.iterrows()}
    lookup.update({str(row.sitename): int(index) for index, row in static.iterrows()})
    missing = requested - set(lookup)
    if missing:
        raise ValueError(f"Unknown station identifiers: {sorted(missing)}")
    return [lookup[token] for token in requested]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", help="Comma-separated siteid or sitename; omit for all 73")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    static, clusters, static_cols, cube, timestamps = prepare_cube()
    names = feature_names(static_cols)
    distance_m = haversine_matrix(static.longitude.to_numpy(float), static.latitude.to_numpy(float))
    selected = parse_stations(args.stations, static)
    results_path = OUT / "v7_lgbm_station_metrics.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    completed = {int(row["station_index"]) for row in results}
    save_json(OUT / "v7_design.json", {
        "model": "station-balanced L2 LightGBM with permutation-invariant multiscale donor summaries",
        "dynamic_items": list(DYNAMIC_ITEMS), "lags": list(LAGS), "feature_count": len(names),
        "no_target_dynamic": True, "outer_protocol": "60/12 selects iterations; refit 72; outer has 72 donors",
        "files_kept": ["v7_design.json", "v7_lgbm_station_metrics.json", "v7_lgbm_station_metrics.csv"],
    })
    print(json.dumps({"output": str(OUT), "features": len(names), "stations_requested": len(selected)}, ensure_ascii=False), flush=True)
    for outer in tqdm(selected, desc="V7 outer LOSO", unit="station"):
        if outer in completed:
            continue
        result = train_one_outer(outer, static, clusters, static_cols, cube, timestamps, distance_m, names)
        results.append(result)
        save_json(results_path, results)
        flat = []
        for row in results:
            flat.append({
                "station_index": row["station_index"], "siteid": row["siteid"], "sitename": row["sitename"],
                "best_iteration": row["best_iteration"], "runtime_seconds": row["runtime_seconds"],
                **{f"validation_{k}": v for k, v in row["validation"].items()},
                **{f"test_{k}": v for k, v in row["test"].items()},
            })
        pd.DataFrame(flat).sort_values("station_index").to_csv(OUT / "v7_lgbm_station_metrics.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

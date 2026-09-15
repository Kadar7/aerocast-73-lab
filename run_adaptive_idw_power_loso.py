"""Leakage-safe adaptive IDW power selection for 73 strict outer targets.

2023-07..2024-06 builds known-station power curves; 2024-07..2025-06
selects the transfer rule using known stations; 2025-07..2026-06 evaluates
the completely excluded outer target. No target history selects its power.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import CFG
from data_pipeline import build_or_load_hourly_cube, haversine_matrix, load_static


POWERS = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
PERIODS = {
    "meta_train": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-06-30 23:00")),
    "meta_valid": (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-06-30 23:00")),
    "outer_test": (pd.Timestamp("2025-07-01"), pd.Timestamp("2026-06-30 23:00")),
}
TEMPERATURES = (0.0, 0.25, 0.5, 1.0)
NEIGHBOR_COUNTS = (1, 2, 3, 5, 10, 15)
NEIGHBOR_POWERS = (0, 1, 2)


def output_root() -> Path:
    requested = os.environ.get("AEROCAST_V7_OUTPUT")
    if requested:
        return Path(requested)
    drive = Path("/content/drive/MyDrive")
    return (drive / "AeroCast_V7_essential") if drive.is_dir() else (Path("/content") / "AeroCast_V7_essential")


def prepare_cube():
    original_start = CFG.train_start
    original_history = CFG.history_hours
    try:
        CFG.train_start = "2023-07-01 00:00:00"
        CFG.history_hours = max(original_history, 73)
        static, clusters, static_cols = load_static(CFG)
        cube, timestamps = build_or_load_hourly_cube(static, CFG)
    finally:
        CFG.train_start = original_start
        CFG.history_hours = original_history
    return static, clusters, static_cols, cube, timestamps


def all_power_predictions(actual: np.ndarray, distance_km: np.ndarray) -> np.ndarray:
    output = np.full((len(POWERS), len(actual), actual.shape[1]), np.nan, dtype="float32")
    for power_index, power in enumerate(tqdm(POWERS, desc="IDW powers", unit="power", leave=False)):
        weights = np.maximum(distance_km, 1.0) ** (-power)
        np.fill_diagonal(weights, 0.0)
        weights /= weights.max(axis=1, keepdims=True)
        weights = weights.astype("float32")
        for start in range(0, len(actual), 2048):
            values = actual[start : start + 2048]
            valid = np.isfinite(values)
            numerator = np.nan_to_num(values, nan=0.0) @ weights.T
            denominator = valid.astype("float32") @ weights.T
            output[power_index, start : start + len(values)] = np.divide(
                numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
            )
    return output


def error_statistics(actual: np.ndarray, prediction: np.ndarray):
    stations = actual.shape[1]
    rmses = np.zeros((stations, len(POWERS)), dtype="float64")
    grams = np.zeros((stations, len(POWERS), len(POWERS)), dtype="float64")
    variances = np.zeros(stations, dtype="float64")
    biases = np.zeros((stations, len(POWERS)), dtype="float64")
    maes = np.zeros((stations, len(POWERS)), dtype="float64")
    for station in range(stations):
        y = actual[:, station].astype(float)
        matrix = prediction[:, :, station].T.astype(float)
        valid = np.isfinite(y) & np.isfinite(matrix).all(axis=1)
        y = y[valid]
        errors = matrix[valid] - y[:, None]
        rmses[station] = np.sqrt(np.mean(errors * errors, axis=0))
        maes[station] = np.mean(np.abs(errors), axis=0)
        biases[station] = np.mean(errors, axis=0)
        grams[station] = errors.T @ errors / len(errors)
        variances[station] = np.mean((y - y.mean()) ** 2)
    return rmses, grams, variances, maes, biases


def weighted_metrics(gram: np.ndarray, variance: float, weights: np.ndarray):
    mse = float(weights @ gram @ weights)
    return np.sqrt(max(mse, 0.0)), 1.0 - mse / variance


def representation_matrices(static: pd.DataFrame, static_cols: list[str], distance_km: np.ndarray):
    raw = static[static_cols].apply(pd.to_numeric, errors="coerce")
    raw = raw.fillna(raw.median()).to_numpy(float)
    z = (raw - raw.mean(axis=0)) / np.where(raw.std(axis=0) < 1e-9, 1.0, raw.std(axis=0))
    static_distance = np.sqrt(np.mean((z[:, None, :] - z[None, :, :]) ** 2, axis=2))
    geo_scale = np.median(distance_km[np.isfinite(distance_km) & (distance_km > 0)])
    hybrid = np.sqrt((distance_km / geo_scale) ** 2 + static_distance ** 2)
    return static_distance, hybrid


def candidate_specs() -> list[tuple[str, int, float, int]]:
    specs: list[tuple[str, int, float, int]] = []
    for family in ("global", "cluster"):
        for temperature in TEMPERATURES:
            specs.append((family, 73, temperature, 0))
    for family in ("geo", "static", "hybrid"):
        for count in NEIGHBOR_COUNTS:
            for temperature in TEMPERATURES:
                for neighbor_power in NEIGHBOR_POWERS:
                    specs.append((family, count, temperature, neighbor_power))
    return specs


def make_weights(
    target: int,
    pool: np.ndarray,
    score: np.ndarray,
    spec: tuple[str, int, float, int],
    clusters: np.ndarray,
    matrices: dict[str, np.ndarray],
) -> np.ndarray:
    family, count, temperature, neighbor_power = spec
    if family == "global":
        selected = pool
    elif family == "cluster":
        selected = pool[clusters[pool] == clusters[target]]
        if len(selected) < 2:
            selected = pool
    else:
        selected = pool[np.argsort(matrices[family][target, pool])[: min(count, len(pool))]]
    if family in {"global", "cluster"} or neighbor_power == 0:
        station_weights = np.ones(len(selected), dtype=float)
    else:
        station_weights = 1.0 / np.maximum(matrices[family][target, selected], 1e-6) ** neighbor_power
    station_weights /= station_weights.sum()
    curve = np.sum(score[selected] * station_weights[:, None], axis=0)
    regret = curve - curve.min()
    if temperature == 0:
        weights = np.zeros(len(POWERS), dtype=float)
        weights[int(np.argmin(curve))] = 1.0
    else:
        weights = np.exp(-regret / temperature)
        weights /= weights.sum()
    return weights


def main() -> None:
    output = output_root()
    output.mkdir(parents=True, exist_ok=True)
    static, clusters, static_cols, cube, timestamps = prepare_cube()
    distance_km = haversine_matrix(static.longitude.to_numpy(float), static.latitude.to_numpy(float)) / 1000.0
    static_distance, hybrid = representation_matrices(static, static_cols, distance_km)
    matrices = {"geo": distance_km, "static": static_distance, "hybrid": hybrid}
    pm25_index = CFG.aq_cube_items.index("PM2.5")
    actual, prediction, statistics = {}, {}, {}
    for name, (start, end) in PERIODS.items():
        indices = np.flatnonzero((timestamps >= start) & (timestamps <= end))
        actual[name] = np.asarray(cube[indices, :, pm25_index], dtype="float32")
        prediction[name] = all_power_predictions(actual[name], distance_km)
        statistics[name] = error_statistics(actual[name], prediction[name])
        print(f"prepared {name}: {len(indices)} hours", flush=True)

    specs = candidate_specs()
    rows = []
    for outer in tqdm(range(len(static)), desc="nested adaptive power", unit="station"):
        known = np.setdiff1d(np.arange(len(static)), np.array([outer]))
        candidate_scores = []
        for spec in specs:
            held_scores = []
            for held in known:
                pool = known[known != held]
                weights = make_weights(held, pool, statistics["meta_train"][0], spec, clusters, matrices)
                held_scores.append(weighted_metrics(statistics["meta_valid"][1][held], statistics["meta_valid"][2][held], weights)[0])
            candidate_scores.append(float(np.mean(held_scores)))
        selected_spec = specs[int(np.argmin(candidate_scores))]
        weights = make_weights(outer, known, statistics["meta_valid"][0], selected_spec, clusters, matrices)
        adaptive_rmse, adaptive_r2 = weighted_metrics(statistics["outer_test"][1][outer], statistics["outer_test"][2][outer], weights)
        fixed = np.zeros(len(POWERS)); fixed[int(np.flatnonzero(POWERS == 2.0)[0])] = 1.0
        fixed_rmse, fixed_r2 = weighted_metrics(statistics["outer_test"][1][outer], statistics["outer_test"][2][outer], fixed)
        rows.append({
            "station_index": outer, "siteid": str(static.loc[outer, "siteid"]), "sitename": str(static.loc[outer, "sitename"]),
            "selector_family": selected_spec[0], "selector_neighbors": selected_spec[1],
            "selector_temperature": selected_spec[2], "selector_neighbor_power": selected_spec[3],
            "dominant_idw_power": float(POWERS[int(np.argmax(weights))]),
            "power_weights": json.dumps(dict(zip(POWERS.astype(str), weights.tolist())), ensure_ascii=False),
            "adaptive_rmse": adaptive_rmse, "adaptive_r2": adaptive_r2,
            "fixed_p2_rmse": fixed_rmse, "fixed_p2_r2": fixed_r2,
            "adaptive_minus_fixed_rmse": adaptive_rmse - fixed_rmse,
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "adaptive_idw_power_station_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "protocol": "outer target excluded; 2023-24 power curves -> 2024-25 nested selector -> 2025-26 outer test",
        "selection_uses_outer_history_or_truth": False,
        "powers": POWERS.tolist(), "stations": len(table),
        "fixed_p2": {"macro_rmse": float(table.fixed_p2_rmse.mean()), "macro_r2": float(table.fixed_p2_r2.mean())},
        "adaptive": {"macro_rmse": float(table.adaptive_rmse.mean()), "macro_r2": float(table.adaptive_r2.mean())},
        "adaptive_station_wins": int((table.adaptive_rmse < table.fixed_p2_rmse).sum()),
        "selector_counts": table[["selector_family", "selector_neighbors", "selector_temperature", "selector_neighbor_power"]].value_counts().rename("stations").reset_index().to_dict("records"),
        "dominant_power_counts": table.dominant_idw_power.value_counts().sort_index().to_dict(),
        "files_kept": ["adaptive_idw_power_summary.json", "adaptive_idw_power_station_metrics.csv"],
    }
    (output / "adaptive_idw_power_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

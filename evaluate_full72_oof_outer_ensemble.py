from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    build_or_load_hourly_cube,
    haversine_matrix,
    load_static,
    standardize_static,
)
from evaluate_outer import load_checkpoint, scaler_from_checkpoint
from evaluate_validation_only_snapshot_ensemble import (
    EPOCHS,
    aligned_predictions,
    fit_convex_weights,
    station_rmse,
    station_rmse_by_epoch,
)
from model import TCNTargetCrossAttention
from sanity_check import audit_dataset
from train_formal import (
    DeviceFeatureBuilder,
    make_index_loader,
    make_vectorized_loader,
    regression_metrics,
    resolve_target,
)


METHOD_SPECS = (
    *(('uniform_top_k', float(k)) for k in (2, 3, 5, 8, 15)),
    *(('softmax_macro_regret', float(x)) for x in (0.25, 0.5, 1.0, 2.0)),
    ('best_epoch_frequency', 0.0),
    *(('convex_ridge', float(x)) for x in (0.0, 0.01, 0.1, 1.0, 10.0)),
)


def load_fold_oof(root: Path, fold: int) -> tuple[pd.DataFrame, np.ndarray]:
    base = None
    columns = []
    for epoch in EPOCHS:
        path = root / f'fold_{fold:02d}' / 'epoch_predictions' / f'epoch_{epoch:03d}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'缺少完整OOF prediction: {path}')
        with np.load(path, allow_pickle=False) as saved:
            current = pd.DataFrame({
                'station_index': saved['station_index'].astype(int),
                'timestamp_ns': saved['timestamp_ns'].astype('int64'),
                'y_true': saved['y_true'].astype('float64'),
            })
            prediction = saved['y_pred'].astype('float64')
        if base is None:
            base = current
        else:
            if not np.array_equal(base.station_index.to_numpy(), current.station_index.to_numpy()):
                raise RuntimeError(f'fold {fold} epoch prediction station順序不一致')
            if not np.array_equal(base.timestamp_ns.to_numpy(), current.timestamp_ns.to_numpy()):
                raise RuntimeError(f'fold {fold} epoch prediction timestamp不一致')
            if not np.allclose(base.y_true, current.y_true, rtol=0, atol=1e-6):
                raise RuntimeError(f'fold {fold} epoch prediction truth不一致')
        columns.append(prediction)
    assert base is not None
    base['timestamp'] = pd.to_datetime(base.pop('timestamp_ns'))
    base['fold'] = fold
    if base.station_index.nunique() != 12:
        raise RuntimeError(f'fold {fold} OOF不是12個validation stations')
    return base, np.stack(columns, axis=1)


def load_all_oof(root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    bases, matrices = [], []
    station_sets = []
    for fold in range(6):
        base, matrix = load_fold_oof(root, fold)
        bases.append(base)
        matrices.append(matrix)
        station_sets.append(set(base.station_index.unique().astype(int)))
    union = set().union(*station_sets)
    if len(union) != 72 or sum(len(x) for x in station_sets) != 72:
        raise RuntimeError('六折OOF沒有形成互斥且完整的72站')
    return pd.concat(bases, ignore_index=True), np.concatenate(matrices, axis=0)


def fit_epoch_weights(
    base: pd.DataFrame,
    matrix: np.ndarray,
    stations: np.ndarray,
    family: str,
    parameter: float,
) -> np.ndarray:
    curves = station_rmse_by_epoch(base, matrix, stations)
    regret = curves - curves.min(axis=1, keepdims=True)
    macro_regret = regret.mean(axis=0)
    if family == 'uniform_top_k':
        k = int(parameter)
        weight = np.zeros(len(EPOCHS), dtype=float)
        weight[np.argsort(macro_regret, kind='stable')[:k]] = 1.0 / k
        return weight
    if family == 'softmax_macro_regret':
        positive = macro_regret[macro_regret > 1e-12]
        scale = float(np.median(positive)) if len(positive) else 1.0
        temperature = max(scale * parameter, 1e-8)
        logits = -macro_regret / temperature
        logits -= logits.max()
        weight = np.exp(logits)
        return weight / weight.sum()
    if family == 'best_epoch_frequency':
        frequency = np.bincount(np.argmin(curves, axis=1), minlength=len(EPOCHS)).astype(float)
        return frequency / frequency.sum()
    if family == 'convex_ridge':
        return fit_convex_weights(base, matrix, stations, parameter)
    raise ValueError(f'未知ensemble family: {family}')


def macro_rmse(base: pd.DataFrame, prediction: np.ndarray, stations: np.ndarray) -> float:
    return float(np.mean([station_rmse(base, prediction, int(s)) for s in stations]))


def station_bias(base: pd.DataFrame, prediction: np.ndarray, station: int) -> float:
    keep = base.station_index.to_numpy(int) == int(station)
    truth = base.y_true.to_numpy(float)[keep]
    return float(np.mean(prediction[keep] - truth))


def fit_fold_calibration(
    base: pd.DataFrame,
    matrix: np.ndarray,
    epoch_weights: np.ndarray,
    selected_method: dict,
) -> tuple[list[dict], bool]:
    """Estimate each fold's offset/reliability using only its 12 OOF stations."""
    deployment_prior = matrix @ epoch_weights
    rows = []
    for fold in range(6):
        stations = np.unique(base.loc[base.fold == fold, 'station_index'].to_numpy(int))
        train_stations = np.unique(base.loc[base.fold != fold, 'station_index'].to_numpy(int))
        evaluation_weights = fit_epoch_weights(
            base, matrix, train_stations,
            selected_method['family'], selected_method['parameter'],
        )
        evaluation_prior = matrix @ evaluation_weights
        evaluation_biases = {
            int(s): station_bias(base, evaluation_prior, int(s)) for s in stations
        }
        deployment_biases = {
            int(s): station_bias(base, deployment_prior, int(s)) for s in stations
        }
        full_bias = float(np.mean(list(deployment_biases.values())))
        raw_rmse = macro_rmse(base, evaluation_prior, stations)
        held_rmse = []
        for held in stations:
            calibration_bias = float(np.mean([
                value for station_id, value in evaluation_biases.items()
                if station_id != int(held)
            ]))
            keep = base.station_index.to_numpy(int) == int(held)
            truth = base.y_true.to_numpy(float)[keep]
            corrected = evaluation_prior[keep] - calibration_bias
            held_rmse.append(float(np.sqrt(np.mean((corrected - truth) ** 2))))
        rows.append({
            'fold': fold,
            'oof_stations': int(len(stations)),
            'oof_macro_bias': full_bias,
            'raw_oof_macro_rmse': raw_rmse,
            'station_loso_bias_corrected_macro_rmse': float(np.mean(held_rmse)),
        })
    raw_score = float(np.mean([row['raw_oof_macro_rmse'] for row in rows]))
    corrected_score = float(np.mean([
        row['station_loso_bias_corrected_macro_rmse'] for row in rows
    ]))
    use_bias_correction = corrected_score < raw_score
    reliability = np.asarray([
        row['station_loso_bias_corrected_macro_rmse']
        if use_bias_correction else row['raw_oof_macro_rmse']
        for row in rows
    ], dtype=float)
    reliability_weights = 1.0 / np.maximum(reliability, 1e-8) ** 2
    reliability_weights /= reliability_weights.sum()
    for row, weight in zip(rows, reliability_weights):
        row['outer_model_weight'] = float(weight)
    return rows, use_bias_correction


def select_method_by_sixfold_oof(
    base: pd.DataFrame, matrix: np.ndarray
) -> tuple[dict, list[dict]]:
    fold_values = sorted(base.fold.unique().astype(int).tolist())
    if len(fold_values) < 2:
        raise ValueError('ensemble method selection至少需要2個OOF folds')
    rows = []
    for family, parameter in METHOD_SPECS:
        held_scores = []
        for fold in fold_values:
            train_stations = np.unique(base.loc[base.fold != fold, 'station_index'].to_numpy(int))
            held_stations = np.unique(base.loc[base.fold == fold, 'station_index'].to_numpy(int))
            weight = fit_epoch_weights(base, matrix, train_stations, family, parameter)
            held_scores.append(macro_rmse(base, matrix @ weight, held_stations))
        rows.append({
            'family': family,
            'parameter': parameter,
            'sixfold_oof_macro_rmse': float(np.mean(held_scores)),
            'fold_rmse': held_scores,
        })
    return min(rows, key=lambda row: row['sixfold_oof_macro_rmse']), rows


def build_outer_fold_context(
    root: Path,
    fold: int,
    static: pd.DataFrame,
    static_cols: list[str],
    cube,
    timestamps,
    distance,
    outer: int,
):
    paths = [root / f'fold_{fold:02d}' / 'epoch_checkpoints' / f'epoch_{e:03d}.pt' for e in EPOCHS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'fold {fold} 缺少{len(missing)}個checkpoints')
    first = load_checkpoint(paths[0])
    train_idx = np.asarray(first['train_indices'], dtype=int)
    val_idx = np.asarray(first['validation_indices'], dtype=int)
    if len(train_idx) != 60 or len(val_idx) != 12 or outer in train_idx or outer in val_idx:
        raise RuntimeError(f'fold {fold} checkpoint protocol錯誤')
    scaler = scaler_from_checkpoint(first)
    static_scaled = standardize_static(static, static_cols, scaler)
    outer_ds = ColdStartStationDataset(
        [outer], train_idx, CFG.test_start, CFG.test_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    if audit_dataset(outer_ds, max_samples=32)['sampled_donor_counts'] != [60]:
        raise RuntimeError(f'fold {fold} outer donors不是60')
    builder = None
    if CFG.device.type == 'cuda':
        builder = DeviceFeatureBuilder(
            train_idx, cube, int(outer_ds.row_times.max()), timestamps,
            static, static_scaled, distance, scaler, outer, CFG.device,
        )
        loader = make_index_loader(outer_ds, False)
    else:
        loader = make_vectorized_loader(
            outer_ds, train_idx, cube, timestamps, static, static_scaled,
            distance, scaler, False,
        )
    return paths, loader, builder


def main() -> None:
    apply_runtime_profile(CFG)
    root = Path(os.environ.get(
        'DL_TCN_CROSSFIT_ROOT',
        str(CFG.formal_output_root / 'crossfit_target_conditioned_snapshots'),
    ))
    base, matrix = load_all_oof(root)
    if matrix.shape[1] != 15:
        raise RuntimeError('OOF prediction matrix不是15 epochs')
    method, method_rows = select_method_by_sixfold_oof(base, matrix)
    stations = np.unique(base.station_index.to_numpy(int))
    final_weights = fit_epoch_weights(
        base, matrix, stations, method['family'], method['parameter']
    )
    fold_calibration, use_bias_correction = fit_fold_calibration(
        base, matrix, final_weights, method
    )
    lock = {
        'protocol': '72 disjoint OOF stations select ensemble; outer truth unused',
        'selection_used_outer_truth': False,
        'oof_stations': int(len(stations)),
        'selected_epoch_method': method,
        'candidate_scores': sorted(method_rows, key=lambda row: row['sixfold_oof_macro_rmse']),
        'weights': [
            {'epoch': int(epoch), 'weight': float(weight)}
            for epoch, weight in zip(EPOCHS, final_weights) if weight > 1e-8
        ],
        'fold_calibration': {
            'selection_used_outer_truth': False,
            'bias_correction_accepted_by_station_loso': use_bias_correction,
            'folds': fold_calibration,
        },
    }
    print('\nLOCKED FULL-72 OOF ENSEMBLE BEFORE OUTER TRUTH')
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)

    static, _, static_cols = load_static()
    outer = resolve_target(static, CFG.target_site)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    fold_base_predictions = []
    fold_calibrated_predictions = []
    outer_base = None
    for fold in range(6):
        print(f'outer inference fold {fold}/5', flush=True)
        paths, loader, builder = build_outer_fold_context(
            root, fold, static, static_cols, cube, timestamps, distance, outer
        )
        model = TCNTargetCrossAttention(static_dim=len(static_cols)).to(CFG.device)
        current_base, current_matrix = aligned_predictions(
            model, paths, loader, CFG.device, static, timestamps, builder
        )
        if outer_base is None:
            outer_base = current_base
        else:
            if not np.array_equal(
                pd.to_datetime(outer_base.timestamp).astype('int64').to_numpy(),
                pd.to_datetime(current_base.timestamp).astype('int64').to_numpy(),
            ):
                raise RuntimeError('六個fold的outer timestamps沒有對齊')
            if not np.allclose(outer_base.y_true, current_base.y_true, rtol=0, atol=1e-6):
                raise RuntimeError('六個fold的outer truth沒有對齊')
        fold_prior = current_matrix @ final_weights
        fold_base_predictions.append(fold_prior)
        correction = fold_calibration[fold]['oof_macro_bias'] if use_bias_correction else 0.0
        fold_calibrated_predictions.append(fold_prior - correction)
    assert outer_base is not None
    raw_equal_prediction = np.mean(fold_base_predictions, axis=0)
    calibrated_equal_prediction = np.mean(fold_calibrated_predictions, axis=0)
    fold_weights = np.asarray(
        [row['outer_model_weight'] for row in fold_calibration], dtype=float
    )
    deployed = np.sum(
        np.stack(fold_calibrated_predictions, axis=0) * fold_weights[:, None], axis=0
    )
    result = {
        'target_station_index': int(outer),
        'siteid': str(static.loc[outer, 'siteid']),
        'sitename': str(static.loc[outer, 'sitename']),
        'selection_used_outer_truth': False,
        'raw_equal_six_model_metrics': regression_metrics(
            outer_base.y_true.to_numpy(float), raw_equal_prediction
        ),
        'bias_corrected_equal_six_model_metrics': regression_metrics(
            outer_base.y_true.to_numpy(float), calibrated_equal_prediction
        ),
        'deployed_method': 'oof_reliability_weighted_bias_corrected_epoch_ensemble',
        'bias_correction_enabled': use_bias_correction,
        'fold_model_weights': [float(x) for x in fold_weights],
        'deployed_metrics': regression_metrics(
            outer_base.y_true.to_numpy(float), deployed
        ),
        'fold_models': 6,
        'snapshots_per_fold': 15,
        'donors_per_fold': 60,
        'files_written': 0,
    }
    print('\nFULL-72 OOF / SIX-MODEL OUTER RESULT')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

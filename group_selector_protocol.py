from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import CFG
from evaluate_full72_oof_outer_ensemble import (
    fit_epoch_weights,
    select_method_by_sixfold_oof,
)
from evaluate_validation_only_snapshot_ensemble import EPOCHS


def _balanced_cluster_folds(
    members: np.ndarray, clusters: np.ndarray, folds: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(folds)]
    cluster_counts = [dict() for _ in range(folds)]
    for cluster in sorted(np.unique(clusters[members])):
        current = members[clusters[members] == cluster].copy()
        rng.shuffle(current)
        for station in current:
            sizes = np.asarray([len(x) for x in buckets])
            minimum = sizes.min()
            candidates = np.flatnonzero(sizes == minimum)
            chosen = min(
                candidates.tolist(),
                key=lambda fold: (cluster_counts[fold].get(int(cluster), 0), fold),
            )
            buckets[chosen].append(int(station))
            cluster_counts[chosen][int(cluster)] = (
                cluster_counts[chosen].get(int(cluster), 0) + 1
            )
    result = [np.asarray(sorted(x), dtype=int) for x in buckets]
    joined = np.concatenate(result)
    if set(joined.tolist()) != set(members.tolist()) or len(joined) != len(np.unique(joined)):
        raise RuntimeError('cluster-balanced folds沒有形成互斥完整分割')
    if max(map(len, result)) - min(map(len, result)) > 1:
        raise RuntimeError('cluster-balanced folds大小不平衡')
    return result


def make_outer_groups(clusters: np.ndarray) -> list[np.ndarray]:
    members = np.arange(len(clusters), dtype=int)
    groups = _balanced_cluster_folds(members, clusters, 6, CFG.seed + 20260904)
    if sorted(map(len, groups)) != [12, 12, 12, 12, 12, 13]:
        raise RuntimeError(f'outer groups大小錯誤: {[len(x) for x in groups]}')
    return groups


def make_selector_inner_folds(
    clusters: np.ndarray, excluded: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    known = np.setdiff1d(np.arange(len(clusters), dtype=int), excluded)
    validation_folds = _balanced_cluster_folds(
        known, clusters, 6, CFG.seed + 20260905 + int(excluded[0])
    )
    folds = [(np.setdiff1d(known, validation), validation) for validation in validation_folds]
    union = np.concatenate([validation for _, validation in folds])
    if set(union.tolist()) != set(known.tolist()) or len(union) != len(np.unique(union)):
        raise RuntimeError('selector inner OOF沒有完整覆蓋known stations')
    return folds


def load_group_oof(root: Path, expected_stations: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    bases, matrices = [], []
    for fold in range(6):
        base = None
        columns = []
        for epoch in EPOCHS:
            path = root / f'fold_{fold:02d}' / 'epoch_predictions' / f'epoch_{epoch:03d}.npz'
            if not path.is_file():
                raise FileNotFoundError(f'缺少group-selector OOF prediction: {path}')
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
                if not np.array_equal(base.station_index, current.station_index):
                    raise RuntimeError(f'group fold {fold} station順序不一致')
                if not np.array_equal(base.timestamp_ns, current.timestamp_ns):
                    raise RuntimeError(f'group fold {fold} timestamp不一致')
                if not np.allclose(base.y_true, current.y_true, rtol=0, atol=1e-6):
                    raise RuntimeError(f'group fold {fold} truth不一致')
            columns.append(prediction)
        assert base is not None
        base['timestamp'] = pd.to_datetime(base.pop('timestamp_ns'))
        base['fold'] = fold
        bases.append(base)
        matrices.append(np.stack(columns, axis=1))
    combined = pd.concat(bases, ignore_index=True)
    observed = np.unique(combined.station_index.to_numpy(int))
    if not np.array_equal(np.sort(observed), np.sort(expected_stations)):
        raise RuntimeError('group-selector OOF station集合錯誤')
    return combined, np.concatenate(matrices, axis=0)


def make_group_lock(
    root: Path,
    group_id: int,
    excluded: np.ndarray,
    static: pd.DataFrame,
) -> dict:
    known = np.setdiff1d(np.arange(len(static), dtype=int), excluded)
    base, matrix = load_group_oof(root, known)
    method, candidates = select_method_by_sixfold_oof(base, matrix)
    weights = fit_epoch_weights(
        base, matrix, known, method['family'], method['parameter']
    )
    return {
        'protocol': (
            'six outer groups; complete held group excluded from six-fold selector; '
            'selector folds use 50/51 training and 10/11 validation stations; '
            'weights locked before each target 72-refit and outer truth'
        ),
        'selection_used_outer_truth': False,
        'outer_group': int(group_id),
        'outer_indices': excluded.tolist(),
        'outer_siteids': static.loc[excluded, 'siteid'].astype(str).tolist(),
        'outer_sitenames': static.loc[excluded, 'sitename'].astype(str).tolist(),
        'selector_known_stations': int(len(known)),
        'selected_method': method,
        'candidate_scores': sorted(
            candidates, key=lambda row: row['sixfold_oof_macro_rmse']
        ),
        'epoch_weights': [float(x) for x in weights],
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)

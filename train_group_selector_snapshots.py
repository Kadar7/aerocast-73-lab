from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from config import CFG, apply_runtime_profile
from data_pipeline import build_or_load_hourly_cube, haversine_matrix, load_static
from group_selector_protocol import make_outer_groups, make_selector_inner_folds
from train_crossfit_snapshots import run_fold
from train_formal import seed_all


def main() -> None:
    seed_all(CFG.seed)
    runtime = apply_runtime_profile(CFG)
    group_id = int(os.environ['DL_TCN_OUTER_GROUP_ID'])
    root = Path(os.environ['DL_TCN_GROUP_SELECTOR_ROOT'])
    static, clusters, static_cols = load_static()
    groups = make_outer_groups(clusters)
    if group_id < 0 or group_id >= len(groups):
        raise ValueError('DL_TCN_OUTER_GROUP_ID必須是0..5')
    excluded = groups[group_id]
    folds = make_selector_inner_folds(clusters, excluded)
    root.mkdir(parents=True, exist_ok=True)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    summaries = []
    for fold_id, (training, validation) in enumerate(folds):
        summaries.append(run_fold(
            fold_id, training, validation, int(excluded[0]),
            static, clusters, static_cols, cube, timestamps, distance,
            root, CFG.device, excluded_indices=excluded,
        ))
    print(json.dumps({
        'status': 'group_selector_complete',
        'group_id': group_id,
        'excluded_siteids': static.loc[excluded, 'siteid'].astype(str).tolist(),
        'selector_folds': len(folds),
        'runtime_profile': runtime,
        'root': str(root),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Fixed group 0 / inner folds 0 and 3. Validation only; no outer or refit."""
from __future__ import annotations
import json
import os
from pathlib import Path


def main():
    os.environ['DL_TCN_MODEL_VERSION'] = 'v2'
    os.environ.setdefault('DL_TCN_MAX_EPOCHS', '15')
    from config import CFG, apply_runtime_profile
    from data_pipeline import load_static, build_or_load_hourly_cube, haversine_matrix
    from group_selector_protocol import make_outer_groups, make_selector_inner_folds
    from train_crossfit_snapshots import run_fold
    from v2_training import v2_config_payload
    root = Path(os.environ.get('DL_TCN_V4_PILOT_ROOT', '/content/DL_TCN_V4_VALIDATION_PILOT'))
    settings = {'training': v2_config_payload(), 'group': 0, 'folds': [0,3],
                'epochs': CFG.max_epochs, 'lr': CFG.learning_rate,
                'weight_decay': CFG.weight_decay, 'dropout': CFG.dropout}
    marker = root / 'pilot_settings.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != settings:
            raise RuntimeError('Pilot settings changed; use a fresh pilot output root')
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError('Unversioned pilot directory; use a fresh output root')
    else:
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(settings,indent=2),encoding='utf-8')
    runtime = apply_runtime_profile(CFG)
    static, clusters, columns = load_static()
    excluded = make_outer_groups(clusters)[0]
    splits = make_selector_inner_folds(clusters, excluded)
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude,static.latitude)
    print(json.dumps({'mode':'validation_only_pilot','runtime':runtime,'settings':settings,
                      'root':str(root)},ensure_ascii=False,indent=2),flush=True)
    for fold_id in (0,3):
        training, validation = splits[fold_id]
        run_fold(fold_id,training,validation,int(excluded[0]),static,clusters,columns,
                 cube,timestamps,distance,root,CFG.device,excluded_indices=excluded)
    print('V4 pilot complete: two validation folds only; no outer metrics or 72-refit.',flush=True)


if __name__ == '__main__':
    main()

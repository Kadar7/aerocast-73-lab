from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from data_pipeline import load_static
from group_selector_protocol import atomic_json, make_group_lock, make_outer_groups
from run_73_station_loso import rebuild_summary, safe_siteid


SCRIPT_DIR = Path(__file__).resolve().parent


def run(script: str, environment: dict) -> None:
    print(f'\nRUN: {sys.executable} {script}', flush=True)
    subprocess.run(
        [sys.executable, script], cwd=SCRIPT_DIR, env=environment, check=True
    )


def remove_tree(path: Path, required_parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != required_parent.resolve():
        raise RuntimeError(f'拒絕刪除非預期暫存目錄: {resolved}')
    if resolved.exists():
        shutil.rmtree(resolved)


def main() -> None:
    output_root = Path(os.environ.get(
        'DL_TCN_GROUP_73_OUTPUT_ROOT',
        '/content/drive/MyDrive/DL_TCN_73_GROUP_SELECTOR',
    ))
    if str(output_root).startswith('/content/drive/') and not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Google Drive尚未掛載')
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = output_root / 'station_summaries'
    predictions = output_root / 'station_predictions'
    locks = output_root / 'group_locks'
    active_selectors = output_root / '_active_selectors'
    active_refits = output_root / '_active_refits'
    for path in (summaries, predictions, locks, active_selectors, active_refits):
        path.mkdir(exist_ok=True)

    static, clusters, _ = load_static()
    groups = make_outer_groups(clusters)
    requested_raw = os.environ.get('DL_TCN_GROUP_IDS', 'all').strip().lower()
    requested = list(range(6)) if requested_raw in {'', 'all'} else [
        int(x.strip()) for x in requested_raw.split(',') if x.strip()
    ]
    if not requested or any(x < 0 or x > 5 for x in requested):
        raise ValueError('DL_TCN_GROUP_IDS只能是all或0..5的逗號清單')
    maximum_new = int(os.environ.get('DL_TCN_MAX_NEW_STATIONS', '0'))
    completed_now = 0

    for group_id in requested:
        excluded = groups[group_id]
        selector_root = active_selectors / f'group_{group_id:02d}'
        lock_path = locks / f'group_{group_id:02d}.json'
        if not lock_path.is_file():
            environment = os.environ.copy()
            environment.update({
                'DL_TCN_OUTER_GROUP_ID': str(group_id),
                'DL_TCN_GROUP_SELECTOR_ROOT': str(selector_root),
                'DL_TCN_SAVE_EPOCH_PREDICTIONS': '1',
                'DL_TCN_MAX_EPOCHS': '15',
            })
            run('train_group_selector_snapshots.py', environment)
            lock = make_group_lock(selector_root, group_id, excluded, static)
            atomic_json(lock_path, lock)
        if selector_root.exists():
            remove_tree(selector_root, active_selectors)

        for target in excluded:
            siteid = str(static.loc[target, 'siteid'])
            sitename = str(static.loc[target, 'sitename'])
            token = safe_siteid(siteid)
            summary_path = summaries / f'site_{token}.json'
            prediction_path = predictions / f'site_{token}.npz'
            active = active_refits / f'site_{token}'
            if summary_path.is_file() and prediction_path.is_file():
                if active.exists():
                    remove_tree(active, active_refits)
                print(f'group {group_id} {siteid} {sitename}: completed, skip', flush=True)
                continue
            if maximum_new > 0 and completed_now >= maximum_new:
                rebuild_summary(output_root)
                print(json.dumps({
                    'status': 'session_limit_reached',
                    'completed_new_stations': completed_now,
                    'output_root': str(output_root),
                }, ensure_ascii=False, indent=2), flush=True)
                return
            active.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update({
                'DL_TCN_TARGET_SITE': siteid,
                'DL_TCN_LOCKED_ENSEMBLE_PATH': str(lock_path),
                'DL_TCN_REFIT_OUTPUT_ROOT': str(output_root),
                'DL_TCN_REFIT_RESUME_PATH': str(active / 'refit_last_state.pt'),
                'DL_TCN_MAX_EPOCHS': '15',
            })
            run('train_refit72_outer_epoch_ensemble.py', environment)
            if not summary_path.is_file() or not prediction_path.is_file():
                raise RuntimeError(f'{siteid} 最終輸出不完整')
            remove_tree(active, active_refits)
            rebuild_summary(output_root)
            completed_now += 1

    rebuild_summary(output_root)
    print(json.dumps({
        'status': 'complete',
        'completed_stations': len(list(summaries.glob('site_*.json'))),
        'output_root': str(output_root),
        'resume_rule': 'rerun same command; completed targets skip, selector/refit resumes',
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

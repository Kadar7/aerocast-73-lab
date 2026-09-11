from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import load_static
from train_formal import regression_metrics
from train_refit72_outer_epoch_ensemble import event_metrics


SCRIPT_DIR = Path(__file__).resolve().parent


def safe_siteid(value: str) -> str:
    cleaned = re.sub(r'[^0-9A-Za-z_-]+', '_', str(value)).strip('_')
    if not cleaned:
        raise ValueError(f'無法建立安全siteid: {value!r}')
    return cleaned


def selected_sites(static: pd.DataFrame) -> list[dict]:
    requested = os.environ.get('DL_TCN_SITE_IDS', 'all').strip()
    records = static[['siteid', 'sitename']].astype(str).to_dict('records')
    if requested.lower() in {'', 'all'}:
        return records
    wanted = {item.strip() for item in requested.split(',') if item.strip()}
    selected = [row for row in records if row['siteid'] in wanted or row['sitename'] in wanted]
    found = {row['siteid'] for row in selected} | {row['sitename'] for row in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f'找不到指定測站: {missing}')
    return selected


def run(command: list[str], environment: dict) -> None:
    print('\nRUN:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=SCRIPT_DIR, env=environment, check=True)


def rebuild_summary(output_root: Path) -> None:
    rows = []
    truths = []
    ensemble_predictions = []
    oracle_predictions = []
    for path in sorted((output_root / 'station_summaries').glob('site_*.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        ensemble = result['ensemble']
        oracle = result['oracle']
        rows.append({
            'siteid': result['siteid'],
            'sitename': result['sitename'],
            'n': result['truth_timestamps'],
            **{f'ensemble_{k}': v for k, v in ensemble['regression'].items()},
            **{f'ensemble_{k}': v for k, v in ensemble['events'].items()},
            'oracle_epoch': oracle['epoch'],
            **{f'oracle_{k}': v for k, v in oracle['regression'].items()},
            **{f'oracle_{k}': v for k, v in oracle['events'].items()},
            'runtime_seconds': result['training_runtime_seconds'],
            'peak_vram_mb': result['peak_vram_mb'],
        })
        prediction_path = (
            output_root / 'station_predictions'
            / f"site_{safe_siteid(result['siteid'])}.npz"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(f'摘要存在但prediction遺失: {prediction_path}')
        with np.load(prediction_path, allow_pickle=False) as saved:
            truths.append(saved['y_true'].astype(float))
            ensemble_predictions.append(saved['ensemble_prediction'].astype(float))
            oracle_predictions.append(saved['oracle_prediction'].astype(float))
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values('siteid')
    path = output_root / 'station_metrics_summary.csv'
    temporary = path.with_suffix('.csv.tmp')
    frame.to_csv(temporary, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)
    truth = np.concatenate(truths)
    ensemble = np.concatenate(ensemble_predictions)
    oracle = np.concatenate(oracle_predictions)
    aggregate = {
        'completed_stations': int(len(frame)),
        'total_truth_timestamps': int(len(truth)),
        'ensemble': {
            'pooled_regression': regression_metrics(truth, ensemble),
            'pooled_events': event_metrics(truth, ensemble),
            'macro_station_rmse': float(frame['ensemble_rmse'].mean()),
            'macro_station_mae': float(frame['ensemble_mae'].mean()),
            'macro_station_r2': float(frame['ensemble_r2'].mean()),
            'macro_station_bias': float(frame['ensemble_bias'].mean()),
        },
        'oracle_reference': {
            'uses_each_outer_truth': True,
            'pooled_regression': regression_metrics(truth, oracle),
            'pooled_events': event_metrics(truth, oracle),
            'macro_station_rmse': float(frame['oracle_rmse'].mean()),
            'macro_station_mae': float(frame['oracle_mae'].mean()),
            'macro_station_r2': float(frame['oracle_r2'].mean()),
            'macro_station_bias': float(frame['oracle_bias'].mean()),
        },
    }
    aggregate_path = output_root / 'aggregate_metrics.json'
    aggregate_tmp = aggregate_path.with_suffix('.json.tmp')
    aggregate_tmp.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(aggregate_tmp, aggregate_path)


def remove_completed_temporary(active_site_root: Path, active_root: Path) -> None:
    resolved_site = active_site_root.resolve()
    resolved_active = active_root.resolve()
    if resolved_site.parent != resolved_active:
        raise RuntimeError(f'拒絕刪除非active子目錄: {resolved_site}')
    if resolved_site.exists():
        shutil.rmtree(resolved_site)


def main() -> None:
    drive_root = Path(os.environ.get(
        'DL_TCN_73_OUTPUT_ROOT', '/content/drive/MyDrive/DL_TCN_73_LOSO'
    ))
    if str(drive_root).startswith('/content/drive/') and not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Google Drive尚未掛載，請先drive.mount("/content/drive")')
    drive_root.mkdir(parents=True, exist_ok=True)
    active_root = drive_root / '_active_station'
    active_root.mkdir(parents=True, exist_ok=True)
    (drive_root / 'station_summaries').mkdir(exist_ok=True)
    (drive_root / 'station_predictions').mkdir(exist_ok=True)

    static, _, _ = load_static()
    sites = selected_sites(static)
    maximum_new = int(os.environ.get('DL_TCN_MAX_NEW_STATIONS', '0'))
    completed_now = 0
    for position, site in enumerate(sites, start=1):
        siteid = str(site['siteid'])
        token = safe_siteid(siteid)
        summary_path = drive_root / 'station_summaries' / f'site_{token}.json'
        prediction_path = drive_root / 'station_predictions' / f'site_{token}.npz'
        if summary_path.is_file() and prediction_path.is_file():
            leftover = active_root / f'site_{token}'
            if leftover.exists():
                remove_completed_temporary(leftover, active_root)
            print(f'[{position}/{len(sites)}] {siteid} {site["sitename"]}: completed, skip', flush=True)
            continue
        if maximum_new > 0 and completed_now >= maximum_new:
            break

        print(f'\n[{position}/{len(sites)}] START {siteid} {site["sitename"]}', flush=True)
        active_site_root = active_root / f'site_{token}'
        crossfit_root = active_site_root / 'crossfit'
        resume_path = active_site_root / 'refit_last_state.pt'
        active_site_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update({
            'DL_TCN_TARGET_SITE': siteid,
            'DL_TCN_CROSSFIT_ROOT': str(crossfit_root),
            'DL_TCN_CROSSFIT_FOLDS': 'all',
            'DL_TCN_SAVE_EPOCH_PREDICTIONS': '1',
            'DL_TCN_MAX_EPOCHS': '15',
            'DL_TCN_REFIT_OUTPUT_ROOT': str(drive_root),
            'DL_TCN_REFIT_RESUME_PATH': str(resume_path),
        })
        run([sys.executable, 'train_crossfit_snapshots.py'], environment)
        run([sys.executable, 'train_refit72_outer_epoch_ensemble.py'], environment)
        if not summary_path.is_file() or not prediction_path.is_file():
            raise RuntimeError(f'{siteid} 執行完成但最終輸出不完整')
        rebuild_summary(drive_root)
        remove_completed_temporary(active_site_root, active_root)
        completed_now += 1
        print(f'[{position}/{len(sites)}] DONE {siteid} {site["sitename"]}', flush=True)

    rebuild_summary(drive_root)
    completed = len(list((drive_root / 'station_summaries').glob('site_*.json')))
    print(json.dumps({
        'status': 'session_complete',
        'completed_stations': completed,
        'requested_stations': len(sites),
        'output_root': str(drive_root),
        'resume_rule': 'rerun same command; completed stations skip and active station resumes',
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

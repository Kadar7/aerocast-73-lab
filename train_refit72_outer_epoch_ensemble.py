from __future__ import annotations

import json
import hashlib
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import CFG, apply_runtime_profile
from data_pipeline import (
    ColdStartStationDataset,
    build_or_load_hourly_cube,
    fit_train_only_scaler,
    haversine_matrix,
    load_static,
    standardize_static,
)
from evaluate_full72_oof_outer_ensemble import (
    fit_epoch_weights,
    load_all_oof,
    select_method_by_sixfold_oof,
)
from sanity_check import audit_dataset, smoke_forward
from v2_training import (
    build_model, build_v2_context, event_pos_weight_from_dataset, smoke_v2,
    train_epoch_v2, validate_epoch_v2, v2_config_payload, v2_enabled,
)
from train_formal import (
    DeviceFeatureBuilder,
    amp_dtype_for,
    cpu_state_dict,
    make_grad_scaler,
    make_index_loader,
    make_loader,
    make_optimizer,
    make_station_sample_weights,
    make_vectorized_loader,
    regression_metrics,
    resolve_target,
    seed_all,
    train_epoch,
    validate_epoch,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def code_revision() -> str | None:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_provenance(static_columns: list[str], model_version: str) -> dict:
    return {
        'code_revision': code_revision(),
        'model_version': model_version,
        'train_period': [CFG.train_start, CFG.train_end],
        'test_period': [CFG.test_start, CFG.test_end],
        'history_hours': int(CFG.history_hours),
        'dynamic_channels': list(CFG.dynamic_items),
        'dynamic_channel_count': int(CFG.n_dynamic_channels),
        'raw_wind_enters_model': False,
        'target_dynamic_history_used': False,
        'static_feature_count': int(len(static_columns)),
        'static_pca': False,
        'static_columns': list(static_columns),
        'static_sha256': file_sha256(CFG.static_path),
        'cluster_sha256': file_sha256(CFG.cluster_path),
        'geometry': ['log1p(distance_km)', 'sin(donor_to_target_bearing)', 'cos(donor_to_target_bearing)'],
        'missing_handling': {
            'dynamic_imputation': False,
            'finite_placeholder_after_scaling': 0.0,
            'binary_mask_same_shape': True,
            'drop_sample_only_when_target_pm25_missing': True,
        },
        'wind': {
            'input_direction': 'meteorological FROM',
            'transport_direction': 'TOWARD=(FROM+180)%360',
            'bearing': 'donor->target',
            'along_positive': 'toward target',
        },
        'event_threshold': float(os.environ.get('DL_TCN_EVENT_THRESHOLD', '35.0')),
        'absolute_error_threshold': float(os.environ.get('DL_TCN_ABS_ERROR_THRESHOLD', '20.0')),
    }


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float('nan')


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    threshold = float(os.environ.get('DL_TCN_EVENT_THRESHOLD', '35.0'))
    absolute_error_threshold = float(os.environ.get('DL_TCN_ABS_ERROR_THRESHOLD', '20.0'))
    observed = y_true >= threshold
    predicted = y_pred >= threshold
    tp = int(np.sum(observed & predicted))
    fp = int(np.sum(~observed & predicted))
    fn = int(np.sum(observed & ~predicted))
    tn = int(np.sum(~observed & ~predicted))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        'event_threshold': threshold,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': safe_divide(2 * precision * recall, precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) else float('nan'),
        'specificity': safe_divide(tn, tn + fp),
        'accuracy': safe_divide(tp + tn, len(y_true)),
        'absolute_error_threshold': absolute_error_threshold,
        'false_high_count': int(np.sum((y_pred - y_true) >= absolute_error_threshold)),
        'false_low_count': int(np.sum((y_true - y_pred) >= absolute_error_threshold)),
        'false_high_rate': float(np.mean((y_pred - y_true) >= absolute_error_threshold)),
        'false_low_rate': float(np.mean((y_true - y_pred) >= absolute_error_threshold)),
    }


def rng_payload() -> dict:
    result = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(payload: dict) -> None:
    random.setstate(payload['python'])
    np.random.set_state(payload['numpy'])
    # Loading a resumable checkpoint with ``map_location=device`` moves these
    # serialized RNG tensors to CUDA as well.  PyTorch's RNG restore APIs
    # require contiguous CPU uint8 tensors.
    cpu_state = payload['torch'].detach().to(
        device='cpu', dtype=torch.uint8
    ).contiguous()
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and 'cuda' in payload:
        cuda_states = [
            state.detach().to(device='cpu', dtype=torch.uint8).contiguous()
            for state in payload['cuda']
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_final_outputs(
    output_root: Path,
    result: dict,
    timestamp_ns: np.ndarray,
    truth: np.ndarray,
    ensemble_prediction: np.ndarray,
    oracle_prediction: np.ndarray,
    epoch_predictions: np.ndarray,
) -> list[str]:
    siteid = str(result['siteid'])
    summary_dir = output_root / 'station_summaries'
    prediction_dir = output_root / 'station_predictions'
    summary_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f'site_{siteid}.json'
    prediction_path = prediction_dir / f'site_{siteid}.npz'
    summary_tmp = summary_path.with_suffix('.json.tmp')
    summary_tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(summary_tmp, summary_path)
    prediction_tmp = prediction_path.with_suffix('.tmp.npz')
    np.savez_compressed(
        prediction_tmp,
        timestamp_ns=timestamp_ns.astype('int64'),
        y_true=truth.astype('float32'),
        ensemble_prediction=ensemble_prediction.astype('float32'),
        oracle_prediction=oracle_prediction.astype('float32'),
        epoch_predictions=epoch_predictions.astype('float32'),
    )
    os.replace(prediction_tmp, prediction_path)
    return [str(summary_path), str(prediction_path)]


def main() -> None:
    seed_all(CFG.seed)
    runtime = apply_runtime_profile(CFG)
    device = CFG.device
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = CFG.enable_tf32
        torch.backends.cudnn.allow_tf32 = CFG.enable_tf32
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    locked_raw = os.environ.get('DL_TCN_LOCKED_ENSEMBLE_PATH', '').strip()
    locked = None
    if locked_raw:
        locked_path = Path(locked_raw)
        locked = json.loads(locked_path.read_text(encoding='utf-8'))
        epoch_weights = np.asarray(locked['epoch_weights'], dtype=float)
        method = dict(locked['selected_method'])
        method_rows = list(locked.get('candidate_scores', []))
        if len(epoch_weights) != 15 or np.any(epoch_weights < 0):
            raise RuntimeError('locked ensemble必須是15個非負權重')
        if not np.isclose(epoch_weights.sum(), 1.0, atol=1e-6):
            raise RuntimeError('locked ensemble權重總和不是1')
        lock_protocol = str(locked['protocol'])
        lock_metadata = {
            key: locked[key]
            for key in ('outer_group', 'outer_siteids', 'selector_known_stations')
            if key in locked
        }
    else:
        crossfit_root = Path(os.environ.get(
            'DL_TCN_CROSSFIT_ROOT',
            str(CFG.formal_output_root / 'crossfit_target_conditioned_snapshots'),
        ))
        oof_base, oof_matrix = load_all_oof(crossfit_root)
        method, method_rows = select_method_by_sixfold_oof(oof_base, oof_matrix)
        oof_stations = np.unique(oof_base.station_index.to_numpy(int))
        epoch_weights = fit_epoch_weights(
            oof_base, oof_matrix, oof_stations,
            method['family'], method['parameter'],
        )
        lock_protocol = '72-station OOF locks epoch ensemble before 72-station refit and outer truth'
        lock_metadata = {}
    lock = {
        'protocol': lock_protocol,
        'selection_used_outer_truth': False,
        'selected_method': method,
        'candidate_scores': sorted(method_rows, key=lambda row: row['sixfold_oof_macro_rmse']),
        'epoch_weights': [
            {'epoch': epoch, 'weight': float(weight)}
            for epoch, weight in enumerate(epoch_weights, start=1) if weight > 1e-8
        ],
    }
    print('\nLOCKED EPOCH ENSEMBLE BEFORE REFIT/OUTER TRUTH')
    print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)

    static, _, static_cols = load_static()
    outer = resolve_target(static, CFG.target_site)
    if locked is not None:
        allowed = {str(x) for x in locked.get('outer_siteids', [])}
        outer_siteid = str(static.loc[outer, 'siteid'])
        if allowed and outer_siteid not in allowed:
            raise RuntimeError(
                f'locked ensemble不屬於outer site {outer_siteid}; allowed={sorted(allowed)}'
            )
    known = np.asarray([idx for idx in range(len(static)) if idx != outer], dtype=int)
    if len(known) != 72:
        raise RuntimeError(f'outer排除後不是72站: {len(known)}')
    cube, timestamps = build_or_load_hourly_cube(static)
    distance = haversine_matrix(static.longitude, static.latitude)
    scaler = fit_train_only_scaler(cube, timestamps, static, static_cols, known)
    static_scaled = standardize_static(static, static_cols, scaler)
    v2_adapter = None
    background_audit = None
    event_pos_weight = None
    if v2_enabled():
        v2_adapter, background_audit = build_v2_context(
            cube, timestamps, static_scaled, distance, known, scaler, static, device
        )
    train_ds = ColdStartStationDataset(
        known, known, CFG.train_start, CFG.train_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    outer_ds = ColdStartStationDataset(
        [outer], known, CFG.test_start, CFG.test_end,
        cube, timestamps, static, static_scaled, distance, scaler,
    )
    if audit_dataset(train_ds, max_samples=64)['sampled_donor_counts'] != [71]:
        raise RuntimeError('72-refit training donors不是71')
    if audit_dataset(outer_ds, max_samples=64)['sampled_donor_counts'] != [72]:
        raise RuntimeError('72-refit outer donors不是72')

    feature_builder = None
    if device.type == 'cuda':
        max_time = int(max(train_ds.row_times.max(), outer_ds.row_times.max()))
        feature_builder = DeviceFeatureBuilder(
            known, cube, max_time, timestamps, static, static_scaled,
            distance, scaler, outer, device,
        )
        train_loader = make_index_loader(train_ds, True)
        outer_loader = make_index_loader(outer_ds, False)
    else:
        train_loader = make_vectorized_loader(
            train_ds, known, cube, timestamps, static, static_scaled,
            distance, scaler, True,
        )
        outer_loader = make_vectorized_loader(
            outer_ds, known, cube, timestamps, static, static_scaled,
            distance, scaler, False,
        )

    base_model = build_model(static_dim=len(static_cols)).to(device)
    if v2_enabled():
        smoke_batch = next(iter(make_loader(train_ds, False, CFG.smoke_batch_size)))
        smoke = smoke_v2(base_model, smoke_batch, device, v2_adapter)
        event_pos_weight = event_pos_weight_from_dataset(train_ds, cube, device)
    else:
        smoke = smoke_forward(
            base_model, make_loader(train_ds, False, CFG.smoke_batch_size), device
        )
    base_model.zero_grad(set_to_none=True)
    optimizer, fused = make_optimizer(base_model, device)
    model = base_model
    if CFG.compile_mode != 'off':
        if not hasattr(torch, 'compile'):
            raise RuntimeError('目前PyTorch沒有torch.compile，請設DL_TCN_COMPILE_MODE=off')
        model = torch.compile(
            # Match the A100 configuration that was benchmarked successfully.
            # Dynamic symbolic shapes can fail inside TorchInductor for this
            # attention graph; fixed-shape compilation preserves all inputs.
            base_model, mode=CFG.compile_mode, fullgraph=False, dynamic=False
        )
    amp_dtype = amp_dtype_for(device)
    grad_scaler = make_grad_scaler(amp_dtype == torch.float16)
    station_weights = make_station_sample_weights(train_ds, len(static), device)
    provenance = run_provenance(
        static_cols, 'v2' if v2_enabled() else 'v1'
    )

    resume_raw = os.environ.get('DL_TCN_REFIT_RESUME_PATH', '').strip()
    resume_path = Path(resume_raw) if resume_raw else None

    print(json.dumps({
        'runtime_profile': runtime,
        'train_samples': len(train_ds),
        'outer_truth_timestamps': len(outer_ds),
        'training_targets': 72,
        'training_donors_per_sample': 71,
        'outer_donors': 72,
        'model_parameters': sum(p.numel() for p in base_model.parameters()),
        'model_version': 'v2' if v2_enabled() else 'v1',
        'provenance': provenance,
        'background': background_audit,
        'smoke': smoke,
        'fused_adamw': fused,
        'files_written': 0,
    }, ensure_ascii=False, indent=2), flush=True)

    outer_base = None
    outer_columns: list[np.ndarray] = []
    history: list[dict] = []
    start_epoch = 1
    peak_vram = 0.0
    if resume_path is not None and resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get('model_version', 'v1') != ('v2' if v2_enabled() else 'v1'):
            raise RuntimeError('72-refit resume model version不一致')
        if v2_enabled() and resume.get('v2_config') != v2_config_payload():
            raise RuntimeError('72-refit resume V2設定不一致，請使用新的output root')
        if int(resume['outer']) != outer or not np.array_equal(resume['known'], known):
            raise RuntimeError('72-refit resume target/protocol不一致')
        base_model.load_state_dict(resume['model_state_dict'])
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        grad_scaler.load_state_dict(resume['grad_scaler_state_dict'])
        restore_rng(resume['rng_state'])
        start_epoch = int(resume['epoch']) + 1
        history = list(resume['history'])
        outer_columns = [np.asarray(x, dtype=float) for x in resume['outer_columns']]
        peak_vram = float(resume.get('peak_vram_mb', 0.0))
        outer_base = pd.DataFrame({
            'station_index': np.asarray(resume['outer_station_index'], dtype=int),
            'timestamp': pd.to_datetime(np.asarray(resume['outer_timestamp_ns'], dtype='int64')),
            'y_true': np.asarray(resume['outer_truth'], dtype=float),
        })
        print(f'72-refit resume from epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, 16):
        epoch_started = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if v2_enabled():
            train_loss = train_epoch_v2(
                model, train_loader, optimizer, grad_scaler, device, epoch,
                v2_adapter, event_pos_weight, feature_builder, station_weights,
            )
        else:
            train_loss = train_epoch(
                model, train_loader, optimizer, grad_scaler, device, epoch,
                feature_builder, station_weights,
            )
        # Outer predictions are collected for the already-locked snapshot
        # ensemble. Outer metrics are deliberately not printed or selected here.
        if v2_enabled():
            _, _, current = validate_epoch_v2(
                model, outer_loader, device, static, timestamps, epoch,
                v2_adapter, feature_builder,
            )
        else:
            _, _, current = validate_epoch(
                model, outer_loader, device, static, timestamps, epoch, feature_builder
            )
        if outer_base is None:
            outer_base = current[['station_index', 'timestamp', 'y_true']].copy()
        else:
            if not np.array_equal(
                pd.to_datetime(outer_base.timestamp).astype('int64').to_numpy(),
                pd.to_datetime(current.timestamp).astype('int64').to_numpy(),
            ):
                raise RuntimeError('72-refit outer snapshot timestamps不一致')
            if not np.allclose(outer_base.y_true, current.y_true, rtol=0, atol=1e-6):
                raise RuntimeError('72-refit outer snapshot truth不一致')
        outer_columns.append(current.y_pred.to_numpy(float))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            peak_vram = max(
                peak_vram,
                torch.cuda.max_memory_allocated(device) / 1024**2,
            )
        row = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'runtime_seconds': time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if resume_path is not None:
            atomic_torch_save({
                'outer': outer,
                'known': known,
                'model_version': 'v2' if v2_enabled() else 'v1',
                'v2_config': v2_config_payload() if v2_enabled() else None,
                'epoch': epoch,
                'model_state_dict': cpu_state_dict(base_model),
                'optimizer_state_dict': optimizer.state_dict(),
                'grad_scaler_state_dict': grad_scaler.state_dict(),
                'rng_state': rng_payload(),
                'history': history,
                'outer_columns': outer_columns,
                'outer_station_index': outer_base.station_index.to_numpy('int16'),
                'outer_timestamp_ns': pd.to_datetime(outer_base.timestamp).astype('int64').to_numpy(),
                'outer_truth': outer_base.y_true.to_numpy('float32'),
                'peak_vram_mb': peak_vram,
            }, resume_path)

    assert outer_base is not None
    outer_matrix = np.stack(outer_columns, axis=1)
    deployed = outer_matrix @ epoch_weights
    truth = outer_base.y_true.to_numpy(float)
    epoch_curve = []
    for epoch, prediction in enumerate(outer_columns, start=1):
        epoch_curve.append({
            'epoch': epoch,
            **regression_metrics(truth, prediction),
        })
    oracle_epoch = int(min(epoch_curve, key=lambda row: row['rmse'])['epoch'])
    oracle_prediction = outer_matrix[:, oracle_epoch - 1]
    selected_regression = regression_metrics(truth, deployed)
    oracle_regression = regression_metrics(truth, oracle_prediction)
    result = {
        'target_station_index': int(outer),
        'siteid': str(static.loc[outer, 'siteid']),
        'sitename': str(static.loc[outer, 'sitename']),
        'selection_used_outer_truth': False,
        'model_version': 'v2' if v2_enabled() else 'v1',
        'provenance': provenance,
        'uses_target_meteorology': False if v2_enabled() else None,
        'v2_config': v2_config_payload() if v2_enabled() else None,
        'background': background_audit,
        'training_targets': 72,
        'training_donors_per_sample': 71,
        'outer_donors': 72,
        'epochs': 15,
        'epoch_selection': {
            'protocol': lock_protocol,
            **lock_metadata,
            'method': method,
            'weights': [float(x) for x in epoch_weights],
        },
        'truth_timestamps': int(len(truth)),
        'ensemble': {
            'regression': selected_regression,
            'events': event_metrics(truth, deployed),
        },
        'oracle': {
            'uses_outer_truth': True,
            'epoch': oracle_epoch,
            'regression': oracle_regression,
            'events': event_metrics(truth, oracle_prediction),
        },
        'epoch_curve': epoch_curve,
        'training_runtime_seconds': float(sum(row['runtime_seconds'] for row in history)),
        'peak_vram_mb': peak_vram,
        'files_written': 0,
    }
    output_raw = os.environ.get('DL_TCN_REFIT_OUTPUT_ROOT', '').strip()
    if output_raw:
        paths = write_final_outputs(
            Path(output_raw), result,
            pd.to_datetime(outer_base.timestamp).astype('int64').to_numpy(),
            truth, deployed, oracle_prediction, outer_matrix,
        )
        result['files_written'] = len(paths)
        result['output_files'] = paths
        # Rewrite the summary once so it also records its final output paths.
        summary_path = Path(paths[0])
        summary_tmp = summary_path.with_suffix('.json.tmp')
        summary_tmp.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        os.replace(summary_tmp, summary_path)
    if resume_path is not None and resume_path.exists():
        resume_path.unlink()
    print('\nREFIT-72 OUTER EPOCH-ENSEMBLE RESULT')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import CFG
from model_v2 import TCNTargetCrossAttentionV2
from train_formal import amp_dtype_for, move_batch, regression_metrics


def v2_enabled() -> bool:
    return os.environ.get("DL_TCN_MODEL_VERSION", "v1").strip().lower() == "v2"


def v2_config_payload() -> dict:
    return {
        "model_version": "v2",
        "training_revision": "full_donor_nested_background_bounded_relation_v4",
        "learning_rate": CFG.learning_rate,
        "weight_decay": CFG.weight_decay,
        "dropout": CFG.dropout,
        "gradient_clip_norm": CFG.gradient_clip_norm,
        "epochs": CFG.max_epochs,
        "background": "target-excluded selection and fit; exact inner ridge LOO",
        "background_scale": 10.0,
        "background_center": 0.0,
        "background_penalties": [0.1, 1.0, 10.0, 100.0],
        "target_meteorology": False,
        "pm25_anomaly_channel": True,
        "static_difference_features": 49,
        "physics_attention": "zero-init signed tanh correction; cap=1; source-pair RMS scales",
        "virtual_target": False,
        "regression_loss": "station-balanced ordinary MSE",
        "donor_mask_probability": 0.0,
        "event_loss_weight": 0.0,
        "virtual_loss_weight": 0.0,
    }


def build_model(static_dim: int):
    if v2_enabled():
        return TCNTargetCrossAttentionV2(static_dim=static_dim)
    from model import TCNTargetCrossAttention
    return TCNTargetCrossAttention(static_dim=static_dim)


@dataclass
class BackgroundFit:
    predicted: np.ndarray
    observed: np.ndarray
    center: float
    scale: float
    ridge_lambda: float
    static_weight: float
    loo_rmse: float


def fit_background(cube, timestamps, static_scaled, distance, source_indices) -> BackgroundFit:
    """No persistent cache: each context owns its source-only lookup table."""
    from background_crossfit import crossfit_background
    source = np.asarray(source_indices, dtype=int)
    tidx = np.flatnonzero(
        (timestamps >= pd.Timestamp(CFG.train_start))
        & (timestamps <= pd.Timestamp(CFG.train_end))
    )
    if not len(tidx):
        raise ValueError("No background training timestamps")
    pm = CFG.aq_cube_items.index("PM2.5")
    observed = np.full(len(static_scaled), np.nan, dtype=np.float64)
    observed[source] = np.nanmean(
        np.asarray(cube[np.ix_(tidx, source, [pm])])[..., 0], axis=0,
    )
    predicted, penalty, alpha, _ = crossfit_background(
        static_scaled, observed, distance, source,
    )
    rmse = float(np.sqrt(np.mean((predicted[source] - observed[source])**2)))
    return BackgroundFit(
        predicted.astype("float32"), observed.astype("float32"),
        0.0, 10.0, penalty, alpha, rmse,
    )


def relation_scales(static_scaled, distance, source, scaler):
    """Fit distance/static RMS to off-diagonal TRAIN source pairs only."""
    source = np.asarray(source, dtype=int)
    off = ~np.eye(len(source), dtype=bool)
    logd = np.log1p(np.asarray(distance)[np.ix_(source, source)] / 1000)
    xs = np.asarray(static_scaled, dtype=np.float64)[source]
    differences = np.mean((xs[:, None] - xs[None, :])**2, axis=-1)
    return np.asarray([
        max(float(np.sqrt(np.mean(logd[off]**2))), 1e-6),
        max(float(scaler.dynamic_std[-2]), 1e-6),
        max(float(scaler.dynamic_std[-1]), 1e-6),
        max(float(np.sqrt(np.mean(differences[off]**2))), 1e-6),
    ], dtype="float32")


class V2BatchAdapter:
    """Adds only source-fitted context. It never reads target meteorology."""

    def __init__(self, background: BackgroundFit, scaler, static, device, prior_scales=None):
        self.device = device
        self.prior_scales = torch.as_tensor(
            np.ones(4, dtype="float32") if prior_scales is None else prior_scales,
            dtype=torch.float32, device=device,
        )
        self.predicted = torch.as_tensor(background.predicted, device=device)
        observed = background.observed.copy()
        observed[~np.isfinite(observed)] = background.center
        self.observed = torch.as_tensor(observed, device=device)
        self.background_center = background.center
        self.background_scale = background.scale
        self.pm_mean = float(scaler.dynamic_mean[CFG.raw_dynamic_items.index("PM2.5")])
        self.pm_std = float(scaler.dynamic_std[CFG.raw_dynamic_items.index("PM2.5")])
        self.wind_mean = torch.as_tensor(scaler.dynamic_mean[-2:], device=device)
        self.wind_std = torch.as_tensor(scaler.dynamic_std[-2:], device=device)
        self.lon = torch.as_tensor(static.longitude.to_numpy("float32"), device=device)
        self.lat = torch.as_tensor(static.latitude.to_numpy("float32"), device=device)
        self.pm_channel = CFG.raw_dynamic_items.index("PM2.5")

    def prepare(self, batch: dict) -> dict:
        batch = dict(batch)
        values = batch["values"].clone()
        donors = batch["donor_indices"]
        safe_donors = donors.clamp_min(0)
        donor_climatology = self.observed[safe_donors]
        anomaly_shift = (donor_climatology - self.pm_mean) / self.pm_std
        pm_mask = batch["mask"][..., self.pm_channel]
        values[..., self.pm_channel] = torch.where(
            pm_mask > 0,
            values[..., self.pm_channel] - anomaly_shift[:, :, None],
            torch.zeros_like(values[..., self.pm_channel]),
        )
        target_raw = self.predicted[batch["target_idx"]]
        batch["values"] = values
        batch["relation_scales"] = self.prior_scales
        batch["physical_wind"] = torch.where(
            batch["mask"][:, :, -1, 9:11] > 0,
            values[:, :, -1, 9:11] * self.wind_std + self.wind_mean,
            0.0,
        )
        batch["donor_background"] = (donor_climatology - self.background_center) / self.background_scale
        batch["target_background_raw"] = target_raw
        batch["target_background"] = (target_raw - self.background_center) / self.background_scale
        return batch



def build_v2_context(cube, timestamps, static_scaled, distance, source_indices, scaler, static, device):
    started = time.perf_counter()
    fit = fit_background(cube, timestamps, static_scaled, distance, source_indices)
    scales = relation_scales(static_scaled, distance, source_indices, scaler)
    elapsed = time.perf_counter() - started
    print(f"V4 background precompute: {elapsed:.3f}s; cached in context memory", flush=True)
    return V2BatchAdapter(fit, scaler, static, device, scales), {
        "method": "nested target-excluded selection+fit; train-period labels only",
        "precompute_seconds": elapsed,
        "relation_scales": scales.tolist(),
        "source_indices": np.asarray(source_indices, dtype=int).tolist(),
        "training_range": [CFG.train_start, CFG.train_end],
        "ridge_lambda": fit.ridge_lambda,
        "static_weight": fit.static_weight,
        "source_loo_rmse": fit.loo_rmse,
        "uses_target_pm25": False,
        "uses_target_meteorology": False,
    }


def _forward(model, batch, need_attention=False):
    return model(
        batch["values"], batch["mask"], batch["donor_static"], batch["geometry"],
        batch["donor_padding_mask"], batch["target_static"], batch["time_features"],
        batch["donor_background"], batch["target_background"],
        batch["target_background_raw"], need_attention, batch["physical_wind"], batch["relation_scales"],
    )


def _v2_loss(prediction, auxiliary, label, sample_weight, event_pos_weight):
    # Weights are fixed N/(number_of_stations * N_i), not batch-normalized.
    return torch.mean(sample_weight * (prediction.float() - label.float()).square())


def event_pos_weight_from_dataset(dataset, cube, device):
    # Compatibility argument only; V4 has no event classification loss.
    return None


def train_epoch_v2(model, loader, optimizer, grad_scaler, device, epoch, adapter,
                   event_pos_weight, feature_builder=None, station_sample_weights=None):
    model.train(); amp_dtype = amp_dtype_for(device); amp_enabled = amp_dtype is not None
    loss_sum = 0.0; sample_count = 0; started = time.perf_counter()
    for batch_number, raw_batch in enumerate(loader, 1):
        batch = feature_builder(raw_batch) if feature_builder is not None else move_batch(raw_batch, device)
        batch = adapter.prepare(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction, auxiliary = _forward(model, batch)
            weights = station_sample_weights[batch["target_idx"]]
            loss = _v2_loss(prediction, auxiliary, batch["label"], weights, event_pos_weight)
        if not torch.isfinite(loss): raise RuntimeError(f"V2 epoch {epoch} batch {batch_number}: loss非finite")
        grad_scaler.scale(loss).backward(); grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip_norm)
        if not torch.isfinite(grad_norm): raise RuntimeError("V2 gradient norm非finite")
        grad_scaler.step(optimizer); grad_scaler.update()
        n = int(batch["label"].numel()); loss_sum += float(loss.detach().cpu()) * n; sample_count += n
        if CFG.progress_every_batches > 0 and batch_number % CFG.progress_every_batches == 0:
            print(f"  V2 epoch {epoch} train {batch_number:,}/{len(loader):,} | {sample_count/max(time.perf_counter()-started,1e-9):,.1f} samples/s", flush=True)
    if sample_count != len(loader.dataset): raise RuntimeError("V2 training coverage錯誤")
    return loss_sum / sample_count


@torch.no_grad()
def validate_epoch_v2(model, loader, device, static, timestamps, epoch, adapter, feature_builder=None):
    model.eval(); amp_dtype = amp_dtype_for(device); amp_enabled = amp_dtype is not None
    ys=[]; ps=[]; ss=[]; ts=[]
    for raw_batch in loader:
        batch = feature_builder(raw_batch) if feature_builder is not None else move_batch(raw_batch, device)
        batch = adapter.prepare(batch)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction, _ = _forward(model, batch)
        if not torch.isfinite(prediction).all(): raise RuntimeError("V2 validation prediction非finite")
        ys.append(batch["label"].float().cpu().numpy()); ps.append(prediction.float().cpu().numpy())
        ss.append(batch["target_idx"].cpu().numpy()); ts.append(batch["time_idx"].cpu().numpy())
    y,p,s,ti = map(np.concatenate, (ys,ps,ss,ts))
    overall = regression_metrics(y,p); rows=[]
    for station in sorted(np.unique(s)):
        keep=s==station
        rows.append({"station_index":int(station),"siteid":str(static.loc[station,"siteid"]),
                     "sitename":str(static.loc[station,"sitename"]),"n":int(keep.sum()),
                     **regression_metrics(y[keep],p[keep])})
    overall["macro_station_rmse"] = float(np.mean([x["rmse"] for x in rows]))
    frame = pd.DataFrame({"station_index":s.astype(int),"timestamp":pd.DatetimeIndex(timestamps[ti]),
                          "y_true":y.astype("float32"),"y_pred":p.astype("float32")})
    return overall, pd.DataFrame(rows), frame


def smoke_v2(model, raw_batch, device, adapter):
    batch = move_batch(raw_batch, device); batch = adapter.prepare(batch)
    model.train(); model.zero_grad(set_to_none=True)
    prediction, auxiliary = _forward(model, batch, True)
    loss = _v2_loss(prediction, auxiliary, batch["label"], torch.ones_like(batch["label"]), None)
    loss.backward()
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss): raise RuntimeError("V2 smoke非finite")
    required = ("tcn.", "donor_projection.", "query_projection.", "cross_attention.", "shared_head.", "residual_head.")
    for prefix in required:
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix)]
        if not grads or all(g is None or float(g.abs().sum()) == 0 for g in grads):
            raise RuntimeError(f"V4 smoke: {prefix}沒有gradient")
        if any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"V4 smoke: {prefix} nonfinite gradient")
    return {"prediction_shape":list(prediction.shape),"loss":float(loss.detach().cpu()),
            "parameters":sum(p.numel() for p in model.parameters()),"attention_finite":bool(torch.isfinite(auxiliary["attention"]).all())}

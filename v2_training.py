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
        "background": "source-only static-ridge+IDW; source target LOO",
        "target_meteorology": False,
        "pm25_anomaly_channel": True,
        "static_difference_features": 49,
        "physics_attention": "distance+along_wind+absolute_cross_wind+static_similarity",
        "virtual_target": "source station-pair spatial mixup",
        "huber_beta": float(os.environ.get("DL_TCN_V2_HUBER_BETA", "5.0")),
        "tail_start": float(os.environ.get("DL_TCN_V2_TAIL_START", "25.0")),
        "event_threshold": float(os.environ.get("DL_TCN_V2_EVENT_THRESHOLD", "35.0")),
        "tail_max_weight": float(os.environ.get("DL_TCN_V2_TAIL_MAX_WEIGHT", "3.0")),
        "event_loss_weight": float(os.environ.get("DL_TCN_V2_EVENT_LOSS_WEIGHT", "0.1")),
        "virtual_loss_weight": float(os.environ.get("DL_TCN_V2_VIRTUAL_LOSS_WEIGHT", "0.1")),
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


def _ridge_predict(x_train, y_train, x_query, penalty: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    query = np.column_stack([np.ones(len(x_query)), x_query])
    regularizer = np.eye(design.shape[1]); regularizer[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty * regularizer, design.T @ y_train)
    return query @ coef


def fit_background(cube, timestamps, static_scaled, distance, source_indices) -> BackgroundFit:
    """Fit using source-station training-period PM2.5 only; source predictions are LOO."""
    source = np.asarray(source_indices, dtype=int)
    tidx = np.flatnonzero(
        (timestamps >= pd.Timestamp(CFG.train_start))
        & (timestamps <= pd.Timestamp(CFG.train_end))
    )
    pm = CFG.aq_cube_items.index("PM2.5")
    # Deliberately index source stations only. Held-out PM2.5 is not even read.
    observed = np.full(len(static_scaled), np.nan, dtype="float64")
    observed[source] = np.nanmean(
        np.asarray(cube[np.ix_(tidx, source, [pm])])[..., 0], axis=0
    )
    if not np.isfinite(observed[source]).all():
        raise RuntimeError("V2 background: source station存在無法估計的全年PM2.5平均")
    x = np.asarray(static_scaled, dtype="float64")
    y = observed[source].astype("float64")
    lambdas = (0.1, 1.0, 10.0, 100.0)
    best = None
    for penalty in lambdas:
        static_loo = np.empty(len(source), dtype="float64")
        idw_loo = np.empty(len(source), dtype="float64")
        for pos, station in enumerate(source):
            keep = np.arange(len(source)) != pos
            static_loo[pos] = _ridge_predict(x[source[keep]], y[keep], x[[station]], penalty)[0]
            d = np.maximum(distance[station, source[keep]] / 1000.0, 1e-3)
            w = 1.0 / np.square(d)
            idw_loo[pos] = np.sum(w * y[keep]) / np.sum(w)
        delta = static_loo - idw_loo
        denom = float(delta @ delta)
        alpha = float(np.clip(((y - idw_loo) @ delta) / denom, 0.0, 1.0)) if denom > 0 else 0.0
        blended = alpha * static_loo + (1.0 - alpha) * idw_loo
        rmse = float(np.sqrt(np.mean(np.square(blended - y))))
        if best is None or rmse < best[0]:
            best = (rmse, penalty, alpha, blended)
    assert best is not None
    rmse, penalty, alpha, source_loo = best
    static_all = _ridge_predict(x[source], y, x, penalty)
    idw_all = np.empty(len(x), dtype="float64")
    for station in range(len(x)):
        available = source[source != station]
        d = np.maximum(distance[station, available] / 1000.0, 1e-3)
        w = 1.0 / np.square(d)
        idw_all[station] = np.sum(w * observed[available]) / np.sum(w)
    predicted = alpha * static_all + (1.0 - alpha) * idw_all
    predicted[source] = source_loo
    center = float(y.mean()); scale = float(y.std())
    if not np.isfinite(scale) or scale < 1e-6: scale = 1.0
    return BackgroundFit(
        predicted.astype("float32"), observed.astype("float32"), center, scale,
        float(penalty), float(alpha), float(rmse),
    )


class V2BatchAdapter:
    """Adds only source-fitted context. It never reads target meteorology."""

    def __init__(self, background: BackgroundFit, scaler, static, device):
        self.device = device
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
        batch["donor_background"] = (donor_climatology - self.background_center) / self.background_scale
        batch["target_background_raw"] = target_raw
        batch["target_background"] = (target_raw - self.background_center) / self.background_scale
        return batch

    @staticmethod
    def _bearing_and_distance(donor_lon, donor_lat, target_lon, target_lat):
        rad = torch.pi / 180.0
        lon1, lat1 = donor_lon * rad, donor_lat * rad
        lon2, lat2 = target_lon[:, None] * rad, target_lat[:, None] * rad
        dlon, dlat = lon2 - lon1, lat2 - lat1
        a = torch.sin(dlat / 2).square() + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2).square()
        distance_km = 2 * 6371.0 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        x = torch.sin(dlon) * torch.cos(lat2)
        y = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
        bearing = torch.atan2(x, y)
        return bearing, distance_km

    def virtual_batch(self, batch: dict) -> tuple[dict, torch.Tensor] | tuple[None, None]:
        # A station-pair spatial interpolation regularizer, deliberately low weight.
        current_pm_ok = batch["mask"][:, :, -1, self.pm_channel] > 0
        valid = current_pm_ok.any(dim=1)
        if not valid.any(): return None, None
        # Keep the complete B dimension so torch.compile sees the same shape on
        # every batch. Invalid virtual rows receive zero loss weight below.
        rows = torch.arange(len(valid), device=self.device)
        ok = current_pm_ok
        selector = (batch["time_idx"] + batch["target_idx"]) % ok.shape[1]
        offsets = torch.arange(ok.shape[1], device=self.device)[None, :]
        candidates = (selector[:, None] + offsets) % ok.shape[1]
        first = torch.argmax(torch.gather(ok, 1, candidates).to(torch.int64), dim=1)
        partner_pos = candidates[torch.arange(len(rows), device=self.device), first]
        alpha = 0.35 + 0.30 * torch.remainder(
            batch["time_idx"] * 1103515245 + batch["target_idx"] * 12345, 997
        ).float() / 996.0

        virtual = {k: (v[rows].clone() if torch.is_tensor(v) and v.ndim and v.shape[0] == len(batch["label"]) else v)
                   for k, v in batch.items()}
        partner_static = virtual["donor_static"][torch.arange(len(rows), device=self.device), partner_pos]
        virtual["target_static"] = alpha[:, None] * virtual["target_static"] + (1-alpha)[:, None] * partner_static
        partner_bg = virtual["donor_background"][torch.arange(len(rows), device=self.device), partner_pos]
        virtual["target_background"] = alpha * virtual["target_background"] + (1-alpha) * partner_bg
        virtual["target_background_raw"] = self.background_center + self.background_scale * virtual["target_background"]
        partner_pm_anom = virtual["values"][torch.arange(len(rows), device=self.device), partner_pos, -1, self.pm_channel]
        partner_raw = partner_pm_anom * self.pm_std + self.observed[virtual["donor_indices"][torch.arange(len(rows), device=self.device), partner_pos]]
        virtual["label"] = alpha * virtual["label"] + (1-alpha) * partner_raw

        donors = virtual["donor_indices"].clamp_min(0)
        target_idx = batch["target_idx"]
        partner_idx = virtual["donor_indices"][torch.arange(len(rows), device=self.device), partner_pos]
        vlon = alpha * self.lon[target_idx] + (1-alpha) * self.lon[partner_idx]
        vlat = alpha * self.lat[target_idx] + (1-alpha) * self.lat[partner_idx]
        bearing_new, distance_new = self._bearing_and_distance(self.lon[donors], self.lat[donors], vlon, vlat)
        sin_new, cos_new = torch.sin(bearing_new), torch.cos(bearing_new)
        virtual["geometry"] = torch.stack([torch.log1p(distance_new), sin_new, cos_new], dim=-1)

        # Rotate the already observed wind vector from the old target frame into
        # the virtual-target frame; no target-side weather is introduced.
        old_sin = batch["geometry"][rows, :, 1]
        old_cos = batch["geometry"][rows, :, 2]
        along = virtual["values"][..., 9] * self.wind_std[0] + self.wind_mean[0]
        cross = virtual["values"][..., 10] * self.wind_std[1] + self.wind_mean[1]
        east = along * old_sin[:, :, None] + cross * old_cos[:, :, None]
        north = along * old_cos[:, :, None] - cross * old_sin[:, :, None]
        new_along = east * sin_new[:, :, None] + north * cos_new[:, :, None]
        new_cross = east * cos_new[:, :, None] - north * sin_new[:, :, None]
        wind_mask = virtual["mask"][..., 9:11]
        virtual["values"][..., 9] = torch.where(wind_mask[..., 0] > 0, (new_along-self.wind_mean[0])/self.wind_std[0], 0)
        virtual["values"][..., 10] = torch.where(wind_mask[..., 1] > 0, (new_cross-self.wind_mean[1])/self.wind_std[1], 0)
        return virtual, valid.to(torch.float32)


def build_v2_context(cube, timestamps, static_scaled, distance, source_indices, scaler, static, device):
    fit = fit_background(cube, timestamps, static_scaled, distance, source_indices)
    return V2BatchAdapter(fit, scaler, static, device), {
        "method": "train-only blended static-ridge + geographic-IDW; source targets use LOO",
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
        batch["target_background_raw"], need_attention,
    )


def _v2_loss(prediction, auxiliary, label, sample_weight, event_pos_weight):
    beta = float(os.environ.get("DL_TCN_V2_HUBER_BETA", "5.0"))
    start = float(os.environ.get("DL_TCN_V2_TAIL_START", "25.0"))
    threshold = float(os.environ.get("DL_TCN_V2_EVENT_THRESHOLD", "35.0"))
    maximum = float(os.environ.get("DL_TCN_V2_TAIL_MAX_WEIGHT", "3.0"))
    tail = 1.0 + (maximum-1.0) * torch.clamp((label-start)/max(threshold-start, 1e-6), 0, 1)
    regression = F.huber_loss(prediction, label, reduction="none", delta=beta)
    event = (label >= threshold).to(label.dtype)
    bce = F.binary_cross_entropy_with_logits(auxiliary["event_logit"], event, reduction="none", pos_weight=event_pos_weight)
    event_weight = float(os.environ.get("DL_TCN_V2_EVENT_LOSS_WEIGHT", "0.1"))
    return torch.mean(sample_weight * (tail * regression + event_weight * bce))


def event_pos_weight_from_dataset(dataset, cube, device):
    pm = CFG.aq_cube_items.index("PM2.5")
    labels = np.asarray(cube[dataset.row_times, dataset.row_targets, pm], dtype="float32")
    positives = int(np.sum(labels >= float(os.environ.get("DL_TCN_V2_EVENT_THRESHOLD", "35"))))
    negatives = len(labels) - positives
    value = min(20.0, negatives / max(positives, 1))
    return torch.tensor(value, dtype=torch.float32, device=device)


def train_epoch_v2(model, loader, optimizer, grad_scaler, device, epoch, adapter,
                   event_pos_weight, feature_builder=None, station_sample_weights=None):
    model.train(); amp_dtype = amp_dtype_for(device); amp_enabled = amp_dtype is not None
    loss_sum = 0.0; sample_count = 0; started = time.perf_counter()
    virtual_weight = float(os.environ.get("DL_TCN_V2_VIRTUAL_LOSS_WEIGHT", "0.1"))
    for batch_number, raw_batch in enumerate(loader, 1):
        batch = feature_builder(raw_batch) if feature_builder is not None else move_batch(raw_batch, device)
        batch = adapter.prepare(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction, auxiliary = _forward(model, batch)
            weights = station_sample_weights[batch["target_idx"]]
            loss = _v2_loss(prediction, auxiliary, batch["label"], weights, event_pos_weight)
            virtual, virtual_valid = adapter.virtual_batch(batch)
            if virtual is not None:
                v_prediction, v_auxiliary = _forward(model, virtual)
                v_loss = _v2_loss(
                    v_prediction, v_auxiliary, virtual["label"],
                    weights * virtual_valid, event_pos_weight,
                )
                loss = loss + virtual_weight * v_loss
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
    loss = F.mse_loss(prediction, batch["label"]); loss.backward()
    if not torch.isfinite(prediction).all() or not torch.isfinite(loss): raise RuntimeError("V2 smoke非finite")
    required = ("tcn.", "donor_projection.", "query_projection.", "cross_attention.", "shared_head.", "residual_head.", "event_head.")
    for prefix in required:
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix)]
        if not grads or all(g is None or float(g.abs().sum()) == 0 for g in grads):
            # event head is exercised separately below.
            if prefix != "event_head.": raise RuntimeError(f"V2 smoke: {prefix}沒有gradient")
    model.zero_grad(set_to_none=True)
    _, event_auxiliary = _forward(model, batch, False)
    event_auxiliary["event_logit"].mean().backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.event_head.parameters()):
        raise RuntimeError("V2 event head沒有finite gradient")
    return {"prediction_shape":list(prediction.shape),"loss":float(loss.detach().cpu()),
            "parameters":sum(p.numel() for p in model.parameters()),"attention_finite":bool(torch.isfinite(auxiliary["attention"]).all())}

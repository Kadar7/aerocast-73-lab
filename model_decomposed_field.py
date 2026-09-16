"""Leakage-safe decomposed field nowcaster for strict unseen-station transfer.

The prediction is deliberately identifiable:

    target PM2.5 = convex current spatial field
                 + time-invariant target offset
                 + zero-centred short-term anomaly.

Only donor histories, donor/target static features, pair geometry and calendar
are model inputs.  Target dynamic history is never accepted by this module.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model_v8 import _MLP, _masked_softmax


class DecomposedFieldNowcaster(nn.Module):
    """Constrained three-part nowcaster.

    Expected tensors follow the V8 protocol:
      values/mask [B,D,24,11], donor_static [B,D,49], geometry [B,D,3],
      donor_padding_mask [B,D], target_static [B,49], calendar [B,24,6],
      current_pm_physical [B,D].

    The five fixed field bases use powers (0, .5, 1, 2, 3) of
    ``(1 + distance_km)^-power``.  A target-conditioned convex gate mixes
    them.  Consequently the spatial field stays inside the observed donor
    range whenever a current donor PM2.5 value exists.
    """

    def __init__(
        self,
        static_dim: int = 49,
        dynamic_channels: int = 11,
        history_hours: int = 24,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        pm_channel_index: int = 7,
        max_static_offset: float = 8.0,
        max_anomaly: float = 25.0,
    ) -> None:
        super().__init__()
        if history_hours != 24 or dynamic_channels != 11:
            raise ValueError("protocol requires 24 history points and 11 channels")
        self.static_dim = int(static_dim)
        self.dynamic_channels = int(dynamic_channels)
        self.history_hours = int(history_hours)
        self.pm_channel_index = int(pm_channel_index)
        self.max_static_offset = float(max_static_offset)
        self.max_anomaly = float(max_anomaly)
        self.register_buffer(
            "field_powers", torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0]),
            persistent=True,
        )

        # Static encoders are shared across target and donors; donor order is
        # never encoded, preserving permutation invariance.
        self.static_encoder = _MLP(static_dim, 64, 24)
        self.calendar_encoder = _MLP(6, 16, 8, layer_norm=False)
        self.relation_encoder = _MLP(24 * 4 + 3, 64, 32)

        # A convex mixture of fixed, auditable spatial estimators.  Initial
        # logits are zero, hence the model starts from their equal average.
        self.field_gate = nn.Sequential(
            nn.Linear(24 + 8 + 11, 48), nn.GELU(), nn.Linear(48, 5)
        )
        nn.init.zeros_(self.field_gate[-1].weight)
        nn.init.zeros_(self.field_gate[-1].bias)

        # The offset is time-invariant and bounded.  It is learned only from
        # target static context and donor-target static relations.
        self.offset_head = nn.Sequential(
            nn.Linear(24 + 32, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

        # Short-term anomaly branch.  Values and masks remain separate inputs;
        # no imputation is introduced.
        self.dynamic_encoder = _MLP(dynamic_channels * 2, 64, 32)
        self.token_projection = nn.Sequential(
            nn.Linear(32 + 32 + 8, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.anomaly_logits = nn.Linear(64, 4)
        self.anomaly_values = nn.Linear(64, 4 * 16)
        self.temporal = nn.GRU(
            input_size=4 * 16 + dynamic_channels + 1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim + 24 + 8 + 11, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )
        nn.init.zeros_(self.anomaly_head[-1].weight)
        nn.init.zeros_(self.anomaly_head[-1].bias)

    def _validate(
        self, values, mask, donor_static, geometry, donor_padding_mask,
        target_static, calendar, current_pm_physical,
    ) -> tuple[int, int]:
        if values.shape != mask.shape or values.ndim != 4:
            raise ValueError("values/mask must be identical [B,D,T,C]")
        batch, donors, history, channels = values.shape
        if (history, channels) != (self.history_hours, self.dynamic_channels):
            raise ValueError("history/channel protocol mismatch")
        expected = {
            "donor_static": (batch, donors, self.static_dim),
            "geometry": (batch, donors, 3),
            "donor_padding_mask": (batch, donors),
            "target_static": (batch, self.static_dim),
            "calendar": (batch, self.history_hours, 6),
            "current_pm_physical": (batch, donors),
        }
        actual = {
            "donor_static": tuple(donor_static.shape),
            "geometry": tuple(geometry.shape),
            "donor_padding_mask": tuple(donor_padding_mask.shape),
            "target_static": tuple(target_static.shape),
            "calendar": tuple(calendar.shape),
            "current_pm_physical": tuple(current_pm_physical.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} expected {shape}, got {actual[name]}")
        return batch, donors

    @staticmethod
    def _batch_scalar(value: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if result.numel() == 1:
            return result.reshape(1).expand(reference.shape[0])
        result = result.reshape(-1)
        if result.shape[0] != reference.shape[0]:
            raise ValueError("fallback mean must be scalar or one value per sample")
        return result

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        donor_static: torch.Tensor,
        geometry: torch.Tensor,
        donor_padding_mask: torch.Tensor,
        target_static: torch.Tensor,
        calendar: torch.Tensor,
        current_pm_physical: torch.Tensor,
        train_pm_mean: float | torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch, donors = self._validate(
            values, mask, donor_static, geometry, donor_padding_mask,
            target_static, calendar, current_pm_physical,
        )
        padding = donor_padding_mask.to(torch.bool)
        mask_bool = mask.to(torch.bool)
        mask_float = mask.to(values.dtype)

        donor_embedding = self.static_encoder(donor_static)
        target_embedding = self.static_encoder(target_static)
        target_expanded = target_embedding[:, None, :].expand(-1, donors, -1)
        relation = self.relation_encoder(torch.cat([
            donor_embedding,
            target_expanded,
            donor_embedding - target_expanded,
            torch.abs(donor_embedding - target_expanded),
            geometry,
        ], dim=-1))
        relation_valid = ~padding
        relation_count = relation_valid.sum(1, keepdim=True).clamp_min(1).to(values.dtype)
        pooled_relation = torch.where(
            relation_valid[..., None], relation, torch.zeros_like(relation)
        ).sum(1) / relation_count

        calendar_embedding = self.calendar_encoder(calendar)
        current_valid = (
            mask_bool[:, :, -1, self.pm_channel_index]
            & ~padding
            & torch.isfinite(current_pm_physical)
        )
        safe_pm = torch.where(current_valid, current_pm_physical, torch.zeros_like(current_pm_physical))
        # geometry[...,0] is log1p(distance_km), so exp(-p*geometry) is
        # exactly (1+distance_km)^(-p), without reconstructing distance.
        basis_logits = -geometry[..., 0, None] * self.field_powers.view(1, 1, -1)
        basis_weights = _masked_softmax(
            basis_logits, current_valid[..., None].expand_as(basis_logits), dim=1
        )
        field_bases = torch.sum(basis_weights * safe_pm[..., None], dim=1)
        basis_min = field_bases.amin(1, keepdim=True)
        basis_max = field_bases.amax(1, keepdim=True)
        basis_mean = field_bases.mean(1, keepdim=True)
        basis_std = field_bases.std(1, keepdim=True, unbiased=False)
        current_count = current_valid.sum(1, keepdim=True).to(values.dtype)
        current_fraction = current_count / (~padding).sum(1, keepdim=True).clamp_min(1).to(values.dtype)
        field_context = torch.cat([
            field_bases,
            basis_min,
            basis_max,
            basis_mean,
            basis_std,
            current_fraction,
            torch.log1p(current_count),
        ], dim=1)
        if field_context.shape[1] != 11:
            raise AssertionError("field context dimension changed")
        gate_logits = self.field_gate(torch.cat([
            target_embedding, calendar_embedding[:, -1], field_context,
        ], dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        spatial_field = torch.sum(gate * field_bases, dim=-1)

        static_offset = self.max_static_offset * torch.tanh(
            self.offset_head(torch.cat([target_embedding, pooled_relation], dim=-1)).squeeze(-1)
        )

        dynamic = self.dynamic_encoder(torch.cat([values, mask_float], dim=-1))
        token = self.token_projection(torch.cat([
            dynamic,
            relation[:, :, None, :].expand(-1, -1, self.history_hours, -1),
            calendar_embedding[:, None, :, :].expand(-1, donors, -1, -1),
        ], dim=-1))
        hour_valid = mask_bool.any(-1) & ~padding[:, :, None]
        logits = self.anomaly_logits(token)
        weights = _masked_softmax(logits, hour_valid[..., None].expand_as(logits), dim=1)
        encoded = self.anomaly_values(token).reshape(batch, donors, self.history_hours, 4, 16)
        pooled = torch.sum(weights[..., None] * encoded, dim=1).reshape(batch, self.history_hours, -1)
        donor_count = (~padding).sum(1).clamp_min(1).to(values.dtype)
        coverage = mask_float.sum(1) / donor_count[:, None, None]
        valid_fraction = hour_valid.sum(1).to(values.dtype) / donor_count[:, None]
        sequence, _ = self.temporal(torch.cat([
            pooled, coverage, valid_fraction[..., None],
        ], dim=-1))
        anomaly_raw = self.anomaly_head(torch.cat([
            sequence[:, -1], target_embedding, calendar_embedding[:, -1], field_context,
        ], dim=-1)).squeeze(-1)
        anomaly = self.max_anomaly * torch.tanh(anomaly_raw / self.max_anomaly)

        has_anchor = current_valid.any(1)
        fallback = self._batch_scalar(train_pm_mean, spatial_field)
        prediction = torch.where(
            has_anchor, spatial_field + static_offset + anomaly, fallback + static_offset
        )
        aux: dict[str, Any] = {
            "spatial_field": spatial_field,
            "static_offset": static_offset,
            "anomaly": anomaly,
            "has_anchor": has_anchor,
            "current_donor_count": current_count.squeeze(1),
            "field_bases": field_bases,
            "field_gate": gate,
            "field_inside_donor_range": (
                (spatial_field >= safe_pm.masked_fill(~current_valid, torch.inf).amin(1))
                & (spatial_field <= safe_pm.masked_fill(~current_valid, -torch.inf).amax(1))
            ) | ~has_anchor,
        }
        if need_weights:
            aux["fixed_field_weights"] = basis_weights
            aux["anomaly_weights"] = weights
        return prediction, aux


def decomposed_loss(
    prediction: torch.Tensor,
    aux: dict[str, torch.Tensor],
    label: torch.Tensor,
    target_idx: torch.Tensor,
    pm_std: float,
    *,
    level_weight: float = 0.35,
    anomaly_center_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Station-balanced objective with identifiable component constraints.

    Batches contain 64 timestamps for each of four stations, but grouping is
    derived from ``target_idx`` rather than relying on that ordering.
    """
    scale = torch.as_tensor(pm_std, dtype=prediction.dtype, device=prediction.device).clamp_min(1e-6)
    main = torch.square((prediction.float() - label.float()) / scale).mean()
    level_terms = []
    anomaly_terms = []
    for station in torch.unique(target_idx):
        keep = target_idx == station
        residual_level = (label[keep].float() - aux["spatial_field"][keep].float()).mean()
        predicted_level = aux["static_offset"][keep].float().mean()
        level_terms.append(torch.square((predicted_level - residual_level) / scale))
        anomaly_terms.append(torch.square(aux["anomaly"][keep].float().mean() / scale))
    level = torch.stack(level_terms).mean()
    anomaly_center = torch.stack(anomaly_terms).mean()
    total = main + level_weight * level + anomaly_center_weight * anomaly_center
    return total, {
        "main_mse": main.detach(),
        "level_mse": level.detach(),
        "anomaly_center_mse": anomaly_center.detach(),
    }

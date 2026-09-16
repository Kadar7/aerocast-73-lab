"""V8-PDC paired output heads on the reviewed V8 donor-field encoder.

The existing V8 already has a signed-affine current-PM skip followed by one
global correction.  PDC does not claim to introduce the first PM reference.
Its exact structural change is a per-donor correction *before* nonnegative
pooling, compared against a parameter-matched no-reference control.

The two experimental arms share every parameter and every tensor operation.
They differ only at the final physical-scale output equation:

``treatment = sum_j w_j * (PM_j,current + sigma_train * delta_j)``
``control   = mu_train + sigma_train * sum_j w_j * delta_j``

Only donors with an observed current PM2.5 value enter the final candidate
set.  An empty set has an exact, non-learned train-mean fallback.
"""
from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from model_v8 import _MLP, _masked_softmax


OutputMode = Literal["treatment", "control"]


def _zero_last_linear(module: nn.Sequential) -> None:
    last = module[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("expected the last head layer to be Linear")
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class PooledDonorCorrection(nn.Module):
    """O(D) V8-PDC model with a shared, permutation-invariant field encoder.

    Tensor contract
    ---------------
    values, mask: [B,D,24,11]
    donor_static: [B,D,49]
    geometry: [B,D,3] where geometry[...,0] is log1p(distance_km)
    donor_padding_mask: [B,D], True means padding
    target_static: [B,49]
    calendar: [B,24,6]
    current_pm_physical: [B,D], NaN when current PM2.5 is unobserved
    """

    def __init__(
        self,
        static_dim: int = 49,
        dynamic_channels: int = 11,
        history_hours: int = 24,
        field_heads: int = 8,
        field_value_dim: int = 16,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        pm_channel_index: int = 7,
        output_mode: OutputMode = "treatment",
    ) -> None:
        super().__init__()
        if history_hours != 24:
            raise ValueError("V8-PDC requires exactly 24 history points")
        if dynamic_channels != 11:
            raise ValueError("V8-PDC requires exactly 11 dynamic channels")
        if field_heads != 8:
            raise ValueError("V8-PDC requires exactly eight field heads")
        if output_mode not in ("treatment", "control"):
            raise ValueError("output_mode must be treatment or control")
        self.static_dim = int(static_dim)
        self.dynamic_channels = int(dynamic_channels)
        self.history_hours = int(history_hours)
        self.field_heads = int(field_heads)
        self.field_value_dim = int(field_value_dim)
        self.hidden_dim = int(hidden_dim)
        self.pm_channel_index = int(pm_channel_index)
        self.output_mode: OutputMode = output_mode

        # This is the reviewed V8 encoder, unchanged in dimensions or order.
        self.static_encoder = _MLP(static_dim, 64, 24)
        self.calendar_encoder = _MLP(6, 16, 8, layer_norm=False)
        self.dynamic_encoder = _MLP(dynamic_channels * 2, 64, 32)
        self.relation_encoder = _MLP(24 * 4 + 3, 64, 32)
        self.token_projection = nn.Sequential(
            nn.Linear(32 + 32 + 8, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.field_logits = nn.Linear(64, field_heads)
        self.field_values = nn.Linear(64, field_heads * field_value_dim)
        field_input_dim = field_heads * field_value_dim + dynamic_channels + 1
        self.field_projection = nn.Sequential(
            nn.Linear(field_input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.target_trunk = nn.Sequential(
            nn.Linear(24 + 8, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 64)
        )

        # Per-donor readout: current token 64 + temporal 96 + target 64 +
        # current coverage context (11 channels + valid-donor fraction) 12.
        donor_readout_dim = 64 + hidden_dim + 64 + dynamic_channels + 1
        self.candidate_correction = nn.Sequential(
            nn.Linear(donor_readout_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.weight_residual = nn.Sequential(
            nn.Linear(donor_readout_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        # At step zero, delta is exactly zero and the weights are exactly the
        # distance prior exp(-2*log1p(distance_km)).
        _zero_last_linear(self.candidate_correction)
        _zero_last_linear(self.weight_residual)

    @staticmethod
    def _scalar_for_batch(value: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if result.numel() == 1:
            return result.reshape(1).expand(reference.shape[0])
        result = result.reshape(-1)
        if result.shape[0] != reference.shape[0]:
            raise ValueError("train PM statistic must be scalar or one value per batch row")
        return result

    def set_output_mode(self, output_mode: OutputMode) -> None:
        if output_mode not in ("treatment", "control"):
            raise ValueError("output_mode must be treatment or control")
        self.output_mode = output_mode

    def _validate_shapes(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        donor_static: torch.Tensor,
        geometry: torch.Tensor,
        donor_padding_mask: torch.Tensor,
        target_static: torch.Tensor,
        calendar: torch.Tensor,
        current_pm_physical: torch.Tensor,
    ) -> tuple[int, int]:
        if values.ndim != 4 or values.shape != mask.shape:
            raise ValueError("values/mask must have identical [B,D,T,C] shapes")
        batch, donors, history, channels = values.shape
        if (history, channels) != (self.history_hours, self.dynamic_channels):
            raise ValueError("unexpected history length or dynamic-channel count")
        expected = {
            "donor_static": (batch, donors, self.static_dim),
            "geometry": (batch, donors, 3),
            "donor_padding_mask": (batch, donors),
            "target_static": (batch, self.static_dim),
            "calendar": (batch, self.history_hours, 6),
            "current_pm_physical": (batch, donors),
        }
        tensors = {
            "donor_static": donor_static,
            "geometry": geometry,
            "donor_padding_mask": donor_padding_mask,
            "target_static": target_static,
            "calendar": calendar,
            "current_pm_physical": current_pm_physical,
        }
        for name, wanted in expected.items():
            if tuple(tensors[name].shape) != wanted:
                raise ValueError(f"{name} expected {wanted}, got {tuple(tensors[name].shape)}")
        return batch, donors

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
        train_pm_std: float | torch.Tensor,
        *,
        output_mode: OutputMode | None = None,
        need_field_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch, donors = self._validate_shapes(
            values, mask, donor_static, geometry, donor_padding_mask,
            target_static, calendar, current_pm_physical,
        )
        mode = self.output_mode if output_mode is None else output_mode
        if mode not in ("treatment", "control"):
            raise ValueError("output_mode must be treatment or control")
        mask = mask.to(dtype=values.dtype)
        padding = donor_padding_mask.to(dtype=torch.bool)

        donor_embedding = self.static_encoder(donor_static)
        target_embedding = self.static_encoder(target_static)
        target_expanded = target_embedding[:, None].expand(-1, donors, -1)
        relation = self.relation_encoder(torch.cat([
            donor_embedding,
            target_expanded,
            donor_embedding - target_expanded,
            torch.abs(donor_embedding - target_expanded),
            geometry,
        ], dim=-1))
        calendar_embedding = self.calendar_encoder(calendar)
        dynamic = self.dynamic_encoder(torch.cat([values, mask], dim=-1))
        token = self.token_projection(torch.cat([
            dynamic,
            relation[:, :, None].expand(-1, -1, self.history_hours, -1),
            calendar_embedding[:, None].expand(-1, donors, -1, -1),
        ], dim=-1))

        hour_valid = mask.to(torch.bool).any(dim=-1) & ~padding[:, :, None]
        field_logits = self.field_logits(token)
        field_valid = hour_valid[..., None].expand_as(field_logits)
        field_weights = _masked_softmax(field_logits, field_valid, dim=1)
        field_values = self.field_values(token).reshape(
            batch, donors, self.history_hours, self.field_heads, self.field_value_dim
        )
        pooled = torch.sum(field_weights[..., None] * field_values, dim=1)
        pooled = pooled.reshape(batch, self.history_hours, -1)
        real_donors = (~padding).sum(dim=1).clamp_min(1).to(values.dtype)
        coverage = mask.sum(dim=1) / real_donors[:, None, None]
        valid_fraction = hour_valid.sum(dim=1).to(values.dtype) / real_donors[:, None]
        field = self.field_projection(torch.cat([
            pooled, coverage, valid_fraction[..., None]
        ], dim=-1))
        temporal_sequence, _ = self.temporal(field)
        temporal_state = temporal_sequence[:, -1]
        target_state = self.target_trunk(torch.cat([
            target_embedding, calendar_embedding[:, -1]
        ], dim=-1))
        current_context = torch.cat([
            coverage[:, -1], valid_fraction[:, -1, None]
        ], dim=-1)

        donor_readout = torch.cat([
            token[:, :, -1],
            temporal_state[:, None].expand(-1, donors, -1),
            target_state[:, None].expand(-1, donors, -1),
            current_context[:, None].expand(-1, donors, -1),
        ], dim=-1)
        delta = self.candidate_correction(donor_readout).squeeze(-1)
        residual_logits = self.weight_residual(donor_readout).squeeze(-1)
        candidate_valid = (
            mask[:, :, -1, self.pm_channel_index].to(torch.bool)
            & ~padding
            & torch.isfinite(current_pm_physical)
        )
        # geometry[...,0] is log1p(distance_km), hence this is proportional
        # to (1 + distance_km)^-2 before the learned residual is introduced.
        weight_logits = -2.0 * geometry[..., 0] + residual_logits
        weights = _masked_softmax(weight_logits, candidate_valid, dim=1)
        valid_pm = torch.where(candidate_valid, current_pm_physical, torch.zeros_like(current_pm_physical))

        pm_mean = self._scalar_for_batch(train_pm_mean, delta[:, 0])
        pm_std = self._scalar_for_batch(train_pm_std, delta[:, 0])
        if not torch.isfinite(pm_mean).all() or not torch.isfinite(pm_std).all():
            raise ValueError("train PM statistics must be finite")
        if bool((pm_std <= 0).any()):
            raise ValueError("train_pm_std must be positive")

        weighted_delta = torch.sum(weights * delta, dim=1)
        correction_component = pm_std * weighted_delta
        reference_component = torch.sum(weights * valid_pm, dim=1)
        if mode == "treatment":
            learned_prediction = reference_component + correction_component
        else:
            learned_prediction = pm_mean + correction_component
        anchor_count = candidate_valid.sum(dim=1)
        has_anchor = anchor_count > 0
        # This is an exact protocol bypass, not a learned fallback.  It also
        # ensures an empty masked softmax can never influence the output.
        prediction = torch.where(has_anchor, learned_prediction, pm_mean)

        aux: dict[str, Any] = {
            "output_mode": mode,
            "delta": delta,
            "candidate_valid": candidate_valid,
            "anchor_count": anchor_count,
            "has_anchor": has_anchor,
            "weights": weights,
            "weight_sum": weights.sum(dim=1),
            "reference_component": reference_component,
            "correction_component": correction_component,
            "current_context": current_context,
            "field_weight_sums": field_weights.sum(dim=1),
        }
        if need_field_weights:
            aux["field_weights"] = field_weights
        return prediction, aux


V8PDC = PooledDonorCorrection

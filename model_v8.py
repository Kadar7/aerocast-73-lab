"""V8 SASFO: an O(D) signed-affine donor-field operator.

This module contains only the model.  Splits, scalers, sampling, checkpointing,
and metrics remain runner responsibilities.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that returns exact zeros, rather than NaN, for an empty set."""
    valid = valid.to(dtype=torch.bool)
    masked = logits.masked_fill(~valid, -torch.inf)
    has_any = valid.any(dim=dim, keepdim=True)
    maximum = masked.amax(dim=dim, keepdim=True)
    maximum = torch.where(has_any, maximum, torch.zeros_like(maximum))
    numerator = torch.where(valid, torch.exp(logits - maximum), torch.zeros_like(logits))
    denominator = numerator.sum(dim=dim, keepdim=True)
    return torch.where(
        has_any,
        numerator / denominator.clamp_min(torch.finfo(logits.dtype).tiny),
        torch.zeros_like(numerator),
    )


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, layer_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SASFO(nn.Module):
    """Signed Affine Spatiotemporal Field Operator.

    Parameters are intentionally small and shared across donors.  Donor order
    is never encoded, so the forward pass is permutation invariant in ``D``.

    Expected tensors
    ----------------
    values, mask: [B, D, 24, 11]
    donor_static: [B, D, 49]
    geometry: [B, D, 3]
    donor_padding_mask: [B, D], True for padding
    target_static: [B, 49]
    calendar: [B, 24, 6]
    current_pm_physical: [B, D] in micrograms/m3
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
        max_skip_l1: float = 3.0,
        pm_channel_index: int = 7,
        zero_guard: float = 1e-8,
    ) -> None:
        super().__init__()
        if history_hours != 24:
            raise ValueError("V8 protocol requires exactly 24 history points")
        if dynamic_channels != 11:
            raise ValueError("V8 protocol requires exactly 11 dynamic channels")
        if field_heads != 8:
            raise ValueError("V8 protocol requires exactly 8 positive field heads")
        if max_skip_l1 < 1.0:
            raise ValueError("max_skip_l1 must be at least one")

        self.static_dim = static_dim
        self.dynamic_channels = dynamic_channels
        self.history_hours = history_hours
        self.field_heads = field_heads
        self.field_value_dim = field_value_dim
        self.hidden_dim = hidden_dim
        self.max_skip_l1 = float(max_skip_l1)
        self.pm_channel_index = pm_channel_index
        self.zero_guard = float(zero_guard)

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
        self.temporal_projection = nn.Linear(hidden_dim, 64)

        # Current-PM affine skip.  The signed direction is structurally
        # zero-sum and its amplitude gives ||w||_1 <= max_skip_l1.
        self.skip_positive_score = nn.Linear(64, 1)
        self.skip_signed_score = nn.Linear(64, 1)
        self.skip_amplitude = nn.Sequential(
            nn.Linear(hidden_dim + 64 + dynamic_channels + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        readout_dim = 64 + 64 + 64 + dynamic_channels + 1
        self.correction_head = nn.Sequential(
            nn.Linear(readout_dim, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 1)
        )

    @staticmethod
    def _scalar_for_batch(value: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if result.numel() == 1:
            return result.reshape(1).expand(reference.shape[0])
        result = result.reshape(-1)
        if result.shape[0] != reference.shape[0]:
            raise ValueError("train PM statistic must be scalar or have one value per batch row")
        return result

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
        if values.shape != mask.shape or values.ndim != 4:
            raise ValueError("values/mask must have identical [B,D,T,C] shapes")
        batch, donors, history, channels = values.shape
        if (history, channels) != (self.history_hours, self.dynamic_channels):
            raise ValueError(
                f"expected history/channels {(self.history_hours, self.dynamic_channels)}, "
                f"got {(history, channels)}"
            )
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
        for name, wanted in expected.items():
            if actual[name] != wanted:
                raise ValueError(f"{name} expected {wanted}, got {actual[name]}")
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
        need_field_weights: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch, donors = self._validate_shapes(
            values,
            mask,
            donor_static,
            geometry,
            donor_padding_mask,
            target_static,
            calendar,
            current_pm_physical,
        )
        mask = mask.to(dtype=values.dtype)
        padding = donor_padding_mask.to(dtype=torch.bool)

        donor_embedding = self.static_encoder(donor_static)
        target_embedding = self.static_encoder(target_static)
        target_expanded = target_embedding[:, None, :].expand(-1, donors, -1)
        relation = self.relation_encoder(
            torch.cat(
                [
                    donor_embedding,
                    target_expanded,
                    donor_embedding - target_expanded,
                    torch.abs(donor_embedding - target_expanded),
                    geometry,
                ],
                dim=-1,
            )
        )
        calendar_embedding = self.calendar_encoder(calendar)
        dynamic = self.dynamic_encoder(torch.cat([values, mask], dim=-1))
        token = self.token_projection(
            torch.cat(
                [
                    dynamic,
                    relation[:, :, None, :].expand(-1, -1, self.history_hours, -1),
                    calendar_embedding[:, None, :, :].expand(-1, donors, -1, -1),
                ],
                dim=-1,
            )
        )

        hour_valid = mask.to(dtype=torch.bool).any(dim=-1) & ~padding[:, :, None]
        field_logits = self.field_logits(token)  # [B,D,T,K]
        field_valid = hour_valid[..., None].expand_as(field_logits)
        field_weights = _masked_softmax(field_logits, field_valid, dim=1)
        field_values = self.field_values(token).reshape(
            batch, donors, self.history_hours, self.field_heads, self.field_value_dim
        )
        pooled = torch.sum(field_weights[..., None] * field_values, dim=1)
        pooled = pooled.reshape(batch, self.history_hours, -1)

        real_donor_count = (~padding).sum(dim=1).clamp_min(1).to(values.dtype)
        coverage = mask.sum(dim=1) / real_donor_count[:, None, None]
        valid_fraction = hour_valid.sum(dim=1).to(values.dtype) / real_donor_count[:, None]
        field = self.field_projection(
            torch.cat([pooled, coverage, valid_fraction[..., None]], dim=-1)
        )
        temporal_sequence, _ = self.temporal(field)
        temporal_state = temporal_sequence[:, -1, :]

        target_state = self.target_trunk(
            torch.cat([target_embedding, calendar_embedding[:, -1, :]], dim=-1)
        )
        temporal_64 = self.temporal_projection(temporal_state)
        current_context = torch.cat([coverage[:, -1, :], valid_fraction[:, -1, None]], dim=-1)

        current_token = token[:, :, -1, :]
        current_pm_valid = (
            mask[:, :, -1, self.pm_channel_index].to(dtype=torch.bool)
            & ~padding
            & torch.isfinite(current_pm_physical)
        )
        valid_expanded = current_pm_valid[..., None]
        positive_logits = self.skip_positive_score(current_token).squeeze(-1) - 2.0 * geometry[..., 0]
        positive = _masked_softmax(positive_logits, current_pm_valid, dim=1)

        signed_raw = torch.tanh(self.skip_signed_score(current_token).squeeze(-1))
        anchor_count = current_pm_valid.sum(dim=1)
        safe_count = anchor_count.clamp_min(1).to(values.dtype)
        signed_mean = torch.where(
            current_pm_valid,
            signed_raw,
            torch.zeros_like(signed_raw),
        ).sum(dim=1, keepdim=True) / safe_count[:, None]
        centered = torch.where(
            current_pm_valid,
            signed_raw - signed_mean,
            torch.zeros_like(signed_raw),
        )
        centered_l1 = centered.abs().sum(dim=1, keepdim=True)
        # Clamp is applied before division.  torch.where never sees an illegal
        # zero denominator, and the false branch is exact zero.
        normalized_direction = centered / centered_l1.clamp_min(self.zero_guard)
        normalized_direction = torch.where(
            centered_l1 > self.zero_guard,
            normalized_direction,
            torch.zeros_like(normalized_direction),
        )

        amplitude_input = torch.cat([temporal_state, target_state, current_context], dim=-1)
        rho = (self.max_skip_l1 - 1.0) * torch.sigmoid(
            self.skip_amplitude(amplitude_input).squeeze(-1)
        )
        skip_weights = positive + rho[:, None] * normalized_direction
        skip_weights = torch.where(current_pm_valid, skip_weights, torch.zeros_like(skip_weights))

        safe_pm = torch.where(
            current_pm_valid,
            current_pm_physical,
            torch.zeros_like(current_pm_physical),
        )
        affine_base = torch.sum(skip_weights * safe_pm, dim=1)
        has_anchor = anchor_count > 0

        correction_input = torch.cat(
            [
                temporal_64,
                target_state,
                temporal_64 * target_state,
                current_context,
            ],
            dim=-1,
        )
        delta = self.correction_head(correction_input).squeeze(-1)
        pm_mean = self._scalar_for_batch(train_pm_mean, delta)
        pm_std = self._scalar_for_batch(train_pm_std, delta)
        if torch.any(~torch.isfinite(pm_mean)) or torch.any(~torch.isfinite(pm_std)):
            raise ValueError("train PM statistics must be finite")
        if torch.any(pm_std <= 0):
            raise ValueError("train_pm_std must be positive")
        anchored_prediction = affine_base + pm_std * delta
        # Exact protocol bypass: without any current PM anchor, do not allow a
        # learned correction to invent an absolute gauge.
        prediction = torch.where(has_anchor, anchored_prediction, pm_mean)

        aux: dict[str, Any] = {
            "affine_base": affine_base,
            "delta": delta,
            "has_anchor": has_anchor,
            "anchor_count": anchor_count,
            "skip_weights": skip_weights,
            "skip_positive_weights": positive,
            "skip_zero_sum_direction": normalized_direction,
            "skip_rho": rho,
            "skip_weight_sum": skip_weights.sum(dim=1),
            "skip_l1": skip_weights.abs().sum(dim=1),
            "current_context": current_context,
            "field_weight_sums": field_weights.sum(dim=1),
        }
        if need_field_weights:
            aux["field_weights"] = field_weights
        return prediction, aux


# Descriptive alias for runners that prefer the long model name.
SignedAffineSpatiotemporalFieldOperator = SASFO

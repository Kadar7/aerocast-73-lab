from __future__ import annotations

import math

import torch
from torch import nn

from config import CFG, Config
from model import SharedTCN


class PhysicsGuidedCrossAttention(nn.Module):
    """Single-query multi-head attention with an explicit transport prior."""

    def __init__(self, cfg: Config = CFG) -> None:
        super().__init__()
        if cfg.attention_dim % cfg.attention_heads:
            raise ValueError("attention_dim必須可被attention_heads整除")
        self.heads = cfg.attention_heads
        self.head_dim = cfg.attention_dim // cfg.attention_heads
        self.q = nn.Linear(cfg.attention_dim, cfg.attention_dim)
        self.k = nn.Linear(cfg.attention_dim, cfg.attention_dim)
        self.v = nn.Linear(cfg.attention_dim, cfg.attention_dim)
        self.out = nn.Linear(cfg.attention_dim, cfg.attention_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        # softplus keeps distance/cross penalties and downwind reward interpretable.
        self.distance_strength = nn.Parameter(torch.full((self.heads,), -1.5))
        self.along_strength = nn.Parameter(torch.full((self.heads,), -1.5))
        self.cross_strength = nn.Parameter(torch.full((self.heads,), -2.0))
        self.static_strength = nn.Parameter(torch.full((self.heads,), -2.0))

    def forward(self, query, tokens, geometry, values, mask, padding_mask, static_similarity,
                need_weights=False, physical_wind=None):
        batch, donors, dim = tokens.shape
        q = self.q(query).view(batch, self.heads, self.head_dim)
        k = self.k(tokens).view(batch, donors, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(tokens).view(batch, donors, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhd,bhnd->bhn", q, k) / math.sqrt(self.head_dim)

        log_distance = geometry[..., 0]
        if physical_wind is None:
            raise ValueError("Physical wind from the source-only scaler is required")
        along = physical_wind[..., 0]
        cross = physical_wind[..., 1]
        wind_ok = mask[:, :, -1, 9] * mask[:, :, -1, 10]
        along = along * wind_ok
        cross = cross * wind_ok
        # donor_static-target_static difference is appended immediately before
        # geometry/background fields; its mean square is passed separately.
        prior = (
            -torch.nn.functional.softplus(self.distance_strength)[None, :, None]
            * log_distance[:, None, :]
            + torch.nn.functional.softplus(self.along_strength)[None, :, None]
            * along[:, None, :]
            - torch.nn.functional.softplus(self.cross_strength)[None, :, None]
            * cross.abs()[:, None, :]
            + torch.nn.functional.softplus(self.static_strength)[None, :, None]
            * static_similarity[:, None, :]
        )
        scores = scores + prior
        scores = scores.masked_fill(padding_mask[:, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attended = torch.einsum("bhn,bhnd->bhd", weights, v).reshape(batch, dim)
        return self.out(attended), (weights if need_weights else None)


class TCNTargetCrossAttentionV2(nn.Module):
    """V2: background/anomaly + static contrasts + transport prior + event head."""

    def __init__(self, static_dim: int = 49, cfg: Config = CFG) -> None:
        super().__init__()
        self.cfg = cfg
        self.static_dim = static_dim
        self.tcn = SharedTCN(cfg)
        donor_dim = cfg.tcn_hidden + static_dim * 2 + 3 + 2
        self.donor_projection = nn.Sequential(
            nn.Linear(donor_dim, cfg.attention_dim), nn.GELU(),
            nn.LayerNorm(cfg.attention_dim),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(static_dim + 4 + 1, cfg.attention_dim), nn.GELU(),
            nn.LayerNorm(cfg.attention_dim),
        )
        self.cross_attention = PhysicsGuidedCrossAttention(cfg)
        self.shared_head = nn.Sequential(
            nn.Linear(cfg.attention_dim * 2, cfg.final_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.residual_head = nn.Linear(cfg.final_hidden, 1)
        self.event_head = nn.Linear(cfg.final_hidden, 1)

    def forward(self, values, mask, donor_static, geometry, donor_padding_mask,
                target_static, time_features, donor_background,
                target_background, target_background_raw,
                need_attention_weights=False, physical_wind=None):
        temporal = self.tcn(values, mask)
        static_difference = donor_static - target_static[:, None, :]
        background_difference = donor_background - target_background[:, None]
        donor_input = torch.cat([
            temporal, donor_static, static_difference, geometry,
            donor_background[..., None], background_difference[..., None],
        ], dim=-1)
        donor_token = self.donor_projection(donor_input)
        query = self.query_projection(torch.cat([
            target_static, time_features, target_background[:, None],
        ], dim=-1))
        attended, attention = self.cross_attention(
            query, donor_token, geometry, values, mask, donor_padding_mask,
            -static_difference.square().mean(dim=-1),
            need_attention_weights, physical_wind,
        )
        hidden = self.shared_head(torch.cat([attended, query], dim=-1))
        prediction = target_background_raw + self.residual_head(hidden).squeeze(-1)
        auxiliary = {
            "attention": attention,
            "event_logit": self.event_head(hidden).squeeze(-1),
            "residual": prediction - target_background_raw,
        }
        return prediction, auxiliary

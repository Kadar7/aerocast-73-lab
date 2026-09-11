from __future__ import annotations
import torch
from torch import nn
from config import CFG, Config

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__(); self.chomp_size = chomp_size
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation), Chomp1d(pad),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation), Chomp1d(pad),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.res = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        self.norm = nn.LayerNorm(out_ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x) + self.res(x)
        return self.norm(y.transpose(1, 2)).transpose(1, 2)

class SharedTCN(nn.Module):
    def __init__(self, cfg: Config = CFG) -> None:
        super().__init__()
        blocks = []
        in_ch = cfg.n_dynamic_channels * 2  # 11 values + 11 masks
        for i, d in enumerate(cfg.tcn_dilations):
            blocks.append(TemporalBlock(in_ch if i == 0 else cfg.tcn_hidden,
                                        cfg.tcn_hidden, cfg.tcn_kernel_size, d, cfg.dropout))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.shape != mask.shape:
            raise ValueError("values/mask shape 不一致")
        B, D, T, C = values.shape
        x = torch.cat([values, mask], dim=-1).reshape(B * D, T, C * 2).transpose(1, 2)
        h = self.blocks(x)[:, :, -1]
        return h.reshape(B, D, -1)

class TCNTargetCrossAttention(nn.Module):
    def __init__(self, static_dim: int = 49, cfg: Config = CFG) -> None:
        super().__init__(); self.cfg = cfg
        self.tcn = SharedTCN(cfg)
        self.donor_projection = nn.Sequential(
            nn.Linear(cfg.tcn_hidden + static_dim + 3, cfg.attention_dim),
            nn.GELU(), nn.LayerNorm(cfg.attention_dim),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(static_dim + 4, cfg.attention_dim),
            nn.GELU(), nn.LayerNorm(cfg.attention_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            cfg.attention_dim, cfg.attention_heads, dropout=cfg.dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(cfg.attention_dim * 2, cfg.final_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.final_hidden, 1),
        )
    def forward(self, values, mask, donor_static, geometry, donor_padding_mask,
                target_static, time_features, need_attention_weights=False):
        temporal = self.tcn(values, mask)
        donor_token = self.donor_projection(torch.cat([temporal, donor_static, geometry], dim=-1))
        query = self.query_projection(torch.cat([target_static, time_features], dim=-1)).unsqueeze(1)
        attn_out, attn_w = self.cross_attention(
            query, donor_token, donor_token, key_padding_mask=donor_padding_mask,
            need_weights=need_attention_weights,
            average_attn_weights=False if need_attention_weights else True,
        )
        pred = self.head(torch.cat([attn_out.squeeze(1), query.squeeze(1)], dim=-1)).squeeze(-1)
        return pred, attn_w

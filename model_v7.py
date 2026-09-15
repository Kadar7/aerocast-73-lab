"""V7 Context-Held-Out Gradient Operator (CHO-GO).

The target never contributes dynamic observations.  Each donor acts once as an
absolute PM2.5 anchor while its entire hash group is removed from the potential
context used for that candidate.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -torch.inf)
    all_masked = ~mask.any(dim=dim, keepdim=True)
    logits = torch.where(all_masked, torch.zeros_like(logits), logits)
    out = torch.softmax(logits.float(), dim=dim).to(logits.dtype)
    return torch.where(all_masked, torch.zeros_like(out), out)


class CausalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, width)
        self.norm2 = nn.GroupNorm(1, width)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def _conv(self, x, conv):
        return conv(nn.functional.pad(x, (2 * self.dilation, 0)))

    def forward(self, x):
        y = self.drop(self.act(self.norm1(self._conv(x, self.conv1))))
        y = self.drop(self.act(self.norm2(self._conv(y, self.conv2))))
        return x + y


class DenseEdgeGraphLayer(nn.Module):
    def __init__(self, width: int = 64, edge_dim: int = 19, heads: int = 4, dropout: float = .1):
        super().__init__()
        self.heads, self.dk = heads, width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.edge_value = nn.Linear(edge_dim, width)
        self.out = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 2, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge, support, knn_mask):
        # edge/query/source: [B,Q,S,E].  Here Q=S=D.
        b, d, w = h.shape
        q = self.q(h).view(b, d, self.heads, self.dk)
        k = self.k(h).view(b, d, self.heads, self.dk)
        v = self.v(h).view(b, d, self.heads, self.dk)
        logits = torch.einsum("bqhd,bshd->bqsh", q, k) / math.sqrt(self.dk)
        logits = logits + self.edge_bias(edge)
        valid = support[:, None, :, None] & knn_mask[..., None]
        attn = _masked_softmax(logits, valid, dim=2)
        ev = self.edge_value(edge).view(b, d, d, self.heads, self.dk)
        value = v[:, None, :, :, :] + ev
        message = torch.sum(attn[..., None] * value, dim=2).reshape(b, d, w)
        x = self.norm1(h + self.drop(self.out(message)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return torch.where(support[..., None], x, torch.zeros_like(x))


class PotentialQuery(nn.Module):
    def __init__(self, width=64, static_dim=32, edge_dim=19, heads=4, dropout=.1):
        super().__init__()
        self.heads, self.dk = heads, width // heads
        self.seed = nn.Sequential(nn.Linear(static_dim + 16 + width, width), nn.GELU(), nn.LayerNorm(width))
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.rel = nn.Sequential(nn.Linear(static_dim * 4 + edge_dim, width), nn.GELU(), nn.Linear(width, width))
        self.bias = nn.Linear(edge_dim, heads)
        self.out = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, width))
        self.head = nn.Sequential(nn.Linear(width + static_dim + 16, 64), nn.GELU(), nn.Linear(64, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, query_static, source_static, source_h, edge, support, local_mask, time_embedding):
        # query_static [B,Q,32], edge/local [B,Q,D,*]
        b, qn, _ = query_static.shape
        d = source_h.shape[1]
        te = time_embedding[:, None, :].expand(-1, qn, -1)
        support_f=support.to(source_h.dtype)
        global_state=(source_h*support_f[...,None]).sum(1)/support_f.sum(1,keepdim=True).clamp_min(1)
        global_state=global_state[:,None].expand(-1,qn,-1)
        seed = self.seed(torch.cat([query_static, te, global_state], -1))
        q = self.q(seed).view(b, qn, self.heads, self.dk)
        k = self.k(source_h).view(b, d, self.heads, self.dk)
        v = self.v(source_h).view(b, d, self.heads, self.dk)
        qs = query_static[:, :, None, :].expand(-1, -1, d, -1)
        ss = source_static[:, None, :, :].expand(-1, qn, -1, -1)
        relation = torch.cat([ss, qs, ss - qs, (ss - qs).abs(), edge], -1)
        rel = self.rel(relation).view(b, qn, d, self.heads, self.dk)
        logits = torch.einsum("bqhd,bshd->bqsh", q, k) / math.sqrt(self.dk) + self.bias(edge)
        valid = support[:, None, :, None] & local_mask[..., None]
        attn = _masked_softmax(logits, valid, dim=2)
        message = torch.sum(attn[..., None] * (v[:, None] + rel), dim=2).reshape(b, qn, -1)
        state = self.norm1(seed + self.drop(self.out(message)))
        state = self.norm2(state + self.drop(self.ff(state)))
        potential = self.head(torch.cat([state, query_static, te], -1)).squeeze(-1)
        return potential, state


class ContextHeldOutGradientOperator(nn.Module):
    def __init__(self, static_dim=49, pm_mean=0.0, pm_std=1.0, dropout=.1, width=64):
        super().__init__()
        self.pm_std = float(pm_std)
        self.pm_mean = float(pm_mean)
        self.static = nn.Sequential(nn.Linear(static_dim, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 32), nn.GELU(), nn.LayerNorm(32))
        self.calendar = nn.Sequential(nn.Linear(6, 32), nn.GELU(), nn.Linear(32, 16))
        self.point = nn.Sequential(nn.Linear(24, width), nn.GELU(), nn.LayerNorm(width))
        self.temporal = nn.Sequential(CausalBlock(width, 1, dropout), CausalBlock(width, 2, dropout), CausalBlock(width, 4, dropout))
        self.node_out = nn.Sequential(nn.Linear(width + 32, width), nn.GELU(), nn.LayerNorm(width))
        self.edge_encoder = nn.Sequential(nn.Linear(19, 64), nn.GELU(), nn.Linear(64, 19))
        self.graph = nn.ModuleList([DenseEdgeGraphLayer(width, 19, 4, dropout) for _ in range(2)])
        self.query = PotentialQuery(width, 32, 19, 4, dropout)

    @staticmethod
    def _nearest_mask(distance, support, k):
        # distance [B,Q,D], support [B,D]
        masked = distance.masked_fill(~support[:, None, :], torch.inf)
        kk = min(k, masked.shape[-1])
        idx = torch.topk(masked, kk, dim=-1, largest=False).indices
        out = torch.zeros_like(masked, dtype=torch.bool)
        out.scatter_(-1, idx, True)
        return out & torch.isfinite(masked)

    def forward(self, values, mask, donor_static, target_static, calendar_history,
                edge_dd, edge_target, donor_groups, donor_padding_mask,
                current_pm_raw, current_pm_mask, distance_dd, distance_target,
                geometry_weight_scale):
        b, d, t, _ = values.shape
        cal = self.calendar(calendar_history)
        node_input = torch.cat([values[..., :9], mask[..., :9], calendar_history[:, None].expand(-1, d, -1, -1)], -1)
        x = self.point(node_input).permute(0, 1, 3, 2).reshape(b * d, -1, t)
        h = self.temporal(x)[..., -1].reshape(b, d, -1)
        ds = self.static(donor_static)
        ts = self.static(target_static)
        h = self.node_out(torch.cat([h, ds], -1))
        # All-missing donors retain only static/calendar plus zero values and
        # zero masks.  They never become PM anchors without current PM truth.
        valid_node = ~donor_padding_mask
        current_time = cal[:, -1]
        candidates, candidate_mask = [], []
        for group in range(4):
            anchor = (donor_groups == group) & (~donor_padding_mask)
            support = (~anchor) & valid_node
            graph_h = h
            diagonal = torch.eye(d,device=distance_dd.device,dtype=torch.bool)[None]
            graph_distance = distance_dd.masked_fill(diagonal, torch.inf)
            knn = self._nearest_mask(graph_distance, support, 8)
            ee = self.edge_encoder(edge_dd)
            for layer in self.graph:
                graph_h = layer(graph_h, ee, support, knn)

            # Query target plus every donor using the identical module/context.
            query_static = torch.cat([ts[:, None], ds], 1)
            query_edge = torch.cat([edge_target[:, None], edge_dd], 1)
            query_dist = torch.cat([distance_target[:, None], distance_dd], 1)
            local = self._nearest_mask(query_dist, support, 16)
            phi, _ = self.query(query_static, ds, graph_h, self.edge_encoder(query_edge), support, local, current_time)
            delta = (phi[:, :1] - phi[:, 1:]) * self.pm_std
            c = current_pm_raw + delta
            cm = anchor & current_pm_mask
            candidates.append(c)
            candidate_mask.append(cm)
        c = torch.stack(candidates, 1)  # [B,4,D]
        cm = torch.stack(candidate_mask, 1)
        dist = distance_target[:, None, :].expand_as(c)
        g = torch.exp(-dist / geometry_weight_scale.clamp_min(1e-6)) + .05
        g = torch.where(cm, g, torch.zeros_like(g))
        denom = g.sum((1, 2))
        pred = (g * c).sum((1, 2)) / denom.clamp_min(1e-8)
        fallback = torch.full_like(pred, self.pm_mean)
        pred = torch.where(denom > 0, pred, fallback)
        aux = {
            "candidate_count": cm.sum((1, 2)),
            "donor_supported": denom > 0,
            "candidate_values": c,
            "candidate_mask": cm,
            "candidate_weights": g / denom[:, None, None].clamp_min(1e-8),
        }
        return pred, aux

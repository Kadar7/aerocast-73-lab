"""V5: identifiable slow climatology correction plus fast dynamic anomaly."""
from __future__ import annotations
import torch
from torch import nn
from config import CFG,Config
from model import SharedTCN
from model_v2 import PhysicsGuidedCrossAttention


class FactorizedResidualKriging(nn.Module):
    def __init__(self,static_dim:int=49,cfg:Config=CFG):
        super().__init__(); self.cfg=cfg; self.static_dim=static_dim
        # Deliberately low-capacity: it learns how to pool legal donor
        # climatology residuals, rather than memorising 50 station labels.
        self.slow_score=nn.Sequential(nn.Linear(5,8),nn.Tanh(),nn.Linear(8,1))
        nn.init.zeros_(self.slow_score[-1].weight); nn.init.zeros_(self.slow_score[-1].bias)
        # Unconstrained signed gain permits extrapolation beyond the donor mean
        # range; it is anchored by the explicit station-level slow loss.
        self.slow_gain=nn.Parameter(torch.tensor(0.5))
        self.tcn=SharedTCN(cfg)
        donor_dim=cfg.tcn_hidden+static_dim*2+3+2
        self.donor_projection=nn.Sequential(
            nn.Linear(donor_dim,cfg.attention_dim),nn.GELU(),nn.LayerNorm(cfg.attention_dim))
        self.query_projection=nn.Sequential(
            nn.Linear(static_dim+4+1,cfg.attention_dim),nn.GELU(),nn.LayerNorm(cfg.attention_dim))
        self.cross_attention=PhysicsGuidedCrossAttention(cfg)
        self.fast_head=nn.Sequential(
            nn.Linear(cfg.attention_dim*2,cfg.final_hidden),nn.GELU(),nn.Dropout(cfg.dropout),
            nn.Linear(cfg.final_hidden,1))

    def forward(self,values,mask,donor_static,geometry,donor_padding_mask,
                target_static,time_features,donor_background,target_background,
                target_background_raw,donor_climatology_raw,
                need_attention_weights=False,physical_wind=None,relation_scales=None):
        static_difference=donor_static-target_static[:,None,:]
        static_distance=static_difference.square().mean(dim=-1,keepdim=True)
        level_difference=(donor_background-target_background[:,None]).unsqueeze(-1)
        slow_features=torch.cat([geometry,static_distance,level_difference],dim=-1)
        slow_logits=self.slow_score(slow_features).squeeze(-1)
        slow_logits=slow_logits.masked_fill(donor_padding_mask,torch.finfo(slow_logits.dtype).min)
        slow_weights=torch.softmax(slow_logits,dim=-1)
        pooled_climatology=torch.sum(slow_weights*donor_climatology_raw,dim=-1)
        slow_correction=self.slow_gain*(pooled_climatology-target_background_raw)

        temporal=self.tcn(values,mask)
        background_difference=donor_background-target_background[:,None]
        donor_input=torch.cat([temporal,donor_static,static_difference,geometry,
                               donor_background[...,None],background_difference[...,None]],dim=-1)
        donor_token=self.donor_projection(donor_input)
        corrected_level=(target_background_raw+slow_correction)/10.0
        query=self.query_projection(torch.cat([target_static,time_features,corrected_level[:,None]],dim=-1))
        attended,attention=self.cross_attention(
            query,donor_token,geometry,values,mask,donor_padding_mask,
            -static_difference.square().mean(dim=-1),need_attention_weights,
            physical_wind,relation_scales)
        anomaly=self.fast_head(torch.cat([attended,query],dim=-1)).squeeze(-1)
        prediction=target_background_raw+slow_correction+anomaly
        return prediction,{
            'slow_correction':slow_correction,'anomaly':anomaly,
            'slow_attention':slow_weights if need_attention_weights else None,
            'attention':attention,
        }

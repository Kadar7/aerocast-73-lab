"""V6 source-climatology anchored fast branch."""
from __future__ import annotations
import torch
from torch import nn
from config import CFG,Config
from model import SharedTCN
from model_v2 import PhysicsGuidedCrossAttention


class AnchoredFactorizedResidualKriging(nn.Module):
    def __init__(self,static_dim:int=49,cfg:Config=CFG,anchored:bool=True):
        super().__init__();self.cfg=cfg;self.static_dim=static_dim;self.anchored=anchored
        self.slow_score=nn.Sequential(nn.Linear(5,8),nn.Tanh(),nn.Linear(8,1))
        nn.init.zeros_(self.slow_score[-1].weight);nn.init.zeros_(self.slow_score[-1].bias)
        self.slow_gain=nn.Parameter(torch.tensor(0.5))
        self.tcn=SharedTCN(cfg)
        donor_dim=cfg.tcn_hidden+static_dim*2+3+2
        self.donor_projection=nn.Sequential(nn.Linear(donor_dim,cfg.attention_dim),nn.GELU(),nn.LayerNorm(cfg.attention_dim))
        self.query_projection=nn.Sequential(nn.Linear(static_dim+4+1,cfg.attention_dim),nn.GELU(),nn.LayerNorm(cfg.attention_dim))
        self.cross_attention=PhysicsGuidedCrossAttention(cfg)
        self.fast_head=nn.Sequential(nn.Linear(cfg.attention_dim*2,cfg.final_hidden),nn.GELU(),
                                     nn.Dropout(cfg.dropout),nn.Linear(cfg.final_hidden,1))

    def _slow(self,donor_static,geometry,donor_padding_mask,target_static,
              donor_background,target_background,target_background_raw,donor_climatology_raw):
        static_difference=donor_static-target_static[:,None,:]
        static_distance=static_difference.square().mean(dim=-1,keepdim=True)
        level_difference=(donor_background-target_background[:,None]).unsqueeze(-1)
        features=torch.cat([geometry,static_distance,level_difference],dim=-1)
        logits=self.slow_score(features).squeeze(-1)
        logits=logits.masked_fill(donor_padding_mask,torch.finfo(logits.dtype).min)
        weights=torch.softmax(logits,dim=-1)
        pooled=torch.sum(weights*donor_climatology_raw,dim=-1)
        return self.slow_gain*(pooled-target_background_raw),weights,static_difference

    def _fast(self,values,mask,donor_static,geometry,donor_padding_mask,target_static,time_features,
              donor_background,target_background,corrected_level,physical_wind,relation_scales,
              static_difference,need_attention_weights=False):
        temporal=self.tcn(values,mask)
        background_difference=donor_background-target_background[:,None]
        donor_input=torch.cat([temporal,donor_static,static_difference,geometry,
                               donor_background[...,None],background_difference[...,None]],dim=-1)
        donor_token=self.donor_projection(donor_input)
        query=self.query_projection(torch.cat([target_static,time_features,corrected_level[:,None]],dim=-1))
        attended,attention=self.cross_attention(query,donor_token,geometry,values,mask,donor_padding_mask,
            -static_difference.square().mean(dim=-1),need_attention_weights,physical_wind,relation_scales)
        scalar=self.fast_head(torch.cat([attended,query],dim=-1)).squeeze(-1)
        return scalar,attention

    @staticmethod
    def _rng_state(device):
        cpu=torch.get_rng_state()
        cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
        return cpu,cuda

    @staticmethod
    def _set_rng_state(state,device):
        torch.set_rng_state(state[0])
        if state[1] is not None: torch.cuda.set_rng_state(state[1],device)

    def forward(self,values,mask,donor_static,geometry,donor_padding_mask,target_static,time_features,
                donor_background,target_background,target_background_raw,donor_climatology_raw,
                reference_values,reference_physical_wind,need_attention_weights=False,
                physical_wind=None,relation_scales=None):
        slow,slow_weights,static_difference=self._slow(donor_static,geometry,donor_padding_mask,target_static,
            donor_background,target_background,target_background_raw,donor_climatology_raw)
        corrected=(target_background_raw+slow)/10.0
        before=self._rng_state(values.device) if self.training and self.anchored else None
        actual,attention=self._fast(values,mask,donor_static,geometry,donor_padding_mask,target_static,time_features,
            donor_background,target_background,corrected,physical_wind,relation_scales,static_difference,
            need_attention_weights)
        if self.anchored:
            after=self._rng_state(values.device) if self.training else None
            if before is not None: self._set_rng_state(before,values.device)
            reference,_=self._fast(reference_values,mask,donor_static,geometry,donor_padding_mask,target_static,
                time_features,donor_background,target_background,corrected,reference_physical_wind,
                relation_scales,static_difference,False)
            if after is not None: self._set_rng_state(after,values.device)
            anomaly=actual-reference
        else:
            reference=torch.zeros_like(actual);anomaly=actual
        prediction=target_background_raw+slow+anomaly
        return prediction,{'slow_correction':slow,'anomaly':anomaly,'actual_fast':actual,
            'reference_fast':reference,'slow_attention':slow_weights if need_attention_weights else None,
            'attention':attention}

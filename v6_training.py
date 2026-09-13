"""Batch adapter, forward and matched ERM step for V6."""
from __future__ import annotations
import torch
from config import CFG
from v5_training import V5BatchAdapter,v5_loss


class V6BatchAdapter(V5BatchAdapter):
    def prepare(self,batch,need_target_mean=True):
        batch=super().prepare(batch,need_target_mean)
        batch['reference_values']=torch.zeros_like(batch['values'])
        wind_observed=batch['mask'][:,:,-1,9:11]>0
        batch['reference_physical_wind']=torch.where(
            wind_observed,self.base.wind_mean.view(1,1,2),torch.zeros_like(batch['physical_wind']))
        return batch


def forward_v6(model,batch,need_attention=False):
    return model(batch['values'],batch['mask'],batch['donor_static'],batch['geometry'],
        batch['donor_padding_mask'],batch['target_static'],batch['time_features'],
        batch['donor_background'],batch['target_background'],batch['target_background_raw'],
        batch['donor_climatology_raw'],batch['reference_values'],batch['reference_physical_wind'],
        need_attention,batch['physical_wind'],batch['relation_scales'])


def ordinary_two_domain_step(model,s_batch,q_batch,optimizer,s_weights,q_weights,
                             slow_weight=1.0,amp_dtype=None):
    model.train();optimizer.zero_grad(set_to_none=True);device=next(model.parameters()).device
    with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_dtype is not None):
        s_loss,s_parts=v5_loss(forward_v6(model,s_batch),s_batch,s_weights,slow_weight)
        q_loss,q_parts=v5_loss(forward_v6(model,q_batch),q_batch,q_weights,slow_weight)
        loss=s_loss+q_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),CFG.gradient_clip_norm,error_if_nonfinite=True)
    optimizer.step()
    return {'support_loss':s_loss.detach(),'query_loss':q_loss.detach()}

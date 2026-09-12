"""Loss, context adapter and first-order MLDG step for V5."""
from __future__ import annotations
from collections import OrderedDict
import numpy as np
import torch
from torch.func import functional_call
from config import CFG
from v2_training import V2BatchAdapter


def balanced_partitions(stations,clusters,n=5,seed=20260913):
    stations=np.asarray(stations,int); groups=[[] for _ in range(n)]; rng=np.random.default_rng(seed)
    for cluster in sorted(np.unique(clusters[stations])):
        members=stations[clusters[stations]==cluster].copy(); rng.shuffle(members)
        local=np.zeros(n,int)
        for station in members:
            sizes=np.array([len(g) for g in groups]); choices=np.flatnonzero((sizes==sizes.min())&(local==local.min()))
            if not len(choices): choices=np.flatnonzero(sizes==sizes.min())
            chosen=int(choices[0]); groups[chosen].append(int(station)); local[chosen]+=1
    groups=[np.asarray(sorted(g),int) for g in groups]
    joined=np.concatenate(groups)
    if len(joined)!=len(stations) or len(np.unique(joined))!=len(stations) or set(joined)!=set(stations):
        raise RuntimeError('Meta partitions do not exactly cover training stations')
    if max(map(len,groups))-min(map(len,groups))>1: raise RuntimeError('Meta partitions unbalanced')
    return groups


class V5BatchAdapter:
    def __init__(self,fit,scaler,static,device,relation_scales,target_mean_labels):
        self.base=V2BatchAdapter(fit,scaler,static,device,relation_scales)
        self.target_means=torch.as_tensor(target_mean_labels,dtype=torch.float32,device=device)

    def prepare(self,batch,need_target_mean=True):
        batch=self.base.prepare(batch)
        donors=batch['donor_indices'].clamp_min(0)
        # Donors are always source stations, so these observed means are legal.
        donor_mu=self.base.observed[donors]
        # Never use a donor background that was fitted with the current
        # support target's mean. Raw donor means are legal because the target
        # itself is absent from donor_indices.
        batch['donor_climatology_raw']=donor_mu
        # Loss-side only. model_v5.forward has no target_mu argument.
        batch['target_mean_label']=self.target_means[batch['target_idx']]
        if need_target_mean and not torch.isfinite(batch['target_mean_label']).all():
            raise RuntimeError('V5 loss-side target mean missing')
        return batch


def forward_v5(model,batch,need_attention=False,params=None):
    args=(batch['values'],batch['mask'],batch['donor_static'],batch['geometry'],
          batch['donor_padding_mask'],batch['target_static'],batch['time_features'],
          batch['donor_background'],batch['target_background'],batch['target_background_raw'],
          batch['donor_climatology_raw'],need_attention,batch['physical_wind'],
          batch['relation_scales'])
    return model(*args) if params is None else functional_call(model,params,args)


def v5_loss(outputs,batch,station_weights,slow_weight=1.0):
    prediction,aux=outputs
    weights=station_weights[batch['target_idx']]
    slow_target=batch['target_mean_label']-batch['target_background_raw']
    final=torch.mean(weights*(prediction.float()-batch['label'].float()).square())
    slow=torch.mean(weights*(aux['slow_correction'].float()-slow_target.float()).square())
    return final+slow_weight*slow,{'final_mse':final.detach(),'slow_mse':slow.detach()}


def first_order_mldg_step(model,s_batch,q_batch,optimizer,s_weights,q_weights,alpha,
                          beta=1.0,slow_weight=1.0,amp_dtype=None,diagnostic=False,do_update=True):
    """gS(theta)+beta*gQ(theta-alpha*gS), one outer AdamW update."""
    model.train(); optimizer.zero_grad(set_to_none=True)
    named=OrderedDict(model.named_parameters()); parameters=tuple(named.values())
    with torch.autocast(device_type=next(model.parameters()).device.type,dtype=amp_dtype,
                        enabled=amp_dtype is not None):
        s_outputs=forward_v5(model,s_batch)
        s_loss,s_parts=v5_loss(s_outputs,s_batch,s_weights,slow_weight)
    g_s=torch.autograd.grad(s_loss,parameters,allow_unused=True)
    fast=OrderedDict((name,p if g is None else p-alpha*g.detach())
                     for (name,p),g in zip(named.items(),g_s))
    q_before=None
    if diagnostic:
        cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        with torch.no_grad(),torch.autocast(device_type=next(model.parameters()).device.type,dtype=amp_dtype,
                                            enabled=amp_dtype is not None):
            q_before=float(v5_loss(forward_v5(model,q_batch),q_batch,q_weights,slow_weight)[0])
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng)
    with torch.autocast(device_type=next(model.parameters()).device.type,dtype=amp_dtype,
                        enabled=amp_dtype is not None):
        q_outputs=forward_v5(model,q_batch,params=fast)
        q_loss,q_parts=v5_loss(q_outputs,q_batch,q_weights,slow_weight)
    g_q=torch.autograd.grad(q_loss,parameters,allow_unused=True)
    diag_values={}
    if diagnostic:
        gs_norm_sq=torch.stack([g.detach().float().square().sum() for g in g_s if g is not None]).sum()
        parameter_norm_sq=torch.stack([p.detach().float().square().sum() for p in parameters]).sum()
        pairs=[(gs,gq) for gs,gq in zip(g_s,g_q) if gs is not None and gq is not None]
        dot=torch.stack([gs.detach().float().mul(gq.detach().float()).sum() for gs,gq in pairs]).sum()
        gq_norm_sq=torch.stack([g.detach().float().square().sum() for g in g_q if g is not None]).sum()
        diag_values={
            'relative_inner_step':float((alpha*torch.sqrt(gs_norm_sq)/torch.sqrt(parameter_norm_sq).clamp_min(1e-12)).cpu()),
            'gradient_cosine':float((dot/torch.sqrt(gs_norm_sq*gq_norm_sq).clamp_min(1e-12)).cpu()),
            'query_loss_before_inner':q_before,
            'query_loss_ratio':float(q_loss.detach().cpu())/max(q_before,1e-12),
        }
    for (_,p),gs,gq in zip(named.items(),g_s,g_q):
        if gs is None and gq is None: continue
        p.grad=(torch.zeros_like(p) if gs is None else gs.detach())+beta*(torch.zeros_like(p) if gq is None else gq.detach())
    # A single fused check avoids one GPU synchronization per parameter.
    torch.nn.utils.clip_grad_norm_(parameters,CFG.gradient_clip_norm,error_if_nonfinite=True)
    if do_update: optimizer.step()
    else: optimizer.zero_grad(set_to_none=True)
    return {'support_loss':s_loss.detach(),'query_loss':q_loss.detach(),**diag_values}


def make_station_weights(targets,station_count,device):
    targets=np.asarray(targets,dtype='int64'); counts=np.bincount(targets,minlength=station_count)
    active=np.flatnonzero(counts); weights=np.zeros(station_count,dtype='float32')
    weights[active]=len(targets)/(len(active)*counts[active])
    return torch.as_tensor(weights,device=device)


def event_metrics(y,p,threshold=35.0):
    y=np.asarray(y);p=np.asarray(p);actual=y>=threshold;predicted=p>=threshold
    tp=int(np.sum(actual&predicted));fp=int(np.sum(~actual&predicted));fn=int(np.sum(actual&~predicted))
    precision=tp/(tp+fp) if tp+fp else float('nan');recall=tp/(tp+fn) if tp+fn else float('nan')
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else float('nan')
    return {'event_tp':tp,'event_fp':fp,'event_fn':fn,'event_precision':precision,'event_recall':recall,'event_f1':f1}

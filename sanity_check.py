from __future__ import annotations
import numpy as np
import torch


def audit_dataset(ds, max_samples=500):
    n=len(ds)
    idx=np.arange(n) if n<=max_samples else np.random.default_rng(123).choice(n,max_samples,replace=False)
    obs=np.zeros(ds.cfg.n_dynamic_channels); total=np.zeros_like(obs); dead=0; dtotal=0
    donor_counts=[]
    for i in idx:
        x=ds[int(i)]; m=x["mask"].numpy(); v=x["values"].numpy()
        if m.shape[1:] != (ds.cfg.history_hours, ds.cfg.n_dynamic_channels): raise RuntimeError("dynamic shape不是D×24×11")
        if not np.isin(m,(0.0,1.0)).all(): raise RuntimeError("dynamic mask不是binary")
        if not np.all(v[m==0] == 0.0): raise RuntimeError("missing dynamic placeholder不是標準化後的0")
        if x["donor_padding_mask"].any(): raise RuntimeError("未padding的單一sample卻標成key padding")
        obs+=m.sum(axis=(0,1)); total+=np.prod(m.shape[:2])
        dead+=int(x["donor_all_missing"].sum()); dtotal+=x["donor_all_missing"].numel(); donor_counts.append(m.shape[0])
    return {
        "dataset_rows":n,
        "target_pm25_truth_timestamps":int(ds.truth_rows),
        "history_eligible_truth_timestamps":int(ds.history_eligible_truth_rows),
        "prediction_to_truth_coverage":float(n/max(ds.truth_rows,1)),
        "sampled_donor_counts":sorted(set(donor_counts)),
        "channel_availability":{k:float(v) for k,v in zip(ds.cfg.dynamic_items,obs/np.maximum(total,1))},
        "fully_missing_real_donor_fraction":float(dead/max(dtotal,1)),
    }


def assert_finite(name,x):
    if not torch.isfinite(x).all(): raise RuntimeError(f"{name}含NaN/Inf")


@torch.no_grad()
def assert_tcn_causal(model, device):
    """Changing time t=23 must not change representations at t<=22."""
    was_training=model.training; model.eval()
    channels=model.cfg.n_dynamic_channels*2
    x=torch.randn(2,channels,model.cfg.history_hours,device=device)
    y1=model.tcn.blocks(x)
    x2=x.clone(); x2[:,:,-1]+=10.0
    y2=model.tcn.blocks(x2)
    earlier_delta=(y1[:,:,:-1]-y2[:,:,:-1]).abs().max().item()
    last_delta=(y1[:,:,-1]-y2[:,:,-1]).abs().max().item()
    if earlier_delta > 1e-6: raise RuntimeError(f"TCN不是causal，future perturbation影響較早輸出: {earlier_delta}")
    if was_training: model.train()
    return {"earlier_max_delta":earlier_delta,"last_step_delta":last_delta}


def smoke_forward(model,loader,device):
    batch=next(iter(loader))
    for k in ("values","mask","donor_static","geometry","target_static","time_features","label"): assert_finite(k,batch[k])
    if batch["values"].shape[-2:] != (model.cfg.history_hours,model.cfg.n_dynamic_channels): raise RuntimeError("batch history/channel shape錯誤")
    if not torch.all((batch["mask"]==0)|(batch["mask"]==1)): raise RuntimeError("batch mask不是binary")
    if not torch.all(batch["values"][batch["mask"]==0]==0): raise RuntimeError("batch missing placeholder不是0")
    if torch.any(batch["donor_padding_mask"].all(dim=1)): raise RuntimeError("某sample所有donor都被padding")
    batch={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
    model=model.to(device); causality=assert_tcn_causal(model,device); model.train(); model.zero_grad(set_to_none=True)
    pred,attn=model(batch["values"],batch["mask"],batch["donor_static"],batch["geometry"],batch["donor_padding_mask"],batch["target_static"],batch["time_features"],True)
    assert_finite("prediction",pred); loss=torch.nn.functional.mse_loss(pred,batch["label"]); assert_finite("loss",loss); loss.backward()
    required=("tcn.","donor_projection.","query_projection.","cross_attention.","head.")
    grad_norms={}
    for prefix in required:
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if not grads or any(g is None for g in grads): raise RuntimeError(f"{prefix}有參數未收到gradient")
        for g in grads: assert_finite(f"{prefix} gradient",g)
        norm=float(sum(g.detach().abs().sum().cpu() for g in grads))
        if norm == 0.0: raise RuntimeError(f"{prefix} gradient全為0")
        grad_norms[prefix.rstrip(".")]=norm
    assert_finite("attention",attn)
    if attn is not None and batch["donor_padding_mask"].any():
        masked=batch["donor_padding_mask"][:,None,None,:].expand_as(attn)
        if attn[masked].abs().max().item() > 1e-7: raise RuntimeError("attention在collate padding上不是0")
    peak=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=="cuda" else 0
    return {"values_shape":list(batch["values"].shape),"prediction_shape":list(pred.shape),"loss":float(loss.detach().cpu()),"peak_vram_mb":float(peak),"parameters":sum(p.numel() for p in model.parameters()),"causality":causality,"gradient_l1_by_module":grad_norms}

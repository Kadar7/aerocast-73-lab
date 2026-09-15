from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
from model_v7 import ContextHeldOutGradientOperator
from v7_training import fixed_hash_groups


def make_batch(b=2,d=12,t=24):
    torch.manual_seed(7)
    values=torch.randn(b,d,t,11);mask=torch.ones_like(values)
    static=torch.randn(b,d,49);target=torch.randn(b,49);calendar=torch.randn(b,t,6)
    edge_dd=torch.randn(b,d,d,19);edge_target=torch.randn(b,d,19)
    distance_dd=torch.rand(b,d,d)*300;distance_dd.diagonal(dim1=1,dim2=2).zero_()
    distance_target=torch.rand(b,d)*300
    groups=torch.tensor([[j%4 for j in range(d)]]*b)
    pad=torch.zeros(b,d,dtype=torch.bool);pm=torch.rand(b,d)*40;pm_mask=torch.ones(b,d,dtype=torch.bool)
    return [values,mask,static,target,calendar,edge_dd,edge_target,groups,pad,pm,pm_mask,distance_dd,distance_target,torch.tensor(100.)]


def main():
    groups=fixed_hash_groups(["1","2","17","84"])
    assert groups.shape==(4,) and ((groups>=0)&(groups<4)).all()
    model=ContextHeldOutGradientOperator(pm_mean=12.,pm_std=7.,dropout=0.).eval()
    batch=make_batch();pred,aux=model(*batch)
    assert pred.shape==(2,) and torch.isfinite(pred).all()
    assert torch.allclose(aux["candidate_weights"].sum((1,2)),torch.ones(2),atol=1e-6)
    permutation=torch.tensor([5,2,10,1,7,0,11,8,4,6,9,3])
    v,m,s,ta,cal,edd,et,g,pad,pm,pmm,dd,dt,rho=batch
    permuted=[v[:,permutation],m[:,permutation],s[:,permutation],ta,cal,
        edd[:,permutation][:,:,permutation],et[:,permutation],g[:,permutation],pad[:,permutation],
        pm[:,permutation],pmm[:,permutation],dd[:,permutation][:,:,permutation],dt[:,permutation],rho]
    pred2,_=model(*permuted)
    assert torch.allclose(pred,pred2,atol=2e-5,rtol=1e-5),(pred,pred2)
    model.train();model.zero_grad(set_to_none=True);prediction,_=model(*batch)
    prediction.sum().backward()
    for prefix in ("static.","point.","temporal.","graph.","query."):
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix)]
        assert grads and any(x is not None and torch.isfinite(x).all() and x.abs().sum()>0 for x in grads),prefix
    print({"status":"ok","prediction":pred.tolist(),"parameters":sum(p.numel() for p in model.parameters())})


if __name__=="__main__":main()

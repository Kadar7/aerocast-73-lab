import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from config import Config
from model_v6 import AnchoredFactorizedResidualKriging
from v2_training import BackgroundFit
from v6_training import V6BatchAdapter,forward_v6


def batch(batch_size=4,donors=3):
    torch.manual_seed(9);targets=torch.tensor([0,0,1,1])[:batch_size]
    return dict(values=torch.randn(batch_size,donors,24,11),mask=torch.ones(batch_size,donors,24,11),
        donor_static=torch.randn(batch_size,donors,49),geometry=torch.randn(batch_size,donors,3),
        donor_padding_mask=torch.zeros(batch_size,donors,dtype=torch.bool),target_static=torch.randn(batch_size,49),
        time_features=torch.randn(batch_size,4),donor_background=torch.randn(batch_size,donors),
        target_background=torch.randn(batch_size),target_background_raw=torch.randn(batch_size)*3+10,
        donor_climatology_raw=torch.randn(batch_size,donors)+10,physical_wind=torch.randn(batch_size,donors,2),
        relation_scales=torch.ones(4),target_idx=targets,label=torch.randn(batch_size)*3+10,
        target_mean_label=torch.randn(batch_size)+10)


def v6_batch():
    result=batch();result['reference_values']=torch.zeros_like(result['values'])
    result['reference_physical_wind']=torch.zeros_like(result['physical_wind'])
    return result


class V6AnchoredTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_actual_equals_reference_zero_anchor_train_and_eval(self):
        model=AnchoredFactorizedResidualKriging(cfg=Config(dropout=.2),anchored=True);b=v6_batch()
        b['values']=b['reference_values'].clone();b['physical_wind']=b['reference_physical_wind'].clone()
        for training in (True,False):
            model.train(training);_,aux=forward_v6(model,b)
            torch.testing.assert_close(aux['anomaly'],torch.zeros_like(aux['anomaly']),rtol=0,atol=0)

    def test_rng_after_anchor_equals_one_actual_forward(self):
        cfg=Config(dropout=.2);control=AnchoredFactorizedResidualKriging(cfg=cfg,anchored=False)
        anchored=AnchoredFactorizedResidualKriging(cfg=cfg,anchored=True);anchored.load_state_dict(control.state_dict())
        control.train();anchored.train();b=v6_batch()
        torch.manual_seed(2026);forward_v6(control,b);after_control=torch.get_rng_state()
        torch.manual_seed(2026);forward_v6(anchored,b);after_anchored=torch.get_rng_state()
        torch.testing.assert_close(after_anchored,after_control)

    def test_anchor_has_finite_shared_gradients(self):
        model=AnchoredFactorizedResidualKriging(cfg=Config(dropout=.1),anchored=True);model.train();b=v6_batch()
        prediction,aux=forward_v6(model,b);(prediction.square().mean()+aux['slow_correction'].square().mean()).backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients);self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        for prefix in ('tcn.','fast_head.'):
            selected=[p.grad for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
            self.assertTrue(selected);self.assertGreater(sum(float(g.abs().sum()) for g in selected),0.)

    def test_reference_wind_respects_missing_mask(self):
        n=4;fit=BackgroundFit(np.array([9,10,11,12],dtype='float32'),np.array([8,10,11,12],dtype='float32'),0.,10.,1.,.5,1.)
        mean=np.zeros(11,dtype='float32');mean[-2:]=[1.5,-2.]
        scaler=SimpleNamespace(dynamic_mean=mean,dynamic_std=np.ones(11,dtype='float32'))
        static=pd.DataFrame({'longitude':np.arange(n),'latitude':np.arange(n)})
        raw=v6_batch();raw['target_idx']=torch.zeros(4,dtype=torch.long);raw['donor_indices']=torch.tensor([[1,2,3]]*4)
        raw['mask'][:,:,-1,9]=0
        adapter=V6BatchAdapter(fit,scaler,static,torch.device('cpu'),np.ones(4,'float32'),np.array([8,10,11,12],'float32'))
        prepared=adapter.prepare(raw)
        self.assertTrue(torch.equal(prepared['reference_physical_wind'][...,0],torch.zeros_like(prepared['reference_physical_wind'][...,0])))
        torch.testing.assert_close(prepared['reference_physical_wind'][...,1],torch.full_like(prepared['reference_physical_wind'][...,1],-2.))

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 required')
    def test_cuda_bf16_anchor_rng_and_gradients(self):
        device=torch.device('cuda');cfg=Config(dropout=.2)
        control=AnchoredFactorizedResidualKriging(cfg=cfg,anchored=False).to(device).train()
        anchored=AnchoredFactorizedResidualKriging(cfg=cfg,anchored=True).to(device).train()
        anchored.load_state_dict(control.state_dict())
        b={key:(value.to(device) if torch.is_tensor(value) else value) for key,value in v6_batch().items()}
        torch.manual_seed(77);torch.cuda.manual_seed_all(77)
        with torch.autocast('cuda',dtype=torch.bfloat16): forward_v6(control,b)
        cpu_after=torch.get_rng_state();cuda_after=torch.cuda.get_rng_state(device)
        torch.manual_seed(77);torch.cuda.manual_seed_all(77)
        equal=dict(b);equal['values']=b['reference_values'];equal['physical_wind']=b['reference_physical_wind']
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction,aux=forward_v6(anchored,equal)
        torch.testing.assert_close(aux['anomaly'].float(),torch.zeros_like(aux['anomaly'].float()),rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),cpu_after);torch.testing.assert_close(torch.cuda.get_rng_state(device),cuda_after)
        anchored.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction,_=forward_v6(anchored,b)
        prediction.float().square().mean().backward()
        for prefix in ('tcn.','fast_head.'):
            selected=[p.grad for name,p in anchored.named_parameters() if name.startswith(prefix) and p.grad is not None]
            self.assertTrue(selected);self.assertTrue(all(torch.isfinite(g).all() for g in selected))
            self.assertGreater(sum(float(g.abs().sum()) for g in selected),0.)


if __name__=='__main__': unittest.main()

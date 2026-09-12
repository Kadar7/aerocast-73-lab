import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from config import CFG,Config
from model_v5 import FactorizedResidualKriging
from v2_training import BackgroundFit
from v5_training import (V5BatchAdapter,balanced_partitions,forward_v5,first_order_mldg_step,
                         make_station_weights,v5_loss,event_metrics)


def batch(batch=4,donors=3):
    torch.manual_seed(9)
    targets=torch.tensor([0,0,1,1])[:batch]
    return dict(values=torch.randn(batch,donors,24,11),mask=torch.ones(batch,donors,24,11),
        donor_static=torch.randn(batch,donors,49),geometry=torch.randn(batch,donors,3),
        donor_padding_mask=torch.zeros(batch,donors,dtype=torch.bool),target_static=torch.randn(batch,49),
        time_features=torch.randn(batch,4),donor_background=torch.randn(batch,donors),
        target_background=torch.randn(batch),target_background_raw=torch.randn(batch)*3+10,
        donor_climatology_raw=torch.randn(batch,donors)+10,physical_wind=torch.randn(batch,donors,2),
        relation_scales=torch.ones(4),target_idx=targets,label=torch.randn(batch)*3+10,
        target_mean_label=torch.randn(batch)+10)


class V5Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_factorization_and_target_mean_not_model_input(self):
        model=FactorizedResidualKriging().eval(); b=batch()
        with torch.no_grad():
            prediction,aux=forward_v5(model,b)
            torch.testing.assert_close(prediction,b['target_background_raw']+aux['slow_correction']+aux['anomaly'])
            changed=dict(b); changed['target_mean_label']=b['target_mean_label']+1000
            torch.testing.assert_close(prediction,forward_v5(model,changed)[0])

    def test_balanced_exact_partitions(self):
        stations=np.arange(50); clusters=np.repeat(np.arange(5),10)
        groups=balanced_partitions(stations,clusters)
        self.assertEqual([len(x) for x in groups],[10]*5)
        np.testing.assert_array_equal(np.sort(np.concatenate(groups)),stations)

    def test_station_time_shards_cover_exactly_once(self):
        stations=np.arange(50);groups=balanced_partitions(stations,np.repeat(np.arange(5),10))
        for epoch in range(5):
            seen=[]
            for cid,query in enumerate(groups):
                self.assertEqual(len(np.union1d(query,np.setdiff1d(stations,query))),50)
                shard=(cid+epoch)%5
                seen.extend((int(s),t) for s in stations for t in range(100) if t%5==shard)
            self.assertEqual(len(seen),5000);self.assertEqual(len(set(seen)),5000)

    def test_meta_step_does_not_leave_fast_weights(self):
        torch.manual_seed(3); model=FactorizedResidualKriging(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        before=copy.deepcopy(model.state_dict()); optimizer_before=copy.deepcopy(optimizer.state_dict())
        b=batch(); weights=make_station_weights(b['target_idx'].numpy(),73,torch.device('cpu'))
        result=first_order_mldg_step(model,b,b,optimizer,weights,weights,1e-4,
                                     diagnostic=True,do_update=False)
        for name,value in model.state_dict().items(): torch.testing.assert_close(value,before[name])
        self.assertEqual(optimizer.state_dict(),optimizer_before)
        self.assertTrue(np.isfinite(result['gradient_cosine']))
        self.assertGreater(result['relative_inner_step'],0)

    def test_alpha_zero_query_loss_replay(self):
        torch.manual_seed(4); model=FactorizedResidualKriging(); optimizer=torch.optim.AdamW(model.parameters(),lr=0.0)
        b=batch(); weights=make_station_weights(b['target_idx'].numpy(),73,torch.device('cpu'))
        result=first_order_mldg_step(model,b,b,optimizer,weights,weights,0.0,diagnostic=True)
        self.assertAlmostEqual(result['query_loss_ratio'],1.0,places=6)

    def test_alpha_zero_matches_two_loss_gradient(self):
        cfg=Config(dropout=0.0);torch.manual_seed(31)
        meta=FactorizedResidualKriging(cfg=cfg);direct=FactorizedResidualKriging(cfg=cfg)
        direct.load_state_dict(meta.state_dict());b=batch();w=make_station_weights(b['target_idx'].numpy(),73,torch.device('cpu'))
        om=torch.optim.SGD(meta.parameters(),lr=1e-4);od=torch.optim.SGD(direct.parameters(),lr=1e-4)
        first_order_mldg_step(meta,b,b,om,w,w,0.0)
        od.zero_grad();ls=v5_loss(forward_v5(direct,b),b,w)[0];lq=v5_loss(forward_v5(direct,b),b,w)[0]
        (ls+lq).backward();torch.nn.utils.clip_grad_norm_(direct.parameters(),CFG.gradient_clip_norm);od.step()
        for a,d in zip(meta.parameters(),direct.parameters()): torch.testing.assert_close(a,d,rtol=1e-5,atol=1e-6)

    def test_target_mean_is_loss_side_only_through_adapter(self):
        n=4;fit=BackgroundFit(np.array([9,10,11,12],dtype='float32'),np.array([8,10,11,12],dtype='float32'),0.,10.,1.,.5,1.)
        scaler=SimpleNamespace(dynamic_mean=np.zeros(11,dtype='float32'),dynamic_std=np.ones(11,dtype='float32'))
        static=pd.DataFrame({'longitude':np.arange(n),'latitude':np.arange(n)})
        means=np.array([8,10,11,12],dtype='float32');changed=means.copy();changed[0]=999
        raw=batch();raw['target_idx']=torch.zeros(4,dtype=torch.long);raw['donor_indices']=torch.tensor([[1,2,3]]*4)
        a=V5BatchAdapter(fit,scaler,static,torch.device('cpu'),np.ones(4,'float32'),means).prepare(dict(raw))
        b=V5BatchAdapter(fit,scaler,static,torch.device('cpu'),np.ones(4,'float32'),changed).prepare(dict(raw))
        for key in ('values','donor_climatology_raw','target_background_raw','physical_wind'):
            torch.testing.assert_close(a[key],b[key])
        self.assertFalse(torch.equal(a['target_mean_label'],b['target_mean_label']))
        model=FactorizedResidualKriging().eval()
        with torch.no_grad(): torch.testing.assert_close(forward_v5(model,a)[0],forward_v5(model,b)[0])

    def test_event_f1_zero_when_all_events_missed(self):
        result=event_metrics(np.array([40.,10.]),np.array([10.,10.]))
        self.assertEqual(result['event_recall'],0.0);self.assertEqual(result['event_f1'],0.0)

if __name__=='__main__': unittest.main()

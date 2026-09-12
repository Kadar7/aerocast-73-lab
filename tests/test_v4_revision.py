import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from background_crossfit import ridge_loo, ridge_predict, crossfit_background
from v2_training import fit_background, relation_scales, _v2_loss, v2_config_payload
from train_formal import make_station_sample_weights
from config import CFG
from model_v2 import PhysicsGuidedCrossAttention


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        torch.set_num_threads(2)

    def test_exact_loo(self):
        for n, p in [(8, 3), (9, 49)]:
            x = self.rng.normal(size=(n,p)); y = self.rng.normal(size=n)
            for lam in (.1, 1., 10., 100.):
                brute = []
                for i in range(n):
                    keep = np.arange(n) != i
                    brute.append(ridge_predict(x[keep], y[keep], x[[i]], lam)[0])
                np.testing.assert_allclose(ridge_loo(x,y,lam), brute, atol=1e-8)

    def test_target_label_exclusion(self):
        x = self.rng.normal(size=(9,49))
        d = np.abs(np.arange(9)[:,None]-np.arange(9)[None,:])*1000.
        source = np.arange(7)
        y = self.rng.normal(20,5,9)
        before = crossfit_background(x,y,d,source)
        for i in source:
            changed=y.copy(); changed[i]+=1000
            after=crossfit_background(x,changed,d,source)
            self.assertAlmostEqual(before[0][i],after[0][i],places=10)
            self.assertEqual(before[3][i],after[3][i])
        y[7:]=1e9
        np.testing.assert_array_equal(before[0],crossfit_background(x,y,d,source)[0])

    def test_background_training_time_and_units(self):
        x=self.rng.normal(size=(7,49)); d=np.abs(np.arange(7)[:,None]-np.arange(7)[None,:])*1000.
        dates=pd.DatetimeIndex([CFG.train_start, CFG.train_end, CFG.test_start])
        cube=self.rng.uniform(1,30,(3,7,len(CFG.aq_cube_items))).astype('float32')
        source=np.arange(5)
        before=fit_background(cube,dates,x,d,source)
        cube[2]=1e6; cube[:,5:]=1e7
        after=fit_background(cube,dates,x,d,source)
        np.testing.assert_array_equal(before.predicted,after.predicted)
        self.assertEqual((before.center,before.scale),(0.,10.))
        scaler=SimpleNamespace(dynamic_std=np.ones(11))
        scales=relation_scales(x,d,source,scaler)
        x[5:]=1e9; d[5:]=1e9; d[:,5:]=1e9
        np.testing.assert_array_equal(scales,relation_scales(x,d,source,scaler))

    def test_station_loss(self):
        targets=np.array([0,1,1,1]); dataset=SimpleNamespace(row_targets=targets)
        class Data:
            row_targets=targets
            def __len__(self): return 4
        weights=make_station_sample_weights(Data(),2,torch.device('cpu'))[torch.tensor(targets)]
        labels=torch.tensor([100.,1.,2.,3.]); predictions=torch.zeros(4)
        expected=(10000+(1+4+9)/3)/2
        self.assertAlmostEqual(_v2_loss(predictions,{},labels,weights,None).item(),expected,places=3)
        self.assertEqual(v2_config_payload()['donor_mask_probability'],0.)

    def test_bounded_signed_prior(self):
        cfg=SimpleNamespace(attention_dim=8,attention_heads=2,dropout=0.)
        model=PhysicsGuidedCrossAttention(cfg)
        self.assertTrue(torch.equal(model.relation_weights,torch.zeros_like(model.relation_weights)))
        with torch.no_grad():
            for layer in (model.q,model.k):
                layer.weight.zero_(); layer.bias.zero_()
            model.relation_weights.fill_(1e3)
        g=torch.tensor([[[100.,0.,1.],[-100.,0.,1.],[0.,0.,1.]]])
        args=[torch.zeros(1,8),torch.randn(1,3,8),g,torch.zeros(1,3,24,11),torch.ones(1,3,24,11),torch.tensor([[False,False,True]]),torch.zeros(1,3)]
        _,attention=model(*args,need_weights=True,physical_wind=torch.zeros(1,3,2),relation_scales=torch.ones(4))
        self.assertTrue(torch.isfinite(attention).all())
        self.assertTrue((attention[:,:,2]==0).all())
        self.assertLessEqual(float((attention[:,:,0]/attention[:,:,1]).max().detach()),float(np.exp(2))+1e-5)
        attention[:,:,0].sum().backward()
        self.assertTrue(torch.isfinite(model.relation_weights.grad).all())

if __name__=='__main__': unittest.main()

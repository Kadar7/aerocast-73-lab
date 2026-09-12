import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
import torch
from diagnose_v4_complete import shuffled_dynamic,prediction_rows
from model_v2 import TCNTargetCrossAttentionV2
from v2_training import _forward

class DiagnosisTests(unittest.TestCase):
    def batch(self):
        torch.manual_seed(12)
        return dict(values=torch.randn(4,3,24,11),mask=torch.ones(4,3,24,11),
            target_idx=torch.zeros(4,dtype=torch.long),donor_indices=torch.tensor([[1,2,3]]*4),
            donor_static=torch.randn(4,3,49),target_static=torch.randn(4,49),
            geometry=torch.randn(4,3,3),time_features=torch.randn(4,4),
            donor_padding_mask=torch.zeros(4,3,dtype=torch.bool),donor_background=torch.randn(4,3),
            target_background=torch.ones(4),target_background_raw=torch.ones(4)*10,
            physical_wind=torch.randn(4,3,2),relation_scales=torch.ones(4))

    def test_shuffle_preserves_query_and_other_channels(self):
        b=self.batch(); original=b['values'].clone(); perm=torch.tensor([3,0,1,2])
        p=shuffled_dynamic(b,perm,True)
        torch.testing.assert_close(b['values'],original)
        torch.testing.assert_close(p['values'][...,7],original[perm,...,7])
        torch.testing.assert_close(p['values'][...,0],original[...,0])
        self.assertIs(p['time_features'],b['time_features'])
        a=shuffled_dynamic(b,perm)
        torch.testing.assert_close(a['physical_wind'],b['physical_wind'][perm])

    def test_reject_cross_target(self):
        b=self.batch(); b['target_idx'][0]=1
        with self.assertRaises(ValueError): shuffled_dynamic(b,torch.tensor([3,0,1,2]))

    def test_model_inference_and_residual_identity(self):
        torch.set_num_threads(2)
        model=TCNTargetCrossAttentionV2().eval()
        with torch.inference_mode():
            for b in (self.batch(),shuffled_dynamic(self.batch(),torch.tensor([3,0,1,2]))):
                p,aux=_forward(model,b)
                self.assertTrue(torch.isfinite(p).all())
                torch.testing.assert_close(p,b['target_background_raw']+aux['residual'])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_station_summary(self):
        rows=prediction_rows(np.array([1.,3.]),np.array([2.,4.]),np.array([0,0]),np.array([4,5]),
                             pd.DataFrame({'siteid':['1'],'sitename':['test']}),np.array([2.]),0,9,'training')
        self.assertEqual(rows[0]['final_bias'],1.)
        self.assertEqual(rows[0]['rmse'],1.)
        self.assertEqual(rows[0]['abs_bias_reduction'],-1.)

if __name__=='__main__': unittest.main()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from v2_training import BackgroundFit, V2BatchAdapter, _forward, _v2_loss, train_epoch_v2, validate_epoch_v2
from train_formal import make_grad_scaler, amp_dtype_for
from model_v2 import TCNTargetCrossAttentionV2

class RevisionTest(unittest.TestCase):
    def test_pipeline_and_backward(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        b,d=4,5
        fit=BackgroundFit(np.ones(8,dtype='float32')*12,np.ones(8,dtype='float32')*10,10,2,1,.5,1)
        scaler=SimpleNamespace(dynamic_mean=np.array([0]*9+[2,-3],dtype='float32'),dynamic_std=np.ones(11,dtype='float32')*2)
        adapter=V2BatchAdapter(fit,scaler,pd.DataFrame({'longitude':[121]*8,'latitude':[24]*8}),torch.device('cpu'))
        batch={'values':torch.randn(b,d,24,11),'mask':torch.ones(b,d,24,11),'donor_indices':torch.arange(d).repeat(b,1),'target_idx':torch.full((b,),7),'time_idx':torch.arange(b),'donor_static':torch.randn(b,d,49),'target_static':torch.randn(b,49),'geometry':torch.randn(b,d,3),'donor_padding_mask':torch.zeros(b,d,dtype=torch.bool),'time_features':torch.randn(b,4),'label':torch.tensor([10.,20.,35.,50.])}
        batch['values'][:,:,-1,9]=-1
        batch['values'][:,:,-1,10]=1.5
        prepared=adapter.prepare(batch)
        self.assertTrue(torch.equal(prepared['physical_wind'],torch.zeros(b,d,2)))
        masked=prepared
        self.assertTrue((~masked['donor_padding_mask']).any(dim=1).all())
        self.assertFalse(masked['donor_padding_mask'].any())
        for key in ['label','target_static','geometry','values','mask']:
            self.assertTrue(torch.equal(masked[key],prepared[key]),key)
        model=TCNTargetCrossAttentionV2()
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
        for epoch in range(2):
            optimizer.zero_grad()
            prediction,aux=_forward(model,masked,True)
            loss=_v2_loss(prediction,aux,masked['label'],torch.ones(b),None)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for prefix in ['tcn.','cross_attention.','residual_head.']:
                grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(any(g.abs().sum()>0 for g in grads))
            self.assertTrue(torch.all(aux['attention'].masked_select(masked['donor_padding_mask'][:,None,:])==0))
            optimizer.step()
        # Exercise the actual training/validation loops, including CUDA AMP when available.
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model=model.to(device)
        adapter=V2BatchAdapter(fit,scaler,pd.DataFrame({'longitude':[121]*8,'latitude':[24]*8}),device)
        class Loader:
            dataset=range(b)
            def __len__(self): return 1
            def __iter__(self): yield batch
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
        grad_scaler=make_grad_scaler(amp_dtype_for(device)==torch.float16)
        static=pd.DataFrame({'siteid':[str(i) for i in range(8)],'sitename':[str(i) for i in range(8)]})
        timestamps=pd.date_range('2024-07-01',periods=b,freq='h')
        for epoch in (1,2):
            loss=train_epoch_v2(model,Loader(),optimizer,grad_scaler,device,epoch,adapter,None,station_sample_weights=torch.ones(8,device=device))
            metrics,stations,predictions=validate_epoch_v2(model,Loader(),device,static,timestamps,epoch,adapter)
            self.assertTrue(np.isfinite(loss) and np.isfinite(metrics['rmse']))
            self.assertEqual(len(predictions),b)
        print(f'Actual loop smoke: device={device}, AMP={amp_dtype_for(device)}, epochs=2, batches/epoch=1')

if __name__=='__main__': unittest.main()

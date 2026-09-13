import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
from diagnose_v5_factorization import station_decomposition,summarize


class FactorizationDiagnosisTests(unittest.TestCase):
    def test_exact_stage_decomposition(self):
        y=np.array([9.,11.]);background=np.array([13.,13.]);slow=np.array([10.,10.]);fast=np.array([0.,0.])
        result=station_decomposition(y,background,slow,slow,fast)
        self.assertAlmostEqual(result['background_bias'],3.)
        self.assertAlmostEqual(result['slow_bias'],0.)
        self.assertAlmostEqual(result['slow_abs_bias_reduction_vs_background'],3.)
        self.assertFalse(result['fast_worsened_slow_bias'])

    def test_summary_routes_fast_failure(self):
        frame=pd.DataFrame([dict(fold=0,method='control',background_rmse=5.,slow_rmse=4.,final_rmse=5.,
            background_bias=2.,slow_bias=.5,final_bias=1.5,slow_abs_bias_reduction_vs_background=1.5,
            final_abs_bias_reduction_vs_slow=-1.,fast_worsened_slow_bias=True)])
        result=summarize(frame).iloc[0]
        self.assertEqual(result.diagnostic_branch,'fast_mean_abs_bias_increased_after_slow')


if __name__=='__main__': unittest.main()

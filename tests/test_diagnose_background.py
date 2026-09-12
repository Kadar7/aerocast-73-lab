import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from diagnose_v4_background import decompose

class DecompositionTest(unittest.TestCase):
    def test_offset(self):
        m=decompose([1,2,3],[4,5,6],10)
        self.assertEqual(m['background_bias'],8)
        self.assertEqual(m['mean_dl_correction'],-5)
        self.assertEqual(m['final_bias'],3)
        self.assertEqual(m['centered_rmse_diagnostic'],0)
        self.assertEqual(m['abs_bias_reduction'],5)
    def test_identity(self):
        m=decompose([2,8,4],[3,6,8],3)
        self.assertAlmostEqual(m['rmse']**2,m['final_bias']**2+m['centered_rmse_diagnostic']**2)
        self.assertAlmostEqual(m['background_bias']+m['mean_dl_correction'],m['final_bias'])
    def test_invalid(self):
        with self.assertRaises(ValueError): decompose([1,np.nan],[1,2],1)

if __name__=='__main__': unittest.main()

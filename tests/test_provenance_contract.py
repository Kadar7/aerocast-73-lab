from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CFG
from train_refit72_outer_epoch_ensemble import run_provenance


class ProvenanceContractTest(unittest.TestCase):
    def test_run_provenance_records_the_fixed_information_contract(self):
        original_static = CFG.static_path
        original_cluster = CFG.cluster_path
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                CFG.static_path = root / 'static.csv'
                CFG.cluster_path = root / 'clusters.csv'
                CFG.static_path.write_text('siteid,sitename,x\n1,A,0\n', encoding='utf-8')
                CFG.cluster_path.write_text('siteid,kmeans_cluster\n1,0\n', encoding='utf-8')
                result = run_provenance([f'x{index}' for index in range(49)], 'v1')
        finally:
            CFG.static_path = original_static
            CFG.cluster_path = original_cluster

        self.assertEqual(result['model_version'], 'v1')
        self.assertEqual(result['dynamic_channel_count'], 11)
        self.assertEqual(result['dynamic_channels'], list(CFG.dynamic_items))
        self.assertFalse(result['raw_wind_enters_model'])
        self.assertFalse(result['target_dynamic_history_used'])
        self.assertEqual(result['static_feature_count'], 49)
        self.assertFalse(result['static_pca'])
        self.assertFalse(result['missing_handling']['dynamic_imputation'])
        self.assertTrue(result['missing_handling']['binary_mask_same_shape'])
        self.assertEqual(result['event_threshold'], 35.0)
        self.assertEqual(len(result['static_sha256']), 64)
        self.assertEqual(len(result['cluster_sha256']), 64)


if __name__ == '__main__':
    unittest.main()

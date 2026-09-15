import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_v8 import SASFO
from v8_sampling import BalancedStationTimeSchedule


def make_batch(batch_size=3, donors=6):
    generator = torch.Generator().manual_seed(20260916)
    values = torch.randn(batch_size, donors, 24, 11, generator=generator)
    mask = torch.ones_like(values)
    geometry = torch.randn(batch_size, donors, 3, generator=generator)
    geometry[..., 0] = geometry[..., 0].abs()
    return {
        "values": values,
        "mask": mask,
        "donor_static": torch.randn(batch_size, donors, 49, generator=generator),
        "geometry": geometry,
        "donor_padding_mask": torch.zeros(batch_size, donors, dtype=torch.bool),
        "target_static": torch.randn(batch_size, 49, generator=generator),
        "calendar": torch.randn(batch_size, 24, 6, generator=generator),
        "current_pm_physical": torch.rand(batch_size, donors, generator=generator) * 25 + 2,
        "train_pm_mean": 12.5,
        "train_pm_std": 5.0,
    }


class SASFOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_donor_permutation_invariance(self):
        model = SASFO(dropout=0.0).eval()
        batch = make_batch()
        with torch.no_grad():
            expected, _ = model(**batch)
            order = torch.tensor([4, 1, 5, 0, 3, 2])
            permuted = dict(batch)
            for key in ("values", "mask", "donor_static", "geometry", "donor_padding_mask", "current_pm_physical"):
                permuted[key] = batch[key][:, order]
            actual, _ = model(**permuted)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_skip_sum_and_structural_l1_bound(self):
        model = SASFO(dropout=0.0).eval()
        _, aux = model(**make_batch())
        torch.testing.assert_close(aux["skip_weight_sum"], torch.ones(3), rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.all(aux["skip_l1"] <= 3.0 + 1e-6))
        torch.testing.assert_close(
            aux["skip_zero_sum_direction"].sum(dim=1), torch.zeros(3), rtol=0, atol=1e-6
        )

    def test_constant_current_field_affine_skip(self):
        model = SASFO(dropout=0.0).eval()
        batch = make_batch()
        batch["current_pm_physical"].fill_(17.25)
        _, aux = model(**batch)
        torch.testing.assert_close(aux["affine_base"], torch.full((3,), 17.25), rtol=1e-6, atol=1e-5)

    def test_zero_and_one_anchor_rules(self):
        model = SASFO(dropout=0.0).eval()
        batch = make_batch(batch_size=2)
        batch["mask"][0, :, -1, 7] = 0
        batch["current_pm_physical"][0, :] = torch.nan
        batch["mask"][1, 1:, -1, 7] = 0
        batch["current_pm_physical"][1, 1:] = torch.nan
        prediction, aux = model(**batch)
        self.assertFalse(bool(aux["has_anchor"][0]))
        self.assertEqual(int(aux["anchor_count"][0]), 0)
        self.assertEqual(float(prediction[0].detach()), 12.5)
        self.assertEqual(int(aux["anchor_count"][1]), 1)
        torch.testing.assert_close(aux["skip_weights"][1, 0], torch.tensor(1.0), rtol=0, atol=1e-6)
        torch.testing.assert_close(aux["skip_l1"][1], torch.tensor(1.0), rtol=0, atol=1e-6)

    def test_forward_backward_gradients_are_finite(self):
        model = SASFO(dropout=0.0).train()
        prediction, _ = model(**make_batch(batch_size=4))
        target = torch.tensor([5.0, 8.0, 13.0, 21.0])
        ((prediction - target).square().mean()).backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        for prefix in ("dynamic_encoder", "field_logits", "temporal", "static_encoder", "correction_head"):
            selected = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            self.assertTrue(selected, prefix)
            self.assertGreater(sum(float(g.abs().sum()) for g in selected), 0.0, prefix)


class BalancedScheduleTests(unittest.TestCase):
    def test_epoch_counts_and_time_cycle(self):
        valid = {station: np.arange(128) + station * 1000 for station in range(8)}
        schedule = BalancedStationTimeSchedule(valid, seed=71)
        station_samples = {station: 0 for station in valid}
        batches = list(schedule.iter_epoch())
        self.assertEqual(len(batches), 256)
        for batch in batches:
            self.assertEqual(batch.station_indices.shape, (256,))
            self.assertEqual(batch.time_indices.shape, (256,))
            self.assertEqual(len(np.unique(batch.stations)), 4)
            for station in batch.stations:
                station_samples[int(station)] += 64
        self.assertEqual(set(station_samples.values()), {8192})
        self.assertEqual(schedule.epoch, 1)
        self.assertEqual(schedule.station_round, 0)
        self.assertEqual(schedule.batch_in_round, 0)

    def test_protocol_step_counts(self):
        valid60 = {station: np.arange(100) for station in range(60)}
        valid72 = {station: np.arange(100) for station in range(72)}
        self.assertEqual(len(BalancedStationTimeSchedule(valid60)), 1920)
        self.assertEqual(len(BalancedStationTimeSchedule(valid72)), 2304)

    def test_exact_resume(self):
        valid = {station: np.arange(173) + station * 1000 for station in range(8)}
        original = BalancedStationTimeSchedule(valid, seed=999)
        for _ in range(19):
            original.next_batch()
        state = original.state_dict()
        restored = BalancedStationTimeSchedule(valid, seed=1)
        restored.load_state_dict(state)
        for _ in range(25):
            left = original.next_batch()
            right = restored.next_batch()
            self.assertEqual(
                (left.epoch, left.station_round, left.batch_in_round, left.batch_index),
                (right.epoch, right.station_round, right.batch_in_round, right.batch_index),
            )
            np.testing.assert_array_equal(left.stations, right.stations)
            np.testing.assert_array_equal(left.times, right.times)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_v8_pdc import PooledDonorCorrection


class V8PDCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.batch, self.donors = 3, 5
        values = torch.randn(self.batch, self.donors, 24, 11)
        mask = torch.ones_like(values)
        self.inputs = {
            "values": values,
            "mask": mask,
            "donor_static": torch.randn(self.batch, self.donors, 49),
            "geometry": torch.randn(self.batch, self.donors, 3),
            "donor_padding_mask": torch.zeros(self.batch, self.donors, dtype=torch.bool),
            "target_static": torch.randn(self.batch, 49),
            "calendar": torch.randn(self.batch, 24, 6),
            "current_pm_physical": torch.rand(self.batch, self.donors) * 20 + 2,
            "train_pm_mean": 11.0,
            "train_pm_std": 5.0,
        }

    def model(self):
        result = PooledDonorCorrection(dropout=0.0)
        result.eval()
        return result

    def test_zero_initialization_and_distance_prior(self):
        model = self.model()
        prediction, aux = model(**self.inputs, output_mode="treatment")
        self.assertTrue(torch.equal(aux["delta"], torch.zeros_like(aux["delta"])))
        expected = torch.softmax(-2.0 * self.inputs["geometry"][..., 0], dim=1)
        self.assertTrue(torch.allclose(aux["weights"], expected, atol=1e-6))
        self.assertTrue(torch.allclose(prediction, (expected * self.inputs["current_pm_physical"]).sum(1), atol=1e-5))

    def test_current_pm_mask_and_exact_fallback(self):
        model = self.model()
        data = {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in self.inputs.items()}
        pm = model.pm_channel_index
        data["mask"][0, :, -1, pm] = 0
        data["current_pm_physical"][0] = torch.nan
        data["mask"][1, 1:, -1, pm] = 0
        data["current_pm_physical"][1, 1:] = torch.nan
        prediction, aux = model(**data, output_mode="treatment")
        self.assertEqual(int(aux["anchor_count"][0]), 0)
        self.assertEqual(float(prediction[0].detach()), 11.0)
        self.assertTrue(torch.equal(aux["weights"][0], torch.zeros_like(aux["weights"][0])))
        self.assertEqual(int(aux["anchor_count"][1]), 1)
        self.assertAlmostEqual(float(aux["weights"][1, 0].detach()), 1.0, places=6)
        self.assertAlmostEqual(float(prediction[1].detach()), float(data["current_pm_physical"][1, 0]), places=5)

    def test_control_treatment_only_final_equation(self):
        model = self.model()
        treatment, ta = model(**self.inputs, output_mode="treatment")
        control, ca = model(**self.inputs, output_mode="control")
        for key in ("delta", "weights", "candidate_valid", "correction_component"):
            self.assertTrue(torch.equal(ta[key], ca[key]))
        expected_difference = ta["reference_component"] - 11.0
        self.assertTrue(torch.allclose(treatment - control, expected_difference, atol=1e-6))
        self.assertEqual(set(model.state_dict()), set(self.model().state_dict()))

    def test_constant_field_exact_at_initialization(self):
        model = self.model()
        data = dict(self.inputs)
        data["current_pm_physical"] = torch.full((self.batch, self.donors), 7.25)
        prediction, aux = model(**data, output_mode="treatment")
        self.assertTrue(torch.allclose(aux["weight_sum"], torch.ones(self.batch), atol=1e-6))
        self.assertTrue(torch.allclose(prediction, torch.full((self.batch,), 7.25), atol=1e-5))

    def test_finite_backward_both_arms(self):
        for mode in ("control", "treatment"):
            model = PooledDonorCorrection(dropout=0.0, output_mode=mode)
            model.train()
            prediction, _ = model(**self.inputs)
            loss = (prediction - torch.tensor([9.0, 10.0, 11.0])).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            last = model.candidate_correction[-1]
            self.assertIsNotNone(last.weight.grad)
            self.assertTrue(bool((last.weight.grad != 0).any()))


if __name__ == "__main__":
    unittest.main()

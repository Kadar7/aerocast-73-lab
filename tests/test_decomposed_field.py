import pathlib
import sys
import unittest

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model_decomposed_field import DecomposedFieldNowcaster, decomposed_loss


class DecomposedFieldTest(unittest.TestCase):
    def batch(self, batch=8, donors=7):
        torch.manual_seed(9)
        values = torch.randn(batch, donors, 24, 11)
        mask = (torch.rand_like(values) > 0.15).float()
        values = values * mask
        current_pm = torch.rand(batch, donors) * 30 + 2
        current_pm[mask[:, :, -1, 7] == 0] = float("nan")
        return dict(
            values=values,
            mask=mask,
            donor_static=torch.randn(batch, donors, 49),
            geometry=torch.cat([
                torch.rand(batch, donors, 1) * 5,
                torch.randn(batch, donors, 2),
            ], dim=-1),
            donor_padding_mask=torch.zeros(batch, donors, dtype=torch.bool),
            target_static=torch.randn(batch, 49),
            calendar=torch.randn(batch, 24, 6),
            current_pm_physical=current_pm,
            train_pm_mean=12.0,
        )

    def test_components_sum_and_field_is_convex(self):
        model = DecomposedFieldNowcaster(dropout=0.0)
        batch = self.batch()
        prediction, aux = model(**batch)
        expected = aux["spatial_field"] + aux["static_offset"] + aux["anomaly"]
        torch.testing.assert_close(prediction, expected)
        self.assertTrue(bool(aux["field_inside_donor_range"].all()))
        torch.testing.assert_close(
            aux["field_gate"].sum(1), torch.ones(prediction.shape[0]), atol=1e-6, rtol=1e-6
        )

    def test_permutation_invariance(self):
        model = DecomposedFieldNowcaster(dropout=0.0).eval()
        batch = self.batch()
        first, _ = model(**batch)
        order = torch.randperm(batch["values"].shape[1])
        changed = dict(batch)
        for key in ("values", "mask", "donor_static", "geometry", "donor_padding_mask", "current_pm_physical"):
            changed[key] = batch[key][:, order]
        second, _ = model(**changed)
        torch.testing.assert_close(first, second, atol=2e-5, rtol=2e-5)

    def test_target_static_changes_offset_not_fixed_bases(self):
        model = DecomposedFieldNowcaster(dropout=0.0).eval()
        batch = self.batch()
        _, first = model(**batch)
        changed = dict(batch)
        changed["target_static"] = batch["target_static"] + 3
        _, second = model(**changed)
        torch.testing.assert_close(first["field_bases"], second["field_bases"])

    def test_loss_has_finite_gradients_for_all_three_parts(self):
        model = DecomposedFieldNowcaster(dropout=0.0)
        batch = self.batch(batch=8)
        prediction, aux = model(**batch)
        labels = torch.rand(8) * 30
        target = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
        loss, parts = decomposed_loss(prediction, aux, labels, target, 8.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
        for prefix in ("field_gate", "offset_head", "dynamic_encoder", "temporal", "anomaly_head"):
            gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertTrue(gradients, prefix)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), prefix)

    def test_anchorless_fallback_is_finite(self):
        model = DecomposedFieldNowcaster(dropout=0.0).eval()
        batch = self.batch()
        batch["mask"][:, :, -1, 7] = 0
        batch["current_pm_physical"][:] = float("nan")
        prediction, aux = model(**batch)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertFalse(bool(aux["has_anchor"].any()))


if __name__ == "__main__":
    unittest.main()

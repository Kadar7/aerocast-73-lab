from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_v8_pdc_paired_pilot import (
    FROZEN_V8_PATH,
    FROZEN_V8_SHA256,
    LOCAL_CHECKPOINT_EVERY_STEPS,
    MAX_EPOCHS,
    REVISION,
    checkpoint_phase_after_batch,
    no_new_negative_station_ids,
    load_frozen_v8_baseline,
    phase_action,
    project_complete_cost,
    select_resume_payload,
    sha256_file,
    standardized_mse,
    validate_existing_manifest,
    validate_protocol_fingerprint,
)


def payload(fingerprint: str, step: int, phase: str = "training", history_length: int = 0) -> dict:
    return {
        "revision": REVISION,
        "protocol_fingerprint": fingerprint,
        "fold": 0,
        "arm": "control",
        "phase": phase,
        "global_step": step,
        "history": [{} for _ in range(history_length)],
    }


class ProtocolTests(unittest.TestCase):
    def test_frozen_v8_artifact_hash(self):
        self.assertEqual(sha256_file(FROZEN_V8_PATH), FROZEN_V8_SHA256)

    def test_frozen_v8_station_ids_and_negative_sets(self):
        baseline = pd.read_csv(FROZEN_V8_PATH, dtype={"siteid": str})
        static = pd.DataFrame({"siteid": [f"x{i}" for i in range(73)],
                               "sitename": [f"s{i}" for i in range(73)]})
        for feature in range(49):
            static[f"f{feature}"] = float(feature)
        for row in baseline.itertuples():
            static.loc[int(row.station_index), ["siteid", "sitename"]] = [str(row.siteid), str(row.sitename)]
        outer = 13
        static.loc[outer, ["siteid", "sitename"]] = ["17", "桃園"]
        splits = []
        all_indices = np.arange(73)
        for fold in range(6):
            if fold in (0, 3):
                validation = baseline.loc[baseline.fold == fold, "station_index"].to_numpy(int)
            else:
                validation = np.array([], dtype=int)
            training = np.setdiff1d(all_indices, np.r_[outer, validation])
            splits.append((training, validation))
        coordinate = np.arange(73, dtype=float)
        distance = np.abs(coordinate[:, None] - coordinate[None, :]) * 1000.0
        frozen = load_frozen_v8_baseline(static, splits, distance, outer)
        self.assertEqual(frozen["folds"]["0"]["negative_r2_station_ids"], [56])
        self.assertEqual(frozen["folds"]["3"]["negative_r2_station_ids"], [])

    def test_protocol_mismatch_is_fatal_for_payload_and_manifest(self):
        with self.assertRaises(RuntimeError):
            validate_protocol_fingerprint({"protocol_fingerprint": "old"}, "new", "unit")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol_manifest.json"
            path.write_text(json.dumps({"protocol_fingerprint": "old"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_existing_manifest(path, "new")

    def test_resume_checks_both_and_selects_highest_global_step(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "local.pt"
            drive = Path(directory) / "drive.pt"
            torch.save(payload("locked", 100), local)
            torch.save(payload("locked", 180, "validation_pending"), drive)
            selected, selected_payload = select_resume_payload(
                local, drive, "locked", 0, "control"
            )
            self.assertEqual(selected, drive)
            self.assertEqual(selected_payload["global_step"], 180)
            torch.save(payload("wrong", 200), local)
            with self.assertRaises(RuntimeError):
                select_resume_payload(local, drive, "locked", 0, "control")

    def test_resume_same_step_prefers_completed_validation_history(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "local.pt"
            drive = Path(directory) / "drive.pt"
            torch.save(payload("locked", 1920, "training", history_length=1), local)
            torch.save(payload("locked", 1920, "validation_pending", history_length=0), drive)
            selected, selected_payload = select_resume_payload(
                local, drive, "locked", 0, "control"
            )
            self.assertEqual(selected, local)
            self.assertEqual(len(selected_payload["history"]), 1)


class PhaseTests(unittest.TestCase):
    def test_pending_validation_and_completed_do_not_train(self):
        self.assertEqual(phase_action("validation_pending", 3, 0), "validate")
        self.assertEqual(phase_action("completed", 8, 5), "finalize")
        self.assertEqual(phase_action("training", 8, 5), "finalize")
        self.assertEqual(phase_action("training", 7, 5), "train")

    def test_periodic_checkpoint_on_last_batch_is_validation_pending(self):
        self.assertEqual(checkpoint_phase_after_batch(4, 3), "validation_pending")
        self.assertEqual(checkpoint_phase_after_batch(3, 3), "training")


class GateAndCostTests(unittest.TestCase):
    def test_standardized_mse(self):
        prediction = torch.tensor([3.0, 7.0])
        label = torch.tensor([1.0, 3.0])
        self.assertAlmostEqual(float(standardized_mse(prediction, label, 2.0)), 2.5)

    def test_negative_station_gate_uses_ids_not_counts(self):
        frozen = {56}
        self.assertTrue(no_new_negative_station_ids(set(), frozen))
        self.assertTrue(no_new_negative_station_ids({56}, frozen))
        self.assertFalse(no_new_negative_station_ids({57}, frozen))
        self.assertFalse(no_new_negative_station_ids({56, 57}, frozen))

    def test_complete_cost_counts(self):
        result = project_complete_cost(
            train_step_seconds=1.0,
            validation_batch_seconds=2.0,
            checkpoint_local_seconds=3.0,
            checkpoint_drive_seconds=5.0,
            setup_seconds=7.0,
            benchmark_seconds=11.0,
            train_steps_per_epoch=1920,
            validation_batches=168,
        )
        runs, epochs = 4, MAX_EPOCHS
        periodic = (1920 + LOCAL_CHECKPOINT_EVERY_STEPS - 1) // LOCAL_CHECKPOINT_EVERY_STEPS
        self.assertEqual(result["train_steps"], runs * epochs * 1920)
        self.assertEqual(result["validation_passes"], runs * (epochs + 1))
        self.assertEqual(result["validation_batches"], runs * (epochs + 1) * 168)
        self.assertEqual(result["periodic_local_writes_per_epoch"], periodic)
        self.assertEqual(result["local_checkpoint_writes"], runs * epochs * (periodic + 3) + runs)
        self.assertEqual(result["drive_checkpoint_writes"], runs * epochs * 3 + runs)


if __name__ == "__main__":
    unittest.main()

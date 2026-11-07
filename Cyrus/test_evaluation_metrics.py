import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_clean_matrix import prediction_metrics


class PredictionMetricTests(unittest.TestCase):
    def test_frame_change_baselines_and_brier_scores(self):
        rows = [
            {
                "sig": [1, None, None],
                "action_effect_delta": {"changed_cells": 2},
                "action_effect_validation": {"frame_change_prediction": {
                    "predicted_probability": 0.8,
                    "predicted_changed": True,
                }},
            },
            {
                "sig": [6, None, None],
                "action_effect_delta": {"changed_cells": 0},
                "action_effect_validation": {"frame_change_prediction": {
                    "predicted_probability": 0.2,
                    "predicted_changed": False,
                }},
            },
            {
                "sig": [6, None, None],
                "action_effect_delta": {"changed_cells": 0},
                "action_effect_validation": {"frame_change_prediction": {
                    "predicted_probability": 0.6,
                    "predicted_changed": False,
                }},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "system2.jsonl"
            log_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            metrics = prediction_metrics(log_path)

        self.assertEqual(metrics["rows"], 3)
        self.assertEqual(metrics["accuracy_all_available_rows"]
                         ["last_outcome_same_signature"]["accuracy"], 1.0)
        self.assertEqual(metrics["accuracy_all_available_rows"]
                         ["full_run_majority_class_in_sample"]["accuracy"],
                         2 / 3)
        self.assertEqual(metrics["brier"]["learned_probability"]["n"], 3)
        self.assertAlmostEqual(metrics["brier"]["learned_probability"]["brier"],
                               0.44 / 3)
        self.assertEqual(sum(row["n"] for row in metrics["calibration_bins"]), 3)
        self.assertTrue(any(row["lower"] == 0.6
                            for row in metrics["calibration_bins"]))
        self.assertFalse(any(row["lower"] == 0.9
                             for row in metrics["calibration_bins"]))


if __name__ == "__main__":
    unittest.main()

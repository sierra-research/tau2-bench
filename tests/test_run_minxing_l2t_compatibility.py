import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_runner_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / "run_minxing_l2t_compatibility.py"
    spec = importlib.util.spec_from_file_location("run_minxing_l2t_compatibility", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MinxingL2TCompatibilityRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner_module()

    def test_metrics_and_collapse_detection(self):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        metrics = self.runner.metrics_from_predictions(y_true, y_pred)
        self.assertEqual(metrics["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 2})
        self.assertEqual(metrics["class_distribution"], {"0": 2, "1": 3})
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (0.5 + 2 / 3) / 2)
        self.assertFalse(metrics["one_class_collapse"])
        self.assertEqual(metrics["collapse_status"], "not_collapsed")

        collapsed = self.runner.metrics_from_predictions(y_true, np.ones_like(y_true))
        self.assertTrue(collapsed["one_class_collapse"])
        self.assertEqual(collapsed["collapse_status"], "collapsed_to_y1")
        self.assertEqual(collapsed["predicted_classes"], [1])

    def test_minxing_split_matches_randomstate_permutation(self):
        train_idx, val_idx = self.runner.minxing_split_indices(93, seed=1)
        expected = np.random.RandomState(1).permutation(93)
        np.testing.assert_array_equal(train_idx, expected[:74])
        np.testing.assert_array_equal(val_idx, expected[74:])

    def test_output_schema_payload_contains_required_fields(self):
        cfg = Namespace(
            dataset="tau2",
            input_data=Path("data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl"),
            mode="label_bce",
            mode_resolved="label_bce",
            seed=1,
            output_dir=Path("results/example"),
            minxing_repo=Path("/tmp/minxing"),
            time_window=(0.05, 3.15),
        )
        train_pred = pd.DataFrame(
            {"y_true": [0, 1, 1], "y_pred": [0, 1, 0], "score": [0.1, 0.9, 0.4]}
        )
        val_pred = pd.DataFrame(
            {"y_true": [0, 1], "y_pred": [0, 1], "score": [0.2, 0.8]}
        )
        payload = self.runner.build_metrics_payload(
            cfg=cfg,
            runtime_sec=1.25,
            device="cpu",
            inputs_used="X only",
            objective="binary cross-entropy on benchmark success label",
            train_pred_df=train_pred,
            val_pred_df=val_pred,
            train_idx=np.array([2, 0, 1]),
            val_idx=np.array([3, 4]),
            x_np=np.zeros((5, 2), dtype=np.float32),
            s=np.zeros((5, 4), dtype=np.float32),
            o_hist=np.zeros((5, 3, 1), dtype=np.float32),
            minxing_changed_files={},
        )
        self.assertEqual(payload["schema_version"], "minxing_l2t_compatibility_v1")
        self.assertEqual(payload["dataset"], "tau2")
        self.assertEqual(payload["mode"], "label_bce")
        self.assertEqual(payload["inputs_used"], "X only")
        self.assertEqual(payload["val_metrics"]["confusion_matrix"], {"tn": 1, "fp": 0, "fn": 0, "tp": 1})
        self.assertEqual(payload["split"]["implementation"], "np.random.RandomState(seed).permutation(N)")
        self.assertFalse(payload["minxing_integrity"]["source_changed"])

    def test_source_fingerprint_change_detection(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_baseline = root / "share_code/experiment/run_baseline.py"
            model_file = root / "share_code/src/train/models.py"
            run_baseline.parent.mkdir(parents=True)
            model_file.parent.mkdir(parents=True)
            run_baseline.write_text("print('baseline')\n", encoding="utf-8")
            model_file.write_text("VALUE = 1\n", encoding="utf-8")

            before = self.runner.source_fingerprint(root)
            model_file.write_text("VALUE = 2\n", encoding="utf-8")
            after = self.runner.source_fingerprint(root)
            changed = self.runner.changed_fingerprints(before, after)
            self.assertIn("share_code/src/train/models.py", changed)
            self.assertNotIn("share_code/experiment/run_baseline.py", changed)

    def test_parse_args_dataset_defaults(self):
        args = self.runner.parse_args(
            [
                "--minxing-repo",
                "/tmp/minxing",
                "--dataset",
                "bfcl",
                "--mode",
                "proposed_only",
                "--output-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.mode_resolved, "proposed")
        self.assertEqual(args.batch_train, 128)
        self.assertTrue(str(args.input_data).endswith("bfcl_v4_non_live_1240_l2t.pkl"))


if __name__ == "__main__":
    unittest.main()

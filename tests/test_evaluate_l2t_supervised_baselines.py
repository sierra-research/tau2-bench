import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_diagnostics_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / "evaluate_l2t_supervised_baselines.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_l2t_supervised_baselines", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class L2TSupervisedBaselineDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diag = load_diagnostics_module()

    def test_minxing_split_matches_randomstate_permutation(self):
        train_idx, val_idx = self.diag.minxing_split_indices(1240, seed=1)
        expected = np.random.RandomState(1).permutation(1240)
        np.testing.assert_array_equal(train_idx, expected[:992])
        np.testing.assert_array_equal(val_idx, expected[992:])
        self.assertEqual(len(train_idx), 992)
        self.assertEqual(len(val_idx), 248)

    def test_artifact_verification_for_generated_bridge_files(self):
        bfcl = self.diag.load_l2t_artifact(self.diag.DEFAULT_BFCL_PKL)
        apibank = self.diag.load_l2t_artifact(self.diag.DEFAULT_APIBANK_PKL)
        self.assertEqual(bfcl["keys"], sorted(bfcl["payload"].keys()))
        self.assertEqual(bfcl["X"].shape, (1240, 17))
        self.assertEqual(bfcl["S"].shape, (1240, 32))
        self.assertEqual(self.diag.class_counts_dict(bfcl["y"]), {"0": 181, "1": 1059})
        self.assertEqual(apibank["X"].shape, (1016, 402))
        self.assertEqual(apibank["S"].shape, (1016, 404))
        self.assertEqual(
            self.diag.class_counts_dict(apibank["y"]), {"0": 508, "1": 508}
        )
        self.assertEqual(
            self.diag.artifact_verification("bfcl", bfcl, seed=1)[
                "minxing_row_split"
            ]["train_class_counts"],
            {"0": 153, "1": 839},
        )
        self.assertEqual(
            self.diag.artifact_verification("apibank", apibank, seed=1)[
                "pair_grouping"
            ]["legacy_minxing_row_split_cross_split_pair_count"],
            160,
        )
        grouped = self.diag.artifact_verification("apibank", apibank, seed=1)[
            "pair_grouping"
        ]["grouped_split"]
        self.assertEqual(grouped["cross_split_group_count"], 0)
        self.assertEqual(grouped["train_size"], 812)
        self.assertEqual(grouped["validation_size"], 204)
        self.assertEqual(grouped["train_class_counts"], {"0": 406, "1": 406})
        self.assertEqual(grouped["validation_class_counts"], {"0": 102, "1": 102})

    def test_apibank_grouped_split_is_deterministic_and_pair_safe(self):
        group_ids = np.asarray(["b", "a", "c", "a", "b", "c", "d", "d"])
        y = np.asarray([0, 0, 1, 1, 1, 0, 0, 1])
        train_idx, val_idx = self.diag.grouped_split_indices(group_ids, seed=3)
        crossing = self.diag.split_group_crossings(group_ids, train_idx, val_idx)
        self.assertEqual(crossing["cross_split_group_count"], 0)
        self.assertGreaterEqual(len(train_idx), 4)
        self.assertLessEqual(len(train_idx), 6)
        self.assertTrue(set(np.unique(y[train_idx])).issubset({0, 1}))

        order = np.asarray([5, 2, 1, 3, 0, 4, 7, 6])
        shuffled_group_ids = group_ids[order]
        shuffled_train_idx, shuffled_val_idx = self.diag.grouped_split_indices(
            shuffled_group_ids, seed=3
        )
        original_train_groups = set(group_ids[train_idx].tolist())
        shuffled_train_groups = set(shuffled_group_ids[shuffled_train_idx].tolist())
        original_val_groups = set(group_ids[val_idx].tolist())
        shuffled_val_groups = set(shuffled_group_ids[shuffled_val_idx].tolist())
        self.assertEqual(original_train_groups, shuffled_train_groups)
        self.assertEqual(original_val_groups, shuffled_val_groups)

    def test_bfcl_subset_masks(self):
        bfcl = self.diag.load_l2t_artifact(self.diag.DEFAULT_BFCL_PKL)
        masks = self.diag.bfcl_subset_masks(bfcl)
        self.assertEqual(int(np.sum(masks["all_categories"])), 1240)
        self.assertEqual(int(np.sum(masks["non_irrelevance"])), 1000)
        self.assertEqual(int(np.sum(masks["irrelevance_only"])), 240)
        self.assertEqual(
            self.diag.class_counts_dict(bfcl["y"][masks["non_irrelevance"]]),
            {"0": 140, "1": 860},
        )
        self.assertEqual(
            self.diag.class_counts_dict(bfcl["y"][masks["irrelevance_only"]]),
            {"0": 41, "1": 199},
        )

    def test_metric_correctness(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_score = np.array([0.1, 0.8, 0.7, 0.9])
        metrics = self.diag.metric_dict(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 2)
        self.assertEqual(metrics["class_0_recall"], 0.5)
        self.assertEqual(metrics["class_1_recall"], 1.0)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)

    def test_leakage_audit_detects_exact_inverse_and_constant_columns(self):
        y = np.asarray([0, 1, 0, 1], dtype=np.int64)
        x = np.asarray(
            [
                [0, 1, 7, 0.2],
                [1, 0, 7, 0.4],
                [0, 1, 7, 0.6],
                [1, 0, 7, 0.8],
            ],
            dtype=np.float32,
        )
        audit = self.diag.audit_feature_matrix(x, y)
        self.assertEqual(audit["exact_y_column_indices"], [0])
        self.assertEqual(audit["exact_inverse_y_column_indices"], [1])
        self.assertEqual(audit["constant_column_indices_first50"], [2])
        self.assertEqual(audit["exact_y_column_count"], 1)
        self.assertEqual(audit["exact_inverse_y_column_count"], 1)
        self.assertEqual(audit["constant_column_count"], 1)

    def test_standardizer_is_fit_on_training_data_only(self):
        x_train = np.array([[0.0, 10.0], [2.0, 14.0], [4.0, 18.0], [6.0, 22.0]])
        y_train = np.array([0, 0, 1, 1])
        x_val = np.array([[100.0, 200.0], [120.0, 240.0]])
        pipeline = Pipeline(
            [
                ("standardize", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=200)),
            ]
        )
        pipeline.fit(x_train, y_train)
        train_mean = x_train.mean(axis=0)
        full_mean = np.vstack([x_train, x_val]).mean(axis=0)
        np.testing.assert_allclose(pipeline.named_steps["standardize"].mean_, train_mean)
        self.assertFalse(
            np.allclose(pipeline.named_steps["standardize"].mean_, full_mean)
        )

    def test_deterministic_evaluation_and_permutation_control(self):
        rng = np.random.RandomState(7)
        x0 = rng.normal(loc=-2.0, scale=0.4, size=(100, 2))
        x1 = rng.normal(loc=2.0, scale=0.4, size=(100, 2))
        x = np.vstack([x0, x1]).astype(np.float32)
        y = np.asarray([0] * 100 + [1] * 100, dtype=np.int64)
        train_idx, val_idx = self.diag.minxing_split_indices(len(y), seed=1)
        model = Pipeline(
            [
                ("standardize", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, random_state=1),
                ),
            ]
        )
        normal = self.diag.fit_evaluate_one(
            dataset="synthetic",
            subset="all",
            view="X-only",
            split_policy="minxing_row_random",
            model_name="logistic",
            model=copy.deepcopy(model),
            x_train=x[train_idx],
            y_train=y[train_idx],
            x_val=x[val_idx],
            y_val=y[val_idx],
            label_permuted=False,
            seed=1,
        )
        permuted_first = self.diag.fit_evaluate_one(
            dataset="synthetic",
            subset="all",
            view="X-only",
            split_policy="minxing_row_random",
            model_name="logistic",
            model=copy.deepcopy(model),
            x_train=x[train_idx],
            y_train=y[train_idx],
            x_val=x[val_idx],
            y_val=y[val_idx],
            label_permuted=True,
            seed=1,
        )
        permuted_second = self.diag.fit_evaluate_one(
            dataset="synthetic",
            subset="all",
            view="X-only",
            split_policy="minxing_row_random",
            model_name="logistic",
            model=copy.deepcopy(model),
            x_train=x[train_idx],
            y_train=y[train_idx],
            x_val=x[val_idx],
            y_val=y[val_idx],
            label_permuted=True,
            seed=1,
        )
        self.assertGreater(normal["balanced_accuracy"], 0.95)
        self.assertLess(permuted_first["balanced_accuracy"], 0.75)
        self.assertEqual(permuted_first, permuted_second)

    def test_repeated_permutation_summary_is_deterministic(self):
        rows = []
        for trial, balanced_accuracy in enumerate([0.4, 0.5, 0.6]):
            rows.append(
                {
                    "dataset": "synthetic",
                    "subset": "all",
                    "split_policy": "pair_grouped",
                    "view": "X-only",
                    "model": "model_a",
                    "label_permuted": True,
                    "permutation_trial": trial,
                    "accuracy": balanced_accuracy + 0.1,
                    "balanced_accuracy": balanced_accuracy,
                    "macro_f1": balanced_accuracy - 0.1,
                }
            )
        summary_first = self.diag.permutation_control_summary(rows)
        summary_second = self.diag.permutation_control_summary(list(reversed(rows)))
        self.assertEqual(summary_first, summary_second)
        self.assertEqual(summary_first[0]["n_trials"], 3)
        self.assertAlmostEqual(summary_first[0]["balanced_accuracy_mean"], 0.5)
        self.assertAlmostEqual(summary_first[0]["balanced_accuracy_min"], 0.4)
        self.assertAlmostEqual(summary_first[0]["balanced_accuracy_max"], 0.6)


if __name__ == "__main__":
    unittest.main()

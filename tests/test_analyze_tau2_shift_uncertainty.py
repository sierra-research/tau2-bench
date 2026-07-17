import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_toolcalling_numerical_representation.py"
INVENTORY_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_toolcalling_shift_inventory.py"
UNCERTAINTY_SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_tau2_shift_uncertainty.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MockEncoder:
    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def encode(
        self,
        texts,
        *,
        batch_size,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        del batch_size, convert_to_numpy, show_progress_bar
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeated = (digest * ((self.dimension // len(digest)) + 1))[: self.dimension]
            row = np.asarray([(value / 255.0) - 0.5 for value in repeated], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(row)
                if norm:
                    row = row / norm
            rows.append(row)
        return np.vstack(rows)


class Tau2ShiftUncertaintyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.numerical = load_module("build_toolcalling_numerical_representation", NUMERICAL_SCRIPT_PATH)
        cls.inventory_builder = load_module("build_toolcalling_shift_inventory", INVENTORY_SCRIPT_PATH)
        cls.uncertainty = load_module("analyze_tau2_shift_uncertainty", UNCERTAINTY_SCRIPT_PATH)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.tmpdir.name) / "numerical"
        cls.numerical.build_outputs(
            output_dir=cls.output_dir,
            full_data=True,
            encoder=MockEncoder(),
            write_outputs=True,
        )
        cls.inventory_jsonl = Path(cls.tmpdir.name) / "inventory.jsonl"
        cls.inventory_summary_json = Path(cls.tmpdir.name) / "inventory_summary.json"
        cls.inventory_report = Path(cls.tmpdir.name) / "inventory.md"
        cls.inventory, _ = cls.inventory_builder.build_outputs(
            numerical_jsonl=cls.output_dir / cls.numerical.FULL_JSONL_NAME,
            numerical_npz=cls.output_dir / cls.numerical.FULL_NPZ_NAME,
            inventory_jsonl=cls.inventory_jsonl,
            summary_json=cls.inventory_summary_json,
            report_path=cls.inventory_report,
            write_outputs=True,
        )
        cls.uncertainty_jsonl = Path(cls.tmpdir.name) / "tau2_shift_uncertainty.jsonl"
        cls.uncertainty_summary_json = Path(cls.tmpdir.name) / "tau2_shift_uncertainty_summary.json"
        cls.uncertainty_report = Path(cls.tmpdir.name) / "tau2_shift_uncertainty.md"
        cls.results, cls.summary = cls.uncertainty.build_outputs(
            numerical_jsonl=cls.output_dir / cls.numerical.FULL_JSONL_NAME,
            numerical_npz=cls.output_dir / cls.numerical.FULL_NPZ_NAME,
            inventory_jsonl=cls.inventory_jsonl,
            inventory_summary_json=cls.inventory_summary_json,
            output_jsonl=cls.uncertainty_jsonl,
            summary_json=cls.uncertainty_summary_json,
            report_path=cls.uncertainty_report,
            write_outputs=True,
        )
        cls.arrays = cls.inventory_builder.load_npz_arrays(cls.output_dir / cls.numerical.FULL_NPZ_NAME)
        numerical_records = cls.numerical.load_jsonl(cls.output_dir / cls.numerical.FULL_JSONL_NAME)
        unified_records = cls.numerical.load_jsonl(cls.numerical.DEFAULT_TAU2_INPUT) + cls.numerical.load_jsonl(
            cls.numerical.DEFAULT_API_BANK_INPUT
        )
        cls.rows = cls.inventory_builder.make_analysis_rows(numerical_records, unified_records, cls.arrays)
        cls.tau2_rows = [row for row in cls.rows if row["source_dataset"] == "tau2"]

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def result_by_id(self, shift_id):
        return next(row for row in self.results if row["shift_id"] == shift_id)

    def inventory_by_id(self, shift_id):
        return next(row for row in self.inventory if row["shift_id"] == shift_id)

    def test_only_tau2_real_benchmark_outcomes_are_analyzed_and_api_bank_is_excluded(self):
        self.assertEqual(len(self.results), 6)
        self.assertTrue(all(row["shift_id"].startswith("tau2_") for row in self.results))
        self.assertTrue(
            all(row["outcome_type"] == self.inventory_builder.TAU2_OUTCOME_TYPE for row in self.results)
        )
        self.assertFalse(any(row["shift_id"].startswith("api_bank_") for row in self.results))

    def test_existing_group_definitions_are_preserved(self):
        for row in self.results:
            inventory_row = self.inventory_by_id(row["shift_id"])
            self.assertEqual(row["source_rule"]["grouping_rule"], inventory_row["grouping_rule"])
            self.assertEqual(row["target_rule"]["grouping_rule"], inventory_row["grouping_rule"])
            self.assertEqual(row["source_rule"]["fields"], inventory_row["group_definition_fields"])
            self.assertEqual(row["target_rule"]["thresholds"], inventory_row["thresholds"])
            self.assertEqual(row["source_n"], inventory_row["source_sample_count"])
            self.assertEqual(row["target_n"], inventory_row["target_sample_count"])
            self.assertEqual(row["source_positive"], inventory_row["source_label_counts"].get("1", 0))
            self.assertEqual(row["target_positive"], inventory_row["target_label_counts"].get("1", 0))

    def test_y_is_not_used_to_define_group_membership(self):
        shift = self.inventory_by_id("tau2_zero_or_one_write_to_two_plus_writes")
        source_rows, target_rows = self.uncertainty.candidate_source_target_rows(shift, self.tau2_rows)
        flipped_rows = [dict(row, y=1 - row["y"]) for row in self.tau2_rows]
        flipped_source, flipped_target = self.uncertainty.candidate_source_target_rows(shift, flipped_rows)
        self.assertEqual(
            {row["sample_id"] for row in source_rows},
            {row["sample_id"] for row in flipped_source},
        )
        self.assertEqual(
            {row["sample_id"] for row in target_rows},
            {row["sample_id"] for row in flipped_target},
        )

    def test_delta_y_is_calculated_correctly(self):
        row = self.result_by_id("tau2_retail_to_airline")
        expected = (row["target_positive"] / row["target_n"]) - (row["source_positive"] / row["source_n"])
        self.assertAlmostEqual(row["delta_y"], expected)

    def test_confidence_intervals_are_finite_and_ordered(self):
        for row in self.results:
            for key in ["delta_y_ci_95", "bootstrap_delta_y_ci_95"]:
                low, high = row[key]
                self.assertTrue(np.isfinite(low), row["shift_id"])
                self.assertTrue(np.isfinite(high), row["shift_id"])
                self.assertLessEqual(low, high, row["shift_id"])
                self.assertGreaterEqual(low, -1.0, row["shift_id"])
                self.assertLessEqual(high, 1.0, row["shift_id"])

    def test_bootstrap_results_are_deterministic(self):
        repeat_results, repeat_summary = self.uncertainty.build_outputs(
            numerical_jsonl=self.output_dir / self.numerical.FULL_JSONL_NAME,
            numerical_npz=self.output_dir / self.numerical.FULL_NPZ_NAME,
            inventory_jsonl=self.inventory_jsonl,
            inventory_summary_json=self.inventory_summary_json,
            output_jsonl=Path(self.tmpdir.name) / "repeat.jsonl",
            summary_json=Path(self.tmpdir.name) / "repeat.json",
            report_path=Path(self.tmpdir.name) / "repeat.md",
            write_outputs=False,
        )
        self.assertEqual(
            [row["bootstrap_delta_y_ci_95"] for row in repeat_results],
            [row["bootstrap_delta_y_ci_95"] for row in self.results],
        )
        self.assertEqual(repeat_summary["bootstrap_configuration"]["seed"], 1)

    def test_classification_rules_for_all_three_thresholds(self):
        ci = (-0.40, -0.20)
        classifications = self.uncertainty.classification_by_threshold(ci)
        self.assertEqual(classifications["0.05"], "candidate_harmful")
        self.assertEqual(classifications["0.10"], "candidate_harmful")
        self.assertEqual(classifications["0.15"], "candidate_harmful")
        self.assertEqual(set(classifications), {"0.05", "0.10", "0.15"})

    def test_candidate_harmless_requires_full_ci_inside_equivalence_region(self):
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.04, 0.04), 0.05),
            "candidate_harmless",
        )
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.06, 0.04), 0.05),
            "inconclusive",
        )

    def test_candidate_harmful_requires_full_ci_below_negative_threshold(self):
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.30, -0.06), 0.05),
            "candidate_harmful",
        )
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.30, -0.05), 0.05),
            "inconclusive",
        )

    def test_candidate_beneficial_requires_full_ci_above_positive_threshold(self):
        self.assertEqual(
            self.uncertainty.classify_delta_ci((0.06, 0.30), 0.05),
            "candidate_beneficial",
        )
        self.assertEqual(
            self.uncertainty.classify_delta_ci((0.05, 0.30), 0.05),
            "inconclusive",
        )

    def test_other_cases_are_inconclusive(self):
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.20, 0.10), 0.05),
            "inconclusive",
        )

    def test_bh_adjusted_p_values_are_valid(self):
        raw = [row["raw_p_value"] for row in self.results]
        adjusted = [row["bh_adjusted_p_value"] for row in self.results]
        self.assertEqual(adjusted, self.uncertainty.bh_adjusted_p_values(raw))
        for raw_p_value, adjusted_p_value in zip(raw, adjusted, strict=True):
            self.assertGreaterEqual(adjusted_p_value, raw_p_value)
            self.assertGreaterEqual(adjusted_p_value, 0.0)
            self.assertLessEqual(adjusted_p_value, 1.0)

    def test_small_group_and_overlap_warnings_are_recorded(self):
        self.assertGreater(self.summary["small_sample_warning_count"], 0)
        for row in self.results:
            if row["source_n"] < self.uncertainty.SMALL_GROUP_WARNING_N or row["target_n"] < self.uncertainty.SMALL_GROUP_WARNING_N:
                self.assertTrue(
                    any("small group size warning" in warning for warning in row["warnings"]),
                    row["shift_id"],
                )
            if row["source_target_overlap_count"] > 0:
                self.assertTrue(
                    any("overlapping source/target groups" in warning for warning in row["warnings"]),
                    row["shift_id"],
                )

    def test_output_schema_validation(self):
        required_keys = {
            "shift_id",
            "shift_family",
            "source_rule",
            "target_rule",
            "source_n",
            "target_n",
            "source_positive",
            "target_positive",
            "source_success_rate",
            "target_success_rate",
            "delta_y",
            "delta_y_ci_method",
            "delta_y_ci_95",
            "bootstrap_delta_y_ci_95",
            "risk_ratio",
            "odds_ratio",
            "raw_p_value",
            "bh_adjusted_p_value",
            "x_centroid_distance",
            "s_centroid_distance",
            "source_target_overlap_count",
            "classification_by_threshold",
            "classification_stability",
            "warnings",
            "outcome_type",
        }
        rows = [json.loads(line) for line in self.uncertainty_jsonl.read_text().splitlines()]
        self.assertEqual(len(rows), len(self.results))
        for row in rows:
            self.assertTrue(required_keys.issubset(row.keys()))
            self.assertEqual(set(row["classification_by_threshold"]), {"0.05", "0.10", "0.15"})
        self.assertTrue(self.uncertainty_summary_json.exists())
        report = self.uncertainty_report.read_text(encoding="utf-8")
        self.assertIn("# Tau2 Shift Uncertainty Analysis", report)
        self.assertIn("Retraining interpretation", report)


if __name__ == "__main__":
    unittest.main()

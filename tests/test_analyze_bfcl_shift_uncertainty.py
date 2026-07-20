import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_bfcl_shift_inventory.py"
UNCERTAINTY_SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_bfcl_shift_uncertainty.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BFCLShiftUncertaintyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory_builder = load_module(
            "build_bfcl_shift_inventory",
            INVENTORY_SCRIPT_PATH,
        )
        cls.uncertainty = load_module(
            "analyze_bfcl_shift_uncertainty",
            UNCERTAINTY_SCRIPT_PATH,
        )
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.inventory_jsonl = Path(cls.tmpdir.name) / "inventory.jsonl"
        cls.inventory_summary_json = Path(cls.tmpdir.name) / "inventory_summary.json"
        cls.inventory_report = Path(cls.tmpdir.name) / "inventory.md"
        cls.audit_report = Path(cls.tmpdir.name) / "audit.md"
        cls.inventory, _ = cls.inventory_builder.build_outputs(
            output_jsonl=cls.inventory_jsonl,
            summary_json=cls.inventory_summary_json,
            report_path=cls.inventory_report,
            audit_report_path=cls.audit_report,
            write_outputs=True,
        )
        cls.output_jsonl = Path(cls.tmpdir.name) / "uncertainty.jsonl"
        cls.summary_json = Path(cls.tmpdir.name) / "uncertainty_summary.json"
        cls.report_path = Path(cls.tmpdir.name) / "uncertainty.md"
        cls.results, cls.summary = cls.uncertainty.build_outputs(
            inventory_jsonl=cls.inventory_jsonl,
            inventory_summary_json=cls.inventory_summary_json,
            output_jsonl=cls.output_jsonl,
            summary_json=cls.summary_json,
            report_path=cls.report_path,
            write_outputs=True,
        )
        cls.rows = cls.inventory_builder.load_jsonl(cls.inventory_builder.DEFAULT_INPUT_JSONL)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def result_by_id(self, shift_id):
        return next(row for row in self.results if row["shift_id"] == shift_id)

    def inventory_by_id(self, shift_id):
        return next(row for row in self.inventory if row["shift_id"] == shift_id)

    def test_all_inventory_rows_are_analyzed_with_preserved_metadata(self):
        self.assertEqual(len(self.results), 6)
        self.assertTrue(all(row["shift_id"].startswith("bfcl_") for row in self.results))
        for row in self.results:
            self.assertEqual(row["model"], "gpt-4o-mini-2024-07-18-FC")
            self.assertEqual(row["label_scope"], "test_case_level")
            self.assertEqual(row["label_origin"], "bfcl_evaluator")
            self.assertFalse(row["is_synthetic"])
            self.assertEqual(row["outcome_type"], "bfcl_test_case_correctness")

    def test_y_is_not_used_to_define_group_membership(self):
        shift = self.inventory_by_id("bfcl_simple_python_to_parallel_multiple")
        source_rows, target_rows = self.uncertainty.source_target_rows(shift, self.rows)
        flipped_rows = [dict(row, y=1 - row["y"]) for row in self.rows]
        flipped_source, flipped_target = self.uncertainty.source_target_rows(
            shift,
            flipped_rows,
        )
        self.assertEqual(
            {row["id"] for row in source_rows},
            {row["id"] for row in flipped_source},
        )
        self.assertEqual(
            {row["id"] for row in target_rows},
            {row["id"] for row in flipped_target},
        )

    def test_delta_y_and_counts_match_dataset_facts(self):
        row = self.result_by_id("bfcl_simple_python_to_parallel_multiple")
        self.assertEqual(row["source_n"], 400)
        self.assertEqual(row["target_n"], 200)
        self.assertEqual(row["source_positive"], 350)
        self.assertEqual(row["target_positive"], 160)
        self.assertAlmostEqual(row["source_success_rate"], 350 / 400)
        self.assertAlmostEqual(row["target_success_rate"], 160 / 200)
        self.assertAlmostEqual(row["delta_y"], -0.075)

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
        self.assertEqual(
            repeat_summary["bootstrap_configuration"]["replicates"],
            10_000,
        )

    def test_classification_rules_use_full_ci(self):
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.30, -0.06), 0.05),
            "candidate_harmful",
        )
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.30, -0.05), 0.05),
            "inconclusive",
        )
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.04, 0.04), 0.05),
            "candidate_harmless",
        )
        self.assertEqual(
            self.uncertainty.classify_delta_ci((-0.06, 0.04), 0.05),
            "inconclusive",
        )
        self.assertEqual(
            self.uncertainty.classify_delta_ci((0.06, 0.30), 0.05),
            "candidate_beneficial",
        )
        self.assertEqual(
            self.uncertainty.classify_delta_ci((0.05, 0.30), 0.05),
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

    def test_behavioral_irrelevance_shift_is_not_primary_complexity(self):
        row = self.result_by_id("bfcl_simple_python_to_irrelevance")
        self.assertEqual(row["shift_type"], "behavioral_abstention")
        self.assertFalse(row["is_primary_complexity_shift"])
        self.assertEqual(self.summary["primary_complexity_shift_count"], 5)
        self.assertEqual(self.summary["behavioral_abstention_shift_count"], 1)

    def test_output_schema_validation(self):
        required_keys = {
            "shift_id",
            "shift_family",
            "shift_type",
            "is_primary_complexity_shift",
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
            "raw_p_value",
            "bh_adjusted_p_value",
            "p_value_method",
            "classification_by_threshold",
            "classification_stability",
            "warnings",
            "label_scope",
            "label_origin",
        }
        rows = [json.loads(line) for line in self.output_jsonl.read_text().splitlines()]
        self.assertEqual(len(rows), len(self.results))
        for row in rows:
            self.assertTrue(required_keys.issubset(row.keys()))
            self.assertEqual(set(row["classification_by_threshold"]), {"0.05", "0.10", "0.15"})
        self.assertTrue(self.summary_json.exists())
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("BFCL Shift Uncertainty", report)
        self.assertIn("No causal claims", report)
        self.assertNotIn("retraining is required", report.lower())
        self.assertNotIn("safe to deploy", report.lower())

    def test_markdown_classification_summary_is_threshold_scoped(self):
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("## Classification Summary by Threshold", report)
        self.assertIn("### d=0.05", report)
        self.assertIn("### d=0.10", report)
        self.assertIn("### d=0.15", report)
        self.assertNotIn("## Candidate Harmless Shifts", report)
        self.assertNotIn("## Inconclusive Shifts", report)

        summary_text = report.split(
            "## Classification Summary by Threshold",
            maxsplit=1,
        )[1].split("## Constraints", maxsplit=1)[0]
        for line in summary_text.splitlines():
            if "bfcl_" in line:
                self.assertRegex(
                    line,
                    r"^- `(?:candidate_harmful|candidate_harmless|"
                    r"candidate_beneficial|inconclusive)`:",
                )

        self.assertIn(
            "- `candidate_harmless`: "
            "`bfcl_simple_python_to_multiple`, "
            "`bfcl_simple_python_to_parallel`",
            summary_text,
        )
        self.assertIn(
            "- `inconclusive`: `bfcl_multiple_to_parallel_multiple`",
            summary_text,
        )


if __name__ == "__main__":
    unittest.main()

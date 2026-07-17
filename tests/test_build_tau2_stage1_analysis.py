import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_tau2_stage1_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_tau2_stage1_analysis", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MockEncoder:
    def __init__(self, dimension: int = 384):
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


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tau2Stage1AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module()
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls.tmpdir.name)
        cls.output_dir = cls.tmp_path / "processed"
        cls.unified_jsonl = cls.output_dir / "unified_toolcalling_tau2_stage1.jsonl"
        cls.unified_manifest = cls.output_dir / "unified_toolcalling_tau2_stage1_manifest.json"
        cls.inventory_jsonl = cls.output_dir / "toolcalling_shift_inventory_tau2_stage1.jsonl"
        cls.inventory_summary = cls.output_dir / "toolcalling_shift_inventory_tau2_stage1_summary.json"
        cls.uncertainty_jsonl = cls.output_dir / "tau2_shift_uncertainty_stage1.jsonl"
        cls.uncertainty_summary = cls.output_dir / "tau2_shift_uncertainty_stage1_summary.json"
        cls.uncertainty_report = cls.tmp_path / "tau2_shift_uncertainty_stage1.md"
        cls.comparison_json = cls.output_dir / "tau2_shift_stage1_comparison.json"
        cls.comparison_report = cls.tmp_path / "tau2_shift_stage1_comparison.md"
        cls.project_log = cls.tmp_path / "toolcalling_shift_project_log.md"
        cls.project_log.write_text("# Log\n", encoding="utf-8")
        cls.protected_paths = [
            cls.builder.DEFAULT_TAU2_INPUT,
            cls.builder.BASELINE_INVENTORY_JSONL,
            cls.builder.BASELINE_UNCERTAINTY_JSONL,
            cls.builder.BASELINE_UNCERTAINTY_SUMMARY,
            cls.builder.BASELINE_UNCERTAINTY_REPORT,
        ]
        cls.before_hashes = {path: file_hash(path) for path in cls.protected_paths}
        cls.result = cls.builder.build_stage1_analysis(
            output_dir=cls.output_dir,
            unified_jsonl=cls.unified_jsonl,
            unified_manifest_json=cls.unified_manifest,
            inventory_jsonl=cls.inventory_jsonl,
            inventory_summary_json=cls.inventory_summary,
            uncertainty_jsonl=cls.uncertainty_jsonl,
            uncertainty_summary_json=cls.uncertainty_summary,
            uncertainty_report=cls.uncertainty_report,
            comparison_json=cls.comparison_json,
            comparison_report=cls.comparison_report,
            project_log=cls.project_log,
            encoder=MockEncoder(),
            append_log=True,
            write_outputs=True,
        )
        cls.combined = cls.builder.load_jsonl(cls.unified_jsonl)
        cls.stage1 = cls.builder.load_jsonl(cls.builder.DEFAULT_STAGE1_RETAINED)
        cls.baseline = cls.builder.load_jsonl(cls.builder.DEFAULT_TAU2_INPUT)
        cls.inventory = cls.builder.load_jsonl(cls.inventory_jsonl)
        cls.baseline_inventory = [
            row
            for row in cls.builder.load_jsonl(cls.builder.BASELINE_INVENTORY_JSONL)
            if row["dataset"] == "tau2"
        ]
        cls.uncertainty = cls.builder.load_jsonl(cls.uncertainty_jsonl)
        cls.comparison = json.loads(cls.comparison_json.read_text(encoding="utf-8"))
        cls.arrays = cls.builder.load_npz_arrays(
            cls.output_dir / cls.builder.DEFAULT_NUMERICAL_NPZ_NAME
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_merge_counts_y_distribution_and_invariants(self):
        manifest = self.result["merge_manifest"]
        self.assertEqual(manifest["total_count"], 105)
        self.assertEqual(manifest["source_counts"], {"baseline": 93, "stage1": 12})
        self.assertEqual(manifest["y_distribution"], {"0": 68, "1": 37})
        self.assertEqual(manifest["stage1_y_distribution"], {"0": 8, "1": 4})
        self.assertTrue(manifest["stage1_sample_ids_all_new"])
        self.assertEqual(manifest["label_scope"], "task_level")
        self.assertEqual(manifest["label_origin"], "tau2_benchmark_reward")
        self.assertFalse(manifest["is_synthetic"])
        self.assertEqual(self.combined[:93], self.baseline)

    def test_no_duplicate_sample_ids_and_stage1_ids_are_new(self):
        sample_ids = [record["sample_id"] for record in self.combined]
        self.assertEqual(len(sample_ids), len(set(sample_ids)))
        baseline_ids = {record["sample_id"] for record in self.baseline}
        stage1_ids = {record["sample_id"] for record in self.stage1}
        self.assertFalse(baseline_ids & stage1_ids)

    def test_numerical_shapes_and_finite_values(self):
        self.assertEqual(self.arrays["X"].shape, (105, 402))
        self.assertEqual(self.arrays["S"].shape, (105, 404))
        self.assertEqual(self.arrays["y"].shape, (105,))
        self.assertEqual(int(np.isnan(self.arrays["X"]).sum()), 0)
        self.assertEqual(int(np.isnan(self.arrays["S"]).sum()), 0)
        self.assertEqual(int(np.isinf(self.arrays["X"]).sum()), 0)
        self.assertEqual(int(np.isinf(self.arrays["S"]).sum()), 0)
        manifest = self.result["numerical_manifest"]
        self.assertEqual(manifest["total_count"], 105)
        self.assertEqual(manifest["count_by_dataset"], {"tau2": 105})

    def test_six_shift_definitions_and_thresholds_remain_unchanged(self):
        self.assertEqual([row["shift_id"] for row in self.inventory], [row["shift_id"] for row in self.baseline_inventory])
        for row, baseline in zip(self.inventory, self.baseline_inventory, strict=True):
            self.assertEqual(row["grouping_rule"], baseline["grouping_rule"])
            self.assertEqual(row["group_definition_fields"], baseline["group_definition_fields"])
            self.assertEqual(row["thresholds"], baseline["thresholds"])
            self.assertFalse(row["group_definition_uses_y"])
        summary = json.loads(self.inventory_summary.read_text(encoding="utf-8"))
        self.assertTrue(summary["definitions_preserved_from_baseline"])
        self.assertEqual(summary["group_definitions_using_y"], [])

    def test_pre_post_comparison_uses_matching_shift_ids(self):
        baseline_ids = [row["shift_id"] for row in self.baseline_inventory]
        self.assertEqual(self.comparison["matching_shift_ids"], baseline_ids)
        self.assertTrue(self.comparison["all_shift_ids_match"])
        self.assertEqual(len(self.comparison["comparison"]), 6)

    def test_ci_widths_and_classifications_are_calculated_correctly(self):
        for row in self.comparison["comparison"]:
            base_ci = row["baseline_delta_y_ci_95"]
            stage_ci = row["stage1_delta_y_ci_95"]
            self.assertAlmostEqual(row["baseline_ci_width"], base_ci[1] - base_ci[0])
            self.assertAlmostEqual(row["stage1_ci_width"], stage_ci[1] - stage_ci[0])
            self.assertEqual(
                row["ci_width_decreased"],
                row["stage1_ci_width"] < row["baseline_ci_width"],
            )
            self.assertEqual(
                row["classification_changed"],
                row["baseline_classification_by_threshold"]
                != row["stage1_classification_by_threshold"],
            )

    def test_stage1_membership_counts_are_correct(self):
        expected = {
            "tau2_retail_to_airline": {"source": 12, "target": 0, "both": 0, "neither": 0},
            "tau2_no_write_to_write_required": {"source": 3, "target": 9, "both": 0, "neither": 0},
            "tau2_zero_or_one_write_to_two_plus_writes": {"source": 4, "target": 8, "both": 0, "neither": 0},
        }
        by_id = {row["shift_id"]: row for row in self.comparison["comparison"]}
        for shift_id, counts in expected.items():
            self.assertEqual(by_id[shift_id]["stage1_membership_counts"], counts)
        for row in self.comparison["comparison"]:
            self.assertEqual(sum(row["stage1_membership_counts"].values()), 12)

    def test_candidate_harmful_shift_threshold_change_is_reported(self):
        self.assertFalse(
            self.comparison[
                "zero_or_one_write_to_two_plus_writes_remains_candidate_harmful_all_thresholds"
            ]
        )
        row = next(
            item
            for item in self.uncertainty
            if item["shift_id"] == "tau2_zero_or_one_write_to_two_plus_writes"
        )
        self.assertEqual(row["classification_by_threshold"]["0.05"], "candidate_harmful")
        self.assertEqual(row["classification_by_threshold"]["0.10"], "candidate_harmful")
        self.assertEqual(row["classification_by_threshold"]["0.15"], "inconclusive")
        self.assertIn(
            "tau2_zero_or_one_write_to_two_plus_writes",
            self.comparison["classification_changed_shift_ids"],
        )

    def test_deterministic_outputs(self):
        repeat = self.builder.build_stage1_analysis(
            output_dir=self.tmp_path / "repeat",
            unified_jsonl=self.tmp_path / "repeat" / "unified.jsonl",
            unified_manifest_json=self.tmp_path / "repeat" / "unified_manifest.json",
            inventory_jsonl=self.tmp_path / "repeat" / "inventory.jsonl",
            inventory_summary_json=self.tmp_path / "repeat" / "inventory_summary.json",
            uncertainty_jsonl=self.tmp_path / "repeat" / "uncertainty.jsonl",
            uncertainty_summary_json=self.tmp_path / "repeat" / "uncertainty_summary.json",
            uncertainty_report=self.tmp_path / "repeat" / "uncertainty.md",
            comparison_json=self.tmp_path / "repeat" / "comparison.json",
            comparison_report=self.tmp_path / "repeat" / "comparison.md",
            project_log=self.tmp_path / "repeat_log.md",
            encoder=MockEncoder(),
            append_log=False,
            write_outputs=True,
        )
        self.assertEqual(
            repeat["comparison"]["comparison"],
            self.result["comparison"]["comparison"],
        )
        self.assertEqual(
            repeat["stage1_uncertainty_rows"],
            self.result["stage1_uncertainty_rows"],
        )

    def test_baseline_outputs_remain_untouched(self):
        after_hashes = {path: file_hash(path) for path in self.protected_paths}
        self.assertEqual(after_hashes, self.before_hashes)
        self.assertTrue(self.result["baseline_outputs_unchanged"])

    def test_reports_and_project_log_are_written(self):
        self.assertIn("Stage 1 task selection was targeted", self.uncertainty_report.read_text(encoding="utf-8"))
        self.assertIn("4/12 successes", self.comparison_report.read_text(encoding="utf-8"))
        log_text = self.project_log.read_text(encoding="utf-8")
        self.assertIn(self.builder.STAGE1_SECTION_TITLE, log_text)
        self.assertIn("### Objective", log_text)
        self.assertIn("### Next step", log_text)


if __name__ == "__main__":
    unittest.main()

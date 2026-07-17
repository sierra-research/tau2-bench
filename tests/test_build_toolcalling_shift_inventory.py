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
FORBIDDEN_GROUP_FIELDS = {
    "corruption_type",
    "is_synthetic",
    "label_origin",
    "validation_error",
    "validation_status",
    "variant",
    "y",
}


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


class ToolcallingShiftInventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.numerical = load_module("build_toolcalling_numerical_representation", NUMERICAL_SCRIPT_PATH)
        cls.inventory_builder = load_module("build_toolcalling_shift_inventory", INVENTORY_SCRIPT_PATH)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.tmpdir.name) / "numerical"
        cls.numerical_records, _, _ = cls.numerical.build_outputs(
            output_dir=cls.output_dir,
            full_data=True,
            encoder=MockEncoder(),
            write_outputs=True,
        )
        cls.inventory_jsonl = Path(cls.tmpdir.name) / "inventory.jsonl"
        cls.summary_json = Path(cls.tmpdir.name) / "summary.json"
        cls.report_path = Path(cls.tmpdir.name) / "inventory.md"
        cls.inventory, cls.summary = cls.inventory_builder.build_outputs(
            numerical_jsonl=cls.output_dir / cls.numerical.FULL_JSONL_NAME,
            numerical_npz=cls.output_dir / cls.numerical.FULL_NPZ_NAME,
            inventory_jsonl=cls.inventory_jsonl,
            summary_json=cls.summary_json,
            report_path=cls.report_path,
            write_outputs=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def by_id(self, shift_id):
        return next(item for item in self.inventory if item["shift_id"] == shift_id)

    def test_inventory_outputs_are_written(self):
        self.assertTrue(self.inventory_jsonl.exists())
        self.assertTrue(self.summary_json.exists())
        self.assertTrue(self.report_path.exists())
        rows = [json.loads(line) for line in self.inventory_jsonl.read_text().splitlines()]
        self.assertEqual(len(rows), len(self.inventory))
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("Tool-Calling Candidate Shift Inventory", report)
        self.assertIn(self.inventory_builder.API_BANK_WARNING, report)

    def test_group_definitions_do_not_use_label_or_label_related_fields(self):
        self.assertEqual(self.summary["group_definitions_using_y"], [])
        for item in self.inventory:
            self.assertFalse(item["group_definition_uses_y"], item["shift_id"])
            field_parts = [
                part
                for field in item["group_definition_fields"]
                for part in field.split(".")
            ]
            self.assertFalse(
                any(forbidden == part for forbidden in FORBIDDEN_GROUP_FIELDS for part in field_parts),
                item["shift_id"],
            )

    def test_tau2_and_api_bank_remain_separated_with_correct_outcome_types(self):
        tau2_items = [item for item in self.inventory if item["dataset"] == "tau2"]
        api_items = [item for item in self.inventory if item["dataset"] == "api_bank"]
        self.assertTrue(tau2_items)
        self.assertTrue(api_items)
        self.assertTrue(
            all(item["outcome_type"] == "real_benchmark_task_outcome" for item in tau2_items)
        )
        self.assertTrue(
            all(item["outcome_type"] == "synthetic_api_call_correctness" for item in api_items)
        )
        self.assertEqual(self.summary["record_count_by_dataset"], {"api_bank": 1016, "tau2": 93})

    def test_api_bank_synthetic_outcome_warning_is_present(self):
        api_items = [item for item in self.inventory if item["dataset"] == "api_bank"]
        for item in api_items:
            self.assertIn(self.inventory_builder.API_BANK_WARNING, item["warnings"])
        self.assertEqual(self.summary["api_bank_warning"], self.inventory_builder.API_BANK_WARNING)

    def test_group_count_and_delta_calculations_for_tau2_domain(self):
        item = self.by_id("tau2_retail_to_airline")
        tau2_records = [
            record for record in self.numerical_records if record["source_dataset"] == "tau2"
        ]
        retail = [record for record in tau2_records if record["metadata"]["domain"] == "retail"]
        airline = [record for record in tau2_records if record["metadata"]["domain"] == "airline"]
        expected_delta = np.mean([record["y"] for record in airline]) - np.mean(
            [record["y"] for record in retail]
        )
        self.assertEqual(item["source_sample_count"], len(retail))
        self.assertEqual(item["target_sample_count"], len(airline))
        self.assertAlmostEqual(item["raw_delta_y"], expected_delta)
        self.assertEqual(
            item["source_label_counts"],
            {str(label): sum(record["y"] == label for record in retail) for label in [0, 1]},
        )

    def test_thresholds_rules_and_centroid_distances_are_recorded(self):
        for item in self.inventory:
            self.assertIsInstance(item["grouping_rule"], str)
            self.assertTrue(item["grouping_rule"])
            self.assertIsInstance(item["thresholds"], dict)
            self.assertGreaterEqual(item["min_group_size"], 1)
            if item["status"] == "eligible":
                self.assertIsNotNone(item["x_centroid_distance"], item["shift_id"])
                self.assertIsNotNone(item["s_centroid_distance"], item["shift_id"])
                self.assertTrue(np.isfinite(item["x_centroid_distance"]), item["shift_id"])
                self.assertTrue(np.isfinite(item["s_centroid_distance"]), item["shift_id"])

    def test_failed_groups_are_reported(self):
        item = self.by_id("api_bank_domain_or_tool_family_comparison")
        self.assertEqual(item["status"], "failed")
        self.assertLess(item["source_sample_count"], item["min_group_size"])
        self.assertLess(item["target_sample_count"], item["min_group_size"])
        self.assertIn("No reliable API-Bank domain metadata is available.", item["warnings"])


if __name__ == "__main__":
    unittest.main()

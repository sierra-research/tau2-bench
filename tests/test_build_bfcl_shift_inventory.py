import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_bfcl_shift_inventory.py"
FORBIDDEN_GROUP_FIELDS = {
    "is_synthetic",
    "label_origin",
    "label_scope",
    "metadata.evaluation_error",
    "metadata.evaluation_error_type",
    "s_raw",
    "y",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BFCLShiftInventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory_builder = load_module(
            "build_bfcl_shift_inventory",
            INVENTORY_SCRIPT_PATH,
        )
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.inventory_jsonl = Path(cls.tmpdir.name) / "inventory.jsonl"
        cls.summary_json = Path(cls.tmpdir.name) / "summary.json"
        cls.report_path = Path(cls.tmpdir.name) / "inventory.md"
        cls.audit_report_path = Path(cls.tmpdir.name) / "audit.md"
        cls.inventory, cls.summary = cls.inventory_builder.build_outputs(
            output_jsonl=cls.inventory_jsonl,
            summary_json=cls.summary_json,
            report_path=cls.report_path,
            audit_report_path=cls.audit_report_path,
            write_outputs=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def by_id(self, shift_id):
        return next(row for row in self.inventory if row["shift_id"] == shift_id)

    def test_dataset_facts_are_validated_and_preserved(self):
        self.assertEqual(self.summary["total"], 1240)
        self.assertEqual(self.summary["model"], "gpt-4o-mini-2024-07-18-FC")
        self.assertEqual(self.summary["label_scope"], "test_case_level")
        self.assertEqual(self.summary["label_origin"], "bfcl_evaluator")
        self.assertFalse(self.summary["is_synthetic"])
        self.assertEqual(self.summary["x_representation_field"], "x_raw")
        self.assertEqual(self.summary["s_representation_field"], "s_raw")
        self.assertEqual(self.summary["y_distribution"], {"0": 181, "1": 1059})
        self.assertEqual(self.summary["categories"]["simple_python"]["total"], 400)
        self.assertEqual(self.summary["categories"]["simple_python"]["y_1"], 350)
        self.assertEqual(self.summary["categories"]["irrelevance"]["total"], 240)
        self.assertEqual(self.summary["categories"]["irrelevance"]["y_0"], 41)

    def test_source_rows_require_x_s_and_y(self):
        rows = self.inventory_builder.load_jsonl(
            self.inventory_builder.DEFAULT_INPUT_JSONL,
        )
        source_summary = json.loads(
            self.inventory_builder.DEFAULT_SOURCE_SUMMARY_JSON.read_text(
                encoding="utf-8",
            ),
        )
        for required_field in ["x_raw", "s_raw", "y"]:
            broken_rows = [dict(row) for row in rows]
            del broken_rows[0][required_field]
            with self.assertRaises(ValueError):
                self.inventory_builder.validate_source_rows(
                    broken_rows,
                    source_summary,
                )

    def test_duplicate_source_row_ids_are_rejected_without_deduplication(self):
        rows = self.inventory_builder.load_jsonl(
            self.inventory_builder.DEFAULT_INPUT_JSONL,
        )
        source_summary = json.loads(
            self.inventory_builder.DEFAULT_SOURCE_SUMMARY_JSON.read_text(
                encoding="utf-8",
            ),
        )
        broken_rows = [dict(row) for row in rows]
        duplicate_id = broken_rows[0]["id"]
        broken_rows[1]["id"] = duplicate_id

        self.assertEqual(len(broken_rows), 1240)
        with self.assertRaises(ValueError) as context:
            self.inventory_builder.validate_source_rows(
                broken_rows,
                source_summary,
            )

        message = str(context.exception)
        self.assertIn("Duplicate BFCL row ids", message)
        self.assertIn("1 duplicate", message)
        self.assertIn(duplicate_id, message)

    def test_primary_and_behavioral_shift_sets_are_separate(self):
        self.assertEqual(self.summary["shift_count"], 6)
        self.assertEqual(self.summary["primary_complexity_shift_count"], 5)
        self.assertEqual(self.summary["behavioral_abstention_shift_count"], 1)
        self.assertEqual(
            set(self.summary["primary_candidate_shift_ids"]),
            {
                "bfcl_simple_python_to_multiple",
                "bfcl_simple_python_to_parallel",
                "bfcl_simple_python_to_parallel_multiple",
                "bfcl_multiple_to_parallel_multiple",
                "bfcl_parallel_to_parallel_multiple",
            },
        )
        behavioral = self.by_id("bfcl_simple_python_to_irrelevance")
        self.assertEqual(behavioral["shift_type"], "behavioral_abstention")
        self.assertFalse(behavioral["is_primary_complexity_shift"])

    def test_group_definitions_do_not_use_y_or_label_related_fields(self):
        self.assertEqual(self.summary["group_definitions_using_y"], [])
        for row in self.inventory:
            self.assertEqual(row["group_definition_fields"], ["category"])
            self.assertFalse(row["group_definition_uses_y"], row["shift_id"])
            self.assertFalse(
                any(field in FORBIDDEN_GROUP_FIELDS for field in row["group_definition_fields"]),
                row["shift_id"],
            )

    def test_group_counts_and_delta_are_category_based(self):
        row = self.by_id("bfcl_simple_python_to_parallel_multiple")
        self.assertEqual(row["source_sample_count"], 400)
        self.assertEqual(row["target_sample_count"], 200)
        self.assertEqual(row["source_label_counts"], {"0": 50, "1": 350})
        self.assertEqual(row["target_label_counts"], {"0": 40, "1": 160})
        self.assertAlmostEqual(row["source_y_mean"], 350 / 400)
        self.assertAlmostEqual(row["target_y_mean"], 160 / 200)
        self.assertAlmostEqual(row["raw_delta_y"], (160 / 200) - (350 / 400))
        self.assertEqual(row["source_target_overlap_count"], 0)

    def test_outputs_are_written(self):
        self.assertTrue(self.inventory_jsonl.exists())
        self.assertTrue(self.summary_json.exists())
        self.assertTrue(self.report_path.exists())
        self.assertTrue(self.audit_report_path.exists())
        rows = [json.loads(line) for line in self.inventory_jsonl.read_text().splitlines()]
        self.assertEqual(len(rows), len(self.inventory))
        self.assertIn("BFCL Shift Inventory", self.report_path.read_text(encoding="utf-8"))
        self.assertIn(
            "BFCL Data Source Audit",
            self.audit_report_path.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()

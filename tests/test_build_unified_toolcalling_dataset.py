import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_unified_toolcalling_dataset.py"
SCHEMA_FIELDS = {
    "sample_id",
    "source_dataset",
    "domain",
    "label_scope",
    "label_origin",
    "is_synthetic",
    "x_raw",
    "s_raw",
    "x_numeric_features",
    "s_numeric_features",
    "y",
    "metadata",
}
MODEL_FACING_FIELDS = {
    "x_raw",
    "s_raw",
    "x_numeric_features",
    "s_numeric_features",
}
LEAKAGE_KEYS = {
    "corruption_type",
    "variant",
    "validation_status",
    "validation_error",
    "label_origin",
    "is_synthetic",
    "y",
}


def load_builder_module():
    spec = importlib.util.spec_from_file_location(
        "build_unified_toolcalling_dataset", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(nested_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(nested_keys(child))
        return keys
    return set()


class UnifiedToolcallingDatasetBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder_module()
        cls.tmpdir = tempfile.TemporaryDirectory()
        output_dir = Path(cls.tmpdir.name) / "outputs"
        cls.tau2_records, cls.api_bank_records, cls.manifest = cls.builder.build_outputs(
            output_dir=output_dir
        )
        cls.records = cls.tau2_records + cls.api_bank_records
        cls.output_dir = output_dir

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_schema_validity(self):
        for record in self.records:
            self.assertEqual(set(record), SCHEMA_FIELDS)
            self.assertIsInstance(record["sample_id"], str)
            self.assertIn(record["source_dataset"], {"tau2", "api_bank"})
            self.assertTrue(record["domain"] is None or isinstance(record["domain"], str))
            self.assertIn(record["label_scope"], {"task_level", "api_call_level"})
            self.assertIsInstance(record["label_origin"], str)
            self.assertIsInstance(record["is_synthetic"], bool)
            self.assertIsInstance(record["x_raw"], dict)
            self.assertIsInstance(record["s_raw"], (dict, list))
            self.assertIsInstance(record["x_numeric_features"], dict)
            self.assertIsInstance(record["s_numeric_features"], dict)
            self.assertIn(record["y"], {0, 1})
            self.assertIsInstance(record["metadata"], dict)

    def test_deterministic_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_dir = Path(tmpdir) / "first"
            second_dir = Path(tmpdir) / "second"
            first = self.builder.build_outputs(output_dir=first_dir)
            second = self.builder.build_outputs(output_dir=second_dir)
            self.assertEqual(
                json.dumps(first[0], sort_keys=True),
                json.dumps(second[0], sort_keys=True),
            )
            self.assertEqual(
                json.dumps(first[1], sort_keys=True),
                json.dumps(second[1], sort_keys=True),
            )
            self.assertEqual(first[2], second[2])
            for file_name in (
                self.builder.TAU2_OUTPUT_NAME,
                self.builder.API_BANK_OUTPUT_NAME,
                self.builder.MANIFEST_OUTPUT_NAME,
            ):
                self.assertEqual(
                    (first_dir / file_name).read_bytes(),
                    (second_dir / file_name).read_bytes(),
                )

    def test_unique_sample_ids(self):
        sample_ids = [record["sample_id"] for record in self.records]
        self.assertEqual(len(sample_ids), len(set(sample_ids)))
        self.assertEqual(self.manifest["duplicate_sample_ids"], [])

    def test_correct_source_specific_metadata(self):
        for record in self.tau2_records:
            self.assertTrue(record["sample_id"].startswith("tau2:"))
            self.assertEqual(record["source_dataset"], "tau2")
            self.assertEqual(record["label_scope"], "task_level")
            self.assertEqual(record["label_origin"], "tau2_benchmark_reward")
            self.assertFalse(record["is_synthetic"])
            self.assertIn(record["domain"], {"retail", "airline"})
            self.assertEqual(record["metadata"]["source_dataset"], "tau2")
            self.assertEqual(record["metadata"]["sample_id"], record["sample_id"])
            self.assertIn("task_id", record["metadata"])
            self.assertIn("reward", record["metadata"])
            self.assertEqual(len(record["x_numeric_features"]), 12)
            self.assertEqual(len(record["s_raw"]), 64)
            self.assertEqual(len(record["s_numeric_features"]), 64)

        for record in self.api_bank_records:
            self.assertTrue(record["sample_id"].startswith("api_bank:"))
            self.assertEqual(record["source_dataset"], "api_bank")
            self.assertIsNone(record["domain"])
            self.assertEqual(record["label_scope"], "api_call_level")
            self.assertEqual(record["metadata"]["source_dataset"], "api_bank")
            self.assertEqual(record["metadata"]["sample_id"], record["sample_id"])
            self.assertIn("original_sample_id", record["metadata"])
            self.assertIn(record["metadata"]["variant"], {"positive", "negative"})
            self.assertIn("corruption_type", record["metadata"])

    def test_no_label_leakage(self):
        for record in self.records:
            keys = set()
            for field in MODEL_FACING_FIELDS:
                keys.update(nested_keys(record[field]))
            self.assertFalse(LEAKAGE_KEYS & keys)
        self.assertEqual(self.manifest["model_facing_leakage_fields"], {})

    def test_correct_total_counts(self):
        self.assertEqual(len(self.tau2_records), 93)
        self.assertEqual(len(self.api_bank_records), 1016)
        self.assertEqual(
            self.manifest["record_count_by_source_dataset"],
            {"tau2": 93, "api_bank": 1016},
        )

    def test_correct_api_bank_synthetic_flag(self):
        counts = Counter(record["is_synthetic"] for record in self.api_bank_records)
        self.assertEqual(counts, Counter({False: 508, True: 508}))
        for record in self.api_bank_records:
            if record["metadata"]["variant"] == "positive":
                self.assertFalse(record["is_synthetic"])
                self.assertEqual(record["label_origin"], "reference_api_call")
                self.assertEqual(record["y"], 1)
            else:
                self.assertTrue(record["is_synthetic"])
                self.assertEqual(record["label_origin"], "synthetic_corruption")
                self.assertEqual(record["y"], 0)

    def test_binary_y_values(self):
        self.assertEqual({record["y"] for record in self.records}, {0, 1})
        self.assertEqual(
            self.manifest["label_distribution_by_source_dataset"],
            {"tau2": {"0": 60, "1": 33}, "api_bank": {"0": 508, "1": 508}},
        )


if __name__ == "__main__":
    unittest.main()

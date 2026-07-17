import hashlib
import importlib.util
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_toolcalling_numerical_representation.py"
LEAKAGE_FIELDS = {
    "corruption_type",
    "is_synthetic",
    "label_origin",
    "validation_error",
    "validation_status",
    "variant",
}


def load_builder_module():
    spec = importlib.util.spec_from_file_location(
        "build_toolcalling_numerical_representation", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MockEncoder:
    def __init__(self, dimension: int = 8):
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
            values = [(value / 255.0) - 0.5 for value in repeated]
            row = np.asarray(values, dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(row)
                if norm:
                    row = row / norm
            rows.append(row)
        return np.vstack(rows)


class ToolcallingNumericalRepresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder_module()
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.tmpdir.name) / "outputs"
        cls.records, cls.manifest, cls.arrays = cls.builder.build_outputs(
            output_dir=cls.output_dir,
            tau2_limit=93,
            apibank_limit=107,
            seed=1,
            encoder=MockEncoder(),
            batch_size=16,
            write_outputs=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_exact_pilot_counts(self):
        counts = Counter(record["source_dataset"] for record in self.records)
        self.assertEqual(len(self.records), 200)
        self.assertEqual(counts, Counter({"api_bank": 107, "tau2": 93}))
        self.assertEqual(self.manifest["total_count"], 200)
        self.assertEqual(
            self.manifest["count_by_dataset"], {"api_bank": 107, "tau2": 93}
        )

    def test_deterministic_sample_selection(self):
        first_ids = [record["sample_id"] for record in self.records]
        second_records, second_manifest, _ = self.builder.build_outputs(
            output_dir=self.output_dir,
            tau2_limit=93,
            apibank_limit=107,
            seed=1,
            encoder=MockEncoder(),
            write_outputs=False,
        )
        self.assertEqual(first_ids, [record["sample_id"] for record in second_records])
        self.assertEqual(
            self.manifest["sampling_summary"], second_manifest["sampling_summary"]
        )

    def test_output_shapes_and_npz_contents(self):
        X = self.arrays["X"]
        S = self.arrays["S"]
        self.assertEqual(X.shape, (200, 26))
        self.assertEqual(S.shape, (200, 28))
        self.assertEqual(self.arrays["y"].shape, (200,))
        with np.load(self.output_dir / self.builder.DEFAULT_NPZ_NAME, allow_pickle=True) as data:
            self.assertEqual(data["X"].shape, X.shape)
            self.assertEqual(data["S"].shape, S.shape)
            self.assertEqual(data["y"].shape, (200,))
            self.assertEqual(data["sample_ids"].shape, (200,))
            self.assertEqual(data["source_dataset"].shape, (200,))
            self.assertEqual(data["label_scope"].shape, (200,))
            self.assertEqual(data["is_synthetic"].shape, (200,))

    def test_shared_dimensions_and_feature_order_are_stable(self):
        self.assertEqual(self.manifest["x_embedding_dimension"], 8)
        self.assertEqual(self.manifest["s_embedding_dimension"], 8)
        self.assertEqual(self.manifest["x_structural_dimension"], 18)
        self.assertEqual(self.manifest["s_structural_dimension"], 20)
        self.assertEqual(
            self.manifest["ordered_x_structural_feature_names"],
            self.builder.X_STRUCTURAL_FEATURE_NAMES,
        )
        self.assertEqual(
            self.manifest["ordered_s_structural_feature_names"],
            self.builder.S_STRUCTURAL_FEATURE_NAMES,
        )
        for record in self.records:
            self.assertEqual(
                list(record["x_structural_features"]),
                self.builder.X_STRUCTURAL_FEATURE_NAMES,
            )
            self.assertEqual(
                list(record["s_structural_features"]),
                self.builder.S_STRUCTURAL_FEATURE_NAMES,
            )

    def test_no_nan_infinite_values_and_normalized_embeddings(self):
        for name in ("X", "S"):
            self.assertEqual(int(np.isnan(self.arrays[name]).sum()), 0)
            self.assertEqual(int(np.isinf(self.arrays[name]).sum()), 0)
        for name in ("x_embeddings", "s_embeddings"):
            norms = np.linalg.norm(self.arrays[name], axis=1)
            np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)
        self.assertEqual(self.manifest["nan_counts"], {"S": 0, "X": 0})
        self.assertEqual(self.manifest["infinite_value_counts"], {"S": 0, "X": 0})

    def test_unique_sample_ids_and_binary_y(self):
        sample_ids = [record["sample_id"] for record in self.records]
        self.assertEqual(len(sample_ids), len(set(sample_ids)))
        self.assertEqual(self.manifest["duplicate_sample_ids"], [])
        self.assertLessEqual(set(self.arrays["y"].tolist()), {0, 1})

    def test_no_label_revealing_fields_in_model_inputs(self):
        for record in self.records:
            combined_text = f"{record['x_text']}\n{record['s_text']}".lower()
            for field in LEAKAGE_FIELDS:
                self.assertNotIn(field, combined_text)
            self.assertFalse(
                LEAKAGE_FIELDS & set(record["x_structural_features"]),
                record["sample_id"],
            )
            self.assertFalse(
                LEAKAGE_FIELDS & set(record["s_structural_features"]),
                record["sample_id"],
            )
        self.assertEqual(
            self.manifest["leakage_audit"]["model_facing_field_name_hits"], {}
        )

    def test_api_bank_pairs_have_identical_x_and_different_s(self):
        api_records_by_pair = defaultdict(list)
        for record in self.records:
            if record["source_dataset"] == "api_bank":
                api_records_by_pair[record["metadata"]["pair_id"]].append(record)

        complete_pairs = [
            members for members in api_records_by_pair.values() if len(members) == 2
        ]
        self.assertEqual(len(complete_pairs), 53)
        self.assertFalse(self.manifest["selected_api_bank_all_pairs_complete"])
        self.assertEqual(self.manifest["selected_api_bank_complete_pair_count"], 53)

        for first, second in complete_pairs:
            self.assertEqual(first["x_text"], second["x_text"])
            self.assertEqual(
                first["x_structural_features"], second["x_structural_features"]
            )
            self.assertNotEqual(first["s_text"], second["s_text"])

    def test_tau2_and_api_bank_metadata_remain_distinguishable(self):
        label_scopes = Counter(record["label_scope"] for record in self.records)
        self.assertEqual(label_scopes["task_level"], 93)
        self.assertEqual(label_scopes["api_call_level"], 107)
        tau2_records = [
            record for record in self.records if record["source_dataset"] == "tau2"
        ]
        api_records = [
            record for record in self.records if record["source_dataset"] == "api_bank"
        ]
        self.assertTrue(
            all(
                record["metadata"]["coarse_tau2_trajectory_serialization"]
                for record in tau2_records
            )
        )
        self.assertTrue(
            all(
                not record["metadata"]["coarse_tau2_trajectory_serialization"]
                for record in api_records
            )
        )

    def test_coarse_tau2_serialization_reported(self):
        self.assertEqual(
            self.manifest["coarse_tau2_trajectory_serialization_count"], 93
        )
        self.assertIn(
            "tau2 trajectories are coarse structural event sequences when raw messages are unavailable.",
            self.manifest["compatibility_warnings"],
        )

    def test_jsonl_does_not_store_embeddings(self):
        rows = [
            json.loads(line)
            for line in (self.output_dir / self.builder.DEFAULT_JSONL_NAME)
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(rows), 200)
        for row in rows:
            self.assertNotIn("X", row)
            self.assertNotIn("S", row)
            self.assertNotIn("embedding", json.dumps(row).lower())

    def test_full_data_mode_counts_shapes_and_integrity(self):
        full_output_dir = Path(self.tmpdir.name) / "full_outputs"
        records, manifest, arrays = self.builder.build_outputs(
            output_dir=full_output_dir,
            full_data=True,
            encoder=MockEncoder(dimension=384),
            batch_size=64,
            write_outputs=True,
        )
        counts = Counter(record["source_dataset"] for record in records)
        self.assertEqual(len(records), 1109)
        self.assertEqual(counts, Counter({"api_bank": 1016, "tau2": 93}))
        self.assertEqual(arrays["X"].shape, (1109, 402))
        self.assertEqual(arrays["S"].shape, (1109, 404))
        self.assertEqual(arrays["y"].shape, (1109,))
        self.assertEqual(manifest["total_count"], 1109)
        self.assertEqual(
            manifest["count_by_dataset"], {"api_bank": 1016, "tau2": 93}
        )
        self.assertEqual(
            manifest["label_scope_counts"], {"api_call_level": 1016, "task_level": 93}
        )
        self.assertEqual(manifest["duplicate_sample_ids"], [])
        self.assertEqual(manifest["nan_counts"], {"S": 0, "X": 0})
        self.assertEqual(manifest["infinite_value_counts"], {"S": 0, "X": 0})
        self.assertEqual(manifest["selected_api_bank_complete_pair_count"], 508)
        self.assertTrue(manifest["selected_api_bank_all_pairs_complete"])
        with np.load(full_output_dir / self.builder.FULL_NPZ_NAME, allow_pickle=True) as data:
            self.assertEqual(data["X"].shape, (1109, 402))
            self.assertEqual(data["S"].shape, (1109, 404))
            self.assertEqual(data["y"].shape, (1109,))

    def test_full_data_mode_is_deterministic(self):
        first_records, first_manifest, first_arrays = self.builder.build_outputs(
            output_dir=Path(self.tmpdir.name) / "deterministic_first",
            full_data=True,
            encoder=MockEncoder(dimension=16),
            write_outputs=False,
        )
        second_records, second_manifest, second_arrays = self.builder.build_outputs(
            output_dir=Path(self.tmpdir.name) / "deterministic_second",
            full_data=True,
            encoder=MockEncoder(dimension=16),
            write_outputs=False,
        )
        self.assertEqual(
            [record["sample_id"] for record in first_records],
            [record["sample_id"] for record in second_records],
        )
        self.assertEqual(first_manifest["sampling_summary"], second_manifest["sampling_summary"])
        np.testing.assert_array_equal(first_arrays["X"], second_arrays["X"])
        np.testing.assert_array_equal(first_arrays["S"], second_arrays["S"])


if __name__ == "__main__":
    unittest.main()

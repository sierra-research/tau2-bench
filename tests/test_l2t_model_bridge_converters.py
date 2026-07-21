import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def load_script(name: str):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class L2TModelBridgeConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bfcl = load_script("convert_bfcl_to_l2t_pkl")
        cls.apibank = load_script("convert_apibank_to_l2t_pkl")
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls.tmpdir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_bfcl_converter_preserves_labels_and_contract(self):
        output_path = self.tmp_path / "bfcl.pkl"
        manifest_path = self.tmp_path / "bfcl_manifest.json"
        payload, manifest = self.bfcl.convert(
            output_path=output_path,
            manifest_path=manifest_path,
            write_outputs=True,
        )

        self.assertTrue(output_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertEqual(payload["X"].shape, (1240, 17))
        self.assertEqual(payload["X"].dtype, np.float32)
        self.assertEqual(payload["y"].shape, (1240,))
        self.assertEqual(payload["y"].dtype, np.int64)
        self.assertEqual(payload["traj"]["s"].shape, (1240, 32))
        self.assertEqual(payload["traj"]["s"].dtype, np.float32)
        self.assertEqual(
            manifest["array_contract"]["class_distribution"], {"0": 181, "1": 1059}
        )
        self.assertEqual(manifest["split"]["seed"], 1)
        self.assertEqual(manifest["split"]["train_size"], 992)
        self.assertEqual(manifest["split"]["validation_size"], 248)
        self.assertEqual(manifest["array_contract"]["duplicate_sample_ids"], [])
        self.assertEqual(
            manifest["array_contract"]["nan_counts"], {"X": 0, "traj_s": 0}
        )
        self.assertEqual(
            manifest["array_contract"]["infinite_value_counts"],
            {"X": 0, "traj_s": 0},
        )
        self.assertEqual(manifest["label_scope"], "test_case_level")
        self.assertEqual(manifest["label_origin"], {"bfcl_evaluator": 1240})
        self.assertIn("sample_ids", payload)
        self.assertEqual(len(payload["sample_ids"]), len(set(payload["sample_ids"])))

    def test_bfcl_model_arrays_exclude_label_metadata_fields(self):
        payload, _ = self.bfcl.convert(write_outputs=False)
        self.assertNotIn("label_origin", payload["feature_names"])
        self.assertNotIn("label_scope", payload["feature_names"])
        self.assertNotIn("is_synthetic", payload["feature_names"])
        self.assertLessEqual(set(np.unique(payload["traj"]["s"]).astype(int)), set(range(8)))
        self.assertTrue(np.isfinite(payload["X"]).all())
        self.assertTrue(np.isfinite(payload["traj"]["s"]).all())

    def test_apibank_converter_uses_existing_numerical_api_rows_only(self):
        output_path = self.tmp_path / "apibank.pkl"
        manifest_path = self.tmp_path / "apibank_manifest.json"
        payload, manifest = self.apibank.convert(
            output_path=output_path,
            manifest_path=manifest_path,
            write_outputs=True,
        )

        self.assertTrue(output_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertEqual(payload["X"].shape, (1016, 402))
        self.assertEqual(payload["X"].dtype, np.float32)
        self.assertEqual(payload["y"].shape, (1016,))
        self.assertEqual(payload["y"].dtype, np.int64)
        self.assertEqual(payload["traj"]["s"].shape, (1016, 404))
        self.assertEqual(payload["traj"]["s"].dtype, np.float32)
        self.assertEqual(
            manifest["array_contract"]["class_distribution"], {"0": 508, "1": 508}
        )
        self.assertEqual(manifest["split"]["seed"], 1)
        self.assertEqual(manifest["split"]["train_size"], 812)
        self.assertEqual(manifest["split"]["validation_size"], 204)
        self.assertIn("group_ids", payload)
        self.assertEqual(len(payload["group_ids"]), 1016)
        self.assertEqual(len(set(payload["group_ids"])), 508)
        self.assertEqual(
            manifest["pair_grouping"]["pair_size_distribution"], {"2": 508}
        )
        self.assertEqual(manifest["pair_grouping"]["pair_count"], 508)
        self.assertEqual(manifest["array_contract"]["duplicate_sample_ids"], [])
        self.assertEqual(
            manifest["array_contract"]["nan_counts"], {"X": 0, "traj_s": 0}
        )
        self.assertEqual(
            manifest["array_contract"]["infinite_value_counts"],
            {"X": 0, "traj_s": 0},
        )
        self.assertEqual(manifest["label_scope"], "api_call_level")
        self.assertEqual(
            manifest["label_origin"],
            {"reference_api_call": 508, "synthetic_corruption": 508},
        )
        self.assertEqual(manifest["synthetic_counts"], {"False": 508, "True": 508})

    def test_apibank_metadata_stays_outside_model_arrays(self):
        payload, manifest = self.apibank.convert(write_outputs=False)
        self.assertIn("metadata", payload)
        self.assertTrue(all(row["source_dataset"] == "api_bank" for row in payload["metadata"]))
        self.assertFalse(
            {
                "label_scope",
                "label_origin",
                "is_synthetic",
                "pair_id",
                "group_ids",
                "corruption_type",
                "validation_status",
                "validation_error",
                "y",
            }
            & set(
                manifest["feature_definitions"]["X"][
                    "ordered_structural_feature_names"
                ]
            )
        )
        self.assertTrue(np.isfinite(payload["X"]).all())
        self.assertTrue(np.isfinite(payload["traj"]["s"]).all())


if __name__ == "__main__":
    unittest.main()

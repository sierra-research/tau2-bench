import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_SCRIPT_PATH = REPO_ROOT / "scripts" / "ingest_tau2_stage1_results.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Tau2Stage1IngestionTest(unittest.TestCase):
    def setUp(self):
        self.ingest = load_module("ingest_tau2_stage1_results", INGEST_SCRIPT_PATH)
        self.tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self.tmpdir.name)
        self.raw_dir = tmp / "raw"
        self.manifest_path = tmp / "manifest.json"
        self.retained_jsonl = tmp / "retained.jsonl"
        self.summary_json = tmp / "summary.json"
        write_json(
            self.manifest_path,
            {
                "tasks": [
                    {
                        "task_id": "50",
                        "domain": "retail",
                        "selection_group": "low_action_one_write",
                    }
                ]
            },
        )
        self.raw_payload = {
            "info": {"environment_info": {"domain_name": "retail"}},
            "tasks": [
                {
                    "id": "50",
                    "evaluation_criteria": {
                        "actions": [],
                        "reward_basis": ["DB"],
                    },
                }
            ],
            "simulations": [
                {
                    "task_id": "50",
                    "termination_reason": "user_stop",
                    "reward_info": {
                        "reward": 1.0,
                        "db_check": {"db_match": True},
                        "action_checks": [],
                        "reward_basis": ["DB"],
                    },
                    "messages": [
                        {"role": "user", "content": "hello", "turn_idx": 0},
                        {"role": "assistant", "content": "done", "turn_idx": 1},
                    ],
                }
            ],
        }
        write_json(self.raw_dir / "task_50.json", self.raw_payload)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_completed_raw_outputs_are_ingested_and_labeled(self):
        retained, summary = self.ingest.build_outputs(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            retained_jsonl=self.retained_jsonl,
            summary_json=self.summary_json,
            write_outputs=True,
        )
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["y"], 1)
        self.assertEqual(retained[0]["label_origin"], "tau2_benchmark_reward")
        self.assertEqual(summary["attempted_count"], 1)
        self.assertEqual(summary["completed_count"], 1)
        self.assertEqual(summary["retained_count"], 1)
        self.assertEqual(summary["filtered_count"], 0)
        self.assertEqual(summary["y_distribution"], {"1": 1})
        self.assertEqual(
            summary["counts_by_stage1_selection_group"],
            {"low_action_one_write": 1},
        )
        self.assertTrue(self.retained_jsonl.exists())
        self.assertTrue(self.summary_json.exists())

    def test_original_filtering_logic_is_reused(self):
        _, summary = self.ingest.build_outputs(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            retained_jsonl=self.retained_jsonl,
            summary_json=self.summary_json,
            write_outputs=False,
        )
        self.assertIn(
            "convert_tau2_results_to_l2t_pkl.get_exclusion_reasons",
            summary["original_filtering_logic_reused"]["get_exclusion_reasons"],
        )
        self.assertEqual(
            summary["original_filtering_logic_reused"]["label_rule"],
            "y = 1 only when reward == 1.0, else y = 0",
        )

    def test_filtered_outputs_report_reasons(self):
        self.raw_payload["simulations"][0]["termination_reason"] = "max_steps"
        write_json(self.raw_dir / "task_50.json", self.raw_payload)
        retained, summary = self.ingest.build_outputs(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            retained_jsonl=self.retained_jsonl,
            summary_json=self.summary_json,
            write_outputs=False,
        )
        self.assertEqual(retained, [])
        self.assertEqual(summary["filtered_count"], 1)
        self.assertEqual(summary["filter_reasons"], {"non_normal_stop:max_steps": 1})

    def test_ingestion_cli_does_not_expose_analysis_update_mode(self):
        with mock.patch.object(sys, "argv", ["ingest_tau2_stage1_results.py"]):
            args = self.ingest.parse_args()
        self.assertFalse(hasattr(args, "update_analysis"))
        self.assertFalse(hasattr(self.ingest, "build_stage1_analysis"))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_tau2_stage1_manifest.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Tau2Stage1ManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_module("build_tau2_stage1_manifest", SCRIPT_PATH)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.output_json = Path(cls.tmpdir.name) / "manifest.json"
        cls.report_path = Path(cls.tmpdir.name) / "manifest.md"
        cls.manifest = cls.builder.build_outputs(
            output_json=cls.output_json,
            report_path=cls.report_path,
            write_outputs=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_correct_12_task_composition(self):
        tasks = self.manifest["tasks"]
        self.assertEqual(len(tasks), 12)
        self.assertEqual(
            self.manifest["composition"],
            {"low_action_one_write": 2, "no_write": 2, "two_plus_writes": 8},
        )

    def test_no_duplicate_task_ids(self):
        task_ids = [task["task_id"] for task in self.manifest["tasks"]]
        self.assertEqual(len(task_ids), len(set(task_ids)))

    def test_no_prior_attempted_or_retained_tasks_selected(self):
        for task in self.manifest["tasks"]:
            self.assertFalse(task["previously_attempted"])
            self.assertFalse(task["previously_retained"])

    def test_no_outcome_based_selection(self):
        self.assertFalse(self.manifest["metadata"]["selection_uses_outcome_labels"])
        self.assertIn(
            "no y, reward, or success/failure",
            self.manifest["metadata"]["selection_policy"],
        )
        for task in self.manifest["tasks"]:
            self.assertFalse(task["selection_uses_outcome_label"])

    def test_selected_tasks_exist_in_local_retail_definitions(self):
        local_tasks = {
            str(task["id"])
            for task in self.builder.load_json(self.builder.DOMAIN_TASK_FILES["retail"])
        }
        for task in self.manifest["tasks"]:
            self.assertEqual(task["domain"], "retail")
            self.assertIn(task["task_id"], local_tasks)

    def test_run_controls_are_recorded(self):
        run = self.manifest["run"]
        self.assertIn("--num-trials 1", run["illustrative_batch_tau2_command_text"])
        self.assertIn(
            "--max-concurrency 1", run["illustrative_batch_tau2_command_text"]
        )
        self.assertIn("authoritative execution path", run["execution_path"])
        self.assertIn("--execute", run["runner_execute_command"])
        self.assertGreater(run["estimated_maximum_llm_calls"], 0)
        self.assertTrue(self.output_json.exists())
        self.assertTrue(self.report_path.exists())


if __name__ == "__main__":
    unittest.main()

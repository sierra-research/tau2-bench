import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCRIPT_PATH = REPO_ROOT / "scripts" / "build_tau2_stage1_manifest.py"
RUNNER_SCRIPT_PATH = REPO_ROOT / "scripts" / "run_tau2_stage1.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Tau2Stage1RunnerTest(unittest.TestCase):
    def setUp(self):
        self.builder = load_module("build_tau2_stage1_manifest", MANIFEST_SCRIPT_PATH)
        self.runner = load_module("run_tau2_stage1", RUNNER_SCRIPT_PATH)
        self.tmpdir = tempfile.TemporaryDirectory()
        tmp = Path(self.tmpdir.name)
        self.manifest_path = tmp / "manifest.json"
        self.status_path = tmp / "status.json"
        self.raw_dir = tmp / "raw"
        self.manifest = self.builder.build_outputs(
            output_json=self.manifest_path,
            report_path=tmp / "manifest.md",
            write_outputs=True,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dry_run_is_default_and_does_not_call_subprocess(self):
        mock_run = mock.Mock()
        status = self.runner.run_stage1(
            manifest_path=self.manifest_path,
            status_path=self.status_path,
            raw_dir=self.raw_dir,
            subprocess_run=mock_run,
        )
        self.assertTrue(status["metadata"]["dry_run"])
        self.assertEqual(status["metadata"]["dry_run_count"], 12)
        self.assertEqual(len(status["tasks"]), 12)
        mock_run.assert_not_called()

    def test_parse_args_accepts_task_id(self):
        with mock.patch.object(
            sys,
            "argv",
            ["run_tau2_stage1.py", "--task-id", "54"],
        ):
            args = self.runner.parse_args()
        self.assertEqual(args.task_id, "54")
        self.assertFalse(args.execute)

    def test_task_54_filter_selects_only_one_task(self):
        mock_run = mock.Mock()
        status = self.runner.run_stage1(
            manifest_path=self.manifest_path,
            status_path=self.status_path,
            raw_dir=self.raw_dir,
            task_id="54",
            subprocess_run=mock_run,
        )
        self.assertEqual(list(status["tasks"]), ["54"])
        self.assertEqual(status["metadata"]["selected_task_ids"], ["54"])
        self.assertEqual(status["metadata"]["dry_run_count"], 1)
        self.assertEqual(status["tasks"]["54"]["status"], "dry_run")
        self.assertEqual(
            self.runner.select_tasks(self.manifest, task_id="54")[0]["selection_group"],
            "two_plus_writes",
        )
        mock_run.assert_not_called()

    def test_task_id_outside_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not present in the manifest"):
            self.runner.run_stage1(
                manifest_path=self.manifest_path,
                status_path=self.status_path,
                raw_dir=self.raw_dir,
                task_id="999999",
                subprocess_run=mock.Mock(),
            )

    def test_status_json_records_only_selected_task_for_canary_dry_run(self):
        self.runner.run_stage1(
            manifest_path=self.manifest_path,
            status_path=self.status_path,
            raw_dir=self.raw_dir,
            task_id="54",
            subprocess_run=mock.Mock(),
        )
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertEqual(list(status["tasks"]), ["54"])
        self.assertEqual(status["metadata"]["dry_run_count"], 1)

    def test_execute_is_required_for_actual_runs(self):
        task = self.manifest["tasks"][0]
        command = self.runner.run_task(
            task,
            raw_dir=self.raw_dir,
            timeout=None,
            execute=False,
            subprocess_run=mock.Mock(),
        )
        self.assertEqual(command["status"], "dry_run")

    def test_completed_tasks_are_resumable(self):
        task_id = self.manifest["tasks"][0]["task_id"]
        write_json(
            self.raw_dir / f"task_{task_id}.json",
            {
                "simulations": [
                    {
                        "task_id": task_id,
                        "termination_reason": "user_stop",
                        "reward_info": {"reward": 1.0},
                    }
                ]
            },
        )
        mock_run = mock.Mock()
        record = self.runner.run_task(
            self.manifest["tasks"][0],
            raw_dir=self.raw_dir,
            timeout=None,
            execute=True,
            subprocess_run=mock_run,
        )
        self.assertEqual(record["status"], "skipped_completed")
        self.assertEqual(record["reward"], 1.0)
        mock_run.assert_not_called()

    def test_completed_task_54_is_skipped_in_later_full_run(self):
        write_json(
            self.raw_dir / "task_54.json",
            {
                "simulations": [
                    {
                        "task_id": "54",
                        "termination_reason": "agent_stop",
                        "reward_info": {"reward": 1.0},
                    }
                ]
            },
        )
        mock_run = mock.Mock(
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="fail")
        )
        status = self.runner.run_stage1(
            manifest_path=self.manifest_path,
            status_path=self.status_path,
            raw_dir=self.raw_dir,
            execute=True,
            subprocess_run=mock_run,
        )
        self.assertEqual(status["tasks"]["54"]["status"], "skipped_completed")
        self.assertEqual(status["tasks"]["54"]["reward"], 1.0)
        called_command = mock_run.call_args.args[0]
        task_ids_index = called_command.index("--task-ids")
        self.assertEqual(called_command[task_ids_index + 1], "55")

    def test_execute_stops_when_observed_cost_reaches_cap(self):
        native_root = Path(self.tmpdir.name) / "native"

        def fake_native_results_path(task_id):
            return native_root / f"task_{task_id}" / "results.json"

        def fake_run(command, **kwargs):
            task_id = command[command.index("--task-ids") + 1]
            write_json(
                fake_native_results_path(task_id),
                {
                    "simulations": [
                        {
                            "task_id": task_id,
                            "termination_reason": "agent_stop",
                            "reward_info": {"reward": 1.0, "cost": 0.03},
                        }
                    ]
                },
            )
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch.object(
            self.runner,
            "native_results_path",
            side_effect=fake_native_results_path,
        ):
            status = self.runner.run_stage1(
                manifest_path=self.manifest_path,
                status_path=self.status_path,
                raw_dir=self.raw_dir,
                execute=True,
                max_total_cost=0.02,
                subprocess_run=mock.Mock(side_effect=fake_run),
            )

        self.assertEqual(status["metadata"]["stop_reason"], "max_total_cost_reached")
        self.assertEqual(status["metadata"]["total_observed_cost"], 0.03)
        self.assertEqual(len(status["tasks"]), 1)
        first_task = next(iter(status["tasks"].values()))
        self.assertEqual(first_task["status"], "completed")

    def test_execute_stops_after_first_error_without_continue_flag(self):
        mock_run = mock.Mock(
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="fail")
        )
        status = self.runner.run_stage1(
            manifest_path=self.manifest_path,
            status_path=self.status_path,
            raw_dir=self.raw_dir,
            execute=True,
            continue_on_error=False,
            subprocess_run=mock_run,
        )

        first_task_id = self.manifest["tasks"][0]["task_id"]
        self.assertEqual(
            status["metadata"]["stop_reason"], f"task_{first_task_id}_error"
        )
        self.assertEqual(list(status["tasks"]), [first_task_id])
        self.assertEqual(status["tasks"][first_task_id]["status"], "error")
        self.assertEqual(mock_run.call_count, 1)

    def test_raw_outputs_are_preserved_after_execution(self):
        task_id = self.manifest["tasks"][0]["task_id"]
        native_path = Path(self.tmpdir.name) / "native" / "results.json"
        native_payload = {
            "simulations": [
                {
                    "task_id": task_id,
                    "termination_reason": "agent_stop",
                    "reward_info": {"reward": 0.0},
                }
            ]
        }

        def fake_run(*args, **kwargs):
            write_json(native_path, native_payload)
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")

        with mock.patch.object(
            self.runner, "native_results_path", return_value=native_path
        ):
            record = self.runner.run_task(
                self.manifest["tasks"][0],
                raw_dir=self.raw_dir,
                timeout=None,
                execute=True,
                subprocess_run=fake_run,
            )

        raw_path = self.raw_dir / f"task_{task_id}.json"
        self.assertEqual(record["status"], "completed")
        self.assertTrue(raw_path.exists())
        self.assertEqual(
            json.loads(raw_path.read_text(encoding="utf-8")), native_payload
        )
        self.assertIn("--task-ids", record["command"])


if __name__ == "__main__":
    unittest.main()

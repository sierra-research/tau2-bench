import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_SCRIPT_PATH = REPO_ROOT / "scripts" / "plan_tau2_additional_sampling.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Tau2AdditionalSamplingPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner = load_module("plan_tau2_additional_sampling", PLANNER_SCRIPT_PATH)
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.output_json = Path(cls.tmpdir.name) / "tau2_additional_sampling_plan.json"
        cls.report_path = Path(cls.tmpdir.name) / "tau2_additional_sampling_plan.md"
        cls.plan = cls.planner.build_outputs(
            output_json=cls.output_json,
            report_path=cls.report_path,
            write_outputs=True,
        )
        cls.uncertainty_rows = cls.planner.load_jsonl(cls.planner.DEFAULT_UNCERTAINTY_JSONL)
        cls.local_tasks = {
            domain: {
                str(task["id"])
                for task in json.loads(path.read_text(encoding="utf-8"))
            }
            for domain, path in cls.planner.DOMAIN_TASK_FILES.items()
            if path.exists()
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def shift_by_id(self, shift_id):
        return next(row for row in self.plan["shifts"] if row["shift_id"] == shift_id)

    def test_all_six_tau2_shifts_are_included_and_api_bank_is_excluded(self):
        shift_ids = [row["shift_id"] for row in self.plan["shifts"]]
        self.assertEqual(len(shift_ids), 6)
        self.assertTrue(all(shift_id.startswith("tau2_") for shift_id in shift_ids))
        self.assertFalse(any("api_bank" in shift_id for shift_id in shift_ids))
        self.assertTrue(self.plan["metadata"]["api_bank_excluded"])

    def test_current_counts_are_preserved_from_uncertainty_inputs(self):
        uncertainty_by_id = {row["shift_id"]: row for row in self.uncertainty_rows}
        for row in self.plan["shifts"]:
            source = uncertainty_by_id[row["shift_id"]]
            self.assertEqual(row["source_n"], source["source_n"])
            self.assertEqual(row["target_n"], source["target_n"])
            self.assertEqual(row["classification_by_threshold"], source["classification_by_threshold"])

    def test_calculations_are_deterministic(self):
        repeat = self.planner.build_outputs(write_outputs=False)
        self.assertEqual(self.plan["shifts"], repeat["shifts"])
        self.assertEqual(
            self.plan["proposed_next_batch"]["stage_1_tasks"],
            repeat["proposed_next_batch"]["stage_1_tasks"],
        )

    def test_requested_precision_and_power_scenarios_are_present(self):
        for row in self.plan["shifts"]:
            self.assertEqual(set(row["precision_planning"]), {"0.15", "0.10", "0.05"})
            self.assertEqual(set(row["power_planning"]), {"0.15", "0.10", "0.05"})
            self.assertEqual(set(row["equivalence_planning"]), {"0.15", "0.10", "0.05"})

    def test_planning_estimates_are_finite_or_explicitly_unavailable(self):
        for row in self.plan["shifts"]:
            for planning_key in ["precision_planning", "power_planning", "equivalence_planning"]:
                for estimate in row[planning_key].values():
                    if estimate["available"]:
                        self.assertGreaterEqual(estimate["required_final_n_per_group"], 0)
                        self.assertGreaterEqual(estimate["additional_source_n"], 0)
                        self.assertGreaterEqual(estimate["additional_target_n"], 0)
                        self.assertGreaterEqual(estimate["additional_total_n"], 0)
                    else:
                        self.assertIn("reason", estimate)
                        self.assertTrue(estimate["reason"])

    def test_no_task_selection_uses_y(self):
        self.assertIn("observed y is not used", self.plan["task_pool_audit"]["task_selection_policy"])
        for task in self.plan["proposed_next_batch"]["stage_1_tasks"]:
            self.assertFalse(task["uses_y"])
            self.assertNotIn("y", task)
        for shift_audit in self.plan["task_pool_audit"]["shift_audits"].values():
            self.assertTrue(shift_audit["task_ids_selectable_without_y"])
            for task in shift_audit["unused_tasks_in_smaller_group"]:
                self.assertFalse(task["uses_y"])

    def test_recommended_tasks_come_from_valid_local_task_definitions(self):
        for task in self.plan["proposed_next_batch"]["stage_1_tasks"]:
            self.assertIn(task["domain"], self.local_tasks)
            self.assertIn(task["task_id"], self.local_tasks[task["domain"]])
        self.assertEqual(self.plan["proposed_next_batch"]["stage_1_batch_size"], 12)

    def test_task_pool_audit_records_retail_and_airline_availability(self):
        domains = self.plan["task_pool_audit"]["domains"]
        self.assertEqual(domains["retail"]["local_task_count"], 114)
        self.assertEqual(domains["retail"]["retained_outcome_count"], 46)
        self.assertEqual(domains["retail"]["unused_by_retained_outcome_count"], 68)
        self.assertEqual(domains["airline"]["local_task_count"], 50)
        self.assertTrue(self.plan["task_pool_audit"]["airline_all_tasks_previously_sampled"])

    def test_telecom_warning_is_included_in_json_and_report(self):
        warning = self.planner.TELECOM_WARNING
        self.assertEqual(self.plan["cost_and_runtime_constraints"]["telecom_warning"], warning)
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn(warning, report)

    def test_output_files_are_written(self):
        self.assertTrue(self.output_json.exists())
        self.assertTrue(self.report_path.exists())
        report = self.report_path.read_text(encoding="utf-8")
        self.assertIn("# Tau2 Additional Sampling Plan", report)
        self.assertIn("## Proposed next batch", report)


if __name__ == "__main__":
    unittest.main()

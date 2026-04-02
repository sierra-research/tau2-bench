"""Integration tests: end-to-end pipeline from Results -> Report."""


from tau2_reliability.extract import extract_task_trial_data
from tau2_reliability.metrics.consistency import compute_all_consistency
from tau2_reliability.models import ReliabilityReport
from tau2_reliability.report import generate_csv, generate_markdown_report


class TestFullPipeline:
    def test_analyze_3_tasks_5_trials(self, make_sim, make_results, tmp_path):
        """Full pipeline: 3 tasks x 5 trials -> report with all metrics."""
        sims = []
        # Task 1: always succeeds, same actions
        for t in range(5):
            sims.append(make_sim(
                task_id="t1", trial=t, reward=1.0,
                action_names=["search", "book"], cost=0.1, duration=30,
            ))
        # Task 2: always fails, same actions
        for t in range(5):
            sims.append(make_sim(
                task_id="t2", trial=t, reward=0.0,
                action_names=["search"], cost=0.05, duration=20,
            ))
        # Task 3: bimodal (3 success, 2 failure, different actions)
        for t in range(3):
            sims.append(make_sim(
                task_id="t3", trial=t, reward=1.0,
                action_names=["search", "book", "confirm"], cost=0.15, duration=45,
            ))
        for t in range(3, 5):
            sims.append(make_sim(
                task_id="t3", trial=t, reward=0.0,
                action_names=["search", "search", "search"], cost=0.2, duration=60,
            ))

        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert len(task_data) == 3

        consistency = compute_all_consistency(task_data)

        # Task 1 and 2 should be consistent, task 3 bimodal
        assert consistency.per_task["t1"]["c_out"] > 0.9
        assert consistency.per_task["t2"]["c_out"] > 0.9
        assert consistency.per_task["t3"]["c_out"] < 0.3

        # Aggregate should be between
        assert 0.0 < consistency.r_con < 1.0

    def test_report_markdown_has_sections(self, make_sim, make_results):
        sims = [
            make_sim(task_id="t1", trial=0, reward=1.0),
            make_sim(task_id="t1", trial=1, reward=0.0),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        consistency = compute_all_consistency(task_data)
        report = ReliabilityReport(
            domain="airline", num_tasks=1, num_trials=2,
            accuracy=0.5, consistency=consistency, r_con=consistency.r_con,
        )
        md = generate_markdown_report(report)
        assert "# Reliability Report" in md
        assert "Consistency" in md
        assert "C_out" in md

    def test_report_csv_output(self, make_sim, make_results, tmp_path):
        sims = [
            make_sim(task_id="t1", trial=0),
            make_sim(task_id="t1", trial=1),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        consistency = compute_all_consistency(task_data)
        report = ReliabilityReport(
            consistency=consistency, r_con=consistency.r_con,
        )
        csv_path = tmp_path / "metrics.csv"
        generate_csv(report, csv_path)
        assert csv_path.exists()
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert "task_id" in df.columns
        assert "c_out" in df.columns

    def test_single_task_works(self, make_sim, make_results):
        sims = [make_sim(task_id="t1", trial=0, reward=1.0)]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        consistency = compute_all_consistency(task_data)
        assert consistency.c_out == 1.0

    def test_infra_errors_filtered(self, make_sim, make_results):
        from tau2.data_model.simulation import TerminationReason
        sims = [
            make_sim(task_id="t1", trial=0, reward=1.0),
            make_sim(task_id="t1", trial=1, reward=1.0,
                     termination=TerminationReason.INFRASTRUCTURE_ERROR),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert task_data[0].num_trials == 1

    def test_action_extraction_from_messages(self, make_sim, make_results):
        sims = [make_sim(
            task_id="t1", trial=0,
            action_names=["get_user", "get_reservation", "cancel_reservation"],
        )]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert task_data[0].action_sequences[0] == [
            "get_user", "get_reservation", "cancel_reservation"
        ]

import json
import math
import os
import subprocess
import sys
from fractions import Fraction

import pytest

from experiments.bayes_at_n import BayesAtNResult, compute_bayes_at_n
from tau2.data_model.elicitation import ElicitationConfig
from tau2.data_model.language import LegacyLanguageContext
from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.metrics.agent_metrics import compute_metrics


@pytest.fixture
def make_results(base_task):
    def make(rewards):
        num_trials = len(rewards[0])
        tasks = [
            base_task.model_copy(deep=True, update={"id": f"task-{i}"})
            for i in range(len(rewards))
        ]
        return Results(
            info=Info(
                git_commit="recorded-source-revision",
                num_trials=num_trials,
                max_steps=100,
                max_errors=10,
                user_info={"implementation": "user_simulator", "llm": "fixed-user"},
                agent_info={
                    "implementation": "llm_agent",
                    "llm": "fixed-agent",
                    "llm_args": {"temperature": 0.7},
                },
                environment_info={"domain_name": "mock", "policy": "fixed policy"},
            ),
            tasks=tasks,
            simulations=[
                SimulationRun(
                    id=f"sim-{i}-{trial}",
                    task_id=task.id,
                    trial=trial,
                    seed=100 + trial,
                    start_time="2026-01-01T00:00:00",
                    end_time="2026-01-01T00:00:01",
                    duration=1.0,
                    termination_reason=TerminationReason.USER_STOP,
                    reward_info=RewardInfo(reward=reward),
                    messages=[],
                    agent_cost=0.1,
                )
                for i, (task, row) in enumerate(zip(tasks, rewards))
                for trial, reward in enumerate(row)
            ],
        )

    return make


@pytest.mark.parametrize(
    ("outcomes", "mean", "variance"),
    [
        ([[0, 0], [1, 1]], Fraction(1, 2), Fraction(3, 160)),
        ([[0, 1], [0, 1]], Fraction(1, 2), Fraction(1, 40)),
        ([[0]], Fraction(1, 3), Fraction(1, 18)),
        ([[1]], Fraction(2, 3), Fraction(1, 18)),
        ([[0] * 5] * 3, Fraction(1, 7), Fraction(1, 196)),
        ([[1] * 5] * 3, Fraction(6, 7), Fraction(1, 196)),
    ],
)
def test_exact_beta_moments(make_results, outcomes, mean, variance):
    summary = compute_bayes_at_n(make_results(outcomes))

    assert summary.posterior_mean == pytest.approx(float(mean), abs=1e-14)
    assert summary.posterior_std == pytest.approx(math.sqrt(variance), abs=1e-14)
    assert summary.num_tasks == len(outcomes)
    assert summary.num_trials == len(outcomes[0])


def test_mixed_counts_match_exact_beta_second_moments(make_results):
    counts = [0, 1, 6, 17, 31]
    num_trials = 31
    outcomes = [[1] * count + [0] * (num_trials - count) for count in counts]
    means = [Fraction(count + 1, num_trials + 2) for count in counts]
    second_moments = [
        Fraction((count + 1) * (count + 2), (num_trials + 2) * (num_trials + 3))
        for count in counts
    ]
    expected_variance = (
        sum(second - mean**2 for second, mean in zip(second_moments, means))
        / len(counts) ** 2
    )

    summary = compute_bayes_at_n(make_results(outcomes))

    assert summary.posterior_mean == pytest.approx(
        float(sum(means) / len(counts)), abs=1e-14
    )
    assert summary.posterior_std == pytest.approx(
        math.sqrt(expected_variance), abs=1e-14
    )


def test_full_success_uses_existing_reward_epsilon(make_results):
    results = make_results([[0.5, 1 - 0.5e-6, 1 + 0.5e-6, 1 - 2e-6]])

    summary = compute_bayes_at_n(results)

    assert [trial.success for trial in summary.tasks[0].outcomes] == [
        False,
        True,
        True,
        False,
    ]
    assert summary.tasks[0].success_count == 2
    assert summary.posterior_mean == 0.5


def test_order_identity_and_input_preservation(make_results):
    results = make_results([[0, 0], [1, 1]])
    before = results.model_dump_json()
    original_metrics = compute_metrics(results).as_dict()
    summary = compute_bayes_at_n(results)
    reordered = results.model_copy(deep=True)
    reordered.tasks.reverse()
    reordered.simulations.reverse()

    assert compute_bayes_at_n(reordered) == summary
    assert results.model_dump_json() == before
    assert compute_metrics(results).as_dict() == original_metrics
    assert original_metrics["pass_hat_1"] == 0.5
    assert original_metrics["pass_hat_2"] == 0.5
    assert summary.info == results.info
    assert summary.tasks[1].task_id == "task-1"
    assert summary.tasks[1].outcomes[1].model_dump() == {
        "simulation_id": "sim-1-1",
        "trial": 1,
        "seed": 101,
        "success": True,
    }
    assert BayesAtNResult.model_validate_json(summary.model_dump_json()) == summary
    summary.info.agent_info.llm_args["temperature"] = 0.0
    assert results.info.agent_info.llm_args["temperature"] == 0.7


def test_accepts_matching_recorded_task_contexts(make_results):
    results = make_results([[0, 1]])
    context = LegacyLanguageContext(language="en", canonical_task_id="task-0")
    results.info.language_tasks = {"task-0": context}
    for simulation in results.simulations:
        simulation.language_context = context.model_copy(deep=True)
        simulation.policy = "task-specific fixed policy"

    summary = compute_bayes_at_n(results)

    assert summary.info.language_tasks == results.info.language_tasks
    assert summary.posterior_mean == 0.5


@pytest.mark.parametrize("num_trials", [0, -1])
def test_rejects_invalid_declared_trial_count(make_results, num_trials):
    results = make_results([[0, 1]])
    results.info.num_trials = num_trials
    with pytest.raises(ValueError, match="positive integer"):
        compute_bayes_at_n(results)


@pytest.mark.parametrize("cohort", ["empty", "duplicate", "wholly_missing", "unknown"])
def test_rejects_invalid_task_cohort(make_results, cohort):
    results = make_results([[0, 1], [1, 0]])
    if cohort == "empty":
        results.tasks = []
        match = "nonempty task cohort"
    elif cohort == "duplicate":
        results.tasks.append(results.tasks[0].model_copy(deep=True))
        match = "duplicate task IDs"
    elif cohort == "wholly_missing":
        results.simulations = results.simulations[:2]
        match = "task-1.*expected 2 trials.*found 0"
    else:
        results.simulations[0].task_id = "unselected-task"
        match = "Unknown simulation task ID"
    with pytest.raises(ValueError, match=match):
        compute_bayes_at_n(results)


@pytest.mark.parametrize("trial", [None, -1, 2])
def test_rejects_invalid_trial_indices(make_results, trial):
    results = make_results([[0, 1]])
    results.simulations[0].trial = trial
    with pytest.raises(ValueError, match="trial index in 0..1"):
        compute_bayes_at_n(results)


def test_rejects_partial_trials(make_results):
    results = make_results([[0, 1], [1, 0]])
    results.simulations.pop()
    with pytest.raises(ValueError, match="task-1.*expected 2 trials.*found 1"):
        compute_bayes_at_n(results)


def test_rejects_duplicate_task_trial_even_with_new_id_and_seed(make_results):
    results = make_results([[0, 1]])
    results.simulations[1].trial = 0
    with pytest.raises(ValueError, match="Duplicate task/trial"):
        compute_bayes_at_n(results)


@pytest.mark.parametrize("simulation_id", ["", "sim-0-0"])
def test_rejects_invalid_simulation_ids(make_results, simulation_id):
    results = make_results([[0, 1], [1, 0]])
    results.simulations[2].id = simulation_id
    with pytest.raises(ValueError, match="Simulation IDs.*nonempty and unique"):
        compute_bayes_at_n(results)


@pytest.mark.parametrize("seed", [None, 100])
def test_rejects_absent_or_reused_seeds(make_results, seed):
    results = make_results([[0, 1]])
    results.simulations[1].seed = seed
    with pytest.raises(ValueError, match="recorded seed|reuses a seed"):
        compute_bayes_at_n(results)


@pytest.mark.parametrize("reward", [None, math.nan, math.inf, -math.inf])
def test_rejects_ungraded_or_nonfinite_rewards(make_results, reward):
    results = make_results([[0, 1]])
    if reward is None:
        results.simulations[0].reward_info = None
    else:
        results.simulations[0].reward_info.reward = reward
    with pytest.raises(ValueError, match="finite graded reward"):
        compute_bayes_at_n(results)


def test_rejects_infrastructure_errors_instead_of_dropping_them(make_results):
    results = make_results([[0, 1]])
    results.simulations[0].termination_reason = TerminationReason.INFRASTRUCTURE_ERROR
    with pytest.raises(ValueError, match="Infrastructure errors.*incomplete cohort"):
        compute_bayes_at_n(results)


@pytest.mark.parametrize("mode_change", ["mode", "ticks", "audio_config"])
def test_rejects_voice_results(make_results, mode_change):
    results = make_results([[0, 1]])
    if mode_change == "mode":
        results.simulations[0].mode = "full_duplex"
    elif mode_change == "ticks":
        results.simulations[0].ticks = []
    else:
        from tau2.data_model.run_config import AudioNativeConfig

        results.info.audio_native_config = AudioNativeConfig()
    with pytest.raises(ValueError, match="text half-duplex"):
        compute_bayes_at_n(results)


def test_rejects_elicitation_and_feedback_retries(make_results):
    results = make_results([[0, 1]])
    results.info.elicitation = ElicitationConfig()
    with pytest.raises(ValueError, match="Elicitation experiments"):
        compute_bayes_at_n(results)
    results.info.elicitation = None
    results.simulations[1].hallucination_retries_used = 1
    with pytest.raises(ValueError, match="Hallucination-feedback retries"):
        compute_bayes_at_n(results)


def test_rejects_changing_task_inputs(make_results):
    results = make_results([[0, 1]])
    results.simulations[1].policy = "changed policy"
    with pytest.raises(ValueError, match="changes policy or language context"):
        compute_bayes_at_n(results)
    results.simulations[1].policy = None
    results.simulations[1].language_context = LegacyLanguageContext(
        language="es", canonical_task_id="task-0"
    )
    with pytest.raises(ValueError, match="changes policy or language context"):
        compute_bayes_at_n(results)


def test_rejects_context_disagreeing_with_recorded_info(make_results):
    results = make_results([[0, 1]])
    results.info.language_tasks = {
        "task-0": LegacyLanguageContext(language="en", canonical_task_id="task-0")
    }
    with pytest.raises(ValueError, match="language context differs from recorded Info"):
        compute_bayes_at_n(results)
    results.info.language_tasks["unselected-task"] = results.info.language_tasks[
        "task-0"
    ]
    with pytest.raises(ValueError, match="contexts differ from the task cohort"):
        compute_bayes_at_n(results)


def _run_cli(*args):
    env = os.environ.copy()
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    return subprocess.run(
        [sys.executable, "-m", "experiments.bayes_at_n", *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize("storage_format", ["json", "dir"])
def test_cli_load_export_and_input_preservation(make_results, tmp_path, storage_format):
    results = make_results([[0, 0], [1, 1]])
    source = tmp_path / "input" / "results.json"
    source.parent.mkdir()
    results.save(source, format=storage_format)
    input_bytes = {
        file: file.read_bytes() for file in source.parent.rglob("*") if file.is_file()
    }
    output = tmp_path / "analysis" / "bayes.json"

    process = _run_cli("--results", source, "--output", output)

    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == json.loads(output.read_text())
    summary = BayesAtNResult.model_validate_json(output.read_text())
    assert summary.posterior_mean == 0.5
    assert summary.posterior_std == pytest.approx(math.sqrt(Fraction(3, 160)))
    assert summary.info == results.info
    assert {task.task_id for task in summary.tasks} == {"task-0", "task-1"}
    assert summary.num_trials == 2
    assert all(len(task.outcomes) == 2 for task in summary.tasks)
    assert {file: file.read_bytes() for file in input_bytes} == input_bytes
    if storage_format == "dir":
        directory_process = _run_cli("--results", source.parent)
        assert directory_process.returncode == 0, directory_process.stderr
        assert json.loads(directory_process.stdout) == json.loads(process.stdout)


def test_cli_rejects_incomplete_results_and_refuses_overwrites(make_results, tmp_path):
    results = make_results([[0, 1]])
    source = tmp_path / "results.json"
    results.save(source)
    original = source.read_bytes()

    overwrite = _run_cli("--results", source, "--output", source)

    assert overwrite.returncode == 2
    assert "File exists" in overwrite.stderr
    assert source.read_bytes() == original
    results.simulations.pop()
    results.save(source)
    output = tmp_path / "bayes.json"

    incomplete = _run_cli("--results", source, "--output", output)

    assert incomplete.returncode == 2
    assert "expected 2 trials" in incomplete.stderr
    assert not output.exists()

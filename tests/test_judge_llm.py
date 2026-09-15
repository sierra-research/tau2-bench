import argparse
import json
from types import SimpleNamespace

import pytest

from tau2.cli import add_run_args
from tau2.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS
from tau2.data_model.simulation import TerminationReason, TextRunConfig
from tau2.data_model.tasks import RewardType
from tau2.evaluator import evaluator_nl_assertions
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2.runner import batch
from tau2.runner import simulation as runner_simulation

NL_ASSERTION = "The agent greeted the user."


@pytest.fixture
def task_with_nl_assertions(base_task):
    """Mock task graded only on NL assertions."""
    task = base_task.model_copy(deep=True)
    task.evaluation_criteria.nl_assertions = [NL_ASSERTION]
    task.evaluation_criteria.reward_basis = [RewardType.NL_ASSERTION]
    return task


@pytest.fixture
def captured_generate(monkeypatch):
    """Capture the kwargs the NL assertions judge passes to `generate`."""
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": NL_ASSERTION,
                            "reasoning": "The agent said hello.",
                            "metExpectation": True,
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(evaluator_nl_assertions, "generate", fake_generate)
    return captured


def test_evaluate_nl_assertions_defaults_to_configured_judge(captured_generate):
    NLAssertionsEvaluator.evaluate_nl_assertions([], [NL_ASSERTION])

    assert captured_generate["model"] == DEFAULT_LLM_NL_ASSERTIONS
    assert (
        captured_generate["temperature"]
        == DEFAULT_LLM_NL_ASSERTIONS_ARGS["temperature"]
    )


def test_evaluate_nl_assertions_uses_custom_judge(captured_generate):
    NLAssertionsEvaluator.evaluate_nl_assertions(
        [],
        [NL_ASSERTION],
        model="gpt-4o",
        model_args={"temperature": 0.7},
    )

    assert captured_generate["model"] == "gpt-4o"
    assert captured_generate["temperature"] == 0.7


def test_calculate_reward_uses_custom_judge(captured_generate, task_with_nl_assertions):
    reward_info = NLAssertionsEvaluator.calculate_reward(
        task=task_with_nl_assertions,
        full_trajectory=[],
        model="gpt-4o",
        model_args={"temperature": 0.7},
    )

    assert reward_info.reward == 1.0
    assert captured_generate["model"] == "gpt-4o"
    assert captured_generate["temperature"] == 0.7


def test_evaluate_simulation_threads_judge_llm(
    captured_generate, task_with_nl_assertions, domain_name
):
    simulation = SimpleNamespace(
        termination_reason=TerminationReason.AGENT_STOP,
        messages=[],
        ticks=None,
    )

    evaluate_simulation(
        simulation=simulation,
        task=task_with_nl_assertions,
        evaluation_type=EvaluationType.NL_ASSERTIONS,
        solo_mode=False,
        domain=domain_name,
        judge_llm="gpt-4o",
        judge_llm_args={"temperature": 0.7},
    )

    assert captured_generate["model"] == "gpt-4o"
    assert captured_generate["temperature"] == 0.7


def test_run_simulation_threads_judge_llm(monkeypatch, base_task):
    captured = {}
    reward_info = SimpleNamespace(reward=1.0)

    def fake_evaluate_simulation(**kwargs):
        captured["judge_llm"] = kwargs["judge_llm"]
        captured["judge_llm_args"] = kwargs["judge_llm_args"]
        return reward_info

    monkeypatch.setattr(
        runner_simulation, "evaluate_simulation", fake_evaluate_simulation
    )

    environment = SimpleNamespace(
        get_policy=lambda: "policy",
        get_domain_name=lambda: "mock",
    )
    orchestrator = SimpleNamespace(
        run=lambda: SimpleNamespace(policy=None, reward_info=None),
        environment=environment,
        task=base_task,
    )

    result = runner_simulation.run_simulation(
        orchestrator,
        judge_llm="gpt-4o",
        judge_llm_args={"temperature": 0.7},
    )

    assert result.reward_info is reward_info
    assert captured == {
        "judge_llm": "gpt-4o",
        "judge_llm_args": {"temperature": 0.7},
    }


def test_run_single_task_uses_config_judge_llm(monkeypatch, base_task):
    captured = {}
    simulation = SimpleNamespace(reward_info=SimpleNamespace(reward=1.0))

    def fake_build_orchestrator(*args, **kwargs):
        return SimpleNamespace(environment=SimpleNamespace(get_policy=lambda: "policy"))

    def fake_run_simulation(orchestrator, **kwargs):
        captured["judge_llm"] = kwargs["judge_llm"]
        captured["judge_llm_args"] = kwargs["judge_llm_args"]
        return simulation

    monkeypatch.setattr(batch, "build_orchestrator", fake_build_orchestrator)
    monkeypatch.setattr(batch, "run_simulation", fake_run_simulation)

    config = TextRunConfig(
        domain="mock",
        judge_llm="gpt-4o",
        judge_llm_args={"temperature": 0.7},
    )

    assert batch.run_single_task(config, base_task) is simulation
    assert captured == {
        "judge_llm": "gpt-4o",
        "judge_llm_args": {"temperature": 0.7},
    }


def test_run_config_defaults_to_configured_judge():
    config = TextRunConfig(domain="mock")

    assert config.judge_llm == DEFAULT_LLM_NL_ASSERTIONS
    assert config.judge_llm_args == DEFAULT_LLM_NL_ASSERTIONS_ARGS


def test_cli_parses_judge_llm_flags():
    parser = argparse.ArgumentParser()
    add_run_args(parser)

    defaults = parser.parse_args(["--domain", "mock"])
    assert defaults.judge_llm == DEFAULT_LLM_NL_ASSERTIONS
    assert defaults.judge_llm_args == DEFAULT_LLM_NL_ASSERTIONS_ARGS

    overridden = parser.parse_args(
        [
            "--domain",
            "mock",
            "--judge-llm",
            "gpt-4o",
            "--judge-llm-args",
            '{"temperature": 0.7}',
        ]
    )
    assert overridden.judge_llm == "gpt-4o"
    assert overridden.judge_llm_args == {"temperature": 0.7}

import json
from unittest.mock import patch

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import (
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import EvaluationCriteria, RewardType, Task, UserScenario
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator

ASSERTION = "The agent reports the order status"
JUDGE_COST = 0.0123
JUDGE_USAGE = {"prompt_tokens": 321, "completion_tokens": 45}


def _task() -> Task:
    return Task(
        id="cost-test",
        user_scenario=UserScenario(instructions="Check my order"),
        evaluation_criteria=EvaluationCriteria(
            nl_assertions=[ASSERTION],
            reward_basis=[RewardType.NL_ASSERTION],
        ),
    )


def _judge_response() -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=json.dumps(
            {
                "results": [
                    {
                        "expectedOutcome": ASSERTION,
                        "metExpectation": True,
                        "reasoning": "The status was reported.",
                    }
                ]
            }
        ),
        cost=JUDGE_COST,
        usage=JUDGE_USAGE,
    )


def test_nl_evaluator_records_judge_cost_and_usage() -> None:
    with patch(
        "tau2.evaluator.evaluator_nl_assertions.generate",
        return_value=_judge_response(),
    ):
        reward_info = NLAssertionsEvaluator.calculate_reward(
            _task(), [UserMessage(role="user", content="Check order #W1")]
        )

    assert reward_info.reward == 1.0
    assert reward_info.evaluator_cost == JUDGE_COST
    assert reward_info.evaluator_usage == JUDGE_USAGE


def test_simulation_serializes_evaluator_cost_separately() -> None:
    simulation = SimulationRun(
        id="sim-cost-test",
        task_id="cost-test",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
        duration=1.0,
        termination_reason=TerminationReason.USER_STOP,
        agent_cost=0.10,
        user_cost=0.05,
        evaluator_cost=JUDGE_COST,
        evaluator_usage=JUDGE_USAGE,
        reward_info=RewardInfo(
            reward=1.0,
            evaluator_cost=JUDGE_COST,
            evaluator_usage=JUDGE_USAGE,
        ),
        messages=[],
    )

    payload = simulation.model_dump(mode="json")
    assert payload["agent_cost"] == 0.10
    assert payload["user_cost"] == 0.05
    assert payload["evaluator_cost"] == JUDGE_COST
    assert payload["evaluator_usage"] == JUDGE_USAGE
    assert payload["reward_info"]["evaluator_cost"] == JUDGE_COST

"""Recorded intake strategies retain their tool contract during re-evaluation."""

import pytest

from tau2.data_model.elicitation import ElicitationConfig, ElicitationStrategy
from tau2.data_model.message import AssistantMessage, Tick, ToolCall
from tau2.data_model.simulation import Info, Results, SimulationRun, TerminationReason
from tau2.data_model.tasks import Action
from tau2.domains.intake.environment import get_environment, get_tasks
from tau2.orchestrator.modes import CommunicationMode
from tau2.scripts.evaluate_trajectories import (
    _build_eval_env_kwargs,
    compute_simulation_rewards,
)


def _recorded_results(strategy, channel, submitted):
    task = get_tasks(strategy=strategy)[0]
    environment = get_environment(strategy=strategy, channel=channel, task=task)
    environment.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=[],
    )
    messages, ticks = [], []
    if submitted:
        action = task.evaluation_criteria.actions[0]
        call = ToolCall(
            id="saved-submission",
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        response = environment.get_response(call)
        assert not response.error
        messages = [AssistantMessage(role="assistant", tool_calls=[call]), response]
        ticks = [
            Tick(
                tick_id=0,
                timestamp="2026-09-17T00:00:01",
                agent_tool_calls=[call],
                agent_tool_results=[response],
            )
        ]
    is_voice = channel == "voice"
    info = Info(
        git_commit="recorded-commit",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        # Default runs still record resolved settings; never infer them from
        # today's default when re-evaluating an existing result.
        elicitation=ElicitationConfig(strategy=strategy),
        agent_info={"implementation": "llm_agent"},
        user_info={
            "implementation": (
                "voice_streaming_user_simulator" if is_voice else "user_simulator"
            )
        },
        environment_info={"domain_name": "intake", "policy": "Recorded policy"},
    )
    simulation = SimulationRun(
        id="saved-call",
        task_id=task.id,
        start_time="2026-09-17T00:00:00",
        end_time="2026-09-17T00:00:02",
        duration=2,
        termination_reason=TerminationReason.USER_STOP,
        messages=None if is_voice else messages,
        ticks=ticks if is_voice else None,
        mode=(
            CommunicationMode.FULL_DUPLEX.value
            if is_voice
            else CommunicationMode.HALF_DUPLEX.value
        ),
    )
    return Results(info=info, tasks=[task], simulations=[simulation])


@pytest.mark.parametrize(
    "strategy,channel",
    [
        (ElicitationStrategy.AGENT_DIRECTED, "voice"),
        (ElicitationStrategy.SCAFFOLDED, "voice"),
        (ElicitationStrategy.SCAFFOLDED, "text"),
    ],
)
@pytest.mark.parametrize("submitted", [False, True])
@pytest.mark.parametrize("fresh_tasks", [False, True])
def test_regrading_preserves_saved_strategy_and_submission_contract(
    strategy, channel, submitted, fresh_tasks, monkeypatch
):
    results = _recorded_results(strategy, channel, submitted)
    if fresh_tasks:
        results.tasks[0].evaluation_criteria.actions = []
    original = results.model_dump_json()

    def refuse_resampling(*args, **kwargs):
        pytest.fail("Re-evaluation must not sample new caller complications")

    monkeypatch.setattr(
        "tau2.domains.intake.complications.sample_complication", refuse_resampling
    )
    rescored = compute_simulation_rewards(results, fresh_tasks=fresh_tasks)

    reward = rescored.simulations[0].reward_info
    assert reward.reward == float(submitted)
    assert reward.db_check.db_match is submitted
    arguments = rescored.tasks[0].evaluation_criteria.actions[0].arguments
    assert ("confirmed_with_user" in arguments) is (
        strategy is ElicitationStrategy.SCAFFOLDED
    )
    assert results.model_dump_json() == original


@pytest.mark.parametrize("fresh_tasks", [False, True])
def test_missing_recorded_strategy_fails_instead_of_using_current_default(fresh_tasks):
    results = _recorded_results(ElicitationStrategy.SCAFFOLDED, "text", False)
    results.info.elicitation = None

    with pytest.raises(ValueError, match="requires recorded elicitation settings"):
        compute_simulation_rewards(results, fresh_tasks=fresh_tasks)


@pytest.mark.parametrize("domain", ["mock", "airline", "retail", "telecom"])
def test_other_domains_keep_existing_environment_kwargs(domain):
    results = _recorded_results(ElicitationStrategy.SCAFFOLDED, "text", False)
    results.info.environment_info.domain_name = domain
    results.info.elicitation = None

    assert (
        _build_eval_env_kwargs(results, results.tasks[0], results.simulations[0])
        is None
    )


def test_banking_preserves_read_log_allowlist():
    results = _recorded_results(ElicitationStrategy.SCAFFOLDED, "text", False)
    results.info.environment_info.domain_name = "banking_knowledge"
    results.info.elicitation = None
    results.tasks[0].evaluation_criteria.actions = [
        Action(
            action_id="read-account",
            requestor="assistant",
            name="call_discoverable_agent_tool",
            arguments={"agent_tool_name": "account_transactions", "arguments": "{}"},
        )
    ]

    assert _build_eval_env_kwargs(
        results, results.tasks[0], results.simulations[0]
    ) == {"read_log_allowlist": {"account_transactions"}}

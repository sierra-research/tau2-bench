"""Post-hoc reviews use the exact elicitation context saved with each call."""

from io import StringIO

import pytest
from rich.console import Console

from tau2.data_model.elicitation import (
    ComplicationKind,
    ComplicationParams,
    SampledComplication,
)
from tau2.data_model.simulation import (
    AuthenticationClassification,
    Results,
    Review,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
    UserOnlyReview,
)
from tau2.domains.intake.environment import get_tasks
from tau2.evaluator.reviewer import ReviewMode
from tau2.runner.helpers import get_info
from tau2.scripts import review_conversation


@pytest.fixture
def recorded_results():
    task = get_tasks(strategy="scaffolded")[0]
    task.user_scenario.instructions.task_instructions = "Recorded caller instructions."
    info = get_info(
        TextRunConfig(domain="intake", elicitation={"strategy": "scaffolded"})
    )
    # Historical names and versions remain valid recorded context without
    # consulting today's registry, prompt templates, or stimulus catalog.
    info.environment_info.domain_name = "intake_free"
    info.environment_info.policy = "Recorded run-level policy."
    info.elicitation_prompt_version = "archived-prompt"
    simulation = SimulationRun(
        id="recorded-call",
        task_id=task.id,
        trial=1,
        seed=42,
        start_time="2026-09-17T00:00:00",
        end_time="2026-09-17T00:00:01",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        policy="Recorded task-specific policy with the pre-call work order.",
        complication=SampledComplication(
            kind=ComplicationKind.SELF_CORRECTION,
            line="Recorded exception: first say the archived decoy, then correct it.",
            params=ComplicationParams(decoy_value="archived decoy"),
            catalog_version="archived-catalog",
        ),
    )
    return Results(info=info, tasks=[task], simulations=[simulation])


@pytest.fixture
def judge_calls(monkeypatch):
    calls = []

    def capture_review(**kwargs):
        calls.append(kwargs)
        if kwargs["mode"] == ReviewMode.FULL:
            return Review(has_errors=False), AuthenticationClassification(
                status="not_needed"
            )
        return UserOnlyReview(has_errors=False), None

    monkeypatch.setattr(review_conversation, "review_simulation", capture_review)

    def refuse_resampling(*args, **kwargs):
        pytest.fail("Post-hoc review must not recreate current run conditions")

    monkeypatch.setattr(
        "tau2.domains.intake.complications.sample_complication", refuse_resampling
    )
    monkeypatch.setattr("tau2.runner.elicitation.user_prompt_task", refuse_resampling)
    return calls


def _review(results, mode, batch=False):
    if batch:
        run = (
            review_conversation.run_full_review
            if mode == ReviewMode.FULL
            else review_conversation.run_user_review
        )
        output = run(
            results,
            results_path="recorded/results.json",
            max_concurrency=1,
            console=Console(file=StringIO()),
        )
        assert output.total_simulations == 1
    else:
        run = (
            review_conversation.review_simulation_full
            if mode == ReviewMode.FULL
            else review_conversation.review_simulation_user
        )
        run(results.simulations[0], results.tasks[0], results)


@pytest.mark.parametrize("mode", [ReviewMode.FULL, ReviewMode.USER])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_review_uses_recorded_context_without_mutating_task(
    recorded_results, judge_calls, mode, batch, structured
):
    task = recorded_results.tasks[0]
    if not structured:
        task.user_scenario.instructions = "Recorded caller instructions."
    original = task.model_dump_json()

    _review(recorded_results, mode, batch)

    assert len(judge_calls) == 1
    reviewed_task = judge_calls[0]["task"]
    assert reviewed_task is not task
    instructions = reviewed_task.user_scenario.instructions
    actual = instructions.task_instructions if structured else instructions
    assert actual == (
        "Recorded caller instructions.\n\n"
        + recorded_results.simulations[0].complication.line
    )
    if mode == ReviewMode.FULL:
        assert judge_calls[0]["policy"] == recorded_results.simulations[0].policy
    assert task.model_dump_json() == original


@pytest.mark.parametrize("mode", [ReviewMode.FULL, ReviewMode.USER])
@pytest.mark.parametrize("clean", [False, True])
def test_clean_and_prompt_silent_calls_do_not_inject_instructions(
    recorded_results, judge_calls, mode, clean
):
    simulation = recorded_results.simulations[0]
    if clean:
        simulation.complication = None
    else:
        simulation.complication = SampledComplication(
            kind=ComplicationKind.MISPRONOUNCED_TERM,
            line="Provenance only: the synthesized term was replaced.",
            params=ComplicationParams(term="token"),
            catalog_version="archived-catalog",
            injected=False,
            mispronunciation="bad token",
        )

    _review(recorded_results, mode)

    assert judge_calls[0]["task"] is recorded_results.tasks[0]


@pytest.mark.parametrize("mode", [ReviewMode.FULL, ReviewMode.USER])
def test_other_domains_keep_existing_review_context(
    recorded_results, judge_calls, mode
):
    recorded_results.info.elicitation = None
    recorded_results.info.environment_info.domain_name = "retail"
    # Without elicitation the recorded language context pins the exact task.
    recorded_results.tasks[0] = get_tasks(strategy="scaffolded")[0]

    _review(recorded_results, mode)

    assert judge_calls[0]["task"] is recorded_results.tasks[0]
    if mode == ReviewMode.FULL:
        assert judge_calls[0]["policy"] == recorded_results.info.environment_info.policy


def test_elicitation_can_review_without_a_run_level_policy(
    recorded_results, judge_calls
):
    recorded_results.info.environment_info.policy = None

    _review(recorded_results, ReviewMode.FULL)

    assert judge_calls[0]["policy"] == recorded_results.simulations[0].policy


def test_elicitation_falls_back_to_recorded_run_policy(recorded_results, judge_calls):
    recorded_results.simulations[0].policy = None

    _review(recorded_results, ReviewMode.FULL)

    assert judge_calls[0]["policy"] == recorded_results.info.environment_info.policy

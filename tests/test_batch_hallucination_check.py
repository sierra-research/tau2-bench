"""Tests for hallucination-check error handling in the batch runner.

The hallucination check is a post-hoc reviewer LLM call that runs after a
simulation has already completed. A reviewer failure (e.g. provider
overloaded after all litellm retries) must not crash the whole batch: the
completed simulation is kept and its hallucination check is recorded as
errored instead.
"""

import json
from types import SimpleNamespace

from tau2.data_model.message import Tick
from tau2.data_model.simulation import (
    AudioNativeConfig,
    HallucinationCheck,
    HallucinationCheckError,
    RunConfig,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.reviewer import check_hallucination as run_hallucination_check
from tau2.registry import registry
from tau2.runner.batch import run_tasks

NAME_ROLES_MARKER = "When asked for your full name"
NATIVE_SPELLOUT_MARKER = "## SPELLING YOUR NAME (NATIVE SCRIPT)"


def _make_config(hallucination_retries: int = 2) -> TextRunConfig:
    return TextRunConfig(
        domain="mock",
        agent="llm_agent",
        user="user_simulator",
        num_trials=1,
        max_concurrency=1,
        hallucination_retries=hallucination_retries,
        # In-process execution: the monkeypatched check_hallucination must be
        # visible (a worker fleet would run the real one in subprocesses).
        workers=0,
    )


def _make_full_duplex_sim(task: Task) -> SimulationRun:
    return SimulationRun(
        id="sim-test-1",
        task_id=task.id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        ticks=[Tick(tick_id=0, timestamp="2026-01-01T00:00:00")],
        trial=0,
    )


def _patch_run_with_retry(monkeypatch, task: Task) -> None:
    def fake_run_with_retry(run_fn, *args, **kwargs) -> SimulationRun:
        return _make_full_duplex_sim(task)

    monkeypatch.setattr("tau2.runner.batch.run_with_retry", fake_run_with_retry)


def test_reviewer_exception_does_not_fail_batch(monkeypatch, base_task: Task):
    """A reviewer LLM failure in check_hallucination must not crash the batch."""
    _patch_run_with_retry(monkeypatch, base_task)

    def failing_check(simulation, task, **kwargs):
        raise RuntimeError("AnthropicError overloaded_error")

    monkeypatch.setattr("tau2.runner.batch.check_hallucination", failing_check)

    results = run_tasks(
        config=_make_config(),
        tasks=[base_task],
        console_display=False,
    )

    assert len(results.simulations) == 1
    sim = results.simulations[0]
    check = sim.hallucination_check
    assert check is not None
    assert check.check_error is not None
    assert "overloaded_error" in check.check_error
    assert check.hallucination_found is False
    assert sim.hallucination_retries_used == 0


def test_clean_hallucination_check_still_recorded(monkeypatch, base_task: Task):
    """Sanity check: a successful check is recorded unchanged (check_error unset)."""
    _patch_run_with_retry(monkeypatch, base_task)

    clean_check = HallucinationCheck(
        reasoning="No fabricated information.",
        hallucination_found=False,
        summary="Clean.",
    )
    monkeypatch.setattr(
        "tau2.runner.batch.check_hallucination",
        lambda simulation, task, **kwargs: clean_check,
    )

    results = run_tasks(
        config=_make_config(),
        tasks=[base_task],
        console_display=False,
    )

    assert len(results.simulations) == 1
    check = results.simulations[0].hallucination_check
    assert check is not None
    assert check.check_error is None
    assert check.summary == "Clean."


def _patch_attempt_runner(monkeypatch, task: Task) -> list[dict]:
    """Replace simulation execution and record every initial/retry call."""
    calls: list[dict] = []

    def fake_run_single_task(config, run_task, **kwargs):
        calls.append(
            {
                "task": run_task,
                "seed": kwargs["seed"],
                "feedback": kwargs["hallucination_feedback"],
            }
        )
        return _make_full_duplex_sim(task).model_copy(
            update={"id": f"sim-attempt-{len(calls)}"}
        )

    def fake_run_with_retry(run_fn, *args, **kwargs):
        return run_fn()

    monkeypatch.setattr("tau2.runner.batch.run_single_task", fake_run_single_task)
    monkeypatch.setattr("tau2.runner.batch.run_with_retry", fake_run_with_retry)
    return calls


def _flagged_check(attempt: int) -> HallucinationCheck:
    return HallucinationCheck(
        reasoning=f"Attempt {attempt} fabricated a value.",
        hallucination_found=True,
        errors=[
            HallucinationCheckError(
                reasoning=f"Fabricated value on attempt {attempt}.",
                user_message=f"fabricated-{attempt}",
                correct_behavior="Say that the value is unknown.",
            )
        ],
        summary=f"Flagged attempt {attempt}.",
    )


def test_hallucination_gate_checks_final_allowed_rerun(monkeypatch, base_task: Task):
    calls = _patch_attempt_runner(monkeypatch, base_task)
    checked_ids: list[str] = []

    def check_until_final_is_clean(simulation, task, **kwargs):
        checked_ids.append(simulation.id)
        attempt = len(checked_ids)
        if attempt < 4:
            return _flagged_check(attempt)
        return HallucinationCheck(
            reasoning="The final attempt is grounded.",
            hallucination_found=False,
            summary="Final attempt clean.",
        )

    monkeypatch.setattr(
        "tau2.runner.batch.check_hallucination", check_until_final_is_clean
    )
    results = run_tasks(
        config=_make_config(hallucination_retries=3),
        tasks=[base_task],
        console_display=False,
    )

    assert len(calls) == 4
    assert checked_ids == [
        "sim-attempt-1",
        "sim-attempt-2",
        "sim-attempt-3",
        "sim-attempt-4",
    ]
    final = results.simulations[0]
    assert final.id == "sim-attempt-4"
    assert final.hallucination_retries_used == 3
    assert final.hallucination_check is not None
    assert final.hallucination_check.hallucination_found is False
    assert final.hallucination_check.summary == "Final attempt clean."


def test_final_flagged_attempt_is_retained_without_fourth_rerun(
    monkeypatch, base_task: Task
):
    calls = _patch_attempt_runner(monkeypatch, base_task)
    checked_ids: list[str] = []

    def always_flag(simulation, task, **kwargs):
        checked_ids.append(simulation.id)
        return _flagged_check(len(checked_ids))

    monkeypatch.setattr("tau2.runner.batch.check_hallucination", always_flag)
    results = run_tasks(
        config=_make_config(hallucination_retries=3),
        tasks=[base_task],
        console_display=False,
    )

    assert len(calls) == 4
    assert checked_ids == [
        "sim-attempt-1",
        "sim-attempt-2",
        "sim-attempt-3",
        "sim-attempt-4",
    ]
    final = results.simulations[0]
    assert final.id == "sim-attempt-4"
    assert final.hallucination_retries_used == 3
    assert final.hallucination_check is not None
    assert final.hallucination_check.hallucination_found is True
    assert final.hallucination_check.summary == "Flagged attempt 4."
    assert final.hallucination_check.errors[0].user_message == "fabricated-4"


def _checked_task(
    monkeypatch,
    *,
    config: RunConfig,
    task: Task,
) -> tuple[Task, str]:
    """Run one offline unit and return the gate's task and rendered prompt."""
    _patch_run_with_retry(monkeypatch, task)
    checked: list[Task] = []
    rendered_prompts: list[str] = []

    def fake_generate(*, messages, **kwargs):
        rendered_prompts.append(messages[-1].content)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "reasoning": "No fabricated information.",
                    "hallucinations": [],
                    "summary": "Clean.",
                }
            ),
            cost=0.0,
        )

    def capture_check(simulation, check_task, **kwargs):
        checked.append(check_task)
        return run_hallucination_check(simulation, check_task)

    monkeypatch.setattr("tau2.evaluator.hallucination_reviewer.generate", fake_generate)
    monkeypatch.setattr("tau2.runner.batch.check_hallucination", capture_check)
    results = run_tasks(config=config, tasks=[task], console_display=False)
    assert len(checked) == 1
    assert len(rendered_prompts) == 1
    assert len(results.tasks) == 1
    assert results.tasks[0].model_dump_json() == task.model_dump_json()
    return checked[0], rendered_prompts[0]


def _localized_config(
    task_set_name: str,
    language: str,
    *,
    retail_name_roles_prompt_version: str | None = None,
) -> VoiceRunConfig:
    return VoiceRunConfig(
        domain="retail",
        task_set_name=task_set_name,
        user_persona_id=language,
        audio_native_config=AudioNativeConfig(),
        num_trials=1,
        max_concurrency=1,
        hallucination_retries=1,
        complication_rate=0.0,
        workers=0,
        retail_name_roles_prompt_version=retail_name_roles_prompt_version,
    )


def test_gate_receives_exact_native_mandarin_user_prompt_task(monkeypatch):
    task = next(
        task
        for task in registry.get_tasks_loader("retail_zh_identity_native")()
        if task.id == "5_zh_identity_native"
    )
    original = task.model_dump_json()

    checked_task, judge_prompt = _checked_task(
        monkeypatch,
        config=_localized_config(
            "retail_zh_identity_native",
            "zh",
            retail_name_roles_prompt_version="v1",
        ),
        task=task,
    )
    prompt = str(checked_task.user_scenario)

    assert "Your first name is 霞, and your last name is 吴" in prompt
    assert "give your last name first and your first name second: 吴霞" in prompt
    assert prompt.count(NAME_ROLES_MARKER) == 1
    assert prompt.count(NATIVE_SPELLOUT_MARKER) == 1
    assert "吴 — 口天吴" in prompt
    assert "霞 — 彩霞的霞" in prompt
    assert "吴霞" in judge_prompt
    assert judge_prompt.count(NAME_ROLES_MARKER) == 1
    assert judge_prompt.count(NATIVE_SPELLOUT_MARKER) == 1
    assert "吴 — 口天吴" in judge_prompt
    assert "霞 — 彩霞的霞" in judge_prompt
    assert task.model_dump_json() == original
    assert NAME_ROLES_MARKER not in str(task.user_scenario)
    assert NATIVE_SPELLOUT_MARKER not in str(task.user_scenario)


def test_gate_receives_exact_korean_identity_user_prompt_task(monkeypatch):
    task = next(
        task
        for task in registry.get_tasks_loader("retail_ko_identity")()
        if task.id.startswith("56_")
    )
    original = task.model_dump_json()

    checked_task, judge_prompt = _checked_task(
        monkeypatch,
        config=_localized_config(
            "retail_ko_identity",
            "ko",
            retail_name_roles_prompt_version="v1",
        ),
        task=task,
    )
    prompt = str(checked_task.user_scenario)

    assert (
        "Your first name is Jaehyun, and your last name is Shin. "
        "When asked for your full name, give your last name first and your "
        "first name second: Shin Jaehyun."
    ) in prompt
    assert prompt.count(NAME_ROLES_MARKER) == 1
    assert judge_prompt.count(NAME_ROLES_MARKER) == 1
    assert NATIVE_SPELLOUT_MARKER not in prompt
    assert task.model_dump_json() == original
    assert NAME_ROLES_MARKER not in str(task.user_scenario)


def test_gate_keeps_english_user_prompt_task_unchanged(monkeypatch):
    task = registry.get_tasks_loader("retail")()[0]
    original = task.model_dump_json()
    config = VoiceRunConfig(
        domain="retail",
        task_set_name="retail",
        audio_native_config=AudioNativeConfig(),
        num_trials=1,
        max_concurrency=1,
        hallucination_retries=1,
        complication_rate=0.0,
        workers=0,
    )

    checked_task, judge_prompt = _checked_task(monkeypatch, config=config, task=task)

    assert checked_task.model_dump_json() == original
    assert str(task.user_scenario) in judge_prompt
    assert task.model_dump_json() == original

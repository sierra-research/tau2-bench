"""Offline coverage for malformed hallucination-reviewer responses."""

from pathlib import Path

import pytest

from tau2.data_model.message import AssistantMessage, Tick, UserMessage
from tau2.data_model.simulation import (
    HallucinationCheck,
    Info,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
)
from tau2.evaluator import hallucination_reviewer
from tau2.evaluator.evaluator import EvaluationType
from tau2.evaluator.hallucination_reviewer import HallucinationReviewResponse
from tau2.runner import batch
from tau2.runner.hallucination_checkpoint import (
    HallucinationReviewCheckpoint,
    load_review_checkpoint,
    review_checkpoint_path,
    review_config_hash,
    save_review_checkpoint,
)
from tau2.utils import llm_utils
from tau2.utils.llm_utils import LLMResponseValidationError


def test_structured_generate_repairs_invalid_json_without_changing_input(monkeypatch):
    messages = [UserMessage(role="user", content="Review the conversation")]
    replies = iter(
        [
            '{"reasoning":"unescaped "quote"","hallucinations":[],"summary":""}',
            '{"reasoning":"checked","hallucinations":[],"summary":"clear"}',
        ]
    )
    calls = []

    def fake_generate_once(model, prompt, **kwargs):
        calls.append((model, list(prompt), kwargs["call_name"]))
        return AssistantMessage(role="assistant", content=next(replies), cost=0.25)

    monkeypatch.setattr(llm_utils, "_generate_once", fake_generate_once)
    result = llm_utils.generate(
        "review-model",
        messages,
        call_name="review_v2.0.0",
        response_model=HallucinationReviewResponse,
        response_validation_retries=2,
    )
    assert HallucinationReviewResponse.model_validate_json(result.content)
    assert result.cost == 0.5
    assert len(messages) == 1
    assert [call[2] for call in calls] == ["review_v2.0.0"] * 2
    assert calls[1][1][-1].content.startswith("Your previous reply was invalid")
    assert calls[1][1][0] is messages[0]
    assert calls[1][1][-2].content.startswith('{"reasoning":"unescaped')


def test_structured_generate_preserves_all_bad_replies(monkeypatch):
    replies = iter(["{bad", '{"reasoning":"ok","hallucinations":"none"}'])
    monkeypatch.setattr(
        llm_utils,
        "_generate_once",
        lambda *_args, **_kwargs: AssistantMessage(
            role="assistant", content=next(replies)
        ),
    )
    with pytest.raises(LLMResponseValidationError) as exc:
        llm_utils.generate(
            "review-model",
            [],
            response_model=HallucinationReviewResponse,
            response_validation_retries=1,
        )
    assert exc.value.raw_responses == [
        "{bad",
        '{"reasoning":"ok","hallucinations":"none"}',
    ]


def test_reviewer_records_model_and_prompt_version(monkeypatch, base_task):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(
            role="assistant",
            content='{"reasoning":"checked","hallucinations":[],"summary":"clear"}',
            cost=0.4,
        )

    monkeypatch.setattr(hallucination_reviewer, "generate", fake_generate)
    review = hallucination_reviewer.FullDuplexHallucinationReviewer.review(
        base_task,
        [Tick(tick_id=0, timestamp="2026-10-15T00:00:00")],
        review_model="review-model",
    )
    assert review.review_status == "completed"
    assert review.hallucination_found is False
    assert review.review_model == "review-model"
    assert review.prompt_version == "2.0.0"
    assert review.response_validation_retries == 3
    assert calls[0]["call_name"] == "llm_judge_hallucination_check_v2.0.0"
    assert calls[0]["response_model"] is HallucinationReviewResponse


def _simulation(task_id: str, sim_id: str) -> SimulationRun:
    return SimulationRun(
        id=sim_id,
        task_id=task_id,
        start_time="2026-10-15T00:00:00",
        end_time="2026-10-15T00:00:01",
        duration=1.0,
        termination_reason=TerminationReason.AGENT_STOP,
        reward_info=RewardInfo(reward=0.5),
        ticks=[Tick(tick_id=0, timestamp="2026-10-15T00:00:00")],
    )


def _context(tmp_path: Path, task_id: str):
    config = TextRunConfig(domain="mock", hallucination_retries=1)
    return batch._BatchContext(
        config=config,
        evaluation_type=EvaluationType.ALL,
        save_dir=tmp_path,
        user_voice_settings=None,
        user_persona_config=None,
        info=Info.model_construct(language_tasks={task_id: None}),
        console_display=False,
    )


def test_review_failure_passes_scored_simulation_with_explicit_flag(
    monkeypatch, tmp_path, base_task
):
    ctx = _context(tmp_path, base_task.id)
    sim_calls = []
    review_calls = []

    def fake_simulate(*_args, **_kwargs):
        sim_calls.append(1)
        return _simulation(base_task.id, "scored-call")

    def fake_review(*_args, **_kwargs):
        review_calls.append(1)
        raise LLMResponseValidationError(
            "HallucinationReviewResponse", ["invalid raw reply"], ValueError()
        )

    monkeypatch.setattr(batch, "run_single_task", fake_simulate)
    monkeypatch.setattr(batch, "check_hallucination", fake_review)

    result = batch.run_unit(ctx, base_task, 0, 42)
    assert result.id == "scored-call"
    assert result.termination_reason == TerminationReason.AGENT_STOP
    assert result.reward_info.reward == 0.5
    assert result.hallucination_check.review_status == "failed"
    assert result.hallucination_check.hallucination_found is None
    assert result.hallucination_check.review_failure.raw_responses == [
        "invalid raw reply"
    ]
    loaded = SimulationRun.model_validate_json(result.model_dump_json())
    assert loaded.reward_info.reward == 0.5
    assert loaded.hallucination_check.review_status == "failed"
    assert loaded.hallucination_check.hallucination_found is None
    assert len(sim_calls) == 1
    assert len(review_calls) == 1

    path = review_checkpoint_path(tmp_path, base_task.id, 0, 42)
    checkpoint = load_review_checkpoint(
        path,
        task_id=base_task.id,
        trial=0,
        seed=42,
        config_hash=review_config_hash(ctx.config),
    )
    assert checkpoint.simulation.id == "scored-call"
    assert checkpoint.simulation.hallucination_check.review_status == "failed"

    # A lost controller response can be delivered again from the completed
    # checkpoint without another scored call or reviewer call.
    assert batch.run_unit(ctx, base_task, 0, 42).id == "scored-call"
    assert len(sim_calls) == 1
    assert len(review_calls) == 1


def test_reviewer_api_error_is_flagged_without_rerunning_voice(
    monkeypatch, tmp_path, base_task
):
    ctx = _context(tmp_path, base_task.id)
    sim_calls = []

    def fake_simulate(*_args, **_kwargs):
        sim_calls.append(1)
        return _simulation(base_task.id, "api-error-call")

    def fake_review(*_args, **_kwargs):
        raise ConnectionError("reviewer unavailable")

    monkeypatch.setattr(batch, "run_single_task", fake_simulate)
    monkeypatch.setattr(batch, "check_hallucination", fake_review)
    result = batch.run_unit(ctx, base_task, 0, 42)
    assert result.id == "api-error-call"
    assert result.hallucination_check.review_status == "failed"
    assert result.hallucination_check.hallucination_found is None
    assert result.hallucination_check.review_failure.error_type == "ConnectionError"
    assert result.hallucination_check.review_failure.raw_responses == []
    assert sim_calls == [1]


def test_result_checkpoint_failure_reuses_reviewed_simulation(
    monkeypatch, tmp_path, base_task
):
    ctx = _context(tmp_path, base_task.id)
    sim_calls = []
    review_calls = []
    save_calls = []

    def fake_simulate(*_args, **_kwargs):
        sim_calls.append(1)
        return _simulation(base_task.id, "save-error-call")

    def fake_review(*_args, **_kwargs):
        review_calls.append(1)
        return HallucinationCheck(
            reasoning="verified", hallucination_found=False, errors=[]
        )

    def fake_save(_result):
        save_calls.append(1)
        if len(save_calls) == 1:
            raise OSError("checkpoint unavailable")

    ctx.save_fn = fake_save
    monkeypatch.setattr(batch, "run_single_task", fake_simulate)
    monkeypatch.setattr(batch, "check_hallucination", fake_review)
    with pytest.raises(OSError, match="checkpoint unavailable"):
        batch.run_unit(ctx, base_task, 0, 42)
    path = review_checkpoint_path(tmp_path, base_task.id, 0, 42)
    assert path.exists()

    result = batch.run_unit(ctx, base_task, 0, 42)
    assert result.id == "save-error-call"
    assert len(sim_calls) == 1
    assert len(review_calls) == 1
    assert len(save_calls) == 2
    assert not path.exists()


def test_delivered_failed_review_leaves_no_checkpoint(monkeypatch, tmp_path, base_task):
    ctx = _context(tmp_path, base_task.id)
    sim_calls = []
    saved = []

    def fake_simulate(*_args, **_kwargs):
        sim_calls.append(1)
        return _simulation(base_task.id, f"call-{len(sim_calls)}")

    def fake_review(*_args, **_kwargs):
        raise ConnectionError("reviewer unavailable")

    ctx.save_fn = saved.append
    monkeypatch.setattr(batch, "run_single_task", fake_simulate)
    monkeypatch.setattr(batch, "check_hallucination", fake_review)
    assert batch.run_unit(ctx, base_task, 0, 42).id == "call-1"
    assert not review_checkpoint_path(tmp_path, base_task.id, 0, 42).exists()

    # A later run of the same unit simulates again rather than replaying.
    assert batch.run_unit(ctx, base_task, 0, 42).id == "call-2"
    assert [r.id for r in saved] == ["call-1", "call-2"]


def test_mismatched_checkpoint_is_discarded(tmp_path, base_task):
    path = review_checkpoint_path(tmp_path, base_task.id, 0, 42)
    save_review_checkpoint(
        path,
        HallucinationReviewCheckpoint(
            task_id=base_task.id,
            trial=0,
            seed=42,
            config_hash="old-config",
            simulation=_simulation(base_task.id, "stale-call"),
        ),
    )
    assert (
        load_review_checkpoint(
            path, task_id=base_task.id, trial=0, seed=42, config_hash="new-config"
        )
        is None
    )
    assert not path.exists()

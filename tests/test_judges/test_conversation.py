# Copyright Sierra
"""Conversation-judge suite: per-dimension/per-turn headline, shadow
composites, caching/resume, and the error channel."""

import pytest
from pydantic import ValidationError

from tau2.data_model.audio_effects import SpeechEffectsResult, UserSpeechInsert
from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    TurnTakingAction,
    UserMessage,
)
from tau2.data_model.simulation import (
    Info,
    QualityJudgeSettings,
    Results,
    RewardInfo,
    SimulationRun,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.data_model.voice import SpeechEnvironment
from tau2.environment.environment import EnvironmentInfo
from tau2.judges.conversation.models import (
    ConversationJudgeConfig,
    EvaConcisenessFailureMode,
    EvaConcisenessTurnResult,
    EvaProgressionDimension,
    EvaProgressionDimensionName,
)
from tau2.judges.conversation.runner import run_conversation_judging
from tau2.judges.conversation.semantic import (
    aggregate_conciseness,
    aggregate_progression,
)
from tau2.judges.quality.harness import evaluate_quality
from tau2.metrics.turn_taking import TurnTakingNAReason


def _tick(
    tick_id: int,
    *,
    user: bool = False,
    agent: bool = False,
    action: str | None = None,
    effect: str | None = None,
    tool: bool = False,
    content: str = "caller",
) -> Tick:
    speech_effects = (
        SpeechEffectsResult(speech_insert=UserSpeechInsert(text="um", type=effect))
        if effect
        else None
    )
    user_chunk = None
    if user or effect:
        user_chunk = UserMessage(
            role="user",
            content=content,
            contains_speech=user,
            turn_taking_action=TurnTakingAction(action=action) if action else None,
            speech_effects=speech_effects,
        )
    return Tick(
        tick_id=tick_id,
        timestamp=str(tick_id),
        tick_duration_seconds=0.5,
        user_chunk=user_chunk,
        agent_chunk=AssistantMessage(
            role="assistant", content="agent", contains_speech=agent
        ),
        agent_tool_calls=(
            [ToolCall(id=f"tool-{tick_id}", name="lookup", arguments={})]
            if tool
            else []
        ),
    )


def _ticks(spec: list[dict]) -> list[Tick]:
    return [_tick(i, **row) for i, row in enumerate(spec)]


_ANSWERED_TURN = (
    [{"user": True, "action": "keep_talking"}] * 2 + [{}] + [{"agent": True}] * 2 + [{}]
)
_UNANSWERED_TURN = [{"user": True, "action": "keep_talking"}] * 2 + [{}] * 3


def _task(task_id: str = "t1") -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="Help the caller."),
        evaluation_criteria=EvaluationCriteria(),
    )


def _sim(sim_id: str, task_id: str, **updates) -> SimulationRun:
    return SimulationRun(
        id=sim_id,
        task_id=task_id,
        start_time="x",
        end_time="y",
        duration=3.0,
        termination_reason="agent_stop",
        ticks=_ticks([{"user": True}] * 2 + [{}] + [{"agent": True}] * 2 + [{}]),
        **updates,
    )


def _results_fixture(sims: list[SimulationRun], tasks: list[Task]) -> Results:
    return Results(
        info=Info(
            git_commit="abc",
            num_trials=1,
            max_steps=10,
            max_errors=1,
            user_info=UserInfo(implementation="user"),
            agent_info={"implementation": "agent"},
            environment_info=EnvironmentInfo(domain_name="mock", policy="policy"),
        ),
        tasks=tasks,
        simulations=sims,
    )


def _install_fake_judges(monkeypatch, calls: dict, *, flag_information_loss=False):
    def fake_conciseness(_sim, *, turn_id, **_kwargs):
        calls["conciseness"] += 1
        return EvaConcisenessTurnResult(
            turn_id=turn_id,
            rating=2,
            failure_modes=[EvaConcisenessFailureMode.VERBOSITY_OR_FILLER],
            evidence="filler-heavy phrasing",
        )

    def fake_progression(_sim, _task, **_kwargs):
        calls["progression"] += 1
        return [
            EvaProgressionDimension(
                name=name,
                flagged=(
                    flag_information_loss
                    and name is EvaProgressionDimensionName.INFORMATION_LOSS
                ),
                rating=(
                    2
                    if flag_information_loss
                    and name is EvaProgressionDimensionName.INFORMATION_LOSS
                    else 3
                ),
                evidence=(
                    "the caller repeated the account id"
                    if flag_information_loss
                    and name is EvaProgressionDimensionName.INFORMATION_LOSS
                    else ""
                ),
            )
            for name in EvaProgressionDimensionName
        ]

    monkeypatch.setattr(
        "tau2.judges.conversation.runner.judge_conciseness_turn", fake_conciseness
    )
    monkeypatch.setattr(
        "tau2.judges.conversation.runner.judge_progression", fake_progression
    )


def test_contradictory_progression_flags_are_rejected():
    with pytest.raises(ValidationError):
        EvaProgressionDimension(
            name=EvaProgressionDimensionName.INFORMATION_LOSS,
            flagged=True,
            rating=3,
            evidence="lost the account id",
        )
    with pytest.raises(ValidationError):
        EvaProgressionDimension(
            name=EvaProgressionDimensionName.INFORMATION_LOSS,
            flagged=False,
            rating=2,
        )


def test_flagged_verdicts_require_evidence():
    # es+pt calibration finding: 27-37% of flagged semantic dimensions carried
    # empty evidence. A flag or deduction without an explanation is
    # unadjudicable and now fails validation (a loud ERROR at the harness).
    with pytest.raises(ValidationError):
        EvaProgressionDimension(
            name=EvaProgressionDimensionName.QUESTION_QUALITY,
            flagged=True,
            rating=2,
            evidence="  ",
        )
    with pytest.raises(ValidationError):
        EvaConcisenessTurnResult(turn_id=0, rating=2)
    # Clean verdicts never require evidence.
    EvaProgressionDimension(
        name=EvaProgressionDimensionName.QUESTION_QUALITY, flagged=False, rating=3
    )
    EvaConcisenessTurnResult(turn_id=0, rating=3)


def test_semantic_transcripts_mark_tick_overlap_barge_ins():
    # Realtime cancellation: every emitted chunk stays "active" (no
    # undelivered gold chunks, no raw_data.was_truncated) — the tick timeline
    # is the ONLY record of the cut. Both semantic transcript renderings must
    # still carry the marker on the cut turn.
    from tau2.data_model.simulation import SimulationRun as _SimulationRun
    from tau2.judges.base import INTERRUPTED_TURN_MARKER
    from tau2.judges.conversation.semantic import (
        _render_transcript,
        extract_agent_turns,
    )

    ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="Qual é o número do seu pedi",
                contains_speech=True,
                utterance_ids=["u1"],
            ),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="do, por fav", contains_speech=True, utterance_ids=["u1"]
            ),
            user_chunk=UserMessage(role="user", content="é o 42", contains_speech=True),
        ),
        Tick(
            tick_id=2,
            timestamp="2",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage.voice(
                content="obrigado", contains_speech=True, utterance_ids=["u2"]
            ),
        ),
    ]
    sim = _SimulationRun(
        id="s-tick-int",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=3.0,
        termination_reason="agent_stop",
        ticks=ticks,
    )
    transcript = _render_transcript(sim)
    assert (
        f"AGENT TURN 0: Qual é o número do seu pedido, por fav "
        f"{INTERRUPTED_TURN_MARKER}" in transcript
    )
    assert f"obrigado {INTERRUPTED_TURN_MARKER}" not in transcript
    turns = dict(extract_agent_turns(sim))
    assert turns[0].endswith(INTERRUPTED_TURN_MARKER)
    assert turns[1] == "obrigado"


def test_semantic_transcripts_mark_interrupted_agent_turns():
    from tau2.data_model.simulation import SimulationRun as _SimulationRun
    from tau2.judges.base import INTERRUPTED_TURN_MARKER
    from tau2.judges.conversation.semantic import (
        _render_transcript,
        extract_agent_turns,
    )
    from tau2.metrics.interaction_quality import extract_spoken_turns

    ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            tick_duration_seconds=0.5,
            agent_chunk=AssistantMessage.voice(
                content="Qual é o número do seu pedi",
                contains_speech=True,
                utterance_ids=["u1"],
                raw_data={"was_truncated": True},
            ),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            tick_duration_seconds=0.5,
            user_chunk=UserMessage(role="user", content="é o 42", contains_speech=True),
        ),
        Tick(
            tick_id=2,
            timestamp="2",
            tick_duration_seconds=0.5,
            agent_chunk=AssistantMessage.voice(
                content="obrigado", contains_speech=True, utterance_ids=["u2"]
            ),
        ),
    ]
    sim = _SimulationRun(
        id="s-int",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.5,
        termination_reason="agent_stop",
        ticks=ticks,
    )
    transcript = _render_transcript(sim)
    assert (
        f"AGENT TURN 0: Qual é o número do seu pedi {INTERRUPTED_TURN_MARKER}"
        in transcript
    )
    assert f"obrigado {INTERRUPTED_TURN_MARKER}" not in transcript
    assert "CALLER: é o 42" in transcript
    # The conciseness turn inputs carry the marker too...
    turns = dict(extract_agent_turns(sim))
    assert turns[0].endswith(INTERRUPTED_TURN_MARKER)
    assert turns[1] == "obrigado"
    # ...while the raw spoken turns never do (deterministic/corpus consumers).
    assert all(
        INTERRUPTED_TURN_MARKER not in turn.text for turn in extract_spoken_turns(sim)
    )
    # Both semantic prompts explain the marker.
    from tau2.judges.conversation.semantic import (
        _CONCISENESS_SYSTEM,
        _PROGRESSION_SYSTEM,
    )

    assert '"[cut off by caller]"' in _CONCISENESS_SYSTEM
    assert '"[cut off by caller]"' in _PROGRESSION_SYSTEM


def test_conciseness_without_agent_turns_is_na():
    assert aggregate_conciseness([]).score is None


def test_semantic_aggregators_follow_eva_rules():
    conciseness = aggregate_conciseness(
        [
            EvaConcisenessTurnResult(turn_id=0, rating=3),
            EvaConcisenessTurnResult(turn_id=1, rating=2, evidence="minor filler"),
            EvaConcisenessTurnResult(turn_id=2, rating=1, evidence="list exhaustion"),
        ]
    )
    assert conciseness.score == pytest.approx(0.5)

    flagged_names = {
        EvaProgressionDimensionName.INFORMATION_LOSS,
        EvaProgressionDimensionName.REDUNDANT_STATEMENTS,
    }
    dimensions = [
        EvaProgressionDimension(
            name=name,
            flagged=name in flagged_names,
            rating=2 if name in flagged_names else 3,
            evidence="repeated turns 2 and 4" if name in flagged_names else "",
        )
        for name in EvaProgressionDimensionName
    ]
    progression = aggregate_progression(dimensions)
    assert progression.rating == 2
    assert progression.score == pytest.approx(0.5)

    dimensions[0] = dimensions[0].model_copy(
        update={"flagged": True, "rating": 1, "evidence": "three wasted lookups"}
    )
    progression = aggregate_progression(dimensions)
    assert progression.rating == 1
    assert progression.score == 0.0


def test_headline_is_per_dimension_and_per_turn_with_composites_in_shadow(
    tmp_path, monkeypatch
):
    task = _task()
    results = _results_fixture([_sim("s1", "t1")], [task])
    results_path = tmp_path / "results.json"
    results.save(results_path, format="json")

    calls = {"conciseness": 0, "progression": 0}
    _install_fake_judges(monkeypatch, calls, flag_information_loss=True)

    artifact = run_conversation_judging(
        [results_path],
        out_path=tmp_path / "conversation.json",
        cache_root=tmp_path / "cache",
        config=ConversationJudgeConfig(
            include_ours=False, ours_llm_judge=False, max_concurrency=1
        ),
    )
    call = artifact.calls[0]
    # HEADLINE: every dimension present once, verdict-level.
    assert {d.name for d in call.progression_dimensions} == set(
        EvaProgressionDimensionName
    )
    flagged = [d for d in call.progression_dimensions if d.flagged]
    assert [d.name for d in flagged] == [EvaProgressionDimensionName.INFORMATION_LOSS]
    # HEADLINE: per-turn conciseness with closed failure modes.
    assert len(call.conciseness_turns) == 1
    assert call.conciseness_turns[0].failure_modes == [
        EvaConcisenessFailureMode.VERBOSITY_OR_FILLER
    ]
    # SHADOW: the composites, labeled, structurally separate.
    shadow = call.shadow_scores
    assert shadow.label == "uncalibrated-shadow"
    assert shadow.eva_x is not None
    assert shadow.eva_x.conversation_progression.rating == 2
    assert shadow.eva_x.conciseness.score == pytest.approx(0.5)
    assert shadow.eva_x.turn_taking.score == pytest.approx(1.0)
    # Conjunction: progression 0.5 >= 0.5, conciseness 0.5 >= 0.5, tt 1.0.
    assert shadow.eva_x.passed is True
    assert shadow.ours is None  # include_ours=False

    # In-verb cell aggregates report the headline per dimension and mode.
    assert len(artifact.cells) == 1
    cell = artifact.cells[0]
    assert cell.language == "en"
    assert cell.domain == "mock"
    dims = {summary.name: summary for summary in cell.dimensions}
    # Severity-split headline: the fake judge flags information_loss at
    # rating 2, so it is a MINOR flag — the major (rating-1) headline stays 0.
    info_loss = dims[EvaProgressionDimensionName.INFORMATION_LOSS]
    assert info_loss.minor_rate == 1.0 and info_loss.minor_count == 1
    assert info_loss.major_rate == 0.0 and info_loss.major_count == 0
    question = dims[EvaProgressionDimensionName.QUESTION_QUALITY]
    assert question.major_rate == 0.0 and question.minor_rate == 0.0
    modes = {summary.mode: summary for summary in cell.conciseness_failure_modes}
    assert modes[EvaConcisenessFailureMode.VERBOSITY_OR_FILLER].rate == 1.0
    assert cell.conciseness_mean_rating == pytest.approx(2.0)
    assert cell.shadow_scores.label == "uncalibrated-shadow"
    assert cell.shadow_scores.eva_x_pass_rate == 1.0


def test_artifact_resumes_each_paid_component(tmp_path, monkeypatch):
    task = _task()
    sim = _sim("s1", "t1")
    results = _results_fixture([sim], [task])
    results_path = tmp_path / "results.json"
    results.save(results_path, format="json")

    calls = {"conciseness": 0, "progression": 0}
    _install_fake_judges(monkeypatch, calls)
    config = ConversationJudgeConfig(ours_llm_judge=False, max_concurrency=1)
    kwargs = {
        "paths": [results_path],
        "out_path": tmp_path / "conversation.json",
        "cache_root": tmp_path / "cache",
        "config": config,
    }
    first = run_conversation_judging(**kwargs)
    second = run_conversation_judging(**kwargs)
    expected_ours = evaluate_quality(
        sim,
        task,
        domain="mock",
        settings=QualityJudgeSettings(llm_judge=False),
    )

    assert calls == {"conciseness": 1, "progression": 1}
    assert first.num_calls == second.num_calls == 1
    assert first.calls[0].shadow_scores.ours == expected_ours
    assert first.calls[0].shadow_scores.eva_x is not None
    assert first.calls[0].errors == []

    # The task contract is part of the cache identity even when the trajectory
    # itself is byte-for-byte unchanged.
    results.tasks = [
        task.model_copy(
            update={"user_scenario": UserScenario(instructions="Changed contract.")}
        )
    ]
    results.save(results_path, format="json")
    run_conversation_judging(**kwargs)
    assert calls == {"conciseness": 2, "progression": 2}


def test_errored_quality_suites_are_retried_not_served_from_cache(
    tmp_path, monkeypatch
):
    """A quality suite with errored factor checks is a partial result: it must
    not be stored, and a poisoned cache from an older build must read as a
    miss — reruns retry the suite instead of resurfacing the errors."""
    import tau2.judges.conversation.runner as runner_mod
    from tau2.data_model.simulation import QualityInfo

    task = _task()
    results = _results_fixture([_sim("s1", "t1")], [task])
    results_path = tmp_path / "results.json"
    results.save(results_path, format="json")
    _install_fake_judges(monkeypatch, {"conciseness": 0, "progression": 0})

    attempts = {"n": 0}

    def fake_quality(sim, task, *, domain, settings):
        attempts["n"] += 1
        errored = attempts["n"] == 1
        return QualityInfo(
            score=None if errored else 1.0,
            rubric_version="r",
            metrics_version="m",
            num_pass=0 if errored else 1,
            num_errors=1 if errored else 0,
        )

    monkeypatch.setattr(runner_mod, "evaluate_quality", fake_quality)
    kwargs = {
        "paths": [results_path],
        "out_path": tmp_path / "conversation.json",
        "cache_root": tmp_path / "cache",
        "config": ConversationJudgeConfig(ours_llm_judge=False, max_concurrency=1),
    }
    first = run_conversation_judging(**kwargs)
    assert first.calls[0].shadow_scores.ours.num_errors == 1
    second = run_conversation_judging(**kwargs)
    assert second.calls[0].shadow_scores.ours.num_errors == 0
    run_conversation_judging(**kwargs)
    # First run errored (not stored), second retried and cached, third reused.
    assert attempts["n"] == 2


def test_broken_inputs_go_to_the_error_channel_not_an_abort(tmp_path, monkeypatch):
    task = _task()
    good = _sim("s-good", "t1")
    orphan = _sim("s-orphan", "t-missing")
    duplicate = good.model_copy()
    mislabeled = _sim(
        "s-mislabeled",
        "t1",
        speech_environment=SpeechEnvironment(persona_id="xx_ghost", language=None),
    )
    results = _results_fixture([good, orphan, duplicate, mislabeled], [task])
    results_path = tmp_path / "results.json"
    results.save(results_path, format="json")

    calls = {"conciseness": 0, "progression": 0}
    _install_fake_judges(monkeypatch, calls)
    artifact = run_conversation_judging(
        [results_path],
        out_path=tmp_path / "conversation.json",
        cache_root=tmp_path / "cache",
        config=ConversationJudgeConfig(
            include_ours=False, ours_llm_judge=False, max_concurrency=1
        ),
    )

    by_id = {call.sim_id: call for call in artifact.calls}
    # The duplicate was dropped upfront; the batch never aborted.
    assert set(by_id) == {"s-good", "s-orphan", "s-mislabeled"}
    assert by_id["s-good"].errors == []
    assert by_id["s-good"].shadow_scores.eva_x is not None
    assert by_id["s-orphan"].errors[0].suite == "input"
    assert "missing task" in by_id["s-orphan"].errors[0].message
    assert by_id["s-orphan"].progression_dimensions == []
    assert by_id["s-mislabeled"].errors[0].suite == "input"
    assert "resolves no language" in by_id["s-mislabeled"].errors[0].message
    assert artifact.num_errors == 2


def test_mute_call_shadow_excludes_turn_taking_from_the_conjunction(
    tmp_path, monkeypatch
):
    # An agent-mute call: the embedded turn-taking component is a reasoned
    # N/A and is excluded from the conjunctive gate; the semantic components
    # still carry the verdict.
    task = _task()
    sim = _sim("s-mute", "t1").model_copy(
        update={"ticks": _ticks(_ANSWERED_TURN + _UNANSWERED_TURN * 4)}
    )
    results = _results_fixture([sim], [task])
    results_path = tmp_path / "results.json"
    results.save(results_path, format="json")

    calls = {"conciseness": 0, "progression": 0}
    _install_fake_judges(monkeypatch, calls)
    artifact = run_conversation_judging(
        [results_path],
        out_path=tmp_path / "conversation.json",
        cache_root=tmp_path / "cache",
        config=ConversationJudgeConfig(
            include_ours=False, ours_llm_judge=False, max_concurrency=1
        ),
    )
    call = artifact.calls[0]
    assert call.errors == []
    eva_x = call.shadow_scores.eva_x
    assert eva_x.turn_taking.score is None
    assert eva_x.turn_taking.not_applicable.reason == TurnTakingNAReason.AGENT_MUTE
    # progression 1.0 and conciseness 0.5 both clear their thresholds.
    assert eva_x.passed is True
    # The headline judgments are unaffected by the dead channel.
    assert len(call.progression_dimensions) == 4


def test_plain_english_speech_environment_keeps_the_documented_default(
    tmp_path, monkeypatch
):
    task = _task()
    sim = _sim("s-en", "t1", speech_environment=SpeechEnvironment())
    results = _results_fixture([sim], [task])
    results_path = tmp_path / "results.json"
    results.save(results_path, format="json")

    calls = {"conciseness": 0, "progression": 0}
    _install_fake_judges(monkeypatch, calls)
    artifact = run_conversation_judging(
        [results_path],
        out_path=tmp_path / "conversation.json",
        cache_root=tmp_path / "cache",
        config=ConversationJudgeConfig(
            include_ours=False, ours_llm_judge=False, max_concurrency=1
        ),
    )
    assert artifact.calls[0].language == "en"
    assert artifact.calls[0].errors == []


def test_cache_survives_run_dir_moves_and_in_place_reevaluation(tmp_path, monkeypatch):
    task = _task()
    sim = _sim("s1", "t1")
    results = _results_fixture([sim], [task])
    first_home = tmp_path / "runs" / "cell-a"
    first_home.mkdir(parents=True)
    results_path = first_home / "results.json"
    results.save(results_path, format="json")

    calls = {"conciseness": 0, "progression": 0}
    _install_fake_judges(monkeypatch, calls)
    config = ConversationJudgeConfig(
        include_ours=False, ours_llm_judge=False, max_concurrency=1
    )
    common = {
        "out_path": tmp_path / "conversation.json",
        "cache_root": tmp_path / "cache",
        "config": config,
    }
    run_conversation_judging([results_path], **common)
    assert calls == {"conciseness": 1, "progression": 1}

    # An in-place re-evaluation rewrites judge-irrelevant fields (rewards);
    # the paid-judgment cache must survive.
    results.simulations = [
        sim.model_copy(update={"reward_info": RewardInfo(reward=1.0)})
    ]
    results.save(results_path, format="json")
    run_conversation_judging([results_path], **common)
    assert calls == {"conciseness": 1, "progression": 1}

    # Relocating the run directory is a normal workflow; the namespace is
    # keyed to the basename plus run metadata, never the absolute path.
    second_home = tmp_path / "archive" / "cell-a"
    second_home.mkdir(parents=True)
    moved_path = second_home / "results.json"
    results.save(moved_path, format="json")
    run_conversation_judging([moved_path], **common)
    assert calls == {"conciseness": 1, "progression": 1}

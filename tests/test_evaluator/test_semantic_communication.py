"""Explicit semantic communication scoring preserves exact task reward checks."""

from types import SimpleNamespace

import pytest

from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.semantic_communication import (
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.simulation import (
    RewardInfo,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
)
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2.orchestrator.modes import CommunicationMode


def task():
    return Task(
        id="fixture",
        user_scenario={"instructions": "Ask about the refund."},
        evaluation_criteria={
            "communicate_info": ["refund is 20 dollars"],
            "reward_basis": ["DB", "COMMUNICATE"],
        },
    )


def settings():
    return SemanticCommunicateSettings(
        model="explicit-test-judge", model_args={"reasoning_effort": "low"}
    )


def scoring(judge=None):
    return CommunicateScoring(judge=judge or settings())


SUBSTRING = CommunicateScoring(scorer="substring")


def simulation():
    return SimulationRun(
        id="sim",
        task_id="fixture",
        start_time="start",
        end_time="end",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            AssistantMessage(role="assistant", content="Recibirá veinte dólares.")
        ],
    )


@pytest.mark.parametrize("language", ["en", "es", "pt", "hi", "zh", "ko"])
def test_explicit_semantic_path_handles_all_languages_and_records_contract(
    monkeypatch, language
):
    import tau2.evaluator.evaluator_communicate as module

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content='{"met": "false", "reasoning": "The required meaning was not conveyed."}'
        )

    monkeypatch.setattr(module, "generate", generate)
    result = CommunicateEvaluator.calculate_reward(
        task(),
        simulation().messages,
        scoring=scoring(),
        customer_language=language,
    )
    assert result.reward == 0 and result.communicate_checks[0].met is False
    assert result.info["communicate_scoring"]["judge"]["model"] == "explicit-test-judge"
    assert calls[0]["call_name"] == "communicate_info_eval"
    assert calls[0]["reasoning_effort"] == "low"


def test_semantic_pass_cannot_override_failed_database_check(monkeypatch):
    import tau2.evaluator.evaluator as evaluator
    import tau2.evaluator.evaluator_communicate as module

    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: SimpleNamespace(
            content='{"met": true, "reasoning": "Equivalent translation conveyed."}'
        ),
    )
    monkeypatch.setattr(
        evaluator.EnvironmentEvaluator,
        "calculate_reward",
        lambda **_: RewardInfo(reward=0, reward_breakdown={RewardType.DB: 0}),
    )
    monkeypatch.setattr(
        evaluator.ActionEvaluator,
        "calculate_reward",
        lambda **_: RewardInfo(reward=1, reward_breakdown={RewardType.ACTION: 1}),
    )
    result = evaluate_simulation(
        simulation(),
        task(),
        EvaluationType.ALL,
        False,
        "retail",
        communicate_scoring=scoring(),
        customer_language="es",
    )
    assert result.reward == 0
    assert result.reward_breakdown[RewardType.COMMUNICATE] == 1
    assert result.reward_breakdown[RewardType.DB] == 0


def test_full_duplex_uses_merged_agent_chunks(monkeypatch):
    import tau2.evaluator.evaluator_communicate as module

    calls = []
    monkeypatch.setattr(
        module,
        "generate",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(content='{"met": true, "reasoning": "Equivalent meaning."}'),
    )
    sim = simulation()
    sim.ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            agent_chunk=AssistantMessage(
                role="assistant", content="veinte ", utterance_ids=["a"]
            ),
        ),
        Tick(
            tick_id=1,
            timestamp="1",
            agent_chunk=AssistantMessage(
                role="assistant", content="dólares", utterance_ids=["a"]
            ),
        ),
    ]
    result = evaluate_simulation(
        sim,
        task(),
        EvaluationType.COMMUNICATE,
        False,
        "retail",
        mode=CommunicationMode.FULL_DUPLEX,
        communicate_scoring=scoring(),
        customer_language="es",
    )
    assert result.reward == 1
    assert len(calls) == 1
    assert calls[0]["messages"][
        1
    ].content == module.COMMUNICATE_JUDGE_USER_PROMPT_TEMPLATE.format(
        info_str="refund is 20 dollars", transcript="assistant: veinte dólares"
    )
    assert calls[0]["messages"][0].content == (
        module.COMMUNICATE_JUDGE_SYSTEM_PROMPT
        + module.COMMUNICATE_JUDGE_LANGUAGE_ADDENDUM_TEMPLATE.format(language="es")
    )


def test_malformed_reply_errors_and_legacy_does_not_call_judge(monkeypatch):
    import tau2.evaluator.evaluator_communicate as module

    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: SimpleNamespace(
            content='{"met": "perhaps", "reasoning": "Unknown."}'
        ),
    )
    with pytest.raises(ValueError):
        CommunicateEvaluator.calculate_reward(
            task(),
            simulation().messages,
            scoring=scoring(),
            customer_language=None,
        )
    monkeypatch.setattr(
        module, "generate", lambda **_: pytest.fail("Legacy matching called a judge")
    )
    assert (
        evaluate_simulation(
            simulation(),
            task(),
            EvaluationType.COMMUNICATE,
            False,
            "retail",
            communicate_scoring=SUBSTRING,
        ).reward
        == 0
    )
    assert SemanticCommunicateSettings().model == "gpt-5.5"
    assert SemanticCommunicateSettings().model_args == {"reasoning_effort": "low"}
    assert TextRunConfig().communicate_scoring == CommunicateScoring()


def test_explicit_semantic_contract_records_and_rejects_mixed_auto_resume(tmp_path):
    from tau2.data_model.simulation import Results
    from tau2.runner.checkpoint import try_resume
    from tau2.runner.helpers import get_info

    config = TextRunConfig(domain="retail", communicate_scoring=scoring())
    info = get_info(config)
    assert info.communicate_scoring == scoring()
    previous = Results(info=info, tasks=[task()], simulations=[])
    path = tmp_path / "results.json"
    previous.save(path)
    different = previous.model_copy(deep=True)
    different.info.communicate_scoring.judge.model = "another-explicit-judge"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="scorer changed"):
        try_resume(path, different, [task()], 1, auto_resume=True)
    assert path.read_bytes() == before


def test_approved_defaults_are_independent_and_forwarded(monkeypatch):
    import tau2.evaluator.evaluator_communicate as module

    calls = []
    monkeypatch.setattr(
        module,
        "generate",
        lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(content='{"met":true,"reasoning":"Equivalent meaning."}')
        ),
    )
    configured = SemanticCommunicateSettings()
    CommunicateEvaluator.calculate_reward(
        task(),
        simulation().messages,
        scoring=scoring(configured),
        customer_language="ko",
    )
    assert calls[0]["model"] == "gpt-5.5"
    assert calls[0]["reasoning_effort"] == "low"
    assert "temperature" not in calls[0]
    configured.model_args["reasoning_effort"] = "medium"
    assert SemanticCommunicateSettings().model_args == {"reasoning_effort": "low"}


@pytest.mark.parametrize("mode", list(CommunicationMode))
@pytest.mark.parametrize(
    "evaluation_type,should_evaluate",
    [
        (EvaluationType.ALL, False),
        (EvaluationType.ALL_WITH_NL_ASSERTIONS, False),
        (EvaluationType.COMMUNICATE, True),
        (EvaluationType.ALL_IGNORE_BASIS, True),
        (EvaluationType.ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS, True),
    ],
)
def test_communication_runs_only_when_required_or_explicit(
    monkeypatch, mode, evaluation_type, should_evaluate
):
    import tau2.evaluator.evaluator as evaluator
    import tau2.evaluator.evaluator_communicate as module

    calls = []
    monkeypatch.setattr(
        module,
        "generate",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(
            content='{"met": false, "reasoning": "Expected information was not stated."}'
        ),
    )
    for name in (
        "EnvironmentEvaluator",
        "FullDuplexEnvironmentEvaluator",
        "ActionEvaluator",
        "FullDuplexActionEvaluator",
        "NLAssertionsEvaluator",
        "FullDuplexNLAssertionsEvaluator",
    ):
        monkeypatch.setattr(
            getattr(evaluator, name),
            "calculate_reward",
            lambda **_: RewardInfo(reward=1),
        )
    selected = task()
    selected.evaluation_criteria.reward_basis = [RewardType.DB, RewardType.NL_ASSERTION]
    selected.evaluation_criteria.communicate_info = [
        "first requirement",
        "second requirement",
    ]
    sim = simulation()
    sim.ticks = []
    result = evaluate_simulation(
        sim,
        selected,
        evaluation_type,
        False,
        "retail",
        mode=mode,
        communicate_scoring=scoring(),
        customer_language="es",
    )
    assert len(calls) == (2 if should_evaluate else 0)
    assert result.reward == (0 if should_evaluate else 1)
    if should_evaluate:
        assert len(result.communicate_checks) == 2
        assert result.reward_breakdown[RewardType.COMMUNICATE] == 0
    else:
        assert result.communicate_checks is None
        assert RewardType.COMMUNICATE not in result.reward_breakdown
        assert result.info["communicate"] is None


@pytest.mark.parametrize(
    "criteria", [None, {"communicate_info": None}, {"communicate_info": []}]
)
@pytest.mark.parametrize("full_duplex", [False, True])
@pytest.mark.parametrize("semantic", [False, True])
def test_missing_communication_requirements_are_neutral_without_judge_calls(
    monkeypatch, criteria, full_duplex, semantic
):
    import tau2.evaluator.evaluator_communicate as module

    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: pytest.fail("Empty communication criteria called a judge"),
        raising=False,
    )
    selected = Task(
        id="empty",
        user_scenario={"instructions": "Fixture"},
        evaluation_criteria=criteria,
    )
    evaluator = (
        module.FullDuplexCommunicateEvaluator
        if full_duplex
        else module.CommunicateEvaluator
    )
    if full_duplex:
        monkeypatch.setattr(
            evaluator,
            "ticks_to_message_history",
            lambda _: pytest.fail("Empty criteria reconstructed a transcript"),
        )
    reward = evaluator.calculate_reward(
        selected,
        [],
        scoring=scoring() if semantic else SUBSTRING,
        customer_language="ko",
    )
    assert reward.reward == 1
    assert not reward.communicate_checks
    assert reward.reward_breakdown == {RewardType.COMMUNICATE: 1}
    if semantic:
        assert reward.communicate_checks == []
        assert reward.info["communicate_scoring"] == scoring().model_dump(mode="json")


def test_legacy_substring_mode_still_strips_commas_and_ignores_caller_text(monkeypatch):
    import tau2.evaluator.evaluator_communicate as module
    from tau2.data_model.message import UserMessage

    monkeypatch.setattr(
        module,
        "generate",
        lambda **_: pytest.fail("Legacy matching called a judge"),
        raising=False,
    )
    selected = task()
    selected.evaluation_criteria.communicate_info = ["1000"]
    passed = module.CommunicateEvaluator.calculate_reward(
        selected,
        [AssistantMessage(role="assistant", content="Your refund is 1,000 dollars.")],
        scoring=SUBSTRING,
    )
    failed = module.CommunicateEvaluator.calculate_reward(
        selected, [UserMessage(role="user", content="1000")], scoring=SUBSTRING
    )
    assert passed.reward == 1
    assert failed.reward == 0

"""Opt-in corrections for released voice-v2 task grading."""

import json

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.simulation import (
    Info,
    Results,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import Action
from tau2.environment.environment import EnvironmentInfo
from tau2.evaluator.task_corrections import (
    ActionEdit,
    ActionMatch,
    GoldVariant,
    TaskCorrection,
    TaskCorrectionMismatch,
    apply_edits,
    communicate_alternatives,
    corrected_task,
    gold_variants,
    load_correction_set,
    task_content_digest,
)
from tau2.multilingual.runtime import load_language
from tau2.run import get_tasks
from tau2.scripts.evaluate_trajectories import compute_simulation_rewards
from tau2.utils.utils import DATA_DIR

SET_NAME = "voice-v2-task-defects-20261029"
LANGUAGES = ("en", "es", "pt", "zh", "ko", "hi")


def _action(name: str, **arguments) -> Action:
    return Action(action_id=f"{name}_0", name=name, arguments=arguments)


# ---- Edits and variants ----


def test_drop_removes_only_matching_actions():
    actions = [
        _action("cancel_reservation", reservation_id="A"),
        _action("cancel_reservation", reservation_id="B"),
    ]
    edit = ActionEdit(
        match=ActionMatch(name="cancel_reservation", arguments={"reservation_id": "B"}),
        op="drop",
    )
    assert [a.arguments for a in apply_edits(actions, [edit])] == [
        {"reservation_id": "A"}
    ]
    # The input list is not mutated.
    assert len(actions) == 2


def test_set_arguments_overwrites_and_keeps_other_arguments():
    actions = [
        _action("cancel_pending_order", order_id="#W1", reason="no longer needed")
    ]
    edit = ActionEdit(
        match=ActionMatch(name="cancel_pending_order", arguments={"order_id": "#W1"}),
        op="set_arguments",
        arguments={"reason": "ordered by mistake"},
    )
    (edited,) = apply_edits(actions, [edit])
    assert edited.arguments == {"order_id": "#W1", "reason": "ordered by mistake"}
    assert actions[0].arguments["reason"] == "no longer needed"


def test_edit_that_matches_nothing_raises():
    edit = ActionEdit(match=ActionMatch(name="cancel_reservation"), op="drop")
    with pytest.raises(ValueError, match="matches no gold action"):
        apply_edits([_action("get_user_details", user_id="u")], [edit])


def test_gold_variants_order_primary_then_alternatives():
    task = get_tasks("airline", task_ids=["39"])[0]
    drop = ActionEdit(
        match=ActionMatch(
            name="cancel_reservation", arguments={"reservation_id": "MSJ4OA"}
        ),
        op="drop",
    )
    correction = TaskCorrection(
        id="X",
        domain="airline",
        canonical_task_id="39",
        summary="s",
        task_digests={"en": [task_content_digest(task)]},
        gold_edits=[drop],
        alternatives=[
            GoldVariant(
                description="also keep LU15PA",
                edits=[
                    ActionEdit(
                        match=ActionMatch(
                            name="cancel_reservation",
                            arguments={"reservation_id": "LU15PA"},
                        ),
                        op="drop",
                    )
                ],
            )
        ],
    )
    variants = gold_variants(task, correction)
    cancelled = [
        sorted(
            a.arguments["reservation_id"] for a in v if a.name == "cancel_reservation"
        )
        for v in variants
    ]
    assert cancelled == [
        ["8C8K4E", "LU15PA"],
        ["8C8K4E"],
    ]


def test_corrected_task_language_selection():
    task = get_tasks("retail", task_ids=["36"])[0]
    correction = load_correction_set(SET_NAME).lookup("retail", "36")
    ko = corrected_task(task, correction, "ko")
    assert ko.evaluation_criteria.nl_assertions[0].endswith("is the 카메라.")
    # Criteria not named by the correction are untouched, and the input is not mutated.
    assert ko.evaluation_criteria.actions == task.evaluation_criteria.actions
    assert task.evaluation_criteria.nl_assertions[0].endswith("is the camera.")
    with pytest.raises(ValueError, match="no NL assertions for language 'fr'"):
        corrected_task(task, correction, "fr")


def test_communicate_alternatives_per_language():
    correction = load_correction_set(SET_NAME).lookup("airline", "7")
    assert communicate_alternatives(correction, "hi") == [["130240"], ["56640"]]
    fixed = corrected_task(get_tasks("airline", task_ids=["7"])[0], correction, "hi")
    assert fixed.evaluation_criteria.communicate_info == ["130240"]


# ---- The shipped correction set ----


def _pack_task(language: str, domain: str, canonical_task_id: str):
    """The shipped task copy of ``language``'s default (regional) voice binding."""
    loaded = load_language(language, domain, mode="voice")
    (task,) = [
        t
        for t in loaded.tasks
        if loaded.metadata[t.id].canonical_task_id == canonical_task_id
    ]
    return task, loaded.scenario_mode


def test_shipped_set_matches_the_canonical_and_localized_tasks():
    """Every edit must still hit each shipped task copy's gold, and every pinned
    digest must equal the shipped task content; drift raises here first."""
    correction_set = load_correction_set(SET_NAME)
    assert {c.id for c in correction_set.corrections} == {
        "T1",
        "T2",
        "T3",
        "T3+T9",
        "T4",
        "T5",
        "T6",
        "T8",
        "T9",
    }
    for correction in correction_set.corrections:
        registry = get_tasks(correction.domain, task_ids=[correction.canonical_task_id])
        assert set(correction.task_digests) == set(LANGUAGES), correction.id
        for language in LANGUAGES:
            task, scenario_mode = _pack_task(
                language, correction.domain, correction.canonical_task_id
            )
            assert scenario_mode == ("pack" if language == "en" else "regional")
            shipped = [task] + (registry if language == "en" else [])
            assert sorted(correction.task_digests[language]) == sorted(
                {task_content_digest(t) for t in shipped}
            ), (correction.id, language)
            for copy in shipped:
                correction.check_applies(
                    copy, language=language, scenario_mode=scenario_mode
                )
                variants = gold_variants(copy, correction, language)
                assert len(variants) == 1 + len(correction.alternatives)
                corrected_task(copy, correction, language)


def test_shipped_set_targets_each_task_once():
    raw = json.loads(
        (DATA_DIR / "tau2" / "task_corrections" / f"{SET_NAME}.json").read_text()
    )
    keys = [(c["domain"], c["canonical_task_id"]) for c in raw["corrections"]]
    assert len(keys) == len(set(keys))


def test_unknown_set_name_raises():
    with pytest.raises(FileNotFoundError):
        load_correction_set("no-such-set")


# ---- Re-grading saved trajectories ----


def _info(domain: str) -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name=domain, policy="test policy"),
    )


def _sim(task_id: str, calls: list[tuple[str, dict]]) -> SimulationRun:
    messages = []
    for i, (name, arguments) in enumerate(calls):
        call_id = f"call_{i}"
        messages.append(
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
            )
        )
        messages.append(
            ToolMessage(id=call_id, role="tool", content="{}", requestor="assistant")
        )
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages,
    )


def _regrade(domain: str, task_id: str, calls, **kwargs):
    task = get_tasks(domain, task_ids=[task_id])[0]
    results = Results(
        info=_info(domain), tasks=[task], simulations=[_sim(task_id, calls)]
    )
    return compute_simulation_rewards(results, **kwargs).simulations[0].reward_info


POLICY_COMPLIANT_39 = [
    ("cancel_reservation", {"reservation_id": "8C8K4E"}),
    ("cancel_reservation", {"reservation_id": "LU15PA"}),
]


def test_airline_39_official_grading_is_unchanged_without_the_flag():
    official = _regrade("airline", "39", POLICY_COMPLIANT_39)
    assert official.db_check.db_match is False
    assert official.reward == 0.0
    assert "task_correction" not in (official.info or {})


def test_airline_39_corrected_gold_accepts_the_policy_compliant_end_state():
    corrected = _regrade(
        "airline", "39", POLICY_COMPLIANT_39, task_corrections=SET_NAME
    )
    assert corrected.db_check.db_match is True
    assert corrected.reward == 1.0
    assert corrected.info["task_correction"]["id"] == "T2"
    assert corrected.info["task_correction"]["gold_variant"] == 0


def test_airline_39_corrected_gold_rejects_cancelling_msj4oa():
    corrected = _regrade(
        "airline",
        "39",
        POLICY_COMPLIANT_39 + [("cancel_reservation", {"reservation_id": "MSJ4OA"})],
        task_corrections=SET_NAME,
    )
    assert corrected.db_check.db_match is False
    assert corrected.reward == 0.0


def test_retail_66_accepts_either_cancel_reason_only_when_corrected():
    calls = [
        (
            "cancel_pending_order",
            {"order_id": "#W3361211", "reason": "ordered by mistake"},
        )
    ]
    task = get_tasks("retail", task_ids=["66"])[0]
    # Retail 66 is NL_ASSERTION-based with no assertions, so no judge is called.
    assert not task.evaluation_criteria.nl_assertions
    assert _regrade("retail", "66", calls).db_check.db_match is False
    corrected = _regrade("retail", "66", calls, task_corrections=SET_NAME)
    assert corrected.db_check.db_match is True
    assert corrected.info["task_correction"]["gold_variant"] == 1
    assert corrected.reward == 1.0


def test_language_specific_edit_arguments():
    correction = load_correction_set(SET_NAME).lookup("airline", "29")
    task = get_tasks("airline", task_ids=["29"])[0]
    for language, amount in (("en", 312), ("zh", 2184), ("hi", 24960)):
        insured = gold_variants(task, correction, language)[1]
        (book,) = [a for a in insured if a.name == "book_reservation"]
        assert book.arguments["insurance"] == "yes"
        assert book.arguments["payment_methods"] == [
            {"payment_id": "credit_card_8003957", "amount": amount}
        ]
    with pytest.raises(ValueError, match="no edit arguments for language 'fr'"):
        gold_variants(task, correction, "fr")


def test_airline_29_insured_booking_matches_only_the_corrected_variant():
    task = get_tasks("airline", task_ids=["29"])[0]
    (book,) = [
        a for a in task.evaluation_criteria.actions if a.name == "book_reservation"
    ]
    insured = {
        **book.arguments,
        "insurance": "yes",
        "payment_methods": [{"payment_id": "credit_card_8003957", "amount": 312}],
    }
    calls = [
        ("cancel_reservation", {"reservation_id": "VA5SGQ"}),
        ("book_reservation", insured),
    ]
    assert _regrade("airline", "29", calls).db_check.db_match is False
    corrected = _regrade("airline", "29", calls, task_corrections=SET_NAME)
    assert corrected.db_check.db_match is True
    assert corrected.info["task_correction"]["gold_variant"] == 1


# ---- Scoping guards ----


def test_correction_refuses_changed_task_content():
    """An older/newer task version with the same id is not "corrected"."""
    task = get_tasks("airline", task_ids=["39"])[0]
    correction = load_correction_set(SET_NAME).lookup("airline", "39")
    correction.check_applies(task, language="en", scenario_mode=None)
    older = task.model_copy(deep=True)
    older.user_scenario.instructions.reason_for_call += " (older wording)"
    with pytest.raises(TaskCorrectionMismatch, match="different task content"):
        correction.check_applies(older, language="en", scenario_mode=None)
    with pytest.raises(TaskCorrectionMismatch, match="no task content for language"):
        correction.check_applies(task, language="fr", scenario_mode="regional")


def test_regrade_refuses_changed_task_content_only_when_corrected():
    task = get_tasks("airline", task_ids=["39"])[0]
    task.user_scenario.instructions.reason_for_call += " (older wording)"
    results = Results(
        info=_info("airline"),
        tasks=[task],
        simulations=[_sim("39", POLICY_COMPLIANT_39)],
    )
    # Official grading never consults the correction set.
    assert compute_simulation_rewards(results).simulations[0].reward_info.reward == 0
    with pytest.raises(TaskCorrectionMismatch, match="different task content"):
        compute_simulation_rewards(results, task_corrections=SET_NAME)


def test_non_english_correction_needs_a_regional_scenario():
    correction = load_correction_set(SET_NAME).lookup("airline", "7")
    regional, mode = _pack_task("ko", "airline", "7")
    assert mode == "regional"
    correction.check_applies(regional, language="ko", scenario_mode="regional")
    for scenario_mode in ("canonical", "pack", None):
        with pytest.raises(TaskCorrectionMismatch, match="scenario mode"):
            correction.check_applies(
                regional, language="ko", scenario_mode=scenario_mode
            )
    # English is written for the canonical scenario in any recorded mode.
    english = get_tasks("airline", task_ids=["7"])[0]
    correction.check_applies(english, language="en", scenario_mode=None)


def _language_results(language: str, domain: str, localize_db: bool, canonical_id):
    from tau2.data_model.run_config import AudioNativeConfig
    from tau2.data_model.simulation import VoiceRunConfig
    from tau2.runner.batch import _load_run_tasks
    from tau2.runner.helpers import get_info

    config = VoiceRunConfig(
        domain=domain,
        language=language,
        localize_db=localize_db,
        audio_native_config=AudioNativeConfig(),
    )
    tasks = _load_run_tasks(config)
    info = get_info(config, tasks=tasks)
    (task,) = [
        t for t in tasks if info.language_tasks[t.id].canonical_task_id == canonical_id
    ]
    return Results(info=info, tasks=[task], simulations=[_sim(task.id, [])]), task


def test_canonical_scenario_korean_regrade_refuses_regional_values():
    """A canonical-mode ko call keeps the USD task: airline 7 grades "1628", and
    the ko regional value "1628000" must not be applied to it."""
    results, task = _language_results("ko", "airline", False, "7")
    assert results.info.language_tasks[task.id].scenario_mode == "canonical"
    assert task.evaluation_criteria.communicate_info == ["1628"]
    assert communicate_alternatives(
        load_correction_set(SET_NAME).lookup("airline", "7"), "ko"
    )[0] == ["1628000"]
    with pytest.raises(TaskCorrectionMismatch, match="scenario mode is 'canonical'"):
        compute_simulation_rewards(results, task_corrections=SET_NAME)


# ---- Judge orchestration and recorded correction identity ----


def test_fresh_tasks_cannot_bypass_saved_task_digest():
    task = get_tasks("airline", task_ids=["39"])[0]
    task.user_scenario.instructions.reason_for_call += " (different scenario)"
    results = Results(
        info=_info("airline"),
        tasks=[task],
        simulations=[_sim("39", POLICY_COMPLIANT_39)],
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        compute_simulation_rewards(results, fresh_tasks=True, task_corrections=SET_NAME)


def test_recorded_correction_identity_changes_with_definition():
    correction_set = load_correction_set(SET_NAME)
    corrected = _regrade(
        "airline", "39", POLICY_COMPLIANT_39, task_corrections=SET_NAME
    )
    note = corrected.info["task_correction"]
    assert note["set_name"] == SET_NAME
    assert note["set_digest"] == correction_set.content_digest()
    correction_set.corrections[0].nl_assertions["*"].append("Different requirement")
    assert note["set_digest"] != correction_set.content_digest()


@pytest.mark.parametrize(
    "db_pass,alternative_pass", [(True, True), (False, True), (True, False)]
)
def test_communication_alternative_preserves_other_rewards(
    monkeypatch, db_pass, alternative_pass
):
    from tau2.data_model.simulation import CommunicateCheck, DBCheck, RewardInfo
    from tau2.data_model.tasks import RewardType
    from tau2.evaluator import task_corrections as module
    from tau2.evaluator.evaluator import EvaluationType

    task = get_tasks("airline", task_ids=["7"])[0]
    correction = load_correction_set(SET_NAME).lookup("airline", "7")
    seen = []

    def evaluate(kind, variant, **kwargs):
        seen.append((kind, variant.evaluation_criteria.communicate_info))
        if kind == EvaluationType.ENV:
            return RewardInfo(
                reward=float(db_pass),
                db_check=DBCheck(db_match=db_pass, db_reward=float(db_pass)),
            )
        if kind == EvaluationType.COMMUNICATE:
            return RewardInfo(
                reward=float(alternative_pass),
                communicate_checks=[
                    CommunicateCheck(
                        info="708", met=alternative_pass, justification="stub"
                    )
                ],
            )
        return RewardInfo(
            reward=0.0,
            db_check=DBCheck(db_match=db_pass, db_reward=float(db_pass)),
            reward_breakdown={
                RewardType.DB: float(db_pass),
                RewardType.COMMUNICATE: 0.0,
            },
            communicate_checks=[
                CommunicateCheck(info="1628", met=False, justification="stub")
            ],
        )

    monkeypatch.setattr(module, "_evaluate", evaluate)
    result = module.evaluate_simulation_with_correction(
        task=task,
        correction=correction,
        language="en",
        evaluation_type=EvaluationType.ALL,
    )
    assert seen[0] == (EvaluationType.ALL, ["1628"])
    assert seen[-1] == (EvaluationType.COMMUNICATE, ["708"])
    assert result.reward == float(db_pass and alternative_pass)
    assert result.reward_breakdown[RewardType.DB] == float(db_pass)
    assert result.reward_breakdown[RewardType.COMMUNICATE] == float(alternative_pass)
    assert result.communicate_checks[0].info == ("708" if alternative_pass else "1628")
    assert result.info["task_correction"]["communicate_alternative"] == (
        1 if alternative_pass else None
    )
    assert task.evaluation_criteria.communicate_info == ["1628"]


def test_replacement_nl_assertions_reach_evaluator(monkeypatch):
    from tau2.data_model.simulation import RewardInfo
    from tau2.evaluator import task_corrections as module
    from tau2.evaluator.evaluator import EvaluationType

    task = get_tasks("retail", task_ids=["105"])[0]
    correction = load_correction_set(SET_NAME).lookup("retail", "105")
    seen = []

    def evaluate(kind, variant, **kwargs):
        seen.append(variant.evaluation_criteria.nl_assertions)
        return RewardInfo(reward=1.0)

    monkeypatch.setattr(module, "_evaluate", evaluate)
    module.evaluate_simulation_with_correction(
        task=task,
        correction=correction,
        language="en",
        evaluation_type=EvaluationType.ALL,
    )
    assert seen == [correction.nl_assertions["*"]]
    assert task.evaluation_criteria.nl_assertions != seen[0]


def test_env_only_accepts_alternative_with_real_evaluator():
    from tau2.evaluator.evaluator import EvaluationType

    calls = [
        (
            "cancel_pending_order",
            {"order_id": "#W3361211", "reason": "ordered by mistake"},
        )
    ]
    assert (
        _regrade("retail", "66", calls, evaluation_type=EvaluationType.ENV).reward == 0
    )
    result = _regrade(
        "retail",
        "66",
        calls,
        evaluation_type=EvaluationType.ENV,
        task_corrections=SET_NAME,
    )
    assert result.reward == 1
    assert result.db_check.db_match
    assert result.info["task_correction"]["gold_variant"] == 1


def test_communicate_only_accepts_alternative_with_real_evaluator():
    from tau2.data_model.semantic_communication import CommunicateScoring
    from tau2.evaluator.evaluator import EvaluationType

    task = get_tasks("airline", task_ids=["7"])[0]
    info = _info("airline")
    info.communicate_scoring = CommunicateScoring(scorer="substring")
    simulation = _sim("7", [])
    simulation.messages = [
        AssistantMessage(role="assistant", content="The other reservations cost 708.")
    ]
    results = Results(info=info, tasks=[task], simulations=[simulation])
    official = compute_simulation_rewards(
        results, evaluation_type=EvaluationType.COMMUNICATE
    )
    assert official.simulations[0].reward_info.reward == 0
    corrected = compute_simulation_rewards(
        results, evaluation_type=EvaluationType.COMMUNICATE, task_corrections=SET_NAME
    )
    result = corrected.simulations[0].reward_info
    assert result.reward == 1
    assert result.communicate_checks[0].info == "708"
    assert result.info["task_correction"]["communicate_alternative"] == 1

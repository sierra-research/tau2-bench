"""Approved language rules compile exactly without inheriting production validation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.data_model.judges import (
    JudgeInput,
    MeasureVerdict,
    NativenessCriterion,
    PlanAction,
    UnitVerdict,
)
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.judges.authoring_models import CandidateJudge
from tau2.judges.authoring_registry import list_authoring_catalog
from tau2.judges.candidates import candidate_catalog
from tau2.judges.catalog import load_catalog
from tau2.judges.criteria import CriteriaJudge
from tau2.judges.run_evidence import RunEvidence
from tau2.judges.suite import execute_suite, plan_suite, read_verdicts
from tau2.judges.validation import production_catalog

ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).parent / "fixtures/naturalness_review.json"
CANDIDATES = ROOT / "data/tau2/judge_catalog"
REVISION = "nativeness-rubric-v21-language-specific-draft2"
LANGUAGES = ("es", "pt", "hi", "ko", "zh")
KO_CONTRACT_ID = "naturalness-v17-rubric-v21-ko-indirect-honorific-v1"


def candidate_path(language):
    return CANDIDATES / f"naturalness-{language}-{REVISION}.json"


@pytest.fixture(params=LANGUAGES)
def reviewed_candidate(request):
    return CandidateJudge.model_validate_json(candidate_path(request.param).read_text())


def test_approved_wording_wrappers_and_identity(reviewed_candidate):
    candidate = reviewed_candidate
    review = json.loads(REVIEW.read_text())
    language = candidate.scope.language
    approved = review["proposed_contract"]
    old = review["old_contract"]
    assert candidate.contract.nativeness_criteria == {
        language: [
            NativenessCriterion.model_validate(row)
            for row in approved["nativeness_criteria"][language]
        ]
    }
    assert candidate.contract.prompt.model_dump() == approved["prompt"]
    assert candidate.contract.model == old["model"]
    assert candidate.contract.model_args == old["model_args"]
    assert candidate.contract.implementation == "naturalness-v17"
    assert candidate.contract.input_exclusions == (
        "injected-opener-and-transfer-line-v1"
    )
    assert candidate.contract.contract_id != old["contract_id"]
    assert candidate.contract.prompt.rubric_version != old["prompt"]["rubric_version"]
    assert candidate.scope.evaluation_level == "utterance"
    assert candidate.scope.agent_gender is None
    assert candidate.status == "unvalidated"
    candidate.review.require(candidate.authoring)


def test_reviewed_v21_is_default_with_validation_still_pending():
    listing = list_authoring_catalog()
    entries = [
        row
        for row in listing.entries
        if row.source == "reviewed_candidate" and row.revision == REVISION
    ]
    assert {row.language for row in entries} == set(LANGUAGES)
    assert len(entries) == 5
    assert all(row.status == "unvalidated" for row in entries)
    production = load_catalog()
    reviewed = json.loads(REVIEW.read_text())["proposed_contract"]
    assert set(production.unvalidated_contract_ids) == {
        reviewed["contract_id"],
        KO_CONTRACT_ID,
        "speech-fidelity-v7",
    }
    assert not any(
        row.source == "validated_contract" and row.diagnostic_id == "naturalness"
        for row in listing.entries
    )
    selected = next(
        contract
        for contract in production.contracts
        if contract.contract_id == reviewed["contract_id"]
    )
    assert selected.prompt.model_dump() == reviewed["prompt"]
    assert {
        language: selected.nativeness_criteria[language] for language in LANGUAGES
    } == {
        language: [
            NativenessCriterion.model_validate(row)
            for row in reviewed["nativeness_criteria"][language]
        ]
        for language in LANGUAGES
    }
    assert [row.factor_id for row in selected.nativeness_criteria["en"]] == [
        "naturalness"
    ]
    assert "contemporary U.S. English" in selected.nativeness_criteria["en"][0].question
    assert "held-out validation is pending" in selected.scope_note
    assert {
        pair.language: pair.contract_id
        for pair in production.pairs
        if pair.measure == "naturalness"
    } == {"en": reviewed["contract_id"]} | {
        language: KO_CONTRACT_ID if language == "ko" else reviewed["contract_id"]
        for language in LANGUAGES
    }
    assert selected.input_exclusions == (
        "injected-opener-transfer-line-and-control-turns-v1"
    )


def test_candidate_cannot_be_loaded_as_validated_registry(reviewed_candidate):
    with pytest.raises(ValueError):
        production_catalog([candidate_path(reviewed_candidate.scope.language)])


def test_exact_approved_criteria_reach_existing_runtime(
    reviewed_candidate, monkeypatch
):
    candidate = reviewed_candidate
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "results": [
                        {
                            "factor_id": "naturalness",
                            "opportunity": True,
                            "violated": False,
                            "severity": 0,
                            "reasoning": "Mocked protocol result; not accuracy evidence.",
                            "quote": "",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    simulation = SimulationRun(
        id="reviewed-naturalness",
        task_id="task",
        trial=0,
        start_time="a",
        end_time="b",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            UserMessage(role="user", content="previous customer"),
            AssistantMessage(role="assistant", content="previous agent"),
            UserMessage(role="user", content="current customer"),
            AssistantMessage(role="assistant", content="current agent"),
        ],
    )
    result = CriteriaJudge(candidate.contract).evaluate(
        RunEvidence(
            simulation=simulation,
            task=Task(id="task", user_scenario={"instructions": "not judge context"}),
            language=candidate.scope.language,
            agent_gender="female",
            caller_gender="male",
            policy="not judge context",
        ),
        "naturalness",
    )
    assert len(result.units) == len(calls) == 2
    for call in calls:
        assert call["model"] == "gpt-5.5"
        assert call["reasoning_effort"] == "xhigh"
        assert call["call_name"] == "naturalness_judge"
        assert call["messages"][0].content == candidate.contract.prompt.system_prompt
        prompt = call["messages"][1].content
        payload = prompt.split("Criteria:\n", 1)[1].split(
            "\n\nAbout the agent being judged:", 1
        )[0]
        assert json.loads(payload) == [
            row.model_dump()
            for row in candidate.contract.nativeness_criteria[candidate.scope.language]
        ]
        assert "Evaluation unit: one agent utterance" in prompt
        assert "Agent gender: female." in prompt
        assert "Caller gender: male." in prompt
        assert "not judge context" not in prompt
    second = calls[1]["messages"][1].content
    assert "current customer" in second and "current agent" in second
    assert "previous agent" not in second and "previous customer" not in second


def test_production_verdicts_are_stale_for_candidate(reviewed_candidate, tmp_path):
    candidate = reviewed_candidate
    inputs = [
        JudgeInput(
            domain="retail",
            run_id="run",
            simulation_id="simulation",
            trial=0,
            language=candidate.scope.language,
            agent_gender="female",
            caller_gender="male",
            input_fingerprint="a" * 64,
            has_audio=False,
        )
    ]
    stored = tmp_path / "production-verdicts.json"
    execute_suite(
        inputs,
        load_catalog(),
        stored,
        only=["naturalness"],
        backend=lambda *_: MeasureVerdict(
            units=[
                UnitVerdict(
                    unit_id="0",
                    opportunity=True,
                    violation=False,
                    evidence="Mocked protocol result; not accuracy evidence.",
                )
            ]
        ),
    )
    before = stored.read_bytes()
    plan = plan_suite(
        inputs,
        candidate_catalog(candidate),
        read_verdicts(stored),
        only=["naturalness"],
    )
    assert len(plan.items) == 1
    assert plan.items[0].action == PlanAction.STALE
    assert plan.items[0].identity.contract_id == candidate.contract.contract_id
    assert plan.items[0].identity.rubric_version == REVISION
    assert stored.read_bytes() == before

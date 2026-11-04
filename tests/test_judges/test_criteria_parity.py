"""Exact prepared requests and verdicts captured before evaluator consolidation."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.data_model.judges import JudgeContract
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import EvaluationCriteria, Task
from tau2.judges.catalog import load_catalog
from tau2.judges.criteria import CriteriaJudge
from tau2.judges.run_evidence import RunEvidence

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/criteria_protocol_v1.json").read_text()
)
HISTORICAL_NATURALNESS = Path(__file__).parent / "fixtures/naturalness_review.json"


def frozen_naturalness_contract():
    """Keep the archived v20 request parity test after v21 becomes the default."""
    payload = json.loads(HISTORICAL_NATURALNESS.read_text())["old_contract"]
    payload["implementation"] = "naturalness-v17"
    payload["input_exclusions"] = "injected-opener-and-transfer-line-v1"
    return JudgeContract.model_validate(payload)


# Deliberate, declared deltas from the archived v1 protocol. Everything else
# (prompt bytes, model, arguments, parsing, verdicts) must match exactly.
#
# 1. Stable per-implementation call names replaced the shared
#    ``nativeness_judge_batch`` label; call_name is telemetry, not request
#    content, so prompts are unchanged.
# 2. Contracts now record their benchmark-speech exclusion rule. The fixture
#    call starts with a caller turn (no injected opener) and never speaks the
#    transfer line, so the exclusions leave its prompts byte-identical.
CURRENT_CALL_NAMES = {
    "naturalness-v16-rubric-v20": "naturalness_judge",
    "hi-gender-v16-rubric-v26": "hindi_gender_judge_batch",
}


def _current_case(case):
    """Apply only the declared deltas above to the untouched archived protocol."""
    case = deepcopy(case)
    if case.get("contract_id") in CURRENT_CALL_NAMES:
        for call in case["calls"]:
            assert call["call_name"] == "nativeness_judge_batch"
            call["call_name"] = CURRENT_CALL_NAMES[case["contract_id"]]
    if "contract" in case:
        assert "input_exclusions" not in case["contract"]
        case["contract"]["input_exclusions"] = "injected-opener-and-transfer-line-v1"
    return _current_naturalness_case(case)


def _current_naturalness_case(case):
    """Apply only the declared factor rename to the untouched archived protocol."""
    if case.get("contract_id") != "naturalness-v16-rubric-v20":
        return case
    case = deepcopy(case)
    assert case["measure"] == "naturalness"
    case["contract_id"] = "naturalness-v17-rubric-v20"
    for result in case["response"]["results"]:
        assert result["factor_id"] == "natural_word_choice"
        result["factor_id"] = "naturalness"
    for call in case["calls"]:
        user_prompt = call["messages"][1]
        old = '"factor_id": "natural_word_choice"'
        assert user_prompt["content"].count(old) == 1
        user_prompt["content"] = user_prompt["content"].replace(
            old, '"factor_id": "naturalness"'
        )
    for unit in case["verdict"]["units"]:
        old = '"factor_id":"natural_word_choice"'
        assert unit["evidence"].count(old) == 1
        unit["evidence"] = unit["evidence"].replace(old, '"factor_id":"naturalness"')
    return case


def _without_creation_times(calls):
    """Message creation times are telemetry, not model request content."""
    return [
        {
            **call,
            "messages": [
                {key: value for key, value in message.items() if key != "timestamp"}
                for message in call["messages"]
            ],
        }
        for call in calls
    ]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_prepared_request_and_verdict_match_frozen_protocol(case, monkeypatch):
    case = _current_case(case)
    calls = []

    def generate(**kwargs):
        calls.append(
            {
                **kwargs,
                "messages": [
                    message.model_dump(mode="json") for message in kwargs["messages"]
                ],
            }
        )
        return SimpleNamespace(content=json.dumps(case["response"]))

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    if "contract" in case:
        contract = JudgeContract.model_validate(case["contract"])
    elif case["contract_id"] == "naturalness-v17-rubric-v20":
        contract = frozen_naturalness_contract()
    else:
        contract = next(
            c for c in load_catalog().contracts if c.contract_id == case["contract_id"]
        )
    recorded_task = Task.model_validate(FIXTURE["task"])
    recorded_task.evaluation_criteria = (
        recorded_task.evaluation_criteria or EvaluationCriteria()
    )
    recorded_task.evaluation_criteria.communicate_info = [
        *(recorded_task.evaluation_criteria.communicate_info or []),
        "PRIVATE_EVALUATOR_GOLDEN",
    ]
    verdict = CriteriaJudge(contract).evaluate(
        RunEvidence(
            simulation=SimulationRun.model_validate(FIXTURE["simulation"]),
            task=recorded_task,
            language=case["language"],
            agent_gender=case["agent_gender"],
            caller_gender=case["caller_gender"],
            policy=FIXTURE["policy"],
        ),
        case["measure"],
    )
    assert "PRIVATE_EVALUATOR_GOLDEN" not in json.dumps(calls)
    assert _without_creation_times(calls) == _without_creation_times(case["calls"])
    assert verdict.model_dump(mode="json") == {**case["verdict"], "unit_errors": []}


@pytest.mark.parametrize("quote", ["", "Not in the delivered speech"])
def test_calibrated_violation_still_requires_exact_quote(quote, monkeypatch):
    response = {
        "results": [
            {
                "factor_id": "naturalness",
                "opportunity": True,
                "violated": True,
                "severity": 1,
                "reasoning": "Fixture violation",
                "quote": quote,
            }
        ]
    }
    monkeypatch.setattr(
        "tau2.judges.criteria.generate",
        lambda **kw: SimpleNamespace(content=json.dumps(response)),
    )
    contract = frozen_naturalness_contract()
    result = CriteriaJudge(contract).evaluate(
        RunEvidence(
            simulation=SimulationRun.model_validate(FIXTURE["simulation"]),
            task=Task.model_validate(FIXTURE["task"]),
            language="es",
            agent_gender="female",
            caller_gender="male",
            policy="",
        ),
        "naturalness",
    )
    assert not result.units
    assert result.unit_errors and all(
        "quote" in error.error.lower() for error in result.unit_errors
    )


def test_diagnostics_outputs_have_generic_names(tmp_path):
    from tau2.judges.run_evidence import progress_path
    from tau2.judges.summary import DiagnosticExport

    directory = tmp_path / "run"
    directory.mkdir()
    assert progress_path(directory) == directory / "diagnostics/verdicts.json"
    assert progress_path(tmp_path / "run.json") == tmp_path / "run.diagnostics.json"
    assert (
        DiagnosticExport.model_fields["schema_version"].default
        == "diagnostics-export-v2"
    )

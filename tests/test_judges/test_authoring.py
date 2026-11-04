"""Independent agent criteria, isolated candidates, and evidence-bound promotion."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau2.judges.authoring import (
    CompileJudgeRequest,
    DraftJudgeRequest,
    compile_judge,
    draft_judge,
)
from tau2.judges.run_evidence import RunEvidence


def test_new_diagnostic_draft_is_agent_owned_and_review_bound(tmp_path, monkeypatch):
    from tau2.judges import authoring
    from tau2.utils.artifacts import ReviewApproval
    from tau2.utils.artifacts import content_digest as digest

    request = DraftJudgeRequest(
        mode="new_diagnostic",
        diagnostic_id="promise_accuracy",
        language="ro",
        variety="Romanian in Romania",
        agent_requirements="Do not promise unsupported delivery dates.",
        domains=["retail"],
        evaluation_level="call",
        agent_gender=None,
        author_model="fixture-author",
        author_model_args={},
        judge_model="fixture-judge",
        judge_model_args={},
    )
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        assert "caller persona" not in kwargs["messages"][1].content.lower()
        return SimpleNamespace(
            content=json.dumps(
                dict(
                    question="Did the agent invent a date?",
                    opportunity="Agent promises a delivery date.",
                    positive_examples="The date is not available.",
                    negative_examples="It will arrive tomorrow without evidence.",
                )
            )
        )

    monkeypatch.setattr(authoring, "generate", generate)
    path = tmp_path / "draft.json"
    draft = draft_judge(request, path)
    assert draft_judge(request, path) == draft and len(calls) == 1
    draft.criteria.negative_examples = "A specific unsupported date."
    with pytest.raises(ValueError, match="review"):
        compile_judge(
            CompileJudgeRequest(
                draft=draft,
                review=ReviewApproval(
                    reviewed_by="Fixture reviewer", content_sha256="0" * 64
                ),
                revision="fixture-v1",
            ),
            tmp_path / "bad.json",
        )
    candidate = compile_judge(
        CompileJudgeRequest(
            draft=draft,
            review=ReviewApproval(
                reviewed_by="Fixture reviewer", content_sha256=digest(draft)
            ),
            revision="fixture-v1",
        ),
        tmp_path / "candidate.json",
    )
    assert candidate.status == "unvalidated"
    assert candidate.scope.measure == "promise_accuracy"
    assert candidate.contract.model == "fixture-judge"


def evidence(candidate):
    from tau2.judges.validation import ValidationRequest
    from tau2.utils.artifacts import content_digest

    units = [
        dict(
            row_id=f"row-{i}",
            input_sha256=f"{i:064x}",
            language="ro",
            domain="retail",
            agent_gender="male" if i % 2 else "female",
        )
        for i in range(20)
    ]
    return ValidationRequest.model_validate(
        dict(
            candidate=candidate.model_dump(),
            dataset=dict(
                revision="heldout-v1",
                source_revision="review-source-v1",
                language_variety=candidate.language_variety,
                frozen_at="2026-01-01T00:00:00Z",
                frozen_before_predictions=True,
                sampling_procedure="Uniform held-out sample.",
                exclusions="None.",
                selection_independent_of_predictions=True,
                held_out_from_development=True,
                development_units=[],
                units=units,
                reviewers=[
                    dict(
                        reviewer_id="human", qualifications="Native Romanian reviewer."
                    )
                ],
                reviewer_instructions="Label unsupported date promises.",
                labels_blind_to_predictions=True,
                adjudication_policy="Independent senior reviewer resolves disagreement.",
            ),
            labels=[
                dict(
                    **u,
                    votes=[
                        dict(reviewer_id="human", opportunity=True, violation=i < 12)
                    ],
                    opportunity=True,
                    violation=i < 12,
                )
                for i, u in enumerate(units)
            ],
            predictions=dict(
                candidate_sha256=content_digest(candidate),
                rows=[
                    dict(**u, status="judged", opportunity=True, violation=i < 12)
                    for i, u in enumerate(units)
                ],
            ),
        )
    )


def test_validation_promotion_custom_id_and_exact_binding(candidate, tmp_path):
    from tau2.judges.validation import (
        PromotionRequest,
        PublishedJudge,
        production_catalog,
        promote_judge,
        validate_judge,
    )
    from tau2.utils.artifacts import ReviewApproval, content_digest

    request = evidence(candidate)
    report = validate_judge(request)
    assert (
        report.passed
        and report.precision == report.recall == report.f1 == report.kappa == 1
    )
    publication = PromotionRequest(
        evidence=request,
        report=report,
        review=ReviewApproval(
            reviewed_by="publication-human", content_sha256=content_digest(report)
        ),
    )
    target = tmp_path / "registry.json"
    promoted = promote_judge(publication, target)
    assert promote_judge(publication, target) == promoted
    loaded = production_catalog([target])
    assert any(
        p.measure == "promise_accuracy" and p.language == "ro" for p in loaded.pairs
    )
    with pytest.raises(ValueError):
        production_catalog([tmp_path / "compiled.json"])
    tampered = promoted.model_dump()
    tampered["publication"]["evidence"]["candidate"]["contract"]["model"] = (
        "different-model"
    )
    with pytest.raises(ValueError, match="reviewed authoring"):
        PublishedJudge.model_validate(tampered)
    request.predictions.rows.pop()
    broken = validate_judge(request)
    assert not broken.complete and not broken.passed
    with pytest.raises(ValueError, match="passing validation"):
        promote_judge(
            PromotionRequest(
                evidence=request,
                report=broken,
                review=ReviewApproval(
                    reviewed_by="human", content_sha256=content_digest(broken)
                ),
            ),
            tmp_path / "bad.json",
        )


def test_opportunity_disagreement_is_classification_error(candidate):
    from tau2.judges.validation import validate_judge

    request = evidence(candidate)
    request.predictions.rows[0].opportunity = False
    request.predictions.rows[0].violation = False
    request.labels[12].opportunity = False
    request.labels[12].votes[0].opportunity = False
    request.predictions.rows[12].violation = True
    report = validate_judge(request)
    assert report.complete and not report.errors
    assert (
        report.true_positive,
        report.false_negative,
        report.false_positive,
        report.true_negative,
    ) == (11, 1, 1, 7)
    assert report.passed


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "overlap",
        "model",
        "error",
        "missing",
        "unblind",
        "scope",
        "insufficient",
        "threshold",
    ],
)
def test_validation_rejects_bad_evidence(candidate, mutation):
    from tau2.judges.validation import validate_judge

    request = evidence(candidate)
    if mutation == "duplicate":
        request.predictions.rows.append(request.predictions.rows[0])
    elif mutation == "overlap":
        request.dataset.development_units.append(request.dataset.units[0])
    elif mutation == "model":
        request.predictions.candidate_sha256 = "a" * 64
    elif mutation == "error":
        row = request.predictions.rows[0]
        row.status = "error"
        row.opportunity = row.violation = None
        row.error = "Provider failure"
    elif mutation == "missing":
        request.predictions.rows.pop()
    elif mutation == "unblind":
        request.dataset.labels_blind_to_predictions = False
    elif mutation == "scope":
        request.dataset.units[0].domain = "airline"
    elif mutation == "insufficient":
        for i in range(9, 12):
            request.labels[i].violation = False
            request.labels[i].votes[0].violation = False
            request.predictions.rows[i].violation = False
    else:
        for row in request.predictions.rows[:3]:
            row.violation = False
    report = validate_judge(request)
    assert not report.passed
    if mutation in ("insufficient", "threshold"):
        assert report.complete
    else:
        assert not report.complete


def test_all_authoring_modes_reference_only_selected_factor(
    tmp_path, monkeypatch, candidate
):
    from tau2.judges import authoring
    from tau2.judges.authoring_models import JudgeDraft
    from tau2.utils.artifacts import ReviewApproval, content_digest

    seen = []
    monkeypatch.setattr(
        authoring,
        "generate",
        lambda **kw: (
            seen.append(kw)
            or SimpleNamespace(content=candidate.authoring.criteria.model_dump_json())
        ),
    )
    base = candidate.authoring.request.model_dump()
    base.update(
        mode="adapt_diagnostic",
        source_contract_id="authoring-catalog",
        source_measure="naturalness",
        diagnostic_id="naturalness",
    )
    draft = draft_judge(DraftJudgeRequest.model_validate(base), tmp_path / "adapt.json")
    assert draft.reference.criterion.factor_id == "naturalness"
    assert len(json.loads(seen[0]["messages"][1].content)["reference"]) < 10
    base.update(
        mode="naturalness_extension",
        diagnostic_id="naturalness",
        evaluation_level="utterance",
        source_contract_id=None,
        source_measure=None,
    )
    draft = draft_judge(
        DraftJudgeRequest.model_validate(base), tmp_path / "natural.json"
    )
    compiled = compile_judge(
        CompileJudgeRequest(
            draft=draft,
            review=ReviewApproval(
                reviewed_by="human", content_sha256=content_digest(draft)
            ),
            revision="v1",
        ),
        tmp_path / "natural-candidate.json",
    )
    assert compiled.contract.implementation == "naturalness-v17"
    assert compiled.contract.input_exclusions == "injected-opener-and-transfer-line-v1"
    assert (
        compiled.contract.nativeness_criteria[draft.request.language][0].factor_id
        == "naturalness"
    )
    assert (
        compiled.contract.prompt.system_prompt == draft.reference.template.system_prompt
    )
    edited = JudgeDraft.model_validate_json((tmp_path / "adapt.json").read_text())
    edited.reference.source_sha256 = "a" * 64
    (tmp_path / "adapt.json").write_text(edited.model_dump_json())
    with pytest.raises(ValueError, match="Stale"):
        draft_judge(edited.request, tmp_path / "adapt.json")


# Digests of selected judging content (model, arguments, prompt, criteria,
# speech material, scope). The v21 naturalness digest pins reviewed wording, not
# held-out validation. Identity/provenance fields are versioned separately.
FROZEN_CONTENT_SHA256 = {
    "nativeness-v15-rubric-v19": "ef1ef0903090df88530826d7eab582bd46f3e97463b2dc8347c52dd354c4896d",
    "naturalness-v17-rubric-v21-language-specific-draft2": "9f1f2ca4cfe54f86ed438d3a8d9329b19a7a3ec629bf423a91e038add107ffa2",
    "naturalness-v17-rubric-v21-ko-indirect-honorific-v1": "f63d0206f2ea15d0c94e6204424b838f097aee16e7b334e0f91994d0865962dc",
    "hi-gender-v16-rubric-v26": "1577c154746efff554d87260f3b515d1fab793e8867ca7438640823011c95652",
    "quality-v7-rubric-v4": "879c417be526c964d1513919cdd5263bb5b12b8bdbb1ee46e272a6f5913b269e",
    "speech-fidelity-v7": "3699eca5008e21ea0106caae747384ad6f97b3aa479da1a82b6fdbad4217ce2c",
}
PAIRS_SHA256 = "2f34e3e24aa7bb3e5b657c8ad0f7d50f12430fd47355e7af38af50cd04f46d0a"


def _sha256(value):
    import hashlib

    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def test_frozen_contract_payloads_are_unchanged():
    from pathlib import Path

    import tau2.judges

    catalog = json.loads(
        (Path(tau2.judges.__file__).parent / "production_judges.json").read_text()
    )
    provenance = {
        "contract_id",
        "source_revision",
        "implementation",
        "input_exclusions",
    }
    assert {
        row["contract_id"]: _sha256(
            {key: value for key, value in row.items() if key not in provenance}
        )
        for row in catalog["contracts"]
    } == FROZEN_CONTENT_SHA256
    assert _sha256(catalog["pairs"]) == PAIRS_SHA256


def test_custom_call_level_uses_real_turns_and_one_model_call(candidate, monkeypatch):
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.judges.criteria import CriteriaJudge

    seen = []
    monkeypatch.setattr(
        "tau2.judges.criteria.generate",
        lambda **kw: (
            seen.append(kw)
            or SimpleNamespace(
                content='{"opportunity":true,"violation":false,"evidence":"supported"}'
            )
        ),
    )
    sim = SimulationRun(
        id="fixture",
        task_id="task",
        start_time="start",
        end_time="end",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=[
            UserMessage(role="user", content="When?"),
            AssistantMessage(
                role="assistant",
                content="Tomorrow. The literal code is {customer_context}.",
            ),
            UserMessage(role="user", content="Sure?"),
            AssistantMessage(role="assistant", content="Yes."),
        ],
    )
    from tau2.data_model.tasks import Task

    result = CriteriaJudge(candidate.contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="task", user_scenario={"instructions": "Fixture caller."}),
            language="ro",
            agent_gender="female",
            caller_gender="male",
            policy="Delivery policy.",
        ),
        "promise_accuracy",
    )
    assert len(seen) == 1 and [u.unit_id for u in result.units] == ["call"]
    assert (
        "Tomorrow." in seen[0]["messages"][1].content
        and "Delivery policy." in seen[0]["messages"][1].content
    )
    assert "The literal code is {customer_context}." in seen[0]["messages"][1].content


def test_cli_candidate_isolation_resume_stale_and_promotion(
    candidate, tmp_path, monkeypatch, capsys
):
    import argparse

    from tau2.data_model.message import AssistantMessage
    from tau2.data_model.simulation import (
        Results,
        RewardInfo,
        SimulationRun,
        TerminationReason,
        TextRunConfig,
    )
    from tau2.data_model.tasks import Task
    from tau2.judges.candidates import run_candidate
    from tau2.judges.cli import add_judges_args
    from tau2.judges.run_evidence import progress_path
    from tau2.judges.summary import build_export
    from tau2.judges.validation import (
        PromotionRequest,
        production_catalog,
        validate_judge,
    )
    from tau2.multilingual.factory.cli import add_factory_args
    from tau2.runner.helpers import get_info
    from tau2.utils.artifacts import ReviewApproval, content_digest

    info = get_info(TextRunConfig(domain="retail"))
    info.language = "ro"
    task = Task(
        id="task",
        user_scenario={
            "instructions": "Private caller instructions must not be agent requirements."
        },
    )
    sim = SimulationRun(
        id="sim",
        task_id="task",
        start_time="start",
        end_time="end",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        trial=0,
        reward_info=RewardInfo(reward=1),
        messages=[AssistantMessage(role="assistant", content="Tomorrow.")],
        agent_voice_gender="female",
    )
    result = Results(info=info, tasks=[task], simulations=[sim])
    path = tmp_path / "result.json"
    result.save(path, format="json")
    before = path.read_bytes()
    calls = []

    def generate(**kw):
        calls.append(kw)
        assert "Private caller instructions" not in kw["messages"][1].content
        return SimpleNamespace(
            content='{"opportunity":true,"violation":false,"evidence":"fixture"}'
        )

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    parser = argparse.ArgumentParser()
    add_judges_args(parser)
    args = parser.parse_args(
        [
            "candidate",
            str(path),
            "--candidate",
            str(tmp_path / "compiled.json"),
            "--output",
            str(tmp_path / "candidate-run"),
            "--dry-run",
        ]
    )
    args.func(args)
    assert not calls and not (tmp_path / "candidate-run").exists()
    args.dry_run = False
    args.func(args)
    assert len(calls) == 1
    args.func(args)
    assert len(calls) == 1
    assert path.read_bytes() == before and not progress_path(path).exists()
    sim.messages[0].content = "Next week."
    result.save(path, format="json")
    stale = run_candidate(path, candidate, tmp_path / "candidate-run", dry_run=True)
    assert not stale.predictions.rows and stale.plan.items[0].action.value == "stale"
    with pytest.raises(ValueError, match="separate"):
        run_candidate(path, candidate, path)
    request = evidence(candidate)
    report = validate_judge(request)
    (tmp_path / "validation.json").write_text(request.model_dump_json())
    args = parser.parse_args(
        [
            "validate",
            "--request",
            str(tmp_path / "validation.json"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    args.func(args)
    promotion = PromotionRequest(
        evidence=request,
        report=report,
        review=ReviewApproval(
            reviewed_by="human", content_sha256=content_digest(report)
        ),
    )
    (tmp_path / "promotion.json").write_text(promotion.model_dump_json())
    args = parser.parse_args(
        [
            "promote",
            "--request",
            str(tmp_path / "promotion.json"),
            "--output",
            str(tmp_path / "registry.json"),
        ]
    )
    args.func(args)
    args = parser.parse_args(
        [
            "suite",
            str(path),
            "--registry",
            str(tmp_path / "registry.json"),
            "--only",
            "promise_accuracy",
            "--dry-run",
        ]
    )
    assert (
        len(
            build_export(
                args.results,
                production_catalog(args.registry),
                only=args.only,
                trials=args.trials,
            ).plan.items
        )
        == 1
        and len(calls) == 1
    )
    args.dry_run = False
    production = build_export(
        args.results,
        production_catalog(args.registry),
        only=args.only,
        trials=args.trials,
        execute=True,
    )
    assert len(calls) == 2 and production.verdicts[0].verdict.units[0].unit_id == "call"
    assert Results.load(path).simulations[0].reward_info.reward == 1
    # Factory compilation remains offline and does not require human prediction evidence.
    factory = argparse.ArgumentParser()
    add_factory_args(factory)
    for old in ("draft-nativeness", "apply-nativeness"):
        with pytest.raises(SystemExit):
            factory.parse_args([old])
    request = CompileJudgeRequest(
        draft=candidate.authoring, review=candidate.review, revision=candidate.revision
    )
    (tmp_path / "compile-request.json").write_text(request.model_dump_json())
    args = factory.parse_args(
        [
            "compile-judge",
            "--request",
            str(tmp_path / "compile-request.json"),
            "--output",
            str(tmp_path / "cli-candidate.json"),
        ]
    )
    args.func(args)
    assert (
        json.loads((tmp_path / "cli-candidate.json").read_text())["status"]
        == "unvalidated"
    )


def test_draft_cli_and_reviewed_adaptation_keep_agent_contract_runnable(
    candidate, tmp_path, monkeypatch
):
    import argparse

    from tau2.judges.authoring_models import JudgeDraft
    from tau2.judges.authoring_registry import list_authoring_catalog
    from tau2.multilingual.factory.cli import add_factory_args
    from tau2.utils.artifacts import ReviewApproval, content_digest

    selected = next(
        p
        for p in list_authoring_catalog().entries
        if p.source == "reviewed_candidate" and p.diagnostic_id == "naturalness"
    )
    request = candidate.authoring.request.model_copy(
        update={
            "mode": "adapt_diagnostic",
            "diagnostic_id": "naturalness",
            "source_contract_id": selected.source_contract_id,
            "source_language": selected.language,
            "source_measure": "naturalness",
        }
    )
    (tmp_path / "request.json").write_text(request.model_dump_json())
    calls = []
    monkeypatch.setattr(
        "tau2.judges.authoring.generate",
        lambda **kw: (
            calls.append(kw)
            or SimpleNamespace(content=candidate.authoring.criteria.model_dump_json())
        ),
    )
    parser = argparse.ArgumentParser()
    add_factory_args(parser)
    args = parser.parse_args(
        [
            "draft-judge",
            "--request",
            str(tmp_path / "request.json"),
            "--output",
            str(tmp_path / "draft-cli.json"),
        ]
    )
    args.func(args)
    draft = JudgeDraft.model_validate_json((tmp_path / "draft-cli.json").read_text())
    assert len(calls) == 1 and calls[0]["call_name"] == "factory_draft_judge"
    assert draft.reference.language == selected.language
    reviewed = compile_judge(
        CompileJudgeRequest(
            draft=draft,
            review=ReviewApproval(
                reviewed_by="human", content_sha256=content_digest(draft)
            ),
            revision="v1",
        ),
        tmp_path / "adapted.json",
    )
    assert reviewed.contract.implementation == "agent-text-v1"
    assert (
        reviewed.contract.nativeness_criteria["ro"][0].factor_id
        == reviewed.scope.measure
    )


@pytest.mark.parametrize("revision", ["1234567", "a" * 64])
def test_candidate_revision_rejects_commit_like_labels_at_intake(revision):
    with pytest.raises(ValidationError) as error:
        CompileJudgeRequest.model_validate({"revision": revision})
    assert any(item["loc"] == ("revision",) for item in error.value.errors())


def test_candidate_export_keeps_unjudged_units_distinct(
    candidate, tmp_path, monkeypatch
):
    from tau2.data_model.judges import (
        JudgeInput,
        MeasureVerdict,
        StoredVerdict,
        SuiteResult,
        UnitError,
        UnitVerdict,
    )
    from tau2.judges.candidates import candidate_catalog, run_candidate
    from tau2.judges.suite import plan_suite

    item = JudgeInput(
        run_id="run",
        simulation_id="sim",
        trial=0,
        language="ro",
        domain="retail",
        agent_gender="female",
        input_fingerprint="a" * 64,
        has_audio=False,
    )
    catalog = candidate_catalog(candidate)
    plan = plan_suite([item], catalog, [])
    verdict = MeasureVerdict(
        units=[
            UnitVerdict(
                unit_id="good", opportunity=True, violation=False, evidence="Recorded"
            )
        ],
        unit_errors=[UnitError(unit_id="bad", kind="input", error="No reference")],
    )
    saved = StoredVerdict(identity=plan.items[0].identity, verdict=verdict)
    monkeypatch.setattr(
        "tau2.judges.candidates.judge_saved_run",
        lambda *a, **kw: SuiteResult(plan=plan, verdicts=[saved]),
    )
    output = run_candidate(tmp_path / "source", candidate, tmp_path / "export")
    good, bad = output.predictions.rows
    assert good.status == "judged" and good.violation is False
    assert bad.status == "error" and bad.opportunity is None and bad.violation is None
    assert bad.error == "No reference" and json.loads(bad.row_id)[-1] == "bad"

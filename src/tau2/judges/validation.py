"""Portable held-out validation and explicit, review-bound catalog publication."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from tau2.data_model.judges import ContractModel, JudgeCatalog
from tau2.judges.authoring_models import (
    CandidateJudge,
    Digest,
)
from tau2.judges.catalog import load_catalog
from tau2.utils.artifacts import ReviewApproval, write_artifact
from tau2.utils.artifacts import content_digest as digest


class EvidenceUnit(ContractModel):
    row_id: Annotated[
        str,
        Field(min_length=1, description="Stable run/simulation/trial/unit identity."),
    ]
    input_sha256: Annotated[
        Digest,
        Field(
            description="Fingerprint of the full recorded call inputs; row_id separately binds the extracted unit."
        ),
    ]
    language: Annotated[str, Field(description="Recorded target language.")]
    domain: Annotated[str, Field(description="Recorded domain.")]
    agent_gender: Annotated[
        Literal["male", "female", "unknown"],
        Field(description="Recorded actual agent gender."),
    ]


class Reviewer(ContractModel):
    reviewer_id: Annotated[
        str, Field(min_length=1, description="Stable human reviewer identifier.")
    ]
    qualifications: Annotated[
        str,
        Field(
            min_length=1,
            description="Language/domain qualifications, subject to publication review.",
        ),
    ]


class HumanVote(ContractModel):
    reviewer_id: Annotated[
        str, Field(description="Qualified reviewer providing this blind label.")
    ]
    opportunity: Annotated[StrictBool, Field(description="Human opportunity label.")]
    violation: Annotated[StrictBool, Field(description="Human violation label.")]

    @model_validator(mode="after")
    def opportunity_required(self):
        if self.violation and not self.opportunity:
            raise ValueError("Human violation requires opportunity")
        return self


class HumanLabel(EvidenceUnit):
    votes: Annotated[
        list[HumanVote],
        Field(
            min_length=1,
            description="Independent blind human labels before adjudication.",
        ),
    ]
    opportunity: Annotated[
        StrictBool, Field(description="Final adjudicated applicability.")
    ]
    violation: Annotated[
        StrictBool, Field(description="Final adjudicated violation label.")
    ]
    adjudicator: Annotated[
        str | None,
        Field(
            default=None,
            description="Reviewer resolving disagreement, required when votes differ.",
        ),
    ]


class Prediction(EvidenceUnit):
    status: Annotated[
        Literal["judged", "error", "missing"],
        Field(description="Failures remain explicit; never counted as nonviolations."),
    ]
    opportunity: Annotated[
        StrictBool | None,
        Field(default=None, description="Judge applicability, absent on failure."),
    ]
    violation: Annotated[
        StrictBool | None,
        Field(default=None, description="Judge violation, absent on failure."),
    ]
    error: Annotated[
        str | None,
        Field(default=None, description="Recorded execution/input/parse error."),
    ]

    @model_validator(mode="after")
    def result_or_error(self):
        if self.status == "judged":
            if (
                self.opportunity is None
                or self.violation is None
                or self.error is not None
            ):
                raise ValueError("Judged prediction needs labels and no error")
            if self.violation and not self.opportunity:
                raise ValueError("Prediction violation requires opportunity")
        elif self.opportunity is not None or self.violation is not None:
            raise ValueError("Failed prediction cannot carry successful labels")
        return self


class PredictionBatch(ContractModel):
    candidate_sha256: Annotated[
        Digest, Field(description="Exact compiled candidate evaluated.")
    ]
    rows: Annotated[
        list[Prediction], Field(description="Stable-ID predictions including failures.")
    ]


class HeldOutDataset(ContractModel):
    revision: Annotated[
        str, Field(min_length=1, description="Immutable held-out dataset revision.")
    ]
    source_revision: Annotated[
        str,
        Field(
            min_length=1,
            description="Accessible source records/revision for independent review.",
        ),
    ]
    language_variety: Annotated[
        str, Field(min_length=1, description="Target variety actually sampled.")
    ]
    frozen_at: Annotated[
        datetime, Field(description="Time dataset membership was frozen.")
    ]
    frozen_before_predictions: Annotated[
        StrictBool,
        Field(description="Human declaration that selection preceded judging."),
    ]
    sampling_procedure: Annotated[
        str,
        Field(
            min_length=1,
            description="Sampling population, inclusion and enrichment procedure.",
        ),
    ]
    exclusions: Annotated[
        str, Field(min_length=1, description="Declared exclusions, or explicitly none.")
    ]
    selection_independent_of_predictions: Annotated[
        StrictBool,
        Field(
            description="No selection by this candidate predictions; needed to estimate recall."
        ),
    ]
    held_out_from_development: Annotated[
        StrictBool,
        Field(
            description="Not used for rubric/model tuning; failures used for tuning require a new dataset."
        ),
    ]
    development_units: Annotated[
        list[EvidenceUnit],
        Field(description="Declared tuning/example units for overlap detection."),
    ]
    units: Annotated[
        list[EvidenceUnit],
        Field(
            min_length=1,
            description="Complete frozen unit population, including nonviolations.",
        ),
    ]
    reviewers: Annotated[
        list[Reviewer],
        Field(min_length=1, description="Human qualifications and stable identities."),
    ]
    reviewer_instructions: Annotated[
        str,
        Field(
            min_length=1,
            description="Human labeling protocol independent of judge outputs.",
        ),
    ]
    labels_blind_to_predictions: Annotated[
        StrictBool, Field(description="Humans did not see judge predictions.")
    ]
    adjudication_policy: Annotated[
        str, Field(min_length=1, description="Disagreement-resolution protocol.")
    ]


class ValidationRequest(ContractModel):
    candidate: Annotated[
        CandidateJudge,
        Field(description="Exact candidate whose accuracy is evaluated."),
    ]
    dataset: Annotated[
        HeldOutDataset,
        Field(description="Frozen held-out population and human process declarations."),
    ]
    labels: Annotated[
        list[HumanLabel],
        Field(description="Row-aligned human evidence, never model-generated labels."),
    ]
    predictions: Annotated[
        PredictionBatch,
        Field(
            description="Exact candidate predictions with stable unit/input identities."
        ),
    ]


class ValidationReport(ContractModel):
    protocol_version: Annotated[
        Literal["heldout-validation-v1"],
        Field(
            default="heldout-validation-v1",
            description="Strict acceptance rule and completeness semantics.",
        ),
    ]
    request_sha256: Annotated[
        Digest,
        Field(description="Exact candidate, dataset, predictions and human evidence."),
    ]
    candidate_sha256: Annotated[
        Digest, Field(description="Exact candidate model/settings/prompt/rubric/scope.")
    ]
    dataset_sha256: Annotated[
        Digest, Field(description="Exact frozen data and review protocol.")
    ]
    expected_units: Annotated[int, Field(description="Frozen evaluation population.")]
    scored_opportunities: Annotated[
        int,
        Field(
            description="Aligned usable opportunities; never includes failed predictions."
        ),
    ]
    no_opportunity_units: Annotated[
        int, Field(description="Aligned human and model no-opportunity units.")
    ]
    errors: Annotated[
        list[str],
        Field(
            description="Missing, duplicate, failed, mismatched or unsupported evidence."
        ),
    ]
    true_positive: Annotated[
        int, Field(description="Human violation correctly predicted.")
    ]
    false_positive: Annotated[
        int, Field(description="Human nonviolation incorrectly flagged.")
    ]
    false_negative: Annotated[int, Field(description="Human violation missed.")]
    true_negative: Annotated[
        int, Field(description="Human nonviolation correctly predicted.")
    ]
    precision: Annotated[
        float | None,
        Field(description="TP/(TP+FP), undefined with no predicted positives."),
    ]
    recall: Annotated[
        float | None,
        Field(description="TP/(TP+FN), undefined with no human positives."),
    ]
    f1: Annotated[float | None, Field(description="Unrounded harmonic mean.")]
    kappa: Annotated[
        float | None,
        Field(
            description="Cohen kappa; required when at least five human negatives exist."
        ),
    ]
    complete: Annotated[
        bool, Field(description="Every frozen unit aligned, applicable and error-free.")
    ]
    passed: Annotated[
        bool,
        Field(description="Complete evidence passes every numerical and support gate."),
    ]


def validate_judge(
    request: ValidationRequest, output: Path | None = None
) -> ValidationReport:
    """Recompute from row-level held-out evidence; declarations remain reviewable."""
    request = ValidationRequest.model_validate(request.model_dump())
    candidate = request.candidate
    data = request.dataset
    errors = []
    if request.predictions.candidate_sha256 != digest(candidate):
        errors.append("Predictions belong to a different candidate")
    for flag in [
        "frozen_before_predictions",
        "selection_independent_of_predictions",
        "held_out_from_development",
        "labels_blind_to_predictions",
    ]:
        if not getattr(data, flag):
            errors.append(f"Required declaration absent: {flag}")
    if data.language_variety != candidate.language_variety:
        errors.append("Dataset variety differs from candidate scope")
    maps = []
    for name, rows in [
        ("dataset", data.units),
        ("human", request.labels),
        ("prediction", request.predictions.rows),
    ]:
        mapping = {r.row_id: r for r in rows}
        if len(mapping) != len(rows):
            errors.append(f"Duplicate {name} row IDs")
        maps.append(mapping)
    frozen, labels, predictions = maps
    for name, mapping in [("human", labels), ("prediction", predictions)]:
        if set(mapping) != set(frozen):
            errors.append(
                f"{name} IDs differ from frozen population (missing={len(set(frozen) - set(mapping))}, extra={len(set(mapping) - set(frozen))})"
            )
    development_ids = {r.row_id for r in data.development_units}
    development_inputs = {r.input_sha256 for r in data.development_units}
    reviewers = {r.reviewer_id for r in data.reviewers}
    if len(reviewers) != len(data.reviewers):
        errors.append("Duplicate reviewer IDs")
    scope = candidate.scope
    if scope.domains is None or set(scope.domains) != {r.domain for r in data.units}:
        errors.append("Evidence does not cover exactly the proposed domain scope")
    expected_genders = (
        {scope.agent_gender} if scope.agent_gender else {"male", "female"}
    )
    if {r.agent_gender for r in data.units} != expected_genders:
        errors.append("Evidence does not cover exactly the proposed agent-gender scope")
    tp = fp = fn = tn = na = 0
    for key, unit in frozen.items():
        if unit.row_id in development_ids or unit.input_sha256 in development_inputs:
            errors.append(f"{key}: held-out/development overlap")
        if unit.language != scope.language:
            errors.append(f"{key}: language outside scope")
        human = labels.get(key)
        prediction = predictions.get(key)
        if human is None or prediction is None:
            continue
        if (
            EvidenceUnit.model_validate(
                human.model_dump(include=set(EvidenceUnit.model_fields))
            )
            != unit
            or EvidenceUnit.model_validate(
                prediction.model_dump(include=set(EvidenceUnit.model_fields))
            )
            != unit
        ):
            errors.append(f"{key}: input identity/scope mismatch")
            continue
        votes = {(v.opportunity, v.violation) for v in human.votes}
        vote_ids = [v.reviewer_id for v in human.votes]
        if len(set(vote_ids)) != len(vote_ids) or not set(vote_ids) <= reviewers:
            errors.append(f"{key}: unknown or duplicate human reviewer")
        if (
            len(votes) > 1 or (human.opportunity, human.violation) not in votes
        ) and human.adjudicator not in reviewers:
            errors.append(f"{key}: missing qualified adjudication")
        if human.adjudicator is not None and human.adjudicator not in reviewers:
            errors.append(f"{key}: unknown adjudicator")
        if human.violation and not human.opportunity:
            errors.append(f"{key}: human violation without opportunity")
            continue
        if prediction.status != "judged":
            errors.append(
                f"{key}: prediction {prediction.status}: {prediction.error or 'no usable result'}"
            )
            continue
        if not human.opportunity and not prediction.opportunity:
            na += 1
            continue
        if human.violation:
            if prediction.violation:
                tp += 1
            else:
                fn += 1
        elif prediction.violation:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    n = tp + fp + fn + tn
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 1
    kappa = ((tp + tn) / n - pe) / (1 - pe) if n and pe < 1 else None
    passed = (
        not errors
        and tp + fn >= 10
        and precision is not None
        and precision > 0.75
        and recall is not None
        and recall > 0.75
        and f1 is not None
        and f1 > 0.80
        and (fp + tn < 5 or (kappa is not None and kappa > 0.60))
    )
    report = ValidationReport(
        request_sha256=digest(request),
        candidate_sha256=digest(candidate),
        dataset_sha256=digest(data),
        expected_units=len(data.units),
        scored_opportunities=n,
        no_opportunity_units=na,
        errors=errors,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=kappa,
        complete=not errors,
        passed=passed,
    )
    if output:
        write_artifact(report, output)
    return report


class PromotionRequest(ContractModel):
    evidence: Annotated[
        ValidationRequest,
        Field(description="Full exact candidate and held-out evidence."),
    ]
    report: Annotated[ValidationReport, Field(description="Recomputed passing report.")]
    review: Annotated[
        ReviewApproval,
        Field(
            description="Human publication approval of the exact report and bound evidence."
        ),
    ]


class PublishedJudge(ContractModel):
    status: Annotated[
        Literal["validated"],
        Field(
            default="validated",
            description="Verified exact-scope production publication.",
        ),
    ]
    publication: Annotated[
        PromotionRequest,
        Field(description="Reproducible evidence and explicit human approval."),
    ]

    @model_validator(mode="after")
    def verified_publication(self):
        computed = validate_judge(self.publication.evidence)
        if computed != self.publication.report or not computed.passed:
            raise ValueError("Promotion requires exact recomputed passing validation")
        self.publication.review.require(computed)
        return self


def promote_judge(request: PromotionRequest, output: Path) -> PublishedJudge:
    """Export a reviewable validated bundle, never mutate packaged production data."""
    artifact = PublishedJudge(publication=request)
    write_artifact(artifact, output)
    return artifact


def production_catalog(registries: list[Path]) -> JudgeCatalog:
    """Load only proof-bearing publications; candidate JSON cannot enable judging."""
    catalog = load_catalog()
    contracts = {c.contract_id: c for c in catalog.contracts}
    pairs = {(p.language, p.measure): p for p in catalog.pairs}
    seen = set()
    revisions = []
    for path in registries:
        published = PublishedJudge.model_validate_json(path.read_text())
        candidate = published.publication.evidence.candidate
        key = (candidate.scope.language, candidate.scope.measure)
        if key in seen:
            raise ValueError(
                "Conflicting registry publications for the same language/diagnostic"
            )
        seen.add(key)
        if (
            candidate.contract.contract_id in contracts
            and contracts[candidate.contract.contract_id] != candidate.contract
        ):
            raise ValueError("Contract ID collision")
        contracts[candidate.contract.contract_id] = candidate.contract
        pairs[key] = candidate.scope
        revisions.append(digest(published))
    return (
        catalog.model_copy(
            update={
                "revision": catalog.revision + "+" + "+".join(sorted(revisions)),
                "contracts": list(contracts.values()),
                "pairs": list(pairs.values()),
            }
        )
        if registries
        else catalog
    )

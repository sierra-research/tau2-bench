"""Typed independent agent authoring and editable review inputs."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from tau2.data_model.judges import (
    ContractModel,
    DiagnosticId,
    JudgeContract,
    NativenessCriterion,
    NeutralLabel,
    Prompt,
    ValidatedPair,
)
from tau2.judges.authoring_catalog import (
    CatalogFactor,
)
from tau2.utils.artifacts import ReviewApproval


class CriteriaDraft(ContractModel):
    question: Annotated[
        str, Field(min_length=1, description="Observable independent agent question.")
    ]
    opportunity: Annotated[
        str, Field(min_length=1, description="Applicability and explicit exclusions.")
    ]
    positive_examples: Annotated[
        str, Field(min_length=1, description="Passing agent rules and examples.")
    ]
    negative_examples: Annotated[
        str,
        Field(
            min_length=1, description="Violating agent rules, examples and exceptions."
        ),
    ]


class AuthoringReference(ContractModel):
    source: Annotated[
        Literal["validated_contract", "authoring_catalog", "reviewed_candidate"],
        Field(
            description="Accuracy-backed source versus candidate-only factor source."
        ),
    ]
    source_revision: Annotated[
        str, Field(description="Exact catalog or contract revision.")
    ]
    source_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Exact source artifact identity."),
    ]
    language: Annotated[
        str | None,
        Field(
            description="Source criterion language; none for language-independent catalog factors."
        ),
    ]
    source_agent_requirements: Annotated[
        str | None,
        Field(
            default=None,
            description="Independent agent requirements from a reviewed candidate source.",
        ),
    ]
    source_review: Annotated[
        ReviewApproval | None,
        Field(
            default=None,
            description="Review attribution of the selected candidate source draft.",
        ),
    ]
    template: Annotated[
        Prompt | None,
        Field(
            default=None,
            description="Frozen reusable naturalness template, absent for generic diagnostics.",
        ),
    ]
    criterion: Annotated[
        NativenessCriterion | CatalogFactor,
        Field(
            description="Exactly the selected criterion, not other languages or caller behavior."
        ),
    ]


class DraftJudgeRequest(ContractModel):
    mode: Annotated[
        Literal["naturalness_extension", "adapt_diagnostic", "new_diagnostic"],
        Field(description="Supported authoring workflow."),
    ]
    diagnostic_id: Annotated[
        DiagnosticId,
        Field(description="Stable proposed diagnostic ID; new IDs are supported."),
    ]
    language: Annotated[
        str,
        Field(
            pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*$", description="Target language."
        ),
    ]
    variety: Annotated[
        str, Field(min_length=1, description="Exact target language variety.")
    ]
    agent_requirements: Annotated[
        str,
        Field(
            min_length=1,
            description="Independent agent expectations; not caller persona instructions.",
        ),
    ]
    domains: Annotated[
        list[str], Field(min_length=1, description="Domains proposed for validation.")
    ]
    evaluation_level: Annotated[
        Literal["call", "utterance"], Field(description="Explicit evaluation unit.")
    ]
    agent_gender: Annotated[
        Literal["male", "female"] | None,
        Field(description="Proposed participant restriction or both genders."),
    ]
    source_contract_id: Annotated[
        str | None,
        Field(default=None, description="Existing catalog contract for adaptation."),
    ]
    source_language: Annotated[
        str | None,
        Field(
            default=None, description="Language of the exact active source criterion."
        ),
    ]
    source_measure: Annotated[
        DiagnosticId | None,
        Field(default=None, description="Catalog diagnostic being adapted."),
    ]
    author_model: Annotated[
        str, Field(min_length=1, description="Explicit author model.")
    ]
    author_model_args: Annotated[
        dict[str, JsonValue], Field(description="Explicit author model arguments.")
    ]
    judge_model: Annotated[
        str,
        Field(
            min_length=1,
            description="Candidate evaluation model, independently selected.",
        ),
    ]
    judge_model_args: Annotated[
        dict[str, JsonValue], Field(description="Candidate evaluation arguments.")
    ]

    @model_validator(mode="after")
    def selected_mode(self):
        if len(set(self.domains)) != len(self.domains):
            raise ValueError("Duplicate domains")
        if self.mode == "naturalness_extension":
            if (
                self.diagnostic_id != "naturalness"
                or self.evaluation_level != "utterance"
            ):
                raise ValueError(
                    "Naturalness extensions use naturalness and utterance units"
                )
        elif self.mode == "adapt_diagnostic":
            if not self.source_contract_id or not self.source_measure:
                raise ValueError("Adaptation requires a selected catalog criterion")
            if (
                self.source_contract_id != "authoring-catalog"
                and not self.source_language
            ):
                raise ValueError(
                    "Active criterion adaptation requires its source language"
                )
        if self.mode != "adapt_diagnostic" and (
            self.source_contract_id or self.source_measure or self.source_language
        ):
            raise ValueError("Source selection applies only to adaptation")
        return self


class JudgeDraft(ContractModel):
    request: Annotated[
        DraftJudgeRequest,
        Field(
            description="Full independent agent authoring inputs and model settings."
        ),
    ]
    prompt_version: Annotated[
        Literal["judge-authoring-v1"],
        Field(
            default="judge-authoring-v1",
            description="Versioned fixed authoring instruction.",
        ),
    ]
    reference: Annotated[
        AuthoringReference | None,
        Field(
            description="Exact selected source criterion and digest; none for an entirely new diagnostic."
        ),
    ]
    criteria: Annotated[
        CriteriaDraft, Field(description="Editable agent criterion for human review.")
    ]


class CompileJudgeRequest(ContractModel):
    draft: Annotated[JudgeDraft, Field(description="Exact edited authoring draft.")]
    review: Annotated[
        ReviewApproval, Field(description="Human approval bound to that draft.")
    ]
    revision: Annotated[
        NeutralLabel,
        Field(
            min_length=1,
            description="New immutable candidate revision; a label, not a commit SHA or digest.",
        ),
    ]


Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CandidateJudge(ContractModel):
    status: Annotated[
        Literal["unvalidated"],
        Field(
            default="unvalidated",
            description="Runnable candidate; not production accuracy evidence.",
        ),
    ]
    revision: Annotated[
        NeutralLabel,
        Field(
            min_length=1,
            description="Immutable candidate revision; a label, not a commit SHA or digest.",
        ),
    ]
    language_variety: Annotated[
        str,
        Field(
            min_length=1,
            description="Exact language variety the independent agent rubric targets.",
        ),
    ]
    contract: Annotated[
        JudgeContract,
        Field(description="Exact implementation, model, prompt and reviewed criteria."),
    ]
    scope: Annotated[
        ValidatedPair,
        Field(description="Requested scope only; not a validation claim."),
    ]
    authoring: Annotated[
        JudgeDraft,
        Field(
            description="Complete reviewed independent agent authoring request and edited criteria."
        ),
    ]
    review: Annotated[
        ReviewApproval, Field(description="Review of the source authoring draft.")
    ]

    @model_validator(mode="after")
    def consistent(self):
        if self.scope.contract_id != self.contract.contract_id:
            raise ValueError("Candidate scope references a different contract")
        self.review.require(self.authoring)
        from tau2.judges.authoring import compile_definition

        expected = compile_definition(self.authoring, self.revision)
        if self.contract != expected:
            raise ValueError("Candidate contract differs from reviewed authoring")
        request = self.authoring.request
        if self.language_variety != request.variety or self.scope != ValidatedPair(
            language=request.language,
            measure=request.diagnostic_id,
            contract_id=self.contract.contract_id,
            evaluation_level=request.evaluation_level,
            agent_gender=request.agent_gender,
            domains=request.domains,
        ):
            raise ValueError("Candidate scope differs from reviewed authoring")
        return self

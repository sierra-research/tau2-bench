# Copyright Sierra
"""Reviewed authoring stage for per-language nativeness rubrics.

``draft-nativeness`` makes a review packet in the language's factory
workspace. It does not touch the pack. A language owner reviews the selected
catalog ids and target-language examples, changes ``review_status`` to
``approved``, and ``apply-nativeness`` adds the resulting ``nativeness:`` block
to ``pack_draft.yaml``. The two-step boundary is deliberate: generated rubric
content never becomes runtime judge material without a human pause.

The model may select ids from the closed catalog and author only the
language-specific rule and examples. Canonical question/opportunity/allowed/
violation text remains in :mod:`tau2.multilingual.nativeness_catalog` and is
composed with the pack content only at judge runtime.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tau2.config import (
    DEFAULT_FACTORY_MODEL,
    DEFAULT_FACTORY_NATIVENESS_REASONING,
)
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.llm import FactoryLLM, get_default_llm, json_model
from tau2.multilingual.factory.state import (
    FactoryProject,
    FactoryStage,
    StageStatus,
)
from tau2.multilingual.loader import normalize_pack_data
from tau2.multilingual.nativeness_catalog import (
    JUDGE_FACTOR_CATALOG,
    get_factor_prompt_base,
)
from tau2.multilingual.schema import (
    LanguagePack,
    LocalizationPackConfig,
    NativenessPackConfig,
    NativenessPackFactorRubric,
)
from tau2.utils.utils import prompt_sha256

NATIVENESS_REVIEW_FILENAME = "nativeness_review.yaml"
NATIVENESS_AUTHOR_CALL_NAME = "factory_draft_nativeness"
NATIVENESS_PROMPT_VERSION = "v1"
NATIVENESS_BLOCK_MARKER = "# --- nativeness rubric (agent speech)"
LEGACY_NATIVENESS_SCAFFOLD_MARKER = "# --- nativeness scoring"

NATIVENESS_AUTHOR_SYSTEM_PROMPT = """\
You author one language pack's agent-speech nativeness rubric for tau2.

Select only factors that genuinely apply to this language from the closed
catalog in the user message. Never invent or rename a factor id. The catalog's
question, opportunity, allowed behavior, and violation are canonical shared
text. Do not repeat, translate, rewrite, or return those fields.

For each selected factor, author only the language-specific material:
- a short name;
- one short, observable rule about the AGENT'S speech;
- one short, observable violation rule about the AGENT'S speech;
- one to three concrete natural AGENT utterances in the target language;
- one to three concrete violating AGENT utterances in the target language.

Use short, simple, straightforward sentences. Examples must be plausible
customer-service agent speech in the target language and native orthography.
Established loanwords are fine when they are natural for the language. Judge
only agent speech, never the customer, task, policy, tool behavior, audio
quality, or whether the answer is factually correct. Do not include research
history, calibration prose, abstract linguistics, evaluator instructions,
severity discussion, or evaluator jargon.

REGIONAL CONSISTENCY is conditional: do not select regional_consistency for
every language. If it genuinely applies, target_variety is REQUIRED and must
name the pack's exact region and variety. Enforce that target, not merely "any
internally consistent variety." Use the pack context (especially persona
locales and variety clauses) as the source of truth. Concrete reference
wording for known packs: Spanish = Peninsular Spanish (Spain; ES-MD personas);
Portuguese = Brazilian Portuguese (Brazil; SP/RJ variation allowed); Mandarin
Chinese = Mainland Putonghua in Simplified Chinese; Korean = South Korean
Standard Korean; Hindi = urban Indian Hindi/Hinglish; English = Standard
American English. Do not output target_variety for any other factor.

Return exactly one JSON object matching the supplied response schema. Do not
return markdown or commentary.\
"""

NATIVENESS_AUTHOR_USER_PROMPT_TEMPLATE = """\
Prompt version: {prompt_version}

TARGET LANGUAGE PACK CONTEXT:
{context_json}

CLOSED NATIVENESS CATALOG (reference and selection only; do not return its canonical fields):
{catalog_json}

RESPONSE JSON SCHEMA:
{schema_json}\
"""

GENERIC_TARGET_VARIETY_RE = re.compile(
    r"\b(?:any|whichever|whatever)\b.*\b(?:variety|dialect|region)\b|"
    r"\b(?:an?|one)\s+(?:internally\s+)?consistent\s+"
    r"(?:regional\s+)?(?:variety|dialect)\b|"
    r"\binternally\s+consistent\b",
    re.IGNORECASE,
)


class ReviewStatus(StrEnum):
    """Human-review state carried by the review artifact."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"


class PersonaNativenessContext(BaseModel):
    """The language-profile material from one draft persona."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: Annotated[str, Field(description="Stable persona id.")]
    locale: Annotated[
        Optional[str],
        Field(description="Pack-declared region/subdivision for this persona."),
    ] = None
    short_description: Annotated[
        str, Field(description="Short speaker/profile description from the pack.")
    ]
    pragmatics_clauses: Annotated[
        list[str],
        Field(description="Reviewed language and register clauses for this persona."),
    ]


class NativenessAuthorContext(BaseModel):
    """Typed pack context supplied to the rubric authoring prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Annotated[str, Field(description="ISO language code.")]
    display_name: Annotated[str, Field(description="Human-readable language name.")]
    scripts: Annotated[
        list[str], Field(description="Scripts declared by the draft personas.")
    ]
    translation_guidance: Annotated[
        Optional[str],
        Field(description="Pack-authored language and translation conventions."),
    ] = None
    agent_language_clause: Annotated[
        Optional[str], Field(description="Agent-side language instruction.")
    ] = None
    personas: Annotated[
        list[PersonaNativenessContext],
        Field(description="Persona language profiles, sorted by id."),
    ]
    localization: Annotated[
        Optional[LocalizationPackConfig],
        Field(description="Draft's typed language-localization conventions."),
    ] = None


class AuthoredNativenessFactor(BaseModel):
    """Only the language-specific content the authoring model may produce."""

    model_config = ConfigDict(extra="forbid")

    factor_id: Annotated[
        str, Field(description="Selected id from the closed nativeness catalog.")
    ]
    nuance: Annotated[
        str,
        Field(
            min_length=1,
            max_length=80,
            description="Short language-specific name for this speech rule.",
        ),
    ]
    target_variety: Annotated[
        Optional[str],
        Field(
            min_length=1,
            max_length=180,
            description="Exact pack target region/variety; required only for "
            "regional_consistency and forbidden for other factors.",
        ),
    ] = None
    agent_rule: Annotated[
        str,
        Field(
            min_length=1,
            max_length=280,
            description="One short observable passing rule about agent speech.",
        ),
    ]
    natural_examples: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=3,
            description="Concrete natural agent utterances in the target language.",
        ),
    ]
    violation_rule: Annotated[
        str,
        Field(
            min_length=1,
            max_length=280,
            description="One short observable violation rule about agent speech.",
        ),
    ]
    violating_examples: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=3,
            description="Concrete violating agent utterances in the target language.",
        ),
    ]

    @field_validator(
        "factor_id",
        "nuance",
        "agent_rule",
        "violation_rule",
    )
    @classmethod
    def _strip_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("nativeness rubric text must not be blank")
        return cleaned

    @field_validator("natural_examples", "violating_examples")
    @classmethod
    def _strip_non_empty_examples(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("nativeness rubric examples must not be blank")
        return cleaned

    @field_validator("target_variety")
    @classmethod
    def _strip_target_variety(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def _regional_consistency_names_one_target(self) -> "AuthoredNativenessFactor":
        if self.factor_id != "regional_consistency":
            if self.target_variety is not None:
                raise ValueError(
                    "target_variety is allowed only for regional_consistency"
                )
            return self
        if self.target_variety is None:
            raise ValueError(
                "regional_consistency requires the pack's explicit target_variety"
            )
        regional_text = " ".join(
            [self.target_variety, self.agent_rule, self.violation_rule]
        )
        if GENERIC_TARGET_VARIETY_RE.search(regional_text):
            raise ValueError(
                "regional_consistency target_variety must name and enforce the "
                "pack's exact region/variety, not accept any internally "
                "consistent variety"
            )
        return self

    def to_pack_rubric(self) -> NativenessPackFactorRubric:
        """Render this structured review row into the existing pack contract."""

        natural = "; ".join(f"“{example}”" for example in self.natural_examples)
        violating = "; ".join(f"“{example}”" for example in self.violating_examples)
        target_prefix = (
            f"Target variety: {self.target_variety}. "
            if self.factor_id == "regional_consistency"
            else ""
        )
        return NativenessPackFactorRubric(
            factor_id=self.factor_id,
            nuance=self.nuance,
            native_does=(
                f"{target_prefix}{self.agent_rule} Natural agent examples: {natural}"
            ),
            ai_likely_does=(
                f"{target_prefix}{self.violation_rule} "
                f"Violating agent examples: {violating}"
            ),
        )


class NativenessAuthorReply(BaseModel):
    """Typed response from the single nativeness authoring call."""

    model_config = ConfigDict(extra="forbid")

    judge_factors: Annotated[
        list[AuthoredNativenessFactor],
        Field(
            min_length=1,
            max_length=8,
            description="Applicable catalog factors with language-specific content.",
        ),
    ]

    @model_validator(mode="after")
    def _validate_through_pack_schema(self) -> "NativenessAuthorReply":
        # This is deliberately the existing closed-catalog + duplicate-id gate.
        # The generator does not maintain a second notion of valid factor ids.
        self.to_pack_config()
        return self

    def to_pack_config(self) -> NativenessPackConfig:
        """Build the existing runtime pack block (live factors by default)."""

        return NativenessPackConfig(
            judge_factors=[factor.to_pack_rubric() for factor in self.judge_factors]
        )


class NativenessReviewProvenance(BaseModel):
    """Reproducible authoring inputs recorded on the review packet."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_name: Annotated[str, Field(description="Stable generate() call name.")]
    prompt_version: Annotated[str, Field(description="Version of the fixed prompt.")]
    prompt_sha256: Annotated[str, Field(description="Hash of fixed prompt material.")]
    catalog_sha256: Annotated[
        str, Field(description="Hash of the closed catalog supplied to the model.")
    ]
    source_pack_sha256: Annotated[
        str, Field(description="Hash of pack_draft.yaml at authoring time.")
    ]
    pack_revision: Annotated[
        int, Field(description="Factory project pack revision at authoring time.")
    ]
    model: Annotated[str, Field(description="LLM model used for authoring.")]
    model_args: Annotated[
        dict[str, str], Field(description="Effective model arguments for authoring.")
    ]


class NativenessReviewArtifact(BaseModel):
    """Human-reviewable packet between generation and pack application."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: Annotated[
        Literal["tau2_factory_nativeness_review"],
        Field(description="Closed artifact discriminator."),
    ] = "tau2_factory_nativeness_review"
    schema_version: Annotated[
        Literal[1], Field(description="Review artifact schema version.")
    ] = 1
    language: Annotated[str, Field(description="ISO language code for this packet.")]
    display_name: Annotated[str, Field(description="Human-readable language name.")]
    review_status: Annotated[
        ReviewStatus,
        Field(
            description="Human gate. Change pending_review to approved only after "
            "reviewing every id, rule, and example."
        ),
    ] = ReviewStatus.PENDING_REVIEW
    provenance: Annotated[
        NativenessReviewProvenance,
        Field(description="Authoring model, prompt, catalog, and source provenance."),
    ]
    judge_factors: Annotated[
        list[AuthoredNativenessFactor],
        Field(min_length=1, max_length=8, description="Rubrics under human review."),
    ]

    @model_validator(mode="after")
    def _validate_through_pack_schema(self) -> "NativenessReviewArtifact":
        self.to_pack_config()
        return self

    def to_pack_config(self) -> NativenessPackConfig:
        """Build and validate the existing runtime ``nativeness:`` contract."""

        return NativenessPackConfig(
            judge_factors=[factor.to_pack_rubric() for factor in self.judge_factors]
        )


class DraftNativenessOutcome(BaseModel):
    """Result of producing or reusing a nativeness review packet."""

    language: Annotated[str, Field(description="ISO language code.")]
    review_path: Annotated[Path, Field(description="Review packet path.")]
    written: Annotated[bool, Field(description="Whether a new packet was written.")]
    artifact: Annotated[
        NativenessReviewArtifact, Field(description="Validated review packet.")
    ]


class ApplyNativenessOutcome(BaseModel):
    """Result of applying an approved packet to the draft pack."""

    language: Annotated[str, Field(description="ISO language code.")]
    pack_draft_path: Annotated[Path, Field(description="Factory draft pack path.")]
    review_path: Annotated[Path, Field(description="Applied review packet path.")]
    written: Annotated[bool, Field(description="Whether the draft changed.")]
    factor_ids: Annotated[
        list[str], Field(description="Applied closed-catalog factor ids.")
    ]


def nativeness_prompt_sha256() -> str:
    """Hash of the fixed, versioned nativeness authoring prompt."""

    return prompt_sha256(
        NATIVENESS_AUTHOR_SYSTEM_PROMPT,
        NATIVENESS_AUTHOR_USER_PROMPT_TEMPLATE,
        NATIVENESS_PROMPT_VERSION,
    )


def _catalog_payload() -> list[dict[str, object]]:
    """Stable catalog view supplied for selection, including canonical bases."""

    payload: list[dict[str, object]] = []
    for factor_id in sorted(JUDGE_FACTOR_CATALOG):
        factor = JUDGE_FACTOR_CATALOG[factor_id]
        base = get_factor_prompt_base(factor_id)
        payload.append(
            {
                "factor_id": factor_id,
                "category": factor.category,
                "default_severity": factor.default_severity,
                "canonical_question": base.question,
                "opportunity": base.opportunity,
                "allowed": base.allowed,
                "violation": base.violation,
            }
        )
    return payload


def nativeness_catalog_sha256() -> str:
    """Hash of the exact closed-catalog view supplied to the model."""

    encoded = json.dumps(
        _catalog_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_draft(lang: str) -> tuple[FactoryProject, Path, str, LanguagePack]:
    project = FactoryProject.load(lang)
    if project is None:
        raise FactoryDraftError(
            f"no factory project for '{lang}'; run `tau2 factory draft` first"
        )
    path = project.pack_draft_path
    if not path.exists():
        raise FactoryDraftError(
            f"no pack draft for '{lang}' at {path}; run `tau2 factory draft` first"
        )
    original = path.read_text()
    raw = yaml.safe_load(original)
    if not isinstance(raw, dict):
        raise FactoryDraftError(f"draft pack {path} is not a YAML mapping")
    try:
        pack = LanguagePack.model_validate(normalize_pack_data(raw, path.parent))
    except ValueError as exc:
        raise FactoryDraftError(
            f"draft pack {path} does not satisfy LanguagePack: {exc}"
        ) from exc
    if pack.language != lang:
        raise FactoryDraftError(
            f"draft pack declares language '{pack.language}', expected '{lang}'"
        )
    return project, path, original, pack


def _review_path(project: FactoryProject) -> Path:
    return project.project_dir / NATIVENESS_REVIEW_FILENAME


def load_nativeness_review(lang: str) -> Optional[NativenessReviewArtifact]:
    """Load and validate a language's review packet, if one exists."""

    project = FactoryProject.load(lang)
    if project is None:
        return None
    path = _review_path(project)
    if not path.exists():
        return None
    raw = yaml.safe_load(path.read_text())
    try:
        artifact = NativenessReviewArtifact.model_validate(raw)
    except ValueError as exc:
        raise FactoryDraftError(
            f"invalid nativeness review packet {path}: {exc}"
        ) from exc
    if artifact.language != lang:
        raise FactoryDraftError(
            f"nativeness review packet {path} is for '{artifact.language}', not '{lang}'"
        )
    return artifact


def _build_context(pack: LanguagePack) -> NativenessAuthorContext:
    scripts = sorted(
        {persona.script for persona in pack.personas.values() if persona.script}
    )
    personas = [
        PersonaNativenessContext(
            persona_id=persona_id,
            locale=persona.locale,
            short_description=persona.short_description,
            pragmatics_clauses=persona.pragmatics_clauses,
        )
        for persona_id, persona in sorted(pack.personas.items())
    ]
    return NativenessAuthorContext(
        language=pack.language,
        display_name=pack.display_name,
        scripts=scripts,
        translation_guidance=pack.translation_guidance,
        agent_language_clause=pack.agent_language_clause,
        personas=personas,
        localization=pack.localization,
    )


def build_nativeness_author_prompt(pack: LanguagePack) -> str:
    """Build the pure user prompt from typed pack context and closed catalog."""

    context = _build_context(pack)
    schema = NativenessAuthorReply.model_json_schema()
    return NATIVENESS_AUTHOR_USER_PROMPT_TEMPLATE.format(
        prompt_version=NATIVENESS_PROMPT_VERSION,
        context_json=context.model_dump_json(indent=2, exclude_none=True),
        catalog_json=json.dumps(_catalog_payload(), ensure_ascii=False, indent=2),
        schema_json=json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True),
    )


def _write_review(path: Path, artifact: NativenessReviewArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(
        artifact.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    try:
        tmp.write_text(rendered)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def draft_nativeness_review(
    lang: str,
    llm: Optional[FactoryLLM] = None,
    *,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> DraftNativenessOutcome:
    """Generate, or idempotently reuse, the pre-apply review packet."""

    project, pack_path, _original, pack = _load_draft(lang)
    review_path = _review_path(project)
    existing = load_nativeness_review(lang)
    if existing is not None:
        target = existing.to_pack_config()
        if pack.nativeness == target:
            return DraftNativenessOutcome(
                language=lang,
                review_path=review_path,
                written=False,
                artifact=existing,
            )
        current_sha = _sha256(pack_path)
        if existing.provenance.source_pack_sha256 != current_sha:
            raise FactoryDraftError(
                f"review packet {review_path} was authored for a different "
                "pack_draft.yaml. Refusing to overwrite it; review/remove the "
                "stale packet explicitly before regenerating"
            )
        return DraftNativenessOutcome(
            language=lang,
            review_path=review_path,
            written=False,
            artifact=existing,
        )
    if pack.nativeness is not None:
        raise FactoryDraftError(
            f"draft pack '{lang}' already has reviewed nativeness content; "
            "refusing to generate a replacement"
        )

    llm = llm if llm is not None else get_default_llm()
    resolved_model = model or project.llm_model or DEFAULT_FACTORY_MODEL
    resolved_reasoning = (
        reasoning_effort
        if reasoning_effort is not None
        else DEFAULT_FACTORY_NATIVENESS_REASONING
    )
    try:
        reply = json_model(
            llm,
            NativenessAuthorReply,
            resolved_model,
            NATIVENESS_AUTHOR_SYSTEM_PROMPT,
            build_nativeness_author_prompt(pack),
            call_name=NATIVENESS_AUTHOR_CALL_NAME,
            reasoning_effort=resolved_reasoning,
        )
    except ValueError as exc:
        raise FactoryDraftError(
            f"LLM call '{NATIVENESS_AUTHOR_CALL_NAME}' returned invalid "
            f"nativeness rubric content: {exc}"
        ) from exc
    artifact = NativenessReviewArtifact(
        language=lang,
        display_name=pack.display_name,
        provenance=NativenessReviewProvenance(
            call_name=NATIVENESS_AUTHOR_CALL_NAME,
            prompt_version=NATIVENESS_PROMPT_VERSION,
            prompt_sha256=nativeness_prompt_sha256(),
            catalog_sha256=nativeness_catalog_sha256(),
            source_pack_sha256=_sha256(pack_path),
            pack_revision=project.pack_revision,
            model=resolved_model,
            model_args={"reasoning_effort": resolved_reasoning},
        ),
        judge_factors=reply.judge_factors,
    )
    _write_review(review_path, artifact)
    return DraftNativenessOutcome(
        language=lang,
        review_path=review_path,
        written=True,
        artifact=artifact,
    )


def _strip_legacy_scaffold(text: str) -> str:
    """Remove only the old commented AUTHOR THIS nativeness scaffold."""

    lines = text.splitlines(keepends=True)
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(LEGACY_NATIVENESS_SCAFFOLD_MARKER)
        ),
        None,
    )
    if start is None:
        return text
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or lines[end].lstrip().startswith("#")
    ):
        if lines[end].startswith("# --- "):
            break
        end += 1
    return "".join(lines[:start] + lines[end:])


def _render_nativeness_block(
    config: NativenessPackConfig, artifact: NativenessReviewArtifact
) -> str:
    body = yaml.safe_dump(
        {
            "nativeness": config.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True
            )
        },
        sort_keys=False,
        allow_unicode=True,
    )
    provenance = artifact.provenance
    return (
        f"{NATIVENESS_BLOCK_MARKER} ---\n"
        "# Applied from the approved factory review packet; canonical shared "
        "factor text remains catalog-owned.\n"
        f"# Generated by prompt {provenance.prompt_version} "
        f"sha256:{provenance.prompt_sha256[:12]} "
        f"(model:{provenance.model}).\n" + body
    )


def apply_nativeness_review(lang: str) -> ApplyNativenessOutcome:
    """Apply an approved review packet to ``pack_draft.yaml`` without overwrite."""

    project, pack_path, original, pack = _load_draft(lang)
    review_path = _review_path(project)
    artifact = load_nativeness_review(lang)
    if artifact is None:
        raise FactoryDraftError(
            f"no nativeness review packet for '{lang}' at {review_path}; run "
            "`tau2 factory draft-nativeness` first"
        )
    target = artifact.to_pack_config()
    factor_ids = [rubric.factor_id for rubric in target.judge_factors]
    if artifact.review_status is not ReviewStatus.APPROVED:
        raise FactoryDraftError(
            f"nativeness review packet {review_path} is '{artifact.review_status}'. "
            "Review every factor and example, then set review_status: approved"
        )
    if pack.nativeness is not None:
        if (
            pack.nativeness == target
            and project.stages[FactoryStage.NATIVENESS] == StageStatus.DONE
        ):
            return ApplyNativenessOutcome(
                language=lang,
                pack_draft_path=pack_path,
                review_path=review_path,
                written=False,
                factor_ids=factor_ids,
            )
        raise FactoryDraftError(
            f"draft pack '{lang}' already has different nativeness content; "
            "or it was inserted outside the recorded apply stage. Refusing to "
            "overwrite reviewed rubrics"
        )
    current_sha = _sha256(pack_path)
    if current_sha != artifact.provenance.source_pack_sha256:
        raise FactoryDraftError(
            f"pack_draft.yaml changed after {review_path} was generated; refusing "
            "to apply a stale review packet"
        )

    base = _strip_legacy_scaffold(original).rstrip("\n")
    new_text = base + "\n\n" + _render_nativeness_block(target, artifact)
    raw = yaml.safe_load(new_text)
    try:
        LanguagePack.model_validate(normalize_pack_data(raw, pack_path.parent))
    except ValueError as exc:
        raise FactoryDraftError(
            f"approved nativeness packet does not produce a valid pack: {exc}"
        ) from exc
    tmp = pack_path.with_suffix(f"{pack_path.suffix}.tmp")
    try:
        tmp.write_text(new_text)
        tmp.replace(pack_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    project.pack_revision += 1
    project.mark_stage(FactoryStage.NATIVENESS, success=True)
    return ApplyNativenessOutcome(
        language=lang,
        pack_draft_path=pack_path,
        review_path=review_path,
        written=True,
        factor_ids=factor_ids,
    )

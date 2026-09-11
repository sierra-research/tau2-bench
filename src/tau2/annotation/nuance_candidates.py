# Copyright Sierra
"""Fixed-prompt pipeline stage: AI candidate nuances for the Nativeness Audit.

Replaces the old skill-embedded "spawn an Opus subagent with this prompt"
step (machine principle: fixed, versioned, in-code prompts through the
``generate()`` seam — never a coding agent improvising content). The prompt
text below is ported VERBATIM from the calibrated subagent prompt; its sha256
is recorded in the output so every candidate file traces to the exact prompt
that produced it.

Output: ``data/tau2/multilingual/_factory/nuance_audit/<iso>.json`` — a
validated, provenance-bearing artifact consumed by the audit tab builder.
"""

import json
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from tau2.annotation.artifacts import provenance_stamp, write_json_artifact
from tau2.annotation.models import AuditNuanceRow, AuditStatus
from tau2.config import (
    DEFAULT_NUANCE_CANDIDATES_MODEL,
    DEFAULT_NUANCE_CANDIDATES_MODEL_ARGS,
)
from tau2.data_model.message import UserMessage
from tau2.utils.llm_utils import extract_json_from_llm_response, generate
from tau2.utils.utils import prompt_sha256

NUANCE_CANDIDATES_CALL_NAME = "annotation_nuance_candidates"
NUANCE_CANDIDATES_PROMPT_VERSION = "v1"

# Ported VERBATIM from the calibrated /nativeness-audit subagent prompt
# ("The subagent prompt" in the old skill). {{LANG}} / {{HINTS}} are filled by
# nuance_candidates_prompt().
NUANCE_CANDIDATES_PROMPT = """You are a senior {{LANG}} linguist and native speaker auditing an AI "user simulator" for a voice benchmark.

CONTEXT
- We built AI callers (LLM-driven, with text-to-speech voice) that role-play CUSTOMERS phoning a
  customer-service agent (airline, retail, telecom): booking/cancelling flights, baggage, refunds,
  account changes. The simulated customer converses entirely in {{LANG}}.
- The underlying model is English-centric generating {{LANG}}. We want every way a real native
  {{LANG}} speaker would notice the caller is NOT a true native — the simulator produces
  non-native "translationese," or omits things natives do.

TASK
Enumerate specific, concrete nuances of natural spoken {{LANG}} (phone customer-service register)
that such an AI is LIKELY to get wrong, overuse, or omit. Each nuance must be a checkable claim a
native reviewer can confirm or reject.

TWO ANGLES (include both)
1) Realism: what the simulated CUSTOMER must do to sound native.
2) Agent benchmark: language the SERVICE AGENT must also handle/produce (e.g. reading back IDs,
   dates, names, emails in native form). Nuances on either side are welcome.

IN SCOPE (scope EMPHASIS differs by language — focus on what actually distinguishes a native
{{LANG}} speaker in this setting)
Word choice & idiom (translationese vs natural phrasing); register/formality (T–V, honorifics,
speech levels); domain terminology (native term vs calque/English); morphosyntax (gender/case/
agreement, measure words, classifiers, particles); how numbers, dates, money, phone numbers, names,
email addresses, confirmation codes and symbols (@ . _ -) are VERBALIZED; code-switching with English
(how much, which words); discourse (fillers, backchannels, politeness routines, openings/closings);
AND tone or intonation that CHANGES MEANING (e.g. lexical tone producing the wrong word in a tonal
language; intonation flipping statement vs question, or altering pragmatic meaning). Reviewers listen
to the audio, so meaning-bearing sound is fair game.

OUT OF SCOPE
Affective / emotional prosody (whether the caller sounds polite, angry, warm, or robotic), raw accent
authenticity, generic voice quality, TTS artifacts, background noise — UNLESS they change meaning or
word identity. Don't log "sounds unnatural/robotic"; DO log "the tone produced the wrong word / changed
the meaning."

QUALITY BAR
- Concrete: name the exact word/form/particle and give the natural native version in {{LANG}} script
  with a short gloss. Vague entries ("uses wrong tone/register") are useless.
- Each row independently verifiable by a native speaker.
- No duplicates, no pronunciation items, no generic filler.
- Order by importance (most frequent / most native-revealing first).
- Aim for 12–20 genuinely distinct, high-value nuances. Quality over quantity.

LANGUAGE-SPECIFIC HINTS (cover where they apply, and go beyond them)
{{HINTS}}

OUTPUT — return ONLY this JSON:
{"nuances":[{
  "category": "<best label; prefer the tab's categories; you may add one>",
  "title": "<short title>",
  "ai_likely_does": "<the failure hypothesis — what the simulator probably does wrong/omits>",
  "native_does": "<what a real {{LANG}} speaker does; concrete, with {{LANG}} example + gloss>",
  "severity": 1 | 2 | 3
}]}
severity: 1 = subtle, 2 = clearly off, 3 = breaks the illusion / risks task confusion.
Do NOT include an example utterance or conversation reference — a human adds those after verifying.

Think carefully and adaptively about {{LANG}}'s typology and the systematic failure modes of an
English-centric LLM before writing.
"""

# Per-language hint lines fed to {{HINTS}} — calibrated prompt material, keyed
# by ISO 639-1 code (ported verbatim from the old skill's hints table).
LANGUAGE_HINTS: dict[str, str] = {
    "es": "usted/tú (+vos); gender agreement; LatAm vs Peninsular (coger, vale, vosotros); anglicisms",
    "hi": "Hinglish code-switching; aap/tum/tu; gender (verb/adj); symbol/number verbalization",
    "ko": "존댓말 / speech levels; Sino vs native numbers + counters; 님/씨 titles; loanwords",
    "pt": "pt-BR vs EP (tô fazendo vs estou a fazer; celular vs telemóvel); você / o senhor; gender",
    "zh": "measure words (量词); 二 vs 两; phone readout (一/幺); 您/你; particles (吧/啊/呢); tone=lexical",
}


class NuanceCandidate(BaseModel):
    """One generated nuance in the on-disk ``<iso>.json`` wire shape."""

    category: Annotated[str, Field(description="Best category label for the tab.")]
    title: Annotated[str, Field(description="Short title of the nuance.")]
    ai_likely_does: Annotated[
        str, Field(description="The failure hypothesis to verify.")
    ]
    native_does: Annotated[
        str, Field(description="What a real native speaker does (with example).")
    ]
    severity: Annotated[int, Field(ge=1, le=3)] = 2

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value):
        # Models occasionally emit "2" or 2.0 — coerce, still bounds-checked.
        return int(float(value)) if isinstance(value, str) else value

    def to_row(self) -> AuditNuanceRow:
        return AuditNuanceRow(
            category=self.category,
            nuance=self.title,
            ai_does=self.ai_likely_does,
            native_does=self.native_does,
            severity=self.severity,
            status=AuditStatus.AUTO,
        )


class NuanceCandidatesResponse(BaseModel):
    """The generator's full JSON reply."""

    nuances: list[NuanceCandidate]


class NuanceCandidatesArtifact(BaseModel):
    """Provenance-bearing output of one language's candidate generation."""

    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    model: Annotated[str, Field(description="LLM used to generate the candidates.")]
    prompt_version: Annotated[str, Field(description="Fixed prompt version tag.")]
    prompt_sha256: Annotated[str, Field(description="Hash of the fixed prompt.")]
    created_at: Annotated[str, Field(description="Artifact creation time in UTC.")]
    git_sha: Annotated[str, Field(description="Repository commit at creation time.")]
    nuances: Annotated[
        list[NuanceCandidate], Field(description="Validated candidate nuances.")
    ]


def nuance_prompt_sha256() -> str:
    """sha256 of the fixed prompt template — recorded in output provenance."""
    return prompt_sha256(NUANCE_CANDIDATES_PROMPT)


def nuance_candidates_prompt(language_display: str, hints: str) -> str:
    """The fixed prompt with its two slots filled."""
    return NUANCE_CANDIDATES_PROMPT.replace("{{LANG}}", language_display).replace(
        "{{HINTS}}", hints
    )


def generate_nuance_candidates(
    iso: str,
    *,
    model: str = DEFAULT_NUANCE_CANDIDATES_MODEL,
    model_args: Optional[dict] = None,
) -> list[NuanceCandidate]:
    """Generate candidate nuances for one language through the generate() seam."""
    from tau2.annotation.sheets.audit import AUDIT_TABS

    tab = AUDIT_TABS.get(iso)
    if tab is None:
        raise ValueError(
            f"unknown audit language '{iso}' (known: {sorted(AUDIT_TABS)})"
        )
    hints = LANGUAGE_HINTS.get(iso, "")
    prompt = nuance_candidates_prompt(tab.tab_name, hints)
    reply = generate(
        model=model,
        messages=[UserMessage(role="user", content=prompt)],
        call_name=NUANCE_CANDIDATES_CALL_NAME,
        **{**DEFAULT_NUANCE_CANDIDATES_MODEL_ARGS, **(model_args or {})},
    )
    payload = json.loads(extract_json_from_llm_response(reply.content or ""))
    response = NuanceCandidatesResponse.model_validate(payload)
    logger.info(f"generated {len(response.nuances)} nuance candidates for {iso}")
    return response.nuances


def write_nuance_candidates(
    iso: str,
    candidates: list[NuanceCandidate],
    out_dir: Path,
    *,
    model: str = DEFAULT_NUANCE_CANDIDATES_MODEL,
) -> Path:
    """Persist candidates as a provenance-bearing ``<iso>.json`` artifact.

    Written atomically through the shared JSON-artifact seam, stamped with
    the write-time provenance (created_at, git_sha).
    """
    payload = NuanceCandidatesArtifact(
        language=iso,
        model=model,
        prompt_version=NUANCE_CANDIDATES_PROMPT_VERSION,
        prompt_sha256=nuance_prompt_sha256(),
        **provenance_stamp(),
        nuances=candidates,
    )
    path = write_json_artifact(Path(out_dir) / f"{iso}.json", payload)
    logger.info(f"wrote {len(candidates)} candidates -> {path}")
    return path


def load_nuance_candidates(path: Path) -> list[NuanceCandidate]:
    """Read a ``<iso>.json`` candidates file back into typed records."""
    # Legacy checked-in artifacts predate provenance metadata; the candidate
    # payload remains the stable input contract for the audit builder.
    return NuanceCandidatesResponse.model_validate_json(Path(path).read_text()).nuances

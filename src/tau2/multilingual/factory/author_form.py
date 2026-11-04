"""Fixed-prompt author-form drafting into an immutable review artifact.

The form carries only locale, register and persona sketches; task behavior,
acoustics and annotation settings are owned elsewhere.
"""

import json
from pathlib import Path

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.multilingual.factory.prompt_rules import (
    AUTHORING_RULES,
    AUTHORING_RULES_VERSION,
)
from tau2.multilingual.factory.schema import (
    AUTOFORM_CALL_NAME,
    AuthorForm,
    AuthorFormArtifact,
    AutoformRequest,
)
from tau2.utils.artifacts import content_digest, write_artifact
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

AUTOFORM_PROMPT_VERSION = "public-author-form-v4"
AUTOFORM_SYSTEM_PROMPT = (
    "You are the Tau-Voice Language Factory author-form generator. From a target "
    "language identity you produce a single JSON author form describing the personas "
    "and locale conventions a realistic voice language pack should respect. "
    "Respond with exactly one JSON object and nothing else."
)


def build_autoform_prompt(request: AutoformRequest) -> str:
    """Render the versioned authoring brief and the model-owned JSON schema."""
    return f"""Produce a Tau-Voice author form for this exact identity:
{request.identity.model_dump_json()}

The form is a human-reviewable seed, not a published language pack.
Return one JSON object matching this schema:
{json.dumps(AuthorForm.model_json_schema(), ensure_ascii=False)}

Requirements:
- Exactly two personas, one male and one female, spanning a language/register axis.
- Describe language use and register. Do not prescribe task attitude or affect;
  those come from each task.
- Locale notes cover regional names, addresses and written entity conventions.
- Preserve identity fields exactly. Do not invent voice IDs or extra fields.
- Locale notes and preferences must be prose strings, not nested objects.
- For English, follow the reviewed persona constraints in the hints; avoid accent
  mandates, nationality restrictions, and grammatical-gender lectures.

Shared authoring contract ({AUTHORING_RULES_VERSION}):
{AUTHORING_RULES}

Author hints:
{request.hints}

Reviewed persona labels (when supplied, preserve exactly for each gender):
{json.dumps(request.persona_names, ensure_ascii=False)}
"""


def run_autoform(request: AutoformRequest, output: Path) -> AuthorFormArtifact:
    """Resume a matching staged draft or generate, validate and persist one.

    Changed inputs/prompt/settings need a fresh path and are rejected before an
    LLM call. Generation failures produce no artifact and are never retried here;
    shared generate() owns provider retries and accounting.
    """
    request = AutoformRequest.model_validate(request.model_dump())
    digest = content_digest(request)
    if output.exists():
        previous = AuthorFormArtifact.model_validate_json(output.read_text())
        if (
            previous.input_sha256 == digest
            and previous.prompt_version == AUTOFORM_PROMPT_VERSION
        ):
            return previous
        raise FileExistsError("Authoring inputs changed; use a new output path")
    reply = generate(
        model=request.generation.model,
        reasoning_effort=request.generation.reasoning_effort,
        messages=[
            SystemMessage(role="system", content=AUTOFORM_SYSTEM_PROMPT),
            UserMessage(role="user", content=build_autoform_prompt(request)),
        ],
        call_name=AUTOFORM_CALL_NAME,
    )
    form = AuthorForm.model_validate_json(
        extract_json_from_llm_response(reply.content or "")
    )
    if form.identity != request.identity:
        raise ValueError(
            "Generated author form disagrees with authoritative locale identity"
        )
    if (
        request.persona_names is not None
        and {persona.gender: persona.name for persona in form.personas}
        != request.persona_names
    ):
        raise ValueError("Generated author form changed reviewed persona labels")
    artifact = AuthorFormArtifact(
        form=form,
        request=request,
        prompt_version=AUTOFORM_PROMPT_VERSION,
        input_sha256=digest,
    )
    write_artifact(artifact, output)
    return artifact

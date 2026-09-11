# Copyright Sierra
"""The human-authored (or autoform-generated) author form: schema, parsing,
fence/field-doc helpers, and the ``factory_autoform`` generation call."""

import re
from pathlib import Path
from typing import Any, Optional, Type

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from tau2.backchannel import BackchannelLevel
from tau2.config import (
    DEFAULT_FACTORY_AUTOFORM_REASONING,
    DEFAULT_FACTORY_MODEL,
)
from tau2.multilingual.factory.llm import FactoryLLM, get_default_llm

# The autoform LLM call name (the observable contract for tests and llm logs).
AUTOFORM_CALL_NAME = "factory_autoform"


class FactoryDraftError(Exception):
    """A drafting/finalize precondition or model-output problem.

    The message is always human-readable and actionable; the CLI prints it
    and exits non-zero.
    """


_FENCE_RE = re.compile(r"```([A-Za-z0-9_-]*)[ \t]*\n(.*?)```", re.DOTALL)


def extract_fenced_blocks(text: str) -> dict[str, list[str]]:
    """All fenced code blocks in ``text``, keyed by (lowercased) info string."""
    blocks: dict[str, list[str]] = {}
    for match in _FENCE_RE.finditer(text):
        blocks.setdefault(match.group(1).lower(), []).append(match.group(2))
    return blocks


def render_field_docs(model: Type[BaseModel]) -> str:
    """Render one model's field documentation from its pydantic metadata.

    Walks ``model_fields`` so the prompt's schema documentation is generated
    from the same ``Field(description=...)`` strings the validator enforces —
    the two cannot drift.
    """
    lines = [f"{model.__name__}:"]
    for name, field in model.model_fields.items():
        requirement = "required" if field.is_required() else "optional"
        description = field.description or "(no description)"
        lines.append(f"- {name} ({requirement}): {description}")
    return "\n".join(lines)


# =============================================================================
# The author form
# =============================================================================


class PersonaSketch(BaseModel):
    """One persona sketch from the author form (the model fleshes it out)."""

    name: str = Field(
        description="The persona's given name, e.g. 'Amara'. Used to derive "
        "the persona_id ('<name>_<language>_v1')."
    )
    sketch: str = Field(
        description="Freeform 2-6 sentence sketch: age, where they live, "
        "occupation, how they speak (register/formality axis, dialect, "
        "code-switching, English tolerance), typical calling environment, "
        "anything distinctive about their LANGUAGE. Do not prescribe an "
        "attitude/affect (friendly, impatient, chatty) — attitude comes from "
        "each task, not the persona."
    )


def _stringify_freeform_value(value: Any) -> str:
    """Render a non-string freeform value (a dict/list the model returned
    instead of prose) into readable, information-preserving text.

    The autoform model sometimes returns ``preferences``/``locale_notes`` as a
    structured JSON object/array even though the schema asks for a freeform
    string. Rather than hard-fail (losing the content) or silently drop it,
    flatten it into a human-readable bullet/line form so every key and value
    survives into the drafted pack.
    """

    def render(node: Any, indent: int = 0) -> list[str]:
        pad = "  " * indent
        if isinstance(node, dict):
            lines: list[str] = []
            for key, val in node.items():
                label = str(key).replace("_", " ")
                if isinstance(val, (dict, list)):
                    lines.append(f"{pad}- {label}:")
                    lines.extend(render(val, indent + 1))
                else:
                    lines.append(f"{pad}- {label}: {_scalar_to_str(val)}")
            return lines
        if isinstance(node, list):
            lines = []
            for item in node:
                if isinstance(item, (dict, list)):
                    lines.extend(render(item, indent))
                else:
                    lines.append(f"{pad}- {_scalar_to_str(item)}")
            return lines
        return [f"{pad}{_scalar_to_str(node)}"]

    return "\n".join(render(value)).strip()


def _scalar_to_str(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return ""
    return str(value)


class AuthorForm(BaseModel):
    """The human-authored input to ``tau2 factory draft``.

    Authored as a YAML mapping (optionally inside a ```yaml fenced block of a
    markdown file) — see ``docs/multilingual/factory_form_example.md``.
    """

    language: str = Field(
        description="ISO 639-1 language code, e.g. 'sw'. Must match --lang."
    )
    display_name: str = Field(
        description="Human-readable language name, e.g. 'Swahili'."
    )
    script: str = Field(
        description="ISO 15924 script code for the matrix language, e.g. "
        "'latn', 'deva'. Must have a registered Unicode range in "
        "tau2.multilingual.invariants.SCRIPT_RANGES (extending it is a "
        "human/engineer task)."
    )
    locale_notes: str = Field(
        default="",
        description="Freeform notes on region/dialect/locale conventions the "
        "drafted personas and guidelines should respect (accent divides, "
        "spelling conventions, how numbers/letters are read aloud, ...).",
    )
    personas: list[PersonaSketch] = Field(
        min_length=2,
        max_length=4,
        description="2-4 persona sketches. Guidance from the Hindi "
        "build: span a register axis (high vs low code-switching, different "
        "sociolects) and any major accent divide.",
    )
    preferences: str = Field(
        default="",
        description="Freeform drafting preferences: backchannel density, "
        "out-of-turn speech style, acoustic environments to prefer, anything "
        "else the drafter should honor.",
    )
    backchannel_level: BackchannelLevel = Field(
        default=BackchannelLevel.MEDIUM,
        description="Backchannel density: low/medium/high.",
    )
    locale_beds: bool = Field(
        default=False,
        description="OPT-IN (default off): scaffold locale acoustic presets "
        "and generate locale background beds. The benchmark default is the "
        "shared English environments (stock indoor/outdoor beds, seed-keyed) "
        "for every language, so acoustic conditions are identical across "
        "locales; only set this for a pack that deliberately studies "
        "locale-specific acoustics.",
    )
    # Typed overrides for the deterministic experiment/pack fields
    # (``pack_assembly``). Leave unset for the derived/documented defaults —
    # the autoform generator never fills them; they are author/CLI decisions.
    domain: Optional[str] = Field(
        default=None,
        description="OVERRIDE (leave unset): experiment domain for the "
        "deterministic experiment block. Unset defers to the CLI --domain or "
        "the documented DEFAULT_MULTILINGUAL_DOMAIN.",
    )
    main_arm_name: Optional[str] = Field(
        default=None,
        description="OVERRIDE (leave unset): experiment arm name. Unset "
        "derives it from display_name's first token, lowercased "
        "('Mandarin Chinese' -> 'mandarin').",
    )
    smoke_task_stem: Optional[str] = Field(
        default=None,
        description="OVERRIDE (leave unset): the task id stem `tau2 "
        "run-preset <preset> --stage smoke` runs. Unset defers to the CLI "
        "flag or the domain profile's default stem.",
    )
    default_out_of_turn_events_per_minute: Optional[float] = Field(
        default=None,
        description="OVERRIDE (leave unset): out-of-turn speech firing rate. "
        "Unset uses the language-level default (1.0).",
    )

    @field_validator("locale_notes", "preferences", mode="before")
    @classmethod
    def _absorb_structured_freeform(cls, v: object) -> object:
        # The quirk-absorbing seam between the (best-effort) autoform LLM JSON
        # and the strict validator: prompt hardening reduces how often the
        # model emits a structured value for these freeform-prose fields, and
        # this coercion guarantees a structured value never hard-fails
        # validation while preserving its content. Strings pass through
        # untouched; typed fields (personas, ...) are not selected here and
        # stay strictly validated.
        if isinstance(v, (dict, list)):
            return _stringify_freeform_value(v)
        return v


def parse_author_form(form_path: Path) -> AuthorForm:
    """Parse an author form file (YAML, or markdown with a ```yaml block).

    Raises:
        FactoryDraftError: With a readable, field-by-field message when the
            file is missing, not valid YAML, or fails form validation.
    """
    form_path = Path(form_path)
    if not form_path.exists():
        raise FactoryDraftError(f"form file does not exist: {form_path}")
    text = form_path.read_text()

    yaml_blocks = extract_fenced_blocks(text).get("yaml", [])
    source = yaml_blocks[0] if yaml_blocks else text
    try:
        data = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise FactoryDraftError(
            f"form file {form_path} is not valid YAML "
            f"(markdown forms must carry the form in a ```yaml block): {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise FactoryDraftError(
            f"form file {form_path} must contain a YAML mapping with the "
            f"AuthorForm fields ({sorted(AuthorForm.model_fields)})"
        )
    try:
        return AuthorForm.model_validate(data)
    except ValidationError as exc:
        lines = [f"form file {form_path} failed validation:"]
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(form)"
            lines.append(f"  - {location}: {error['msg']}")
        lines.append(
            "See docs/multilingual/factory_form_example.md for a complete example."
        )
        raise FactoryDraftError("\n".join(lines)) from exc


# =============================================================================
# Autoform: LLM-generate the author form from a language identity
# =============================================================================

AUTOFORM_SYSTEM_PROMPT = """\
You are the Tau-Voice Language Factory author-form generator. From a target \
language identity you produce a single JSON author form describing the personas \
and locale conventions a realistic voice language pack should respect. You \
respond with exactly one JSON object and nothing else."""


def build_autoform_prompt(
    language: str, display_name: str, script: str, hints: str = ""
) -> str:
    """The single autoform prompt. Schema rendered from the pydantic models so
    it can't drift from what ``AuthorForm.model_validate`` will enforce."""
    schema = render_field_docs(AuthorForm) + "\n\n" + render_field_docs(PersonaSketch)
    hint_block = (
        f"\n\n## Additional author hints (honor these)\n\n{hints.strip()}\n"
        if hints.strip()
        else ""
    )
    return f"""\
Produce a Tau-Voice author form for {display_name} ('{language}', script \
'{script}'). The author form is the human-reviewable seed from which a full \
voice language pack is drafted.

Respond with ONE JSON object and nothing else, matching this schema:

{schema}

Requirements:
- Exactly 2 personas. One must read as male and one as female — the pack
  invariant requires exactly one of each, and each sketch must make the
  intended gender unambiguous.
- The two personas must SPAN A REGISTER AXIS for {display_name}: e.g. one
  higher-register / formal / low-code-switching speaker and one casual /
  high-code-switching speaker; where the language has a major regional or
  accent divide, span that too.
- locale_notes must capture region/dialect conventions and how numbers,
  emails, and letters / booking codes are read aloud in {display_name}.
- Set language='{language}', display_name='{display_name}', and
  script='{script}' exactly.
- locale_notes and preferences MUST each be a single plain JSON string (prose,
  newlines allowed) — NOT a nested object or array. Put any structure inline as
  readable sentences or a dash list inside the string.
- Do NOT include voice ids or any field outside the schema.
- Do NOT set locale_beds — locale acoustic beds are an engineer opt-in
  decision (default off: every pack runs on the shared benchmark
  environments).{hint_block}

Respond with ONLY the JSON object."""


def generate_author_form(
    language: str,
    display_name: str,
    script: str,
    *,
    hints: str = "",
    llm: Optional[FactoryLLM] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> AuthorForm:
    """LLM-generate a validated ``AuthorForm`` from a language identity.

    ``language``/``display_name``/``script`` are AUTHORITATIVE — they are
    forced onto the result so the form can never disagree with the caller's
    --lang. Raises ``FactoryDraftError`` (field-by-field, mirroring
    ``parse_author_form``) when the model output fails ``AuthorForm``
    validation.
    """
    llm = llm if llm is not None else get_default_llm()
    model = model or DEFAULT_FACTORY_MODEL
    if reasoning_effort is None:
        reasoning_effort = DEFAULT_FACTORY_AUTOFORM_REASONING

    data = llm.json_call(
        model,
        AUTOFORM_SYSTEM_PROMPT,
        build_autoform_prompt(language, display_name, script, hints),
        call_name=AUTOFORM_CALL_NAME,
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(data, dict):
        raise FactoryDraftError(
            f"autoform LLM call '{AUTOFORM_CALL_NAME}' returned a "
            f"{type(data).__name__}, expected a JSON object with the AuthorForm "
            f"fields ({sorted(AuthorForm.model_fields)})"
        )
    # Identity fields are authoritative — never taken from the model.
    data["language"] = language
    data["display_name"] = display_name
    data["script"] = script
    # Locale beds are an engineer opt-in (edit the reviewed form to enable);
    # the autoform model can never flip it.
    data.pop("locale_beds", None)
    # (Structured locale_notes/preferences replies are absorbed by AuthorForm's
    # own before-validators — the quirk handling lives ON the boundary model.)
    try:
        return AuthorForm.model_validate(data)
    except ValidationError as exc:
        lines = ["autoform produced an invalid AuthorForm:"]
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(form)"
            lines.append(f"  - {location}: {error['msg']}")
        raise FactoryDraftError("\n".join(lines)) from exc

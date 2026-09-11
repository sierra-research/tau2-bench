# Copyright Sierra
"""LLM drafting of a language pack: the chained creative calls, deterministic
assembly, guardrail-gated targeted repair, and the ``run_draft`` entry point."""

from pathlib import Path
from typing import Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from tau2.config import DEFAULT_FACTORY_DRAFT_REASONING, DEFAULT_FACTORY_MODEL
from tau2.multilingual.delivery_catalog import DELIVERY_FACTOR_CATALOG
from tau2.multilingual.factory import pack_assembly, paths
from tau2.multilingual.factory.author_form import (
    AuthorForm,
    FactoryDraftError,
    extract_fenced_blocks,
    parse_author_form,
)
from tau2.multilingual.factory.draft_prompts import (
    DRAFT_ARTIFACTS_SYSTEM_PROMPT,
    DRAFT_PERSONAS_PROMPT_VERSION,
    DRAFT_PERSONAS_SYSTEM_PROMPT,
    DRAFT_TEXT_ARTIFACTS_SYSTEM_PROMPT,
    build_artifacts_prompt,
    build_artifacts_repair_prompt,
    build_personas_prompt,
    build_personas_repair_prompt,
    build_text_artifacts_prompt,
    build_text_artifacts_repair_prompt,
    personas_prompt_sha256,
)
from tau2.multilingual.factory.guardrails import (
    GuardrailReport,
    validate_pack_draft,
)
from tau2.multilingual.factory.llm import FactoryLLM, get_default_llm
from tau2.multilingual.factory.localization_drafting import (
    draft_localization,
    localization_prompt_sha256,
)
from tau2.multilingual.factory.state import (
    FactoryProject,
    FactoryStage,
    factory_file_path,
)
from tau2.multilingual.schema import LocalizationPackConfig, MultilingualPersonaConfig
from tau2.utils.utils import prompt_sha256

# LLM call names (these are the observable contract for tests and llm logs).
#
# Drafting is split along the cross-artifact COHERENCE SEAM into two chained
# creative calls (the deterministic fields are code-generated in
# ``pack_assembly`` and never asked of the model):
#  - Call A (``factory_draft_personas``): the linguistically-coupled persona /
#    voice content + translation_guidance, authored together;
#  - Call B (``factory_draft_artifacts``): the guidelines markdown, authored
#    GIVEN Call A's personas so the guidelines conventions stay consistent with
#    the persona phrases.
# Repair is TARGETED: a guardrail failure localized to the guidelines re-runs
# only Call B; anything touching persona/pack content re-runs Call A (and then
# Call B, which depends on it).
DRAFT_PERSONAS_CALL_NAME = "factory_draft_personas"
DRAFT_ARTIFACTS_CALL_NAME = "factory_draft_artifacts"
DRAFT_TEXT_ARTIFACTS_CALL_NAME = "factory_draft_text_artifacts"
REPAIR_PERSONAS_CALL_NAME = "factory_draft_personas_repair"
REPAIR_ARTIFACTS_CALL_NAME = "factory_draft_artifacts_repair"
REPAIR_TEXT_ARTIFACTS_CALL_NAME = "factory_draft_text_artifacts_repair"


def _resolve_draft_model(
    model: Optional[str], project_model: Optional[str] = None
) -> str:
    """Drafting model: explicit arg (CLI --model) > project pin > default."""
    return model or project_model or DEFAULT_FACTORY_MODEL


def _resolve_draft_reasoning(reasoning_effort: Optional[str]) -> Optional[str]:
    """Drafting reasoning effort: explicit arg (CLI --reasoning) > default."""
    return (
        reasoning_effort
        if reasoning_effort is not None
        else DEFAULT_FACTORY_DRAFT_REASONING
    )


# The guidelines draft filename inside the factory workspace. The draft pack's
# guidelines_voice_path is normalized to this so the workspace validates
# self-contained; the author's DECLARED final filename is recorded on the
# project and restored by finalize.
GUIDELINES_DRAFT_FILENAME = "guidelines_draft.md"

# The TEXT (half-duplex chat) guidelines draft filename inside the factory
# workspace. Parallel to GUIDELINES_DRAFT_FILENAME (voice); the draft pack's
# guidelines_text_path is normalized to this so the workspace validates
# self-contained, and finalize restores the declared final filename.
GUIDELINES_TEXT_DRAFT_FILENAME = "guidelines_text_draft.md"


# =============================================================================
# Model-output parsing and post-processing
# =============================================================================


class CreativePersonas(BaseModel):
    """Call A's parsed output: persona content + translation_guidance."""

    personas: list[MultilingualPersonaConfig] = Field(
        description="Validated authored personas (voice_id stripped)"
    )
    translation_guidance: Optional[str] = None
    agent_greeting: Optional[str] = None
    stripped_voice_ids: dict[str, str] = Field(default_factory=dict)


class GuidelinesArtifact(BaseModel):
    """Call B's (voice) or Call C's (text) parsed output: one guidelines
    markdown body."""

    markdown: str = ""


class PackDraft(BaseModel):
    """A parsed, post-processed drafting (or repair) response.

    The two creative calls + the deterministic fields are assembled into this:
    ``pack_yaml`` is the full normalized pack draft text and
    ``guidelines_markdown`` the guidelines file body.
    """

    pack_yaml: str = Field(description="The pack draft YAML text (normalized)")
    guidelines_markdown: str = Field(description="The guidelines markdown text")
    text_guidelines_markdown: str = Field(
        default="",
        description="The TEXT (typed chat) guidelines markdown text (Call C)",
    )
    declared_guidelines_filename: Optional[str] = Field(
        default=None,
        description="The PACK-RELATIVE guidelines path for the final pack "
        "(``guidelines/simulation_guidelines_voice_<lang>.md``, see "
        "factory.paths), recorded before guidelines_voice_path is normalized "
        "to the workspace draft name; finalize restores it.",
    )
    declared_text_guidelines_filename: Optional[str] = Field(
        default=None,
        description="The PACK-RELATIVE TEXT guidelines path for the final "
        "pack, recorded before guidelines_text_path is normalized to the "
        "workspace draft name; finalize restores it.",
    )
    stripped_voice_ids: dict[str, str] = Field(
        default_factory=dict,
        description="persona_id -> voice_id the model emitted despite the "
        "leave-unset instruction (stripped; voice selection is a human "
        "bottleneck).",
    )
    # The parsed creative outputs are retained so TARGETED repair can re-run
    # exactly one call and re-assemble without re-sending the others.
    creative_personas: CreativePersonas
    derived_artifacts: GuidelinesArtifact
    derived_text_artifacts: GuidelinesArtifact
    localization: Optional[LocalizationPackConfig] = Field(
        default=None,
        description="Call D's validated spoken-localization block (symbol "
        "readouts, date/number examples, domain glossary); carried through "
        "repair re-assembly unchanged (it is schema-validated at draft time).",
    )


def _parse_personas_response(
    data: Union[dict, list], *, call_name: str, allow_acoustic: bool = True
) -> CreativePersonas:
    """Parse Call A's JSON into personas + translation_guidance (voice_id stripped).

    ``allow_acoustic=False`` (a pack on the shared benchmark environments —
    no scaffolded presets) also strips any ``acoustic_preset_id`` the model
    emitted despite the prompt, so a stray reference can't dangle."""
    if not isinstance(data, dict):
        raise FactoryDraftError(
            f"LLM call '{call_name}' returned a {type(data).__name__}, expected a "
            "JSON object with keys 'personas'/'translation_guidance'"
        )
    personas = data.get("personas")
    if not isinstance(personas, list):
        raise FactoryDraftError(
            f"LLM call '{call_name}' must return a 'personas' LIST "
            f"(got {type(personas).__name__})"
        )
    stripped: dict[str, str] = {}
    cleaned: list[MultilingualPersonaConfig] = []
    for idx, persona in enumerate(personas):
        if not isinstance(persona, dict):
            raise FactoryDraftError(
                f"LLM call '{call_name}' returned a non-dict persona at index "
                f"{idx} (got {type(persona).__name__}); expected an object per "
                "persona — silently dropping it would yield too few personas"
            )
        persona = dict(persona)
        voice_id = persona.pop("voice_id", None)
        if voice_id:
            stripped[str(persona.get("persona_id", "?"))] = str(voice_id)
        if not allow_acoustic:
            persona.pop("acoustic_preset_id", None)
        try:
            cleaned.append(MultilingualPersonaConfig.model_validate(persona))
        except ValidationError as exc:
            raise FactoryDraftError(
                f"LLM call '{call_name}' returned an invalid persona at index "
                f"{idx}: {exc}"
            ) from exc
    return CreativePersonas(
        personas=cleaned,
        translation_guidance=data.get("translation_guidance"),
        agent_greeting=data.get("agent_greeting"),
        stripped_voice_ids=stripped,
    )


def _parse_artifacts_response(response: str, *, call_name: str) -> GuidelinesArtifact:
    """Parse Call B / Call C: a single markdown block (the guidelines body)."""
    blocks = extract_fenced_blocks(response)
    markdown_blocks = blocks.get("markdown", []) + blocks.get("md", [])
    if not markdown_blocks:
        raise FactoryDraftError(
            f"LLM call '{call_name}' returned no ```markdown block "
            f"(found blocks: {sorted(blocks) or 'none'})"
        )
    return GuidelinesArtifact(markdown=markdown_blocks[0] or "")


def _assemble_pack_yaml(
    form: AuthorForm,
    personas: CreativePersonas,
    *,
    localization: Optional[LocalizationPackConfig] = None,
    domain: Optional[str] = None,
    smoke_task_stem: Optional[str] = None,
) -> str:
    """Merge deterministic fields + Call A + Call B into one pack draft YAML.

    Deterministic fields are the base; creative content overlays them on the
    keys it owns. ``guidelines_voice_path`` is normalized to the workspace draft
    filename so the workspace validates self-contained.
    """
    data: dict = pack_assembly.build_deterministic_pack_fields(
        form,
        domain=domain,
        smoke_task_stem=smoke_task_stem,
    )
    # Creative persona content (keyed by persona_id, the loader's shape).
    persona_map: dict[str, dict] = {}
    for persona in personas.personas:
        persona_map[persona.persona_id] = persona.model_dump(
            mode="json", exclude_none=True
        )
    data["personas"] = persona_map

    # Now that persona locales are known, re-namespace the generated bed paths
    # from the bare language code to the derived ``language_COUNTRY`` dir (es ->
    # es_ES), so the pack's background_noise_files match the locale-bed
    # naming convention — no post-hoc rename needed.
    from tau2.multilingual.factory.paths import derive_locale_dir

    locale_dir = derive_locale_dir(
        form.language, [pd.get("locale") for pd in persona_map.values()]
    )
    if locale_dir == form.language:
        # No persona carried a country-parseable ISO 3166-2 locale, so the beds
        # fall back to the bare-language dir instead of the language_COUNTRY
        # convention (hi_IN, es_ES). Functional but inconsistent — surface it so
        # the author can add a persona locale (e.g. "ES-MD") and regenerate.
        logger.warning(
            f"No persona locale for '{form.language}' yields a country — acoustic "
            f"beds will use the bare '{form.language}/' dir, not the "
            f"'{form.language}_COUNTRY' convention. Add an ISO 3166-2 locale "
            "(e.g. 'ES-MD') to a persona for locale-namespaced beds."
        )
    if data.get("acoustic_presets"):
        pack_assembly.localize_acoustic_bed_paths(
            data["acoustic_presets"], form.language, locale_dir
        )
        # Keep personas on the generated locale beds: if the model parked one on
        # the shared-English office preset and left a locale bed unused, move it.
        pack_assembly.reassign_personas_to_locale_beds(
            persona_map, data["acoustic_presets"]
        )
    if personas.translation_guidance is not None:
        data["translation_guidance"] = personas.translation_guidance
    if personas.agent_greeting is not None:
        data["agent_greeting"] = personas.agent_greeting
    data["backchannel_level"] = form.backchannel_level.value
    data["guidelines_voice_path"] = GUIDELINES_DRAFT_FILENAME
    data["guidelines_text_path"] = GUIDELINES_TEXT_DRAFT_FILENAME
    if localization is not None:
        data["localization"] = localization.model_dump(mode="json", exclude_none=True)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + (
        "\n" + delivery_scaffold_comment()
    )


def _catalog_menu(catalog: dict) -> str:
    """A scaffold comment's factor-id menu, one ``#``-prefixed line per
    catalog category."""
    by_category: dict[str, list[str]] = {}
    for fid, f in catalog.items():
        by_category.setdefault(f.category, []).append(fid)
    return "\n".join(
        f"#     {cat}: {', '.join(sorted(ids))}"
        for cat, ids in sorted(by_category.items())
    )


def delivery_scaffold_comment() -> str:
    """A commented `delivery:` scaffold appended to every draft pack.

    Deliberately a COMMENT, not generated content: delivery judge factors must
    be authored by a human SELECTING from the closed catalog
    (``tau2.multilingual.delivery_catalog``) and writing the per-language
    listening instruction — the factory never invents factors. The scaffold
    lists the catalog menu so the author can uncomment + fill it. Unlike
    nativeness, a pack with no ``delivery:`` block is still judged
    language-specifically: the delivery judge falls back to the fixed
    native-listener prompt for the language.
    """
    return (
        "# --- delivery scoring (audio-layer; AUTHOR THIS once annotation\n"
        "# confirms this language's speech-layer defects) ----------------------\n"
        "# Select factor_id(s) from the closed catalog below (you may NOT invent\n"
        "# new ids — the loader rejects unknown ones) and write what the audio\n"
        "# judge should listen for. Leaving this commented is safe: the judge\n"
        "# then uses the fixed native-listener fallback for this language.\n"
        "# Catalog factor ids by category:\n"
        f"{_catalog_menu(DELIVERY_FACTOR_CATALOG)}\n"
        "#\n"
        "# delivery:\n"
        "#   judge_factors:\n"
        "#     - factor_id: tone_meaning_flip\n"
        "#       listen_for: <what to listen for in the delivered audio>\n"
    )


def _assemble_draft(
    form: AuthorForm,
    personas: CreativePersonas,
    artifacts: GuidelinesArtifact,
    text_artifacts: GuidelinesArtifact,
    *,
    localization: Optional[LocalizationPackConfig] = None,
    domain: Optional[str] = None,
    smoke_task_stem: Optional[str] = None,
) -> PackDraft:
    pack_yaml = _assemble_pack_yaml(
        form,
        personas,
        localization=localization,
        domain=domain,
        smoke_task_stem=smoke_task_stem,
    )
    return PackDraft(
        pack_yaml=pack_yaml,
        guidelines_markdown=artifacts.markdown,
        text_guidelines_markdown=text_artifacts.markdown,
        declared_guidelines_filename=paths.default_voice_guidelines_filename(
            form.language
        ),
        declared_text_guidelines_filename=paths.default_text_guidelines_filename(
            form.language
        ),
        stripped_voice_ids=dict(personas.stripped_voice_ids),
        creative_personas=personas,
        derived_artifacts=artifacts,
        derived_text_artifacts=text_artifacts,
        localization=localization,
    )


def draft_pack(
    form: AuthorForm,
    llm: Optional[FactoryLLM] = None,
    *,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    domain: Optional[str] = None,
    smoke_task_stem: Optional[str] = None,
) -> PackDraft:
    """The chained creative calls + deterministic assembly: form in, draft out.

    Call A authors the personas + translation_guidance; Call B derives the
    guidelines GIVEN Call A's personas; Call C the TEXT guidelines; Call D the
    spoken-localization block GIVEN Call A's translation guidance + persona
    registers; code generates the deterministic fields (including
    ``backchannel_level`` from the form) and assembles all into the pack draft.
    """
    llm = llm if llm is not None else get_default_llm()
    model = _resolve_draft_model(model)
    reasoning_effort = _resolve_draft_reasoning(reasoning_effort)

    preset_ids = pack_assembly.acoustic_preset_ids(form)
    personas_data = llm.json_call(
        model,
        DRAFT_PERSONAS_SYSTEM_PROMPT,
        build_personas_prompt(form, preset_ids),
        call_name=DRAFT_PERSONAS_CALL_NAME,
        reasoning_effort=reasoning_effort,
    )
    personas = _parse_personas_response(
        personas_data,
        call_name=DRAFT_PERSONAS_CALL_NAME,
        allow_acoustic=bool(preset_ids),
    )
    artifacts_response = llm.chat(
        model,
        DRAFT_ARTIFACTS_SYSTEM_PROMPT,
        build_artifacts_prompt(form, personas.personas),
        call_name=DRAFT_ARTIFACTS_CALL_NAME,
        reasoning_effort=reasoning_effort,
    )
    artifacts = _parse_artifacts_response(
        artifacts_response, call_name=DRAFT_ARTIFACTS_CALL_NAME
    )
    # Call C: derive the TEXT (typed chat) guidelines GIVEN Call A's personas,
    # in parallel to Call B's voice guidelines.
    text_artifacts_response = llm.chat(
        model,
        DRAFT_TEXT_ARTIFACTS_SYSTEM_PROMPT,
        build_text_artifacts_prompt(form, personas.personas),
        call_name=DRAFT_TEXT_ARTIFACTS_CALL_NAME,
        reasoning_effort=reasoning_effort,
    )
    text_artifacts = _parse_artifacts_response(
        text_artifacts_response, call_name=DRAFT_TEXT_ARTIFACTS_CALL_NAME
    )
    # Call D: the spoken-localization block (native symbol readouts, the
    # date/numeral consistency examples, the speech conventions + worked
    # spoken-value examples, the guideline example palettes, the domain-term
    # glossary), grounded on Call A's translation guidance + persona
    # registers + backchannel phrases and on Call B's voice guidelines (so
    # the drafted conventions agree with the guidelines' native content).
    # Validated against the closed catalogs inside draft_localization itself.
    localization = draft_localization(
        form.language,
        form.display_name,
        form.script,
        # Same precedence as the deterministic experiments entry — without
        # this the glossary is drafted against the DEFAULT domain's catalog
        # and the draft dead-ends on the missing-glossary guardrail.
        domain=pack_assembly.resolve_domain(form, domain),
        translation_guidance=personas.translation_guidance,
        persona_summaries=[p.short_description for p in personas.personas],
        speech_guidelines=artifacts.markdown,
        backchannel_phrases=sorted(
            {
                str(phrase)
                for p in personas.personas
                for phrase in (p.backchannel_phrases or [])
            }
        ),
        llm=llm,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    return _assemble_draft(
        form,
        personas,
        artifacts,
        text_artifacts,
        localization=localization,
        domain=domain,
        smoke_task_stem=smoke_task_stem,
    )


# =============================================================================
# The draft stage entry point
# =============================================================================


def drafting_prompt_sha256() -> str:
    """sha256 over the fixed prompt material of draft Calls A-D.

    Covers Call A's persona material (system prompt + neutrality rule +
    version, via ``personas_prompt_sha256``), the Call B/C artifact system
    prompts, and Call D's localization material (system prompt + version, via
    ``localization_prompt_sha256``) — one recordable provenance hash for the
    whole drafting chain.
    """
    return prompt_sha256(
        personas_prompt_sha256(),
        DRAFT_ARTIFACTS_SYSTEM_PROMPT,
        DRAFT_TEXT_ARTIFACTS_SYSTEM_PROMPT,
        localization_prompt_sha256(),
    )


class DraftOutcome(BaseModel):
    """What ``run_draft`` did, for the CLI and tests."""

    language: str
    pack_draft_path: Path
    guidelines_draft_path: Path
    text_guidelines_draft_path: Optional[Path] = None
    pack_revision: int = Field(description="The project revision after the draft")
    stripped_voice_ids: dict[str, str] = Field(default_factory=dict)
    repair_rounds: int = Field(
        default=0, description="Number of auto-repair rounds that ran"
    )
    repair_attempted: bool = False
    report: GuardrailReport
    model: str = Field(default="", description="The LLM model that ran draft Calls A-D")
    personas_prompt_version: str = DRAFT_PERSONAS_PROMPT_VERSION
    prompt_sha256: str = Field(
        default_factory=drafting_prompt_sha256,
        description="sha256 over the fixed prompt material of draft Calls "
        "A-D (see drafting_prompt_sha256)",
    )


def _write_draft(project: FactoryProject, draft: PackDraft) -> tuple[Path, Path, Path]:
    pack_path = project.pack_draft_path
    guidelines_path = factory_file_path(project.language, GUIDELINES_DRAFT_FILENAME)
    text_guidelines_path = factory_file_path(
        project.language, GUIDELINES_TEXT_DRAFT_FILENAME
    )
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    # Guidelines files first, pack.yaml last (matches finalize's
    # discoverability ordering; the draft validator resolves both paths).
    guidelines_path.write_text(draft.guidelines_markdown)
    text_guidelines_path.write_text(draft.text_guidelines_markdown)
    pack_path.write_text(draft.pack_yaml)
    return pack_path, guidelines_path, text_guidelines_path


# Substrings that localize a guardrail problem to Call C (the derived TEXT
# guidelines) vs Call B (the derived VOICE guidelines). Guidelines-BODY
# guardrails phrase their messages around the draft file NAME (path.name), so
# the two draft filenames disambiguate voice from text. We deliberately match
# only the FILENAMES, never the path FIELD names: a ``guidelines_voice_path`` /
# ``guidelines_text_path`` schema error is a pack-field (Call A / assembly)
# problem, and matching the field name would misroute it to a guidelines-only
# repair that cannot fix it. Everything matching NEITHER filename (schema/pack
# problems included) re-runs Call A. ``guidelines_draft.md`` is not a substring
# of ``guidelines_text_draft.md``, so the two never collide.
_TEXT_ARTIFACT_MARKERS = (GUIDELINES_TEXT_DRAFT_FILENAME,)
_VOICE_ARTIFACT_MARKERS = (GUIDELINES_DRAFT_FILENAME,)


def _problem_is_text_artifact(problem: str) -> bool:
    p = problem.lower()
    return any(marker.lower() in p for marker in _TEXT_ARTIFACT_MARKERS)


def _problem_is_voice_artifact(problem: str) -> bool:
    p = problem.lower()
    if _problem_is_text_artifact(problem):
        return False
    return any(marker.lower() in p for marker in _VOICE_ARTIFACT_MARKERS)


def _repair_targets(problems: list[str]) -> tuple[bool, bool, bool]:
    """Which calls a set of guardrail problems requires re-running.

    Returns ``(rerun_personas, rerun_voice_artifacts, rerun_text_artifacts)``.
    Artifact-only problems re-run just the affected guidelines call (Call B for
    voice, Call C for text); any persona/pack problem re-runs Call A and then
    re-derives BOTH guidelines (they depend on Call A's personas).
    """
    rerun_voice = False
    rerun_text = False
    rerun_personas = False
    for problem in problems:
        if _problem_is_text_artifact(problem):
            rerun_text = True
        elif _problem_is_voice_artifact(problem):
            rerun_voice = True
        else:
            rerun_personas = True
    if rerun_personas:
        # A persona/pack problem is present -> re-run A; re-derive B and C.
        return True, True, True
    return False, rerun_voice, rerun_text


def run_draft(
    lang: str,
    form_path: Path,
    llm: Optional[FactoryLLM] = None,
    *,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_repair_rounds: int = 1,
    domain: Optional[str] = None,
    smoke_task_stem: Optional[str] = None,
) -> DraftOutcome:
    """Draft a pack from a form into the factory workspace, gated by guardrails.

    Two chained creative calls (personas, then the derived artifacts) plus the
    deterministic code-generated fields, then up to ``max_repair_rounds``
    TARGETED auto-repair rounds (default 1): a guardrail failure localized to
    the guidelines / decision prompt re-runs only Call B; a persona/pack
    problem re-runs Call A and then re-derives Call B. Stops as soon as the
    draft is guardrail-clean or the budget is exhausted. Any remaining problems
    are surfaced for a direct edit of ``pack_draft.yaml`` plus ``tau2 factory
    validate --draft <lang>``.
    """
    llm = llm if llm is not None else get_default_llm()
    form = parse_author_form(form_path)
    if form.language != lang:
        raise FactoryDraftError(
            f"--lang is '{lang}' but the form declares language "
            f"'{form.language}' — refusing to draft a mismatched pack"
        )

    project = FactoryProject.load_or_create(lang, script=form.script)
    # A re-draft may correct the script or domain: ``load_or_create`` ignores
    # its defaults when the project already exists, so refresh explicitly —
    # the form (plus CLI --domain) is the source of truth, not the first draft.
    project.script = form.script
    project.domain = pack_assembly.resolve_domain(form, domain)
    model = _resolve_draft_model(model, project.llm_model)
    reasoning_effort = _resolve_draft_reasoning(reasoning_effort)

    draft = draft_pack(
        form,
        llm,
        model=model,
        reasoning_effort=reasoning_effort,
        domain=domain,
        smoke_task_stem=smoke_task_stem,
    )
    pack_path, guidelines_path, _text_guidelines_path = _write_draft(project, draft)
    report = validate_pack_draft(project.project_dir)
    stripped = dict(draft.stripped_voice_ids)
    declared = draft.declared_guidelines_filename
    declared_text = draft.declared_text_guidelines_filename
    repair_rounds = 0

    while not report.ok and repair_rounds < max_repair_rounds:
        repair_rounds += 1
        rerun_personas, rerun_artifacts, rerun_text_artifacts = _repair_targets(
            report.problems
        )
        logger.warning(
            f"Draft for '{lang}' failed {len(report.problems)} guardrail(s); "
            f"auto-repair round {repair_rounds}/{max_repair_rounds} "
            f"(personas={rerun_personas}, voice_artifacts={rerun_artifacts}, "
            f"text_artifacts={rerun_text_artifacts})"
        )
        personas = draft.creative_personas
        artifacts = draft.derived_artifacts
        text_artifacts = draft.derived_text_artifacts
        if rerun_personas:
            personas_data = llm.json_call(
                model,
                DRAFT_PERSONAS_SYSTEM_PROMPT,
                build_personas_repair_prompt(
                    form,
                    yaml.safe_dump(
                        {
                            "personas": [
                                p.model_dump(mode="json", exclude_none=True)
                                for p in personas.personas
                            ]
                        },
                        sort_keys=False,
                        allow_unicode=True,
                    ),
                    report.problems,
                ),
                call_name=REPAIR_PERSONAS_CALL_NAME,
                reasoning_effort=reasoning_effort,
            )
            repaired = _parse_personas_response(
                personas_data,
                call_name=REPAIR_PERSONAS_CALL_NAME,
                allow_acoustic=form.locale_beds,
            )
            # The persona repair prompt sends only a `personas` fragment, so the
            # model may omit `translation_guidance` / `agent_greeting`. Carry the
            # prior values forward when omitted (None) so assembly does not drop
            # previously-good content after a persona-only guardrail fix.
            if repaired.translation_guidance is None:
                repaired.translation_guidance = personas.translation_guidance
            if repaired.agent_greeting is None:
                repaired.agent_greeting = personas.agent_greeting
            stripped.update(repaired.stripped_voice_ids)
            personas = repaired

        if rerun_artifacts:
            response = llm.chat(
                model,
                DRAFT_ARTIFACTS_SYSTEM_PROMPT,
                build_artifacts_repair_prompt(
                    form,
                    personas.personas,
                    artifacts.markdown,
                    report.problems,
                ),
                call_name=REPAIR_ARTIFACTS_CALL_NAME,
                reasoning_effort=reasoning_effort,
            )
            new_artifacts = _parse_artifacts_response(
                response, call_name=REPAIR_ARTIFACTS_CALL_NAME
            )
            # Keep prior guidelines if a repair returns a blank markdown block.
            if not new_artifacts.markdown.strip():
                new_artifacts.markdown = artifacts.markdown
            artifacts = new_artifacts

        if rerun_text_artifacts:
            text_response = llm.chat(
                model,
                DRAFT_TEXT_ARTIFACTS_SYSTEM_PROMPT,
                build_text_artifacts_repair_prompt(
                    form,
                    personas.personas,
                    text_artifacts.markdown,
                    report.problems,
                ),
                call_name=REPAIR_TEXT_ARTIFACTS_CALL_NAME,
                reasoning_effort=reasoning_effort,
            )
            new_text_artifacts = _parse_artifacts_response(
                text_response, call_name=REPAIR_TEXT_ARTIFACTS_CALL_NAME
            )
            # Keep prior text guidelines if a repair returns a blank block.
            if not new_text_artifacts.markdown.strip():
                new_text_artifacts.markdown = text_artifacts.markdown
            text_artifacts = new_text_artifacts

        draft = _assemble_draft(
            form,
            personas,
            artifacts,
            text_artifacts,
            localization=draft.localization,
            domain=domain,
            smoke_task_stem=smoke_task_stem,
        )
        draft.stripped_voice_ids = dict(stripped)
        declared = draft.declared_guidelines_filename
        declared_text = draft.declared_text_guidelines_filename
        pack_path, guidelines_path, _text_guidelines_path = _write_draft(project, draft)
        report = validate_pack_draft(project.project_dir)

    project.pack_revision += 1
    project.guidelines_final_filename = (
        declared or paths.default_voice_guidelines_filename(lang)
    )
    project.guidelines_text_final_filename = (
        declared_text or paths.default_text_guidelines_filename(lang)
    )
    project.mark_stage(FactoryStage.DRAFT, success=report.ok)

    if stripped:
        logger.warning(
            f"Stripped model-emitted voice_id(s) (human bottleneck — pin "
            f"after codec audition): {stripped}"
        )
    return DraftOutcome(
        language=lang,
        pack_draft_path=pack_path,
        guidelines_draft_path=guidelines_path,
        text_guidelines_draft_path=_text_guidelines_path,
        pack_revision=project.pack_revision,
        stripped_voice_ids=stripped,
        repair_rounds=repair_rounds,
        repair_attempted=repair_rounds > 0,
        report=report,
        model=model,
    )

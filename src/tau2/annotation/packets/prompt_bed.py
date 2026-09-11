# Copyright Sierra
"""The prompt + bed review packet: one page per language, built for native
annotators to review (a) every LANGUAGE-BEARING prompt in the benchmark —
each rendered with the language's real pack data — (b) the pack personas'
pinned voices as audio samples, and (c) every locale's background-noise beds.

Only per-language content is included; global, language-independent prompt
templates are deliberately absent (noise to a native-speaker reviewer). The
prompt sections group into four categories:

- Runtime — user simulator: the voice user-sim system prompt,
  the rendered backchannel decision prompt, the
  pack's spoken inserts (greeting, backchannel/out-of-turn phrases), and —
  for identity-swapped languages — sample rows of the caller identity map.
- Runtime — agent: the audio-native and text agent system prompts as real
  runs build them (domain policy + the pack's agent_language_clause).
- Judges & evaluators: nativeness judge user prompts (in runtime request batches),
  delivery judge user prompt (pack delivery rubric), communicate /
  NL-assertions judges with their rendered language addenda.
- Factory build-time: translator/fixer/verifier systems (with the pack's
  translation_guidance), localization drafting, autoform, and the
  nuance-candidates prompt.

Rendered sections go through the REAL runtime code paths — the same
``VoiceRunConfig`` → ``build_voice_user`` / ``build_agent`` sequences the
orchestrator builders run, and the judges' own render functions — never a
re-implementation of prompt assembly (a hand-assembled render once shipped
truncated wrong content). Runtime-only inputs (transcripts, audio) appear
as visible bracketed stand-ins captioned with what fills them.
Construction is offline — no LLM calls at build time — with ONE opt-out
online step: the persona voice samples render through ElevenLabs TTS (the
``factory voice-samples`` core) so reviewers hear each pinned voice; pass
``--no-voice-samples`` to skip it and build fully offline.

The page follows the packet conventions: localStorage state namespaced by the
content-derived batch id, per-rater storage, and a browser CSV export whose
headers are generated from ``PromptBedRow`` (the ONE manifest-less ingest
path; see ``packets.forms``). Bed audio is copied next to the HTML and played
via relative paths, like sim-packet audio.
"""

import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.annotation.artifacts import (
    ArtifactManifest,
    derive_batch_id,
    git_sha,
    write_json_artifact,
)
from tau2.annotation.models import PromptBedItemKind, PromptBedRow
from tau2.annotation.packets.builder import DEFAULT_OUTPUT_ROOT, PACKET_MANIFEST_NAME
from tau2.annotation.packets.forms import PROMPT_BED_KIND
from tau2.annotation.packets.page import _config_json, _env, _static

PROMPT_BED_LABEL = "Prompt & Background-Bed Review"

# The marker the packet substitutes into the backchannel decision prompt's
# ``{conversation_history}`` slot (filled per tick at runtime).
CONVERSATION_HISTORY_PLACEHOLDER = "[conversation so far]"

PromptCategory = Literal["runtime_user", "runtime_agent", "judges", "factory"]

# Category key -> (page heading, one-paragraph blurb). Order is page order.
PROMPT_CATEGORIES: dict[str, tuple[str, str]] = {
    "runtime_user": (
        "Runtime — user simulator (the caller)",
        "The prompts that drive the SIMULATED CALLER during a live run. These "
        "matter most for your language: they contain the localized guidelines, "
        "persona, scenario, and spoken phrases.",
    ),
    "runtime_agent": (
        "Runtime — agent (the service rep under test)",
        "The system prompts the AGENT being benchmarked runs with, ending in "
        "the pack-authored language clause telling it how to respond in your "
        "language.",
    ),
    "judges": (
        "Judges & evaluators",
        "The scoring prompts whose rubric text comes from your language pack: "
        "nativeness (per nuance), audio delivery, and the task-success judges "
        "with their rendered language addenda.",
    ),
    "factory": (
        "Factory build-time (pack authoring)",
        "The prompts the language FACTORY used to author this pack: "
        "translation, localization drafting, the author form, and nuance "
        "candidates.",
    ),
}

# Visible stand-ins for inputs that only exist at runtime. Rendered prompts
# use these where a template slot is filled per call/tick/run.
RUNTIME_TRANSCRIPT_PLACEHOLDER = "[agent speech transcript — filled per run]"
RUNTIME_AGENT_CONTEXT_PLACEHOLDER = "[agent role + gender — filled per run]"
RUNTIME_EXPECTED_TEXT_PLACEHOLDER = (
    "[expected_synthesis_text — the agent utterance's gold transcript, "
    "filled per audio clip]"
)

# How many identity-map rows the identity-swap section samples (deterministic:
# the first N callers by English name; the full map stays in the repo file).
IDENTITY_EXAMPLE_COUNT = 8

BED_TYPE_LABELS: dict[str, str] = {
    "outdoor": "outdoor street",
    "indoor_tv": "indoor TV / kitchen",
    "shared_office": "office (shared, all languages)",
}


class PromptBedPacketOptions(BaseModel):
    """Everything one ``tau2 annotate prompt-bed-packet`` invocation needs."""

    language: Annotated[
        str, Field(description="ISO 639-1 code of a registered language pack.")
    ]
    domain: Annotated[str, Field(description="Domain for the task scenario.")] = (
        "airline"
    )
    task_index: Annotated[
        int,
        Field(
            ge=0,
            description="Index into the language's main run-arm task set "
            "(the identity variant when one exists, mirroring run presets).",
        ),
    ] = 0
    task_set: Annotated[
        Optional[str],
        Field(
            description="Override the task set to render instead of the "
            "language's main run-arm set — the pre-run review gate for "
            "variant sets (e.g. retail_hi_identity_native renders the "
            "native-script DB clause and the name spell-out payload). Must "
            "be a registered set of the packet's domain."
        ),
    ] = None
    out_dir: Annotated[
        Optional[Path],
        Field(
            description="Output ROOT; the packet lands at <out>/<language>/ "
            "(default data/annotations/prompt_bed_review)."
        ),
    ] = None
    voice_samples: Annotated[
        bool,
        Field(
            description="Render each pack persona's pinned voice into an "
            "audio sample (ElevenLabs TTS — the ONE online step). False "
            "builds a fully offline packet with no voices section."
        ),
    ] = True

    @property
    def packet_key(self) -> str:
        """What names the packet: the language, or the overridden task set —
        so a variant packet never collides with the language's main one."""
        return self.task_set or self.language

    @property
    def packet_dir(self) -> Path:
        root = self.out_dir or (DEFAULT_OUTPUT_ROOT / "prompt_bed_review")
        return Path(root) / self.packet_key


class PromptSectionVM(BaseModel):
    """One rendered prompt section of the page (exact text, no truncation)."""

    item_id: Annotated[str, Field(description="Stable section id (a CSV key).")]
    category: Annotated[
        PromptCategory,
        Field(description="Page group this section renders under."),
    ]
    title: Annotated[str, Field(description="Section heading.")]
    caption: Annotated[
        str, Field(description="One-paragraph English note: what this controls.")
    ]
    text: Annotated[str, Field(description="The EXACT rendered text.")]


class VoiceSampleVM(BaseModel):
    """One persona voice-sample row (MP3 rendered into the packet)."""

    item_id: Annotated[str, Field(description="Stable voice id (a CSV key).")]
    persona_id: Annotated[str, Field(description="Pack persona id.")]
    gender: Annotated[str, Field(description="Persona gender tag.")]
    description: Annotated[
        str, Field(description="The persona's short description (who speaks).")
    ]
    rel_href: Annotated[
        str, Field(description="Packet-relative audio path (the <audio> src).")
    ]


class BedFileVM(BaseModel):
    """One background bed row of the page (audio copied into the packet)."""

    item_id: Annotated[str, Field(description="Stable bed id (a CSV key).")]
    locale: Annotated[
        str, Field(description="Locale dir name, or 'shared' for the office bed.")
    ]
    bed_type: Annotated[str, Field(description="outdoor / indoor_tv / shared_office.")]
    label: Annotated[str, Field(description="Readable bed label.")]
    src: Annotated[Path, Field(description="Source wav in the repo data tree.")]
    rel_href: Annotated[
        str, Field(description="Packet-relative audio path (the <audio> src).")
    ]


# ---------------------------------------------------------------------------
# Prompt sections (through the REAL runtime build path)
# ---------------------------------------------------------------------------


def _load_task(
    domain: str, language: str, task_index: int, task_set: Optional[str] = None
):
    """The task a real matrix run for this language would run at this index.

    ``task_set`` overrides the matrix resolver for variant sets (the
    native-script identity sets, e.g. ``retail_hi_identity_native``) so the
    review gate can render exactly what such a run would build. The override
    must belong to the packet's domain — a cross-domain set would render the
    wrong policy under the right heading.
    """
    from tau2.multilingual.run_presets import matrix_task_set_name
    from tau2.registry import registry

    if task_set is not None and not task_set.startswith(f"{domain}_"):
        raise ValueError(
            f"task set override '{task_set}' does not belong to domain "
            f"'{domain}' (expected a '{domain}_*' set)"
        )
    task_set_name = task_set or matrix_task_set_name(domain, language)
    tasks = registry.get_tasks_loader(task_set_name)()
    if task_index >= len(tasks):
        raise ValueError(
            f"task index {task_index} out of range: task set "
            f"'{task_set_name}' has {len(tasks)} tasks"
        )
    return task_set_name, tasks[task_index]


def _build_runtime_user(
    *,
    domain: str,
    language: str,
    task_set_name: str,
    task,
    environment,
):
    """Construct the voice user simulator EXACTLY as the runner does.

    Mirrors ``build_voice_orchestrator``'s user-side sequence with the run
    presets' defaults (seed 42, regular speech complexity, bare-language-code
    persona rotation gender-pinned by the caller-gender sidecar).
    """
    from tau2.config import (
        DEFAULT_MATRIX_SEED,
        DEFAULT_MATRIX_SPEECH_COMPLEXITY,
    )
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.multilingual.registry import resolve_run_language
    from tau2.runner.build import build_voice_user, user_prompt_task

    config = VoiceRunConfig(
        domain=domain,
        task_set_name=task_set_name,
        user_persona_id=language,
        audio_native_config=AudioNativeConfig(),
        speech_complexity=DEFAULT_MATRIX_SPEECH_COMPLEXITY,
        seed=DEFAULT_MATRIX_SEED,
    )
    user_language = resolve_run_language(language)
    prompt_task = user_prompt_task(config, task, user_language)
    return build_voice_user(
        environment,
        prompt_task,
        config.audio_native_config,
        speech_complexity=config.speech_complexity,
        seed=config.seed or DEFAULT_MATRIX_SEED,
        persona_seed=config.seed,
        domain=domain,
        user_persona_id=config.user_persona_id,
    )


def _speech_phrases_text(user) -> str:
    """The persona's runtime spoken inserts (not LLM prompts, but per-language
    content a run injects into the call): backchannels + out-of-turn speech."""
    lines = ["Backchannel phrases (spoken while the agent talks):"]
    lines += [f"  - {p}" for p in user.backchannel_phrases]
    speech_env = getattr(user.voice_settings, "speech_environment", None)
    non_directed = list(getattr(speech_env, "non_directed_phrases", None) or [])
    lines.append("")
    lines.append("Out-of-turn (non-directed) phrases (spoken away from the call):")
    if non_directed:
        lines += [f"  - {p}" for p in non_directed]
    else:
        lines.append("  (none configured for this environment preset)")
    return "\n".join(lines)


def _composite(*parts: tuple[str, str]) -> str:
    """Join labeled prompt pieces into one reviewable block, labels visible."""
    return "\n\n".join(f"───── {label} ─────\n{text.strip()}" for label, text in parts)


def _pack_script(pack) -> Optional[str]:
    """The pack's script code, resolved from its personas (as the factory
    does for localization backfill); None when no persona declares one."""
    scripts = sorted(
        {p.script for p in pack.personas.values() if getattr(p, "script", None)}
    )
    return scripts[0] if scripts else None


def _native_identity_swap_section(language: str, domain: str) -> PromptSectionVM:
    """Sample rows of the NATIVE-SCRIPT identity map, each with its full
    spell-out payload — the review surface for the zh gloss catalog and the
    hi akshara spellouts before any native-DB run. Fail-loud when the map is
    missing: a native packet without it would review the wrong variant."""
    import json

    from tau2.multilingual.factory.entity_localization import (
        native_identity_map_path,
    )

    path = native_identity_map_path(language, domain)
    if not path.is_file():
        raise FileNotFoundError(
            f"native identity map not found: {path} — build it with "
            f"`tau2 factory localize-entities --lang {language} --domain "
            f"{domain} --native-script` before rendering a native packet"
        )
    identity_map: dict[str, dict] = json.loads(path.read_text())
    shown = sorted(identity_map.items())[:IDENTITY_EXAMPLE_COUNT]
    lines = [
        f"{len(identity_map)} callers carry native-script DB names for this "
        f"language; the first {len(shown)} (by original caller key), each "
        "with the exact spell-out payload the user simulator receives:",
    ]
    for caller_key, identity in shown:
        name = f"{identity['first_name']} {identity['last_name']}"
        romanized = (
            f"{identity['romanized_first_name']} {identity['romanized_last_name']}"
        )
        lines += [
            "",
            f"  {caller_key} → {name} (romanized: {romanized}) — "
            f"{identity['email']} ({identity['gender']})",
        ]
        lines += [f"      {line}" for line in identity["character_clarifications"]]
    lines += ["", f"Full map: {path.name} (in the language pack directory)"]
    return PromptSectionVM(
        item_id="native_identity_swap_examples",
        category="runtime_user",
        title="Native-script caller identities — names and spell-out payloads",
        caption="On this task-set variant the customer DB stores first/last "
        "names in the native script, and the caller spells their name from "
        "the payload shown under each row (zh: one gloss per character, "
        "never pinyin; hi: akshara-by-akshara with matra clarifications). "
        "Review both the native name forms and every payload line — a wrong "
        "gloss or matra here ships verbatim into every call.",
        text="\n".join(lines),
    )


def _identity_swap_section(language: str, domain: str) -> Optional[PromptSectionVM]:
    """Sample rows of the caller identity map (English caller → localized
    name/email/gender), or None for a language without one (the English
    baseline keeps canonical US callers)."""
    import json

    from tau2.multilingual.factory.entity_localization import identity_map_path

    path = identity_map_path(language, domain)
    if not path.is_file():
        return None
    identity_map: dict[str, dict] = json.loads(path.read_text())
    lines = [
        f"{len(identity_map)} callers are renamed for this language; the "
        f"first {min(IDENTITY_EXAMPLE_COUNT, len(identity_map))} (by original "
        "caller key):",
        "",
    ]
    # The map value schema is profile-driven (telecom: name+email; airline:
    # full identity incl. address) — render the fields every profile shares
    # and append the city when the profile localizes addresses.
    for caller_key, identity in sorted(identity_map.items())[:IDENTITY_EXAMPLE_COUNT]:
        name = f"{identity['first_name']} {identity['last_name']}"
        city = (identity.get("address") or {}).get("city")
        lines.append(
            f"  {caller_key} → {name} — {identity['email']} "
            f"({identity['gender']}{f', {city}' if city else ''})"
        )
    lines += ["", f"Full map: {path.name} (in the language pack directory)"]
    return PromptSectionVM(
        item_id="identity_swap_examples",
        category="runtime_user",
        title="Caller identity swap — example rows",
        caption="Each task's caller is deterministically renamed from the "
        "canonical English identity to a locale identity (name + email; "
        "the voice is gender-pinned to the new name). Check the sampled "
        "names and emails read like real people from this locale — a wrong "
        "gender mapping or an unnatural name/email is an issue.",
        text="\n".join(lines),
    )


def _runtime_user_sections(
    pack, user, backchannel_prompt: str, identity_section: Optional[PromptSectionVM]
) -> list[PromptSectionVM]:
    """User-simulator prompts rendered for this language."""
    cat: PromptCategory = "runtime_user"
    return [
        PromptSectionVM(
            item_id="system_prompt_english",
            category=cat,
            title="Voice user-simulator system prompt",
            caption="The full system prompt the voice user simulator runs "
            "with: English simulation guidelines, the persona and "
            "localization blocks for your language, a mandatory "
            "target-language directive, and the task-0 scenario with English "
            "instructions but localized names/entities. This text fully "
            "controls how the simulated caller behaves and speaks.",
            text=user.system_prompt,
        ),
        PromptSectionVM(
            item_id="backchannel_decision_prompt",
            category=cat,
            title="Backchannel decision prompt (rendered for this language)",
            caption="The prompt that decides, mid-call, whether the caller "
            "murmurs a short acknowledgement while the agent is talking — "
            "rendered with this language's density level and the persona's "
            f"real phrases. The literal '{CONVERSATION_HISTORY_PLACEHOLDER}' "
            "marks where the live conversation is inserted each time.",
            text=backchannel_prompt,
        ),
        PromptSectionVM(
            item_id="agent_greeting",
            category=cat,
            title="Agent greeting",
            caption="The agent's opening line for this language (the caller "
            "hears this first). It should be a natural service greeting.",
            text=pack.agent_greeting or "(this pack has no agent_greeting)",
        ),
        PromptSectionVM(
            item_id="speech_phrases",
            category=cat,
            title="Runtime spoken inserts — backchannels and out-of-turn speech",
            caption="Not an LLM prompt: these pack-authored phrases are "
            "injected directly into the call audio (backchannels while the "
            "agent talks; out-of-turn asides to people in the room). They "
            "must sound like things a real caller would actually say.",
            text=_speech_phrases_text(user),
        ),
        *([identity_section] if identity_section else []),
    ]


def _runtime_agent_sections(
    language: str, locale: str | None, environment, task=None
) -> list[PromptSectionVM]:
    """Agent-side prompts, built by the REAL build_agent path.

    ``task`` matters: build_agent derives the native-script-DB clause flip
    from the task id, so a native-variant packet renders the agent prompt a
    native run would actually carry."""
    from tau2.data_model.simulation import AudioNativeConfig
    from tau2.runner.build import build_agent

    voice_agent = build_agent(
        "discrete_time_audio_native_agent",
        environment,
        audio_native_config=AudioNativeConfig(),
        language=language,
        locale=locale,
        task=task,
    )
    text_agent = build_agent(
        "llm_agent", environment, language=language, locale=locale, task=task
    )
    domain_name = environment.get_domain_name()

    cat: PromptCategory = "runtime_agent"
    return [
        PromptSectionVM(
            item_id="agent_system_prompt_voice",
            category=cat,
            title="Voice agent system prompt (audio-native, as a real voice "
            "run builds it)",
            caption="The COMPLETE system prompt of the agent under test on a "
            f"voice run: the voice-agent instruction, the full {domain_name} "
            "domain policy, and — at the very end — this language pack's "
            "agent language clause telling the agent the caller's original "
            "locale and how to respond in their language. Review the response "
            "instructions especially: they are pack-authored text.",
            text=voice_agent.system_prompt,
        ),
        PromptSectionVM(
            item_id="agent_system_prompt_text",
            category=cat,
            title="Text agent system prompt (half-duplex chat, as a real text "
            "run builds it)",
            caption="The agent system prompt on TEXT (typed chat) runs: agent "
            "instruction + domain policy + the same pack agent_language_clause "
            "appended for your language.",
            text=text_agent.system_prompt,
        ),
    ]


def _judge_sections(language: str, pack) -> list[PromptSectionVM]:
    """The scoring prompts whose rubric text comes from this language's pack."""
    from tau2.evaluator.evaluator_communicate import (
        COMMUNICATE_JUDGE_LANGUAGE_ADDENDUM_TEMPLATE,
        COMMUNICATE_JUDGE_SYSTEM_PROMPT,
        COMMUNICATE_JUDGE_USER_PROMPT_TEMPLATE,
    )
    from tau2.evaluator.evaluator_nl_assertions import (
        NL_ASSERTIONS_JUDGE_LANGUAGE_ADDENDUM_TEMPLATE,
        NL_ASSERTIONS_JUDGE_SYSTEM_PROMPT,
        NL_ASSERTIONS_JUDGE_USER_PROMPT_TEMPLATE,
    )
    from tau2.judges.delivery.factors import build_delivery_rubric
    from tau2.judges.delivery.judge import (
        build_user_prompt as build_delivery_user_prompt,
    )
    from tau2.judges.nativeness.factors import (
        NativenessFactorConfig,
        judge_factors_for,
    )
    from tau2.judges.nativeness.harness import (
        ISOLATED_UTTERANCE_FACTOR_IDS,
        utterance_factor_batches,
    )
    from tau2.judges.nativeness.judge import (
        NativenessJudgeCriterion,
        NativenessJudgeInput,
    )
    from tau2.judges.nativeness.judge import (
        build_user_prompt as build_nativeness_user_prompt,
    )

    script = _pack_script(pack)

    # Nativeness judge: render the exact request groups used at runtime. The
    # packet keeps placeholders for per-simulation text and context.
    factors = [
        factor
        for factor in judge_factors_for(language)
        if factor.type == "judge" and factor.enabled
    ]
    factor_prompts: list[tuple[str, str]] = []

    def add_factor_prompt(
        level: Literal["call", "utterance"],
        selected: list[NativenessFactorConfig],
        heading: str,
    ) -> None:
        request = NativenessJudgeInput(
            language=language,
            evaluation_level=level,
            criteria=[
                NativenessJudgeCriterion.from_rubric(factor.id, factor.params)
                for factor in selected
            ],
            agent_text=RUNTIME_TRANSCRIPT_PLACEHOLDER,
            customer_context=(
                "{{ALL_CUSTOMER_TURNS}}"
                if level == "call"
                else "{{PRECEDING_CUSTOMER_TURN}}"
            ),
            agent_context=RUNTIME_AGENT_CONTEXT_PLACEHOLDER,
        )
        factor_prompts.append((heading, build_nativeness_user_prompt(request)))

    call_factors = [factor for factor in factors if factor.evaluation_level == "call"]
    if call_factors:
        add_factor_prompt("call", call_factors, "CALL-LEVEL BATCH")

    utterance_factors = [
        factor for factor in factors if factor.evaluation_level == "utterance"
    ]
    for batch in utterance_factor_batches(utterance_factors):
        isolated = len(batch) == 1 and batch[0].id in ISOLATED_UTTERANCE_FACTOR_IDS
        heading = (
            f"UTTERANCE-LEVEL ISOLATED — {batch[0].id}"
            if isolated
            else "UTTERANCE-LEVEL SHARED BATCH"
        )
        add_factor_prompt("utterance", batch, heading)
    nativeness_user_text = (
        _composite(*factor_prompts)
        if factor_prompts
        else "(this language pack defines no nativeness judge factors — the "
        "nativeness LLM judge does not run for this language)"
    )

    # Delivery judge: the full user prompt with this language's rubric
    # (pack factors or the native-listener fallback) + the output shape.
    delivery_user_text = build_delivery_user_prompt(
        RUNTIME_EXPECTED_TEXT_PLACEHOLDER,
        build_delivery_rubric(language),
        was_interrupted=False,
    )

    # The runtime judges append the language addendum only on non-English
    # runs (see evaluator_communicate / evaluator_nl_assertions) — mirror
    # that so an English packet shows the prompts English runs actually use.
    localized_judges = language.lower() != "en"
    communicate_system = COMMUNICATE_JUDGE_SYSTEM_PROMPT
    nl_system = NL_ASSERTIONS_JUDGE_SYSTEM_PROMPT
    if localized_judges:
        communicate_system += COMMUNICATE_JUDGE_LANGUAGE_ADDENDUM_TEMPLATE.format(
            language=language
        )
        script_note = f" (script: {script})" if script else ""
        nl_system += NL_ASSERTIONS_JUDGE_LANGUAGE_ADDENDUM_TEMPLATE.format(
            language=language, script_note=script_note
        )
    addendum_label = (
        "SYSTEM PROMPT (with language addendum)"
        if localized_judges
        else "SYSTEM PROMPT (English run — no language addendum)"
    )

    cat: PromptCategory = "judges"
    return [
        PromptSectionVM(
            item_id="nativeness_judge_user_prompts",
            category=cat,
            title="Nativeness judge — runtime user-prompt batches",
            caption="The exact call-level, isolated utterance-level, and shared "
            "utterance-level request groups used at runtime. The nuance / "
            "native-does / AI-likely-does text is pack-authored — review it as "
            "carefully as the runtime prompts: it defines what counts as native. "
            "Bracketed […] stand-ins mark the transcript and agent context filled "
            "per run.",
            text=nativeness_user_text,
        ),
        PromptSectionVM(
            item_id="delivery_judge_user_prompt",
            category=cat,
            title="Delivery judge — user prompt with this language's rubric",
            caption="The delivery judge's user prompt rendered for this "
            "language: the rubric block comes from the pack's delivery "
            "factors (or the native-listener fallback when a pack defines "
            "none), followed by the fixed output-shape instructions. The "
            "audio clip itself is attached at runtime.",
            text=delivery_user_text,
        ),
        PromptSectionVM(
            item_id="communicate_judge_prompts",
            category=cat,
            title="Communicate-info judge (task success on non-English runs)",
            caption="Decides whether the agent conveyed each expected piece "
            "of information; on non-English runs it replaces exact substring "
            "matching, so pass@1 depends on it. The language addendum below "
            "is rendered for this language; the user prompt's {…} slots take "
            "the expected info and transcript per check.",
            text=_composite(
                (addendum_label, communicate_system),
                ("USER PROMPT TEMPLATE", COMMUNICATE_JUDGE_USER_PROMPT_TEMPLATE),
            ),
        ),
        PromptSectionVM(
            item_id="nl_assertions_judge_prompts",
            category=cat,
            title="NL-assertions judge (expected-outcome grading)",
            caption="Grades each natural-language expected outcome of a task. "
            "The language addendum below is rendered with this language's "
            "code and script; the user prompt's {…} slots take the "
            "conversation and assertions per run.",
            text=_composite(
                (addendum_label, nl_system),
                ("USER PROMPT TEMPLATE", NL_ASSERTIONS_JUDGE_USER_PROMPT_TEMPLATE),
            ),
        ),
    ]


def _factory_sections(language: str, pack, domain: str) -> list[PromptSectionVM]:
    """Factory build-time prompts, rendered with this pack's real data."""
    from tau2.annotation.nuance_candidates import (
        LANGUAGE_HINTS,
        nuance_candidates_prompt,
    )
    from tau2.multilingual.factory.author_form import (
        AUTOFORM_SYSTEM_PROMPT,
        build_autoform_prompt,
    )
    from tau2.multilingual.factory.localization_drafting import (
        DRAFT_LOCALIZATION_SYSTEM_PROMPT,
        build_localization_prompt,
    )
    from tau2.multilingual.factory.translation_rows import build_prompt_context

    script = _pack_script(pack) or "latn"
    prompt_ctx = build_prompt_context(language, script, domain=domain)
    localization_prompt = build_localization_prompt(
        language,
        pack.display_name,
        script,
        domain=domain,
        translation_guidance=pack.translation_guidance,
        persona_summaries=[
            p.short_description for _, p in sorted(pack.personas.items())
        ],
    )
    cat: PromptCategory = "factory"
    return [
        PromptSectionVM(
            item_id="factory_translator_system_prompt",
            category=cat,
            title="Translation factory — translator system prompt (rendered "
            "for this language)",
            caption="The system prompt of the task translator, rendered with "
            "this pack's author-written translation_guidance. The per-task "
            "user prompts carry the English text and protected values at "
            "build time. Every localized scenario you reviewed above was "
            "produced under this prompt.",
            text=prompt_ctx.translator_system,
        ),
        PromptSectionVM(
            item_id="factory_fixer_system_prompt",
            category=cat,
            title="Translation factory — fixer system prompt (rendered for "
            "this language)",
            caption="The system prompt of the translation FIXER, which "
            "repairs translations the verifier flagged. Same pack "
            "translation_guidance block as the translator.",
            text=prompt_ctx.fixer_system,
        ),
        PromptSectionVM(
            item_id="factory_verifier_system_prompt",
            category=cat,
            title="Translation factory — verifier system prompt (rendered "
            "for this language)",
            caption="The system prompt of the translation VERIFIER, which "
            "checks meaning preservation and naturalness of each translated "
            "task before it ships.",
            text=prompt_ctx.verifier_system,
        ),
        PromptSectionVM(
            item_id="factory_localization_drafting_prompts",
            category=cat,
            title="Localization drafting prompt (rendered with this pack's data)",
            caption="The one-call prompt that drafted this pack's "
            "localization block (native symbol readouts, date/number "
            "examples, the domain glossary). Rendered with the real "
            "translation guidance and persona register summaries.",
            text=_composite(
                ("SYSTEM PROMPT", DRAFT_LOCALIZATION_SYSTEM_PROMPT),
                ("USER PROMPT (rendered)", localization_prompt),
            ),
        ),
        PromptSectionVM(
            item_id="factory_autoform_prompts",
            category=cat,
            title="Author-form (autoform) prompt (rendered for this language)",
            caption="The prompt that generates the pack's author form — the "
            "human-reviewable seed (persona sketches, locale notes) a new "
            "language pack is drafted from.",
            text=_composite(
                ("SYSTEM PROMPT", AUTOFORM_SYSTEM_PROMPT),
                (
                    "USER PROMPT (rendered)",
                    build_autoform_prompt(language, pack.display_name, script),
                ),
            ),
        ),
        PromptSectionVM(
            item_id="factory_nuance_candidates_prompt",
            category=cat,
            title="Nuance-candidates prompt (rendered for this language)",
            caption="The prompt that proposes candidate nativeness nuances "
            "for this language during audits — the pipeline that feeds the "
            "nativeness judge factors reviewed above.",
            text=nuance_candidates_prompt(
                pack.display_name, LANGUAGE_HINTS.get(language, "")
            ),
        ),
    ]


def _render_prompt_sections(
    opts: PromptBedPacketOptions,
) -> tuple[list[PromptSectionVM], dict]:
    """Render every per-language prompt section (four categories) + provenance."""
    from tau2.judges.delivery.judge import DELIVERY_JUDGE_PROMPT_VERSION
    from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION
    from tau2.multilingual.english_prompts import TARGET_LANGUAGE_DIRECTIVE_VERSION
    from tau2.multilingual.registry import get_language_pack
    from tau2.runner.build import build_environment

    language, domain = opts.language, opts.domain
    pack = get_language_pack(language)
    if pack is None:
        raise ValueError(f"no registered language pack for '{language}'")

    task_set_name, task = _load_task(domain, language, opts.task_index, opts.task_set)
    environment = build_environment(domain)

    user = _build_runtime_user(
        domain=domain,
        language=language,
        task_set_name=task_set_name,
        task=task,
        environment=environment,
    )
    persona_id = getattr(user.persona_config, "persona_id", None)

    backchannel_prompt = user.backchannel_decision_prompt.replace(
        "{conversation_history}", CONVERSATION_HISTORY_PLACEHOLDER
    )

    from tau2.multilingual.native_script import is_native_identity_task_id

    identity_section = (
        _native_identity_swap_section(language, domain)
        if is_native_identity_task_id(str(task.id))
        else _identity_swap_section(language, domain)
    )
    sections = [
        *_runtime_user_sections(pack, user, backchannel_prompt, identity_section),
        *_runtime_agent_sections(
            language,
            getattr(user.persona_config, "locale", None),
            environment,
            task=task,
        ),
        *_judge_sections(language, pack),
        *_factory_sections(language, pack, domain),
    ]
    ids = [s.item_id for s in sections]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"duplicate prompt section ids: {ids}")

    provenance = {
        "task_set_name": task_set_name,
        "task_id": str(task.id),
        "task_index": opts.task_index,
        "persona_id": persona_id,
        "target_language_directive_version": TARGET_LANGUAGE_DIRECTIVE_VERSION,
        "nativeness_judge_prompt_version": NATIVENESS_JUDGE_PROMPT_VERSION,
        "delivery_judge_prompt_version": DELIVERY_JUDGE_PROMPT_VERSION,
        "backchannel_level": getattr(pack.backchannel_level, "value", None),
        "sections_by_category": {
            key: [s.item_id for s in sections if s.category == key]
            for key in PROMPT_CATEGORIES
        },
        "prompt_sha256": {
            s.item_id: hashlib.sha256(s.text.encode()).hexdigest() for s in sections
        },
    }
    return sections, provenance


# ---------------------------------------------------------------------------
# Persona voice samples (the one online step; skippable)
# ---------------------------------------------------------------------------


def _render_voice_samples(
    language: str, pack, packet_dir: Path
) -> tuple[list[VoiceSampleVM], dict[str, str]]:
    """Render each pack persona's pinned voice into ``<packet>/voices/`` and
    return the rows + their sha256 hashes.

    Goes through the ``factory voice-samples`` core (same audition script and
    TTS parameters as a human voice audition). Any failed render is a LOUD
    error — a packet silently missing a persona's voice would hide exactly
    the defect the voices section exists to catch.
    """
    from tau2.multilingual.factory.voice_samples import render_voice_samples

    out_dir = packet_dir / "voices"
    results = render_voice_samples(language, out_dir=out_dir, label="packet")
    failed = [f"{r.label}: {r.error}" for r in results if r.error or r.path is None]
    if failed:
        raise RuntimeError(
            f"voice sample render failed for '{language}': {'; '.join(failed)}"
        )
    by_persona = {r.label.removesuffix("_packet"): r for r in results}
    voices: list[VoiceSampleVM] = []
    hashes: dict[str, str] = {}
    for persona_id, persona in sorted(pack.personas.items()):
        result = by_persona.get(persona_id)
        if result is None:
            raise RuntimeError(
                f"no voice sample rendered for persona '{persona_id}' — "
                f"got {sorted(by_persona)}"
            )
        item_id = f"voice_{language}_{persona_id}"
        voices.append(
            VoiceSampleVM(
                item_id=item_id,
                persona_id=persona_id,
                gender=persona.tags.get("gender", "unknown"),
                description=persona.short_description,
                rel_href=f"voices/{result.path.name}",
            )
        )
        hashes[item_id] = hashlib.sha256(result.path.read_bytes()).hexdigest()
    return voices, hashes


# ---------------------------------------------------------------------------
# Beds (every locale, both beds, plus the shared office bed)
# ---------------------------------------------------------------------------


def _discover_bed_files() -> list[BedFileVM]:
    """Every locale's two beds + the shared office bed, from the data tree.

    A locale dir missing either bed is a LOUD error — a silently absent bed
    row would hide exactly the defect this packet exists to catch.
    """
    from tau2 import voice_config
    from tau2.multilingual.factory.paths import (
        OUTDOOR_BED_BASENAME,
        SHARED_OFFICE_BED_BASENAME,
        TV_KITCHEN_BED_BASENAME,
    )

    root = Path(voice_config.BACKGROUND_NOISE_CONTINUOUS_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"background-bed directory not found: {root}")
    # Only language_COUNTRY dirs are locales; the tree also carries shared
    # ingredient dirs (e.g. continuous/shared/kitchen_ambience.wav, a mix
    # source for the indoor beds) that own no per-locale bed pair.
    locale_dirs = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and re.fullmatch(r"[a-z]{2,3}_[A-Z]{2}", p.name)
    )
    if not locale_dirs:
        raise FileNotFoundError(f"no locale bed directories under {root}")

    beds: list[BedFileVM] = []
    missing: list[str] = []
    for locale_dir in locale_dirs:
        for bed_type, basename in (
            ("outdoor", OUTDOOR_BED_BASENAME),
            ("indoor_tv", TV_KITCHEN_BED_BASENAME),
        ):
            src = locale_dir / basename
            if not src.is_file():
                missing.append(f"{locale_dir.name}/{basename}")
                continue
            beds.append(
                BedFileVM(
                    item_id=f"bed_{locale_dir.name}_{bed_type}",
                    locale=locale_dir.name,
                    bed_type=bed_type,
                    label=BED_TYPE_LABELS[bed_type],
                    src=src,
                    rel_href=f"beds/{locale_dir.name}/{basename}",
                )
            )
    shared = root / SHARED_OFFICE_BED_BASENAME
    if not shared.is_file():
        missing.append(SHARED_OFFICE_BED_BASENAME)
    if missing:
        raise FileNotFoundError(f"bed files missing under {root}: {', '.join(missing)}")
    beds.append(
        BedFileVM(
            item_id="bed_shared_office",
            locale="shared",
            bed_type="shared_office",
            label=BED_TYPE_LABELS["shared_office"],
            src=shared,
            rel_href=f"beds/{SHARED_OFFICE_BED_BASENAME}",
        )
    )
    return beds


def _copy_beds(beds: list[BedFileVM], packet_dir: Path) -> dict[str, str]:
    """Copy every bed into the packet; returns {item_id: sha256}."""
    hashes: dict[str, str] = {}
    for bed in beds:
        dest = packet_dir / bed.rel_href
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bed.src, dest)
        h = hashlib.sha256()
        with open(dest, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b""):
                h.update(chunk)
        hashes[bed.item_id] = h.hexdigest()
    return hashes


# ---------------------------------------------------------------------------
# Page + manifest
# ---------------------------------------------------------------------------


def _render_page(
    *,
    batch_id: str,
    batch_name: str,
    language: str,
    display_name: str,
    sections: list[PromptSectionVM],
    voices: list[VoiceSampleVM],
    beds: list[BedFileVM],
) -> str:
    items = [
        *(
            {
                "item_id": s.item_id,
                "row_kind": PromptBedItemKind.PROMPT.value,
                "locale": "",
                "bed_type": "",
            }
            for s in sections
        ),
        *(
            {
                "item_id": v.item_id,
                "row_kind": PromptBedItemKind.VOICE.value,
                "locale": "",
                "bed_type": "",
            }
            for v in voices
        ),
        *(
            {
                "item_id": b.item_id,
                "row_kind": PromptBedItemKind.BED.value,
                "locale": b.locale,
                "bed_type": b.bed_type,
            }
            for b in beds
        ),
    ]
    packet_config = {
        "batch_id": batch_id,
        "batch_name": batch_name,
        "kind": PROMPT_BED_KIND,
        "language": language,
        "csv_headers": PromptBedRow.headers(),
        "items": items,
    }
    groups = [
        {
            "key": key,
            "title": title,
            "blurb": blurb,
            "sections": [s for s in sections if s.category == key],
        }
        for key, (title, blurb) in PROMPT_CATEGORIES.items()
    ]
    return (
        _env()
        .get_template("prompt_bed.html.j2")
        .render(
            packet_config=_config_json(packet_config),
            css=_static("index.css") + "\n" + _static("prompt_bed.css"),
            js=_static("form_core.js") + "\n" + _static("prompt_bed_form.js"),
            batch_name=batch_name,
            task_label=PROMPT_BED_LABEL,
            display_name=display_name,
            groups=groups,
            voices=voices,
            beds=beds,
        )
    )


def build_prompt_bed_packet(opts: PromptBedPacketOptions) -> Path:
    """Build one language's prompt + bed review packet; returns the manifest."""
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack(opts.language)
    if pack is None:
        raise ValueError(f"no registered language pack for '{opts.language}'")

    packet_dir = opts.packet_dir
    if packet_dir.exists() and any(packet_dir.iterdir()):
        raise FileExistsError(
            f"output directory already exists: {packet_dir} — remove it to "
            "rebuild (packets are regenerated whole, never patched)"
        )

    # Beds first: discovery is cheap and its failure modes (missing files)
    # should surface before the heavier prompt render.
    beds = _discover_bed_files()
    sections, prompt_provenance = _render_prompt_sections(opts)

    packet_dir.mkdir(parents=True, exist_ok=True)
    bed_hashes = _copy_beds(beds, packet_dir)
    voices: list[VoiceSampleVM] = []
    voice_hashes: dict[str, str] = {}
    if opts.voice_samples:
        voices, voice_hashes = _render_voice_samples(opts.language, pack, packet_dir)

    batch_name = f"prompt_bed_review_{opts.packet_key}"
    batch_id = derive_batch_id(
        PROMPT_BED_KIND,
        batch_name,
        {**prompt_provenance["prompt_sha256"], **voice_hashes, **bed_hashes},
    )

    html = _render_page(
        batch_id=batch_id,
        batch_name=batch_name,
        language=opts.language,
        display_name=pack.display_name,
        sections=sections,
        voices=voices,
        beds=beds,
    )
    (packet_dir / "index.html").write_text(html)

    manifest = ArtifactManifest(
        kind=PROMPT_BED_KIND,
        batch_id=batch_id,
        batch_name=batch_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        language=opts.language,
        domain=opts.domain,
        form="prompt_bed_review",
        provenance={
            **prompt_provenance,
            "voices": {
                v.item_id: {
                    "path": v.rel_href,
                    "persona_id": v.persona_id,
                    "sha256": voice_hashes[v.item_id],
                }
                for v in voices
            }
            if opts.voice_samples
            else "skipped (--no-voice-samples)",
            "beds": {
                bed.item_id: {"path": bed.rel_href, "sha256": bed_hashes[bed.item_id]}
                for bed in beds
            },
        },
        files=[],
        entries=None,
    )
    manifest_path = write_json_artifact(packet_dir / PACKET_MANIFEST_NAME, manifest)
    logger.info(
        f"prompt+bed packet '{batch_name}' ({batch_id}): "
        f"{len(sections)} prompt sections + {len(voices)} voice samples + "
        f"{len(beds)} beds -> {packet_dir}"
    )
    return manifest_path

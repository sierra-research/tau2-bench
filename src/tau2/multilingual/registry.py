# Copyright Sierra
"""Registry for Language Packs.

Packs are registered into a module-level dict, and shared code looks them up
by language code or persona id. Registration also makes each pack persona
available as a ``VoicePersona`` so the TTS voice-id plumbing
(``get_elevenlabs_voice_id``, speech generators) works.

All getters lazily trigger pack discovery (``tau2.multilingual.loader``), so
callers never import language packs explicitly. With no packs installed, every
lookup is a cheap miss and English behavior applies.
"""

import random
import re
from collections.abc import Iterable
from typing import Optional

from loguru import logger

from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN
from tau2.data_model.voice_personas import (
    VoicePersona,
    _resolve_voice_id,
    register_voice_persona,
)
from tau2.multilingual.schema import LanguagePack, MultilingualPersonaConfig
from tau2.multilingual.varieties import language_variety_name, locale_display_name

_LANGUAGE_PACKS: dict[str, LanguagePack] = {}


class PersonaResolutionError(RuntimeError):
    """A run cannot honour the caller persona its config or its tasks require.

    Raised in place of falling back to a stock English voice. That fallback is
    unrecoverable once it happens: the simulation is written, scored, and
    shipped with ``speech_environment.persona_id`` null and an English speaker
    reading Korean, and nothing downstream distinguishes it from a real call.
    A run that cannot speak as its caller must die instead.
    """


def _reset_for_tests() -> None:
    """TEST-ONLY: drop all registered packs.

    Voice personas registered as a side effect of ``register_language_pack``
    are NOT touched here — tests snapshot and restore
    ``tau2.data_model.voice_personas`` themselves (see
    ``tests/test_multilingual/factory_testing/conftest_helpers.py``). Never
    call this from production code.
    """
    _LANGUAGE_PACKS.clear()


def register_language_pack(pack: LanguagePack) -> None:
    """Register a language pack and its personas.

    Each persona is also registered as a ``VoicePersona`` (name=persona_id) so
    voice-id resolution and audio generation work through the existing
    registries. Persona voice ids honor the ``TAU2_VOICE_ID_<PERSONA_ID>``
    env-var override, like all other voice personas.

    Raises:
        ValueError: If the language or any persona_id is already registered.
    """
    if pack.language in _LANGUAGE_PACKS:
        raise ValueError(f"Language pack '{pack.language}' already registered")

    for persona in pack.personas.values():
        voice_persona = VoicePersona(
            elevenlabs_voice_id=_resolve_voice_id(
                persona.persona_id, persona.voice_id or ""
            ),
            name=persona.persona_id,
            display_name=persona.display_name,
            short_description=persona.short_description,
            prompt=persona.tts_voice_prompt,
            complexity="regular",
            language=pack.language,
        )
        register_voice_persona(voice_persona)

    _LANGUAGE_PACKS[pack.language] = pack
    logger.info(
        f"Registered language pack '{pack.language}' "
        f"({len(pack.personas)} personas: {sorted(pack.personas)})"
    )


def get_language_pack(language: str) -> Optional[LanguagePack]:
    """Get the pack for a language code, or None if none is registered.

    English now ships as its own pack (``en``, the shared baseline), so
    ``get_language_pack('en')`` returns it; a None result means the code has
    no pack and the legacy hardcoded-English defaults apply.
    """
    _ensure_loaded()
    return _LANGUAGE_PACKS.get(language)


def get_agent_language_clause(
    language: Optional[str],
    locale: Optional[str] = None,
    *,
    native_script_db: bool = False,
) -> Optional[str]:
    """Render the agent-side language clause for one resolved caller persona.

    The pack owns language-wide response conventions. The resolved persona's
    locale adds caller-background context without asserting where the caller is
    physically located during the scenario. Keeping this composition here makes
    it identical for text and voice agents and for every benchmark domain.

    The locale renders as its human-readable place phrase and the language
    carries its pinned variety name (``tau2.multilingual.varieties`` — the same
    catalogs the user-simulator directive reads): a raw code like 'ES-MD' is
    useless to the model. An unmapped locale or language raises at render time
    (see the varieties module for the fail-loudly contract).

    ``native_script_db`` is set for a run on a ``_identity_native`` task set
    (the native-script DB ablation): the pack's
    ``agent_native_script_db_clause`` is appended so the prompt tells the
    truth about the DB's script. A native-variant run on a pack without that
    clause raises — the default clause implies Latin-script tool arguments,
    which would be a lie about this DB.
    """
    if language is None:
        return None
    pack = get_language_pack(language)
    if pack is None:
        return None

    sections: list[str] = []
    if locale:
        sections.append(
            f"The customer is originally from {locale_display_name(locale)} and "
            f"speaks {pack.display_name} — expect "
            f"{language_variety_name(pack.language)}. This describes their "
            "linguistic and regional background, not necessarily their current "
            "physical location."
        )
    if pack.agent_language_clause:
        sections.append(pack.agent_language_clause)
    if native_script_db:
        if not pack.agent_native_script_db_clause:
            raise ValueError(
                f"Language pack '{pack.language}' has no "
                "agent_native_script_db_clause, but this run's tasks are a "
                "native-script identity set — running it would tell the agent "
                "the DB is something it is not. Author the clause in the "
                "pack.yaml before running the native variant."
            )
        sections.append(pack.agent_native_script_db_clause)
    return "\n".join(sections) or None


def list_language_packs() -> list[str]:
    """List registered language codes."""
    _ensure_loaded()
    return sorted(_LANGUAGE_PACKS)


def get_localization_guidelines(
    language: Optional[str],
    mode: str = "voice",
    domain: Optional[str] = None,
) -> Optional[str]:
    """Rendered spoken-localization prompt section for a language, or None.

    The single lookup seam the user simulators call: resolves the pack, and
    renders its ``localization`` block (date/numeral rule, speech
    conventions, domain glossary) via
    ``LocalizationPackConfig.to_prompt_text``. None when the language is
    None, has no pack, or the pack carries no localization data —
    English/unlocalized runs are unaffected.

    Args:
        language: ISO 639-1 code of an active language pack, or None.
        mode: "voice" or "text" (text omits the spoken-only sub-sections).
        domain: The run's benchmark domain; selects which of the pack's
            per-domain glossaries renders (None renders no glossary — the
            language-mechanics sections are domain-independent).
    """
    if language is None:
        return None
    pack = get_language_pack(language)
    if pack is None or pack.localization is None:
        return None
    return pack.localization.to_prompt_text(pack.display_name, mode=mode, domain=domain)


def get_guideline_example_items(
    language: Optional[str], kind_id: str, gender: Optional[str] = None
) -> Optional[list[str]]:
    """A pack's native utterances for a guideline example kind, or None.

    The lookup seam behind the English voice guidelines' ``<EXAMPLE:kind>``
    slots. None (no language, no pack, no localization block, or the kind
    not carried) renders the kind's fixed English default.

    ``gender`` is the speaking persona's gender tag. Where the pack carries a
    male realization of a kind (languages that mark speaker gender in these
    mechanics — Japanese pronouns, sentence-final particles, softeners), male
    speakers render it; everyone else renders the shared palette.
    """
    if language is None:
        return None
    pack = get_language_pack(language)
    if pack is None or pack.localization is None:
        return None
    return pack.localization.guideline_example_items(kind_id, gender)


def get_spellout_section(language: Optional[str]) -> Optional[str]:
    """Rendered 'Speaking Special Characters and Numbers' section, or None.

    The single owner of the symbol table + worked value readouts: the voice
    user simulator substitutes this into the English voice guidelines'
    ``<SPOKEN_VALUES_SPELLOUT>`` slot so english-prompt-mode runs voice
    symbols and values natively. None when the language is None, has no pack,
    or the pack lacks the data — the fixed English section applies (see
    ``tau2.user.user_simulator.ENGLISH_SPELLOUT_SECTION``).

    Args:
        language: ISO 639-1 code of an active language pack, or None.
    """
    if language is None:
        return None
    pack = get_language_pack(language)
    if pack is None or pack.localization is None:
        return None
    return pack.localization.to_spellout_section(pack.display_name)


def get_spelled_entity_tables(language: Optional[str]):
    """The pack's pre-TTS spelled-entity word tables, or None.

    This lookup returns a
    ``SpelledEntityTables`` ONLY when the language's pack opts in
    (``localization.spelled_entity_normalization: true``). None (no language,
    no pack, no localization block, or the flag off) means the TTS-bound text
    is passed through untouched — languages whose TTS already renders codes
    correctly are unaffected.

    Args:
        language: ISO 639-1 code of an active language pack, or None.
    """
    if language is None:
        return None
    pack = get_language_pack(language)
    if pack is None or pack.localization is None:
        return None
    return pack.localization.spelled_entity_tables()


def resolve_text_input_style(language: Optional[str], style) -> Optional[str]:
    """Validate a run's text input style against a language's pack.

    The single build-time gate for the ``--text-input-style`` arm knob:
    returns the rendered directive block (fixed versioned template + the
    pack's worked examples; see ``tau2.multilingual.text_input_catalog``), or
    None for the default ``native_script`` style (which renders nothing —
    the baseline arm is untouched).

    Raises:
        ValueError: If a non-default style is requested for a run without a
            resolved language pack, or for a pack that does not declare the
            style. A silent fallback to native_script would corrupt the arm.
    """
    from tau2.multilingual.text_input_catalog import (
        TextInputStyle,
        render_input_style_directive,
    )

    resolved = TextInputStyle(style)
    if resolved is TextInputStyle.NATIVE_SCRIPT:
        return None
    pack = get_language_pack(language) if language else None
    if pack is None:
        raise ValueError(
            f"--text-input-style '{resolved.value}' requires a language-pack "
            "run (--user-persona-id naming a registered language), but this "
            "run resolved no language pack."
        )
    spec = pack.text_input.style_spec(resolved) if pack.text_input else None
    if spec is None:
        supported = sorted(
            s.style.value for s in (pack.text_input.styles if pack.text_input else [])
        )
        raise ValueError(
            f"Language pack '{pack.language}' does not support text input "
            f"style '{resolved.value}' (supported beyond native_script: "
            f"{supported or 'none'})."
        )
    return render_input_style_directive(resolved, pack.display_name, spec.examples)


def get_multilingual_persona(
    persona_id: str,
) -> Optional[tuple[LanguagePack, MultilingualPersonaConfig]]:
    """Look up a persona across all packs by its persona_id.

    Returns the (pack, persona) pair, or None if the id does not belong to
    any language pack (e.g. it is a plain English voice persona).
    """
    _ensure_loaded()
    for pack in _LANGUAGE_PACKS.values():
        persona = pack.get_persona(persona_id)
        if persona is not None:
            return pack, persona
    return None


def persona_rotation_key(task_id: Optional[str]) -> Optional[int]:
    """Round-robin key for bare-language-code persona assignment, or None.

    Keys on the task id's FIRST run of digits ('12' in 'task_12_v3'), so a
    numeric suffix never perturbs the rotation of the task's own number.
    None when the id is missing or has no digits — callers then fall back to
    the first element of the seeded persona order. The single owner of the
    keying: voice (``sample_voice_config``) and text
    (``resolve_task_persona``) both call this, so the two sides always make
    the identical persona choice for a given (task_id, rotation seed).
    """
    match = re.search(r"\d+", task_id or "")
    return int(match.group(0)) if match else None


def language_persona_order(persona_ids: list[str], rotation_seed: int) -> list[str]:
    """Deterministic seed-keyed shuffle of a pack's persona ids.

    Seeded with a derived string, so the order is stable across processes and
    runs (string seeds do not go through Python's salted ``hash()``) while still
    varying with the seed. Shared by voice (``sample_voice_config``) and text
    (``resolve_task_persona``) so both use one balanced round-robin.
    """
    return random.Random(f"tau2-language-persona-order:{rotation_seed}").sample(
        persona_ids, len(persona_ids)
    )


def resolve_task_persona(
    persona_name: str,
    task_id: Optional[str] = None,
    run_seed: Optional[int] = None,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
) -> str:
    """Resolve a persona override to a concrete persona id for one task.

    Mirrors the bare-language-code assignment in
    ``tau2.user_simulation_voice_presets.sample_voice_config`` so text runs get
    the same persona as voice runs without reimplementing it:

    - A full persona id (e.g. ``'imran_hindi_v1'``) or a plain English persona
      name passes through unchanged. An id that names neither (see
      :func:`require_persona_override`) raises rather than passing through to
      a stock English voice.
    - A bare language code with a registered pack (e.g. ``'hi'``) is assigned
      one of that pack's personas. When the task has a caller-gender sidecar
      entry (see ``tau2 factory localize-entities`` — covers plain and
      ``_identity`` localized task ids), the pool is first restricted to
      same-gender personas so the persona's gender (first-person grammatical
      agreement, display name) always matches the task's caller name — the
      same pinning the voice path applies. Within the pool the assignment is
      a seed-rotated round-robin keyed by ``persona_rotation_key`` (the task
      id's first digit run; contiguous task numbers split evenly, counts
      differing by at most 1), rotated by ``run_seed`` so different seeds
      vary the pairing. Without task-id digits it falls back to the
      deterministic first element of the seeded order.

    The rotation key must be the RUN-level seed (constant across a run's tasks)
    for the balance guarantee to hold. ``domain`` locates the caller-gender
    sidecar (``data/tau2/multilingual/<lang>/<domain>_caller_gender_<lang>.json``).

    Raises:
        PersonaResolutionError: If ``persona_name`` names no known persona.
    """
    require_persona_override(persona_name)
    pack = get_language_pack(persona_name)
    if pack is None:
        return persona_name
    persona_ids = sorted(pack.personas)
    if task_id:
        from tau2.multilingual.factory.entity_localization import (
            caller_gender_for_task,
        )

        caller_gender = caller_gender_for_task(task_id, persona_name, domain)
        if caller_gender:
            matching = [
                pid
                for pid in persona_ids
                if (pack.personas[pid].tags or {}).get("gender", "").lower()
                == caller_gender.lower()
            ]
            persona_ids = matching or persona_ids
    if len(persona_ids) == 1:
        return persona_ids[0]
    order = language_persona_order(persona_ids, run_seed if run_seed is not None else 0)
    key = persona_rotation_key(task_id)
    if key is not None:
        return order[key % len(order)]
    return order[0]


def require_persona_override(identifier: str) -> None:
    """Assert a ``--user-persona-id`` value names a caller we can speak as.

    The override is defined over exactly three forms: a registered language
    code (``'ko'`` — assign one of that pack's personas per task), a pack
    persona id (``'jihun_ko_v1'``), and a stock English voice persona
    (``'matt_delaney'``). Anything else — a typo, a language whose pack is not
    installed, a pack carrying no personas — raises.

    Raises:
        PersonaResolutionError: If the identifier resolves to no persona. The
            alternative is what this guard exists to prevent: the samplers
            treat an unresolvable id as "no language pack here" and hand the
            run a stock English voice, which is silent data corruption.
    """
    from tau2.data_model.voice_personas import ALL_PERSONAS

    pack = get_language_pack(identifier)  # also forces pack discovery
    if pack is not None:
        if not pack.personas:
            raise PersonaResolutionError(
                f"Language pack '{identifier}' carries no personas, so the run "
                "has no caller to speak as."
            )
        return
    if identifier in ALL_PERSONAS:
        return
    raise PersonaResolutionError(
        f"Unknown user persona '{identifier}'. Expected a registered language "
        f"code ({', '.join(list_language_packs())}), a language-pack persona "
        "id, or a stock voice persona name."
    )


def require_caller_persona(
    user_persona_id: Optional[str],
    task_ids: Iterable[str],
    domain: str,
) -> None:
    """Assert a run can speak as the caller its config and its tasks require.

    Two ways a run silently produces unusable calls, both fatal here:

    1. ``--user-persona-id`` names nothing resolvable (see
       :func:`require_persona_override`).
    2. The tasks come from a localized set (``<domain>_<lang>`` or
       ``<domain>_<lang>_identity``) but the run's persona resolves to a
       different language — or to no language at all, which is exactly what an
       omitted ``--user-persona-id`` looks like. A Korean scenario read by a
       stock English voice is corrupt data that scores and ships like any
       other call.

    Plain English runs are untouched: base task ids belong to no localized set,
    impose no language, and keep the stock-persona sampling they always had.

    Args:
        user_persona_id: The run's ``--user-persona-id``, or None.
        task_ids: The ids of every task the run will simulate.
        domain: The run's benchmark domain — localized set names are
            ``<domain>_<suffix>``, so the language is only readable off a task
            id together with its domain.

    Raises:
        PersonaResolutionError: On either condition above.
    """
    if user_persona_id is not None:
        require_persona_override(user_persona_id)
    run_language = (
        resolve_run_language(user_persona_id) if user_persona_id is not None else None
    )

    from tau2.multilingual.english_prompts import task_language

    mismatched: dict[str, list[str]] = {}
    for task_id in task_ids:
        language = task_language(task_id, domain)
        if language is not None and language != run_language:
            mismatched.setdefault(language, []).append(task_id)
    if not mismatched:
        return

    summary = "; ".join(
        f"{language}: {len(ids)} task(s) e.g. {ids[0]}"
        for language, ids in sorted(mismatched.items())
    )
    configured = (
        f"--user-persona-id '{user_persona_id}' (language "
        f"{run_language or 'none — a stock English voice'})"
        if user_persona_id is not None
        else "no --user-persona-id, so a stock English voice"
    )
    raise PersonaResolutionError(
        f"Localized tasks require a caller persona this run does not have. "
        f"The run is configured with {configured}, but its tasks belong to "
        f"localized task sets of other languages ({summary}). Re-run with "
        f"--user-persona-id <language code> matching the task set."
    )


def resolve_run_language(identifier: str) -> Optional[str]:
    """Resolve a persona-override identifier to a run language code.

    The ``--user-persona-id`` override accepts either a pack persona id
    (e.g. 'rishika_hindi_v1') or a bare language code (e.g. 'hi', meaning
    "sample among that pack's personas per task"). Returns the pack's
    language for both forms, or None for anything else (plain English
    personas and unknown ids — English behavior).
    """
    multilingual = get_multilingual_persona(identifier)
    if multilingual is not None:
        return multilingual[0].language
    pack = get_language_pack(identifier)
    if pack is not None:
        return pack.language
    return None


def _ensure_loaded() -> None:
    from tau2.multilingual.loader import load_language_packs

    load_language_packs()

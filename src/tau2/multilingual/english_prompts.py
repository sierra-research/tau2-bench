# Copyright Sierra
"""English prompts for multilingual runs — how every multilingual run works.

The user simulator reads ENGLISH global guidelines and ENGLISH task
instructions, plus a fixed directive that every word spoken to the agent must
be in the target language. Rationale: translating the prompt measurably
degrades instruction-following (first-turn ``###STOP###`` in 15/16 languages vs
zero in English; progressive-disclosure salience loss), while the speech itself
stays native. The alternative — the pack's localized guidelines plus translated
task instructions — is the retired native arm, quarantined in
deleted 2026-08-03; nothing selects it at runtime.

Two pieces live here:

- :func:`render_target_language_directive` — the fixed, versioned directive
  block appended to the English guidelines (prompt text is calibrated material;
  bump the version constant on any change).
- :func:`english_user_task_variant` — the English task instructions for a
  LOCALIZED task, with entity localization preserved: identity-variant sets
  rename the caller to a locale identity via ``initialization_data`` (see
  ``tau2.multilingual.invariants``), and that rename is re-applied to the
  English source prose so the sim still presents the locale name/user_id/email.
  Everything else on the task (id, initial_state, evaluation_criteria) is
  untouched — only the user-sim scenario changes.
"""

from functools import lru_cache
from typing import Optional

from tau2.data_model.tasks import Task, UserScenario
from tau2.multilingual.domain_profiles import CallerIdentityKind, get_domain_profile
from tau2.multilingual.invariants import (
    AddressProseFormat,
    IdentityRename,
    identity_rename,
)
from tau2.multilingual.names import NameOrder, name_order_for_language
from tau2.multilingual.native_script import is_identity_task_id
from tau2.multilingual.registry import get_language_pack
from tau2.multilingual.task_sets import discover_localized_task_files

# ---------------------------------------------------------------------------
# Target-language directive (fixed, versioned prompt block)
# ---------------------------------------------------------------------------

TARGET_LANGUAGE_DIRECTIVE_VERSION = "v3"

# Appended to the English guidelines for every multilingual persona.
# Calibrated prompt material: never edit in place without
# bumping TARGET_LANGUAGE_DIRECTIVE_VERSION. Superseded versions stay
# retrievable below for provenance/comparison (results recorded before the
# bump were produced by them).
TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V1 = """
## LANGUAGE OF THE CALL (MANDATORY)

You are a native {language_name} speaker. These guidelines and your scenario are written in English for precision, but the phone call itself is NOT in English: every word you speak to the agent MUST be in {language_name}.
- Never switch to English because your instructions are in English. Express any scenario wording you need in natural, spoken {language_name} — do not translate literally.
- Concrete values (names, IDs, emails, confirmation codes, flight numbers, dates, amounts) stay exactly as written in your scenario; say them aloud the way a native {language_name} speaker naturally reads them out.
- Special tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) are control signals, not speech: when the guidelines call for one, emit it verbatim, exactly as written.
""".strip()

# v2 adds the three language-GENERIC rules the localized guideline files carry
# natively but the English arm was missing: native orthography, English
# spelling for code-switched words, and the persona-precedence pointer.
# Language-parameterized only — per-language conventions (symbol readouts,
# phone/amount/spelling) live in the pack's localization data, never here.
TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V2 = """
## LANGUAGE OF THE CALL (MANDATORY)

You are a native {language_name} speaker. These guidelines and your scenario are written in English for precision, but the phone call itself is NOT in English: every word you speak to the agent MUST be in {language_name}.
- Never switch to English because your instructions are in English. Express any scenario wording you need in natural, spoken {language_name} — do not translate literally.
- Write {language_name} with its native orthography: the language's own script, accents/diacritics, and language-specific punctuation (e.g. Spanish tildes and opening ¿ ¡), so your words are pronounced correctly.
- If your persona code-switches into English, write those words in their English spelling inside the {language_name} sentence (e.g. "Quiero el refund de mi pedido") — never transliterate them phonetically.
- Concrete values (names, IDs, emails, confirmation codes, flight numbers, dates, amounts) stay exactly as written in your scenario; say them aloud the way a native {language_name} speaker naturally reads them out.
- Your dialect/variety, register, form of address, and how much English you mix come from the persona guidelines in this prompt — always follow the persona. These guidelines say HOW to behave on the call; the persona says WHO you are.
- Special tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) are control signals, not speech: when the guidelines call for one, emit it verbatim, exactly as written.
""".strip()

# v3 implements the owner ruling (2026-08-20) that a prompt naming the
# caller's language names their location/variety too: the opening sentence
# now states WHERE the speaker is from (the resolved persona's locale,
# rendered human-readable) and WHICH variety of the language they speak (the
# per-language pinned variety) — both from the fixed catalogs in
# ``tau2.multilingual.varieties``, the same source the agent-side language
# clause reads. The directive NAMES the variety; the persona guidelines
# (which v2's persona-precedence bullet already points at) elaborate it.
TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V3 = """
## LANGUAGE OF THE CALL (MANDATORY)

You are a native {language_name} speaker from {origin}, and your {language_name} is {variety_name}. These guidelines and your scenario are written in English for precision, but the phone call itself is NOT in English: every word you speak to the agent MUST be in {language_name}.
- Never switch to English because your instructions are in English. Express any scenario wording you need in natural, spoken {language_name} — do not translate literally.
- Write {language_name} with its native orthography: the language's own script, accents/diacritics, and language-specific punctuation (e.g. Spanish tildes and opening ¿ ¡), so your words are pronounced correctly.
- If your persona code-switches into English, write those words in their English spelling inside the {language_name} sentence (e.g. "Quiero el refund de mi pedido") — never transliterate them phonetically.
- Concrete values (names, IDs, emails, confirmation codes, flight numbers, dates, amounts) stay exactly as written in your scenario; say them aloud the way a native {language_name} speaker naturally reads them out.
- Your dialect/variety, register, form of address, and how much English you mix come from the persona guidelines in this prompt — always follow the persona. These guidelines say HOW to behave on the call; the persona says WHO you are.
- Special tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) are control signals, not speech: when the guidelines call for one, emit it verbatim, exactly as written.
""".strip()


def render_target_language_directive(language: str, locale: Optional[str]) -> str:
    """The directive block for one target language + resolved persona locale.

    ``language`` is an ISO 639-1 code with a registered pack; the pack's
    ``display_name`` (e.g. 'German') parameterizes the fixed template.
    ``locale`` is the RESOLVED persona's locale code (e.g. 'ES-MD'): the v3
    template names the speaker's origin and the language's pinned variety, so
    it renders from the run's concrete caller, not a pack-wide guess.

    Raises:
        ValueError: If the language has no registered pack, the locale is
            missing, or either code is absent from the variety catalogs
            (``tau2.multilingual.varieties``). Rendering a directive without
            the location/variety pin would silently ship an under-specified
            arm — a run that cannot state who its caller is must die instead.
    """
    from tau2.multilingual.varieties import language_variety_name, locale_display_name

    pack = get_language_pack(language)
    if pack is None:
        raise ValueError(
            f"Target-language directive: '{language}' has no registered "
            "language pack, so the directive cannot name the speaker's "
            "variety and origin."
        )
    if not locale:
        raise ValueError(
            f"Target-language directive: the resolved persona for '{language}' "
            "carries no locale, so the directive cannot name the speaker's "
            "origin. Pack personas must pin a locale."
        )
    return TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V3.format(
        language_name=pack.display_name,
        origin=locale_display_name(locale),
        variety_name=language_variety_name(language),
    )


# ---------------------------------------------------------------------------
# End-of-prompt reminder (fixed, versioned prompt block)
# ---------------------------------------------------------------------------

TARGET_LANGUAGE_REMINDER_VERSION = "v1"

# Appended after the scenario block for a multilingual persona: a terse
# recency anchor restating ONLY the language mandate and the special-token
# contract, not the directive. It renders as-is for every language — the
# retired native arm substituted the pack's translated equivalent here
# (the retired native arm's ``end_reminder``); nothing does
# now. Calibrated prompt material: never edit in place without
# bumping TARGET_LANGUAGE_REMINDER_VERSION.
TARGET_LANGUAGE_REMINDER_TEMPLATE_V1 = """
## FINAL REMINDER
Every word you speak to the agent is in {language_name} — never in English.
Special tokens (###STOP###, ###TRANSFER###, ###OUT-OF-SCOPE###) stay verbatim, in ASCII, exactly as written.
""".strip()


def render_target_language_reminder(language: str) -> str:
    """The end-of-prompt reminder for one target language.

    Same resolution as :func:`render_target_language_directive`: the pack's
    ``display_name`` parameterizes the fixed template, with the bare code as
    fallback for unregistered languages.
    """
    pack = get_language_pack(language)
    language_name = pack.display_name if pack is not None else language
    return TARGET_LANGUAGE_REMINDER_TEMPLATE_V1.format(language_name=language_name)


# ---------------------------------------------------------------------------
# English task instructions (entity localization preserved)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_domain_db(domain: str) -> Optional[dict]:
    """The domain's raw DB dict (for identity-rename resolution), or None.

    Delegates to ``localize_lib.load_domain_db`` (the single owner of domain-DB
    loading, JSON/TOML probing included), cached per domain (the DB can be
    large and this runs per task swap); consumers treat the returned dict as
    READ-ONLY — ``identity_rename`` only scans it.

    Identity-variant localized tasks patch the caller identity via
    ``initialization_data``; resolving the OLD identity (name/email/phone) requires
    the domain DB, exactly as the localization invariants do
    (``tau2.multilingual.invariants.identity_rename``). A domain without a
    JSON/TOML DB file simply gets no rename support (plain localized sets do
    not need it: their entities are byte-identical to the English source).
    """
    from tau2.multilingual.localize_lib import load_domain_db

    try:
        return load_domain_db(domain)
    except FileNotFoundError:
        return None


def _caller_identity_kind(domain: str) -> CallerIdentityKind:
    """How ``domain`` anchors its caller — the shape the prose swap needs.

    An unprofiled domain (test fixtures) gets the handle shape: the literal
    string pairs only, which is what every domain did before the prose-anchored
    shape existed.
    """
    try:
        return get_domain_profile(domain).caller_identity
    except KeyError:
        return CallerIdentityKind.USER_ID_HANDLE


def _renamed(
    text: Optional[str],
    rename: Optional[IdentityRename],
    caller_identity: CallerIdentityKind,
    address_format: Optional[AddressProseFormat] = None,
) -> Optional[str]:
    """``text`` with the identity rename applied for this caller shape.

    Thin wrapper over :meth:`IdentityRename.apply_to_prose` — the swap itself
    is shared with the identity-set emitter, whose committed prose this must
    reproduce exactly (``_check_scenario_consistency`` compares them).
    """
    if text is None or rename is None:
        return text
    return rename.apply_to_prose(text, caller_identity, address_format)


def _english_scenario(
    localized_task: Task,
    source_task: Task,
    domain_db: Optional[dict],
    caller_identity: CallerIdentityKind,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
    address_format: Optional[AddressProseFormat] = None,
) -> UserScenario:
    """The English-source user scenario with the caller rename re-applied.

    For plain localized tasks the rename is None (entities are byte-identical
    to the source by the localization invariants) and the source scenario is
    returned as-is. For identity-variant tasks the locale identity (full name,
    user_id, email) replaces the source caller's in every prose field, so
    English instruction text never undoes entity localization.

    ``name_order`` must be the LOCALE's: the English instructions still carry
    the locale caller's name, and writing it given-first here would hand the
    user simulator a different name from the one in the agent's record.
    """
    rename = identity_rename(
        source_task.model_dump(),
        localized_task.model_dump(),
        domain_db,
        name_order,
    )
    scenario = source_task.user_scenario.model_copy(deep=True)
    scenario.persona = _renamed(
        scenario.persona, rename, caller_identity, address_format
    )
    instructions = scenario.instructions
    if isinstance(instructions, str):
        scenario.instructions = _renamed(
            instructions, rename, caller_identity, address_format
        )
        return scenario
    for field in ("reason_for_call", "known_info", "unknown_info", "task_instructions"):
        setattr(
            instructions,
            field,
            _renamed(
                getattr(instructions, field),
                rename,
                caller_identity,
                address_format,
            ),
        )
    return scenario


def english_user_task_variant(
    task: Task,
    *,
    domain: str,
    task_set_name: Optional[str],
) -> Task:
    """The task with its user scenario swapped to English (entities localized).

    Given a task from a LOCALIZED task set (``<domain>_<suffix>``, registered
    by filename convention), returns a copy whose ``user_scenario`` is the
    English source task's — with the identity-variant caller rename re-applied
    (see :func:`_english_scenario`). Every other field (id, initial_state,
    evaluation_criteria) is the localized task's, so the environment and the
    evaluator are untouched; only what the USER SIMULATOR reads changes.

    Tasks that are not from a localized set (base English runs, or an English
    task set paired with a multilingual persona) pass through unchanged —
    their instructions are already English.

    Raises:
        ValueError: If the localized task has no English source task — English
            prompt mode cannot run it, and silently keeping the translated
            instructions would corrupt the arm.
    """
    if task_set_name is None:
        # A run handed a Task directly (no task_set_name on the config) still
        # needs the swap when the task came from a localized set; the id's
        # ``_<suffix>`` tail identifies it.
        task_set_name = _infer_localized_task_set(task.id, domain)
    if task_set_name is None or task_set_name == domain:
        return task
    if task_set_name not in discover_localized_task_files():
        return task
    suffix = task_set_name.removeprefix(f"{domain}_")
    source_id = task.id.removesuffix(f"_{suffix}")
    source_task = next(
        (t for t in _load_source_tasks(domain) if t.id == source_id), None
    )
    if source_id == task.id or source_task is None:
        raise ValueError(
            f"English prompts: localized task '{task.id}' (task set "
            f"'{task_set_name}') has no English source task '{source_id}' in "
            f"domain '{domain}' — cannot build English task instructions."
        )
    language = task_language(task.id, domain)
    address_format = None
    if language and is_identity_task_id(task.id):
        from tau2.multilingual.factory.entity_localization import load_corpus

        address_format = load_corpus(language).address_format
    english_scenario = _english_scenario(
        task,
        source_task,
        _load_domain_db(domain),
        _caller_identity_kind(domain),
        name_order_for_language(language),
        address_format,
    )
    _check_scenario_consistency(task, english_scenario, domain)
    return task.model_copy(update={"user_scenario": english_scenario})


def _infer_localized_task_set(task_id: str, domain: str) -> Optional[str]:
    """The domain's localized task set a task id belongs to, or None.

    Matches the id's ``_<suffix>`` tail against the discovered localized task
    sets (filename convention). The longest suffix wins so identity variants
    (``3_de_identity`` -> ``airline_de_identity``) beat their plain sets
    (``airline_de``). Base English ids match nothing and pass through.
    """
    best_suffix: Optional[str] = None
    for name in discover_localized_task_files():
        if not name.startswith(f"{domain}_"):
            continue
        suffix = name.removeprefix(f"{domain}_")
        if task_id.endswith(f"_{suffix}"):
            if best_suffix is None or len(suffix) > len(best_suffix):
                best_suffix = suffix
    return f"{domain}_{best_suffix}" if best_suffix else None


def task_language(task_id: str, domain: str) -> Optional[str]:
    """The ISO 639-1 language a localized task id belongs to, or None.

    A localized set is named ``<domain>_<lang>`` or ``<domain>_<lang>_identity``,
    so the language is the first token of the matched suffix. Base English ids
    (the un-localized ``tasks.json`` sets) match no localized set and return
    None — callers decide what that means (usually English).
    """
    task_set = _infer_localized_task_set(task_id, domain)
    if task_set is None:
        return None
    return task_set.removeprefix(f"{domain}_").split("_")[0]


@lru_cache(maxsize=None)
def _load_source_tasks(domain: str) -> tuple[Task, ...]:
    """The domain's multilingual SOURCE task set (English), by profile.

    The localized sets were generated from the domain profile's
    ``source_tasks_filename`` — for a pool domain (telecom) that is the
    diversified seed split ``tasks_multilingual.json``, NOT the registry
    default ``tasks.json``. Loading the registry default here once swapped
    seed scenarios for their same-id pool ancestors (phone-auth known_info,
    ``unknown_info`` silently dropped), corrupting every english-prompt-mode
    name-auth sim. Domains without a profile fall back to the registry
    loader.

    Cached per domain (loading re-parses the whole task file, and the swap
    runs once per simulated task); returned as a tuple, and the Task models
    are READ-ONLY — the swap ``model_copy``s before changing anything.
    """
    from tau2.multilingual.localize_lib import data_dir
    from tau2.utils import load_file

    try:
        profile = get_domain_profile(domain)
    except KeyError:
        from tau2.registry import registry

        return tuple(registry.get_tasks_loader(domain)())
    path = data_dir() / "tau2" / "domains" / domain / profile.source_tasks_filename
    tasks = load_file(path)
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]
    return tuple(Task.model_validate(task) for task in tasks)


def _check_scenario_consistency(
    localized_task: Task, english_scenario: UserScenario, domain: str
) -> None:
    """Fail loud when the English swap disagrees with the localized scenario.

    The swap must never change what the caller knows — only the language it
    is written in. ``unknown_info`` is the tripwire: both sides carry it or
    neither does, and for a domain whose task prose is NOT translated
    (telecom) the whole scenario is byte-identical to the source after the
    identity rename, so the fields must match exactly. A mismatch means the
    source lookup resolved the wrong task variant — raising here turns a
    silently corrupted arm into a failed run.
    """
    localized = localized_task.user_scenario.instructions
    english = english_scenario.instructions
    if isinstance(localized, str) or isinstance(english, str):
        return
    if (localized.unknown_info is None) != (english.unknown_info is None):
        raise ValueError(
            f"English prompts: task '{localized_task.id}' "
            f"unknown_info mismatch — localized carries "
            f"{localized.unknown_info!r} but the English source scenario "
            f"carries {english.unknown_info!r}. The English swap resolved "
            "the wrong source task variant."
        )
    if get_domain_profile(domain).tasks_translated:
        return
    for field in ("known_info", "unknown_info"):
        if getattr(localized, field) != getattr(english, field):
            raise ValueError(
                f"English prompts: task '{localized_task.id}' "
                f"{field} mismatch on untranslated domain '{domain}' — "
                f"localized {getattr(localized, field)!r} != English source "
                f"{getattr(english, field)!r}. The English swap resolved the "
                "wrong source task variant."
            )

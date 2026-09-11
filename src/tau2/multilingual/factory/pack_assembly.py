# Copyright Sierra
"""Deterministic, code-generated pieces of a drafted language pack.

Generates the mechanical/templated pack fields in code from the
:class:`AuthorForm` + language identity + CLI args, so drafting need not ask
the model for them. What is deterministic here:

- **the ``experiments`` entries** — ``preset_name`` is
  ``multilingual_v1_<main_arm_name>``, ``task_set_suffix`` is the language
  code, ``domain`` comes from the CLI; ``main_arm_name`` derives from the
  display name (form may override) with non-default domains suffixed
  ``_<domain>``, and ``smoke_task_stem`` from the form/CLI/default.
- **``agent_language_clause``** — templated from the display name + script name.
- **the firing rates** (``default_out_of_turn_events_per_minute``) —
  language-level defaults, overridable by the form.
- **the ``acoustic_presets`` scaffolding** — ids/display_names/bed-path
  structure; continuous beds are namespaced under ``<lang>/<file>.wav`` while
  the office bed and bursts reuse the shared top-level English files.

The result is a partial pack dict the drafter MERGES UNDER the creative LLM
output (creative content wins on any key these functions do not own).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN
from tau2.multilingual.factory.paths import (
    OUTDOOR_BED_BASENAME,
    SHARED_OFFICE_BED_BASENAME,
    TV_KITCHEN_BED_BASENAME,
)

if TYPE_CHECKING:
    # Type-annotation only (form attributes are read by duck typing at
    # runtime).
    from tau2.multilingual.factory.author_form import AuthorForm

# The one domain whose experiments keep the BARE language arm name (no
# ``_<domain>`` suffix). Frozen at "airline" — the first-shipped domain named
# its presets before multi-domain packs existed, and renaming them would churn
# every shipped pack, preset, and save path. This is a NAMING legacy, not the
# authoring default above.
UNSUFFIXED_ARM_DOMAIN = "airline"

# Language-level firing-rate defaults (overridable by the form). Chosen to sit
# in the band the shipped packs use (hi 1.0).
DEFAULT_OUT_OF_TURN_EVENTS_PER_MINUTE = 1.0

# Human-readable script names for the agent_language_clause, keyed by ISO 15924
# code (lowercased). Unknown scripts fall back to a generic "the native script"
# phrasing so a new language still produces a sensible clause.
_SCRIPT_DISPLAY_NAMES: dict[str, str] = {
    "deva": "Devanagari script",
    "hans": "Simplified Chinese characters",
    "hant": "Traditional Chinese characters",
    "latn": "Latin script",
    "arab": "Arabic script",
    "cyrl": "Cyrillic script",
    "beng": "Bengali script",
    "gujr": "Gujarati script",
    "guru": "Gurmukhi script",
    "taml": "Tamil script",
    "telu": "Telugu script",
    "knda": "Kannada script",
    "mlym": "Malayalam script",
    "thai": "Thai script",
    "hang": "Hangul",
    "jpan": "Japanese script",
    "grek": "Greek script",
    "hebr": "Hebrew script",
}

_FIRST_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# The headline-experiment preset-naming convention. SINGLE SOURCE OF TRUTH:
# both the deterministic generator (build_experiment_block) and the guardrail
# (guardrails._check_experiment_block) derive the expected preset_name through
# this helper, so the convention is defined exactly once and they cannot drift.
PRESET_NAME_PREFIX = "multilingual_v1_"


def preset_name_for(main_arm_name: str) -> str:
    """The experiment ``preset_name`` for a given ``main_arm_name``.

    The convention is ``multilingual_v1_<main_arm_name>`` and the registry
    presets key on it. Defined here once so the generator and the guardrail
    share it rather than each hardcoding the pattern string.
    """
    return f"{PRESET_NAME_PREFIX}{main_arm_name}"


def derive_main_arm_name(form: AuthorForm) -> str:
    """The short arm name for the language's DEFAULT-domain experiment.

    Derived from the display name's first alphanumeric token, lowercased
    ("Mandarin Chinese" -> "mandarin", "Hindi" -> "hindi"). This is a NAMING
    convention, not a decision, so it is deterministic — but an author can
    override it via the typed ``main_arm_name`` form field.
    """
    if form.main_arm_name:
        return form.main_arm_name.strip()
    match = _FIRST_TOKEN_RE.search(form.display_name)
    token = match.group(0).lower() if match else form.language
    return token or form.language


def main_arm_name_for(base_arm_name: str, domain: str) -> str:
    """The main arm name for one domain's experiments entry.

    The legacy first domain (airline) keeps the bare language arm name (no
    rename churn for the shipped presets); every other domain is suffixed
    ``_<domain>`` so its preset (``multilingual_v1_<arm>``) and save paths are
    distinct. Shared by the generator and the guardrail so the convention
    lives once.
    """
    if domain == UNSUFFIXED_ARM_DOMAIN:
        return base_arm_name
    return f"{base_arm_name}_{domain}"


def resolve_domain(form: AuthorForm, domain: Optional[str] = None) -> str:
    """The experiment domain: CLI arg > typed form override > documented default.

    Shared by ``build_experiment_block`` and ``run_draft`` (which records it on
    the ``FactoryProject``) so the precedence is defined exactly once.
    """
    return domain or form.domain or DEFAULT_MULTILINGUAL_DOMAIN


def default_smoke_task_stem_for(domain: str) -> str:
    """The domain's documented default smoke stem, from its profile.

    Raises :class:`FactoryDraftError` for an unprofiled domain — a stem cannot
    be invented, so drafting such a domain must pass one explicitly.
    """
    from tau2.multilingual.domain_profiles import get_domain_profile
    from tau2.multilingual.factory.author_form import FactoryDraftError

    try:
        return get_domain_profile(domain).default_smoke_task_stem
    except KeyError as exc:
        raise FactoryDraftError(
            f"domain '{domain}' has no profile (and so no default smoke "
            "stem) — pass --smoke-task-stem explicitly"
        ) from exc


def script_display_name(script: str) -> str:
    """Human-readable script name for the agent_language_clause."""
    return _SCRIPT_DISPLAY_NAMES.get((script or "").lower(), "the native script")


def build_agent_language_clause(form: AuthorForm) -> str:
    """Template the agent-side language clause from display name + script.

    Mirrors the hi/ro shipped variants: a script clause (which script for
    the matrix language, Latin for English insertions) plus the constant
    "mirror the user's code-switching register / keep tool calls in English"
    tail. Deterministic, so the model never re-authors it.
    """
    script_name = script_display_name(form.script)
    return (
        f"Respond in {form.display_name} using {script_name} for "
        f"{form.display_name} and Latin script for English insertions; mirror "
        f"the user's code-switching register; keep all tool calls, arguments, "
        f"and structured outputs in English exactly as specified."
    )


def build_experiment_block(
    form: AuthorForm,
    *,
    domain: Optional[str] = None,
    smoke_task_stem: Optional[str] = None,
) -> dict[str, Any]:
    """One deterministic ``experiments`` entry.

    ``preset_name``/``task_set_suffix``/``main_arm_name``/``domain`` are
    derived; ``smoke_task_stem`` is a genuine decision sourced from the args
    / form-override / a documented default (never invented). ``description`` is
    a neutral, derivable summary — the model is no longer asked for it. The
    shared English baseline is its own pack/preset, so there is no per-language
    baseline persona here; language results are compared against it post-hoc.
    """
    resolved_domain = resolve_domain(form, domain)
    main_arm = main_arm_name_for(derive_main_arm_name(form), resolved_domain)
    resolved_smoke = (
        smoke_task_stem
        or form.smoke_task_stem
        or default_smoke_task_stem_for(resolved_domain)
    )
    return {
        "preset_name": preset_name_for(main_arm),
        "description": (
            f"{form.display_name} v1 headline experiment: the pack's personas "
            f"on the {resolved_domain} tasks."
        ),
        "domain": resolved_domain,
        "main_arm_name": main_arm,
        "task_set_suffix": form.language,
        "smoke_task_stem": str(resolved_smoke),
    }


def build_acoustic_presets(form: AuthorForm) -> dict[str, dict[str, Any]]:
    """The deterministic ``acoustic_presets`` scaffolding.

    Empty unless the form opts into locale beds (``locale_beds: true``) — the
    benchmark default is the shared English environments, selected at run
    time by the stock indoor/outdoor logic in ``sample_voice_config``, so a
    pack without presets needs no acoustic scaffolding at all.

    When opted in: three presets matching the hi shape — an outdoor-traffic
    bed and a multi-generational-household bed (both locale-flavored
    continuous beds, namespaced under ``<lang>/`` with the SAME base filename
    the per-locale generator writes), and a generic office bed that reuses
    the shared top-level English ambience. Bursts always reuse the shared
    English files. Ids/display_names are structural; the model never
    free-authors them.
    """
    if not form.locale_beds:
        return {}
    lang = form.language
    return {
        f"{lang}_outdoor_traffic": {
            "id": f"{lang}_outdoor_traffic",
            "display_name": f"{form.display_name} Outdoor Traffic",
            "background_noise_files": [f"{lang}/{OUTDOOR_BED_BASENAME}"],
            "burst_noise_files": ["car_horn.wav", "engine_idling.wav"],
        },
        f"{lang}_office": {
            "id": f"{lang}_office",
            "display_name": f"{form.display_name} Office",
            "background_noise_files": [SHARED_OFFICE_BED_BASENAME],
            "burst_noise_files": [],
        },
        f"{lang}_household_multi_gen": {
            "id": f"{lang}_household_multi_gen",
            "display_name": f"{form.display_name} Multi-Generational Household",
            "background_noise_files": [f"{lang}/{TV_KITCHEN_BED_BASENAME}"],
            "burst_noise_files": ["dog_bark.wav"],
        },
    }


def localize_acoustic_bed_paths(
    presets: dict[str, dict[str, Any]], language: str, locale_dir: str
) -> dict[str, dict[str, Any]]:
    """Re-namespace the locale beds from ``<lang>/`` to ``<locale_dir>/`` in place.

    :func:`build_acoustic_presets` namespaces the two generated locale beds under
    the bare language code because persona locales aren't known yet at that point.
    Once personas are drafted, the assembler calls this with the derived
    ``language_COUNTRY`` dir (see ``paths.derive_locale_dir``) so the
    pack's ``background_noise_files`` match where ``generate-assets`` writes the
    wavs (e.g. ``es/busy_street...`` -> ``es_ES/busy_street...``). Shared English
    beds (office) and bursts have no ``<lang>/`` prefix and are left untouched.
    No-op when ``locale_dir`` equals the bare language code.
    """
    if locale_dir == language:
        return presets
    prefix = f"{language}/"
    for preset in presets.values():
        files = preset.get("background_noise_files") or []
        preset["background_noise_files"] = [
            f"{locale_dir}/{str(f).rsplit('/', 1)[-1]}"
            if str(f).startswith(prefix)
            else f
            for f in files
        ]
    return presets


def _is_locale_bed(preset: dict[str, Any]) -> bool:
    """True if a preset's continuous bed is a locale-specific generated bed.

    Locale beds (outdoor / household) are namespaced under a subfolder, so their
    path contains ``/``; the shared English office bed (``people_talking.wav``)
    and bursts are bare filenames.
    """
    files = preset.get("background_noise_files") or []
    return bool(files) and "/" in str(files[0])


def reassign_personas_to_locale_beds(
    personas: dict[str, dict[str, Any]], acoustic_presets: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Keep personas on the generated LOCALE beds, off the shared-English office bed.

    The draft model assigns each persona an ``acoustic_preset_id`` from the persona
    sketch, and sometimes picks the ``office`` preset — which references the SHARED
    ENGLISH ``people_talking.wav``, not a locale bed. That leaves a generated locale
    bed (outdoor / household) UNUSED and puts that persona on generic English
    ambience instead of locale-flavored audio. When a locale bed would otherwise go
    unused, reassign a persona that's on a shared (non-locale) preset to it, so every
    generated locale bed is used and no persona is stranded on the English office
    bed. Matches the hi/zh pattern (both personas on locale beds, office unused).
    Mutates the persona dicts in place. No-op once all locale beds are covered.
    """
    locale_preset_ids = {
        pid for pid, p in acoustic_presets.items() if _is_locale_bed(p)
    }
    used = {p.get("acoustic_preset_id") for p in personas.values()}
    unused_locale = [
        pid for pid in acoustic_presets if pid in locale_preset_ids and pid not in used
    ]
    if not unused_locale:
        return personas
    for persona in personas.values():
        if not unused_locale:
            break
        current = acoustic_presets.get(persona.get("acoustic_preset_id"))
        if current is not None and not _is_locale_bed(current):
            persona["acoustic_preset_id"] = unused_locale.pop(0)
    return personas


def build_deterministic_pack_fields(
    form: AuthorForm,
    *,
    domain: Optional[str] = None,
    smoke_task_stem: Optional[str] = None,
    out_of_turn_events_per_minute: Optional[float] = None,
) -> dict[str, Any]:
    """All deterministic pack fields, as a partial pack dict to merge in.

    The drafter overlays the creative LLM output ON TOP of this dict (creative
    content wins on keys it owns), so these fields are always present and
    internally consistent regardless of model behavior. ``acoustic_presets``
    here is the canonical scaffolding; the creative call wires each persona's
    ``acoustic_preset_id`` to one of these ids.
    """
    oot = (
        out_of_turn_events_per_minute
        if out_of_turn_events_per_minute is not None
        else form.default_out_of_turn_events_per_minute
    )
    fields = {
        "language": form.language,
        "display_name": form.display_name,
        "agent_language_clause": build_agent_language_clause(form),
        "default_out_of_turn_events_per_minute": (
            oot if oot is not None else DEFAULT_OUT_OF_TURN_EVENTS_PER_MINUTE
        ),
        "acoustic_presets": build_acoustic_presets(form),
        "experiments": [
            build_experiment_block(
                form,
                domain=domain,
                smoke_task_stem=smoke_task_stem,
            )
        ],
    }
    if not fields["acoustic_presets"]:  # shared-environment default: no key at all
        del fields["acoustic_presets"]
    return fields


def acoustic_preset_ids(form: AuthorForm) -> list[str]:
    """The deterministic preset ids, for prompting the creative call."""
    return list(build_acoustic_presets(form))


# =============================================================================
# `tau2 factory add-experiment` — append one domain's experiments entry to a
# SHIPPED pack.yaml (the run-17-times step of opening a new domain, hence a
# verb). Deterministic and idempotent: the entry is code-generated from the
# pack's own identity + the naming conventions above; a pack that already has
# the domain is skipped.
# =============================================================================


class AddExperimentOutcome(BaseModel):
    """What one ``tau2 factory add-experiment`` invocation did."""

    language: str
    domain: str
    pack_path: Path
    written: bool = Field(
        description="False when the pack already had the domain (skip)."
    )
    preset_name: str
    main_arm_name: str
    original: str = Field(
        default="",
        description="pack.yaml text before the write — lets the caller roll "
        "the file back if post-write guardrails reject the merged pack.",
    )


def _derive_base_arm_name(display_name: str, language: str) -> str:
    """The bare arm name from a shipped pack's display name (same convention
    as :func:`derive_main_arm_name`, without an AuthorForm)."""
    match = _FIRST_TOKEN_RE.search(display_name)
    token = match.group(0).lower() if match else language
    return token or language


def _smoke_stem_problem(domain: str, lang: str, stem: str) -> Optional[str]:
    """A problem string when the smoke stem names no task in the arm the
    preset would run; None when it does — or when no arm file is on disk yet
    (preset generation fails loudly on a missing set, so nothing to check).
    """
    import json

    from tau2.multilingual.run_presets import resolve_main_task_suffix
    from tau2.multilingual.task_sets import discover_localized_task_files

    task_files = discover_localized_task_files()
    suffix = resolve_main_task_suffix(domain, lang, task_files=task_files)
    path = task_files.get(f"{domain}_{suffix}")
    if path is None:
        return None
    with open(path) as fp:
        task_ids = {task["id"] for task in json.load(fp)}
    smoke_id = f"{stem}_{suffix}"
    if smoke_id not in task_ids:
        return (
            f"smoke task '{smoke_id}' (from stem '{stem}') is not a task id "
            f"in {path.name} — pick one task id from the {domain} arm, "
            "without the language suffix"
        )
    return None


def run_add_experiment(
    lang: str,
    domain: str,
    *,
    smoke_task_stem: Optional[str] = None,
) -> "AddExperimentOutcome":
    """Append the ``experiments`` entry for one domain to a shipped pack.

    The entry is deterministic: the base arm name comes from the pack's
    bare-named (airline) entry when present (so 'hindi' begets
    'hindi_telecom'), else from the display name; ``preset_name``/
    ``main_arm_name`` follow the shared naming helpers. ``smoke_task_stem``
    defaults to the domain profile's stem; pass one explicitly to override.

    The write is append-only INSIDE the ``experiments`` block: every other
    line of pack.yaml stays byte-identical. The merged pack is re-validated
    (each entry as ``ExperimentSpec`` + the whole file as ``LanguagePack``)
    before anything is written. Idempotent: an existing entry for the domain
    skips without writing.
    """
    import yaml as _yaml

    from tau2.multilingual.domain_profiles import get_domain_profile
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.backfill_lib import (
        load_shipped_pack,
        validate_merged_pack,
    )
    from tau2.multilingual.schema import ExperimentSpec

    try:
        get_domain_profile(domain)
    except KeyError as exc:
        raise FactoryDraftError(str(exc)) from exc

    pack = load_shipped_pack(lang)
    raw = _yaml.safe_load(pack.original)
    existing_entries = raw.get("experiments") or []
    specs = [ExperimentSpec.model_validate(entry) for entry in existing_entries]

    for spec in specs:
        if spec.domain == domain:
            return AddExperimentOutcome(
                language=lang,
                domain=domain,
                pack_path=pack.pack_path,
                written=False,
                preset_name=spec.preset_name,
                main_arm_name=spec.main_arm_name,
            )

    unsuffixed_spec = next(
        (s for s in specs if s.domain == UNSUFFIXED_ARM_DOMAIN), None
    )
    base_arm = (
        unsuffixed_spec.main_arm_name
        if unsuffixed_spec is not None
        else _derive_base_arm_name(pack.display_name, lang)
    )
    main_arm = main_arm_name_for(base_arm, domain)

    if smoke_task_stem is None:
        smoke_task_stem = default_smoke_task_stem_for(domain)

    smoke_problem = _smoke_stem_problem(domain, lang, str(smoke_task_stem))
    if smoke_problem is not None:
        raise FactoryDraftError(smoke_problem)

    entry = {
        "preset_name": preset_name_for(main_arm),
        "description": (
            f"{pack.display_name} v1 {domain} experiment: the pack's "
            f"personas on the {domain} tasks."
        ),
        "domain": domain,
        "main_arm_name": main_arm,
        "task_set_suffix": lang,
        "smoke_task_stem": str(smoke_task_stem),
    }
    ExperimentSpec.model_validate(entry)
    rendered = _yaml.safe_dump([entry], sort_keys=False, allow_unicode=True)

    lines = pack.original.splitlines(keepends=True)
    key_index = next(
        (
            i
            for i, line in enumerate(lines)
            # Tolerant block-style match (trailing spaces/comments): missing
            # the key here would send us to the append path, and a SECOND
            # top-level 'experiments:' key wins YAML last-key-wins — the
            # existing entries would be silently dropped on every load.
            if re.match(r"^experiments:\s*(#.*)?$", line)
        ),
        None,
    )
    if key_index is None:
        if "experiments" in raw:
            raise FactoryDraftError(
                "pack.yaml carries an 'experiments' key this verb cannot "
                "locate as a block-style 'experiments:' line (flow-style "
                "list?). Appending would write a duplicate top-level key and "
                "the existing entries would be silently dropped on load — "
                "re-format the key as a block-style list first."
            )
        base_text = pack.original.rstrip("\n") + "\n"
        new_text = f"{base_text}experiments:\n{rendered}"
    else:
        end = key_index + 1
        while end < len(lines):
            line = lines[end]
            if line.strip() and not line.startswith(("- ", "  ")):
                break
            if not line.strip():
                # Blank line: part of the block only if list content resumes.
                lookahead = end
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead >= len(lines) or not lines[lookahead].startswith(
                    ("- ", "  ")
                ):
                    break
            end += 1
        # Insert before any trailing blank lines inside the block bound.
        while end > key_index + 1 and not lines[end - 1].strip():
            end -= 1
        new_text = "".join(lines[:end]) + rendered + "".join(lines[end:])

    top_level_keys = sum(
        1 for line in new_text.splitlines() if line.startswith("experiments:")
    )
    if top_level_keys != 1:
        raise FactoryDraftError(
            f"merged pack would carry {top_level_keys} top-level "
            "'experiments:' keys — refusing to write (YAML last-key-wins "
            "would silently drop all but the last)."
        )
    merged_entries = _yaml.safe_load(new_text).get("experiments") or []
    seen: set[str] = set()
    for merged_entry in merged_entries:
        merged_spec = ExperimentSpec.model_validate(merged_entry)
        if merged_spec.domain in seen:
            raise FactoryDraftError(
                f"merged pack would carry duplicate experiments entries for "
                f"domain '{merged_spec.domain}'"
            )
        seen.add(merged_spec.domain)
    validate_merged_pack(new_text, pack.pack_path.parent)

    pack.pack_path.write_text(new_text)
    return AddExperimentOutcome(
        language=lang,
        domain=domain,
        pack_path=pack.pack_path,
        written=True,
        preset_name=entry["preset_name"],
        main_arm_name=main_arm,
        original=pack.original,
    )

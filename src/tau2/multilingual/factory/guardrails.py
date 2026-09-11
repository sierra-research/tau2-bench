# Copyright Sierra
"""Deterministic guardrails for LLM-drafted language packs.

A factory draft is model output, so before anything downstream consumes it we
re-run every rule the runtime would hit — schema validation with the loader's
exact normalization, the backchannel-prompt format contract, the
count and brevity bounds on both short-phrase inventories (backchannel
continuers, out-of-turn speech), script-range coverage, noise-file existence,
tag vocabulary and completeness, persona display_name ↔ tags.gender
consistency — and report ALL problems at once as human-readable strings
instead of raising.

``validate_pack_draft`` checks a ``pack_draft.yaml`` inside a factory
workspace folder; ``validate_final_pack`` runs the identical checks against a
real ``data/tau2/multilingual/<lang>/pack.yaml``.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

from tau2.multilingual.brevity import (
    BACKCHANNEL_INVENTORY,
    NON_DIRECTED_INVENTORY,
    phrase_inventory_problems,
)
from tau2.multilingual.factory.name_genders import known_given_name_gender
from tau2.multilingual.factory.pack_assembly import (
    UNSUFFIXED_ARM_DOMAIN,
    preset_name_for,
)
from tau2.multilingual.factory.state import PACK_DRAFT_FILENAME
from tau2.multilingual.invariants import SCRIPT_RANGES
from tau2.multilingual.loader import normalize_pack_data
from tau2.multilingual.localize_lib import multilingual_data_dir
from tau2.multilingual.schema import ExperimentSpec, LanguagePack
from tau2.multilingual.tags import (
    REQUIRED_TAG_DIMENSIONS,
    TAG_VOCABULARY,
)
from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR, BURST_NOISE_DIR

# How many available noise files to list when a referenced one is missing.
MAX_LISTED_NOISE_FILES = 30


class GuardrailReport(BaseModel):
    """Outcome of a guardrail run: pass/fail plus every problem found.

    ``problems`` gate the draft (non-empty => not ok). ``notes`` is a separate
    non-fatal tier for expected-pending state (e.g. an opted-in pack's locale
    beds awaiting `the retired locale-bed generator`) and never affects
    ``ok``.
    """

    ok: bool = Field(description="True iff no problems were found")
    problems: list[str] = Field(
        default_factory=list,
        description="Human-readable problems, in check order (empty == pass)",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Non-fatal informational notes (do NOT affect ok): "
        "expected-pending state such as a locale bed a pack declares but does "
        "not ship.",
    )

    @classmethod
    def from_problems(
        cls, problems: list[str], notes: Optional[list[str]] = None
    ) -> "GuardrailReport":
        return cls(ok=not problems, problems=problems, notes=notes or [])


def validate_pack_draft(pack_dir: Path) -> GuardrailReport:
    """Run all guardrails over ``<pack_dir>/pack_draft.yaml``."""
    return _validate_pack_file(Path(pack_dir) / PACK_DRAFT_FILENAME, draft=True)


def validate_final_pack(lang: str) -> GuardrailReport:
    """Run the same guardrails over the real ``<lang>/pack.yaml``."""
    return _validate_pack_file(
        multilingual_data_dir() / lang / "pack.yaml", draft=False
    )


def _validate_pack_file(pack_path: Path, *, draft: bool = False) -> GuardrailReport:
    problems: list[str] = []

    # ------------------------------------------------------------------
    # 1. The file parses to a mapping with only known top-level keys.
    # ------------------------------------------------------------------
    if not pack_path.exists():
        return GuardrailReport.from_problems([f"pack file does not exist: {pack_path}"])
    try:
        data = yaml.safe_load(pack_path.read_text())
    except yaml.YAMLError as exc:
        return GuardrailReport.from_problems(
            [f"{pack_path.name} is not valid YAML: {exc}"]
        )
    if not isinstance(data, dict):
        return GuardrailReport.from_problems(
            [f"{pack_path.name} is not a YAML mapping"]
        )

    allowed_keys = set(LanguagePack.model_fields) | {"experiments"}
    for key in sorted(set(data) - allowed_keys):
        problems.append(
            f"unknown top-level key '{key}' (allowed: {sorted(allowed_keys)})"
        )

    # ------------------------------------------------------------------
    # 1b. Experiments-list invariants: each entry validates as the typed
    #     ExperimentSpec, preset_name follows the
    #     multilingual_v1_<main_arm_name> convention, task_set_suffix is the
    #     language code (registry presets key on these), domains are unique,
    #     each entry's task set exists on disk, and the pack carries a full
    #     glossary for every domain it has an experiment for. Only checked
    #     when an experiments list is present.
    # ------------------------------------------------------------------
    experiments = data.get("experiments")
    pack_language = data.get("language")
    experiment_notes: list[str] = []
    if experiments is not None and not isinstance(experiments, list):
        problems.append("'experiments' must be a list of experiment entries")
    elif isinstance(experiments, list):
        experiment_problems, experiment_notes = _check_experiments(
            experiments, pack_language, data
        )
        problems.extend(experiment_problems)

    # ------------------------------------------------------------------
    # 2. Loader normalization + schema validation (the SAME
    #    ``loader.normalize_pack_data`` the runtime uses, so the rules can
    #    never drift).
    # ------------------------------------------------------------------
    data = normalize_pack_data(data, pack_path.parent)

    try:
        LanguagePack.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"])
            prefix = f"schema: {location}: " if location else "schema: "
            problems.append(prefix + error["msg"])
    except Exception as exc:  # e.g. malformed persona entries pre-validation
        problems.append(f"schema validation failed: {exc}")

    personas = data.get("personas")
    personas = personas if isinstance(personas, dict) else {}

    # ------------------------------------------------------------------
    # 3. Every persona script has a registered Unicode range.
    # ------------------------------------------------------------------
    for persona_id, persona in personas.items():
        if not isinstance(persona, dict):
            continue
        script = persona.get("script")
        if not script:
            problems.append(
                f"persona '{persona_id}' declares no script "
                "(ISO 15924 code, e.g. 'latn', 'deva')"
            )
        elif script not in SCRIPT_RANGES:
            problems.append(
                f"persona '{persona_id}' uses script '{script}' which has no "
                "registered Unicode range — an engineer must extend "
                "SCRIPT_RANGES in tau2.multilingual.invariants "
                f"(known scripts: {sorted(SCRIPT_RANGES)})"
            )

    # ------------------------------------------------------------------
    # 4a. The two short-phrase inventories: backchannel continuers and
    #     out-of-turn speech. Both are drawn from uniformly at random with
    #     zero context, so both are bounded in COUNT (a long mixed list emits
    #     the wrong thing at a random moment) and in LENGTH (a long entry has
    #     stopped being a continuer / a glance away from the mic). Both bounds
    #     live in tau2.multilingual.brevity.
    # ------------------------------------------------------------------
    for persona_id, persona in personas.items():
        if not isinstance(persona, dict):
            continue
        for inventory in (BACKCHANNEL_INVENTORY, NON_DIRECTED_INVENTORY):
            problems.extend(
                phrase_inventory_problems(
                    persona_id,
                    persona.get(inventory.field),
                    inventory=inventory,
                    language=str(pack_language or ""),
                    script=persona.get("script"),
                )
            )

    # ------------------------------------------------------------------
    # 4b. Every persona's acoustic_preset_id resolves to a preset in the pack.
    #     The schema validator also enforces this, but a clear factory-level
    #     message (and coverage even when schema validation bailed earlier) is
    #     worth the few lines — this is a genuine cross-reference invariant.
    # ------------------------------------------------------------------
    presets = data.get("acoustic_presets")
    presets = presets if isinstance(presets, dict) else {}
    preset_ids = set(presets)
    for persona_id, persona in personas.items():
        if not isinstance(persona, dict):
            continue
        preset_ref = persona.get("acoustic_preset_id")
        if preset_ref is not None and preset_ref not in preset_ids:
            problems.append(
                f"persona '{persona_id}' references acoustic preset "
                f"'{preset_ref}' which is not defined in acoustic_presets "
                f"(defined: {sorted(preset_ids)})"
            )

    # ------------------------------------------------------------------
    # 5. Acoustic preset noise files exist where the runtime resolves them.
    #    Missing LOCALE-namespaced continuous beds are non-fatal (bed
    #    production is retired, so a declared-but-absent locale bed is an
    #    authoring leftover); pending_beds carries them for the caller to
    #    surface as an informational note. Missing top-level/shared files stay
    #    hard errors.
    # ------------------------------------------------------------------
    noise_problems, pending_beds = _check_noise_files(presets)
    problems.extend(noise_problems)

    # ------------------------------------------------------------------
    # 6. Persona tags: in-vocabulary AND complete (factory-level requirement).
    # ------------------------------------------------------------------
    for persona_id, persona in personas.items():
        if not isinstance(persona, dict):
            continue
        tags = persona.get("tags") or {}
        if not isinstance(tags, dict):
            problems.append(f"persona '{persona_id}': tags must be a mapping")
            continue
        for key, value in tags.items():
            if key not in TAG_VOCABULARY:
                problems.append(
                    f"persona '{persona_id}': unknown tag dimension '{key}' "
                    f"(known: {sorted(TAG_VOCABULARY)})"
                )
            elif value not in TAG_VOCABULARY[key]:
                problems.append(
                    f"persona '{persona_id}': tag '{key}' has invalid value "
                    f"'{value}' (allowed: {TAG_VOCABULARY[key]})"
                )
        for dimension in REQUIRED_TAG_DIMENSIONS:
            if dimension not in tags:
                problems.append(
                    f"persona '{persona_id}': missing required tag dimension "
                    f"'{dimension}' (factory requires "
                    f"{REQUIRED_TAG_DIMENSIONS} on every persona)"
                )

    # ------------------------------------------------------------------
    # 6b. display_name ↔ tags.gender consistency — cheap insurance against
    #     Olivia-tagged-male / Noah-tagged-female pack authoring. Advisory by
    #     construction: only names in the closed unambiguous given-name catalog
    #     are checked; a name the catalog does not know passes silently.
    # ------------------------------------------------------------------
    for persona_id, persona in personas.items():
        if not isinstance(persona, dict):
            continue
        tags = persona.get("tags")
        tag_gender = tags.get("gender") if isinstance(tags, dict) else None
        display_name = persona.get("display_name")
        if not display_name or tag_gender not in ("male", "female"):
            continue
        catalog_gender = known_given_name_gender(str(display_name))
        if catalog_gender is not None and catalog_gender != tag_gender:
            problems.append(
                f"persona '{persona_id}': display_name '{display_name}' is a "
                f"known {catalog_gender} given name but tags.gender is "
                f"'{tag_gender}' — fix the name or the tag (or, if the name is "
                "genuinely unisex, remove it from GIVEN_NAME_GENDERS in "
                "tau2.multilingual.factory.name_genders)"
            )

    # ------------------------------------------------------------------
    # 7. Guidelines body carries the <PERSONA_GUIDELINES> slot + ASCII control
    #    tokens. The runtime injects each persona's content into the literal
    #    <PERSONA_GUIDELINES> slot and emits ###STOP###/###TRANSFER###/
    #    ###OUT-OF-SCOPE### control tokens — a guidelines file missing the slot
    #    or with a re-scripted (non-ASCII) control token silently breaks the
    #    run, so these are genuine invariants (safe hard guardrails). Only
    #    checked when the guidelines file resolved + exists (the schema
    #    validator already reports a missing file).
    # ------------------------------------------------------------------
    guidelines_path_str = data.get("guidelines_voice_path")
    if guidelines_path_str:
        guidelines_path = Path(guidelines_path_str)
        if guidelines_path.exists():
            problems.extend(_check_guidelines_body(guidelines_path, draft=draft))

    # ------------------------------------------------------------------
    # 7b. The TEXT guidelines body carries the same slot + control tokens
    #     (the runtime injects the persona slot and emits the same control
    #     tokens in text mode), and on drafts must be fully localized. It must
    #     also NOT be a copy of the voice guidelines — text and voice are
    #     distinct documents (text drops spoken-form conventions), and a
    #     copy would silently ship voice-only instructions into text runs.
    # ------------------------------------------------------------------
    text_guidelines_path_str = data.get("guidelines_text_path")
    if text_guidelines_path_str:
        text_guidelines_path = Path(text_guidelines_path_str)
        if text_guidelines_path.exists():
            problems.extend(_check_guidelines_body(text_guidelines_path, draft=draft))
            if guidelines_path_str:
                problems.extend(
                    _check_text_not_copy_of_voice(
                        text_guidelines_path, Path(guidelines_path_str)
                    )
                )

    return GuardrailReport.from_problems(
        problems, notes=experiment_notes + pending_beds
    )


# The control tokens the runtime emits; they must survive verbatim (ASCII) in
# the localized guidelines — never translated or re-scripted.
_GUIDELINES_CONTROL_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")
_GUIDELINES_PERSONA_SLOT = "<PERSONA_GUIDELINES>"

# Signature ENGLISH instruction phrases from the source guidelines. A pack that
# is fully written in the target language renders these in that language, so
# their verbatim survival means the instructions were left in English with only
# the examples localized (the "English-skeleton" regression that made some packs
# inconsistent). Checked on DRAFTS only: shipped packs predate the
# full-translation contract and are accepted as-is (see _check_guidelines_body).
_GUIDELINES_ENGLISH_INSTRUCTION_MARKERS = (
    "Only share information that is explicitly",
    "Make the agent work for information",
    "Start with minimal information",
    "Strictly follow the scenario instructions",
    "Never fabricate, guess, or infer",
    "Do not end the conversation prematurely",
    "Use vague initial statements",
    "Sometimes forget details",
)
# Two or more verbatim markers is unambiguous skeleton; a fully localized pack
# matches none. (A lone coincidental match stays under the bar.)
_GUIDELINES_ENGLISH_MARKER_THRESHOLD = 2


def _check_guidelines_body(path: Path, *, draft: bool = False) -> list[str]:
    """The localized guidelines body must keep the persona slot + a control token.

    On ``draft`` packs it additionally enforces the full-translation contract:
    the instruction prose itself must be authored in the target language, not
    left in English with only the examples localized. This is draft-only so it
    drives the draft repair loop toward a fully localized pack without
    retroactively failing the already-shipped English-skeleton packs.
    """
    problems: list[str] = []
    try:
        text = path.read_text()
    except OSError as exc:
        return [f"guidelines file {path.name} could not be read: {exc}"]
    if _GUIDELINES_PERSONA_SLOT not in text:
        problems.append(
            f"guidelines file {path.name} is missing the literal "
            f"{_GUIDELINES_PERSONA_SLOT} slot (the runtime injects each "
            "persona's content there)"
        )
    if not any(token in text for token in _GUIDELINES_CONTROL_TOKENS):
        problems.append(
            f"guidelines file {path.name} contains none of the ASCII control "
            f"tokens {list(_GUIDELINES_CONTROL_TOKENS)} — control tokens must "
            "stay verbatim in ASCII, never translated or re-scripted"
        )
    if draft:
        leaked = [m for m in _GUIDELINES_ENGLISH_INSTRUCTION_MARKERS if m in text]
        if len(leaked) >= _GUIDELINES_ENGLISH_MARKER_THRESHOLD:
            problems.append(
                f"guidelines file {path.name} still has English instruction prose "
                f"(e.g. {leaked[:3]}) — the guidelines must be written ENTIRELY in "
                "the target language (translate every heading and instruction); "
                "only the <PERSONA_GUIDELINES> slot and the ASCII control tokens "
                "stay verbatim in English"
            )
    return problems


# Text guidelines this close to the voice guidelines are treated as a copy.
# Text and voice are distinct documents; text drops spoken-form conventions,
# so a near-identical file means the voice guidelines were reused, silently
# shipping voice-only instructions (spell-it-out, backchannels) into text runs.
_TEXT_VOICE_SIMILARITY_THRESHOLD = 0.92


def _check_text_not_copy_of_voice(text_path: Path, voice_path: Path) -> list[str]:
    """The text guidelines must not be (near-)identical to the voice guidelines."""
    import difflib

    try:
        text_body = text_path.read_text()
        voice_body = voice_path.read_text()
    except OSError:
        return []  # a read problem is already reported by the body check
    ratio = difflib.SequenceMatcher(None, text_body.split(), voice_body.split()).ratio()
    if ratio >= _TEXT_VOICE_SIMILARITY_THRESHOLD:
        return [
            f"text guidelines file {text_path.name} is {ratio:.0%} identical to "
            f"the voice guidelines {voice_path.name} — text and voice must be "
            "distinct documents (text drops spoken-form conventions like "
            "spell-it-out, backchannels, and silence handling). Re-author the "
            "text guidelines for typed chat instead of copying the voice file."
        ]
    return []


def _check_experiments(
    experiments: list, pack_language: object, data: dict
) -> tuple[list[str], list[str]]:
    """Deterministic invariants over the ``experiments`` list.

    Each entry must validate as the SAME typed ``ExperimentSpec`` the runtime
    loader enforces (a broken pack must fail HERE at validate time, not at
    ``tau2 run-preset`` time). On top of the schema, per entry:

    - ``preset_name`` must be ``multilingual_v1_<main_arm_name>``;
    - non-default domains must carry the ``_<domain>`` arm-name suffix, so
      their presets and save paths can never collide with the headline arm;
    - ``task_set_suffix`` must equal the pack language;
    - one entry per domain; preset/arm names unique across entries;
    - a pack that carries a ``localization`` block must gloss every domain it
      has an experiment for with FULL catalog coverage (backfill with ``tau2
      factory draft-localization --lang <lang> --domain <domain>``). A pack
      with NO localization block
      (English) is exempt.

    The naming rules hold by construction now that ``pack_assembly``
    code-generates the entries; the guardrail catches hand-edits that drift
    from the convention (registry presets key on them).

    Returns ``(problems, notes)``. A missing localized task file
    (``<domain>_tasks_<suffix>.json``) is a NOTE, not a problem: the factory
    order produces it AFTER the draft (the retired task-translation loop /
    ``localize-entities``), and preset generation already warns loudly if it
    is still absent at run time.
    """
    from tau2.multilingual.localization_catalog import get_domain_term_catalog
    from tau2.multilingual.task_sets import discover_localized_task_files

    problems: list[str] = []
    notes: list[str] = []
    task_files = discover_localized_task_files()
    seen_domains: set[str] = set()
    seen_preset_names: set[str] = set()
    seen_arm_names: set[str] = set()
    localization = data.get("localization")
    glossaries = (
        localization.get("domain_glossaries")
        if isinstance(localization, dict)
        else None
    ) or {}
    for i, experiment in enumerate(experiments):
        label = f"experiments[{i}]"
        if not isinstance(experiment, dict):
            problems.append(f"{label}: must be a mapping")
            continue
        try:
            spec = ExperimentSpec.model_validate(experiment)
        except ValidationError as exc:
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"])
                prefix = f"{label}.{location}: " if location else f"{label}: "
                problems.append(prefix + error["msg"])
            continue
        if spec.domain in seen_domains:
            problems.append(
                f"{label}: duplicate entry for domain '{spec.domain}' — "
                "one experiments entry per domain"
            )
        seen_domains.add(spec.domain)
        # Cross-entry uniqueness: preset generation keys on these names, and
        # a duplicate is DROPPED there with only a log warning — validate
        # must fail loud instead.
        if spec.preset_name in seen_preset_names:
            problems.append(
                f"{label}: duplicate preset_name '{spec.preset_name}' across "
                "experiments entries — preset generation would silently drop "
                "one of them"
            )
        seen_preset_names.add(spec.preset_name)
        if spec.main_arm_name in seen_arm_names:
            problems.append(
                f"{label}: duplicate main_arm_name '{spec.main_arm_name}' "
                "across experiments entries — arm names key save paths and "
                "must stay distinct"
            )
        seen_arm_names.add(spec.main_arm_name)
        # Derive the expected names through the SAME helpers the generator
        # uses (pack_assembly.preset_name_for / main_arm_name_for) so the
        # conventions live in one place and the check can never re-encode a
        # stale pattern string.
        expected = preset_name_for(spec.main_arm_name)
        if spec.preset_name != expected:
            problems.append(
                f"{label}.preset_name '{spec.preset_name}' must follow the "
                f"convention 'multilingual_v1_<main_arm_name>' "
                f"(expected '{expected}' for main_arm_name "
                f"'{spec.main_arm_name}')"
            )
        if spec.domain != UNSUFFIXED_ARM_DOMAIN and not spec.main_arm_name.endswith(
            f"_{spec.domain}"
        ):
            problems.append(
                f"{label}.main_arm_name '{spec.main_arm_name}' must end "
                f"with '_{spec.domain}' (every domain but the original "
                f"'{UNSUFFIXED_ARM_DOMAIN}' is suffixed so presets and save "
                "paths stay distinct)"
            )
        if pack_language is not None and spec.task_set_suffix != pack_language:
            problems.append(
                f"{label}.task_set_suffix '{spec.task_set_suffix}' must "
                f"equal the pack language '{pack_language}'"
            )
        glossed = glossaries.get(spec.domain)
        if localization is not None and not glossed:
            problems.append(
                f"{label}: the pack has no '{spec.domain}' glossary under "
                "localization.domain_glossaries — run `tau2 factory "
                f"draft-localization --lang {pack_language} --domain "
                f"{spec.domain}`"
            )
        elif localization is not None:
            # FULL catalog coverage, not mere presence: a stub glossary would
            # ship a mostly-English localization section. (Drafting enforces
            # this at write time; the guardrail catches hand-edits.)
            try:
                catalog = set(get_domain_term_catalog(spec.domain))
            except KeyError:
                catalog = set()  # unknown domain is reported by other checks
            covered = {
                entry.get("term_id") for entry in glossed if isinstance(entry, dict)
            }
            missing = sorted(catalog - covered)
            if missing:
                problems.append(
                    f"{label}: the '{spec.domain}' glossary is missing "
                    f"{len(missing)} catalog term(s) (e.g. {missing[0]!r}) — "
                    "full catalog coverage is required; re-run `tau2 factory "
                    f"draft-localization --lang {pack_language} --domain "
                    f"{spec.domain}`"
                )
        task_set = f"{spec.domain}_{spec.task_set_suffix}"
        if task_set not in task_files:
            notes.append(
                f"{label}: task set '{task_set}' has no "
                f"{spec.domain}_tasks_{spec.task_set_suffix}.json under "
                "data/tau2/multilingual/ yet — produce it before "
                f"`tau2 run-preset {spec.preset_name}`"
            )
    return problems, notes


def _check_noise_files(presets: dict) -> tuple[list[str], list[str]]:
    """Every preset noise file exists where the runtime would load it.

    Mirrors ``tau2.user_simulation_voice_presets.sample_voice_config``:
    background files resolve under ``BACKGROUND_NOISE_CONTINUOUS_DIR`` and
    burst files under ``BURST_NOISE_DIR`` (both inside
    ``data/voice/background_noise_audio_pcm_mono_verified/``).

    Returns ``(problems, notes)``. A missing file is normally a HARD problem.
    The one exception, to break the draft -> finalize -> generate-beds
    ORDERING TRAP for an opted-in pack: a missing CONTINUOUS bed under a
    LOCALE SUBFOLDER (path contains a ``/``) is NON-fatal — it is expected to
    be produced by `the retired locale-bed generator`, which writes exactly
    that path. It
    is surfaced as a NOTE so draft and finalize still pass. Top-level/shared
    beds (no subfolder) must already exist in the repo, so a missing one stays
    a hard problem; bursts are always shared/top-level and stay hard too. This
    preserves fail-loud: skip generate-beds and the locale wav is genuinely
    absent — the runtime voice loader fails when it can't find it and the
    ``noise_wavs`` checklist never flips — rather than silently substituting
    the shared English bed.
    """
    problems: list[str] = []
    notes: list[str] = []
    for preset_id, preset in presets.items():
        if not isinstance(preset, dict):
            continue
        for kind, base_dir in (
            ("background_noise_files", BACKGROUND_NOISE_CONTINUOUS_DIR),
            ("burst_noise_files", BURST_NOISE_DIR),
        ):
            for filename in preset.get(kind) or []:
                if (base_dir / filename).exists():
                    continue
                # A locale-namespaced continuous bed (path has a subfolder) is
                # expected to be generated later — note it, don't block.
                is_locale_continuous_bed = (
                    kind == "background_noise_files" and "/" in str(filename).strip("/")
                )
                if is_locale_continuous_bed:
                    notes.append(
                        f"acoustic preset '{preset_id}': locale continuous bed "
                        f"'{filename}' not present yet under {base_dir} — "
                        "and locale-bed production is retired. Draft/finalize "
                        "pass; the file must exist before any voice run."
                    )
                    continue
                problems.append(
                    f"acoustic preset '{preset_id}': {kind} entry "
                    f"'{filename}' not found under {base_dir} — "
                    f"available files: {_available_noise_files(base_dir)}"
                )
    return problems, notes


def _available_noise_files(base_dir: Path) -> list[str]:
    if not base_dir.is_dir():
        return []
    files = sorted(str(path.relative_to(base_dir)) for path in base_dir.rglob("*.wav"))
    if len(files) > MAX_LISTED_NOISE_FILES:
        extra = len(files) - MAX_LISTED_NOISE_FILES
        files = files[:MAX_LISTED_NOISE_FILES] + [f"... (+{extra} more)"]
    return files

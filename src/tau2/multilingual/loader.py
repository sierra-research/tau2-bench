# Copyright Sierra
"""Discovery/loading of language packs.

Packs are discovered lazily by the registry getters: a
``data/tau2/multilingual/<lang>/pack.yaml`` file holds the whole
``LanguagePack`` — personas, prompts, presets — plus an optional
``experiments`` list (one entry per benchmark domain the pack runs) consumed
by ``tau2.multilingual.run_presets``. Dropping the folder in is enough; no
Python required. A pack.yaml that fails ``LanguagePack`` / ``ExperimentSpec``
validation (or a duplicate registration) fails loudly.

This module owns the ONE pack.yaml normalization (:func:`normalize_pack_data`)
— the factory guardrails and backfill verbs reuse it, so what the runtime
loads and what the factory validates can never drift.
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml
from loguru import logger

if TYPE_CHECKING:
    from tau2.multilingual.schema import ExperimentSpec

_loaded = False

# Serializes discovery so concurrent voice runs (ThreadPoolExecutor workers
# calling the lazy registry getters) never observe a partially-populated
# registry. See load_language_packs.
_load_lock = threading.Lock()

# Typed ``experiments`` entries from pack.yaml, keyed by language code (one
# spec per domain the pack runs); consumed by tau2.multilingual.run_presets
# to generate one multilingual_v1_* preset per spec.
_EXPERIMENTS: dict[str, list["ExperimentSpec"]] = {}


def _reset_for_tests() -> None:
    """TEST-ONLY: forget all discovery state so packs reload from disk.

    Lets tests point ``tau2.utils.DATA_DIR`` at a temporary multilingual dir
    and have the next registry lookup rediscover packs from there. Callers
    must snapshot and restore the prior state (see
    ``tests/test_multilingual/factory_testing/conftest_helpers.py``). Never
    call this from production code.
    """
    global _loaded
    _loaded = False
    _EXPERIMENTS.clear()


def get_language_experiments(language: str) -> list["ExperimentSpec"]:
    """The pack.yaml ``experiments`` entries for a language ([] if none)."""
    load_language_packs()
    return list(_EXPERIMENTS.get(language, []))


def get_language_experiment(language: str, domain: str) -> Optional["ExperimentSpec"]:
    """The language's experiment spec for one domain, if the pack has one."""
    for spec in get_language_experiments(language):
        if spec.domain == domain:
            return spec
    return None


def get_all_language_experiments() -> dict[str, list["ExperimentSpec"]]:
    """All pack.yaml ``experiments`` entries, keyed by language code."""
    load_language_packs()
    return {language: list(specs) for language, specs in _EXPERIMENTS.items()}


def normalize_pack_data(data: dict, pack_dir: Path) -> dict:
    """The loader's exact pack.yaml normalization, as one shared function.

    Returns a shallow-copied mapping ready for ``LanguagePack.model_validate``:

    - the ``experiments`` list is removed (it is a sibling of the pack model,
      each entry validated separately as ``ExperimentSpec``);
    - ``guidelines_voice_path`` / ``guidelines_text_path`` are resolved against
      the pack folder (they are authored relative to it);
    - authoring convenience: list-form ``personas`` / ``acoustic_presets`` are
      keyed into dicts by ``persona_id`` / preset ``id``.

    Shared by the runtime loader, the factory guardrails, and the backfill
    verbs, so their normalization rules can never drift.
    """
    out = dict(data)
    out.pop("experiments", None)
    for guidelines_key in ("guidelines_voice_path", "guidelines_text_path"):
        guidelines = out.get(guidelines_key)
        if guidelines and not Path(guidelines).is_absolute():
            out[guidelines_key] = str(Path(pack_dir) / guidelines)
    if isinstance(out.get("personas"), list):
        out["personas"] = {p["persona_id"]: p for p in out["personas"]}
    if isinstance(out.get("acoustic_presets"), list):
        out["acoustic_presets"] = {a["id"]: a for a in out["acoustic_presets"]}
    return out


def _load_yaml_pack(pack_path: Path) -> None:
    """Validate and register one pack.yaml; stash its typed experiments."""
    from tau2.multilingual.registry import register_language_pack
    from tau2.multilingual.schema import ExperimentSpec, LanguagePack

    data = yaml.safe_load(pack_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{pack_path} is not a YAML mapping")
    experiments = data.get("experiments")
    if experiments is not None and not isinstance(experiments, list):
        raise ValueError(
            f"{pack_path}: 'experiments' must be a list of experiment "
            "entries (one per domain)"
        )

    pack = LanguagePack.model_validate(normalize_pack_data(data, pack_path.parent))
    specs = [ExperimentSpec.model_validate(entry) for entry in experiments or []]
    seen_domains: set[str] = set()
    for spec in specs:
        if spec.domain in seen_domains:
            raise ValueError(
                f"{pack_path}: duplicate experiments entry for domain "
                f"'{spec.domain}' — one spec per domain"
            )
        seen_domains.add(spec.domain)
    register_language_pack(pack)
    if specs:
        _EXPERIMENTS[pack.language] = specs


def load_language_packs() -> None:
    """Discover and register all language packs (idempotent + thread-safe).

    Double-checked locking: ``_loaded`` is set ``True`` only AFTER every pack
    is registered, so a concurrent worker either sees a fully-populated
    registry or waits on the lock, never a half-registered one. On failure the
    registries are rolled back to their pre-load snapshot so retries start from
    a clean cold registry.
    """
    global _loaded
    if _loaded:
        return

    with _load_lock:
        if _loaded:
            return

        # Snapshot everything registration mutates so a mid-load failure can
        # roll back to a clean cold state (packs + experiments live here;
        # personas are registered as a side effect in the registry/voice
        # modules).
        from tau2.data_model import voice_personas
        from tau2.multilingual import registry
        from tau2.multilingual.localize_lib import multilingual_data_dir

        packs_snapshot = dict(registry._LANGUAGE_PACKS)
        experiments_snapshot = dict(_EXPERIMENTS)
        personas_snapshot = dict(voice_personas.ALL_PERSONAS)
        persona_names_snapshot = list(voice_personas.ALL_PERSONA_NAMES)

        try:
            # data/tau2/multilingual/<lang>/pack.yaml
            base = multilingual_data_dir()
            if base.is_dir():
                for pack_path in sorted(base.glob("*/pack.yaml")):
                    try:
                        _load_yaml_pack(pack_path)
                    except Exception:
                        logger.exception(f"Failed to load language pack '{pack_path}'")
                        raise

            _loaded = True
        except Exception:
            registry._LANGUAGE_PACKS.clear()
            registry._LANGUAGE_PACKS.update(packs_snapshot)
            _EXPERIMENTS.clear()
            _EXPERIMENTS.update(experiments_snapshot)
            voice_personas.ALL_PERSONAS.clear()
            voice_personas.ALL_PERSONAS.update(personas_snapshot)
            voice_personas.ALL_PERSONA_NAMES[:] = persona_names_snapshot
            raise

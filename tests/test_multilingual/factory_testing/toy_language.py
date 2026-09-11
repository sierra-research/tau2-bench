# Copyright Sierra
"""The toy language fixture: Toylang ('tl') and the 'toyair' mini domain.

A complete, valid Latin-script language pack plus a 4-task mini domain (2
users in the db) and a pre-localized task-set variant that passes
``check_task_set_localization``. Static data lives in ``fixtures/``; the
helpers here load it, normalize it the way the loader does, and materialize
it into temporary directories for isolated tests.

Persona tags
------------
TODO(factory tags PR): ``MultilingualPersonaConfig`` does not have a ``tags``
field yet (a parallel PR adds it). Until it lands, the toy personas' tag
dimensions live in the ``TOY_PERSONA_TAGS`` sidecar dict below, and helpers
check ``MultilingualPersonaConfig.model_fields`` at runtime: when the field
exists, the tags are injected into the pack dict / written pack.yaml, so this
package works unchanged both before and after that PR merges. Inline the tags
into ``fixtures/tl/pack.yaml`` once the schema field exists.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import yaml

from tau2.multilingual.schema import LanguagePack, MultilingualPersonaConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TOY_PACK_DIR = FIXTURES_DIR / "tl"
TOY_DOMAIN_DIR = FIXTURES_DIR / "toyair"

TOY_LANGUAGE = "tl"
TOY_SCRIPT = "latn"
TOY_DOMAIN = "toyair"
TOY_GUIDELINES_FILENAME = "simulation_guidelines_voice_tl.md"
TOY_LOCALIZED_TASKS_FILENAME = f"{TOY_DOMAIN}_tasks_{TOY_LANGUAGE}.json"

# Sidecar persona tags (see module docstring). One full tag dict per persona,
# every dimension present for both, so tag-pivot tests have real targets.
TOY_PERSONA_TAGS: dict[str, dict[str, str]] = {
    "tessa_tl_v1": {
        "code_switch": "high",
        "formality": "low",
        "english_tolerance": "high",
        "age_band": "20s",
        "gender": "female",
        "register": "urban_professional",
        "closing_style": "brisk",
    },
    "orin_tl_v1": {
        "code_switch": "low",
        "formality": "high",
        "english_tolerance": "low",
        "age_band": "50s",
        "gender": "male",
        "register": "retiree",
        "closing_style": "extended",
    },
}


def personas_support_tags() -> bool:
    """Whether the (parallel-PR) ``tags`` field exists on the schema yet."""
    return "tags" in MultilingualPersonaConfig.model_fields


def toy_pack_yaml_dict(include_experiments: bool = True) -> dict:
    """The raw pack.yaml mapping, with sidecar tags injected when supported.

    This is the *pre-normalization* authoring form (relative guidelines path,
    optional ``experiments`` list) — exactly what would sit on disk.
    """
    data = yaml.safe_load((TOY_PACK_DIR / "pack.yaml").read_text())
    if personas_support_tags():
        for persona_id, tags in TOY_PERSONA_TAGS.items():
            data["personas"][persona_id]["tags"] = dict(tags)
    if not include_experiments:
        data.pop("experiments", None)
    return data


def normalize_pack_dict(data: dict, pack_dir: Path) -> dict:
    """Loader-style normalization (mirrors ``loader._load_yaml_pack``).

    Pops the ``experiments`` list, resolves ``guidelines_voice_path``
    relative to ``pack_dir``, and converts persona/preset lists to dicts.
    Returns a normalized copy ready for ``LanguagePack.model_validate``.
    """
    data = copy.deepcopy(data)
    data.pop("experiments", None)
    guidelines = data.get("guidelines_voice_path")
    if guidelines and not Path(guidelines).is_absolute():
        data["guidelines_voice_path"] = str(pack_dir / guidelines)
    if isinstance(data.get("personas"), list):
        data["personas"] = {p["persona_id"]: p for p in data["personas"]}
    if isinstance(data.get("acoustic_presets"), list):
        data["acoustic_presets"] = {a["id"]: a for a in data["acoustic_presets"]}
    return data


def toy_pack_dict(pack_dir: Path = TOY_PACK_DIR) -> dict:
    """The normalized, validation-ready toy pack dict.

    ``pack_dir`` must contain the guidelines markdown (the fixtures dir does;
    pass the folder from ``write_toy_pack_dir`` for tmp copies).
    """
    return normalize_pack_dict(toy_pack_yaml_dict(), pack_dir)


def load_toy_pack() -> LanguagePack:
    """Build the toy ``LanguagePack`` object directly — no registry involved."""
    return LanguagePack.model_validate(toy_pack_dict())


def write_toy_pack_dir(tmp_path: Path) -> Path:
    """Write the toy pack into ``tmp_path``, mirroring
    ``data/tau2/multilingual/<lang>/``: pack.yaml (tags injected when the
    schema supports them) + the guidelines markdown. Returns the language
    folder (``tmp_path / 'tl'``).
    """
    lang_dir = tmp_path / TOY_LANGUAGE
    lang_dir.mkdir(parents=True, exist_ok=True)
    (lang_dir / "pack.yaml").write_text(
        yaml.safe_dump(toy_pack_yaml_dict(), sort_keys=False, allow_unicode=True)
    )
    shutil.copy(
        TOY_PACK_DIR / TOY_GUIDELINES_FILENAME, lang_dir / TOY_GUIDELINES_FILENAME
    )
    return lang_dir


def toy_tasks() -> list[dict]:
    """The 4 English toyair tasks (fresh copies)."""
    with open(TOY_DOMAIN_DIR / "tasks.json") as fp:
        return json.load(fp)


def toy_tasks_localized() -> list[dict]:
    """The pre-localized Toylang variant ('_tl' ids; passes the invariants)."""
    with open(TOY_PACK_DIR / TOY_LOCALIZED_TASKS_FILENAME) as fp:
        return json.load(fp)


def toy_db() -> dict:
    """The toyair mini db (2 users, 2 reservations)."""
    with open(TOY_DOMAIN_DIR / "db.json") as fp:
        return json.load(fp)

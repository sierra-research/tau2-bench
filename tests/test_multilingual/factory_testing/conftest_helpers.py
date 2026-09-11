# Copyright Sierra
"""Pytest fixtures for isolated language-pack environments.

Exposed to the suite via ``tests/test_multilingual/conftest.py``. The core
fixture, ``isolated_pack_env``, gives a test a temporary DATA_DIR containing
ONLY the toy Toylang pack and the toyair mini domain, with every piece of
global pack state (loader flag, experiment blocks, pack registry, voice
personas) reset for the test and restored afterwards — so tests can exercise
discovery/registration/injection end to end without touching the real hi
pack or leaking state into other tests.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
import tau2.multilingual.run_presets as ml_run_presets
import tau2.utils
import tau2.utils.utils
from tau2.multilingual.domain_profiles import (
    DOMAIN_PROFILES,
    CallerIdentityKind,
    DomainLocalizationProfile,
)
from test_multilingual.factory_testing.toy_language import (
    TOY_DOMAIN,
    TOY_DOMAIN_DIR,
    TOY_LOCALIZED_TASKS_FILENAME,
    TOY_PACK_DIR,
    write_toy_pack_dir,
)


@dataclass(frozen=True)
class IsolatedPackEnv:
    """Paths inside the temporary DATA_DIR set up by ``isolated_pack_env``."""

    data_dir: Path  # the tmp DATA_DIR (tau2.utils.DATA_DIR points here)
    multilingual_dir: Path  # <data_dir>/tau2/multilingual
    pack_dir: Path  # <multilingual_dir>/tl (pack.yaml + guidelines)
    domain_dir: Path  # <data_dir>/tau2/domains/toyair (tasks.json + db.json)
    localized_tasks_fixture: Path  # pre-localized tl task set (NOT installed)

    def install_localized_tasks(self) -> Path:
        """Copy the pre-localized tl task set into the pack folder, where the
        filename convention (``task_sets.py``) will discover it."""
        dest = self.pack_dir / TOY_LOCALIZED_TASKS_FILENAME
        shutil.copy(self.localized_tasks_fixture, dest)
        return dest


@pytest.fixture
def isolated_pack_env(tmp_path, monkeypatch) -> IsolatedPackEnv:
    """A tmp DATA_DIR with only the toy pack + toyair domain; state restored.

    - Copies the toy pack into ``<tmp>/data/tau2/multilingual/tl/`` and the
      toyair domain into ``<tmp>/data/tau2/domains/toyair/``.
    - Monkeypatches ``tau2.utils.DATA_DIR`` (and the defining module — every
      multilingual path helper resolves through it dynamically) and sets
      ``TAU2_DATA_DIR`` for subprocesses.
    - Resets loader/registry state via the test-only ``_reset_for_tests``
      hooks so the next lookup rediscovers packs from the tmp dir; snapshots
      and restores everything (including voice personas) afterwards.
    """
    saved_loaded = ml_loader._loaded
    saved_experiments = dict(ml_loader._EXPERIMENTS)
    saved_packs = dict(ml_registry._LANGUAGE_PACKS)
    saved_personas = dict(voice_personas.ALL_PERSONAS)
    saved_persona_names = list(voice_personas.ALL_PERSONA_NAMES)

    data_dir = tmp_path / "data"
    multilingual_dir = data_dir / "tau2" / "multilingual"
    pack_dir = write_toy_pack_dir(multilingual_dir)
    domain_dir = data_dir / "tau2" / "domains" / TOY_DOMAIN
    domain_dir.mkdir(parents=True)
    shutil.copy(TOY_DOMAIN_DIR / "tasks.json", domain_dir / "tasks.json")
    shutil.copy(TOY_DOMAIN_DIR / "db.json", domain_dir / "db.json")

    monkeypatch.setattr(tau2.utils, "DATA_DIR", data_dir)
    monkeypatch.setattr(tau2.utils.utils, "DATA_DIR", data_dir)
    monkeypatch.setenv("TAU2_DATA_DIR", str(data_dir))
    # The domain-profile catalog is closed (unknown domains raise at every
    # lookup seam), so the fixture that installs the toyair domain must also
    # register its profile — airline-shaped, matching the toy fixtures.
    monkeypatch.setitem(
        DOMAIN_PROFILES,
        TOY_DOMAIN,
        DomainLocalizationProfile(
            domain=TOY_DOMAIN,
            source_tasks_filename="tasks.json",
            caller_identity=CallerIdentityKind.USER_ID_HANDLE,
            identity_swap_supported=True,
            tasks_translated=True,
            vocabulary_context=(
                "Calls are to the toy airline's customer service line."
            ),
            translation_example_entities=(
                "user ids like 'ana_lee_1001' and reservation codes"
            ),
            default_smoke_task_stem="1",
        ),
    )

    ml_loader._reset_for_tests()
    ml_registry._reset_for_tests()
    # The lazily-built run-preset table caches what the loader discovered —
    # drop it so presets regenerate from the tmp data dir (and again on
    # teardown so later tests regenerate from the real one).
    ml_run_presets._reset_presets_cache_for_tests()

    try:
        yield IsolatedPackEnv(
            data_dir=data_dir,
            multilingual_dir=multilingual_dir,
            pack_dir=pack_dir,
            domain_dir=domain_dir,
            localized_tasks_fixture=TOY_PACK_DIR / TOY_LOCALIZED_TASKS_FILENAME,
        )
    finally:
        ml_loader._reset_for_tests()
        ml_registry._reset_for_tests()
        ml_run_presets._reset_presets_cache_for_tests()
        ml_loader._loaded = saved_loaded
        ml_loader._EXPERIMENTS.update(saved_experiments)
        ml_registry._LANGUAGE_PACKS.update(saved_packs)
        voice_personas.ALL_PERSONAS.clear()
        voice_personas.ALL_PERSONAS.update(saved_personas)
        voice_personas.ALL_PERSONA_NAMES[:] = saved_persona_names

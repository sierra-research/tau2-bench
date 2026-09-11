# Copyright Sierra
"""Shared fixtures for the multilingual test suite.

The isolated-pack-env implementation lives in
``test_multilingual/factory_testing/conftest_helpers.py`` (an importable
package, so factory PR test suites can reuse them directly too). This
conftest additionally hosts the suite's small shared helpers: the
registry-snapshot fixture (opt-in via ``pytest.mark.usefixtures``), the
voice-simulator builder, the first-persona lookup, and the isolated factory
workspace fixture.
"""

from pathlib import Path

import pytest

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
import tau2.multilingual.varieties as ml_varieties
from test_multilingual.factory_testing.conftest_helpers import (  # noqa: F401
    isolated_pack_env,
)

# Catalog entries for the suite's synthetic 'xx' test language (the variety
# catalogs are closed and fail loudly at prompt render, so synthetic packs
# need entries just like real ones do).
TEST_LOCALE = "XX-TS"
TEST_LOCALE_DISPLAY = "Testville, Testland"
TEST_VARIETY = "Test-Standard Testlang (Testland)"


@pytest.fixture
def clean_registries():
    """Snapshot/restore the pack registry and voice persona registry.

    Also seeds the closed variety catalogs with the synthetic 'xx' language's
    entries (restored on teardown), so tests registering 'xx' packs can render
    locale/variety-pinned prompts without touching the shipped catalogs.

    NOT autouse: tests that register packs directly opt in per module with
    ``pytestmark = pytest.mark.usefixtures("clean_registries")``.
    """
    saved_packs = dict(ml_registry._LANGUAGE_PACKS)
    saved_personas = dict(voice_personas.ALL_PERSONAS)
    saved_names = list(voice_personas.ALL_PERSONA_NAMES)
    saved_loaded = ml_loader._loaded
    saved_locales = dict(ml_varieties.LOCALE_DISPLAY_NAMES)
    saved_varieties = dict(ml_varieties.LANGUAGE_VARIETY_NAMES)
    ml_loader._loaded = True  # skip discovery; tests register packs directly
    ml_varieties.LOCALE_DISPLAY_NAMES[TEST_LOCALE] = TEST_LOCALE_DISPLAY
    ml_varieties.LANGUAGE_VARIETY_NAMES["xx"] = TEST_VARIETY
    yield
    ml_registry._LANGUAGE_PACKS.clear()
    ml_registry._LANGUAGE_PACKS.update(saved_packs)
    voice_personas.ALL_PERSONAS.clear()
    voice_personas.ALL_PERSONAS.update(saved_personas)
    voice_personas.ALL_PERSONA_NAMES[:] = saved_names
    ml_loader._loaded = saved_loaded
    ml_varieties.LOCALE_DISPLAY_NAMES.clear()
    ml_varieties.LOCALE_DISPLAY_NAMES.update(saved_locales)
    ml_varieties.LANGUAGE_VARIETY_NAMES.clear()
    ml_varieties.LANGUAGE_VARIETY_NAMES.update(saved_varieties)


def make_voice_sim(
    persona_config,
    *,
    llm: str = "dummy",
    instructions: str = "You are a test user.",
    **kwargs,
):
    """A VoiceStreamingUserSimulator with the suite's standard voice settings
    (no transcription, default synthesis). Voice imports stay lazy so the
    conftest itself never requires the voice extra."""
    from tau2.data_model.voice import SynthesisConfig, VoiceSettings
    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    return VoiceStreamingUserSimulator(
        tools=None,
        instructions=instructions,
        llm=llm,
        voice_settings=VoiceSettings(
            transcription_config=None,
            synthesis_config=SynthesisConfig(),
        ),
        persona_config=persona_config,
        **kwargs,
    )


def first_persona(language: str):
    """The alphabetically-first persona of a registered language pack."""
    pack = ml_registry.get_language_pack(language)
    return pack.personas[sorted(pack.personas)[0]]


@pytest.fixture
def factory_dir(tmp_path, monkeypatch) -> Path:
    """An isolated factory workspace root under a temporary DATA_DIR."""
    import tau2.multilingual.factory.state as factory_state
    import tau2.utils
    import tau2.utils.utils

    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    monkeypatch.setattr(tau2.utils.utils, "DATA_DIR", tmp_path)
    return factory_state.factory_root_dir()

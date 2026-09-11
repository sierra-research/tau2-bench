# Copyright Sierra
"""``tau2 factory generate-assets`` is VOICE-ONLY: checklist gating.

Since the shared-English-acoustics decision, generate-assets produces persona
voices only (locale-bed production is retired and deleted). These
tests pin the checklist semantics of the voice-only flow:

- ``voice_ids`` flips only when every persona voice succeeded;
- ``noise_wavs`` flips trivially for a pack with NO acoustic presets (the
  benchmark default — there are no locale wavs to deliver), gated on passing
  guardrails;
- ``noise_wavs`` is NOT touched for an opted-in pack (that was the retired
  bed verb's job).
"""

from tau2.multilingual.factory import asset_generation
from tau2.multilingual.factory.asset_generation import generate_assets
from tau2.multilingual.factory.voice_generation import (
    PersonaVoiceResult,
    PersonaVoiceStatus,
)


class _Checklist:
    def __init__(self):
        self.voice_ids = False
        self.noise_wavs = False


class _FakeProject:
    def __init__(self):
        self.checklist = _Checklist()
        self.saved = False

    def save(self):
        self.saved = True


class _Report:
    def __init__(self, ok, problems=None):
        self.ok = ok
        self.problems = problems or []


class _Pack:
    def __init__(self, acoustic_presets=None):
        self.acoustic_presets = acoustic_presets or {}
        self.personas = {}


def _voice_result(status):
    return PersonaVoiceResult(
        persona_id="fatima_hi_v1",
        display_name="Fatima",
        voice_id="v123" if status is PersonaVoiceStatus.CREATED else None,
        status=status,
    )


def _patch(monkeypatch, *, voice_statuses, report_ok, pack):
    project = _FakeProject()
    monkeypatch.setattr(
        asset_generation,
        "generate_pack_voices",
        lambda *a, **k: [_voice_result(s) for s in voice_statuses],
    )
    monkeypatch.setattr(asset_generation, "pack_locale_dir", lambda lang: None)
    monkeypatch.setattr(asset_generation, "get_language_pack", lambda lang: pack)
    monkeypatch.setattr(
        asset_generation.FactoryProject,
        "load",
        classmethod(lambda cls, lang: project),
    )
    monkeypatch.setattr(
        asset_generation, "validate_final_pack", lambda lang: _Report(report_ok)
    )
    return project


def test_voice_success_flips_voice_ids_and_trivial_noise_wavs(monkeypatch):
    """Default pack (no presets): voices pin, noise_wavs trivially satisfied."""
    project = _patch(
        monkeypatch,
        voice_statuses=[PersonaVoiceStatus.CREATED, PersonaVoiceStatus.SKIPPED],
        report_ok=True,
        pack=_Pack(),
    )
    outcome = generate_assets("hi")
    assert project.checklist.voice_ids is True
    assert project.checklist.noise_wavs is True
    assert project.saved is True
    assert outcome.guardrail_ok is True


def test_voice_error_does_not_flip_voice_ids(monkeypatch):
    project = _patch(
        monkeypatch,
        voice_statuses=[PersonaVoiceStatus.CREATED, PersonaVoiceStatus.ERROR],
        report_ok=True,
        pack=_Pack(),
    )
    generate_assets("hi")
    assert project.checklist.voice_ids is False


def test_guardrail_failure_blocks_trivial_noise_wavs(monkeypatch):
    project = _patch(
        monkeypatch,
        voice_statuses=[PersonaVoiceStatus.CREATED],
        report_ok=False,
        pack=_Pack(),
    )
    generate_assets("hi")
    assert project.checklist.noise_wavs is False


def test_opted_in_pack_leaves_noise_wavs_untouched(monkeypatch):
    """A pack WITH acoustic presets: noise_wavs is generate-beds' to flip."""
    project = _patch(
        monkeypatch,
        voice_statuses=[PersonaVoiceStatus.CREATED],
        report_ok=True,
        pack=_Pack(acoustic_presets={"hi_outdoor_traffic": object()}),
    )
    generate_assets("hi")
    assert project.checklist.voice_ids is True
    assert project.checklist.noise_wavs is False


def test_dry_run_touches_no_checklist(monkeypatch):
    project = _patch(
        monkeypatch,
        voice_statuses=[PersonaVoiceStatus.SKIPPED],
        report_ok=True,
        pack=_Pack(),
    )
    generate_assets("hi", dry_run=True)
    assert project.saved is False

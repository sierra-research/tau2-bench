# Copyright Sierra
"""``tau2 factory generate-assets`` — design + auto-pin persona voices.

VOICE-ONLY since the shared-English-acoustics decision (2026-07-21): every
language runs on the stock English environment selection, so the factory's
asset stage produces exactly one deliverable — an ElevenLabs voice for every
persona, designed from its ``tts_voice_prompt`` (the one-male/one-female
invariant is enforced first) and auto-pinned into pack.yaml.
Already-pinned personas are skipped unless ``force``.

Locale background beds are retired: their generator was deleted on
2026-08-03 and every language runs on shared stock-English acoustics. This
stage is voice-only and never touches ``noise_wavs``.

On a successful real run this flips the ``voice_ids`` :class:`HumanChecklist`
flag — auto-pinned, no human audition required — and re-runs the final-pack
guardrails so a broken reference fails loudly. A pack with no
``acoustic_presets`` (the benchmark default) also gets its ``noise_wavs``
checklist item flipped: there are no locale wavs to deliver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from tau2.multilingual.factory.guardrails import validate_final_pack
from tau2.multilingual.factory.paths import derive_locale_dir
from tau2.multilingual.factory.state import FactoryProject
from tau2.multilingual.factory.voice_generation import (
    PersonaVoiceResult,
    PersonaVoiceStatus,
    audition_text_for,
    generate_pack_voices,
)
from tau2.multilingual.registry import get_language_pack
from tau2.utils import DATA_DIR
from tau2.voice.utils.voice_design import VOICE_DESIGN_MODEL


class AssetGenerationOutcome(BaseModel):
    """What one ``tau2 factory generate-assets`` run produced."""

    language: str
    voice_results: list[PersonaVoiceResult] = Field(default_factory=list)
    voice_rationale_path: Optional[Path] = None
    readme_path: Optional[Path] = None
    guardrail_ok: bool = True
    guardrail_problems: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)


def generate_assets(
    language: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> AssetGenerationOutcome:
    """Design + auto-pin persona voices for a finalized language pack."""
    outcome = AssetGenerationOutcome(language=language)

    outcome.voice_results = generate_pack_voices(language, force=force, dry_run=dry_run)
    outcome.lines = [
        f"  voice  {r.persona_id:22s} {r.status.value:8s} {r.voice_id or ''}"
        for r in outcome.voice_results
    ]

    # Persist voice provenance + a locale README next to the pack's committed
    # audio assets (if any), committed alongside the pack.
    if not dry_run:
        locale_dir = pack_locale_dir(language)
        if locale_dir is not None:
            if outcome.voice_results:
                outcome.voice_rationale_path = write_voice_rationale(
                    language, locale_dir, outcome.voice_results
                )
            outcome.readme_path = write_locale_readme(language, locale_dir)

    # Re-validate the final pack BEFORE persisting any checklist flag, so a
    # partially-generated or broken pack can never be recorded as "done".
    report = validate_final_pack(language)
    outcome.guardrail_ok = report.ok
    outcome.guardrail_problems = report.problems

    # Flip the human-bottleneck checklist only on a fully successful real run
    # (auto-pinned): every persona voice must have been produced (no errors).
    if not dry_run:
        project = FactoryProject.load(language)
        if project is not None:
            voices_ok = bool(outcome.voice_results) and all(
                r.status in (PersonaVoiceStatus.CREATED, PersonaVoiceStatus.SKIPPED)
                for r in outcome.voice_results
            )
            if voices_ok:
                project.checklist.voice_ids = True
            # A pack with no acoustic presets (the shared-English benchmark
            # default) has no locale wavs to deliver — the noise_wavs item is
            # trivially satisfied. Opted-in packs flip it via
            # the retired locale-bed generator.
            pack = get_language_pack(language)
            if pack is not None and not pack.acoustic_presets and report.ok:
                project.checklist.noise_wavs = True
            project.save()
        else:
            logger.info(
                f"No factory project for '{language}' (hand-authored pack) — "
                "skipping checklist update."
            )

    return outcome


def locale_asset_dir(locale_dir: str) -> Path:
    """The on-disk locale asset dir (voice provenance, README, locale beds)."""
    return DATA_DIR / "voice" / "background_noise_audio_pcm_mono_verified" / locale_dir


def pack_locale_dir(language: str) -> Optional[str]:
    """The ``language_COUNTRY`` asset dir for a pack, from its persona locales."""
    pack = get_language_pack(language)
    if pack is None:
        return None
    locales = [
        getattr(p, "locale", None) for p in getattr(pack, "personas", {}).values()
    ]
    return derive_locale_dir(language, locales)


def write_voice_rationale(
    language: str, locale_dir: str, voice_results: list[PersonaVoiceResult]
) -> Optional[Path]:
    """Persist per-persona voice-design provenance as YAML in the locale dir.

    The design *prompt* itself already lives in each persona's ``tts_voice_prompt``
    in pack.yaml; this file makes the rest of the provenance auditable — the
    ElevenLabs model, the gender-matched audition text, the resulting voice_id,
    and this run's status — mirroring ``acoustic_rationale.yaml`` for beds.
    """
    pack = get_language_pack(language)
    if pack is None:
        return None
    # .value: this dict is YAML-dumped below (safe_dump rejects Enum objects).
    status_by_pid = {r.persona_id: r.status.value for r in (voice_results or [])}
    personas: dict[str, dict] = {}
    for pid, persona in getattr(pack, "personas", {}).items():
        tags = getattr(persona, "tags", None) or {}
        gender = tags.get("gender") if isinstance(tags, dict) else None
        personas[pid] = {
            "voice_id": getattr(persona, "voice_id", None),
            "gender": gender,
            "voice_design_model": VOICE_DESIGN_MODEL,
            "voice_design_prompt": getattr(persona, "tts_voice_prompt", ""),
            "audition_text": audition_text_for(language, gender),
            "status": status_by_pid.get(pid, "preexisting"),
        }
    payload = {
        "language": language,
        "generator": "tau2.multilingual.factory.voice_generation (ElevenLabs Voice Design)",
        "note": (
            "Voice provenance. The design prompt is each persona's "
            "tts_voice_prompt in pack.yaml; voice_id is the saved ElevenLabs "
            "voice. Regenerate: tau2 factory generate-assets --lang "
            f"{language} --force."
        ),
        "personas": personas,
    }
    path = locale_asset_dir(locale_dir) / "voice_rationale.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    return path


def write_locale_readme(language: str, locale_dir: str) -> Optional[Path]:
    """Write a README for the locale asset dir (skips an existing hand-edited one)."""
    path = locale_asset_dir(locale_dir) / "README.md"
    if path.exists():
        return path
    pack = get_language_pack(language)
    display = getattr(pack, "display_name", None) or language
    content = f"""# {display} voice + acoustic assets

Generated by `tau2 factory generate-assets --lang {language}` (ElevenLabs).
All audio is **16 kHz PCM mono WAV**.

- **voices**: designed per persona from its `tts_voice_prompt`; pinned `voice_id`s
  live in `pack.yaml`.
- **continuous beds** (`continuous/{locale_dir}/`, only if present): LEGACY
  generated locale beds (outdoor traffic + TV-news-over-kitchen). The benchmark
  default is the shared stock-English acoustics; beds are produced only by the
  explicit the retired locale-bed generator verb.
- **bursts**: constant across locales — presets reference the shared English
  burst files (`car_horn.wav`, `engine_idling.wav`, `dog_bark.wav`), NOT per-locale
  copies.

## Provenance
- Voices: [`voice_rationale.yaml`](./voice_rationale.yaml) (design prompt, model,
  audition text, voice_id per persona).
- Beds (if present): [`acoustic_rationale.yaml`](./acoustic_rationale.yaml)
  (prompts, scripts, models, durations).

Regenerate voices: `tau2 factory generate-assets --lang {language} --force`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path

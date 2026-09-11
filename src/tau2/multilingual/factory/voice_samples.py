# Copyright Sierra
"""Render audition MP3 samples for pack persona voices (ElevenLabs TTS).

Programmatic core behind ``tau2 factory voice-samples``: renders each persona's
locale audition script (:data:`voice_generation._AUDITION_TEXT`) through its
pinned ElevenLabs voice into an output folder for a human codec audition — the
voice analogue of the regenerated-beds reverify folder. Extra explicit
voice_ids (e.g. the rejected options from a prior audition round) can be
rendered alongside with their own labels for A/B comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from tau2.multilingual.factory.voice_generation import (
    _DEFAULT_AUDITION_TEXT,
    _pack_path,
    _persona_gender,
    audition_text_for,
)

# TTS parameters for audition renders. eleven_v3 accepts (and validates) an
# explicit language_code pin — see ElevenLabsTTSConfig.language_code.
SAMPLE_TTS_MODEL_ID = "eleven_v3"
SAMPLE_OUTPUT_FORMAT = "mp3_44100_128"
SAMPLE_TTS_SEED = 42


class ExtraVoice(BaseModel):
    """An explicit non-pack voice to render for comparison."""

    label: str = Field(description="Filename label, e.g. 'minh_bad_optionA'")
    gender: str = Field(description="'male'/'female' — picks the audition variant")
    voice_id: str = Field(description="ElevenLabs voice id")

    @classmethod
    def parse(cls, spec: str) -> "ExtraVoice":
        """Parse the CLI form ``LABEL:GENDER:VOICE_ID``."""
        parts = spec.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"--extra must be LABEL:GENDER:VOICE_ID (got {spec!r})")
        return cls(label=parts[0], gender=parts[1], voice_id=parts[2])


class VoiceSampleResult(BaseModel):
    """One rendered (or failed) audition sample."""

    language: str
    label: str = Field(description="What this sample is (persona+label or extra label)")
    voice_id: str
    path: Optional[Path] = Field(default=None, description="Output MP3 (None on error)")
    error: str = ""


def _render_sample(client, *, voice_id: str, text: str, language: str, path: Path):
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=SAMPLE_TTS_MODEL_ID,
        output_format=SAMPLE_OUTPUT_FORMAT,
        seed=SAMPLE_TTS_SEED,
        language_code=language,
    )
    path.write_bytes(b"".join(audio))


def render_voice_samples(
    language: str,
    *,
    out_dir: Path,
    label: str = "pinned",
    personas: Optional[list[str]] = None,
    extras: Optional[list[ExtraVoice]] = None,
    english: bool = False,
    client=None,
) -> list[VoiceSampleResult]:
    """Render audition samples for a pack's pinned persona voices (+ extras).

    Pack personas render as ``<lang>_<persona_id>_<gender>_<label>.mp3``;
    extras as ``<lang>_<extra.label>_<gender>.mp3``. ``personas`` (persona ids)
    restricts which pack personas render; extras always render. ``english``
    renders the English audition script instead of the language's native one —
    for judging delivery qualities (pace, clarity, distance) without a language
    barrier. Errors are captured per sample so one failed render never loses
    the rest.
    """
    if client is None:
        from elevenlabs import ElevenLabs

        client = ElevenLabs()

    out_dir.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(_pack_path(language).read_text())
    results: list[VoiceSampleResult] = []

    def _run(label_: str, voice_id: str, gender: Optional[str], filename: str):
        path = out_dir / filename
        if english:
            text, tts_language = _DEFAULT_AUDITION_TEXT, "en"
        else:
            text, tts_language = audition_text_for(language, gender), language
        try:
            _render_sample(
                client, voice_id=voice_id, text=text, language=tts_language, path=path
            )
        except Exception as e:  # noqa: BLE001 — per-sample isolation
            logger.error(f"voice sample failed for {label_}: {e}")
            results.append(
                VoiceSampleResult(
                    language=language, label=label_, voice_id=voice_id, error=str(e)
                )
            )
            return
        results.append(
            VoiceSampleResult(
                language=language, label=label_, voice_id=voice_id, path=path
            )
        )
        logger.info(f"rendered {path.name}")

    for pid, p in (data.get("personas") or {}).items():
        if personas and pid not in personas:
            continue
        gender = _persona_gender(p) or "unknown"
        voice_id = str(p.get("voice_id") or "").strip()
        if not voice_id:
            results.append(
                VoiceSampleResult(
                    language=language,
                    label=f"{pid}_{label}",
                    voice_id="",
                    error="no voice_id pinned",
                )
            )
            continue
        _run(
            f"{pid}_{label}",
            voice_id,
            gender,
            f"{language}_{pid}_{gender}_{label}.mp3",
        )

    for extra in extras or []:
        _run(
            extra.label,
            extra.voice_id,
            extra.gender,
            f"{language}_{extra.label}_{extra.gender}.mp3",
        )

    return results

# Copyright Sierra
"""Cheap smoke test that the ElevenLabs Voice Design and Sound Effects endpoints
are reachable: one small call each, a typed pass/fail outcome per endpoint. The
voice-design preview is not saved to the account.

Library-shaped like the sibling factory verbs: :func:`run_asset_smoke` returns
an :class:`AssetSmokeOutcome`; the CLI (``tau2 factory smoke-assets``) renders
it and exits non-zero on failure.
"""

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.voice.utils.audio_io import save_wav_file

# Model ids — v3 everywhere one exists; SFX has no v3 model so use the newest v2.
VOICE_DESIGN_MODEL = "eleven_ttv_v3"
SOUND_EFFECTS_MODEL = "eleven_text_to_sound_v2"

# A Hindi audition line + an India-flavored sound-effect prompt so the smoke
# also sanity-checks the multilingual / locale path we actually use. Voice
# Design requires the audition `text` to be at least 100 characters.
HINDI_AUDITION_TEXT = (
    "नमस्ते, मैं अपने account के बारे में थोड़ी help चाहता था। मैंने आज सुबह "
    "login करने की कोशिश की थी, but बार-बार password गलत बता रहा है। मुझे लगता है "
    "मैंने कुछ change नहीं किया, तो ज़रा check करके बता दीजिए क्या issue है।"
)
SFX_PROMPT = "busy Indian street, continuous car and auto-rickshaw horns honking"


class EndpointCheck(BaseModel):
    """One endpoint's smoke result."""

    name: str = Field(description="Human-readable endpoint name")
    passed: bool
    detail: str = Field(
        default="",
        description="PASS detail (what came back) or FAIL reason (the raw error)",
    )


class AssetSmokeOutcome(BaseModel):
    """What ``tau2 factory smoke-assets`` produced."""

    out_dir: Path = Field(description="Where the smoke artifacts were written")
    checks: list[EndpointCheck]

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)


def _check_voice_design(client, out_dir: Path) -> EndpointCheck:
    """Confirm Voice Design returns a usable preview (does not save a voice)."""
    name = "Voice Design (text_to_voice.design)"
    try:
        result = client.text_to_voice.design(
            voice_description=(
                "A warm, confident middle-aged Indian man speaking natural "
                "Hinglish in an urban Indian accent, mobile phone-call quality."
            ),
            text=HINDI_AUDITION_TEXT,
            model_id=VOICE_DESIGN_MODEL,
            loudness=0.5,
            guidance_scale=38,
            seed=42,
            auto_generate_text=False,
        )
    except Exception as e:  # noqa: BLE001 — surface the raw API error
        return EndpointCheck(name=name, passed=False, detail=f"design call raised: {e}")

    previews = getattr(result, "previews", None)
    if not previews:
        return EndpointCheck(name=name, passed=False, detail="no previews returned")

    preview = previews[0]
    gen_id = getattr(preview, "generated_voice_id", None)
    b64 = getattr(preview, "audio_base_64", None)
    if b64:
        import base64

        (out_dir / "voice_design_preview.mp3").write_bytes(base64.b64decode(b64))
    return EndpointCheck(
        name=name,
        passed=True,
        detail=(
            f"{len(previews)} preview(s), generated_voice_id={gen_id} "
            "(not saved to account)"
        ),
    )


def _check_sound_effects(client, out_dir: Path) -> EndpointCheck:
    """Confirm the Sound-Effects endpoint returns raw PCM we can wrap as a WAV."""
    name = "Sound Effects (text_to_sound_effects.convert)"
    try:
        chunks = client.text_to_sound_effects.convert(
            text=SFX_PROMPT,
            duration_seconds=3.0,
            loop=True,
            model_id=SOUND_EFFECTS_MODEL,
            output_format="pcm_16000",
        )
        raw = b"".join(chunks)
    except Exception as e:  # noqa: BLE001
        return EndpointCheck(
            name=name, passed=False, detail=f"convert call raised: {e}"
        )

    if not raw:
        return EndpointCheck(name=name, passed=False, detail="empty audio returned")

    # Sound-effects PCM is STEREO interleaved (TTS is mono) — wrap as 2-channel.
    audio = AudioData(
        data=raw,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=16000, channels=2
        ),
    )
    out_path = out_dir / "sound_effect.wav"
    save_wav_file(audio, out_path)
    return EndpointCheck(
        name=name,
        passed=True,
        detail=f"{len(raw)} bytes, {audio.duration:.2f}s @ 16kHz PCM16 -> {out_path}",
    )


def run_asset_smoke() -> AssetSmokeOutcome:
    """One small live call per ElevenLabs endpoint the factory depends on."""
    # Env loading belongs to the entry point, not module import (sibling verbs
    # load .env the same way).
    from dotenv import load_dotenv

    load_dotenv()
    from elevenlabs import ElevenLabs

    client = ElevenLabs()  # reads ELEVENLABS_API_KEY from env
    out_dir = Path(tempfile.mkdtemp(prefix="tau2_el_smoke_"))
    return AssetSmokeOutcome(
        out_dir=out_dir,
        checks=[
            _check_voice_design(client, out_dir),
            _check_sound_effects(client, out_dir),
        ],
    )

"""ElevenLabs-compatible HTTP facade for a local Qwen3-TTS model."""

import json
import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16_000
SUPPORTED_OUTPUT_FORMAT = "pcm_16000"
SPEECH_TAG_PATTERN = re.compile(r"\[(cough|sneeze|sniffle)\]", re.IGNORECASE)
PAUSE_TAG_PATTERN = re.compile(r"\[pause\]", re.IGNORECASE)


@dataclass(frozen=True)
class VoiceReference:
    ref_audio: str
    ref_text: str
    language: str


class TextToSpeechRequest(BaseModel):
    """Subset of the ElevenLabs request consumed by this service."""

    model_config = ConfigDict(extra="allow")

    text: str
    model_id: str | None = None
    seed: int | None = None
    voice_settings: dict[str, Any] | None = None


class QwenTTSService:
    """Own the model and serialize GPU inference requests."""

    def __init__(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        model_path = _required_env("QWEN_TTS_MODEL_PATH")
        dtype_name = os.getenv("QWEN_TTS_DTYPE", "bfloat16")
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported QWEN_TTS_DTYPE: {dtype_name}") from exc

        model_kwargs: dict[str, Any] = {
            "device_map": os.getenv("QWEN_TTS_DEVICE", "cuda:0"),
            "dtype": dtype,
        }
        attention = os.getenv("QWEN_TTS_ATTN_IMPLEMENTATION", "flash_attention_2")
        if attention:
            model_kwargs["attn_implementation"] = attention

        self.model = Qwen3TTSModel.from_pretrained(model_path, **model_kwargs)
        self.torch = torch
        self.default_voice = VoiceReference(
            ref_audio=_required_env("QWEN_TTS_REF_AUDIO"),
            ref_text=_required_env("QWEN_TTS_REF_TEXT"),
            language=os.getenv("QWEN_TTS_LANGUAGE", "English"),
        )
        self.voices = self._load_voices(os.getenv("QWEN_TTS_VOICES_FILE"))
        self.inference_lock = threading.Lock()

    def synthesize(self, text: str, voice_id: str, seed: int | None) -> bytes:
        reference = self.voices.get(voice_id, self.default_voice)
        normalized_text = _normalize_text(text)
        if not normalized_text:
            return np.zeros(TARGET_SAMPLE_RATE // 5, dtype="<i2").tobytes()

        with self.inference_lock, self.torch.inference_mode():
            if seed is not None:
                self.torch.manual_seed(seed)
                if self.torch.cuda.is_available():
                    self.torch.cuda.manual_seed_all(seed)
            wavs, sample_rate = self.model.generate_voice_clone(
                text=normalized_text,
                language=reference.language,
                ref_audio=reference.ref_audio,
                ref_text=reference.ref_text,
            )

        if len(wavs) == 0:
            raise RuntimeError("Qwen3-TTS returned no waveform")
        return _to_pcm16_mono(wavs[0], sample_rate)

    @staticmethod
    def _load_voices(path: str | None) -> dict[str, VoiceReference]:
        if path is None:
            return {}
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return {
            voice_id: VoiceReference(
                ref_audio=config["ref_audio"],
                ref_text=config["ref_text"],
                language=config.get("language", "English"),
            )
            for voice_id, config in raw.items()
        }


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _normalize_text(text: str) -> str:
    text = PAUSE_TAG_PATTERN.sub("...", text)
    return SPEECH_TAG_PATTERN.sub("", text).strip()


def _to_pcm16_mono(waveform: Any, source_rate: int) -> bytes:
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    audio = np.asarray(waveform, dtype=np.float32).squeeze()
    if audio.ndim == 2:
        audio = audio.mean(axis=0 if audio.shape[0] <= 2 else 1)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape {audio.shape}")
    if source_rate != TARGET_SAMPLE_RATE:
        divisor = np.gcd(source_rate, TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            source_rate // divisor,
        )
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _check_api_key(received: str | None) -> None:
    expected = os.getenv("QWEN_TTS_API_KEY")
    if expected and received != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.tts = QwenTTSService()
    yield


app = FastAPI(title="Qwen3-TTS ElevenLabs Compatibility Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/text-to-speech/{voice_id}")
def text_to_speech(
    voice_id: str,
    request: TextToSpeechRequest,
    output_format: str = Query(default=SUPPORTED_OUTPUT_FORMAT),
    xi_api_key: str | None = Header(default=None, alias="xi-api-key"),
) -> Response:
    _check_api_key(xi_api_key)
    if output_format != SUPPORTED_OUTPUT_FORMAT:
        raise HTTPException(
            status_code=400,
            detail=f"Only {SUPPORTED_OUTPUT_FORMAT} is supported",
        )
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    try:
        audio = app.state.tts.synthesize(request.text, voice_id, request.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=audio, media_type="audio/pcm")

"""Extended-thinking Live runs reject the imported SDK before provider calls."""

import pytest
from google import genai

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.voice.audio_native.gemini.provider import require_extended_thinking_sdk

MODEL = "gemini-3.8-live-extended-thinking"


@pytest.mark.parametrize(
    "version", ["1.63.0", "2.25.0", "2.26.0", "2.30.0", "2.30.1rc1"]
)
def test_extended_thinking_rejects_old_or_unpinned_sdk(monkeypatch, version):
    monkeypatch.setattr(genai, "__version__", version)
    config = VoiceRunConfig(
        domain="intake",
        audio_native_config=AudioNativeConfig(provider="gemini", model=MODEL),
    )
    with pytest.raises(ValueError, match="google-genai>=2.30.1"):
        config.validate()


@pytest.mark.parametrize("version", ["2.30.1", "2.31.0"])
def test_extended_thinking_accepts_new_sdk(monkeypatch, version):
    monkeypatch.setattr(genai, "__version__", version)
    require_extended_thinking_sdk(MODEL)


def test_other_gemini_live_models_do_not_need_new_sdk(monkeypatch):
    monkeypatch.setattr(genai, "__version__", "2.25.0")
    require_extended_thinking_sdk("gemini-live-2.5-flash")

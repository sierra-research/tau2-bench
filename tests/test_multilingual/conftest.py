"""Reviewed runtime prompt fixtures."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def voice_extra():
    """Gate only tests constructing voice runtime objects, not text/config tests."""
    for module in ("pyaudio", "elevenlabs", "scipy", "pydub", "websockets"):
        pytest.importorskip(module, reason="Requires the voice extra")


@pytest.fixture
def english_prompt_content():
    return json.loads(
        (Path(__file__).parent / "fixtures/reviewed_prompt_content.json").read_text()
    )["en"]

"""Phonic classifies agent speech from the audio played in each tick.

The flag uses the OpenAI Live adapter's peak test and must not change the
audio bytes.
"""

import audioop
from types import SimpleNamespace

import pytest

from tau2.voice.audio_native.phonic.discrete_time_adapter import mark_played_speech


@pytest.mark.parametrize("amplitude, expected", [(0, False), (24, False), (1000, True)])
def test_contains_speech_from_played_audio(amplitude, expected):
    sample = int(amplitude).to_bytes(2, "little", signed=True)
    payload = audioop.lin2ulaw(sample * 1600, 2)
    result = SimpleNamespace(agent_audio_data=payload, contains_speech=None)
    mark_played_speech(result)
    assert result.agent_audio_data == payload
    assert result.contains_speech is expected


def test_empty_tick_is_not_speech():
    result = SimpleNamespace(agent_audio_data=b"", contains_speech=None)
    mark_played_speech(result)
    assert result.contains_speech is False

"""In-turn insert labels must describe the text actually sent to TTS."""

import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tau2.agent.base.voice import VoiceMixin
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.audio_effects import (
    ChannelEffectsResult,
    SourceEffectsResult,
    SpeechEffectsResult,
    UserSpeechInsert,
)
from tau2.data_model.message import UserMessage
from tau2.data_model.voice import SpeechEnvironment, SynthesisConfig, VoiceSettings
from tau2.voice.utils.text_effects import insert_speech_text


@pytest.mark.parametrize(
    "text,min_words,pronunciations,draw_insert,inserted",
    [
        ("Yes please", 3, None, True, False),
        ("我想取消我的预订谢谢", 3, None, True, False),
        ("Okay", 3, None, True, False),
        ("Yes please proceed", 3, None, True, True),
        ("Yes please proceed", 4, None, True, False),
        ("Yes please", 2, None, True, True),
        ("Yes please proceed", 3, None, False, False),
        ("Narbut", 3, {"Narbut": "narbutt"}, True, False),
        ("ABC", 3, {"ABC": "ay bee see"}, True, True),
    ],
)
def test_speech_insert_label_matches_tts_input(
    monkeypatch, text, min_words, pronunciations, draw_insert, inserted
):
    tic = UserSpeechInsert(text="[cough]", type="vocal_tic")
    speech_effects = SpeechEffectsResult(
        speech_insert=tic if draw_insert else None, dynamic_muffling_enabled=True
    )
    source_effects = SourceEffectsResult(
        speech_insert=UserSpeechInsert(text="Excuse me", type="non_directed_phrase")
    )
    channel_effects = ChannelEffectsResult(frame_drops_enabled=True, frame_drop_ms=20)
    schedule = Mock(return_value=(speech_effects, source_effects, channel_effects))
    monkeypatch.setattr("tau2.agent.base.voice.generate_turn_effects", schedule)

    audio = AudioData(
        data=b"\0\0" * 160,
        format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=16000),
    )
    synthesize = Mock(return_value=audio)
    monkeypatch.setattr("tau2.agent.base.voice.synthesize_voice", synthesize)
    config = SynthesisConfig()
    config.speech_effects_config.min_words_for_vocal_tics = min_words
    environment = SpeechEnvironment(voice_seed=42)
    mixin = VoiceMixin(
        voice_settings=VoiceSettings(
            synthesis_config=config,
            speech_environment=environment,
            pronunciation_map=pronunciations,
        )
    )
    batch_effects = Mock(return_value=audio)
    monkeypatch.setattr(mixin, "apply_batch_effects", batch_effects)

    result = mixin.synthesize_voice(
        UserMessage(role="user", content=text),
        SimpleNamespace(noise_generator=None),
        effects_turn_idx=7,
        add_background_noise=False,
    )

    expected_text = (pronunciations or {}).get(text, text)
    if inserted:
        expected_text = insert_speech_text(
            expected_text, tic, random.Random(49), min_words=min_words, in_turn=True
        )
    synthesize.assert_called_once()
    assert synthesize.call_args.kwargs["text"] == expected_text
    assert result.audio_script_gold == expected_text
    assert result.content == text
    assert result.speech_effects.speech_insert == (tic if inserted else None)
    assert result.speech_effects.dynamic_muffling_enabled
    assert result.source_effects == source_effects
    assert result.channel_effects == channel_effects
    assert batch_effects.call_args.kwargs["speech_effects"] == result.speech_effects
    schedule.assert_called_once_with(seed=42, turn_idx=7, synthesis_config=config)

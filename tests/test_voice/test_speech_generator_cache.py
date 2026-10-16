"""Tests for process-cached out-of-turn speech generation."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.audio_effects import UserSpeechInsert
from tau2.data_model.voice import ElevenLabsTTSConfig
from tau2.voice.synthesis.audio_effects import speech_generator


@pytest.fixture(autouse=True)
def empty_speech_cache():
    speech_generator.clear_out_of_turn_speech_cache()
    yield
    speech_generator.clear_out_of_turn_speech_cache()


def _audio() -> AudioData:
    return AudioData(
        data=b"\x00\x00" * 100,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=16000,
            channels=1,
        ),
    )


def _generator(voice_id: str = "voice_test"):
    return speech_generator.OutOfTurnSpeechGenerator(
        voice_id=voice_id,
        provider_config=ElevenLabsTTSConfig(voice_id=voice_id),
        target_sample_rate=16000,
    )


def test_reuses_fixed_clips_across_simulator_instances(monkeypatch):
    calls: list[str] = []

    def fake_synthesize_voice(text, **_kwargs):
        calls.append(text)
        return _audio()

    monkeypatch.setattr(speech_generator, "synthesize_voice", fake_synthesize_voice)
    items = [
        UserSpeechInsert(text=".[cough]", type="vocal_tic"),
        UserSpeechInsert(text="Hold on.", type="non_directed_phrase"),
    ]

    first = _generator()
    second = _generator()
    first.generate_all(items)
    second.generate_all(items)

    assert calls == [".[cough]", "Hold on."]
    assert first.generated_audio == second.generated_audio


def test_does_not_share_clips_between_persona_voices(monkeypatch):
    call_count = 0

    def fake_synthesize_voice(**_kwargs):
        nonlocal call_count
        call_count += 1
        return _audio()

    monkeypatch.setattr(speech_generator, "synthesize_voice", fake_synthesize_voice)
    item = UserSpeechInsert(text="One second.", type="non_directed_phrase")

    _generator("voice_one").generate_all([item])
    _generator("voice_two").generate_all([item])

    assert call_count == 2


def test_singleflights_concurrent_cache_misses(monkeypatch):
    call_count = 0
    call_lock = Lock()
    request_started = Event()
    release_request = Event()

    def blocking_synthesize_voice(**_kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        request_started.set()
        release_request.wait(timeout=5)
        return _audio()

    monkeypatch.setattr(
        speech_generator,
        "synthesize_voice",
        blocking_synthesize_voice,
    )
    item = UserSpeechInsert(text="One second.", type="non_directed_phrase")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_generator().generate_all, [item])
        assert request_started.wait(timeout=2)
        second = executor.submit(_generator().generate_all, [item])
        time.sleep(0.05)
        with call_lock:
            calls_while_blocked = call_count
        release_request.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert calls_while_blocked == 1
    assert call_count == 1


def test_materializes_cache_misses_sequentially(monkeypatch):
    active = 0
    peak = 0
    state_lock = Lock()

    def tracked_synthesize_voice(**_kwargs):
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with state_lock:
            active -= 1
        return _audio()

    monkeypatch.setattr(
        speech_generator,
        "synthesize_voice",
        tracked_synthesize_voice,
    )
    items = [
        UserSpeechInsert(text=f"Phrase {index}", type="non_directed_phrase")
        for index in range(4)
    ]

    _generator().generate_all(items)

    assert peak == 1


def test_failed_generation_is_retried_instead_of_cached(monkeypatch):
    call_count = 0

    def flaky_synthesize_voice(**_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("temporary failure")
        return _audio()

    monkeypatch.setattr(
        speech_generator,
        "synthesize_voice",
        flaky_synthesize_voice,
    )
    item = UserSpeechInsert(text="Try again.", type="non_directed_phrase")

    first = _generator()
    second = _generator()
    first.generate_all([item])
    second.generate_all([item])

    assert not first.has_audio(item)
    assert second.has_audio(item)
    assert call_count == 2

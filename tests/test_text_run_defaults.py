# Copyright Sierra
"""Modality-sensitive defaults for ordinary ``tau2 run`` invocations."""

from tau2.cli import _resolve_run_mode_defaults
from tau2.config import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_TEXT_MAX_CONCURRENCY,
    DEFAULT_TEXT_WORKERS,
)
from tau2.data_model.simulation import Score, TextRunConfig


def test_text_run_defaults_to_reward_and_eight_workers_at_ten_each():
    scores, concurrency, workers = _resolve_run_mode_defaults(
        audio_native=False,
        scores=None,
        max_concurrency=None,
        workers=None,
    )

    assert scores == {Score.REWARD}
    assert concurrency == DEFAULT_TEXT_MAX_CONCURRENCY == 10
    assert workers == DEFAULT_TEXT_WORKERS == 8

    config = TextRunConfig()
    assert config.scores == {Score.REWARD}
    assert config.max_concurrency == 10
    assert config.workers == 8


def test_voice_defaults_are_unchanged():
    scores, concurrency, workers = _resolve_run_mode_defaults(
        audio_native=True,
        scores=None,
        max_concurrency=None,
        workers=None,
    )

    assert scores == {Score.REWARD, Score.QUALITY, Score.NATIVENESS}
    assert concurrency == DEFAULT_MAX_CONCURRENCY == 3
    assert workers == 0


def test_explicit_run_settings_override_modality_defaults():
    scores, concurrency, workers = _resolve_run_mode_defaults(
        audio_native=False,
        scores="reward,quality",
        max_concurrency=4,
        workers=0,
    )

    assert scores == {Score.REWARD, Score.QUALITY}
    assert concurrency == 4
    assert workers == 0

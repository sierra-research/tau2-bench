# Copyright Sierra
"""Audio-quality survivors: the ingest row contract and agent-speech spans.

The packet builders shipped with the concluded audio-quality waves and are
gone; the row model still dispatches returned browser CSVs and the span
helpers still feed the audio-quality judge's packet input contract.
"""

import pytest

from tau2.annotation.models import AudioIssueRating, AudioQualityRow
from tau2.annotation.packets.audio_quality import (
    agent_speech_tick_spans,
    agent_speech_transcript,
)
from tau2.annotation.packets.forms import AUDIO_QUALITY_KIND, match_browser_kind
from tau2.data_model.message import AssistantMessage, Tick
from test_annotation.conftest import hi_sim

TICK_MS = 200


def _speech_tick(tick_id: int, utterance_id: str) -> Tick:
    return Tick(
        tick_id=tick_id,
        timestamp="2026-01-01T00:00:00",
        tick_duration_seconds=TICK_MS / 1000,
        agent_chunk=AssistantMessage(
            role="assistant",
            content="audio",
            contains_speech=True,
            utterance_ids=[utterance_id],
        ),
    )


def test_audio_quality_row_completion_contract():
    ratings = {
        dimension: "0"
        for dimension in (
            "mispronunciation",
            "word_substitution",
            "missing_word",
            "extra_or_hallucinated_word",
            "number_date_currency",
            "email_url_code",
            "punctuation_or_formatting",
            "acronym_brand_name",
            "clipped_or_garbled",
            "other",
            "intonation",
        )
    }
    row = AudioQualityRow(
        clip_id="clip_001", listened_fully=True, completed=True, **ratings
    )
    assert row.mispronunciation is AudioIssueRating.NO_ISSUE
    assert AudioQualityRow.from_cells(row.to_cells()) == row
    assert match_browser_kind(set(AudioQualityRow.headers())) == AUDIO_QUALITY_KIND

    with pytest.raises(ValueError, match="unrated dimensions"):
        AudioQualityRow(clip_id="clip_001", listened_fully=True, completed=True)
    with pytest.raises(ValueError, match="unreviewed LLM findings"):
        AudioQualityRow(
            clip_id="clip_001",
            listened_fully=True,
            completed=True,
            adjudication_required=True,
            llm_total_findings=1,
            **ratings,
        )


def test_agent_speech_spans_split_on_utterance_ids_and_skip_silence():
    sim = hi_sim("s1", "t1")
    sim.ticks = [
        _speech_tick(0, "u1"),
        _speech_tick(1, "u1"),
        Tick(
            tick_id=2,
            timestamp="2026-01-01T00:00:00",
            tick_duration_seconds=TICK_MS / 1000,
        ),
        Tick(
            tick_id=3,
            timestamp="2026-01-01T00:00:00",
            tick_duration_seconds=TICK_MS / 1000,
        ),
        _speech_tick(4, "u2"),
        _speech_tick(5, "u2"),
    ]

    assert agent_speech_tick_spans(sim) == [(0, 1), (4, 5)]


def test_agent_transcript_matches_agent_speech_spans_only():
    sim = hi_sim("s1", "t1")
    sim.ticks = [
        Tick(
            tick_id=0,
            timestamp="2026-01-01T00:00:00",
            tick_duration_seconds=TICK_MS / 1000,
            agent_chunk=AssistantMessage(
                role="assistant", content="not spoken", contains_speech=False
            ),
        ),
        _speech_tick(1, "u1").model_copy(
            update={
                "agent_chunk": AssistantMessage.voice(
                    content="Hola", utterance_ids=["u1"]
                )
            }
        ),
        _speech_tick(2, "u1").model_copy(
            update={
                "agent_chunk": AssistantMessage.voice(content=".", utterance_ids=["u1"])
            }
        ),
        _speech_tick(4, "u2").model_copy(
            update={
                "agent_chunk": AssistantMessage.voice(
                    content="Adiós.", utterance_ids=["u2"]
                )
            }
        ),
    ]

    assert agent_speech_transcript(sim) == ["Hola.", "Adiós."]

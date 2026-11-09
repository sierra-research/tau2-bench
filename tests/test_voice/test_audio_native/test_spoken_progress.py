"""Positions of placed word ends in received audio (SpokenProgress)."""

import pytest

from tau2.voice.audio_native.avartha_pipecat.discrete_time_adapter import (
    _SpokenTranscript,
)
from tau2.voice.audio_native.spoken_progress import SourceStream, placed_word_end


def tracker(*spans):
    """Received audio at half the source rate: one byte per two source samples."""
    progress = _SpokenTranscript()
    for start, end in spans:
        progress.add_audio((end - start) // 2, (start, end))
    return progress


def test_source_positions_map_onto_received_audio_through_gaps():
    progress = tracker((0, 100), (100, 100), (300, 400))
    # Inside a span, rounded up to the received byte that ends it.
    assert progress._received_position(1) == 1
    assert progress._received_position(100) == 50
    # In a gap (audio never received), at the next received audio.
    assert progress._received_position(200) == 50
    assert progress._received_position(350) == 75
    # Not received yet.
    assert progress._received_position(401) is None
    # Received positions back to source, rounded down; a zero-length span
    # contributes its end.
    assert progress._source_position(50) == 100
    assert progress._source_position(51) == 302
    assert progress._source_position(1000) == 400


def test_a_cut_before_any_placed_audio_played_keeps_no_cut_text():
    progress = _SpokenTranscript()
    progress.add_text("Hello there.")
    progress.add_audio(40)
    progress.add_audio(50, (100, 200))
    progress.get_transcript_for_audio(40)
    progress.interrupt("u-1")
    progress.add_word("Hello", 150)

    assert progress.take_reference_amendments() == []


def test_the_source_count_ends_at_a_sample_rate_change():
    stream = SourceStream()
    assert stream.take(480, 24000) == (0, 480)
    assert stream.take(240, 24000) == (480, 720)
    assert stream.take(160, 16000) is None
    assert stream.take(480, 24000) is None


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            {
                "pcm_stream_offset": 7,
                "pcm_sample_rate_hz": 8000,
                "pcm_boundary": "word_end",
            },
            (7, 8000),
        ),
        ({"pcm_stream_offset": 7, "pcm_sample_rate_hz": 8000}, None),
        (
            {
                "pcm_stream_offset": "7",
                "pcm_sample_rate_hz": 8000,
                "pcm_boundary": "word_end",
            },
            None,
        ),
    ],
)
def test_placed_word_end_parsing(data, expected):
    assert placed_word_end(data) == expected

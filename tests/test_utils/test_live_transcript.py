"""Playback-based Live groups and conservative caption association."""

import copy
import json
import wave

import pytest

from tau2.data_model.message import AssistantMessage, Tick, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.transcript import LiveTranscriptFragment, LiveTranscriptTick
from tau2.judges.run_evidence import load_audio_clips
from tau2.utils.transcript.live import require_live_alignment
from tau2.utils.transcript.views import (
    agent_messages,
    build_agent_turns,
    utterance_groups,
)
from tau2.voice.audio_native.tick_result import TickResult


def fragment(text, start, end, event="event", speaker="agent"):
    return LiveTranscriptFragment(
        event_id=event, text=text, start_ms=start, end_ms=end, speaker=speaker
    )


def timeline(fragments, speech, caller=(), length=30):
    return [
        Tick(
            tick_id=i,
            timestamp="2026-09-28T00:00:00",
            tick_duration_seconds=0.2,
            agent_chunk=AssistantMessage(
                role="assistant",
                content="",
                contains_speech=i in speech,
                raw_data={
                    "live_transcript": LiveTranscriptTick(
                        session_id="test-session",
                        fragments=fragments.get(i, []),
                        speech_intervals_ms=[(0, 200)] if i in speech else [],
                    ).model_dump()
                },
            ),
            user_chunk=UserMessage(
                role="user",
                content="caller" if i in caller else "",
                contains_speech=i in caller,
                utterance_ids=["caller"],
            ),
        )
        for i in range(length)
    ]


def metadata(message):
    return message.raw_data["live_segment"]


def simulation(ticks):
    return SimulationRun(
        id="test",
        task_id="0",
        seed=42,
        messages=[],
        ticks=ticks,
        start_time="2026-09-28T00:00:00",
        end_time="2026-09-28T00:00:06",
        duration=6,
        termination_reason=TerminationReason.AGENT_STOP,
    )


def audio_file(tmp_path):
    path = tmp_path / "both.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        # Distinguishable channels let the test assert exact copied samples.
        wav.writeframes(b"\x01\x00\x02\x00" * (6 * 8000))
    return path


def test_source_clock_drift_does_not_split_continuous_audio():
    ticks = timeline(
        {
            5: [fragment("十六万", 1000, 1400, "a")],
            8: [fragment("四千八百七十一", 100000, 102000, "b")],
        },
        range(5, 15),
    )
    before = copy.deepcopy(ticks)
    messages = agent_messages(ticks)
    assert [m.content for m in messages] == ["十六万四千八百七十一"]
    assert metadata(messages[0])["association_status"] == "stable"
    assert (
        utterance_groups(ticks, "agent")[0].start_tick,
        utterance_groups(ticks, "agent")[0].end_tick,
    ) == (5, 14)
    assert ticks == before


def test_audio_silence_splits_even_when_provider_time_is_continuous():
    ticks = timeline(
        {3: [fragment("First", 0, 200, "a")], 22: [fragment(" Second", 200, 400, "b")]},
        [*range(2, 7), *range(21, 28)],
    )
    messages = agent_messages(ticks)
    assert [m.content for m in messages] == ["First", " Second"]
    assert metadata(messages[1])["boundary"] == "audio_silence"
    require_live_alignment(ticks)


def test_late_batched_captions_preserve_text_and_missing_audio_flags():
    ticks = timeline(
        {29: [fragment("First", 1000, 1600, "a"), fragment("Second", 4000, 4600, "b")]},
        [5, 6, 7, 20, 21, 22],
    )
    messages = agent_messages(ticks)
    assert "".join(m.content for m in messages) == "FirstSecond"
    with pytest.raises(ValueError, match="Unresolved Live"):
        require_live_alignment(ticks)


def test_tick_straddling_padded_windows_belongs_to_earlier_clip():
    ticks = timeline(
        {
            5: [fragment("First", 0, 200, "a")],
            8: [fragment(" boundary", 200, 400, "b")],
            12: [fragment("Second", 400, 600, "c")],
        },
        [5, 12],
    )
    # First speech ends at 1.10 s; second starts at 2.60 s: a 1.50 s gap.
    ticks[5].agent_chunk.raw_data["live_transcript"]["speech_intervals_ms"] = [(0, 100)]
    ticks[12].agent_chunk.raw_data["live_transcript"]["speech_intervals_ms"] = []
    ticks[13].agent_chunk.raw_data["live_transcript"]["speech_intervals_ms"] = [
        (0, 100)
    ]
    # Boundary is 1.85 s, inside tick 9 (1.8–2.0 s).
    ticks[9].agent_chunk.raw_data["live_transcript"]["fragments"] = ticks[
        8
    ].agent_chunk.raw_data["live_transcript"]["fragments"]
    ticks[8].agent_chunk.raw_data["live_transcript"]["fragments"] = []
    messages = agent_messages(ticks)
    assert [m.content for m in messages] == ["First boundary", "Second"]
    assert metadata(messages[0])["earlier_clip_tie_tick_ids"] == [9]
    assert metadata(messages[0])["playback_end_us"] == 1850000
    assert metadata(messages[1])["playback_start_us"] == 1850000
    require_live_alignment(ticks)


def test_text_in_long_silence_is_retained_without_blocking_judging():
    ticks = timeline(
        {
            5: [fragment("First", 0, 200, "a")],
            14: [fragment(" orphan", 200, 400, "b")],
            23: [fragment("Second", 400, 600, "c")],
        },
        [5, 23],
    )
    assert "".join(m.content for m in agent_messages(ticks)) == "First orphanSecond"
    assert metadata(agent_messages(ticks)[0])["outside_window_tick_ids"] == [14]
    require_live_alignment(ticks)


def test_short_pause_and_backchannel_do_not_split_or_prove_truncation():
    ticks = timeline(
        {
            5: [fragment("178만", 1000, 1600, "a")],
            12: [fragment(" 6천 원", 2200, 2800, "b")],
        },
        [5, 6, 7, 11, 12, 13],
        caller=[12],
    )
    turns = build_agent_turns(simulation(ticks))
    assert len(turns) == 1
    assert turns[0].text == "178만 6천 원"
    assert turns[0].overlapping_user_text == "caller"
    assert not turns[0].interrupted


def test_caller_speech_does_not_override_approved_silence_threshold():
    ticks = timeline(
        {
            5: [fragment("First", 1000, 1600, "a")],
            12: [fragment("Second", 2400, 3000, "b")],
        },
        [5, 6, 7, 12, 13, 14],
        caller=[9, 10],
    )
    messages = agent_messages(ticks)
    assert len(messages) == 1
    assert messages[0].content == "FirstSecond"


def test_unmatched_audio_is_retained_and_blocks_judging():
    ticks = timeline({}, [5, 6])
    messages = agent_messages(ticks)
    assert len(messages) == 1 and messages[0].content == ""
    assert metadata(messages[0])["association_status"] == "unmatched_audio"
    with pytest.raises(ValueError, match="no transcript"):
        require_live_alignment(ticks)


def test_unmatched_text_is_preserved_and_blocks_judging():
    ticks = timeline({5: [fragment("Missing audio", 0, 200)]}, [])
    assert agent_messages(ticks)[0].content == "Missing audio"
    assert metadata(agent_messages(ticks)[0])["association_status"] == "unmatched_text"
    with pytest.raises(ValueError, match="no recorded speech"):
        require_live_alignment(ticks)


@pytest.mark.parametrize("gap_ms,count", [(1499, 1), (1500, 2), (1501, 2)])
def test_exact_approved_silence_threshold(gap_ms, count):
    ticks = timeline({}, [])
    for a, b in [(0, 100), (100 + gap_ms, 200 + gap_ms)]:
        for i in range(a // 200, (b - 1) // 200 + 1):
            ticks[i].agent_chunk.raw_data["live_transcript"][
                "speech_intervals_ms"
            ].append((max(a - i * 200, 0), min(b - i * 200, 200)))
    messages = agent_messages(ticks)
    assert len(messages) == count
    assert all(metadata(m)["silence_threshold_ms"] == 1500 for m in messages)


def test_ids_are_deterministic_and_independent_of_wall_clock():
    ticks = timeline({5: [fragment("Hello", 1000, 1600)]}, range(5, 12))
    ids = [m.utterance_ids for m in agent_messages(ticks)]
    for t in ticks:
        t.timestamp = "2099-01-01T00:00:00"
        t.wall_clock_duration_seconds = 60
    assert [m.utterance_ids for m in agent_messages(ticks)] == ids


def test_session_boundary_cannot_merge_audio():
    ticks = timeline(
        {5: [fragment("One", 0, 200)], 6: [fragment("Two", 0, 200)]}, [5, 6]
    )
    for t in ticks[6:]:
        t.agent_chunk.raw_data["live_transcript"]["session_id"] = "next-session"
    assert [m.content for m in agent_messages(ticks)] == ["One", "Two"]


def test_provider_timestamp_regression_does_not_reorder_text():
    ticks = timeline(
        {5: [fragment("Later", 2000, 2200, "a"), fragment("Earlier", 1000, 1200, "b")]},
        [5, 6],
    )
    assert agent_messages(ticks)[0].content == "LaterEarlier"


def test_native_ids_and_legacy_no_id_behavior_unchanged():
    ticks = timeline({}, [])[:3]
    for i, t in enumerate(ticks):
        t.agent_chunk = AssistantMessage(
            role="assistant",
            content=str(i),
            utterance_ids=["native"] if i < 2 else None,
        )
    assert [m.content for m in agent_messages(ticks)] == ["01", "2"]


def test_metadata_survives_serialization():
    result = TickResult(
        tick_number=1,
        audio_sent_bytes=0,
        audio_sent_duration_ms=0,
        events=[object()],
        live_transcript=LiveTranscriptTick(
            session_id="s", fragments=[fragment("안녕", 0, 200)]
        ),
    )
    saved = json.loads(result.model_dump_json())
    assert "events" not in saved
    assert saved["live_transcript"]["fragments"][0]["text"] == "안녕"


def test_fidelity_uses_exact_subtick_audio_bounds_and_shared_id(tmp_path):
    import base64
    import io

    ticks = timeline({5: [fragment("Recorded text", 80000, 81000)]}, [])
    ticks[5].agent_chunk.raw_data["live_transcript"]["speech_intervals_ms"] = [
        (20, 180)
    ]
    sim = simulation(ticks)
    clip = load_audio_clips(audio_file(tmp_path), sim)[0]
    assert clip.segment_id == build_agent_turns(sim)[0].segment_id
    assert clip.duration_seconds == 1.66
    with wave.open(io.BytesIO(base64.b64decode(clip.wav_base64)), "rb") as wav:
        assert wav.readframes(wav.getnframes()) == b"\x02\x00" * 13280


def test_silent_harness_greeting_is_not_fidelity_opportunity(tmp_path):
    ticks = timeline({}, [])
    ticks[0].agent_chunk = AssistantMessage(
        role="assistant", content="Hello", contains_speech=False
    )
    assert load_audio_clips(audio_file(tmp_path), simulation(ticks)) == []


@pytest.mark.parametrize("cut_tick", [9, 10])
def test_explicit_truncation_follows_playback_not_caption_clock(cut_tick):
    ticks = timeline(
        {
            5: [fragment("First", 100000, 102000, "a")],
            22: [fragment("Second", 104000, 105000, "b")],
        },
        [*range(5, 10), *range(21, 28)],
    )
    ticks[cut_tick].agent_chunk.raw_data["was_truncated"] = True
    assert [t.interrupted for t in build_agent_turns(simulation(ticks))] == [
        True,
        False,
    ]


def test_invalid_activity_interval_rejected():
    ticks = timeline({}, [])
    ticks[0].agent_chunk.raw_data["live_transcript"]["speech_intervals_ms"] = [(0, 201)]
    with pytest.raises(ValueError, match="outside"):
        agent_messages(ticks)


def test_provider_timestamp_changes_do_not_change_group_ids_or_bounds():
    ticks = timeline({5: [fragment("Hello", 1000, 1200)]}, [5, 6])
    before = agent_messages(ticks)[0]
    f = ticks[5].agent_chunk.raw_data["live_transcript"]["fragments"][0]
    f.update(start_ms=900000, end_ms=1, ingress_audio_bytes=999999)
    after = agent_messages(ticks)[0]
    assert after.utterance_ids == before.utterance_ids
    assert metadata(after)["playback_start_us"] == metadata(before)["playback_start_us"]
    assert metadata(after)["playback_end_us"] == metadata(before)["playback_end_us"]


def test_padding_stops_at_session_edges():
    ticks = timeline(
        {0: [fragment("Start", 0, 200)], 29: [fragment("End", 0, 200)]}, [0, 29]
    )
    messages = agent_messages(ticks)
    assert metadata(messages[0])["playback_start_us"] == 0
    assert metadata(messages[1])["playback_end_us"] == 6000000


@pytest.mark.parametrize("contains_speech", [True, False])
def test_non_live_audio_and_missing_ids_do_not_enable_padding(
    tmp_path, contains_speech
):
    ticks = timeline({}, [])
    for t in ticks:
        t.agent_chunk = None
    ticks[5].agent_chunk = AssistantMessage(
        role="assistant", content="Other provider", contains_speech=contains_speech
    )
    clip = load_audio_clips(audio_file(tmp_path), simulation(ticks))[0]
    assert clip.duration_seconds == 0.2
    assert clip.segment_id is None


def test_legacy_live_recovery_uses_right_channel_and_preserves_source(tmp_path):
    import struct

    from tau2.utils.transcript.live_audio import recover_live_audio

    ticks = timeline({}, [], length=3)
    for i, tick in enumerate(ticks):
        tick.agent_chunk.raw_data = {"tick_number": i}
        tick.agent_chunk.content = "Recorded" if i == 1 else ""
    ticks[0].agent_chunk.content = "Unspoken harness greeting"
    ticks[0].agent_chunk.raw_data = None
    sim = simulation(ticks)
    before = sim.model_dump_json()
    path = tmp_path / "legacy.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        # Caller channel is active throughout; only the middle agent tick speaks.
        for i in range(3):
            wav.writeframes(struct.pack("<hh", 32000, 64 if i == 1 else 0) * 1600)
    recovered = recover_live_audio(sim, path, provider="openai_live")
    assert sim.model_dump_json() == before
    messages = agent_messages(recovered.ticks)
    assert [m.content for m in messages] == ["Recorded"]
    assert metadata(messages[0])["evidence_origins"] == ["recorded_audio"]
    assert metadata(messages[0])["speech_start_us"] == 200000
    assert metadata(messages[0])["speech_end_us"] == 400000
    assert metadata(messages[0])["fragments"][0]["start_ms"] is None
    require_live_alignment(recovered.ticks)
    assert load_audio_clips(path, recovered)[0].duration_seconds == 0.6


def test_legacy_recovery_requires_explicit_live_provider(tmp_path):
    from tau2.utils.transcript.live_audio import recover_live_audio

    sim = simulation(timeline({}, []))
    for tick in sim.ticks:
        tick.agent_chunk.raw_data = {"tick_number": tick.tick_id}
    assert recover_live_audio(sim, tmp_path / "missing.wav", provider="openai") is sim


def test_recorded_live_evidence_is_not_replaced_by_recovery(tmp_path):
    from tau2.utils.transcript.live_audio import recover_live_audio

    sim = simulation(timeline({5: [fragment("Saved", 1000, 1200)]}, [5]))
    assert (
        recover_live_audio(sim, tmp_path / "missing.wav", provider="openai_live") is sim
    )


@pytest.mark.parametrize("measure", ["naturalness", "speech_fidelity"])
def test_judges_skip_only_uncaptioned_live_group(tmp_path, monkeypatch, measure):
    from types import SimpleNamespace

    from tau2.data_model.tasks import Task
    from tau2.judges.catalog import load_catalog
    from tau2.judges.criteria import CriteriaJudge
    from tau2.judges.fidelity import FidelityJudge
    from tau2.judges.run_evidence import AudioEvidence, RunEvidence

    ticks = timeline({3: [fragment("First", 0, 200)]}, [*range(2, 7), *range(21, 28)])
    sim = simulation(ticks)
    path = audio_file(tmp_path)
    response = (
        '{"findings": []}'
        if measure == "speech_fidelity"
        else json.dumps(
            {
                "results": [
                    {
                        "factor_id": "naturalness",
                        "opportunity": True,
                        "violated": False,
                        "severity": 0,
                        "reasoning": "Natural",
                        "quote": "",
                    }
                ]
            }
        )
    )
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=response)

    adapter = "fidelity" if measure == "speech_fidelity" else "criteria"
    monkeypatch.setattr(f"tau2.judges.{adapter}.generate", generate)
    contract_id = (
        "speech-fidelity-v7"
        if measure == "speech_fidelity"
        else "naturalness-v17-rubric-v21-language-specific-draft2"
    )
    contract = next(c for c in load_catalog().contracts if c.contract_id == contract_id)
    judge = (
        FidelityJudge(contract)
        if measure == "speech_fidelity"
        else CriteriaJudge(contract)
    )
    evidence = RunEvidence(
        simulation=sim,
        task=Task(id="0", user_scenario={"instructions": "Fixture"}),
        language="es",
        audio=AudioEvidence(path=path, sha256="a" * 64, error=None),
    )
    result = judge.evaluate(evidence, measure)
    assert len(calls) == len(result.units) == len(result.unit_errors) == 1
    assert result.unit_errors[0].kind == "input"
    assert "no transcript" in result.unit_errors[0].error
    assert result.units[0].unit_id != result.unit_errors[0].unit_id
    monkeypatch.setattr(
        f"tau2.judges.{adapter}.generate",
        lambda **kw: pytest.fail("Successful unit was repeated"),
    )
    assert judge.resume(evidence, measure, result) == result


def test_all_uncaptioned_audio_remains_unjudged(tmp_path, monkeypatch):
    from tau2.data_model.tasks import Task
    from tau2.judges.catalog import load_catalog
    from tau2.judges.fidelity import FidelityJudge
    from tau2.judges.run_evidence import AudioEvidence, RunEvidence

    sim = simulation(timeline({}, range(2, 7)))
    monkeypatch.setattr(
        "tau2.judges.fidelity.generate", lambda **kw: pytest.fail("No reference")
    )
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    result = FidelityJudge(contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="0", user_scenario={"instructions": "Fixture"}),
            language="es",
            audio=AudioEvidence(path=audio_file(tmp_path), sha256="a" * 64, error=None),
        ),
        "speech_fidelity",
    )
    assert not result.units and len(result.unit_errors) == 1

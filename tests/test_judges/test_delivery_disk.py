# Copyright Sierra
"""Delivery rejudge-from-disk: channel split, tick slicing, the harness
``wav_source`` seam, and the streaming/CLI integration.

Synthetic stereo fixture: the caller channel (left) carries a LOUD constant
marker and the agent channel (right) a distinct constant per tick — so a
wrong-channel split, an off-by-one tick slice, or a broken timeline all fail
on exact sample values, not on plausible-looking audio.
"""

import base64
import io
import wave
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.simulation import (
    DeliveryJudgeSettings,
    DeliveryUtteranceResult,
    JudgeOutcome,
    Results,
)
from tau2.judges.cli import add_judges_args, run_judges_rejudge
from tau2.judges.delivery.disk_audio import (
    DiskAudioError,
    SimAgentAudio,
    find_both_wav,
    sim_tick_duration_ms,
    slice_tick_span,
    split_agent_channel,
)
from tau2.judges.delivery.harness import evaluate_delivery
from tau2.judges.export import JudgeStreamStats, iter_judged_sims_detailed
from tau2.voice.utils.audio_io import save_wav_file
from tau2.voice.utils.audio_preprocessing import convert_to_stereo
from test_judges.conftest import hi_sim, make_hi_results

RATE = 8000
TICK_MS = 200
SPT = RATE * TICK_MS // 1000  # samples per tick

# Agent-channel value per tick: ticks 0-1 = utterance u0, tick 2 = quiet
# (caller speaking), tick 3 = utterance u1.
AGENT_TICK_VALUES = [1000, 2000, 10, 4000]
USER_MARKER = 30000  # loud constant on the caller channel


def _mono(values_per_tick) -> AudioData:
    data = np.concatenate([np.full(SPT, v, dtype=np.int16) for v in values_per_tick])
    return AudioData(
        data=data.tobytes(),
        format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=RATE),
    )


def _stereo(agent_values) -> AudioData:
    user = _mono([USER_MARKER] * len(agent_values))
    return convert_to_stereo(user, _mono(agent_values))


def _speech_tick(tick_id, content, utterance_ids):
    return Tick(
        tick_id=tick_id,
        timestamp="2026-01-01T00:00:00",
        tick_duration_seconds=TICK_MS / 1000.0,
        agent_chunk=AssistantMessage(
            role="assistant",
            content=content,
            audio_script_gold=content,
            utterance_ids=utterance_ids,
        ),
    )


def _voice_ticks():
    # Spans: [(0, 1), (3, 3)]; tick 2 is the caller's turn (no agent chunk).
    return [
        _speech_tick(0, "aapka ", ["u0"]),
        _speech_tick(1, "code JMO1MG hai", ["u0"]),
        Tick(
            tick_id=2,
            timestamp="2026-01-01T00:00:00",
            tick_duration_seconds=TICK_MS / 1000.0,
        ),
        _speech_tick(3, "dhanyavaad", ["u1"]),
    ]


def _disk_sim(sim_id="s1", task_id="t1"):
    sim = hi_sim(sim_id, task_id)
    sim.ticks = _voice_ticks()
    return sim


def _write_both_wav(results_dir: Path, sim, agent_values) -> Path:
    audio_dir = (
        results_dir / "tasks" / f"task_{sim.task_id}" / f"sim_{sim.id}" / "audio"
    )
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / "both.wav"
    save_wav_file(_stereo(agent_values), path)
    return path


def _wav_values(wav_b64: str) -> np.ndarray:
    with wave.open(io.BytesIO(base64.b64decode(wav_b64))) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == RATE
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


@pytest.fixture
def disk_run(tmp_path):
    """A dir-format hi run whose sim s1 has ticks + a tick-aligned both.wav."""
    sim = _disk_sim()
    run_dir = make_hi_results(tmp_path, [sim])
    _write_both_wav(run_dir, sim, AGENT_TICK_VALUES)
    return run_dir, sim


# ---------------------------------------------------------------------------
# Channel split + tick slicing (pure)
# ---------------------------------------------------------------------------


def test_split_agent_channel_takes_the_right_channel():
    agent = split_agent_channel(_stereo(AGENT_TICK_VALUES))
    assert agent.format.channels == 1
    values = np.frombuffer(agent.data, dtype=np.int16)
    assert len(values) == 4 * SPT
    # The caller marker must NOT appear: a wrong split fails here instantly.
    assert set(np.unique(values)) == set(AGENT_TICK_VALUES)


def test_split_agent_channel_rejects_mono_half_duplex():
    with pytest.raises(DiskAudioError, match="half-duplex"):
        split_agent_channel(_mono(AGENT_TICK_VALUES))


def test_split_agent_channel_rejects_companded():
    bad = AudioData(
        data=b"\x00" * 64,
        format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=RATE, channels=2),
    )
    with pytest.raises(DiskAudioError, match="companded"):
        split_agent_channel(bad)


def test_slice_tick_span_exact_and_clamped():
    agent = split_agent_channel(_stereo(AGENT_TICK_VALUES))
    piece = slice_tick_span(agent, 0, 1, TICK_MS)
    values = np.frombuffer(piece.data, dtype=np.int16)
    assert list(values[:SPT]) == [1000] * SPT
    assert list(values[SPT:]) == [2000] * SPT
    # End past the audio clamps (the merge end-pads; last tick may run short).
    clamped = slice_tick_span(agent, 3, 9, TICK_MS)
    assert np.frombuffer(clamped.data, dtype=np.int16).tolist() == [4000] * SPT


def test_slice_starting_past_audio_is_loud():
    agent = split_agent_channel(_stereo(AGENT_TICK_VALUES[:2]))
    with pytest.raises(DiskAudioError, match="does not match the ticks"):
        slice_tick_span(agent, 3, 3, TICK_MS)


# ---------------------------------------------------------------------------
# SimAgentAudio (per-utterance source from disk)
# ---------------------------------------------------------------------------


def test_sim_agent_audio_slices_by_utterance(disk_run):
    run_dir, sim = disk_run
    audio = SimAgentAudio(sim, run_dir)
    assert audio.spans == [(0, 1), (3, 3)]
    assert sim_tick_duration_ms(sim) == TICK_MS

    u0 = _wav_values(audio.wav_b64_for(0))
    assert list(np.unique(u0)) == [1000, 2000] and len(u0) == 2 * SPT
    u1 = _wav_values(audio.wav_b64_for(1))
    assert u1.tolist() == [4000] * SPT

    with pytest.raises(DiskAudioError, match="out of range"):
        audio.slice_utterance(2)


def test_sim_agent_audio_missing_wav_is_loud(tmp_path):
    sim = _disk_sim()
    run_dir = make_hi_results(tmp_path, [sim])
    assert find_both_wav(run_dir, sim) is None
    with pytest.raises(DiskAudioError, match="no both.wav"):
        SimAgentAudio(sim, run_dir)


def test_find_both_wav_supports_artifacts_layout(tmp_path):
    sim = _disk_sim()
    flat = tmp_path / "artifacts" / f"task_{sim.task_id}" / f"sim_{sim.id}"
    flat.mkdir(parents=True)
    save_wav_file(_stereo(AGENT_TICK_VALUES), flat / "both.wav")
    assert find_both_wav(tmp_path, sim) == flat / "both.wav"


# ---------------------------------------------------------------------------
# The harness seam: evaluate_delivery with disk audio
# ---------------------------------------------------------------------------


def _capture_judge(captured):
    def _judge(wav_b64, expected_text, *, utterance_idx, **kwargs):
        captured[utterance_idx] = (wav_b64, expected_text)
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx,
            expected_text=expected_text,
            outcome=JudgeOutcome.PASS,
        )

    return _judge


def test_evaluate_delivery_from_disk(disk_run, monkeypatch):
    run_dir, sim = disk_run
    captured = {}
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge",
        _capture_judge(captured),
    )
    info = evaluate_delivery(
        sim,
        "hi",
        settings=DeliveryJudgeSettings(sample_rate=1.0),
        wav_source=SimAgentAudio(sim, run_dir).wav_b64_for,
    )
    assert info is not None and info.num_judged == 2
    # The judge heard EXACTLY the tick-sliced agent audio per utterance.
    assert list(np.unique(_wav_values(captured[0][0]))) == [1000, 2000]
    assert _wav_values(captured[1][0]).tolist() == [4000] * SPT
    assert captured[0][1] == "aapka code JMO1MG hai"
    assert captured[1][1] == "dhanyavaad"


# ---------------------------------------------------------------------------
# Streaming integration (iter_judged_sims_detailed) + CLI
# ---------------------------------------------------------------------------


def _two_sim_run(tmp_path):
    """s1 has a both.wav; s2 is a voice sim whose audio is missing."""
    s1, s2 = _disk_sim("s1", "t1"), _disk_sim("s2", "t1")
    run_dir = make_hi_results(tmp_path, [s1, s2])
    _write_both_wav(run_dir, s1, AGENT_TICK_VALUES)
    return run_dir


def test_iter_judged_sims_delivery_from_disk(tmp_path, monkeypatch):
    run_dir = _two_sim_run(tmp_path)
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _capture_judge({})
    )
    stats = JudgeStreamStats()
    items = {
        j.sim.id: j
        for j in iter_judged_sims_detailed(
            run_dir,
            delivery_settings=DeliveryJudgeSettings(sample_rate=1.0),
            stats=stats,
        )
    }
    # Nativeness verdicts were complete -> reused; delivery judged from disk.
    assert stats.reused == 2 and stats.judged == 0
    assert stats.delivery_judged == 1
    assert stats.delivery_unavailable == 1
    assert stats.delivery_unavailable_sim_ids == ["s2"]
    # Freshly paid delivery verdicts must be persisted despite nativeness reuse.
    assert items["s1"].was_judged is True
    assert items["s1"].sim.delivery_info.num_judged == 2
    assert items["s2"].was_judged is False
    assert items["s2"].sim.delivery_info is None


def test_iter_judged_sims_reuses_stored_delivery(tmp_path, monkeypatch):
    run_dir = _two_sim_run(tmp_path)
    calls = {"n": 0}

    def _judge(wav_b64, expected_text, *, utterance_idx, **kwargs):
        calls["n"] += 1
        return DeliveryUtteranceResult(
            utterance_idx=utterance_idx, outcome=JudgeOutcome.PASS
        )

    monkeypatch.setattr("tau2.judges.delivery.harness.run_delivery_judge", _judge)
    settings = DeliveryJudgeSettings(sample_rate=1.0)

    from tau2.judges.export import save_judged_results

    first = {
        j.sim.id: j.sim
        for j in iter_judged_sims_detailed(run_dir, delivery_settings=settings)
        if j.was_judged
    }
    save_judged_results(run_dir, first)
    assert calls["n"] == 2

    stats = JudgeStreamStats()
    list(iter_judged_sims_detailed(run_dir, delivery_settings=settings, stats=stats))
    assert calls["n"] == 2  # no new judge calls
    assert stats.delivery_reused == 1 and stats.delivery_unavailable == 1


def test_iter_judged_sims_delivery_heals_error_verdicts(tmp_path, monkeypatch):
    """Stored delivery verdicts containing an ERROR utterance (failed judge
    call) are NOT reusable — the sim is re-judged so gap-filling heals the
    error instead of treating it as done forever."""
    from tau2.data_model.simulation import DeliveryInfo
    from tau2.judges.export import has_complete_delivery_verdicts

    sim = _disk_sim("s1", "t1")
    from tau2.judges.delivery.judge import DELIVERY_JUDGE_PROMPT_VERSION

    sim.delivery_info = DeliveryInfo(
        num_judged=1,
        num_errors=1,
        judge_prompt_version=DELIVERY_JUDGE_PROMPT_VERSION,
        utterance_results=[
            DeliveryUtteranceResult(
                utterance_idx=0, outcome=JudgeOutcome.ERROR, summary="api down"
            )
        ],
    )
    assert has_complete_delivery_verdicts(sim) is False
    run_dir = make_hi_results(tmp_path, [sim])
    _write_both_wav(run_dir, sim, AGENT_TICK_VALUES)

    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _capture_judge({})
    )
    stats = JudgeStreamStats()
    items = {
        j.sim.id: j
        for j in iter_judged_sims_detailed(
            run_dir,
            delivery_settings=DeliveryJudgeSettings(sample_rate=1.0),
            stats=stats,
        )
    }
    assert stats.delivery_judged == 1 and stats.delivery_reused == 0
    assert items["s1"].was_judged is True
    assert items["s1"].sim.delivery_info.num_errors == 0


def test_has_complete_delivery_verdicts_predicate():
    from tau2.data_model.simulation import DeliveryInfo
    from tau2.judges.export import has_complete_delivery_verdicts

    sim = _disk_sim()
    sim.delivery_info = None
    assert has_complete_delivery_verdicts(sim) is False
    # Empty result (judge never actually judged an utterance) -> re-judge.
    from tau2.judges.delivery.judge import DELIVERY_JUDGE_PROMPT_VERSION

    sim.delivery_info = DeliveryInfo()
    assert has_complete_delivery_verdicts(sim) is False
    sim.delivery_info = DeliveryInfo(
        num_judged=1,
        judge_prompt_version=DELIVERY_JUDGE_PROMPT_VERSION,
        utterance_results=[
            DeliveryUtteranceResult(utterance_idx=0, outcome=JudgeOutcome.PASS)
        ],
    )
    assert has_complete_delivery_verdicts(sim) is True
    # Complete verdicts from an OLDER delivery prompt -> stale, NOT reusable.
    sim.delivery_info.judge_prompt_version = "v6"
    assert has_complete_delivery_verdicts(sim) is False


def test_iter_judged_sims_delivery_judges_no_nativeness_sims(tmp_path, monkeypatch):
    """Delivery is defined for every language: a voice sim with NO nativeness
    axis (e.g. an English run) is still delivery-judged from disk and yielded
    as a delivery-only JudgedSim for persistence."""
    sim = _disk_sim("en1", "t1")
    sim.nativeness_info = None  # English: nativeness never applies
    run_dir = make_hi_results(tmp_path, [sim])
    _write_both_wav(run_dir, sim, AGENT_TICK_VALUES)

    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _capture_judge({})
    )
    stats = JudgeStreamStats()
    items = list(
        iter_judged_sims_detailed(
            run_dir,
            delivery_settings=DeliveryJudgeSettings(sample_rate=1.0),
            stats=stats,
        )
    )
    assert stats.delivery_judged == 1
    assert (stats.judged, stats.reused, stats.skipped) == (0, 0, 0)
    assert len(items) == 1 and items[0].was_judged is True
    assert items[0].sim.nativeness_info is None
    assert items[0].sim.delivery_info is not None
    assert items[0].sim.delivery_info.num_judged == 2

    # Without delivery settings the same sim is not part of any axis and is
    # not yielded (the nativeness-only stream shape is unchanged).
    assert list(iter_judged_sims_detailed(run_dir)) == []


def test_rejudge_cli_delivery_flag(tmp_path, monkeypatch):
    run_dir = _two_sim_run(tmp_path)
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _capture_judge({})
    )
    run_judges_rejudge(
        Namespace(
            results=[str(run_dir)],
            rejudge=False,
            judge_model=None,
            judge_args=None,
            max_concurrency=2,
            output=None,
            delivery=True,
            delivery_sample_rate=1.0,
            delivery_model=None,
        )
    )
    stored = {s.id: s for s in Results.load(run_dir).simulations}
    assert stored["s1"].delivery_info is not None
    assert stored["s1"].delivery_info.num_judged == 2
    assert stored["s2"].delivery_info is None


def test_rejudge_cli_delivery_only_forces_delivery_keeps_nativeness(
    tmp_path, monkeypatch
):
    """--rejudge --delivery --delivery-only re-judges the delivery axis while
    the stored nativeness verdicts stream through untouched (the nativeness
    judge is never invoked — it is not even mocked here)."""
    run_dir = _two_sim_run(tmp_path)
    monkeypatch.setattr(
        "tau2.judges.delivery.harness.run_delivery_judge", _capture_judge({})
    )
    run_judges_rejudge(
        Namespace(
            results=[str(run_dir)],
            rejudge=True,
            judge_model=None,
            judge_args=None,
            max_concurrency=2,
            output=None,
            delivery=True,
            delivery_only=True,
            delivery_sample_rate=1.0,
            delivery_model=None,
        )
    )
    stored = {s.id: s for s in Results.load(run_dir).simulations}
    assert stored["s1"].delivery_info is not None
    assert stored["s1"].delivery_info.num_judged == 2
    assert stored["s1"].nativeness_info.judge_model == "stored-judge"
    assert stored["s2"].nativeness_info.judge_model == "stored-judge"


def test_rejudge_cli_delivery_only_requires_delivery(tmp_path):
    run_dir = _two_sim_run(tmp_path)
    with pytest.raises(SystemExit, match="--delivery-only requires --delivery"):
        run_judges_rejudge(
            Namespace(
                results=[str(run_dir)],
                rejudge=True,
                judge_model=None,
                judge_args=None,
                max_concurrency=2,
                output=None,
                delivery=False,
                delivery_only=True,
                delivery_sample_rate=1.0,
                delivery_model=None,
            )
        )


def test_cli_arg_shapes_parse():
    import argparse

    parser = argparse.ArgumentParser(prog="tau2 judges")
    add_judges_args(parser)
    for argv in (
        ["rejudge", "r", "--delivery", "--delivery-sample-rate", "0.5"],
        ["rejudge", "r", "--delivery", "--delivery-model", "m"],
        ["rejudge", "r", "--rejudge", "--delivery", "--delivery-only"],
    ):
        args = parser.parse_args(argv)
        assert callable(args.func)

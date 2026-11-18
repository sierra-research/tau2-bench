"""A played-word-end adapter's clip cut at its end stops at its last text tick.

avartha_pipecat shows a word on the tick whose played audio carries its end. A
cut that catches a long token (a code read digit by digit) mid-sound leaves the
token out of the recorded text while its partial audio stays in the group's
ticks. When the line's last cut comes at or after its last text tick, the clip
ends at that tick and is judged as interrupted. Other adapters keep the full
tick span.
"""

import base64
import io
import json
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from tau2.data_model.judges import ImplementationFamily
from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.judges.catalog import load_catalog
from tau2.judges.fidelity import FidelityJudge
from tau2.judges.run_evidence import (
    PLAYED_WORD_END_TEXT_PROVIDERS,
    RunEvidence,
    load_audio_clips,
    recorded_provider,
)
from tau2.utils.transcript.views import text_covered_tick_span, utterance_groups

PIPECAT = "avartha_pipecat"
RATE = 8000
TICK = 0.2

# A recorded avartha_pipecat cut (Portuguese airline call, ticks renumbered
# from 0). The line was "E os 930 reais já voltaram pro gift card 6941833.";
# the caller cut while the agent read the code digit by digit, so "card" is
# the last word shown (tick 15) while speech continues through tick 22 and the
# cut tick 23 plays nothing. The next reply starts after 2.4 s of silence.
# (content, utterance_id, contains_speech, was_truncated)
RECORDED_CUT = [
    (None, "u17", True, False),
    (" E os", "u17", True, False),
    *[(None, "u17", True, False)] * 6,
    (" 930 reais", "u17", True, False),
    (" já", "u17", True, False),
    (None, "u17", True, False),
    (" voltaram", "u17", True, False),
    (" pro", "u17", True, False),
    (" gift", "u17", True, False),
    (None, "u17", True, False),
    (" card", "u17", True, False),
    *[(None, "u17", True, False)] * 7,
    (None, "u17", False, True),
    *[(None, None, False, False)] * 11,
    (" 6941833,", "u18", True, False),
    (None, "u18", True, False),
]
LAST_WORD_TICK = 15
CUT_TICK = 23


def _ticks(rows, gold=None):
    ticks = []
    for i, (content, uid, speech, cut) in enumerate(rows):
        raw = {"was_truncated": True} if cut else {}
        ticks.append(
            Tick(
                tick_id=i,
                timestamp="t",
                tick_duration_seconds=TICK,
                agent_chunk=AssistantMessage(
                    role="assistant",
                    content=content,
                    contains_speech=speech,
                    utterance_ids=[uid] if uid else None,
                    raw_data=raw,
                    # Markup golds merge only alongside chunk audio.
                    audio_content=base64.b64encode(b"\0").decode() if gold else None,
                    audio_script_gold=gold(i) if gold and uid else None,
                ),
            )
        )
    return ticks


def _simulation(ticks):
    return SimulationRun(
        id="s",
        task_id="t",
        trial=0,
        start_time="a",
        end_time="b",
        duration=len(ticks) * TICK,
        termination_reason=TerminationReason.MAX_STEPS,
        ticks=ticks,
    )


def _audio(tmp_path, rows, path=None):
    """Agent channel: tick index i as a constant sample where speech played."""
    samples = round(TICK * RATE)
    agent = np.concatenate(
        [
            np.full(samples, (i + 1) if speech else 0, dtype="<i2")
            for i, (_, _, speech, _) in enumerate(rows)
        ]
    )
    stereo = np.stack([np.zeros_like(agent), agent], axis=1)
    path = path or tmp_path / "both.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(stereo.tobytes())
    return path


def _clip_ticks(clip):
    """Distinct tick markers carried by a clip's samples, in order."""
    with wave.open(io.BytesIO(base64.b64decode(clip.wav_base64))) as f:
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return sorted({int(v) - 1 for v in pcm if v})


def test_recorded_cut_clip_ends_with_its_last_shown_word(tmp_path):
    sim = _simulation(_ticks(RECORDED_CUT))
    clips = load_audio_clips(_audio(tmp_path, RECORDED_CUT), sim, provider=PIPECAT)
    cut, following = clips
    assert cut.reference == "E os 930 reais já voltaram pro gift card"
    # Speech through "card"'s tick; the partly read code is not in the clip.
    assert _clip_ticks(cut) == list(range(0, LAST_WORD_TICK + 1))
    assert cut.duration_seconds == pytest.approx((LAST_WORD_TICK + 1) * TICK)
    # The cut survives merging the group's chunks.
    assert cut.interrupted is True
    assert following.reference == "6941833,"
    assert following.interrupted is False


def test_group_span_and_delivered_span():
    ticks = _ticks(RECORDED_CUT)
    cut, following = utterance_groups(ticks, "agent")[:2]
    assert (cut.start_tick, cut.end_tick) == (0, CUT_TICK)
    assert text_covered_tick_span(ticks, cut) == (0, LAST_WORD_TICK)
    assert text_covered_tick_span(ticks, following) == (
        following.start_tick,
        following.end_tick,
    )


def test_uncut_group_keeps_its_full_span(tmp_path):
    rows = [(r[0], r[1], r[2], False) for r in RECORDED_CUT]
    clips = load_audio_clips(
        _audio(tmp_path, rows), _simulation(_ticks(rows)), provider=PIPECAT
    )
    assert _clip_ticks(clips[0]) == list(range(0, CUT_TICK))
    assert clips[0].duration_seconds == pytest.approx((CUT_TICK + 1) * TICK)
    assert clips[0].interrupted is False


def test_word_kept_on_the_cut_tick_keeps_the_cut_tick(tmp_path):
    """A word the cut caught mostly played is shown on the cut tick."""
    rows = list(RECORDED_CUT)
    rows[CUT_TICK] = (" 69", "u17", False, True)
    clip = load_audio_clips(
        _audio(tmp_path, rows), _simulation(_ticks(rows)), provider=PIPECAT
    )[0]
    assert clip.reference.endswith("gift card 69")
    assert clip.duration_seconds == pytest.approx((CUT_TICK + 1) * TICK)
    assert clip.interrupted is True


def test_cut_on_a_middle_chunk_marks_the_merged_clip_interrupted(tmp_path):
    """An explicit cut on any chunk of a multi-chunk group is an interruption."""
    rows = [
        (" One", "u", True, False),
        (" two", "u", True, True),
        (" three", "u", True, False),
    ]
    clip = load_audio_clips(
        _audio(tmp_path, rows), _simulation(_ticks(rows)), provider=PIPECAT
    )[0]
    assert clip.reference == "One two three"
    assert clip.interrupted is True
    assert clip.duration_seconds == pytest.approx(3 * TICK)


def test_markup_gold_groups_keep_their_full_span(tmp_path):
    """Gold references come from delivered synthesis chunks, not tick text."""
    rows = [
        (" One", "u", True, False),
        (None, "u", True, False),
        (None, "u", False, True),
    ]

    def gold(_):
        return (
            '<message uuid="m" active="0">'
            "<chunk id=0>One two</chunk><chunk id=1> three</chunk></message>"
        )

    ticks = _ticks(rows, gold=gold)
    group = utterance_groups(ticks, "agent")[0]
    assert text_covered_tick_span(ticks, group) == (0, 2)
    clip = load_audio_clips(
        _audio(tmp_path, rows), _simulation(ticks), provider=PIPECAT
    )[0]
    assert clip.reference == "One two"
    assert clip.duration_seconds == pytest.approx(3 * TICK)
    assert clip.interrupted is True


def test_cut_group_without_text_keeps_its_span():
    rows = [(None, "u", True, False), (None, "u", False, True)]
    ticks = _ticks(rows)
    group = utterance_groups(ticks, "agent")[0]
    assert text_covered_tick_span(ticks, group) == (0, 1)


@pytest.mark.parametrize("provider", [None, "gemini", "openai"])
def test_other_adapters_keep_the_full_cut_span(tmp_path, provider):
    """Proportional transcripts can show text ahead of its audio: no trim."""
    sim = _simulation(_ticks(RECORDED_CUT))
    cut = load_audio_clips(_audio(tmp_path, RECORDED_CUT), sim, provider=provider)[0]
    assert _clip_ticks(cut) == list(range(0, CUT_TICK))
    assert cut.duration_seconds == pytest.approx((CUT_TICK + 1) * TICK)
    # The recorded cut is still an interruption for every adapter.
    assert cut.interrupted is True


def test_fidelity_contract_names_the_cut_clip_rule():
    """Verdicts judged before the rule record the retired implementation."""
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    assert contract.implementation == ImplementationFamily.FIDELITY
    assert ImplementationFamily.FIDELITY != ImplementationFamily.FIDELITY_V6
    assert (
        ImplementationFamily("speech-fidelity-v6") is ImplementationFamily.FIDELITY_V6
    )


def test_speech_after_a_mid_line_cut_keeps_the_full_span(tmp_path):
    """Only a cut at or after the last text tick ends the clip there."""
    rows = [
        (" One", "u", True, False),
        (" two", "u", True, True),
        (" three", "u", True, False),
        (None, "u", True, False),
    ]
    ticks = _ticks(rows)
    assert text_covered_tick_span(ticks, utterance_groups(ticks, "agent")[0]) == (0, 3)
    clip = load_audio_clips(
        _audio(tmp_path, rows), _simulation(ticks), provider=PIPECAT
    )[0]
    assert clip.duration_seconds == pytest.approx(4 * TICK)
    assert clip.interrupted is True


def test_fidelity_judge_reads_the_recorded_provider(tmp_path, monkeypatch):
    """RunEvidence carries the provider SavedRun reads from run metadata."""
    sim = _simulation(_ticks(RECORDED_CUT))
    _audio(tmp_path, RECORDED_CUT, tmp_path / "tasks/task_t/sim_s/audio/both.wav")
    monkeypatch.setattr(
        "tau2.judges.fidelity.generate",
        lambda **_: SimpleNamespace(content=json.dumps({"findings": []})),
    )
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )

    def first_clip(provider):
        result = FidelityJudge(contract).evaluate(
            RunEvidence(
                simulation=sim,
                task=Task(id="t", user_scenario={"instructions": "Fixture"}),
                results_path=tmp_path,
                language="pt",
                audio_native_provider=provider,
            ),
            "speech_fidelity",
        )
        return json.loads(result.units[0].evidence)

    trimmed, full = first_clip(PIPECAT), first_clip(None)
    assert trimmed["clip_duration_seconds"] == pytest.approx(
        (LAST_WORD_TICK + 1) * TICK
    )
    assert full["clip_duration_seconds"] == pytest.approx((CUT_TICK + 1) * TICK)
    assert trimmed["was_interrupted"] is True and full["was_interrupted"] is True


def _metadata(info_provider, conditions_provider):
    from tau2.data_model.benchmark import VoiceBenchmarkConditions

    def config(provider):
        return None if provider is None else SimpleNamespace(provider=provider)

    conditions = (
        VoiceBenchmarkConditions.model_construct(
            audio_native_config=config(conditions_provider)
        )
        if conditions_provider
        else None
    )
    return SimpleNamespace(
        info=SimpleNamespace(
            audio_native_config=config(info_provider),
            benchmark=None
            if conditions is None
            else SimpleNamespace(conditions=conditions),
        )
    )


def test_recorded_provider_reads_either_record_and_rejects_a_conflict():
    assert recorded_provider(_metadata(PIPECAT, None)) == PIPECAT
    assert recorded_provider(_metadata(None, PIPECAT)) == PIPECAT
    assert recorded_provider(_metadata(None, None)) is None
    with pytest.raises(ValueError, match="conflicts"):
        recorded_provider(_metadata(PIPECAT, "gemini"))


def test_trimmed_providers_are_pinned_to_the_fidelity_implementation():
    """Adding a provider changes saved runs' fidelity input: bump FIDELITY too."""
    assert PLAYED_WORD_END_TEXT_PROVIDERS == {"avartha_pipecat"}
    assert ImplementationFamily.FIDELITY == "speech-fidelity-v6-cut-clip-bounds"

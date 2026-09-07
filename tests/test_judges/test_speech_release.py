"""Offline contracts for the tau-Elicitation speech-judge runner."""

import base64
import json
import wave
from io import BytesIO
from pathlib import Path

import pytest

from tau2.data_model.message import UserMessage
from tau2.judges.speech import (
    SpeechJudgeConfig,
    _agent_channel,
    _slice_wav_b64,
    build_user_prompt,
    load_prompt_spec,
    rejudge_results,
)
from tau2.utils.llm_utils import _format_messages_for_logging, to_litellm_messages


def _stereo_wav(path: Path) -> None:
    frames = b"".join(
        left.to_bytes(2, "little", signed=True)
        + right.to_bytes(2, "little", signed=True)
        for left, right in ((1, 11), (2, 12), (3, 13), (4, 14))
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(4)
        output.writeframes(frames)


def test_prompt_asset_parses_and_renders_exact_v6_shape() -> None:
    spec = load_prompt_spec()
    assert spec.system.startswith("You are a careful evaluator")
    assert '"fidelity" | "intonation"' in spec.output_shape
    rendered = build_user_prompt("Expected words", was_interrupted=True)
    assert "Reference transcript (expected_synthesis_text):\nExpected words" in rendered
    assert "was_interrupted: true" in rendered
    assert "Target locale/language:\nen" in rendered
    assert "{rubric_section}" not in rendered


def test_multimodal_message_is_forwarded_and_audio_is_redacted() -> None:
    message = UserMessage(role="user", content="Listen", audio_content="UklGRg==")
    converted = to_litellm_messages([message])
    assert converted[0]["content"][1] == {
        "type": "input_audio",
        "input_audio": {"data": "UklGRg==", "format": "wav"},
    }
    logged = _format_messages_for_logging(converted)
    assert logged[0]["content"][1]["input_audio"]["data"] == "<audio: 8 b64 chars>"


def test_stereo_right_channel_and_tick_slice(tmp_path: Path) -> None:
    source = tmp_path / "both.wav"
    _stereo_wav(source)
    pcm, rate, width = _agent_channel(source)
    assert rate == 4
    assert width == 2
    assert [
        int.from_bytes(pcm[i : i + 2], "little", signed=True)
        for i in range(0, len(pcm), 2)
    ] == [11, 12, 13, 14]
    encoded = _slice_wav_b64(pcm, rate, width, 1, 2, 0.25)
    with wave.open(BytesIO(base64.b64decode(encoded)), "rb") as sliced:
        values = sliced.readframes(sliced.getnframes())
    assert [
        int.from_bytes(values[i : i + 2], "little", signed=True)
        for i in range(0, len(values), 2)
    ] == [12, 13]


def test_frozen_paper_root_requires_separate_output(tmp_path: Path) -> None:
    source = tmp_path / "data" / "simulations" / "paper_runs" / "tau-elicit"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="refusing to modify a frozen"):
        rejudge_results(
            source,
            output=None,
            replace_existing=True,
            limit_sims=1,
            config=SpeechJudgeConfig(),
        )


def test_output_preserves_unknown_simulation_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "cell"
    simulations = source / "simulations"
    simulations.mkdir(parents=True)
    sim_path = simulations / "sim.json"
    raw = {"id": "s", "future_field": {"keep": True}}
    sim_path.write_text(json.dumps(raw))

    class StubSimulation:
        pass

    monkeypatch.setattr(
        "tau2.judges.speech.SimulationRun.model_validate", lambda _raw: StubSimulation()
    )
    monkeypatch.setattr(
        "tau2.judges.speech.judge_simulation",
        lambda *_args: type(
            "Info",
            (),
            {"model_dump": lambda self, mode: {"judge_prompt_version": "v6"}},
        )(),
    )
    output = tmp_path / "out"
    report = rejudge_results(
        source,
        output=output,
        replace_existing=True,
        limit_sims=None,
        config=SpeechJudgeConfig(),
    )
    saved = json.loads((output / "simulations" / "sim.json").read_text())
    assert report.judged == 1
    assert saved["future_field"] == {"keep": True}
    assert saved["delivery_info"]["judge_prompt_version"] == "v6"

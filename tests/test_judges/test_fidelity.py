"""Speech-fidelity audio slicing and clip-truncation decisions."""

import json
import struct
import wave
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.judges.catalog import load_catalog
from tau2.judges.fidelity import FidelityJudge, FidelityReply, Finding
from tau2.judges.run_evidence import RunEvidence, inspect_audio


def audio_fixture(tmp_path):
    sim = SimulationRun(
        id="s",
        task_id="t",
        trial=0,
        start_time="a",
        end_time="b",
        duration=4,
        termination_reason=TerminationReason.MAX_STEPS,
        ticks=[
            Tick(
                tick_id=i,
                timestamp="tick",
                tick_duration_seconds=1,
                agent_chunk=AssistantMessage(
                    role="assistant",
                    content="hello" if i == 0 else "",
                    contains_speech=True,
                    utterance_ids=["u"],
                ),
            )
            for i in range(4)
        ],
    )
    path = tmp_path / "tasks/task_t/sim_s/audio/both.wav"
    path.parent.mkdir(parents=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(struct.pack("<hh", 100, 200) * 32000)
    return sim, path


def finding(category="mispronunciation", time_range="1.0-1.5", **updates):
    return {
        "category": category,
        "time_range": time_range,
        "issue": "audible discrepancy",
        "severity": 2,
        "confidence": 0.9,
        "is_truncation": False,
        **updates,
    }


def test_finding_category_is_required_and_closed():
    missing = '{"findings":[{"issue":"tail"}]}'
    unknown = '{"findings":[{"category":"unknown","issue":"tail"}]}'
    for response in (missing, unknown):
        with pytest.raises(ValidationError):
            FidelityReply.model_validate_json(response)
    for updates in ({"is_truncation": "false"}, {"severity": 4}, {"confidence": 1.1}):
        with pytest.raises(ValidationError):
            FidelityReply.model_validate({"findings": [finding(**updates)]})
    with pytest.raises(ValidationError):
        FidelityReply.model_validate(
            {"findings": [finding("clip_truncation", is_truncation=False)]}
        )


def test_reply_schema_is_the_parsed_reply():
    """The provider-enforced schema admits only replies the judge can parse."""
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    schema = contract.model_args["response_format"]["json_schema"]["schema"]
    assert schema["required"] == list(FidelityReply.model_fields)
    item = schema["properties"]["findings"]["items"]
    assert set(item["required"]) == set(item["properties"]) == set(Finding.model_fields)
    assert item["additionalProperties"] is False
    categories = Finding.model_json_schema()["properties"]["category"]["enum"]
    assert item["properties"]["category"]["enum"] == categories
    severity = item["properties"]["severity"]
    assert (severity["minimum"], severity["maximum"]) == (1, 3)


def test_a_severity_zero_finding_is_the_rubrics_no_issue_answer():
    """The rubric defines severity 0 as no clear issue, to be returned as an
    empty list. A reply that still lists one failed validation, which left the
    clip, and so its whole call, unjudged."""

    clean = FidelityReply.model_validate({"findings": [finding(severity=0)]})
    mixed = FidelityReply.model_validate(
        {"findings": [finding(severity=0), finding(severity=2)]}
    )

    assert clean.findings == []
    assert [f.severity for f in mixed.findings] == [2]
    for severity in (-1, "0", None):
        with pytest.raises(ValidationError):
            FidelityReply.model_validate({"findings": [finding(severity=severity)]})


@pytest.mark.parametrize("marker", ["###STOP###", "###TRANSFER###"])
def test_control_marker_excludes_whole_audio_utterance(tmp_path, monkeypatch, marker):
    sim, _ = audio_fixture(tmp_path)
    ticks = []
    for i, tick in enumerate(sim.ticks):
        text = ("hello", f"unfinished {marker}", "goodbye", "")[i]
        ticks.append(
            tick.model_copy(
                update={
                    "agent_chunk": tick.agent_chunk.model_copy(
                        update={"content": text, "utterance_ids": [f"u{i}"]}
                    )
                }
            )
        )
    sim = sim.model_copy(update={"ticks": ticks})
    seen = []

    def generate(**kwargs):
        seen.append(kwargs["messages"][1].content)
        return SimpleNamespace(content='{"findings": []}')

    monkeypatch.setattr("tau2.judges.fidelity.generate", generate)
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    result = FidelityJudge(contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="t", user_scenario={"instructions": "Fixture"}),
            results_path=tmp_path,
            language="en",
        ),
        "speech_fidelity",
    )
    assert [unit.unit_id for unit in result.units] == ["0", "2"]
    assert len(seen) == 2
    assert all("unfinished" not in prompt for prompt in seen)


def test_terminal_speech_clip_is_excluded(tmp_path, monkeypatch):
    sim, _ = audio_fixture(tmp_path)
    ticks = [
        tick.model_copy(
            update={
                "agent_chunk": tick.agent_chunk.model_copy(
                    update={
                        "content": ("hello", "goodbye")[i],
                        "utterance_ids": [f"u{i}"],
                    }
                )
            }
        )
        for i, tick in enumerate(sim.ticks[:2])
    ]
    sim = sim.model_copy(
        update={"ticks": ticks, "termination_reason": TerminationReason.USER_STOP}
    )
    seen = []

    def generate(**kwargs):
        seen.append(kwargs["messages"][1].content)
        return SimpleNamespace(content='{"findings": []}')

    monkeypatch.setattr("tau2.judges.fidelity.generate", generate)
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    result = FidelityJudge(contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="t", user_scenario={"instructions": "Fixture"}),
            results_path=tmp_path,
            language="en",
        ),
        "speech_fidelity",
    )
    assert [unit.unit_id for unit in result.units] == ["0"]
    assert len(seen) == 1 and "goodbye" not in seen[0]


def test_audio_digest_exact_model_and_filter(tmp_path, monkeypatch):
    sim, path = audio_fixture(tmp_path)
    seen = []

    def generate(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "findings": [
                        finding("clip_truncation", "3.0-3.5", is_truncation=True),
                        finding("word_substitution", "3.0-3.5"),
                    ]
                }
            )
        )

    monkeypatch.setattr("tau2.judges.fidelity.generate", generate)
    evidence = inspect_audio(tmp_path, sim)
    assert evidence.sha256 and evidence.error is None
    c = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    result = FidelityJudge(c).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id=sim.task_id, user_scenario={"instructions": "Fixture"}),
            results_path=tmp_path,
            language="en",
        ),
        "speech_fidelity",
    )
    assert len(result.units) == 1 and result.units[0].violation
    saved_evidence = json.loads(result.units[0].evidence)
    assert saved_evidence["filter_version"] == "explicit-truncation-v1"
    findings = saved_evidence["findings"]
    assert [f["excluded_as_truncation"] for f in findings] == [True, False]
    assert all("excluded_by_final_second_filter" not in f for f in findings)
    assert findings[1]["time_range"] == "3.0-3.5"
    assert seen[0]["call_name"] == "speech_fidelity_judge"
    assert seen[0]["model"] == "gemini/gemini-3.1-pro-preview"
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["messages"][1].audio_content
    system_prompt = seen[0]["messages"][0].content
    assert "Prefer precision over recall for speech fidelity." in system_prompt
    assert "both axes" not in system_prompt
    prompt = seen[0]["messages"][1].content
    assert "native listener of English" in prompt
    assert "factor_checks" not in prompt
    assert "time_range" in prompt
    assert "clip_truncation" in prompt
    assert "was_interrupted: false" in prompt
    before = evidence.sha256
    path.write_bytes(path.read_bytes() + b"changed")
    assert inspect_audio(tmp_path, sim).sha256 != before


def test_missing_and_malformed_audio_are_distinct(tmp_path):
    sim, path = audio_fixture(tmp_path)
    path.write_bytes(b"bad wav")
    assert inspect_audio(tmp_path, sim).error
    path.unlink()
    evidence = inspect_audio(tmp_path, sim)
    assert evidence.path is None and evidence.error is None


def test_truncation_only_is_unscored_and_interruption_comes_from_recording(
    tmp_path, monkeypatch
):
    sim, _ = audio_fixture(tmp_path)
    for tick in sim.ticks[1:]:
        tick.agent_chunk = None
    sim.ticks[0].agent_chunk.raw_data = {"was_truncated": True}
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {"findings": [finding("missing_word", "0.5-1.0", is_truncation=True)]}
            )
        )

    monkeypatch.setattr("tau2.judges.fidelity.generate", generate)
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    result = FidelityJudge(contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="t", user_scenario={"instructions": "Fixture"}),
            results_path=tmp_path,
            language="en",
        ),
        "speech_fidelity",
    )
    assert len(result.units) == 1 and not result.units[0].violation
    assert not result.unit_errors
    assert "was_interrupted: true" in calls[0]["messages"][1].content
    saved_evidence = json.loads(result.units[0].evidence)
    assert saved_evidence["was_interrupted"] is True
    assert saved_evidence["findings"][0]["excluded_as_truncation"] is True


def test_missing_finding_category_is_a_unit_error(tmp_path, monkeypatch):
    sim, _ = audio_fixture(tmp_path)
    monkeypatch.setattr(
        "tau2.judges.fidelity.generate",
        lambda **kw: SimpleNamespace(content='{"findings":[{"issue":"tail"}]}'),
    )
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    result = FidelityJudge(contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="t", user_scenario={"instructions": "Fixture"}),
            results_path=tmp_path,
            language="en",
        ),
        "speech_fidelity",
    )
    assert not result.units and len(result.unit_errors) == 1


def test_mandarin_tone_flip_stays_in_fidelity_prompt(tmp_path, monkeypatch):
    sim, _ = audio_fixture(tmp_path)
    seen = []

    def generate(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(content='{"findings": []}')

    monkeypatch.setattr("tau2.judges.fidelity.generate", generate)
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    FidelityJudge(contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="t", user_scenario={"instructions": "Fixture"}),
            results_path=tmp_path,
            language="zh",
        ),
        "speech_fidelity",
    )
    prompt = seen[0]["messages"][1].content
    assert "买 mǎi (buy) spoken as 卖 mài (sell)" in prompt
    assert "factor_checks" not in prompt


def test_fidelity_audio_reaches_completion_boundary(tmp_path, monkeypatch):
    from litellm import ModelResponse

    import tau2.utils.llm_utils as llm_utils

    sim, _ = audio_fixture(tmp_path)
    seen = []

    def completion(**kwargs):
        seen.append(kwargs)
        return ModelResponse(
            model="fixture",
            choices=[
                {
                    "message": {"role": "assistant", "content": '{"findings": []}'},
                    "finish_reason": "stop",
                }
            ],
        )

    monkeypatch.setattr(llm_utils, "completion", completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda _: 0)
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    FidelityJudge(contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id=sim.task_id, user_scenario={"instructions": "Fixture"}),
            results_path=tmp_path,
            language="en",
        ),
        "speech_fidelity",
    )
    content = seen[0]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["input_audio"]["format"] == "wav"
    assert content[1]["input_audio"]["data"].startswith("UklG")


def test_fidelity_retains_successes_and_retries_only_failed_clip(tmp_path, monkeypatch):
    sim, _ = audio_fixture(tmp_path)
    for tick in sim.ticks[2:]:
        tick.agent_chunk.utterance_ids = ["v"]
        tick.agent_chunk.content = "goodbye" if tick.tick_id == 2 else ""
    replies = iter(
        [
            SimpleNamespace(content='{"findings": []}'),
            SimpleNamespace(
                content='{"findings": [',
                raw_data={
                    "choices": [{"finish_reason": "length"}],
                    "usage": {"completion_tokens": 8192},
                },
            ),
        ]
    )
    monkeypatch.setattr("tau2.judges.fidelity.generate", lambda **kw: next(replies))
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    evidence = RunEvidence(
        simulation=sim,
        task=Task(id="t", user_scenario={"instructions": "Fixture"}),
        results_path=tmp_path,
        language="en",
    )
    judge = FidelityJudge(contract)
    result = judge.evaluate(evidence, "speech_fidelity")
    assert len(result.units) == len(result.unit_errors) == 1
    error = result.unit_errors[0]
    assert error.finish_reason == "length"
    assert error.response_text == '{"findings": ['
    assert error.usage == {"completion_tokens": 8192}
    calls = []

    def retry(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content='{"findings": []}')

    monkeypatch.setattr("tau2.judges.fidelity.generate", retry)
    resumed = judge.resume(evidence, "speech_fidelity", result)
    assert len(calls) == 1 and not resumed.unit_errors and len(resumed.units) == 2
    assert resumed.units[0] == result.units[0]


def test_valid_json_at_output_limit_is_not_a_clean_verdict(tmp_path, monkeypatch):
    sim, _ = audio_fixture(tmp_path)
    monkeypatch.setattr(
        "tau2.judges.fidelity.generate",
        lambda **kw: SimpleNamespace(
            content='{"findings": []}',
            raw_data={"choices": [{"finish_reason": "length"}]},
        ),
    )
    contract = next(
        c for c in load_catalog().contracts if c.contract_id == "speech-fidelity-v7"
    )
    result = FidelityJudge(contract).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="t", user_scenario={"instructions": "Fixture"}),
            results_path=tmp_path,
            language="en",
        ),
        "speech_fidelity",
    )
    assert not result.units and len(result.unit_errors) == 1

"""Benchmark-authored opener and transfer line never reach model-language judges."""

import json
import struct
import wave
from types import SimpleNamespace

import pytest

from tau2.data_model.judges import InputExclusions, Measure
from tau2.data_model.message import AssistantMessage, Tick, UserMessage
from tau2.data_model.simulation import (
    Results,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.judges.catalog import load_catalog
from tau2.judges.criteria import CriteriaJudge
from tau2.judges.exclusions import (
    BENCHMARK_TRANSFER_MESSAGE,
    injected_opener,
    judged_agent_turns,
    without_benchmark_transfer_message,
)
from tau2.judges.fidelity import FidelityJudge
from tau2.judges.run_evidence import RunEvidence, SavedRun
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.runner.helpers import get_info

OPENER = "Hola, ¿en qué puedo ayudarle?"
BOTH = InputExclusions.OPENER_AND_TRANSFER
NATURALNESS = InputExclusions.OPENER_TRANSFER_AND_CONTROL_TURNS


def text_sim(*messages):
    return SimulationRun(
        id="s",
        task_id="t",
        trial=0,
        start_time="a",
        end_time="b",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=list(messages),
    )


def agent(text, **kwargs):
    return AssistantMessage(role="assistant", content=text, **kwargs)


def user(text):
    return UserMessage(role="user", content=text)


def contract(identifier):
    return next(c for c in load_catalog().contracts if c.contract_id == identifier)


@pytest.mark.parametrize(
    ("text", "interrupted", "expected"),
    [
        (BENCHMARK_TRANSFER_MESSAGE, False, ""),
        (
            f"Le transfiero ahora. {BENCHMARK_TRANSFER_MESSAGE}",
            False,
            "Le transfiero ahora.",
        ),
        (
            "Le transfiero. you are being transferred to a human agent. please hold on. ###TRANSFER###",
            False,
            "Le transfiero.",
        ),
        ("Le transfiero. YOU ARE BEING TRANSF", True, "Le transfiero."),
        # A truncated prefix is kept when nothing cut the turn off.
        (
            "Le transfiero. YOU ARE BEING TRANSF",
            False,
            "Le transfiero. YOU ARE BEING TRANSF",
        ),
        # Too short to be distinguishable from ordinary English.
        ("Le transfiero. YOU ARE BEING", True, "Le transfiero. YOU ARE BEING"),
        ("Gracias. ###STOP###", False, "Gracias."),
        # Non-Latin speech after a cut-off prefix is the agent's own and stays.
        (
            "Le transfiero. YOU ARE BEING TRANSFERRED 请稍等，马上回来",
            True,
            "Le transfiero. YOU ARE BEING TRANSFERRED 请稍等，马上回来",
        ),
        # Turns without the line are returned byte-identical.
        ("Hola,  ¿en qué  puedo ayudar? ", False, "Hola,  ¿en qué  puedo ayudar? "),
    ],
)
def test_transfer_line_removal(text, interrupted, expected):
    assert without_benchmark_transfer_message(text, interrupted=interrupted) == expected


def test_text_opener_is_identified_structurally():
    sim = text_sim(agent(OPENER), user("Ayuda"), agent(OPENER), agent("Claro."))
    assert injected_opener(sim, OPENER)
    turns = judged_agent_turns(sim, OPENER, BOTH)
    # The first message is the injected opener; the model repeating it is judged.
    assert [(t.index, t.text) for t in turns] == [(1, OPENER), (2, "Claro.")]
    assert len(judged_agent_turns(sim, OPENER, InputExclusions.NONE)) == 3
    # A first message that is not the recorded opener is model speech.
    assert not injected_opener(sim, "Otro saludo")
    assert len(judged_agent_turns(sim, "Otro saludo", BOTH)) == 3
    assert len(judged_agent_turns(sim, None, BOTH)) == 3
    # Calls starting with the caller have no injected opener.
    assert not injected_opener(text_sim(user("Hola"), agent(OPENER)), OPENER)


def test_transfer_only_turn_is_dropped_but_opener_rule_keeps_it():
    sim = text_sim(agent(OPENER), user("Humano"), agent(BENCHMARK_TRANSFER_MESSAGE))
    assert judged_agent_turns(sim, OPENER, BOTH) == []
    assert [
        t.text for t in judged_agent_turns(sim, OPENER, InputExclusions.OPENER)
    ] == [BENCHMARK_TRANSFER_MESSAGE]


@pytest.mark.parametrize("marker", ["###STOP###", "###TRANSFER###"])
def test_naturalness_skips_entire_control_turn(marker):
    sim = text_sim(
        agent(OPENER),
        user(f"Caller says {marker}"),
        agent(f"Incomplete phrase {marker}"),
        agent("Complete phrase."),
    )
    assert [
        (t.index, t.text) for t in judged_agent_turns(sim, OPENER, NATURALNESS)
    ] == [(2, "Complete phrase.")]


def duplex_sim(opener_speech: bool):
    ticks = [
        Tick(
            tick_id=0,
            timestamp="tick",
            tick_duration_seconds=1,
            agent_chunk=agent(OPENER, contains_speech=opener_speech, cost=0.0),
        )
    ]
    ticks += [
        Tick(
            tick_id=i,
            timestamp="tick",
            tick_duration_seconds=1,
            agent_chunk=agent(
                OPENER if i == 1 else "", contains_speech=True, utterance_ids=["u"]
            ),
        )
        for i in (1, 2, 3)
    ]
    return SimulationRun(
        id="s",
        task_id="t",
        trial=0,
        start_time="a",
        end_time="b",
        duration=4,
        termination_reason=TerminationReason.MAX_STEPS,
        ticks=ticks,
    )


def test_full_duplex_opener_spoken_or_text_only():
    spoken = duplex_sim(opener_speech=True)
    assert [t.text for t in judged_agent_turns(spoken, OPENER, BOTH)] == [OPENER]
    assert len(judged_agent_turns(spoken, OPENER, InputExclusions.NONE)) == 2
    # Audio-native openers are text over silence and never become a turn, so a
    # model utterance that repeats the opener verbatim is still judged.
    silent = duplex_sim(opener_speech=False)
    assert [t.text for t in judged_agent_turns(silent, OPENER, BOTH)] == [OPENER]


def test_utterance_overlapping_call_end_is_excluded():
    sim = duplex_sim(opener_speech=False)
    sim = sim.model_copy(update={"termination_reason": TerminationReason.USER_STOP})
    assert judged_agent_turns(sim, OPENER, NATURALNESS) == []
    assert [t.text for t in judged_agent_turns(sim, OPENER, BOTH)] == [OPENER]


def test_criteria_prompts_exclude_opener_and_transfer_line(monkeypatch):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
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

    monkeypatch.setattr("tau2.judges.criteria.generate", generate)
    sim = text_sim(
        agent(OPENER),
        user("Quiero un humano"),
        agent(f"De acuerdo. {BENCHMARK_TRANSFER_MESSAGE}"),
    )
    judge = contract("naturalness-v17-rubric-v21-language-specific-draft2")
    assert judge.input_exclusions == NATURALNESS
    result = CriteriaJudge(judge).evaluate(
        RunEvidence(
            simulation=sim,
            task=Task(id="t", user_scenario={"instructions": "Help."}),
            language="es",
            opener=OPENER,
        ),
        Measure.NATURALNESS,
    )
    assert [unit.unit_id for unit in result.units] == ["1"]
    prompt = calls[0]["messages"][1].content
    assert "De acuerdo." in prompt
    assert OPENER not in prompt and "TRANSFERRED" not in prompt


def test_fidelity_skips_injected_opener_clip(tmp_path, monkeypatch):
    sim = duplex_sim(opener_speech=False)
    path = tmp_path / "tasks/task_t/sim_s/audio/both.wav"
    path.parent.mkdir(parents=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(struct.pack("<hh", 100, 200) * 32000)
    seen = []

    def generate(**kwargs):
        seen.append(kwargs["messages"][1].content)
        return SimpleNamespace(content='{"findings": []}')

    monkeypatch.setattr("tau2.judges.fidelity.generate", generate)
    judge = FidelityJudge(contract("speech-fidelity-v7"))
    evidence = RunEvidence(
        simulation=sim,
        task=Task(id="t", user_scenario={"instructions": "Fixture"}),
        results_path=tmp_path,
        language="es",
        opener=OPENER,
    )
    result = judge.evaluate(evidence, "speech_fidelity")
    assert [unit.unit_id for unit in result.units] == ["1"]
    assert len(seen) == 1
    # Without a recorded opener, the tick-0 clip is judged like any other.
    seen.clear()
    judge.evaluate(evidence.model_copy(update={"opener": None}), "speech_fidelity")
    assert len(seen) == 2


def test_unlabeled_saved_run_is_english_with_default_opener(tmp_path):
    # Results saved before every run recorded its language carry none.
    info = get_info(TextRunConfig(domain="retail", num_trials=1)).model_copy(
        update={"language": None, "language_tasks": {}}
    )
    assert info.benchmark is None
    task = Task(id="0", user_scenario={"instructions": "Fixture caller."})
    Results(
        info=info,
        tasks=[task],
        simulations=[
            text_sim(agent(DEFAULT_FIRST_AGENT_MESSAGE.content), user("Hi")).model_copy(
                update={"task_id": "0"}
            )
        ],
    ).save(tmp_path / "run.json", format="json")
    saved = SavedRun(tmp_path / "run.json")
    assert saved.language == "en"
    evidence = saved.load(saved.inputs[0])
    assert evidence.language == "en"
    assert evidence.opener == DEFAULT_FIRST_AGENT_MESSAGE.content
    assert judged_agent_turns(evidence.simulation, evidence.opener, BOTH) == []


def test_recorded_opener_prefers_task_opener(tmp_path):
    info = get_info(TextRunConfig(domain="retail", num_trials=1))
    task = Task(
        id="0", user_scenario={"instructions": "Fixture caller."}, agent_opener=OPENER
    )
    Results(
        info=info,
        tasks=[task],
        simulations=[text_sim(agent(OPENER)).model_copy(update={"task_id": "0"})],
    ).save(tmp_path / "run.json", format="json")
    saved = SavedRun(tmp_path / "run.json")
    assert saved.load(saved.inputs[0]).opener == OPENER


def test_contracts_record_their_exclusion_rule():
    rules = {c.contract_id: c.input_exclusions for c in load_catalog().contracts}
    assert rules == {
        "nativeness-v15-rubric-v19": BOTH,
        "naturalness-v17-rubric-v21-language-specific-draft2": NATURALNESS,
        "naturalness-v17-rubric-v21-ko-indirect-honorific-v1": NATURALNESS,
        "hi-gender-v16-rubric-v26": BOTH,
        "quality-v7-rubric-v4": InputExclusions.NONE,
        "speech-fidelity-v7": InputExclusions.OPENER_AND_CONTROL_TURNS,
    }


def test_provenance_rejects_source_commits():
    from pydantic import ValidationError

    from tau2.data_model.judges import JudgeContract

    payload = contract("quality-v7-rubric-v4").model_dump()
    payload["source_revision"] = "6813fceab54fe6c7c44733be9935e6d38aaca848"
    with pytest.raises(ValidationError, match="neutral package label"):
        JudgeContract.model_validate(payload)

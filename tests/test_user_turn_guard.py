from unittest.mock import patch

import pytest

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.evaluator.evaluator import EvaluationType
from tau2.metrics.agent_metrics import compute_metrics, display_metrics
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.simulation import run_simulation
from tau2.user.turn_guard import (
    check_user_message,
    detect_multi_turn,
    truncate_to_first_turn,
)
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import STOP
from tau2.utils.display import ConsoleDisplay

# First user message of retail task 9 from issue #710 (gpt-oss-120b).
ISSUE_710 = (
    "Yeah, hi. I need to exchange two things. The water bottle\u2014want a bigger size. "
    "And the desk lamp\u2014need a brighter one, preferably with an AC adapter.My order "
    "number is 8020\u2011MEI\u20112023.Can you confirm the exchange?Actually, never mind the "
    "bottle. Just the desk lamp\u2014brighter, AC adapter if possible. No water bottle "
    "exchange.###STOP###"
)


def test_issue_example_detected():
    assert "glued_sentences" in detect_multi_turn(ISSUE_710)


def test_issue_example_truncated_and_stop_dropped():
    out = truncate_to_first_turn(ISSUE_710)
    assert out.endswith("AC adapter.")
    assert "###STOP###" not in out
    assert not UserSimulator.is_stop(_um(out))


@pytest.mark.parametrize(
    "text",
    [ISSUE_710, "I need a refund.Can you check it?Thanks, bye.###STOP###"],
)
def test_stop_at_end_is_caught_only_by_glued_sentences(text):
    # is_stop matches ###STOP### anywhere in the message, so a merged message
    # that ends in STOP ends the call. Nothing follows the token, so the
    # text-after-STOP check can't see it; only the glue check does.
    assert text.endswith(STOP)
    assert UserSimulator.is_stop(_um(text))
    assert detect_multi_turn(text) == ["glued_sentences"]
    assert not UserSimulator.is_stop(_um(truncate_to_first_turn(text)))


@pytest.mark.parametrize(
    "text, first_turn",
    [
        ("I have my receipt handy.Okay, what next?", "I have my receipt handy."),
        ("Can you help me.My order is #W2378156.", "Can you help me."),
        ("It's order #W2378156.Thanks!", "It's order #W2378156."),
        ("I can't log in.I tried twice.", "I can't log in."),
        ("Is that right?Actually, wait.", "Is that right?"),
        ("It broke (again).Can you help?", "It broke (again)."),
        # A title isn't a sentence end, so the cut is at the real one.
        ("I'm Mrs.Davis.Can you help?", "I'm Mrs.Davis."),
        # Spanish and Portuguese use spaces, so they are checked too.
        ("Gracias, señora.Puede ver mi pedido?", "Gracias, señora."),
    ],
)
def test_latin_glue_flagged_and_cut(text, first_turn):
    assert detect_multi_turn(text) == ["glued_sentences"]
    assert truncate_to_first_turn(text) == first_turn


@pytest.mark.parametrize(
    "text",
    [
        "Some items, e.g.the lamp, are fine.",
        "Some items, e.g.The lamp, are fine.",
        "I'm on app version 2.0.1, the 2.0.Beta build crashed.",
        "Build 1.2.RC3 and firmware v1.2.3-beta are installed.",
        "Python 3.12.Final and build 10.0.19045.Update both fail.",
        "My doctor, Dr.Lee, has a Ph.D in chemistry.",
        "Ask Mr.Smith at the St.Louis store.",
        "My name is Mrs.Davis, and Prof.Kim or mr.Jones referred me.",
        "La Sra.García y la Srta.Ruiz me atendieron con la Dra.López.",
        "A Profa.Silva e o sr.Costa me indicaram.",
        "I compared Samsung vs.Apple phones.",
        "I attached photo.JPG and the Node.JS logs.",
        "My email is Mei.Davis@example.com.",
        "See https://shop.example.com/Help.Center for details.",
        "I bought it on Amazon.Com last week.",
    ],
)
def test_glue_false_positives_not_flagged_or_cut(text):
    assert detect_multi_turn(text) == []
    assert truncate_to_first_turn(text) == text


@pytest.mark.parametrize(
    "text",
    [
        # Chinese, Japanese and Thai don't put a space after a full stop.
        "您好，我想退一下货。订单号是#W2378156。麻烦您帮我看看，可以吗？",
        "我买的是AirPods Pro.Apple官网说可以换货.您能帮我处理一下吗?",
        "我买的是 AirPods Pro.Apple 官网说可以换货。",
        "こんにちは。注文をキャンセルしたいです。注文番号は#W2378156です。",
        "MacBook Air.Appleストアで買いました.返品できますか?",
        "สวัสดีครับ ผมอยากยกเลิกคำสั่งซื้อ #W2378156 ครับ ขอบคุณครับ",
        "ผมสั่งซื้อ iPad.Apple บอกว่าเปลี่ยนได้ครับ",
    ],
)
def test_unspaced_scripts_not_flagged(text):
    assert detect_multi_turn(text) == []
    assert check_user_message(text, "truncate").repaired is None


def test_text_after_stop():
    assert "text_after_control_token" in detect_multi_turn(
        "Thanks! ###STOP### Agent: bye"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Thanks, bye. ###STOP###.",
        "Thanks! **###STOP###**",
        "Bye. ###STOP### ###STOP###",
    ],
)
def test_punctuation_after_stop_is_not_another_turn(text):
    # As in merged_turn_cue, only a letter or digit after the token counts.
    assert detect_multi_turn(text) == []
    assert check_user_message(text, "truncate").repaired is None


def test_agent_impersonation():
    text = "I want to return my order.\nAgent: Sure, what's the order id?\nIt's #W123."
    assert "agent_impersonation" in detect_multi_turn(text)
    assert truncate_to_first_turn(text) == "I want to return my order."


def test_clean_messages_not_flagged():
    for text in [
        "Hi, I'd like to cancel order #W2378156. Thanks!",
        "That's all, thank you. ###STOP###",
        "The price was $4.1 and I live in the U.S. now.",
    ]:
        assert detect_multi_turn(text) == [], text


def test_legit_stop_kept_on_truncate():
    assert (
        truncate_to_first_turn("Great, thanks. ###STOP###")
        == "Great, thanks. ###STOP###"
    )


def test_modes():
    assert check_user_message(ISSUE_710, "off").issues == []
    d = check_user_message(ISSUE_710, "detect")
    assert d.is_multi_turn and d.repaired is None
    assert check_user_message(ISSUE_710, "truncate").repaired is not None


def _um(text):
    from tau2.data_model.message import UserMessage

    return UserMessage(role="user", content=text)


def _am(text):
    return AssistantMessage(role="assistant", content=text)


def _run(mode):
    user = UserSimulator(llm="mock", instructions="x", turn_guard=mode)
    state = user.get_init_state()
    fake = AssistantMessage(role="assistant", content=ISSUE_710, raw_data={})
    with patch("tau2.user.user_simulator.generate", return_value=fake):
        msg, _ = user.generate_next_message(
            AssistantMessage(role="assistant", content="Hi, how can I help?"), state
        )
    return msg


def test_repro_default_leak_still_ends_call_in_detect_mode():
    # Reproduces #710: the leaked ###STOP### ends the call before the agent acts.
    msg = _run("detect")
    assert UserSimulator.is_stop(msg)
    assert msg.raw_data["turn_guard"]["issues"]


def test_truncate_mode_keeps_call_alive():
    msg = _run("truncate")
    assert not UserSimulator.is_stop(msg)
    assert msg.raw_data["turn_guard"]["original"] == ISSUE_710


def test_truncate_rewrites_simulator_history():
    # The simulator must not "remember" the cut text: its state and the next
    # LLM request carry only the first turn. The original is kept in raw_data,
    # which is never sent to the model.
    user = UserSimulator(llm="mock", instructions="x", turn_guard="truncate")
    state = user.get_init_state()
    replies = [_am(ISSUE_710), _am("The brighter one, please.")]  # raw_data=None
    with patch("tau2.user.user_simulator.generate", side_effect=replies) as llm:
        first, state = user.generate_next_message(_am("Hi, how can I help?"), state)
        user.generate_next_message(_am("Which lamp?"), state)

    first_turn = truncate_to_first_turn(ISSUE_710)
    assert first.content == first_turn
    assert first.raw_data["turn_guard"]["original"] == ISSUE_710
    # The system prompt explains ###STOP###, so check the conversation turns.
    sent = [
        m.content or "" for m in llm.call_args.kwargs["messages"] if m.role != "system"
    ]
    assert first_turn in sent
    for history in (sent, [m.content or "" for m in state.messages]):
        assert not any("never mind the bottle" in c for c in history)
        assert not any(STOP in c for c in history)


def _orchestrated_run(mode, get_environment, base_task):
    env = get_environment()
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="mock")
    user = UserSimulator(llm="mock", instructions="x", turn_guard=mode)
    user_replies = iter([ISSUE_710, "That's all, thanks. ###STOP###"])
    with (
        patch(
            "tau2.user.user_simulator.generate",
            side_effect=lambda **_: _am(next(user_replies)),
        ),
        patch(
            "tau2.agent.llm_agent.generate",
            side_effect=lambda **_: _am("Sure. Which lamp would you like?"),
        ),
    ):
        orchestrator = Orchestrator(
            domain="mock",
            agent=agent,
            user=user,
            environment=env,
            task=base_task,
            max_steps=10,
        )
        # ACTION scoring needs no LLM judge.
        return run_simulation(orchestrator, evaluation_type=EvaluationType.ACTION)


def test_truncate_records_merge_counts_in_simulation_info(get_environment, base_task):
    sim = _orchestrated_run("truncate", get_environment, base_task)
    assert sim.termination_reason == TerminationReason.USER_STOP
    # The merge counts are stored with the score in the same SimulationRun.
    assert sim.reward_info is not None
    assert sim.info == {
        "turn_guard": {"mode": "truncate", "user_messages": 2, "merged_messages": 1}
    }


def test_detect_leaves_simulation_info_unset(get_environment, base_task):
    sim = _orchestrated_run("detect", get_environment, base_task)
    assert sim.info is None
    # #710: the leaked STOP ends the call on the first user message.
    assert [m.content for m in sim.messages if m.role == "user"] == [ISSUE_710]


def _results(*sim_infos):
    sims = [
        SimulationRun(
            id=f"sim-{i}",
            task_id="t0",
            trial=i,
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=60.0,
            termination_reason=TerminationReason.USER_STOP,
            reward_info=RewardInfo(reward=1.0),
            info=info,
        )
        for i, info in enumerate(sim_infos)
    ]
    info = Info(
        git_commit="abc123",
        num_trials=len(sims),
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
    )
    task = Task(
        id="t0",
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )
    return Results(info=info, tasks=[task], simulations=sims)


def test_metrics_show_merge_rate_next_to_reward(capsys):
    merged = {
        "turn_guard": {"mode": "truncate", "user_messages": 3, "merged_messages": 1}
    }
    clean = {
        "turn_guard": {"mode": "truncate", "user_messages": 2, "merged_messages": 0}
    }
    metrics = compute_metrics(_results(merged, clean))
    assert metrics.avg_reward == 1.0
    assert metrics.sims_with_turn_guard_info == 2
    assert metrics.sims_with_merged_user_turns == 1

    with ConsoleDisplay.console.capture() as table:
        ConsoleDisplay.display_agent_metrics(metrics)
    display_metrics(metrics)
    for out in (table.get().lower(), capsys.readouterr().out.lower()):
        reward, merges = out.index("average reward"), out.index("merged user turns")
        assert reward < merges < out.index("cost")
        assert "1/2" in out[merges:]


def test_metrics_without_truncation_show_no_merge_rate(capsys):
    metrics = compute_metrics(_results(None, None))
    assert metrics.sims_with_turn_guard_info == 0

    with ConsoleDisplay.console.capture() as table:
        ConsoleDisplay.display_agent_metrics(metrics)
    display_metrics(metrics)
    for out in (table.get().lower(), capsys.readouterr().out.lower()):
        assert "merged user turns" not in out

from unittest.mock import patch

from tau2.data_model.message import AssistantMessage
from tau2.user.turn_guard import (
    check_user_message,
    detect_multi_turn,
    truncate_to_first_turn,
)
from tau2.user.user_simulator import UserSimulator

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


def test_text_after_stop():
    assert "text_after_control_token" in detect_multi_turn(
        "Thanks! ###STOP### Agent: bye"
    )


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

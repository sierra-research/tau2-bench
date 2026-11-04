"""The agent-directed elicitation domain.

Covers:

- the agent-directed tool surface: submit_fields omits confirmed_with_user;
  the other agent tools share the scaffolded implementation;
- silent user-side measurement tools (note_spell_request,
  note_readback) and their invisibility to the agent-visible tool surface;
- the voice-only environment (channel="text" fails loud) and policy
  without spelling or read-back mandates;
- the in-memory task rewrite: swapped caller instructions with a drift
  guard, golden submit_fields args stripped, everything else byte-equal to
  the frozen set — which itself stays untouched on disk;
- golden replay through the evaluator: the rewritten golden actions score
  reward 1.0 against the agent-directed environment;
- the versioned agent policy and its verification sentence.
"""

import json

import pytest

from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.intake.agent_directed import (
    AGENT_DIRECTED_POLICY_PATH,
    AGENT_DIRECTED_USER_TASK_INSTRUCTIONS,
    AgentDirectedIntakeUserTools,
    _agent_directed_task_variant,
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.intake.call_frame import USER_TASK_INSTRUCTIONS
from tau2.domains.intake.environment import get_tasks as get_shared_tasks
from tau2.domains.intake.environment import load_frozen_tasks
from tau2.domains.intake.time_display import TIME_CALLER_GUIDANCE
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.domains.intake.utils import (
    INTAKE_SCAFFOLDED_POLICY_PATH,
    INTAKE_SCAFFOLDED_TEXT_POLICY_PATH,
)
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

AGENT_TOOLS = {
    "log_capture",
    "submit_fields",
    "get_today",
    "end_call",
}
NOTE_TOOLS = {"note_spell_request", "note_readback"}


@pytest.fixture(scope="module")
def env():
    return get_environment()


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return get_tasks()


@pytest.fixture(scope="module")
def scaffolded_tasks() -> list[Task]:
    return get_shared_tasks(strategy="scaffolded")


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def test_agent_tool_surface(env):
    assert {t.name for t in env.get_tools()} == AGENT_TOOLS


def test_submit_fields_schema_drops_confirmed_with_user(env):
    tool = next(t for t in env.get_tools() if t.name == "submit_fields")
    params = tool.openai_schema["function"]["parameters"]
    assert set(params["properties"]) == {"record_id", "fields"}
    assert set(params["required"]) == {"record_id", "fields"}
    assert "confirmed_with_user" not in json.dumps(tool.openai_schema)


def test_user_tool_surface_adds_measurement_notes(env):
    assert {t.name for t in env.get_user_tools()} == {"get_entity", *NOTE_TOOLS}


def test_note_tools_never_on_agent_surface(env):
    """User-side measurement tools are invisible to the agent: absent from
    the agent tool list and from the assistant tool descriptions."""
    assert NOTE_TOOLS.isdisjoint({t.name for t in env.get_tools()})
    agent_desc = env.get_tools_description("assistant")
    for name in NOTE_TOOLS:
        assert name not in agent_desc
    user_desc = env.get_tools_description("user")
    for name in NOTE_TOOLS:
        assert name in user_desc


def test_note_tool_calls_route_as_user_requestor_only(env):
    """The orchestrator routes tool calls by requestor; a note call resolves
    through the USER toolkit and the agent toolkit has no such tool."""
    assert env.user_tools.has_tool("note_spell_request")
    assert env.user_tools.has_tool("note_readback")
    assert not env.tools.has_tool("note_spell_request")
    assert not env.tools.has_tool("note_readback")


def test_submit_fields_uses_shared_record_writes(env, tasks):
    """Agent-directed submission uses the shared folding and record writes."""
    task = tasks[0]
    init = task.initial_state
    env2 = get_environment()
    env2.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=[],
    )
    record_id, fields = _golden_submission(task)
    result = env2.use_tool("submit_fields", record_id=record_id, fields=fields)
    assert "Fields submitted" in result
    with pytest.raises(Exception, match="no missing fields"):
        env2.use_tool("submit_fields", record_id=record_id, fields=fields)


# ---------------------------------------------------------------------------
# User-side note tools
# ---------------------------------------------------------------------------


def _user_tools() -> AgentDirectedIntakeUserTools:
    tools = AgentDirectedIntakeUserTools(IntakeUserDB())
    tools.set_entities({"contact_phone": "555-014-2233"})
    return tools


def test_note_spell_request_known_field():
    assert (
        _user_tools().note_spell_request("contact_phone")
        == "Noted spell request for contact_phone."
    )


def test_note_readback_records_affirmation_state():
    tools = _user_tools()
    assert "affirmed" in tools.note_readback("contact_phone", i_affirmed=True)
    assert "corrected" in tools.note_readback("contact_phone", i_affirmed=False)


@pytest.mark.parametrize("tool", ["note_spell_request", "note_readback"])
def test_note_tools_fail_loud_on_unknown_field(tool):
    tools = _user_tools()
    kwargs = {"field": "shoe_size"}
    if tool == "note_readback":
        kwargs["i_affirmed"] = True
    with pytest.raises(ValueError, match="contact_phone"):
        getattr(tools, tool)(**kwargs)


# ---------------------------------------------------------------------------
# Environment: policy + voice-only
# ---------------------------------------------------------------------------


def test_environment_is_voice_only():
    with pytest.raises(ValueError, match="voice-only"):
        get_environment(channel="text")


def test_agent_directed_policy_has_no_protocol_mandates(env):
    from tau2.domains.intake.agent_directed import POLICY_VERIFICATION_SENTENCE

    policy = env.get_policy()
    assert policy == AGENT_DIRECTED_POLICY_PATH.read_text()
    # The agent-directed policy names verification once, with spelling as
    # a worked example, inside its one verification sentence — and nowhere
    # else: outside that sentence the rigid-protocol machinery must be
    # absent (no spelling mandate, no read-backs, no per-entity
    # confirmation, no attestation).
    assert policy.count(POLICY_VERIFICATION_SENTENCE) == 1
    stripped = policy.replace(POLICY_VERIFICATION_SENTENCE, "", 1)
    for banned in (
        "spell",
        "Spell",
        "verify",
        "read back",
        "read-back",
        "read it back",
        "confirmed_with_user",
        "Pinning the written form",
        "explicit yes",
    ):
        assert banned not in stripped, banned
    # Both strategies require logging and a single submission.
    assert "log_capture" in policy
    assert "exactly one `submit_fields` call" in policy
    assert "never include `submit_fields`" in policy
    assert "submit nothing" in policy


def test_scaffolded_policies_require_spelling_and_confirmation():
    """The scaffolded strategy explicitly prescribes the capture protocol."""
    scaffolded = INTAKE_SCAFFOLDED_POLICY_PATH.read_text()
    assert "Pinning the written form" in scaffolded
    assert "letter by letter" in scaffolded
    assert "confirmed" in INTAKE_SCAFFOLDED_TEXT_POLICY_PATH.read_text()


# ---------------------------------------------------------------------------
# Task rewrite
# ---------------------------------------------------------------------------


def test_variant_swaps_instructions_on_every_task(tasks, scaffolded_tasks):
    assert len(tasks) == len(scaffolded_tasks) == 200
    for task in tasks:
        instructions = task.user_scenario.instructions
        expected = AGENT_DIRECTED_USER_TASK_INSTRUCTIONS
        if task.id.startswith("intake_times_"):
            expected += "\n\n# Intake written clarification\n" + TIME_CALLER_GUIDANCE
        assert instructions.task_instructions == expected


def test_variant_preserves_everything_else(tasks, scaffolded_tasks):
    for agent_directed, scaffolded in zip(tasks, scaffolded_tasks):
        assert agent_directed.id == scaffolded.id
        assert agent_directed.agent_opener == scaffolded.agent_opener
        assert agent_directed.user_scenario.persona == scaffolded.user_scenario.persona
        agent_directed_instructions = agent_directed.user_scenario.instructions
        scaffolded_instructions = scaffolded.user_scenario.instructions
        assert (
            agent_directed_instructions.known_info == scaffolded_instructions.known_info
        )
        assert (
            agent_directed_instructions.unknown_info
            == scaffolded_instructions.unknown_info
        )
        assert (
            agent_directed_instructions.reason_for_call
            == scaffolded_instructions.reason_for_call
        )
        assert (
            agent_directed.initial_state.model_dump()
            == scaffolded.initial_state.model_dump()
        )


def test_variant_strips_confirmed_with_user_from_golden_actions(
    tasks, scaffolded_tasks
):
    for agent_directed, scaffolded in zip(tasks, scaffolded_tasks):
        for action in agent_directed.evaluation_criteria.actions or []:
            if action.name == "submit_fields":
                assert "confirmed_with_user" not in action.arguments
        # The scaffolded set still carries the attestation (deep-copy guard:
        # the rewrite must never mutate the shared loader's objects).
        for action in scaffolded.evaluation_criteria.actions or []:
            if action.name == "submit_fields":
                assert "confirmed_with_user" in action.arguments


def test_variant_instructions_carry_the_agent_directed_contract():
    text = AGENT_DIRECTED_USER_TASK_INSTRUCTIONS
    assert "never volunteer a spelling" in text
    assert "note_spell_request" in text
    assert "note_readback" in text
    assert "i_affirmed" in text
    # The shared grounding rules survive the swap.
    assert 'exactly "Hello?"' in text
    assert "get_entity" in text


def test_drift_guard_fails_loud_on_changed_frozen_instructions(scaffolded_tasks):
    drifted = scaffolded_tasks[0].model_copy(deep=True)
    drifted.user_scenario.instructions.task_instructions = "something else"
    with pytest.raises(ValueError, match="drifted"):
        _agent_directed_task_variant(drifted)


def test_base_split_covers_every_task(tasks):
    splits = get_tasks_split()
    assert set(splits["base"]) == {t.id for t in tasks}


def test_shared_call_frame_matches_frozen_set():
    """The drift guard's premise: every frozen task carries the canonical
    call-frame instructions verbatim."""
    for task in load_frozen_tasks():
        assert (
            task.user_scenario.instructions.task_instructions == USER_TASK_INSTRUCTIONS
        )


# ---------------------------------------------------------------------------
# Golden replay (reward plumbing)
# ---------------------------------------------------------------------------


def _golden_submission(task: Task) -> tuple[str, dict]:
    action = next(
        a for a in task.evaluation_criteria.actions if a.name == "submit_fields"
    )
    return action.arguments["record_id"], dict(action.arguments["fields"])


def _replay_reward(task: Task, actions: list[tuple[str, str, dict]]) -> float:
    env = get_environment()
    init = task.initial_state
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=[],
    )
    messages: list[Message] = []
    for index, (requestor, name, arguments) in enumerate(actions):
        tool_call = ToolCall(
            id=f"sim_{index}", name=name, arguments=arguments, requestor=requestor
        )
        messages.append(
            AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])
        )
        messages.append(env.get_response(tool_call))
    return EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=messages,
    ).reward


@pytest.mark.parametrize("index", range(200))
def test_golden_replay_scores_one(tasks, index):
    task = tasks[index]
    actions = [
        (a.requestor, a.name, json.loads(json.dumps(a.arguments)))
        for a in task.evaluation_criteria.actions
    ]
    assert _replay_reward(task, actions) == 1.0


def test_no_submission_scores_zero(tasks):
    assert _replay_reward(tasks[0], []) == 0.0


# ---------------------------------------------------------------------------
# Versioned policy contract
# ---------------------------------------------------------------------------

_VERIFICATION_SENTENCE = (
    "Getting each value exactly right is what matters: if you are not sure "
    "you heard a value exactly, verify it with the person before recording "
    "it — for example by asking them to spell it letter by letter."
)
_VERIFICATION_ANCHOR = "How you get each value right is up to you."


def test_agent_directed_policy_version_and_sentence():
    from tau2.domains.intake.agent_directed import (
        AGENT_DIRECTED_POLICY_VERSION,
        POLICY_VERIFICATION_SENTENCE,
    )

    assert AGENT_DIRECTED_POLICY_VERSION == "1.4.0"
    assert POLICY_VERIFICATION_SENTENCE == _VERIFICATION_SENTENCE


def test_agent_directed_policy_carries_verification_sentence_once_anchored(env):
    """The environment exposes the pinned verification policy verbatim."""
    prompt = env.get_policy()
    assert prompt.count(_VERIFICATION_SENTENCE) == 1
    assert f"{_VERIFICATION_ANCHOR} {_VERIFICATION_SENTENCE}" in prompt
    assert env.get_policy() == AGENT_DIRECTED_POLICY_PATH.read_text()


def test_policy_load_fails_loud_on_verification_sentence_drift(tmp_path):
    """load_agent_directed_policy guards the versioned prose: a policy file
    without the anchored verification sentence must never load silently."""
    from unittest import mock

    from tau2.domains.intake import agent_directed

    drifted = tmp_path / "agent_directed_policy.md"
    drifted.write_text(
        AGENT_DIRECTED_POLICY_PATH.read_text().replace(
            f" {_VERIFICATION_SENTENCE}", "", 1
        )
    )
    with mock.patch.object(agent_directed, "AGENT_DIRECTED_POLICY_PATH", drifted):
        with pytest.raises(ValueError, match="drifted"):
            agent_directed.load_agent_directed_policy()

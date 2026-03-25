"""
Integration test: user tool call flow through StreamingMixin + orchestrator.

Verifies the full chain that was broken by the EnvironmentMessage bug:
  user emits tool_call → orchestrator executes against environment →
  results delivered next tick via tool_results parameter →
  user processes result → queues speech → delivers via turn-taking.

Uses mock streaming participants with the mock domain's real user tools
(check_notifications, dismiss_notification) so the entire environment
routing path (requestor="user" → use_user_tool) is exercised.
"""

from typing import Callable, List

from tau2.agent.base.streaming import (
    StreamingMixin,
    StreamingState,
    TurnTakingAction,
    basic_turn_taking_policy,
)
from tau2.agent.base_agent import FullDuplexAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.user.user_simulator_base import STOP, FullDuplexUser
from tau2.utils.utils import get_now


# ---------------------------------------------------------------------------
# Minimal state types
# ---------------------------------------------------------------------------
class AgentState(StreamingState[UserMessage, AssistantMessage]):
    pass


class UserState(StreamingState[AssistantMessage, UserMessage]):
    pass


# ---------------------------------------------------------------------------
# Mock agent: always replies, never makes tool calls
# ---------------------------------------------------------------------------
class SimpleAgent(
    StreamingMixin[UserMessage, AssistantMessage, AgentState],
    FullDuplexAgent[AgentState],
):
    def __init__(self, tools: List[Tool], domain_policy: str):
        self.tools = tools
        self.domain_policy = domain_policy

    def get_init_state(self, message_history=None):
        return AgentState()

    def speech_detection(self, chunk):
        return isinstance(chunk, UserMessage) and chunk.contains_speech

    @classmethod
    def is_stop(cls, message):
        return False

    def _next_turn_taking_action(self, state):
        action, info = basic_turn_taking_policy(
            state,
            wait_to_respond_threshold_other=1,
            wait_to_respond_threshold_self=1,
        )
        return TurnTakingAction(action=action, info=info)

    def _perform_turn_taking_action(self, state, action):
        if action.action == "keep_talking":
            chunk = state.output_streaming_queue.pop(0)
            chunk.timestamp = get_now()
            state.time_since_last_talk = 0
        elif action.action == "generate_message":
            msg = AssistantMessage(
                role="assistant",
                content="Sure, let me help.",
                contains_speech=True,
                chunk_id=0,
                is_final_chunk=True,
            )
            state.input_turn_taking_buffer = []
            state.output_streaming_queue.append(msg)
            chunk = state.output_streaming_queue.pop(0)
            chunk.timestamp = get_now()
            state.time_since_last_talk = 0
        else:
            state.time_since_last_talk += 1
            chunk = AssistantMessage(
                role="assistant", content=None, contains_speech=False
            )
        chunk.turn_taking_action = action
        return chunk, state

    def _create_chunk_messages(self, full_message):
        full_message.chunk_id = 0
        full_message.is_final_chunk = True
        return [full_message]

    def _process_tool_result(self, tool_result, state):
        state.time_since_last_talk += 1
        return (
            AssistantMessage(role="assistant", content=None, contains_speech=False),
            state,
        )

    def _emit_waiting_chunk(self, state):
        state.time_since_last_talk += 1
        return (
            AssistantMessage(role="assistant", content=None, contains_speech=False),
            state,
        )


# ---------------------------------------------------------------------------
# Mock user: calls check_notifications, then speaks, then stops
# ---------------------------------------------------------------------------
class ToolCallingUser(
    StreamingMixin[AssistantMessage, UserMessage, UserState],
    FullDuplexUser[UserState],
):
    def __init__(self, instructions=None, tools=None):
        self.instructions = instructions
        self.tools = tools or []
        self._generate_count = 0

    def get_init_state(self, message_history=None):
        return UserState()

    def speech_detection(self, chunk):
        return isinstance(chunk, AssistantMessage) and chunk.contains_speech

    @classmethod
    def is_stop(cls, message):
        return (
            isinstance(message, UserMessage)
            and message.content
            and STOP in message.content
        )

    def _next_turn_taking_action(self, state):
        action, info = basic_turn_taking_policy(
            state,
            wait_to_respond_threshold_other=1,
            wait_to_respond_threshold_self=1,
        )
        return TurnTakingAction(action=action, info=info)

    def _perform_turn_taking_action(self, state, action):
        if action.action == "keep_talking":
            chunk = state.output_streaming_queue.pop(0)
            chunk.timestamp = get_now()
            state.time_since_last_talk = 0
            if not state.output_streaming_queue and state.delivering_tool_result_speech:
                state.delivering_tool_result_speech = False
        elif action.action == "generate_message":
            state.input_turn_taking_buffer = []
            self._generate_count += 1
            if self._generate_count == 1:
                chunk = UserMessage(
                    role="user",
                    content="",
                    contains_speech=False,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="check_notifications",
                            arguments={},
                            requestor="user",
                        )
                    ],
                )
            else:
                chunk = UserMessage(
                    role="user",
                    content=f"Thanks, got it. {STOP}",
                    contains_speech=True,
                    chunk_id=0,
                    is_final_chunk=True,
                )
            state.time_since_last_talk = 0
        else:
            state.time_since_last_talk += 1
            chunk = UserMessage(role="user", content=None, contains_speech=False)
        chunk.turn_taking_action = action
        return chunk, state

    def _create_chunk_messages(self, full_message):
        full_message.chunk_id = 0
        full_message.is_final_chunk = True
        return [full_message]

    def _process_tool_result(self, tool_result, state):
        msg = UserMessage(
            role="user",
            content="I see my notifications.",
            contains_speech=True,
            chunk_id=0,
            is_final_chunk=True,
        )
        state.output_streaming_queue.append(msg)
        state.delivering_tool_result_speech = True
        chunk, state = self._emit_waiting_chunk(state)
        return chunk, state

    def _emit_waiting_chunk(self, state):
        state.time_since_last_talk += 1
        return UserMessage(role="user", content=None, contains_speech=False), state


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_user_tool_call_flow(
    domain_name: str,
    get_environment: Callable[[], Environment],
    task_with_user_tools: Task,
):
    """Full chain: user calls check_notifications → orchestrator executes →
    results delivered → user processes and speaks → conversation ends."""
    environment = get_environment()
    agent = SimpleAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
    )
    user = ToolCallingUser(instructions="test")

    orchestrator = FullDuplexOrchestrator(
        domain=domain_name,
        user=user,
        agent=agent,
        environment=environment,
        task=task_with_user_tools,
    )
    orchestrator.initialize()

    for _ in range(20):
        if orchestrator.done:
            break
        orchestrator.step()

    ticks = orchestrator.get_ticks()

    # User made a tool call
    tool_call_ticks = [t for t in ticks if t.user_tool_calls]
    assert len(tool_call_ticks) > 0, "User should have made a tool call"

    # Tool was executed and results generated
    tool_result_ticks = [t for t in ticks if t.user_tool_results]
    assert len(tool_result_ticks) > 0, "Tool results should have been produced"

    # Tool result contains actual notification data from mock environment
    first_result = tool_result_ticks[0].user_tool_results[0]
    assert (
        "notif_1" in first_result.content
        or "notification" in first_result.content.lower()
    )

    # User spoke after receiving tool results
    tool_call_tick_id = tool_call_ticks[0].tick_id
    user_spoke_after = any(
        t.tick_id > tool_call_tick_id and t.user_chunk and t.user_chunk.contains_speech
        for t in ticks
    )
    assert user_spoke_after, "User should have spoken after receiving tool results"

    # Simulation completed (user sent [STOP])
    assert orchestrator.done, "Simulation should have completed"

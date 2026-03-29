#!/usr/bin/env python3
"""
ReAct (Reasoning + Acting) agent example.

This example implements the ReAct pattern where the agent explicitly
reasons before each action:

    1. THINK  -- analyze the situation and decide what to do
    2. ACT    -- call a tool or respond to the user
    3. OBSERVE -- receive the tool result (handled by the orchestrator)

This loop repeats until the agent responds to the user with text
(no tool call), which ends the turn.

The key difference from the standard LLMAgent is the two-phase prompting:
- Phase 1: Generate a reasoning trace (no tools available)
- Phase 2: Choose an action based on the reasoning (tools available)

This pattern often improves tool-use accuracy because the agent
explicitly plans before acting.

Usage:
    python examples/agents/react_agent.py
"""

from typing import Optional

from loguru import logger

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.toolkit import Tool
from tau2.utils.llm_utils import generate

# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """\
You are a customer service agent that helps users according to the policy below.
Always follow the policy strictly. Use the provided tools when needed.

<policy>
{domain_policy}
</policy>"""

THINK_PROMPT = """\
Before responding, reason step-by-step about what to do next.

Consider:
- What is the user asking for?
- What information do I already have?
- What tool should I call, or should I respond directly?
- Does the policy require any specific steps?

Write your reasoning concisely."""

ACT_PROMPT = """\
Based on your reasoning above, now take action:
- If you need information or need to perform an operation, call the appropriate tool.
- If you have all the information needed, respond to the user directly.

Your reasoning was:
{reasoning}"""


# =============================================================================
# ReAct Agent
# =============================================================================


class ReActAgentState:
    """Conversation state for the ReAct agent."""

    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
    ):
        self.system_messages = system_messages
        self.messages = messages


class ReActAgent(HalfDuplexAgent[ReActAgentState]):
    """A ReAct agent that reasons before acting.

    Each turn follows the pattern:
        1. THINK: Ask the LLM to reason about the situation (no tools).
        2. ACT: Ask the LLM to take action based on its reasoning (with tools).

    The reasoning trace is injected into the action prompt so the LLM
    can refer to its own analysis when choosing a tool or composing a response.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "openai/gpt-4.1-mini",
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ReActAgentState:
        system_content = SYSTEM_PROMPT.format(domain_policy=self.domain_policy)
        system_messages = [SystemMessage(role="system", content=system_content)]
        messages = list(message_history) if message_history else []

        return ReActAgentState(
            system_messages=system_messages,
            messages=messages,
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: ReActAgentState,
    ) -> tuple[AssistantMessage, ReActAgentState]:
        # Add incoming message(s) to state
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Phase 1: THINK -- reason about the situation (no tools)
        reasoning = self._think(state)
        logger.debug(f"[ReAct] Reasoning: {reasoning[:200]}")

        # Phase 2: ACT -- choose an action based on reasoning (with tools)
        assistant_message = self._act(state, reasoning)

        # Store the action in conversation history
        # (reasoning is ephemeral -- not stored in history)
        state.messages.append(assistant_message)

        return assistant_message, state

    def _think(self, state: ReActAgentState) -> str:
        """Phase 1: Generate a reasoning trace without tools."""
        think_messages = (
            state.system_messages
            + state.messages
            + [UserMessage(role="user", content=THINK_PROMPT)]
        )

        # Call LLM without tools -- forces text-only reasoning
        response = generate(
            model=self.llm,
            tools=[],  # No tools available during thinking
            messages=think_messages,
            call_name="react_think",
            **self.llm_args,
        )

        return str(response.content) if response.content else ""

    def _act(self, state: ReActAgentState, reasoning: str) -> AssistantMessage:
        """Phase 2: Choose an action based on the reasoning trace."""
        act_instruction = ACT_PROMPT.format(reasoning=reasoning)
        act_messages = (
            state.system_messages
            + state.messages
            + [UserMessage(role="user", content=act_instruction)]
        )

        # Call LLM with tools -- can choose to call a tool or respond
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=act_messages,
            call_name="react_act",
            **self.llm_args,
        )

        return response


# =============================================================================
# Factory and registration
# =============================================================================


def create_react_agent(tools, domain_policy, **kwargs):
    """Factory function for the registry."""
    return ReActAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-4.1-mini"),
        llm_args=kwargs.get("llm_args"),
    )


# =============================================================================
# Run it
# =============================================================================

if __name__ == "__main__":
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import get_tasks, run_single_task

    # Register the ReAct agent
    registry.register_agent_factory(create_react_agent, "react_agent")

    # Run on a mock task
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    result = run_single_task(
        TextRunConfig(
            domain="mock",
            agent="react_agent",
            llm_agent="openai/gpt-4.1-mini",
        ),
        tasks[0],
        seed=42,
    )

    print(f"\nTask: {result.task_id}")
    print(f"Reward: {result.reward_info.reward}")
    print(f"Messages: {len(result.messages)}")
    print()

    # Print conversation
    for msg in result.messages:
        role = msg.role.value if hasattr(msg.role, "value") else msg.role
        if msg.content:
            print(f"  [{role}] {str(msg.content)[:120]}")
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            names = [tc.name for tc in msg.tool_calls]
            print(f"  [{role}] Tool calls: {names}")
        else:
            print(f"  [{role}] (tool result)")

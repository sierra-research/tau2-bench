#!/usr/bin/env python3
"""
Minimal custom agent example.

This example shows how to create the simplest possible tau2 agent,
register it, and run it against the mock domain -- all in one file.

No API keys are needed (uses the mock domain with a real LLM).

Usage:
    python examples/agents/minimal_text_agent.py

What this does:
    1. Defines a custom agent that wraps an LLM with a system prompt
    2. Registers it with the tau2 registry
    3. Runs it on a single mock task
    4. Prints the result
"""

from typing import Optional

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from tau2.environment.toolkit import Tool
from tau2.utils.llm_utils import generate

# =============================================================================
# Step 1: Define your agent
# =============================================================================


class MinimalAgentState:
    """Simple state container for the MinimalAgent."""

    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
    ):
        self.system_messages = system_messages
        self.messages = messages


class MinimalAgent(HalfDuplexAgent[MinimalAgentState]):
    """A minimal agent that uses an LLM to respond to messages.

    The state holds the conversation history as tau2 Message objects,
    which is the format expected by the generate() utility.
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
    ) -> MinimalAgentState:
        """Build the initial conversation state with a system prompt."""
        system_prompt = (
            f"You are a helpful customer service agent.\n\n"
            f"## Domain Policy\n{self.domain_policy}\n\n"
            f"Follow the policy strictly. Use the provided tools to help the user."
        )
        return MinimalAgentState(
            system_messages=[SystemMessage(role="system", content=system_prompt)],
            messages=list(message_history) if message_history else [],
        )

    def generate_next_message(
        self, message: UserMessage, state: MinimalAgentState
    ) -> tuple[AssistantMessage, MinimalAgentState]:
        """Generate a response using the LLM."""
        # Add the incoming message to state
        state.messages.append(message)

        # Call the LLM with tools (generate expects tau2 Message objects)
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            **self.llm_args,
        )

        # Add the response to the conversation and return
        state.messages.append(response)
        return response, state


# =============================================================================
# Step 2: Create a factory function
# =============================================================================


def create_minimal_agent(tools, domain_policy, **kwargs):
    """Factory function for the registry.

    The registry calls this with:
        tools: list[Tool] -- the domain's tools
        domain_policy: str -- the domain's policy document
        **kwargs may include: llm, llm_args, task, solo_mode
    """
    return MinimalAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm", "openai/gpt-4.1-mini"),
        llm_args=kwargs.get("llm_args"),
    )


# =============================================================================
# Step 3: Register and run
# =============================================================================

if __name__ == "__main__":
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.runner import get_tasks, run_single_task

    # Register our agent
    registry.register_agent_factory(create_minimal_agent, "minimal_agent")

    # --- Option A: Run a single task programmatically ---
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    result = run_single_task(
        TextRunConfig(
            domain="mock",
            agent="minimal_agent",
            llm_agent="openai/gpt-4.1-mini",
        ),
        tasks[0],
        seed=42,
    )
    print(f"\nTask: {result.task_id}")
    print(f"Reward: {result.reward_info.reward}")
    print(f"Messages: {len(result.messages)}")

    # --- Option B: Run all mock tasks via run_domain ---
    # Uncomment to run a full evaluation:
    #
    # results = run_domain(
    #     TextRunConfig(
    #         domain="mock",
    #         agent="minimal_agent",
    #         llm_agent="openai/gpt-4.1-mini",
    #         num_trials=1,
    #         max_concurrency=1,
    #     )
    # )

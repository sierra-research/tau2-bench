"""A2A Agent implementation for tau2-bench."""

from typing import List, Optional

import httpx
from loguru import logger

from tau2.a2a.client import A2AClient
from tau2.a2a.models import A2AAgentState, A2AConfig
from tau2.a2a.translation import (
    a2a_to_tau2_assistant_message,
    tau2_to_a2a_message_content,
)
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, Message
from tau2.environment.tool import Tool


class A2AAgent(HalfDuplexAgent):
    """
    Agent that communicates with remote A2A-compliant agents.

    Implements the HalfDuplexAgent interface by translating tau2 messages
    to A2A protocol format, sending them via HTTP, and parsing responses
    back to tau2 AssistantMessage format.
    """

    def __init__(
        self,
        config: A2AConfig,
        tools: List[Tool],
        domain_policy: str,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        Initialize A2A agent.

        Args:
            config: A2A configuration (endpoint, auth, timeout)
            tools: List of tools available in this domain
            domain_policy: Domain-specific policy text
            http_client: Optional HTTP client for testing (uses config if None)
        """
        super().__init__(tools=tools, domain_policy=domain_policy)

        self.config = config
        self.client = A2AClient(config=config, http_client=http_client)

        logger.info(
            f"Initialized A2AAgent (endpoint={config.endpoint}, "
            f"timeout={config.timeout}, num_tools={len(tools)})"
        )

    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> A2AAgentState:
        """
        Get the initial state of the agent.

        Args:
            message_history: Optional message history to initialize with

        Returns:
            Fresh A2AAgentState with no context_id
        """
        logger.trace(
            f"Initializing A2A agent state "
            f"(history_length={len(message_history or [])})"
        )
        return A2AAgentState(
            context_id=None,
            conversation_history=message_history or [],
            agent_card=None,
            request_count=0,
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: A2AAgentState,
    ) -> tuple[AssistantMessage, A2AAgentState]:
        """Respond to a user or tool message via the remote A2A agent."""
        import asyncio

        async def _async_generate():
            tools_for_translation = self.tools if message.role == "user" else None
            is_first_message = state.request_count == 0
            policy_for_translation = self.domain_policy if is_first_message else None

            a2a_content = tau2_to_a2a_message_content(
                message,
                tools=tools_for_translation,
                domain_policy=policy_for_translation,
                is_first_message=is_first_message,
            )

            logger.debug(
                f"Sending message to A2A agent (role={message.role}, "
                f"length={len(a2a_content)}, context_id={state.context_id})"
            )

            if state.context_id is None:
                logger.trace(
                    f"A2A context lifecycle: first message "
                    f"(request_count={state.request_count})"
                )
            else:
                logger.trace(
                    f"A2A context lifecycle: reusing context "
                    f"(context_id={state.context_id}, "
                    f"request_count={state.request_count})"
                )

            # Send message to A2A agent
            response_content, new_context_id = await self.client.send_message(
                message_content=a2a_content,
                context_id=state.context_id,
            )

            logger.debug(
                f"Received response from A2A agent "
                f"(length={len(response_content)}, "
                f"context_id={new_context_id})"
            )

            if state.context_id is None and new_context_id is not None:
                logger.trace(
                    f"A2A context lifecycle: new context created "
                    f"(context_id={new_context_id})"
                )
            elif state.context_id != new_context_id:
                logger.warning(
                    f"A2A context lifecycle: context changed unexpectedly "
                    f"(old={state.context_id}, new={new_context_id})"
                )

            assistant_msg = a2a_to_tau2_assistant_message(response_content)

            new_conversation_history = state.conversation_history + [
                message,
                assistant_msg,
            ]

            new_state = A2AAgentState(
                context_id=new_context_id or state.context_id,
                conversation_history=new_conversation_history,
                agent_card=state.agent_card,
                request_count=state.request_count + 1,
            )

            return assistant_msg, new_state

        # Run async function synchronously.
        # This method is called from thread pool workers (via run_in_executor in
        # run_tau2_evaluation.py), which never have a running event loop.
        # Using asyncio.run() creates a fresh event loop for this thread.
        #
        # IMPORTANT: Do NOT use nested ThreadPoolExecutor here - it causes deadlock
        # when multiple concurrent evaluations run.
        return asyncio.run(_async_generate())

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[A2AAgentState] = None,
    ) -> None:
        """
        Stop the agent and release resources.

        Args:
            message: The last message to the agent.
            state: The agent state.
        """
        import asyncio

        async def _async_close():
            await self.client.close()

        asyncio.run(_async_close())
        logger.debug("A2AAgent stopped and resources cleaned up")


def create_a2a_agent(tools, domain_policy, **kwargs):
    """Factory function for A2AAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - a2a_agent_args (dict): A2A configuration with keys:
                - endpoint (str, required): A2A agent endpoint URL
                - auth_token (str, optional): Bearer token for authentication
                - timeout (int, optional): Response timeout in seconds (default 300)
    """
    a2a_agent_args = kwargs.get("a2a_agent_args") or {}
    config = A2AConfig(
        endpoint=a2a_agent_args["endpoint"],
        auth_token=a2a_agent_args.get("auth_token"),
        timeout=a2a_agent_args.get("timeout", 300),
    )
    return A2AAgent(config=config, tools=tools, domain_policy=domain_policy)

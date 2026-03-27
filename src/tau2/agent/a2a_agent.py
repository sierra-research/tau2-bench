"""A2A Agent implementation for tau2-bench."""

from typing import List, Optional

import httpx
from a2a.client import Client, ClientConfig, ClientFactory
from a2a.client.client_factory import minimal_agent_card
from a2a.types import Message as A2AMessage
from a2a.utils.message import get_message_text
from loguru import logger

from tau2.a2a.models import A2AAgentState, A2AConfig
from tau2.a2a.translation import (
    a2a_to_tau2_assistant_message,
    format_tools_as_text,
    tau2_to_a2a_message,
)
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, Message, ToolMessage
from tau2.environment.tool import Tool

# Can refactor into shared utils. Currently borrowed from tau2.voice.audio_native.async_loop
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop


def create_a2a_client(config: A2AConfig) -> Client:
    """Create an SDK Client from A2AConfig.

    Uses minimal_agent_card() to bootstrap a client without requiring
    an upfront agent card discovery call.

    Args:
        config: tau2 A2A configuration

    Returns:
        SDK Client configured for the endpoint
    """
    httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.timeout, connect=config.connect_timeout),
        verify=config.verify_ssl,
        headers=(
            {"Authorization": f"Bearer {config.auth_token}"}
            if config.auth_token
            else {}
        ),
        follow_redirects=True,
    )
    card = minimal_agent_card(config.endpoint)
    factory = ClientFactory(
        ClientConfig(httpx_client=httpx_client, streaming=False),
    )
    return factory.create(card)


class A2AAgent(HalfDuplexAgent):
    """
    Agent that communicates with remote A2A-compliant agents.

    Implements the HalfDuplexAgent interface by translating tau2 messages
    to A2A protocol format, sending them via the SDK Client, and parsing
    responses back to tau2 AssistantMessage format.

    Uses ``BackgroundAsyncLoop`` to bridge the sync HalfDuplexAgent
    interface with the async A2A SDK, preserving httpx connection
    pooling across calls.
    """

    def __init__(
        self,
        config: A2AConfig,
        tools: List[Tool],
        domain_policy: str,
        client: Optional[Client] = None,
    ):
        """
        Initialize A2A agent.

        Args:
            config: A2A configuration (endpoint, auth, timeout)
            tools: List of tools available in this domain
            domain_policy: Domain-specific policy text
            client: Optional SDK Client for testing (created from config if None)
        """
        super().__init__(tools=tools, domain_policy=domain_policy)

        self.config = config

        self._bg_loop = BackgroundAsyncLoop()
        self._bg_loop.start()

        self._client = client or create_a2a_client(config)

        self._valid_tool_names = {tool.name for tool in tools}

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
            request_count=0,
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: A2AAgentState,
    ) -> tuple[AssistantMessage, A2AAgentState]:
        """Respond to a user or tool message via the remote A2A agent."""

        async def _async_generate():
            tools_for_translation = self.tools if message.role == "user" else None
            is_first_message = state.request_count == 0
            policy_for_translation = self.domain_policy if is_first_message else None

            a2a_msg = tau2_to_a2a_message(
                message,
                tools=tools_for_translation,
                domain_policy=policy_for_translation,
                is_first_message=is_first_message,
            )

            # On tool errors, enhance message with available tools for self-correction
            if isinstance(message, ToolMessage) and message.error:
                tool_text = format_tools_as_text(self.tools)
                if tool_text:
                    from a2a.client.helpers import create_text_message_object
                    from a2a.types import Role as A2ARole

                    current_text = get_message_text(a2a_msg)
                    enhanced_text = (
                        current_text
                        + f"\n\n{tool_text}"
                        + "\nTo use a tool, respond with JSON: "
                        '{"tool_call": {"name": "tool_name", "arguments": {"param1": "value"}}}'
                    )
                    a2a_msg = create_text_message_object(A2ARole.user, enhanced_text)

            logger.debug(
                f"Sending message to A2A agent (role={message.role}, "
                f"context_id={state.context_id})"
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

            # Set context_id on the outgoing message
            a2a_msg.context_id = state.context_id

            # Send message via SDK client
            result = None
            async for event in self._client.send_message(a2a_msg):
                if isinstance(event, A2AMessage):
                    result = event
                else:
                    # ClientEvent is tuple[Task, UpdateEvent]
                    result = event[0]
                break

            if result is None:
                logger.warning("A2A agent returned no events")
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content="I apologize, but I was unable to generate a response. Could you please rephrase your request?",
                    tool_calls=None,
                )
                return assistant_msg, state

            # Extract context_id from result
            new_context_id = result.context_id

            logger.debug(
                f"Received response from A2A agent (context_id={new_context_id})"
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

            assistant_msg = a2a_to_tau2_assistant_message(result)

            # Log invalid tool calls for diagnostics
            if assistant_msg.is_tool_call():
                invalid = [
                    tc.name
                    for tc in assistant_msg.tool_calls
                    if tc.name not in self._valid_tool_names
                ]
                if invalid:
                    logger.debug(f"A2A agent produced invalid tool call(s): {invalid}")

            new_conversation_history = state.conversation_history + [
                message,
                assistant_msg,
            ]

            new_state = A2AAgentState(
                context_id=new_context_id or state.context_id,
                conversation_history=new_conversation_history,
                request_count=state.request_count + 1,
            )

            return assistant_msg, new_state

        return self._bg_loop.run_coroutine(
            _async_generate(), timeout=float(self.config.timeout)
        )

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
        if self._bg_loop.is_running:
            self._bg_loop.run_coroutine(self._client.close(), timeout=5.0)
            self._bg_loop.stop()
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

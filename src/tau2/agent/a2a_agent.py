"""A2A Agent implementation for tau2-bench."""

from typing import Any

import httpx
from loguru import logger

from tau2.a2a.client import A2AClient
from tau2.a2a.metrics import AggregatedMetrics, ProtocolMetrics
from tau2.a2a.models import A2AAgentState, A2AConfig
from tau2.a2a.translation import (
    a2a_to_tau2_assistant_message,
    tau2_to_a2a_message_content,
)
from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, Message
from tau2.environment.tool import Tool


class A2AAgent(LocalAgent):
    """
    Agent that communicates with remote A2A-compliant agents.

    Implements the BaseAgent interface by:
    - Translating tau2 messages to A2A protocol format
    - Sending messages via HTTP to remote A2A agent
    - Parsing A2A responses back to tau2 AssistantMessage format
    - Managing session context across multi-turn conversations
    """

    def __init__(
        self,
        config: A2AConfig,
        tools: list[Tool],
        domain_policy: str,
        http_client: httpx.AsyncClient | None = None,
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
            "Initialized A2AAgent",
            endpoint=config.endpoint,
            timeout=config.timeout,
            num_tools=len(tools),
        )

    def get_init_state(
        self,
        message_history: list[Message] | None = None,
    ) -> A2AAgentState:
        """
        Get the initial state of the agent.

        Args:
            message_history: Optional message history to initialize with

        Returns:
            Fresh A2AAgentState with no context_id (will be set on first response)
        """
        logger.trace(
            "Initializing A2A agent state",
            context_id=None,
            message_history_length=len(message_history or []),
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
        """
        Produce the next assistant message by sending the provided input to the remote A2A agent and update the agent state.
        
        Parameters:
            message: The incoming user or tool-result message to deliver to the remote agent.
            state: The current A2AAgentState (context, conversation history, request count).
        
        Returns:
            A tuple of (AssistantMessage, A2AAgentState) where the AssistantMessage is the agent's reply and the A2AAgentState is the updated state with a possibly new context_id, extended conversation history, and incremented request_count.
        """
        import asyncio

        # Async/sync bridge: Run async HTTP operations in synchronous context
        async def _async_generate():
            # Translate tau2 message to A2A content
            # Include tools for user messages so agent knows what's available
            """
            Send the current tau2 message to the remote A2A agent and return the produced assistant message along with an updated A2AAgentState.
            
            The function translates the provided tau2 message into A2A format (including available tools for user messages), sends it to the remote agent via the internal A2A client, translates the agent's A2A response into a tau2 AssistantMessage, and constructs a new state that appends the turn to the conversation history, preserves or updates the context_id returned by the agent, and increments the request count.
            
            Returns:
                tuple[AssistantMessage, A2AAgentState]: The assistant message generated from the A2A response and the updated agent state.
            """
            tools_for_translation = self.tools if message.role == "user" else None
            a2a_content = tau2_to_a2a_message_content(message, tools=tools_for_translation)

            logger.debug(
                "Sending message to A2A agent",
                role=message.role,
                content_length=len(a2a_content),
                context_id=state.context_id,
            )

            # Debug: Log context_id lifecycle - before request
            if state.context_id is None:
                logger.trace(
                    "A2A context_id lifecycle: First message, no context yet",
                    request_count=state.request_count,
                )
            else:
                logger.trace(
                    "A2A context_id lifecycle: Reusing existing context",
                    context_id=state.context_id,
                    request_count=state.request_count,
                )

            # Send message to A2A agent
            response_content, new_context_id = await self.client.send_message(
                message_content=a2a_content,
                context_id=state.context_id,
            )

            logger.debug(
                "Received response from A2A agent",
                response_length=len(response_content),
                new_context_id=new_context_id,
            )

            # Debug: Log context_id lifecycle - after response
            if state.context_id is None and new_context_id is not None:
                logger.trace(
                    "A2A context_id lifecycle: New context created by agent",
                    new_context_id=new_context_id,
                    request_count=state.request_count,
                )
            elif state.context_id == new_context_id:
                logger.trace(
                    "A2A context_id lifecycle: Context persisted across turns",
                    context_id=new_context_id,
                    request_count=state.request_count,
                )
            elif state.context_id != new_context_id:
                logger.warning(
                    "A2A context_id lifecycle: Context changed unexpectedly",
                    old_context_id=state.context_id,
                    new_context_id=new_context_id,
                    request_count=state.request_count,
                )

            # Translate A2A response to tau2 AssistantMessage
            assistant_msg = a2a_to_tau2_assistant_message(response_content)

            # Update state
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
        # when multiple concurrent evaluations run, as each nested executor blocks
        # its parent worker thread waiting on future.result().
        # See: specs/007-datadog-project/resolve-tau2agent-concurrency.md
        return asyncio.run(_async_generate())

    def stop(
        self,
        message: ValidAgentInputMessage | None = None,
        state: A2AAgentState | None = None,
    ) -> None:
        """
        Stop the agent and release its resources.
        
        Closes the agent's internal HTTP client by running its asynchronous close routine; the implementation will create or reuse an event loop as needed to perform the shutdown.
        
        Parameters:
            message (ValidAgentInputMessage | None): Ignored; present for interface compatibility.
            state (A2AAgentState | None): Ignored; present for interface compatibility.
        """
        import asyncio

        # Close HTTP client if needed
        async def _async_close():
            await self.client.close()

        # Run async close synchronously - same pattern as generate_next_message()
        # See: specs/007-datadog-project/resolve-tau2agent-concurrency.md
        asyncio.run(_async_close())

        logger.debug("A2AAgent stopped and resources cleaned up")

    def get_protocol_metrics(self) -> list[ProtocolMetrics]:
        """
        Get all collected protocol metrics from the A2A client.

        Returns:
            List of ProtocolMetrics for all A2A requests made by this agent
        """
        return self.client.get_metrics()

    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """
        Get aggregated protocol metrics summary.

        Returns:
            AggregatedMetrics with computed summary statistics
        """
        metrics = self.get_protocol_metrics()
        return AggregatedMetrics.from_protocol_metrics(metrics)

    def export_metrics_json(self, task_id: str | None = None) -> dict[str, Any]:
        """
        Export protocol metrics in JSON format for tau2-bench results.

        This format follows the specification in data-model.md and can be
        integrated into tau2-bench's results export.

        Args:
            task_id: Optional task identifier for context

        Returns:
            Dictionary with protocol metrics and summary in tau2-bench format
        """
        protocol_metrics = self.get_protocol_metrics()
        aggregated_metrics = self.get_aggregated_metrics()

        return {
            "task_id": task_id,
            "agent_type": "a2a_agent",
            "protocol_metrics": [m.to_dict() for m in protocol_metrics],
            "summary": aggregated_metrics.model_dump(),
        }

    def clear_metrics(self) -> None:
        """Clear all collected protocol metrics."""
        self.client.clear_metrics()

    @classmethod
    def from_cli_args(
        cls,
        llm: str,
        llm_args: dict,
        tools: list[Tool],
        domain_policy: str,
    ) -> "A2AAgent":
        """
        Create A2AAgent from CLI arguments.

        This follows tau2-bench's agent construction pattern where:
        - llm parameter contains the A2A endpoint
        - llm_args contains auth_token and timeout

        Args:
            llm: A2A agent endpoint URL
            llm_args: Dict with optional 'auth_token' and 'timeout' keys
            tools: List of available tools
            domain_policy: Domain policy text

        Returns:
            Configured A2AAgent instance
        """
        config = A2AConfig(
            endpoint=llm,
            auth_token=llm_args.get("auth_token"),
            timeout=llm_args.get("timeout", 300),
        )

        return cls(
            config=config,
            tools=tools,
            domain_policy=domain_policy,
        )
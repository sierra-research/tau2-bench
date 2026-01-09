"""
LangChain Agent implementation for tau2-bench.

This agent uses LangGraph's create_react_agent to provide a conversational
agent that can interact with domain tools.
"""

import os
from copy import deepcopy
from typing import Callable, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage as LangChainSystemMessage,
)
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.utils.llm_utils import get_response_cost, get_response_usage
from tau2.environment.tool import Tool


class CostTrackingCallback(BaseCallbackHandler):
    """Callback to track LLM costs from LangChain calls."""

    def __init__(self):
        super().__init__()
        self.last_cost = None
        self.last_usage = None

    def on_llm_end(self, response, **kwargs):
        """Called when LLM call ends. Extract cost and usage from response."""
        try:
            # LangChain's ChatOpenAI uses LiteLLM, which stores cost in response_metadata
            if hasattr(response, "response_metadata"):
                metadata = response.response_metadata
                if metadata:
                    # Try to get cost
                    if "cost" in metadata:
                        self.last_cost = metadata["cost"]
                    # Try to get usage
                    if "usage" in metadata:
                        self.last_usage = metadata["usage"]

                    # Also check for LiteLLM response in _hidden_params
                    if "_hidden_params" in metadata:
                        hidden = metadata.get("_hidden_params", {})
                        if "response" in hidden:
                            from tau2.utils.llm_utils import (
                                get_response_cost,
                                get_response_usage,
                            )

                            try:
                                litellm_response = hidden["response"]
                                if self.last_cost is None:
                                    self.last_cost = get_response_cost(litellm_response)
                                if self.last_usage is None:
                                    self.last_usage = get_response_usage(
                                        litellm_response
                                    )
                            except Exception:
                                pass
        except Exception as e:
            logger.debug(f"Error extracting cost from callback: {e}")

    def reset(self):
        """Reset tracked costs (call before each agent invocation)."""
        self.last_cost = None
        self.last_usage = None


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class LangChainAgentState(BaseModel):
    """The state of the LangChain agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    langchain_messages: list  # Store LangChain messages for the agent
    tool_execution_cache: dict = {}  # Cache of tool_call_id -> ToolMessage for evaluation replay


class LangChainAgent(LocalAgent[LangChainAgentState]):
    """
    A LangChain agent that uses LangGraph's create_react_agent.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tool_executor: Optional[Callable] = None,
    ):
        """
        Initialize the LangChainAgent.

        Args:
            tools: List of tau2 Tool objects
            domain_policy: Domain policy document
            llm: LLM model name (e.g., "openai/gpt-oss-120b")
            llm_args: Additional LLM parameters
            base_url: Optional base URL for the LLM API
            api_key: Optional API key (if not provided, will use environment variable)
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.base_url = base_url
        self.api_key = api_key
        self.tool_executor = (
            tool_executor  # Callback to execute tools through orchestrator
        )

        # Convert tau2 tools to LangChain tools
        self.langchain_tools = self._convert_tools_to_langchain(tools)

        # Initialize LangChain LLM
        self.langchain_llm = self._create_langchain_llm()

        # Create LangGraph agent
        self.system_prompt_text = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

        # Log tool count for debugging
        logger.debug(
            f"Creating LangGraph agent with {len(self.langchain_tools)} tools: "
            f"{[t.name for t in self.langchain_tools[:5]]}"
        )

        # Create a callback handler to track LLM costs
        self.cost_tracker = CostTrackingCallback()

        self.agent = create_react_agent(
            self.langchain_llm,
            self.langchain_tools,
            prompt=self.system_prompt_text,
        )

    def _create_langchain_llm(self) -> ChatOpenAI:
        """Create a LangChain ChatOpenAI instance."""
        llm_kwargs = deepcopy(self.llm_args) if self.llm_args else {}

        # Handle base_url and api_key
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Handle api_key: use provided value, or try to load from environment if missing/empty
        api_key_provided = self.api_key and self.api_key.strip()
        if api_key_provided:
            llm_kwargs["api_key"] = self.api_key
        elif "api_key" not in llm_kwargs or not llm_kwargs.get("api_key", "").strip():
            # Try to get from environment if using Nebius API
            # Check if base_url points to Nebius or model name contains "nebius"
            is_nebius = (self.base_url and "nebius" in self.base_url.lower()) or (
                self.llm and "nebius" in self.llm.lower()
            )
            if is_nebius:
                api_key = os.getenv("NEBIUS_API_KEY")
                if api_key:
                    llm_kwargs["api_key"] = api_key
                    logger.debug("Loaded NEBIUS_API_KEY from environment")
                else:
                    logger.warning(
                        "Nebius API detected but no api_key provided and NEBIUS_API_KEY not found in environment"
                    )

        # Ensure api_key is set (required by OpenAI client)
        if "api_key" not in llm_kwargs or not llm_kwargs.get("api_key", "").strip():
            raise ValueError(
                "api_key must be provided either as a parameter, in llm_args, or via NEBIUS_API_KEY environment variable"
            )

        return ChatOpenAI(model=self.llm, **llm_kwargs)

    def _convert_tools_to_langchain(self, tools: List[Tool]) -> List[StructuredTool]:
        """Convert tau2 Tool objects to LangChain StructuredTool objects.

        IMPORTANT: If tool_executor is provided, tools will be executed through
        the orchestrator's environment to ensure state consistency. Otherwise,
        they execute directly (which can cause state mismatches during evaluation).
        """
        langchain_tools = []

        for tool in tools:
            # Create a wrapper function that executes tools
            # Use a closure to capture the tool and tool_executor
            def make_tool_func(tau2_tool: Tool, executor=None):
                def tool_func(**kwargs):
                    # If we have a tool_executor (from orchestrator), use it
                    # This executes tools through an isolated environment instance
                    # that doesn't affect the main environment state
                    if executor is not None:
                        # Create a ToolCall object for the executor
                        from tau2.data_model.message import ToolCall
                        import uuid

                        tool_call = ToolCall(
                            id=f"langgraph-{uuid.uuid4()}",
                            name=tau2_tool.name,
                            arguments=kwargs,
                        )

                        # Execute through isolated environment (doesn't affect main environment)
                        tool_message = executor(tool_call)

                        # Return the content (LangGraph expects a string)
                        if isinstance(tool_message.content, (dict, list)):
                            import json

                            return json.dumps(tool_message.content, indent=2)
                        return str(tool_message.content) if tool_message.content else ""
                    else:
                        # Fallback: execute directly (legacy behavior)
                        try:
                            result = tau2_tool(**kwargs)
                            if isinstance(result, (dict, list)):
                                import json

                                return json.dumps(result, indent=2)
                            return str(result) if result is not None else ""
                        except Exception as e:
                            logger.error(f"Error calling tool {tau2_tool.name}: {e}")
                            import traceback

                            logger.debug(traceback.format_exc())
                            return f"Error: {str(e)}"

                # Set function name and docstring for better debugging
                tool_func.__name__ = tau2_tool.name
                tool_func.__doc__ = tau2_tool._get_description()
                return tool_func

            # Get the function signature from the tool
            tool_func = make_tool_func(tool, self.tool_executor)

            # Create LangChain StructuredTool
            langchain_tool = StructuredTool.from_function(
                func=tool_func,
                name=tool.name,
                description=tool._get_description(),
                args_schema=tool.params,
            )
            langchain_tools.append(langchain_tool)

        return langchain_tools

    @property
    def system_prompt(self) -> str:
        return self.system_prompt_text

    def _tau2_to_langchain_message(self, message: Message) -> Optional[object]:
        """Convert a tau2 message to a LangChain message."""
        if isinstance(message, UserMessage):
            return HumanMessage(content=message.content or "")
        elif isinstance(message, AssistantMessage):
            # LangGraph agent handles assistant messages internally
            # We'll convert tool calls separately if needed
            return AIMessage(content=message.content or "")
        elif isinstance(message, ToolMessage):
            # Tool messages are handled by LangGraph internally
            from langchain_core.messages import ToolMessage as LangChainToolMessage

            return LangChainToolMessage(
                content=message.content or "", tool_call_id=message.id
            )
        elif isinstance(message, SystemMessage):
            return LangChainSystemMessage(content=message.content or "")
        return None

    def _langchain_to_tau2_assistant_message(
        self, langchain_messages: list
    ) -> AssistantMessage:
        """Convert LangGraph agent output to tau2 AssistantMessage."""
        # LangGraph returns all messages, we need to find AIMessages with tool_calls or content
        # Look backwards through messages to find the last AIMessage
        # Priority: AIMessage with tool_calls > AIMessage with content
        content = None
        tool_calls = None
        cost = None
        usage = None

        # Find the last AIMessage with tool_calls first (highest priority)
        ai_msg_with_tools = None
        ai_msg_with_content = None

        for msg in reversed(langchain_messages):
            if isinstance(msg, AIMessage):
                # Check for tool calls first
                if (
                    hasattr(msg, "tool_calls")
                    and msg.tool_calls
                    and not ai_msg_with_tools
                ):
                    ai_msg_with_tools = msg
                # Also track last AIMessage with content
                if hasattr(msg, "content") and msg.content and not ai_msg_with_content:
                    ai_msg_with_content = msg

        # Prefer AIMessage with tool_calls over one with just content
        target_msg = ai_msg_with_tools or ai_msg_with_content

        if target_msg:
            # Extract cost and usage from response_metadata if available
            # LangChain's ChatOpenAI stores LiteLLM response in response_metadata
            if hasattr(target_msg, "response_metadata"):
                metadata = target_msg.response_metadata
                if metadata:
                    # Try to extract cost from metadata
                    # LiteLLM stores cost in various places depending on version
                    if "cost" in metadata:
                        cost = metadata["cost"]
                    elif "_hidden_params" in metadata:
                        # LiteLLM sometimes stores response in _hidden_params
                        hidden = metadata.get("_hidden_params", {})
                        if "response" in hidden:
                            response = hidden["response"]
                            try:
                                cost = get_response_cost(response)
                            except Exception:
                                pass

                    # Try to extract usage from metadata
                    if "usage" in metadata:
                        usage = metadata["usage"]
                    elif "_hidden_params" in metadata:
                        hidden = metadata.get("_hidden_params", {})
                        if "response" in hidden:
                            response = hidden["response"]
                            try:
                                usage = get_response_usage(response)
                            except Exception:
                                pass

            # Check for tool calls first (prioritize tool calls over content)
            if hasattr(target_msg, "tool_calls") and target_msg.tool_calls:
                tool_calls = []
                for tc in target_msg.tool_calls:
                    # Handle different tool call formats
                    if isinstance(tc, dict):
                        # Extract arguments - could be in 'args' or 'arguments'
                        args = tc.get("args", tc.get("arguments", {}))
                        # If args is a string, try to parse as JSON
                        if isinstance(args, str):
                            import json

                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                args = {}
                        tool_calls.append(
                            ToolCall(
                                id=tc.get("id", ""),
                                name=tc.get("name", ""),
                                arguments=args,
                            )
                        )
                    else:
                        # LangChain tool call object
                        args = getattr(tc, "args", getattr(tc, "arguments", {}))
                        if isinstance(args, str):
                            import json

                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                args = {}
                        tool_calls.append(
                            ToolCall(
                                id=getattr(tc, "id", ""),
                                name=getattr(tc, "name", ""),
                                arguments=args,
                            )
                        )
                # If we have tool calls, don't include content (tau2 requirement)
                content = None
            else:
                # No tool calls, check for content
                if hasattr(target_msg, "content") and target_msg.content:
                    content = target_msg.content

        # If no content and no tool calls, provide a default message
        if not content and not tool_calls:
            content = "I'm ready to help you."

        return AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls or None,
            cost=cost,
            usage=usage,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LangChainAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )

        # Convert message history to LangChain format
        langchain_messages = []
        for msg in message_history:
            lc_msg = self._tau2_to_langchain_message(msg)
            if lc_msg:
                langchain_messages.append(lc_msg)

        return LangChainAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
            langchain_messages=langchain_messages,
            tool_execution_cache={},  # Initialize empty cache for recording tool executions
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LangChainAgentState
    ) -> tuple[AssistantMessage, LangChainAgentState]:
        """
        Respond to a user or tool message using LangGraph agent.
        """
        # Add the new message to state
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            # Convert tool messages to LangChain format
            for tm in message.tool_messages:
                lc_msg = self._tau2_to_langchain_message(tm)
                if lc_msg:
                    state.langchain_messages.append(lc_msg)
        else:
            state.messages.append(message)
            lc_msg = self._tau2_to_langchain_message(message)
            if lc_msg:
                state.langchain_messages.append(lc_msg)

        # Reset cost tracker before invocation
        self.cost_tracker.reset()

        # Invoke LangGraph agent with current message history
        # LangGraph's create_react_agent executes tools internally, but it should
        # return AIMessages with tool_calls BEFORE executing them
        # We need to extract those tool calls from the message sequence
        try:
            result = self.agent.invoke({"messages": state.langchain_messages})

            # Extract the updated messages from LangGraph result
            # LangGraph returns a dict with "messages" key containing the full conversation
            updated_langchain_messages = result.get(
                "messages", state.langchain_messages
            )

            # Find new messages (those not in previous state)
            previous_msg_count = len(state.langchain_messages)
            new_messages = updated_langchain_messages[previous_msg_count:]

            # Debug: Log what messages we got back
            if new_messages:
                logger.debug(
                    f"LangGraph returned {len(new_messages)} new messages "
                    f"(total: {len(updated_langchain_messages)}, previous: {previous_msg_count})"
                )
                for i, msg in enumerate(new_messages):
                    msg_type = type(msg).__name__
                    has_tool_calls = hasattr(msg, "tool_calls") and bool(
                        getattr(msg, "tool_calls", None)
                    )
                    has_content = hasattr(msg, "content") and bool(
                        getattr(msg, "content", None)
                    )
                    logger.debug(
                        f"  New message {i}: {msg_type}, "
                        f"has_content={has_content}, has_tool_calls={has_tool_calls}"
                    )
                    if has_tool_calls:
                        tool_calls_list = getattr(msg, "tool_calls", [])
                        logger.debug(
                            f"    Tool calls: {[tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in tool_calls_list]}"
                        )
            else:
                logger.warning(
                    f"No new messages from LangGraph agent! "
                    f"Previous: {previous_msg_count}, Updated: {len(updated_langchain_messages)}"
                )

            # IMPORTANT: LangGraph's create_react_agent executes tools internally
            # The message sequence will be: AIMessage (with tool_calls) -> ToolMessage -> AIMessage (final response)
            # LangGraph may execute multiple tools in sequence during a single invoke() call.
            # We need to extract ALL tool calls from ALL AIMessages with tool_calls in new messages,
            # combine them into a single AssistantMessage, and return that to the orchestrator.
            # Otherwise, return the LAST AIMessage with content from new messages only

            # Collect ALL AIMessages with tool_calls from new messages
            ai_msgs_with_tool_calls = []
            ai_msg_with_content = None

            for msg in new_messages:
                if isinstance(msg, AIMessage):
                    # Collect all AIMessages with tool calls
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        ai_msgs_with_tool_calls.append(msg)
                        logger.debug(
                            f"Found AIMessage with {len(msg.tool_calls)} tool calls: "
                            f"{[tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in msg.tool_calls]}"
                        )
                    # Also track the last AIMessage with content (for when there are no tool calls)
                    if hasattr(msg, "content") and msg.content:
                        ai_msg_with_content = msg

            if ai_msgs_with_tool_calls:
                # Extract ALL tool calls from ALL AIMessages with tool_calls
                # Combine them into a single AssistantMessage
                all_tool_calls = []
                for ai_msg in ai_msgs_with_tool_calls:
                    if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                        all_tool_calls.extend(ai_msg.tool_calls)

                logger.debug(
                    f"Extracting {len(all_tool_calls)} tool calls from {len(ai_msgs_with_tool_calls)} AIMessages"
                )

                # Create a combined AIMessage with all tool calls
                # Use the first AIMessage as a base and replace its tool_calls
                combined_ai_msg = ai_msgs_with_tool_calls[0]
                combined_ai_msg.tool_calls = all_tool_calls

                # Convert to tau2 format
                assistant_message = self._langchain_to_tau2_assistant_message(
                    [combined_ai_msg]
                )
            elif ai_msg_with_content:
                # No tool calls found, use the last AIMessage with content from new messages
                assistant_message = self._langchain_to_tau2_assistant_message(
                    [ai_msg_with_content]  # Just this one message
                )
            else:
                # Fallback: no AIMessages in new messages, look at all messages
                logger.warning(
                    "No AIMessages found in new messages, falling back to all messages"
                )
                assistant_message = self._langchain_to_tau2_assistant_message(
                    updated_langchain_messages
                )

            # Add cost and usage from callback tracker if not already set
            if (
                assistant_message.cost is None
                and self.cost_tracker.last_cost is not None
            ):
                assistant_message.cost = self.cost_tracker.last_cost
            if (
                assistant_message.usage is None
                and self.cost_tracker.last_usage is not None
            ):
                assistant_message.usage = self.cost_tracker.last_usage

            # Log what we extracted
            if assistant_message.is_tool_call():
                logger.debug(
                    f"Extracted {len(assistant_message.tool_calls)} tool calls: "
                    f"{[tc.name for tc in assistant_message.tool_calls]}"
                )
            elif assistant_message.has_text_content():
                logger.debug(
                    f"Extracted text response (no tool calls): {assistant_message.content[:100]}..."
                )

            # Validate the message format
            from tau2.agent.base import validate_message_format

            is_valid, error_msg = validate_message_format(assistant_message)
            if not is_valid:
                logger.warning(f"Invalid message format: {error_msg}")
                # Try to fix: if both content and tool_calls exist, prefer tool_calls
                if (
                    assistant_message.has_text_content()
                    and assistant_message.is_tool_call()
                ):
                    assistant_message.content = None

            # Update state with new messages
            state.messages.append(assistant_message)
            state.langchain_messages = updated_langchain_messages

        except Exception as e:
            logger.error(f"Error in LangGraph agent invocation: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            # Fallback: create a simple text response
            assistant_message = AssistantMessage(
                role="assistant",
                content=f"I apologize, but I encountered an error: {str(e)}",
            )
            state.messages.append(assistant_message)

        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed
        # Recreate LLM with new seed
        self.langchain_llm = self._create_langchain_llm()
        self.agent = create_react_agent(
            self.langchain_llm,
            self.langchain_tools,
            prompt=self.system_prompt_text,
        )

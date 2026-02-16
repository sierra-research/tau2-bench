import asyncio
import functools
import json
import os
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Callable, Coroutine

from loguru import logger
from pctx_client import Pctx
from pctx_client import tool as pctx_tool
from pctx_client.tool_descriptions import PRESCRIPTIVE_DESCRIPTIONS

from tau2.agent.base import ValidAgentInputMessage
from tau2.agent.llm_agent import (
    LLMAgent,
    LLMAgentState,
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.environment import Environment
from tau2.environment.tool import as_tool
from tau2.utils.utils import get_now

# fs mode addendum
PCTX_FS_ADDENDUM = """

Available functions in `{namespace}` namespace:
{function_list}

Use pctx_execute_typescript to call functions as `await {namespace}.functionName({{ args }})`.

**Communication vs Action:**
- When you need information from the user (dates, preferences, clarifications): ASK them, don't try to figure it out yourself
- When you have specific information: use tools to get details or take actions
- Don't do exhaustive searches - be conversational and gather requirements first

Write operations (cancel/book/update/send_certificate) execute database changes.
Multi-step pattern: gather info → verify policy → execute action → communicate result.""".strip()


class LLMPctxAgent(LLMAgent):
    def __init__(
        self,
        env: Environment,
        llm: str | None = None,
        llm_args: dict | None = None,
    ):
        self.internal_messages: list[Message] = []
        self.current_execute_callbacks: list[tuple[ToolCall, ToolMessage]] = []
        self.env = env
        self.pctx_mode = os.environ.get("PCTX_MODE", "code").lower()

        # Create a persistent event loop for this agent instance
        self._loop = asyncio.new_event_loop()

        # convert env tools to pctx tools for code-mode registration
        env_tools = list(env.tools.tools.values()) if env.tools else []
        pctx_env_tools = [
            pctx_tool(self._track_pctx_tool(fn), namespace=env.domain_name)
            for fn in env_tools
        ]
        self.pctx = Pctx(tools=pctx_env_tools)
        self.code_mode_fns = self._get_sync_code_mode_fns()
        tau_tools = [as_tool(t) for t in self.code_mode_fns.values()]

        super().__init__(tau_tools, env.get_policy(), llm, llm_args)

    @property
    def system_prompt(self) -> str:
        """Override to add fs mode context."""
        base_prompt = SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION
        )

        if self.pctx_mode == "fs":
            # Generate function list from env tools
            env_tools = list(self.env.tools.tools.values()) if self.env.tools else []
            function_list = "\n".join([f"- {fn.__name__}" for fn in env_tools])

            fs_context = PCTX_FS_ADDENDUM.format(
                namespace=self.env.domain_name, function_list=function_list
            )
            return base_prompt + "\n\n" + fs_context

        return base_prompt

    def __del__(self):
        """Clean up the persistent event loop when the agent is deleted."""
        if hasattr(self, "_loop") and not self._loop.is_closed():
            self._loop.close()

    def _track_pctx_tool(self, fn: Callable) -> Callable:
        """
        Returns wrapped callable tracing arguments as tool calls
        without altering the original type signature
        """

        @functools.wraps(fn)
        def tracked(**kwargs):
            tool_call_id = str(uuid.uuid4())
            tool_call = ToolCall(id=tool_call_id, name=fn.__name__, arguments=kwargs)

            logger.debug(f"[PCTX] Env Call - {tool_call.name}\n{tool_call.arguments}")

            result = None
            error = False
            try:
                result = fn(**kwargs)
            except Exception as e:
                result = f"Error: {e}"
                error = True

            content = self.env.to_json_str(result)

            logger.debug(
                f"[PCTX] Env Response - {tool_call.name} (error={error})\n{content}"
            )

            tool_msg = ToolMessage(
                id=tool_call_id,
                role="tool",
                content=content,
                error=error,
            )
            self.current_execute_callbacks.append((tool_call, tool_msg))

            return result

        return tracked

    def _handle_pctx_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        error = False
        logger.debug(
            f"[PCTX] Call - {tool_call.name}\n{tool_call.arguments.get('functions', tool_call.arguments.get('code', tool_call.arguments))}"
        )
        try:
            resp = self.code_mode_fns[tool_call.name](**tool_call.arguments)
        except Exception as e:
            resp = f"Error: {e}"
            error = True

        if tool_call.name == "pctx_execute":
            logger.debug(f"[PCTX] Response - {tool_call.name} (error={error})\n{resp}")
        else:
            logger.debug(f"[PCTX] Response - {tool_call.name} (error={error})")

        return ToolMessage(
            id=tool_call.id,
            content=json.dumps(resp),
            requestor=tool_call.requestor,
            role="tool",
            error=error,
        )

    def _run_in_loop(self, coro: Coroutine):
        """Run a coroutine in the agent's persistent event loop from sync code."""
        asyncio.set_event_loop(self._loop)
        try:
            return self._loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)

    def _get_sync_code_mode_fns(self) -> dict[str, Callable]:
        """Get synchronous wrapper functions for pctx tools.

        Mode is controlled by PCTX_MODE environment variable:
        - "code" (default): Traditional discovery workflow (list_functions, get_function_details, execute)
        - "fs": Filesystem exploration workflow (execute_bash, execute_typescript)
        """
        mode = os.environ.get("PCTX_MODE", "code").lower()

        if mode == "fs":
            # Filesystem mode: bash exploration + typescript execution
            def pctx_execute_bash(command: str) -> str:
                return self._run_in_loop(self.pctx.execute_bash(command)).markdown()

            pctx_execute_bash.__doc__ = PRESCRIPTIVE_DESCRIPTIONS["execute_bash"]

            def pctx_execute_typescript(code: str) -> str:
                return self._run_in_loop(self.pctx.execute(code)).markdown()

            pctx_execute_typescript.__doc__ = PRESCRIPTIVE_DESCRIPTIONS[
                "execute_typescript"
            ]

            return {
                "pctx_execute_bash": pctx_execute_bash,
                "pctx_execute_typescript": pctx_execute_typescript,
            }
        else:
            # Code mode (default): Traditional discovery workflow
            def pctx_list_functions() -> str:
                return self._run_in_loop(self.pctx.list_functions()).code

            pctx_list_functions.__doc__ = PRESCRIPTIVE_DESCRIPTIONS["list_functions"]

            def pctx_get_function_details(functions: list[str]) -> str:
                return self._run_in_loop(self.pctx.get_function_details(functions)).code

            pctx_get_function_details.__doc__ = PRESCRIPTIVE_DESCRIPTIONS[
                "get_function_details"
            ]

            def pctx_execute(code: str) -> str:
                return self._run_in_loop(self.pctx.execute(code)).markdown()

            pctx_execute.__doc__ = PRESCRIPTIVE_DESCRIPTIONS["execute"]

            return {
                "pctx_list_functions": pctx_list_functions,
                "pctx_get_function_details": pctx_get_function_details,
                "pctx_execute": pctx_execute,
            }

    def connect(self):
        try:
            self._run_in_loop(self.pctx.connect())
            logger.debug(
                f"[PCTX] - connected to server: session_id={self.pctx._session_id}"
            )
        except Exception as e:
            logger.error(f"[PCTX] - connect failed: {e}")
            raise  # Re-raise to fail the task if we can't connect

    def disconnect(self):
        try:
            self._run_in_loop(self.pctx.disconnect())
            logger.debug(f"[PCTX] - disconnected from server")
        except Exception as e:
            logger.warning(f"[PCTX] - disconnect failed (ignoring): {e}")
            # Ignore disconnect errors - the session work is already done

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        msg, state = super().generate_next_message(message=message, state=state)

        # TODO: track these internal messages
        iteration = 0
        while msg.is_tool_call():
            iteration += 1
            logger.debug(
                f"[PCTX] Tool call iteration {iteration}\n\ttool call(s): {len(msg.tool_calls)}\n\tmessage content: {msg.content}"
            )

            expanded_assistant_msg = deepcopy(msg)
            expanded_assistant_msg.tool_calls = []

            tool_msgs = []
            execute_tool_msgs = []
            for pctx_tool_call in msg.tool_calls:
                self.current_execute_callbacks.clear()

                before_handle = get_now()
                pctx_tool_msg = self._handle_pctx_tool_call(pctx_tool_call)
                pctx_tool_msg.timestamp = before_handle
                tool_msgs.append(pctx_tool_msg)

                expanded_assistant_msg.tool_calls.append(pctx_tool_call)
                expanded_assistant_msg.tool_calls.extend(
                    map(lambda e: e[0], self.current_execute_callbacks)
                )
                execute_tool_msgs.extend(
                    map(lambda e: e[1], self.current_execute_callbacks)
                )

            self.internal_messages.append(expanded_assistant_msg)
            self.internal_messages.extend(tool_msgs)
            self.internal_messages.extend(execute_tool_msgs)

            # Packaging multiple tool messages into a MultiToolMessage
            if len(tool_msgs) > 1:
                logger.debug(
                    f"[PCTX] Packaging {len(tool_msgs)} tool messages into MultiToolMessage"
                )
                next_msg = MultiToolMessage(
                    role="tool",
                    tool_messages=tool_msgs,
                )
            else:
                next_msg = tool_msgs[0]

            # call model again
            logger.debug(f"[PCTX] Calling model again with tool results")
            msg, state = super().generate_next_message(message=next_msg, state=state)

        logger.debug(
            f"[PCTX] Returning final message after {iteration} tool call iteration(s)"
        )
        # final msg will automatically be added to the trajectory so we should avoid
        # adding to self.internal_messages (double counting)
        return msg, state

    def get_internal_messages(self) -> list[Message]:
        return self.internal_messages

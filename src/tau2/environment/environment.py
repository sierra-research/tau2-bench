import json
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall, InitializationData
from tau2.environment.critic import (
    GATE_BLOCKED_MARKER,
    REASONING_ARG,
    review_write_action,
    verdict_cache_key,
)
from tau2.environment.db import DB
from tau2.environment.tool import Tool
from tau2.environment.toolkit import ToolKitBase, ToolSignature, get_tool_signatures

# Tool names that gate their own inner dispatch generically (based on the
# mutates_state of whatever underlying tool is actually being invoked) and so
# are excluded from the outer, environment-level gate below -- gating them a
# second time at this layer would require a top-level `_reasoning` argument
# on the *wrapper* call itself, which doesn't make sense since the wrapper's
# real arguments are JSON-encoded inside a string argument.
_SELF_GATING_TOOL_NAMES = {"call_discoverable_agent_tool"}


class EnvironmentInfo(BaseModel):
    """
    Environment information.
    """

    domain_name: str = Field(description="The name of the domain.")
    policy: str = Field(description="The policy of the agent.")
    tool_defs: Optional[dict[str, ToolSignature]] = Field(
        description="The tool definitions of the environment.", default=None
    )


class Environment:
    """
    Environment
    """

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: Optional[ToolKitBase] = None,
        user_tools: Optional[ToolKitBase] = None,
        solo_mode: bool = False,
        enable_write_critic: bool = False,
    ):
        """
        Environment
        Args:
            domain_name: The name of the domain.
            policy: The policy of the domain.
            tools: The tools available to the assistant in the domain.
            user_tools: The tools available to the user in the domain.
            solo_mode: The agent will have access to both user and assistant tools.
            enable_write_critic: If True, every assistant-initiated mutating
                (write) tool call -- identified generically via the tool's
                ``mutates_state`` metadata, not by name -- is required to
                include a ``_reasoning`` argument and is reviewed by a
                second, independent LLM critic (see ``tau2.environment.critic``)
                before it is actually executed. Off by default so this
                framework-level behavior doesn't silently change other
                domains; opt in per-domain in that domain's ``get_environment``.
        """
        self.domain_name = domain_name
        self.policy = policy
        self.tools = tools
        self.user_tools = user_tools
        self.solo_mode = solo_mode
        self.enable_write_critic = enable_write_critic
        # Cache of critic verdicts keyed by verdict_cache_key(tool_name, args),
        # scoped to THIS Environment instance only. This is a same-instance
        # optimization (e.g. the agent issuing the exact same call twice
        # within one live conversation) -- it is NOT what makes replay
        # deterministic. Replay/evaluation always constructs a brand-new
        # Environment per attempt (see evaluator_env.py), so this cache is
        # empty on every replay regardless. The actual replay-determinism
        # fix is GATE_BLOCKED_MARKER: every gate rejection is tagged with it,
        # and Environment.set_state()'s replay loop recognizes that marker in
        # the *recorded* response and skips re-executing (and re-critiquing)
        # that call entirely, rather than relying on any cache surviving
        # across instances.
        self._critic_verdict_cache: dict[str, Optional[str]] = {}
        # Per-tool-name count of write-critic blocks (missing-reasoning OR a
        # critic rejection) issued so far this conversation. Capped at 1: the
        # agent gets one corrective error per tool name to react to, but a
        # second attempt on that same tool name is let through unconditionally
        # rather than gated again. This bounds the worst case where a stubborn
        # or wrong verdict keeps rejecting corrected retries -- burning
        # through the simulation's error budget and causing a premature
        # TOO_MANY_ERRORS termination -- to at most one rejection per tool.
        self._critic_block_counts: dict[str, int] = {}
        if self.solo_mode:
            self.validate_solo_mode()
        self.sync_tools()

    def get_domain_name(self) -> str:
        """
        Get the name of the domain.
        """
        return self.domain_name

    def get_policy(self) -> str:
        """
        Get the policy of the domain.
        """
        return self.policy

    def get_tools(self) -> list[Tool]:
        """
        Get the tools of the domain.
        """
        if self.tools is None:
            raise ValueError("Tools not available")
        return list(self.tools.get_tools().values())

    def get_user_tools(self, include: Optional[list[str]] = None) -> list[Tool]:
        """
        Get the user tools of the domain, optionally filtered by name.

        Args:
            include: If provided, only return tools whose names are in this list.
                If None, return all user tools (no filtering).

        Returns:
            A list of Tool objects available to the user.
        """
        if self.user_tools is None:
            raise ValueError("User tools not available")
        return list(self.user_tools.get_tools(include=include).values())

    def get_tools_description(
        self, env_type: Literal["user", "assistant"]
    ) -> Optional[str]:
        """
        Return a description of the user tools.
        """
        if env_type == "user":
            tool_kit = self.user_tools
        elif env_type == "assistant":
            tool_kit = self.tools
        else:
            raise ValueError(f"Invalid environment type: {env_type}")
        if tool_kit is None:
            return None
        tools = sorted(tool_kit.get_tools().values(), key=lambda x: x.name)
        return "\n\n".join(
            [f"{i + 1}. {t.name}\n{t.short_desc}" for i, t in enumerate(tools)]
        )

    def _has_tool(self, tool_name: str) -> bool:
        """Check if a tool exists in the environment.

        Checks toolkit tools and user tools.
        """
        if self.tools is not None and self.tools.has_tool(tool_name):
            return True
        if self.user_tools is not None and self.user_tools.has_tool(tool_name):
            return True
        return False

    def _is_mutating_tool(self, tool_name: str) -> bool:
        """Check if a tool mutates environment state.

        Looks up ``mutates_state`` on the underlying function via the toolkit.
        Falls back to ``True`` (assume mutation) if the tool or attribute
        cannot be found.
        """
        for toolkit in (self.tools, self.user_tools):
            if toolkit is not None and toolkit.has_tool(tool_name):
                return toolkit.tool_mutates_state(tool_name)
        return True  # safe fallback: assume mutation

    def _check_write_critic_gate(
        self,
        tool_name: str,
        requestor: str,
        arguments: dict,
        conversation_history: Optional[list],
    ) -> Optional[str]:
        """Generic write-action critic gate.

        For any assistant-initiated tool call whose underlying tool mutates
        state (identified generically via ``mutates_state`` metadata, never
        by tool name), require a ``_reasoning`` argument and run it past a
        second, independent LLM critic (``tau2.environment.critic``) before
        the tool actually executes.

        Mutates ``arguments`` in place by popping the harness-level
        ``_reasoning`` field so it never reaches the underlying tool's real
        signature.

        Returns an error string (blocking the write) if the gate should
        reject the call, else None (safe to proceed).
        """
        if not self.enable_write_critic:
            return None
        if requestor != "assistant":
            return None
        if tool_name in _SELF_GATING_TOOL_NAMES:
            return None
        if not self._has_tool(tool_name) or not self._is_mutating_tool(tool_name):
            return None

        agent_reasoning = arguments.pop(REASONING_ARG, None)

        # Cap: at most one block per tool name per conversation (see
        # _critic_block_counts docstring in __init__). Once this tool has
        # already been blocked once, let any further attempt through
        # unconditionally rather than risk the gate itself exhausting the
        # simulation's error budget.
        if self._critic_block_counts.get(tool_name, 0) >= 1:
            return None

        if not agent_reasoning or not str(agent_reasoning).strip():
            self._critic_block_counts[tool_name] = (
                self._critic_block_counts.get(tool_name, 0) + 1
            )
            return (
                f"Error: {GATE_BLOCKED_MARKER} '{tool_name}' is a state-changing action "
                f"and requires a '{REASONING_ARG}' argument explaining, in 1-2 sentences, "
                "why this specific action and these specific argument values are correct "
                "given the policy/evidence gathered so far in this conversation. Please "
                f"retry including '{REASONING_ARG}' in the arguments."
            )

        cache_key = verdict_cache_key(tool_name, arguments)
        if cache_key in self._critic_verdict_cache:
            return self._critic_verdict_cache[cache_key]

        verdict = review_write_action(
            tool_name=tool_name,
            args_dict=arguments,
            agent_reasoning=str(agent_reasoning),
            conversation_history=conversation_history,
        )
        self._critic_verdict_cache[cache_key] = verdict
        if verdict is not None:
            self._critic_block_counts[tool_name] = (
                self._critic_block_counts.get(tool_name, 0) + 1
            )
        return verdict

    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a tool available to the assistant of the domain.
        """
        if self.tools is None:
            raise ValueError("Tools not available")
        return self.tools.use_tool(tool_name=tool_name, **kwargs)

    def use_user_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a tool available to the user of the domain.
        """
        if self.user_tools is None:
            raise ValueError("User tools not available")
        return self.user_tools.use_tool(tool_name=tool_name, **kwargs)

    def make_tool_call(
        self,
        tool_name: str,
        requestor: Literal["user", "assistant"] = "assistant",
        **kwargs,
    ) -> Any:
        """
        Make a tool call based on the requestor.
        Args:
            tool_name: The name of the tool to call.
            requestor: The requestor of the tool call.
            kwargs: The arguments to pass to the tool.
        Returns:
            The response of the tool call.

        Note: This does not call sync_tools.
        """
        if requestor == "user":
            if self.solo_mode:
                raise ValueError("User tool calls are not allowed in solo mode")
            return self.use_user_tool(tool_name=tool_name, **kwargs)
        elif requestor == "assistant":
            if self.solo_mode and self.user_tools is not None:
                if self.user_tools.has_tool(tool_name):
                    return self.use_user_tool(tool_name=tool_name, **kwargs)
            return self.use_tool(tool_name=tool_name, **kwargs)
        else:
            raise ValueError(f"Invalid requestor: {requestor}")

    def sync_tools(self):
        """
        Sync the user and assistant tools.
        Subclass should override this method if tools need to be synced.
        """
        pass

    def run_env_function_call(self, env_function_call: EnvFunctionCall) -> Any:
        """
        Runs any function available on agent environment or user environment.
        """
        env_type = env_function_call.env_type
        func_name = env_function_call.func_name
        if env_type == "user":
            tool_kit = self.user_tools
        elif env_type == "assistant":
            tool_kit = self.tools
        else:
            raise ValueError(f"Invalid environment type: {env_type}")
        func = getattr(tool_kit, func_name)
        if func is None:
            raise ValueError(f"Function {func_name} not found in {env_type} tools")
        res = func(**env_function_call.arguments)
        self.sync_tools()
        return res

    def run_env_assertion(
        self,
        assertion: EnvAssertion,
        raise_assertion_error: bool = True,
    ) -> bool:
        """
        Runs any assertion function on agent tools or user tools.
        """
        if not isinstance(assertion, EnvAssertion):
            raise ValueError(f"Assertion must be an EnvAssertion. Got {assertion}")
        res = self.run_env_function_call(assertion)
        if not isinstance(res, bool):
            raise ValueError(
                f"Function {assertion.func_name} returned {type(res)} instead of bool"
            )
        assert_pass = res == assertion.assert_value
        if raise_assertion_error:
            assert assert_pass, assertion.message or f"Assertion failed: {assertion}"
        return assert_pass

    def run_env_function_calls(self, env_function_calls: list[EnvFunctionCall]) -> None:
        """
        Run a list of environment function calls. If the function call is an assertion,
        an assertion check will be performed.
        """
        for env_function_call in env_function_calls:
            if isinstance(env_function_call, EnvAssertion):
                self.run_env_assertion(env_function_call, raise_assertion_error=True)
            else:
                self.run_env_function_call(env_function_call)

    def get_info(self, include_tool_info: bool = False) -> EnvironmentInfo:
        """
        Get environment information.
        """
        return EnvironmentInfo(
            domain_name=self.domain_name,
            policy=self.policy,
            tool_defs=(
                get_tool_signatures(self.tools)
                if self.tools is not None and include_tool_info
                else None
            ),
            user_tool_defs=(
                get_tool_signatures(self.user_tools)
                if self.user_tools is not None and include_tool_info
                else None
            ),
        )

    def check_db(self, reference: DB) -> bool:
        """
        Compare the agent database with the reference
        """
        return self.get_db_hash() == reference.get_hash()

    def check_user_db(self, reference: DB) -> bool:
        """
        Compare the user database with the reference
        """
        return self.get_user_db_hash() == reference.get_hash()

    def get_db_hash(self) -> Optional[str]:
        """
        Get a hash of the agent database
        Returns None if the database is not available
        """
        if self.tools is None:
            return None
        return self.tools.get_db_hash()

    def get_user_db_hash(self) -> Optional[str]:
        """
        Get a hash of the user database
        Returns None if the database is not available
        """
        if self.user_tools is None:
            return None
        return self.user_tools.get_db_hash()

    def set_state(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
        strict: bool = True,
    ):
        """
        Set the state of the environment given initialization data and a list of messages.

        Args:
            strict: When True (default), raise if a replayed mutating tool call
                returns different content than the recorded ToolMessage. When
                False, log a warning instead and continue the replay. Lenient
                mode is intended for re-grading historical trajectories whose
                recorded tool outputs contain cosmetic drift against current
                tool code (e.g. numeric argument echoes rendered as `25` by the
                code that produced them but `25.0` after numeric-argument
                normalization); the state mutation is applied identically
                either way.
        """
        if self.solo_mode:
            assert all(
                [not isinstance(message, UserMessage) for message in message_history]
            ), "User messages are not allowed in solo mode"

        def get_actions_from_messages(
            messages: list[Message],
        ) -> list[tuple[ToolCall, ToolMessage]]:
            """
            Get the actions from the messages.
            """
            messages = deepcopy(messages)[::-1]
            actions = []
            while messages:
                message = messages.pop()
                if isinstance(message, ToolMessage):
                    raise ValueError(
                        "Tool message not expected. Tool messages should always follow a tool call."
                    )
                if (
                    isinstance(message, (AssistantMessage, UserMessage))
                    and message.is_tool_call()
                ):
                    tool_calls = message.tool_calls
                    for tc in tool_calls:
                        if len(messages) == 0:
                            raise ValueError("Tool message expected. Got None.")
                        tm = messages.pop()
                        if not isinstance(tm, ToolMessage):
                            raise ValueError(f"Tool message expected. Got {type(tm)}")
                        if tc.id != tm.id:
                            raise ValueError(
                                f"Tool call id mismatch. Got {tc.id} and {tm.id}"
                            )
                        actions.append((tc, tm))

            return actions

        if initialization_data is not None:
            if initialization_data.agent_data is not None:
                self.tools.update_db(initialization_data.agent_data)
                # Sync user_tools.db to point to the same db instance as tools.db
                # This is necessary because update_db creates a new db instance
                if self.user_tools is not None and self.user_tools.db is not None:
                    self.user_tools.db = self.tools.db
            if initialization_data.user_data is not None:
                self.user_tools.update_db(initialization_data.user_data)
                # Sync tools.db to point to the same db instance as user_tools.db
                if self.tools is not None and self.tools.db is not None:
                    self.tools.db = self.user_tools.db

        if initialization_actions is not None:
            for action in initialization_actions:
                self.run_env_function_call(action)

        action_responses = get_actions_from_messages(message_history)
        for tool_call, expected_response in action_responses:
            if not self._has_tool(tool_call.name):
                # Hallucinated tool name. The live env returned a
                # ToolMessage(error=True) for this call and made no state
                # change, so replay it as a no-op. The agent's subsequent
                # recovery (if any) will still be replayed and determine
                # the final state. Repeated hallucination is bounded
                # upstream by the orchestrator's max_errors guard, which
                # ends the live sim with TerminationReason.TOO_MANY_ERRORS
                # before evaluation runs.
                logger.debug(
                    f"Skipping unknown tool '{tool_call.name}' during replay "
                    "(no-op, matching live env behavior on hallucinated tools)."
                )
                continue
            # Non-mutating tools (reads, thinks, etc.) don't change state --
            # skip them to avoid re-execution and non-deterministic output
            # comparison issues.
            if not self._is_mutating_tool(tool_call.name):
                continue
            # A recorded response that was blocked by the generic write-critic
            # gate (missing reasoning, or a critic rejection) never mutated
            # state live, and the exact wording of a critic rejection is a
            # non-deterministic LLM output that must not be replayed. Detect
            # this purely from the recorded content and skip re-execution
            # entirely -- no mutation to reproduce, nothing to compare.
            if (
                isinstance(expected_response.content, str)
                and GATE_BLOCKED_MARKER in expected_response.content
            ):
                continue
            # For every other (successful) mutating call, replay it with the
            # critic gate bypassed: the gate already had its say live, and
            # the actual tool output for a successful call is deterministic
            # (no LLM wording embedded in it), so bypassing here just
            # reproduces the same state mutation without re-invoking a live,
            # non-deterministic LLM check against a fresh/empty conversation
            # context.
            response = self.get_response(tool_call, skip_write_critic=True)
            try:
                content = json.loads(response.content)
            except json.JSONDecodeError:
                content = response.content
            try:
                expected_content = json.loads(expected_response.content)
            except json.JSONDecodeError:
                expected_content = expected_response.content
            if content != expected_content:
                if strict:
                    raise ValueError(
                        f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\nExpected:\n{expected_response}"
                    )
                logger.warning(
                    f"Replayed tool call '{tool_call.name}' returned different "
                    f"content than the recorded ToolMessage; continuing because "
                    f"strict=False. Recorded output may predate current tool "
                    f"code.\nTool call:\n{tool_call}"
                )
        self.sync_tools()

    @classmethod
    def to_json_str(cls, resp: Any) -> str:
        """
        Convert a response to a JSON string.
        """

        def _process(resp: Any) -> str:
            if isinstance(resp, BaseModel):
                return resp.model_dump()
            elif isinstance(resp, str):
                return resp
            elif resp is None:
                return resp
            elif isinstance(resp, (int, float, bool)):
                return str(resp)
            elif isinstance(resp, list):
                return [_process(item) for item in resp]
            elif isinstance(resp, tuple):
                return tuple(_process(item) for item in resp)
            elif isinstance(resp, dict):
                return {k: _process(v) for k, v in resp.items()}
            elif isinstance(resp, (datetime, date)):
                # TODO: this did not fix the error: Object of type date is not JSON serializable
                return resp.isoformat()
            else:
                raise ValueError(f"Unsupported type: {type(resp)}")

        if not isinstance(resp, str):
            return json.dumps(_process(resp), default=str)  # FIXME: add default=str
        return resp

    def set_solo_mode(self, solo_mode: bool):
        """
        Set the solo mode of the environment.
        """
        self.solo_mode = solo_mode
        if solo_mode:
            self.validate_solo_mode()

    def validate_solo_mode(self) -> None:
        """
        Validate the tool call in solo mode.
        """
        assistant_tool_names = set(self.tools.get_tools().keys())
        user_tool_names = (
            set(self.user_tools.get_tools().keys())
            if self.user_tools is not None
            else set()
        )
        overlap = assistant_tool_names & user_tool_names
        if len(overlap) > 0:
            raise ValueError(f"Tool names overlap: {overlap}")

    def get_response(
        self,
        message: ToolCall,
        conversation_history: Optional[list] = None,
        skip_write_critic: bool = False,
    ) -> ToolMessage:
        """
        Get the response of the domain. This also calls sync_tools.
        Args:
            message: The message to get the response for.
            conversation_history: Recent conversation messages so far this
                task (assistant/tool messages). Threaded generically into the
                toolkit (via ``set_conversation_context``) before dispatch so
                any tool -- including the write-action critic gate below --
                can see what policy/evidence has already been surfaced.
            skip_write_critic: If True, bypass the write-action critic gate
                entirely and just re-apply the tool call's state mutation
                directly. Used during replay (``set_state`` / evaluation):
                replay re-executes a tool call that was ALREADY approved and
                executed once during the live run, against a fresh Environment
                instance with no conversation history available, purely to
                reproduce its state mutation deterministically -- re-running a
                live, non-deterministic LLM critic there (possibly with a
                different or empty context than the original call saw) would
                risk a different verdict/wording than the original response,
                which breaks the replay-consistency check this method's
                caller performs. The critic already had its say the first
                time the call was actually made live.
        Returns:
            The response of the tool call.
        """
        error = False
        arguments = dict(message.arguments)
        if self.tools is not None:
            self.tools.set_conversation_context(conversation_history)
            self.tools.set_skip_write_critic(skip_write_critic)
        if self.user_tools is not None:
            self.user_tools.set_conversation_context(conversation_history)
            self.user_tools.set_skip_write_critic(skip_write_critic)

        if not skip_write_critic:
            gate_error = self._check_write_critic_gate(
                message.name, message.requestor, arguments, conversation_history
            )
            if gate_error is not None:
                return ToolMessage(
                    id=message.id,
                    content=gate_error,
                    requestor=message.requestor,
                    role="tool",
                    error=True,
                )
        else:
            # Still strip the harness-level reasoning field so it doesn't
            # leak through to the underlying tool's real signature.
            arguments.pop(REASONING_ARG, None)

        try:
            resp = self.make_tool_call(
                message.name, requestor=message.requestor, **arguments
            )
            self.sync_tools()
        except Exception as e:
            resp = f"Error: {e}"
            error = True
        logger.debug(f"Response: {resp}")
        resp = self.to_json_str(resp)
        return ToolMessage(
            id=message.id,
            content=resp,
            requestor=message.requestor,
            role="tool",
            error=error,
        )

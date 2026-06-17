#!/usr/bin/env python3
"""
Run the Return-and-Exchange agent (Claude + skills + supervisor) against
τ-bench retail evaluation.

Your agent repo uses its own mock tools (lookup_order, etc.). τ-bench scores
against the retail domain tools (return_delivered_order_items, etc.) and DB.
This adapter keeps your prompts/skills/supervisor but drives τ-bench's retail
tool API so scores are meaningful.

Setup (PowerShell, from tau2-bench root):

    uv pip install anthropic PyYAML
    # .env in tau2-bench: ANTHROPIC_API_KEY=... and OPENAI_API_KEY=... (user sim)

    uv run python examples/agents/return_exchange_agent_tau2.py `
        --return-agent-path "C:\dev\Return-and-Exchange-agent-main" `
        --task-ids 0 1 2 `
        --user-llm openai/gpt-4.1-mini

Full retail base split:

    uv run python examples/agents/return_exchange_agent_tau2.py `
        --return-agent-path "C:\dev\Return-and-Exchange-agent-main"
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
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
from tau2.data_model.simulation import TextRunConfig
from tau2.environment.toolkit import Tool
from tau2.registry import registry
from tau2.runner import run_domain

DEFAULT_RETURN_AGENT_PATH = os.environ.get(
    "RETURN_AGENT_PATH",
    "C:\\dev\\Return-and-Exchange-agent-main",
)

AGENT_NAME = "return_exchange_agent"


def _add_return_agent_to_path(return_agent_path: Path) -> None:
    path_str = str(return_agent_path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _clean_json_schema(schema: dict) -> dict:
    """Strip Pydantic extras so Anthropic accepts the tool input_schema."""
    if not schema:
        return {"type": "object", "properties": {}}

    cleaned = copy.deepcopy(schema)
    for key in ("title", "description", "$defs", "definitions"):
        cleaned.pop(key, None)

    if "properties" in cleaned:
        for prop in cleaned["properties"].values():
            if isinstance(prop, dict):
                prop.pop("title", None)

    if cleaned.get("type") is None:
        cleaned["type"] = "object"
    return cleaned


def tau2_tools_to_anthropic(tools: list[Tool]) -> list[dict[str, Any]]:
    anthropic_tools = []
    for tool in tools:
        fn = tool.openai_schema["function"]
        anthropic_tools.append(
            {
                "name": fn["name"],
                "description": fn["description"],
                "input_schema": _clean_json_schema(fn["parameters"]),
            }
        )
    return anthropic_tools


def _anthropic_messages_from_history(
    messages: list[APICompatibleMessage],
) -> list[dict[str, Any]]:
    """Convert τ-bench conversation history to Anthropic API messages."""
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, UserMessage):
            if msg.content:
                anthropic_messages.append({"role": "user", "content": msg.content})
            continue

        if isinstance(msg, AssistantMessage):
            if msg.is_tool_call():
                blocks: list[dict[str, Any]] = []
                for tc in msg.tool_calls or []:
                    tool_id = tc.id or f"toolu_{uuid.uuid4().hex[:12]}"
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": blocks})
            elif msg.content:
                anthropic_messages.append({"role": "assistant", "content": msg.content})
            continue

        if isinstance(msg, ToolMessage):
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.id,
                            "content": msg.content or "",
                        }
                    ],
                }
            )
            continue

        if isinstance(msg, MultiToolMessage):
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tm.id,
                            "content": tm.content or "",
                        }
                        for tm in msg.tool_messages
                    ],
                }
            )

    return anthropic_messages


def _customer_messages_for_supervisor(
    messages: list[APICompatibleMessage],
) -> list[dict[str, str]]:
    """Plain {role, content} list for the Return-and-Exchange supervisor."""
    convo: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, UserMessage) and msg.content:
            convo.append({"role": "user", "content": msg.content})
        elif (
            isinstance(msg, AssistantMessage) and msg.content and not msg.is_tool_call()
        ):
            convo.append({"role": "assistant", "content": msg.content})
    return convo


def _trace_from_history(messages: list[APICompatibleMessage]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}

    for msg in messages:
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                entry = {"tool": tc.name, "input": tc.arguments}
                trace.append(entry)
                if tc.id:
                    pending[tc.id] = entry
        if isinstance(msg, ToolMessage) and msg.id in pending:
            try:
                pending[msg.id]["result"] = json.loads(msg.content or "{}")
            except json.JSONDecodeError:
                pending[msg.id]["result"] = msg.content
        if isinstance(msg, MultiToolMessage):
            for tm in msg.tool_messages:
                if tm.id in pending:
                    try:
                        pending[tm.id]["result"] = json.loads(tm.content or "{}")
                    except json.JSONDecodeError:
                        pending[tm.id]["result"] = tm.content

    return trace


def _build_system_prompt(domain_policy: str, return_agent_path: Path) -> str:
    skill_block = ""
    try:
        from skills import assemble_skill_prompt

        skill_block = assemble_skill_prompt()
    except ImportError:
        skill_block = (
            "(Skills module not found — install return-agent path or set "
            "RETURN_AGENT_PATH.)"
        )

    return f"""You are a retail customer service agent handling returns and exchanges.

Follow the domain policy below exactly. Use only the provided tools. Do not invent
data or procedures. Authenticate the customer before sharing order details. Before
any action that updates the database (cancel, modify, return, exchange), list the
action details and obtain explicit user confirmation (yes) to proceed.

Make at most one tool call per turn. Do not send user text and a tool call in the
same message.

## τ-bench retail policy
{domain_policy}

## Return-and-exchange skills (from your agent)
{skill_block}

When the request is resolved or must escalate, give a clear final message to the
customer."""


class ReturnExchangeTau2AgentState:
    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
    ):
        self.system_messages = system_messages
        self.messages = messages


class ReturnExchangeTau2Agent(HalfDuplexAgent[ReturnExchangeTau2AgentState]):
    """Claude agent using Return-and-Exchange prompts against τ-bench retail tools."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        return_agent_path: str,
        use_supervisor: bool = True,
        model: str = "claude-sonnet-4-6",
        **kwargs,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.return_agent_path = Path(return_agent_path)
        self.use_supervisor = use_supervisor
        self.model = model

        _add_return_agent_to_path(self.return_agent_path)

        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or api_key.startswith("<"):
            raise ValueError(
                "Set ANTHROPIC_API_KEY in tau2-bench .env (Return-and-Exchange "
                "agent uses Claude via Anthropic API)."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._anthropic_tools = tau2_tools_to_anthropic(tools)

        if self.use_supervisor:
            from supervisor import supervised_reply

            self._supervised_reply = supervised_reply
        else:
            self._supervised_reply = None

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ReturnExchangeTau2AgentState:
        system = _build_system_prompt(self.domain_policy, self.return_agent_path)
        history: list[APICompatibleMessage] = []
        if message_history:
            history = [
                m for m in message_history if isinstance(m, APICompatibleMessage)
            ]
        return ReturnExchangeTau2AgentState(
            system_messages=[SystemMessage(role="system", content=system)],
            messages=history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: ReturnExchangeTau2AgentState
    ) -> tuple[AssistantMessage, ReturnExchangeTau2AgentState]:
        state.messages.append(message)

        system_text = state.system_messages[0].content or ""
        anthropic_messages = _anthropic_messages_from_history(state.messages)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_text,
            tools=self._anthropic_tools,
            messages=anthropic_messages,
        )

        if response.stop_reason == "tool_use":
            tool_calls: list[ToolCall] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input),
                    )
                )
                break  # retail policy: one tool per turn

            assistant_message = AssistantMessage.text(
                content=None, tool_calls=tool_calls
            )
            state.messages.append(assistant_message)
            return assistant_message, state

        draft = "".join(b.text for b in response.content if b.type == "text")

        if self._supervised_reply is not None:
            customer_msgs = _customer_messages_for_supervisor(state.messages)
            trace = _trace_from_history(state.messages)
            final_text, _verdict = self._supervised_reply(
                customer_msgs, draft, trace, client=self._client
            )
        else:
            final_text = draft

        assistant_message = AssistantMessage.text(content=final_text)
        state.messages.append(assistant_message)
        return assistant_message, state


def create_return_exchange_agent(tools, domain_policy, **kwargs):
    return_agent_path = kwargs.get("return_agent_path") or DEFAULT_RETURN_AGENT_PATH
    use_supervisor = kwargs.get("use_supervisor", True)
    model = kwargs.get("return_agent_model", "claude-sonnet-4-6")
    return ReturnExchangeTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        return_agent_path=return_agent_path,
        use_supervisor=use_supervisor,
        model=model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Return-and-Exchange agent on τ-bench retail domain."
    )
    parser.add_argument(
        "--return-agent-path",
        default=DEFAULT_RETURN_AGENT_PATH,
        help="Path to Return-and-Exchange-agent repo (default: RETURN_AGENT_PATH env)",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        default=None,
        help="Retail task IDs (default: full base split)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of evaluation trials",
    )
    parser.add_argument(
        "--user-llm",
        default="openai/gpt-4.1-mini",
        help="LiteLLM model for τ-bench user simulator",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Parallel simulations (keep 1 while debugging)",
    )
    parser.add_argument(
        "--save-to",
        default="return-exchange-agent-retail",
        help="Output folder under data/simulations/",
    )
    parser.add_argument(
        "--no-supervisor",
        action="store_true",
        help="Disable Return-and-Exchange supervisor layer",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model for the agent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    return_path = Path(args.return_agent_path)
    if not return_path.is_dir():
        raise SystemExit(
            f"Return-and-Exchange agent not found at {return_path}. "
            "Clone https://github.com/annagibaeva/Return-and-Exchange-agent "
            "or pass --return-agent-path."
        )

    def factory_with_path(tools, domain_policy, **kwargs):
        kwargs["return_agent_path"] = str(return_path)
        kwargs["use_supervisor"] = not args.no_supervisor
        kwargs["return_agent_model"] = args.model
        return create_return_exchange_agent(tools, domain_policy, **kwargs)

    registry.register_agent_factory(factory_with_path, AGENT_NAME)

    config = TextRunConfig(
        domain="retail",
        agent=AGENT_NAME,
        llm_agent="anthropic/claude-sonnet-4-6",
        llm_user=args.user_llm,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        task_ids=args.task_ids,
        task_split_name="base" if args.task_ids is None else None,
        save_to=args.save_to,
    )

    print(f"Return-and-Exchange agent path: {return_path}")
    print(f"Tasks: {args.task_ids or 'full base split'}")
    print(f"Results → data/simulations/{args.save_to}/")
    print()

    run_domain(config)
    print()
    print("Done. View results:")
    print("  uv run tau2 view")


if __name__ == "__main__":
    main()

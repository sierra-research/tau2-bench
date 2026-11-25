#!/usr/bin/env python3
"""Stateful bridge to the pinned official tau2-bench environment and scorer."""

from __future__ import annotations

import argparse
import base64
import queue
import threading
import uuid
import numpy as np
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


# Transport only: the official embedder model, texts, vector dtype and retrieval
# algorithms stay unchanged. Credentials remain in the Vela Process, never here.
from tau2.knowledge.embedders.openai_embedder import OpenAIEmbedder

EMBEDDING_QUEUE = queue.Queue()
EMBEDDING_PENDING = {}
EMBEDDING_LOCK = threading.Lock()


def _embedding_init(self, model="text-embedding-ada-002", api_key=None):
    if model != "text-embedding-3-large":
        raise ValueError("AllTools requires text-embedding-3-large")
    self.model = model


def _embedding_embed(self, texts):
    if not texts:
        raise ValueError("No text to embed.")
    request_id = uuid.uuid4().hex
    pending = {"event": threading.Event()}
    with EMBEDDING_LOCK:
        EMBEDDING_PENDING[request_id] = pending
    EMBEDDING_QUEUE.put({"id": request_id, "model": self.model, "input": texts})
    try:
        if not pending["event"].wait(600):
            raise RuntimeError("embedding_transport_timeout")
        if pending.get("error"):
            raise RuntimeError("embedding_provider_failed:" + pending["error"])
        raw = base64.b64decode(pending["vectors"])
        return np.frombuffer(raw, dtype="<f8").reshape(len(texts), 3072).copy()
    finally:
        with EMBEDDING_LOCK:
            EMBEDDING_PENDING.pop(request_id, None)


def embedding_poll(payload):
    requests = []
    while not EMBEDDING_QUEUE.empty():
        try:
            requests.append(EMBEDDING_QUEUE.get_nowait())
        except queue.Empty:
            break
    return {"requests": requests}


def embedding_result(payload):
    with EMBEDDING_LOCK:
        pending = EMBEDDING_PENDING.get(payload["id"])
        if pending is None:
            raise ValueError("unknown_embedding_request")
        pending.update({k: payload[k] for k in ("vectors", "error") if k in payload})
        pending["event"].set()
    return {"ok": True}


OpenAIEmbedder.__init__ = _embedding_init
OpenAIEmbedder.embed = _embedding_embed


TASKS = {task.id: task for task in get_tasks()}
STATE: dict[str, Any] = {}


def _derive_read_log_allowlist(task) -> set[str]:
    allowlist = set()
    for action in task.evaluation_criteria.actions or []:
        if action.name == "call_discoverable_agent_tool":
            name = (action.arguments or {}).get("agent_tool_name")
            if name:
                allowlist.add(name)
    return allowlist


def _user_guidelines(use_tools: bool) -> str:
    filename = "simulation_guidelines_tools.md" if use_tools else "simulation_guidelines.md"
    path = f"/opt/tau2-bench/data/tau2/user_simulator/{filename}"
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _message(item: dict[str, Any]):
    role = item["role"]
    if role in {"assistant", "user"}:
        calls = [ToolCall.model_validate(call) for call in item.get("tool_calls") or []]
        cls = AssistantMessage if role == "assistant" else UserMessage
        return cls(role=role, content=item.get("content"), tool_calls=calls or None)
    if role == "tool":
        return ToolMessage.model_validate(item)
    raise ValueError(f"unsupported role: {role}")


def initialize(payload: dict[str, Any]) -> dict[str, Any]:
    task = TASKS[payload["task_id"]]
    env_kwargs = {
        "retrieval_variant": payload["retrieval_variant"],
        "task": task,
        "read_log_allowlist": _derive_read_log_allowlist(task),
    }
    if env_kwargs["retrieval_variant"] != "alltools":
        raise ValueError("This runtime requires retrieval_variant=alltools")
    env = get_environment(**env_kwargs)
    init = task.initial_state
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=list(init.message_history or []) if init else [],
    )
    STATE.clear()
    STATE.update(task=task, env=env, env_kwargs=env_kwargs)
    agent_tools = [tool.openai_schema for tool in env.tools.get_tools().values()]
    selected_user_tools = task.user_tools or []
    user_tools = [
        tool.openai_schema
        for tool in env.user_tools.get_tools(include=selected_user_tools).values()
    ]
    return {
        "task_id": task.id,
        "retrieval_variant": env_kwargs["retrieval_variant"],
        "policy": env.policy,
        # Upstream build_user passes str(task.user_scenario), not the raw
        # instructions field. This preserves the Persona / Instructions labels
        # and indentation implemented by UserScenario.__str__.
        "scenario": str(task.user_scenario),
        "scenario_rendered": True,
        "user_guidelines": _user_guidelines(use_tools=bool(user_tools)),
        "agent_tools": agent_tools,
        "user_tools": user_tools,
        "reward_basis": [str(x.value) for x in task.evaluation_criteria.reward_basis],
        "nl_assertions": task.evaluation_criteria.nl_assertions or [],
    }


def call_tool(payload: dict[str, Any]) -> dict[str, Any]:
    if not STATE:
        raise RuntimeError("bridge is not initialized")
    requestor = payload["requestor"]
    call = ToolCall(
        id=payload["id"],
        name=payload["name"],
        arguments=payload.get("arguments") or {},
        requestor=requestor,
    )
    result = STATE["env"].get_response(call)
    return result.model_dump(mode="json")


def gold_trajectory(_payload: dict[str, Any]) -> dict[str, Any]:
    """Create an executable reference trajectory from official golden actions."""
    if not STATE:
        raise RuntimeError("bridge is not initialized")
    task = STATE["task"]
    messages: list[dict[str, Any]] = [
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]
    for index, action in enumerate(task.evaluation_criteria.actions or []):
        call = ToolCall(
            id=f"gold_{index}",
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        message_cls = AssistantMessage if action.requestor == "assistant" else UserMessage
        call_message = message_cls(role=action.requestor, tool_calls=[call])
        result = STATE["env"].get_response(call)
        messages.append(call_message.model_dump(mode="json"))
        messages.append(result.model_dump(mode="json"))
    assertions = task.evaluation_criteria.nl_assertions or []
    if assertions:
        messages.append({"role": "assistant", "content": " ".join(assertions)})
    messages.append({"role": "user", "content": "Thanks. ###STOP###"})
    return {"messages": messages, "termination_reason": "user_stop"}


def grade(payload: dict[str, Any]) -> dict[str, Any]:
    if not STATE:
        raise RuntimeError("bridge is not initialized")
    task = STATE["task"]
    trajectory = [_message(item) for item in payload["messages"]]
    env_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory,
        env_kwargs=STATE["env_kwargs"],
    )
    action_info = ActionEvaluator.calculate_reward(task=task, full_trajectory=trajectory)
    basis = {str(x.value) for x in task.evaluation_criteria.reward_basis}
    components: dict[str, float] = {}
    reward = 1.0
    if basis & {"DB", "ENV_ASSERTION"}:
        reward *= env_info.reward
        components["environment"] = env_info.reward
    if "ACTION" in basis:
        reward *= action_info.reward
        components["action"] = action_info.reward
    return {
        "reward_without_nl": reward,
        "components": components,
        "reward_basis": sorted(basis),
        "db_check": env_info.db_check.model_dump(mode="json") if env_info.db_check else None,
        "action_checks": [x.model_dump(mode="json") for x in action_info.action_checks or []],
    }


class Handler(BaseHTTPRequestHandler):
    ROUTES = {
        "/embedding/poll": embedding_poll,
        "/embedding/result": embedding_result,
        "/init": initialize,
        "/tool": call_tool,
        "/gold": gold_trajectory,
        "/grade": grade,
    }

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "tasks": len(TASKS)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            size = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(size) or b"{}")
            self._send(200, self.ROUTES[self.path](payload))
        except Exception as exc:
            self._send(500, {"error": type(exc).__name__, "detail": str(exc)})

    def log_message(self, *_args):
        return

    def _send(self, status: int, payload: dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

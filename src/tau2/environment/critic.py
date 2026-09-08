"""Generic second-agent critic for mutating (write) tool calls.

This replaces an earlier approach that hand-wrote a separate verifier
function per risky tool name (see git history). That approach was rejected
as hardcoded: it only covered a hand-picked list of 2-3 tools, with
bespoke prompts and hardcoded KB doc IDs baked into Python code.

This module instead provides ONE generic critic that can review any
mutating/write tool call: given the tool name, its proposed arguments, the
calling agent's stated reasoning, and the recent conversation context
(which contains whatever policy/evidence -- e.g. KB_search results -- the
agent has already surfaced), it asks a second, independent LLM call whether
the proposed action is supported by that context. No tool names or policy
text are hardcoded here.

Determinism / replay: callers are responsible for caching the verdict for
a given (tool_name, arguments) pair within a single environment/task
instance (see `verdict_cache_key`), so that a task-level retry that
re-executes the same tool call does not re-invoke this (non-deterministic)
LLM call and risk producing a differently-worded verdict than the first
run, which would break replay-consistency checks.
"""

import hashlib
import json
from typing import Any, Dict, Optional

from loguru import logger

from tau2.data_model.message import UserMessage
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

# Cheap/fast model used for the critic calls. Falls back to a small model so
# the check doesn't meaningfully add cost/latency vs the main agent call.
CRITIC_LLM_MODEL = "claude-sonnet-5"

# Harness-level argument name the calling agent must supply on any gated
# write action, explaining its reasoning. Popped before the underlying tool
# is actually invoked -- it is never a real tool parameter.
REASONING_ARG = "_reasoning"

# Every error string this gate can return (missing-reasoning, or a critic
# rejection) is prefixed with this marker. It lets replay (Environment.set_state)
# recognize -- purely by inspecting the *recorded* response content, without
# needing to re-invoke any LLM -- that a given historical tool call was
# blocked by the gate and therefore never mutated state, so replay can safely
# skip re-executing it instead of re-running a non-deterministic LLM check.
GATE_BLOCKED_MARKER = "[write-critic-gate-blocked]"

MAX_CONTEXT_MESSAGES = 40


def verdict_cache_key(tool_name: str, args_dict: Dict[str, Any]) -> str:
    """Stable cache key for a (tool_name, arguments) pair.

    Used to cache the critic's verdict so a replay of the exact same tool
    call (e.g. after a task-level retry) reuses the first verdict instead of
    re-invoking the (non-deterministic) LLM critic.
    """
    normalized = json.dumps(args_dict, sort_keys=True, default=str)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{tool_name}:{digest}"


def _format_conversation_context(conversation_history: Optional[list]) -> str:
    if not conversation_history:
        return "(no prior conversation context available)"
    recent = list(conversation_history)[-MAX_CONTEXT_MESSAGES:]
    lines = []
    for msg in recent:
        role = getattr(msg, "role", msg.__class__.__name__)
        try:
            lines.append(f"[{role}]\n{msg}")
        except Exception:  # noqa: BLE001
            continue
    return "\n\n".join(lines) if lines else "(no prior conversation context available)"


def review_write_action(
    tool_name: str,
    args_dict: Dict[str, Any],
    agent_reasoning: str,
    conversation_history: Optional[list] = None,
) -> Optional[str]:
    """Generic second-agent review of a proposed mutating tool call.

    Args:
        tool_name: Name of the tool the first agent is about to call.
        args_dict: The proposed arguments (the actual tool arguments, not
            including the harness-level `_reasoning` field).
        agent_reasoning: The first agent's own stated reasoning for this
            action and these argument values.
        conversation_history: Recent conversation messages (assistant/tool
            messages so far this conversation), used as the evidence/policy
            context the critic checks the action against -- e.g. KB_search
            results and prior tool results already retrieved.

    Returns:
        None if the action is compliant (or if the check could not be
        completed -- fail open), or an error string to return to the
        calling agent (blocking the write) if the critic rejects it.
    """
    context_str = _format_conversation_context(conversation_history)

    prompt = f"""You are a second, independent reviewer for a bank customer-service agent. The
first agent is about to take an action that changes state (a write/mutating action) and has
stated its own reasoning for it. Your job is to critically review the proposed action against
whatever policy and evidence has already been surfaced in the conversation below -- do not
defer to the first agent's reasoning just because it sounds confident; check it yourself.

Look especially for:
- An irreversible or high-impact action taken without adequate verification.
- A numeric value, classification, or category that does not match evidence already present
  in the conversation (e.g. a policy document, a transaction record, a prior tool result).
- An action that is premature given what has (or hasn't) been checked so far.
- Reasoning that doesn't actually follow from the policy/evidence cited.

Recent conversation context (this is the evidence/policy record available to the first agent
-- includes prior tool calls/results and any knowledge-base search results already retrieved):
--- BEGIN CONVERSATION CONTEXT ---
{context_str}
--- END CONVERSATION CONTEXT ---

Proposed tool call:
- tool_name: {tool_name}
- arguments: {json.dumps(args_dict, default=str)}

First agent's stated reasoning for this action and these argument values:
"{agent_reasoning}"

Determine whether this proposed action complies with the policy/evidence already surfaced in
the conversation above. If the conversation context is too thin to tell either way, prefer NOT
rejecting (fail open / compliant=true) -- only reject when something is actually unsupported,
wrong, or premature given what's in the context.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"compliant": true or false, "explanation": "<one or two sentences>"}}
"""

    try:
        response = generate(
            model=CRITIC_LLM_MODEL,
            messages=[UserMessage(role="user", content=prompt)],
            call_name="review_write_action",
            num_retries=1,
        )
        content = response.content or ""
        json_str = extract_json_from_llm_response(content)
        verdict = json.loads(json_str)
        compliant = bool(verdict.get("compliant", True))
        if compliant:
            return None
        explanation = verdict.get(
            "explanation", "This action does not appear supported by the evidence/policy surfaced so far."
        )
        return (
            f"Error: {GATE_BLOCKED_MARKER} proposed call to '{tool_name}' was rejected by "
            f"compliance review. {explanation} Please re-check the policy/evidence available "
            "in this conversation and retry with corrected arguments (or gather more evidence "
            "first)."
        )
    except Exception as e:  # noqa: BLE001 - fail open on any critic error
        logger.warning(f"Generic write-action critic call failed for '{tool_name}': {e}")
        return None

"""Harness-owned canonical Qwen3 format/parse codec.

This module is the single source of truth for turning tau2 messages into a
Qwen3 prompt string and for parsing a vanilla text-completion response back
into structured ``(reasoning, content, tool_calls)``. The exact same
functions are reused to format ToolMind SFT training data, so the model sees
identical formatting at train and eval time *by construction*.

Why this exists
---------------
The baseline path relies on vLLM's server-side chat template + tool-call
parser + reasoning parser when hitting ``/v1/chat/completions``. This module
moves both prompt FORMATTING and output PARSING into the harness so we can
serve a *vanilla* text-in/text-out vLLM (``/v1/completions``) with NO
``--enable-auto-tool-choice`` / ``--tool-call-parser`` / ``--reasoning-parser``
flags, while staying bit-compatible with the native path.

Formatting
----------
``render_chat`` is a thin wrapper over ``tokenizer.apply_chat_template`` (which
IS the canonical Qwen3 template) so it is guaranteed identical to what vLLM's
chat endpoint would build. It is kept isolated so the backing tokenizer can be
swapped/pinned.

Parsing
-------
``parse_completion`` reproduces vLLM's non-streaming semantics. vLLM runs the
reasoning parser FIRST (splitting ``reasoning_content`` from ``content``) and
then runs the tool parser on the resulting content
(``vllm/entrypoints/openai/chat_completion/serving.py``: the reasoning parser's
``extract_reasoning`` is called on ``output.text``, then
``_parse_tool_calls_from_content`` runs on the returned content — "tool calls
are extracted exclusively from the content").

Because the CPU virtualenv used here ships ``transformers``/``tokenizers`` but
NOT ``vllm`` (importing vllm pulls in heavy GPU deps that do not install
cleanly), the two parser classes are FAITHFULLY PORTED here rather than
imported. The logic mirrors, line for line:

  * Tool parsing: ``vllm/tool_parsers/hermes_tool_parser.py``
    ``Hermes2ProToolParser.extract_tool_calls`` (non-streaming).
  * Reasoning parsing: ``vllm/reasoning/qwen3_reasoning_parser.py``
    ``Qwen3ReasoningParser.extract_reasoning`` (non-streaming).

Source reference: vllm-project/vllm @ commit
2616f67faaa735a3e0d9c17968fa91f242d36c56 (fetched via the GitHub API while
building this module). Behavior is pinned by tests in
``tests/test_qwen3_codec.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

# Imported at module load (under the import lock): transformers' lazy loader is
# not thread-safe on first attribute access and races under tau2's concurrency.
from transformers import AutoTokenizer

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

# Default tokenizer that provides the canonical Qwen3 chat template. PIN THIS to
# the exact checkpoint you serve / train on so the prompt we render is the same
# one the model was trained and is served with. Override without code changes via
# the env var TAU2_QWEN3_TOKENIZER (set it to an HF id or a local checkpoint dir).
DEFAULT_TOKENIZER_ID = os.environ.get("TAU2_QWEN3_TOKENIZER", "Qwen/Qwen3-8B")

# Qwen3 turn terminator. Use as the stop sequence for /v1/completions so the
# server stops at the end of the assistant turn instead of hallucinating the
# next turn.
STOP: list[str] = ["<|im_end|>"]

# Tags (kept here so both the renderer and parser share constants).
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

# Generation-prompt tail emitted by the Qwen3 template; used to locate the
# final assistant turn when building training loss masks.
ASSISTANT_HEADER = "<|im_start|>assistant\n"


@dataclass
class ParsedOutput:
    """Structured result of parsing a vanilla completion.

    Mirrors the tuple vLLM produces: reasoning content, message content, and
    tool calls. ``tool_calls`` is ``None`` (not an empty list) when no tools
    were called, matching tau2's ``AssistantMessage`` convention.
    """

    reasoning: Optional[str]
    content: Optional[str]
    tool_calls: Optional[list[ToolCall]]


# ---------------------------------------------------------------------------
# Message <-> chat-template dict conversion
# ---------------------------------------------------------------------------


def messages_to_chat_dicts(messages: list[Message]) -> list[dict]:
    """Convert tau2 messages into the dicts ``apply_chat_template`` expects.

    Important differences from ``llm_utils.to_litellm_messages``:

    * Consecutive ``ToolMessage``s are emitted as separate ``role="tool"``
      dicts; the Qwen3 template itself merges them into one user turn wrapping
      multiple ``<tool_response>`` blocks.
    * ``tool_calls[*].function.arguments`` is passed as a **dict** (not a JSON
      string). The Qwen3 template renders it via ``tojson``, producing
      ``{"city": "Tokyo"}`` with canonical spacing. (Passing a pre-serialized
      string would render without spaces and break parity.)
    """
    out: list[dict] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            out.append({"role": "system", "content": message.content or ""})
        elif isinstance(message, UserMessage):
            out.append({"role": "user", "content": message.content or ""})
        elif isinstance(message, AssistantMessage):
            d: dict[str, Any] = {
                "role": "assistant",
                "content": message.content if message.content is not None else "",
            }
            if message.is_tool_call():
                d["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            out.append(d)
        elif isinstance(message, ToolMessage):
            out.append({"role": "tool", "content": message.content or ""})
        else:
            raise ValueError(f"Unsupported message type for Qwen3 codec: {message!r}")
    return out


def _normalize_conversation_dicts(conversations: list[dict]) -> list[dict]:
    """Normalize ToolMind-style conversation dicts for the chat template.

    ToolMind records already use OpenAI-ish role dicts. We mainly ensure that
    assistant ``tool_calls[*].function.arguments`` is a dict (not a JSON
    string) so it renders identically to the eval path.
    """
    normalized: list[dict] = []
    for turn in conversations:
        role = turn.get("role")
        if role == "assistant" and turn.get("tool_calls"):
            new_calls = []
            for tc in turn["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                new_calls.append(
                    {
                        "type": "function",
                        "function": {"name": fn["name"], "arguments": args},
                    }
                )
            turn = {**turn, "tool_calls": new_calls}
            if turn.get("content") is None:
                turn["content"] = ""
        normalized.append(turn)
    return normalized


# ---------------------------------------------------------------------------
# Rendering (canonical: thin wrapper over apply_chat_template)
# ---------------------------------------------------------------------------


_TOKENIZER_CACHE: dict[str, Any] = {}
_TOKENIZER_LOCK = threading.Lock()


def _get_tokenizer(tokenizer_id: str):
    """Load (and cache) a tokenizer, thread-safely.

    Uses double-checked locking so that concurrent tau2 worker threads sharing
    a ``tokenizer_id`` load it exactly once instead of racing on first access.
    """
    tok = _TOKENIZER_CACHE.get(tokenizer_id)
    if tok is None:
        with _TOKENIZER_LOCK:
            tok = _TOKENIZER_CACHE.get(tokenizer_id)
            if tok is None:
                tok = AutoTokenizer.from_pretrained(tokenizer_id)
                _TOKENIZER_CACHE[tokenizer_id] = tok
    return tok


def get_stop_tokens(tokenizer_id: str = DEFAULT_TOKENIZER_ID) -> list[str]:
    """Stop sequence(s) for ``/v1/completions``, derived from the served model.

    Rendering is model-agnostic via ``tokenizer_id``; the stop sequence must be
    too, or a non-Qwen model never halts at its turn boundary. We use the
    tokenizer's own ``eos_token`` (Qwen3 -> ``<|im_end|>``, Llama-3 -> ``<|eot_id|>``),
    falling back to the module ``STOP`` constant if it exposes none.
    """
    tok = _get_tokenizer(tokenizer_id)
    eos = getattr(tok, "eos_token", None)
    return [eos] if eos else list(STOP)


def chat_template_signature(tokenizer_id: str = DEFAULT_TOKENIZER_ID) -> str:
    """Return a short, stable hash of the tokenizer's chat template.

    Train/eval parity hinges on rendering with the SAME chat template the model
    is served with. Compute this at training time and again against the served
    checkpoint; if they differ, your rendered prompts diverge from what the model
    expects. (Hashes the raw ``chat_template`` string, so it's robust to
    tokenizer-version bumps that leave the template untouched.)
    """
    tok = _get_tokenizer(tokenizer_id)
    template = getattr(tok, "chat_template", None)
    if not template:
        raise ValueError(f"Tokenizer {tokenizer_id!r} has no chat_template to pin.")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def assert_template_matches(
    expected_signature: str, tokenizer_id: str = DEFAULT_TOKENIZER_ID
) -> None:
    """Raise if the tokenizer's chat-template signature != ``expected_signature``.

    Call at serving/training startup to fail fast on a train/eval template skew:

        assert_template_matches(TRAINED_TEMPLATE_SIG, tokenizer_id=served_ckpt)
    """
    actual = chat_template_signature(tokenizer_id)
    if actual != expected_signature:
        raise ValueError(
            f"Chat-template mismatch for {tokenizer_id!r}: expected "
            f"{expected_signature!r}, got {actual!r}. The renderer's template "
            "differs from what you trained/served on — prompts will diverge."
        )


def render_chat(
    messages: list[Message],
    tools: Optional[list[dict]] = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
) -> str:
    """Render a canonical Qwen3 prompt string from tau2 messages.

    This is intentionally a thin wrapper over ``apply_chat_template`` — that
    template IS the canonical Qwen3 format, identical to what vLLM's chat
    endpoint builds server-side. Kept isolated so the tokenizer can be swapped.

    Args:
        messages: tau2 messages.
        tools: OpenAI-style tool schemas (``{type:function, function:{...}}``)
            or ``None``.
        add_generation_prompt: append the ``<|im_start|>assistant\\n`` tail.
        enable_thinking: Qwen3 thinking switch. ``False`` injects the empty
            ``<think>\\n\\n</think>`` block.
        tokenizer_id: HF id of the tokenizer providing the chat template.
    """
    tok = _get_tokenizer(tokenizer_id)
    return tok.apply_chat_template(
        messages_to_chat_dicts(messages),
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def render_chat_dicts(
    conversations: list[dict],
    tools: Optional[list[dict]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
) -> str:
    """Render directly from OpenAI-style conversation dicts (ToolMind path)."""
    tok = _get_tokenizer(tokenizer_id)
    return tok.apply_chat_template(
        _normalize_conversation_dicts(conversations),
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


# ---------------------------------------------------------------------------
# Reasoning parser (ported from vllm Qwen3ReasoningParser.extract_reasoning)
# ---------------------------------------------------------------------------


def _extract_reasoning(
    model_output: str, thinking_enabled: bool
) -> tuple[Optional[str], Optional[str]]:
    """Port of ``Qwen3ReasoningParser.extract_reasoning`` (non-streaming).

    Returns ``(reasoning, content)``. See the vLLM source referenced in the
    module docstring; the branch structure is reproduced exactly.
    """
    # Strip <think> if present in the generated output.
    parts = model_output.partition(THINK_START)
    model_output = parts[2] if parts[1] else parts[0]

    if THINK_END in model_output:
        reasoning, _, content = model_output.partition(THINK_END)
        return reasoning, content or None

    if not thinking_enabled:
        # Thinking explicitly disabled -> everything is content.
        return None, model_output

    # No </think> -- check for implicit reasoning end via <tool_call>.
    tool_call_index = model_output.find(TOOL_CALL_START)
    if tool_call_index != -1:
        reasoning = model_output[:tool_call_index]
        content = model_output[tool_call_index:]
        return reasoning or None, content or None

    # Thinking enabled but no </think>: output was truncated -> all reasoning.
    return model_output, None


# ---------------------------------------------------------------------------
# Tool parser (ported from vllm Hermes2ProToolParser.extract_tool_calls)
# ---------------------------------------------------------------------------

_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
)


def _make_tool_call_id() -> str:
    """vLLM synthesizes random ids; mirror that (``chatcmpl-tool-<hex>``)."""
    return f"chatcmpl-tool-{uuid.uuid4().hex}"


def _extract_tool_calls(
    model_output: str,
) -> tuple[bool, list[ToolCall], Optional[str]]:
    """Port of ``Hermes2ProToolParser.extract_tool_calls`` (non-streaming).

    Returns ``(tools_called, tool_calls, content)``. ``content`` is the text
    before the first ``<tool_call>`` (``None`` if empty). On any parse error,
    falls back to ``(False, [], model_output)`` exactly like vLLM.
    """
    if TOOL_CALL_START not in model_output:
        return False, [], model_output

    try:
        # Two possible captures: between tags, or tag-to-EOS. findall yields
        # tuples where exactly one element is the JSON body.
        function_call_tuples = _TOOL_CALL_REGEX.findall(model_output)
        raw_function_calls = [
            json.loads(match[0] if match[0] else match[1])
            for match in function_call_tuples
        ]
        tool_calls = [
            ToolCall(
                id=_make_tool_call_id(),
                name=fc["name"],
                arguments=fc["arguments"],
            )
            for fc in raw_function_calls
        ]
        content = model_output[: model_output.find(TOOL_CALL_START)]
        # Treat a whitespace-only remainder as no content. The Qwen3 template
        # puts a blank line between </think> and <tool_call>, so the text before
        # the tool call is often "\n\n"; vLLM normalizes that to None on a pure
        # tool-call turn, and we must match to keep replayed history identical.
        return True, tool_calls, (content if content.strip() else None)
    except Exception:
        # Mirror vLLM: swallow and treat the whole thing as content.
        return False, [], model_output


# ---------------------------------------------------------------------------
# Combined parse (reasoning FIRST, then tools — matching vLLM serving order)
# ---------------------------------------------------------------------------


def parse_completion(text: str, thinking_enabled: bool = True) -> ParsedOutput:
    """Parse a vanilla ``/v1/completions`` response into structured fields.

    Reproduces vLLM's non-streaming order of operations:
      1. reasoning parser splits ``(reasoning, content)`` from the raw text;
      2. the Hermes tool parser runs on the resulting ``content`` only.

    Tool-call ids are synthesized (vLLM uses random ids).
    """
    reasoning, content = _extract_reasoning(text, thinking_enabled)

    tool_calls: Optional[list[ToolCall]] = None
    if content is not None:
        tools_called, parsed_calls, tool_content = _extract_tool_calls(content)
        if tools_called:
            tool_calls = parsed_calls
            content = tool_content  # text before first <tool_call>, or None
    return ParsedOutput(reasoning=reasoning, content=content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Training-data formatting (proves same codec serves train + eval)
# ---------------------------------------------------------------------------


def format_training_example(
    record: dict,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    enable_thinking: bool = True,
) -> dict:
    """Format one ToolMind-style record into an SFT training example.

    A ToolMind record is ``{"conversations": [...], "tools": [...]}`` where
    ``conversations`` ends on the target assistant turn (the only turn carrying
    ``<think>``). This uses the SAME ``render_chat`` codec as eval, then builds
    an assistant-only loss mask: every token before the final
    ``<|im_start|>assistant`` header is labeled ``-100``.

    Returns ``{"text", "input_ids", "labels"}``.
    """
    conversations = record["conversations"]
    tools = record.get("tools")

    tok = _get_tokenizer(tokenizer_id)
    norm = _normalize_conversation_dicts(conversations)

    # Full rendered conversation (no trailing generation prompt: the final
    # assistant turn IS the target).
    full_text = tok.apply_chat_template(
        norm,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    # Prefix = everything up to and including the LAST assistant header. Tokens
    # in the prefix are masked; the assistant body (+ closing <|im_end|>) is
    # the supervised target.
    header_idx = full_text.rindex(ASSISTANT_HEADER)
    prefix_text = full_text[: header_idx + len(ASSISTANT_HEADER)]

    input_ids = tok.encode(full_text, add_special_tokens=False)
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    labels = [-100] * n_prefix + list(input_ids[n_prefix:])
    # Defensive: lengths must match (prefix is a token prefix of full).
    if len(labels) != len(input_ids):
        labels = labels[: len(input_ids)]

    return {"text": full_text, "input_ids": input_ids, "labels": labels}

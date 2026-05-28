"""Optional tokenizer-backed length functions for LengthFilter(metric="tokens").

Kept separate from `core`/`nodes` so the base framework has zero heavy deps.
These import transformers / tiktoken lazily, only when actually called.

To match the eval harness, prefer `qwen3_token_counter`: it counts the tokens of
the *chat-templated* example (the same string the model is trained/served on),
not just raw content.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .core import Record

LengthFn = Callable[[Record], float]


@lru_cache(maxsize=4)
def _qwen_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def qwen3_token_counter(
    model: str = "Qwen/Qwen3-8B", enable_thinking: bool = True
) -> LengthFn:
    """Return a length_fn counting full-sequence Qwen3 chat-template tokens.

    Mirrors how the example is tokenized at train/eval time: tools are injected
    into the system block and the whole conversation is rendered before counting.
    """
    tok = _qwen_tokenizer(model)

    def count(record: Record) -> float:
        msgs = record.get("conversations") or record.get("messages") or []
        tools = record.get("tools") or None
        # default empty content so pure tool-call turns don't break the template
        msgs = [{**m, "content": (m.get("content") or "")} for m in msgs]
        text = tok.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=False
        )
        return len(tok(text, add_special_tokens=False)["input_ids"])

    return count


@lru_cache(maxsize=4)
def _tiktoken_enc(encoding: str):
    import tiktoken

    return tiktoken.get_encoding(encoding)


def tiktoken_counter(encoding: str = "cl100k_base") -> LengthFn:
    """Cheap approximate token count over concatenated message contents."""
    from .nodes.length import _content_text

    enc = _tiktoken_enc(encoding)

    def count(record: Record) -> float:
        return len(enc.encode(_content_text(record), disallowed_special=()))

    return count

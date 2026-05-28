"""LengthFilter — the first concrete curation operator.

Drops records whose "length" falls outside [min, max]. Length is measured by a
configurable `metric` so the same node works for several common curation needs:

  * ``"chars"``     — total characters across all message contents
  * ``"messages"``  — number of conversation turns (messages)
  * ``"words"``     — whitespace-delimited word count across contents
  * ``"tokens"``    — token count; requires a `length_fn` to be supplied (so the
                       core package stays dependency-free). See
                       ``curation.length_metrics.qwen3_token_counter`` for a
                       ready-made tokenizer-backed counter.

Records are ToolMind-style ``{"conversations": [...], "tools": [...]}`` but the
metric extraction is tolerant of a few common message schemas.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from ..core import FilterNode, Record, register

# A length function maps a record -> a numeric length.
LengthFn = Callable[[Record], float]

_MESSAGE_KEYS = ("conversations", "messages", "conversation", "turns")


def _messages(record: Record) -> list[dict]:
    for k in _MESSAGE_KEYS:
        v = record.get(k)
        if v:
            return v
    return []


def _content_text(record: Record) -> str:
    parts = []
    for m in _messages(record):
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):  # structured/multimodal content blocks
            parts.extend(str(b.get("text", b)) if isinstance(b, dict) else str(b) for b in c)
    return "\n".join(parts)


def _chars(record: Record) -> float:
    return len(_content_text(record))


def _words(record: Record) -> float:
    return len(_content_text(record).split())


def _num_messages(record: Record) -> float:
    return len(_messages(record))


_BUILTIN_METRICS: dict[str, LengthFn] = {
    "chars": _chars,
    "words": _words,
    "messages": _num_messages,
}


@register("length_filter")
class LengthFilter(FilterNode):
    """Keep records with min_len <= length(record) <= max_len.

    Args:
        metric: one of {"chars", "words", "messages", "tokens"}. For "tokens" you
            must pass `length_fn` (a callable record->int), otherwise a ValueError
            is raised at construction time.
        min_len: inclusive lower bound (None = no lower bound).
        max_len: inclusive upper bound (None = no upper bound).
        length_fn: optional explicit length callable; overrides `metric`.
        name: optional node name for stats/reporting.
    """

    def __init__(
        self,
        metric: str = "chars",
        min_len: Optional[float] = None,
        max_len: Optional[float] = None,
        length_fn: Optional[LengthFn] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.metric = metric
        self.min_len = min_len
        self.max_len = max_len

        if length_fn is not None:
            self._len: LengthFn = length_fn
        elif metric in _BUILTIN_METRICS:
            self._len = _BUILTIN_METRICS[metric]
        elif metric == "tokens":
            raise ValueError(
                "metric='tokens' requires a `length_fn` (e.g. "
                "curation.length_metrics.qwen3_token_counter()). Builtin metrics: "
                f"{sorted(_BUILTIN_METRICS)}"
            )
        else:
            raise ValueError(
                f"Unknown metric {metric!r}. Use one of {sorted(_BUILTIN_METRICS)} "
                "+ 'tokens', or pass length_fn."
            )

        if min_len is None and max_len is None:
            raise ValueError("LengthFilter needs at least one of min_len / max_len.")
        if min_len is not None and max_len is not None and min_len > max_len:
            raise ValueError(f"min_len ({min_len}) > max_len ({max_len}).")

    def keep(self, record: Record) -> bool:
        n = self._len(record)
        if self.min_len is not None and n < self.min_len:
            return False
        if self.max_len is not None and n > self.max_len:
            return False
        return True

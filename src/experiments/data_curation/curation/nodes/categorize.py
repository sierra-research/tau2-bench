"""Categorization — annotating nodes that label each record.

These don't change the data; they attach a category string that samplers/filters
downstream consume. `ToolUseCategorizer` is the "categorization job" for ToolMind:
it buckets each trajectory by conversational structure. `NumericBucketizer` is a
generic helper that maps any numeric feature into named ranges.
"""
from __future__ import annotations

from typing import Optional

from ..core import MapNode, Record, register

_MESSAGE_KEYS = ("conversations", "messages", "conversation", "turns")


def _messages(record: Record) -> list[dict]:
    for k in _MESSAGE_KEYS:
        v = record.get(k)
        if v:
            return v
    return []


def _has_tool_call(messages: list[dict]) -> bool:
    return any(m.get("tool_calls") for m in messages)


@register("categorize_tool_use")
class ToolUseCategorizer(MapNode):
    """Label each record by tool-use structure, written to `field` (default "category").

    Categories:
        chat_only     — no tool call anywhere
        single_call   — has a tool call, <= 2 turns
        multi_turn    — has a tool call, 3..10 turns
        long_agentic  — has a tool call, > 10 turns

    Also stashes raw features under `<field>_features` for inspection.
    """

    def __init__(
        self,
        field: str = "category",
        multi_turn_max: int = 10,
        single_turn_max: int = 2,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.field = field
        self.multi_turn_max = multi_turn_max
        self.single_turn_max = single_turn_max
        from collections import Counter

        self._dist: Counter = Counter()

    def _label(self, messages: list[dict]) -> str:
        if not _has_tool_call(messages):
            return "chat_only"
        n = len(messages)
        if n <= self.single_turn_max:
            return "single_call"
        if n <= self.multi_turn_max:
            return "multi_turn"
        return "long_agentic"

    def transform(self, record: Record) -> Record:
        msgs = _messages(record)
        label = self._label(msgs)
        self._dist[label] += 1
        self.stats.extra["distribution"] = dict(self._dist)
        out = dict(record)
        out[self.field] = label
        out[f"{self.field}_features"] = {
            "n_turns": len(msgs),
            "n_tools": len(record.get("tools") or []),
            "has_tool_call": _has_tool_call(msgs),
        }
        return out


@register("bucketize")
class NumericBucketizer(MapNode):
    """Generic: map a numeric feature into named buckets.

    `metric` is one of {"turns", "tools"} (cheap, dependency-free). `edges` is an
    ascending list of upper bounds; `labels` has len(edges)+1 names.

        NumericBucketizer(metric="turns", edges=[2, 10], labels=["short","mid","long"])
    """

    _METRICS = {
        "turns": lambda r: len(_messages(r)),
        "tools": lambda r: len(r.get("tools") or []),
    }

    def __init__(
        self,
        metric: str,
        edges: list[float],
        labels: list[str],
        field: str = "bucket",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if metric not in self._METRICS:
            raise ValueError(f"Unknown metric {metric!r}; use {sorted(self._METRICS)}")
        if len(labels) != len(edges) + 1:
            raise ValueError("labels must have exactly len(edges)+1 entries")
        if list(edges) != sorted(edges):
            raise ValueError("edges must be ascending")
        self.metric = metric
        self.edges = list(edges)
        self.labels = list(labels)
        self.field = field

    def transform(self, record: Record) -> Record:
        v = self._METRICS[self.metric](record)
        idx = len(self.edges)
        for i, e in enumerate(self.edges):
            if v <= e:
                idx = i
                break
        out = dict(record)
        out[self.field] = self.labels[idx]
        return out

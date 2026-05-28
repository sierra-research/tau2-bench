"""Core primitives for declarative data curation.

A curation job is a directed graph of **nodes** (operators). Each node consumes a
stream of records and produces a stream of records. Because every node has the
same `Iterable[Record] -> Iterator[Record]` shape, nodes compose freely: a linear
`Pipeline` is just the common case of a graph that is a path.

Design goals:
  * **Streaming** — records flow one at a time so we can process multi-GB JSONL
    without loading it into memory.
  * **Observable** — every node tracks how many records it saw / emitted / dropped,
    which is the whole point of data curation (you want to know what you threw away).
  * **Declarative** — nodes self-register, so a pipeline can be built from a plain
    dict / YAML config (see `registry` + `Pipeline.from_config`).

The first concrete operator lives in `curation.nodes.length`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Optional

# A record is just a JSON-like dict (e.g. a ToolMind example {"conversations", "tools"}).
# Keeping it as a plain dict keeps nodes decoupled from any particular schema.
Record = dict[str, Any]


@dataclass
class NodeStats:
    """Per-node counters, populated as records flow through."""

    name: str
    seen: int = 0
    emitted: int = 0
    dropped: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def drop_rate(self) -> float:
        return self.dropped / self.seen if self.seen else 0.0

    def as_dict(self) -> dict[str, Any]:
        d = {
            "node": self.name,
            "seen": self.seen,
            "emitted": self.emitted,
            "dropped": self.dropped,
            "drop_rate": round(self.drop_rate, 4),
        }
        if self.extra:
            d["extra"] = self.extra
        return d


class Node(ABC):
    """A curation operator: a stream transform over records.

    Subclass and implement `process`, or more commonly subclass `FilterNode` /
    `MapNode` which implement the streaming bookkeeping for you.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or type(self).__name__
        self.stats = NodeStats(self.name)

    @abstractmethod
    def process(self, records: Iterable[Record]) -> Iterator[Record]:
        """Consume records, yield transformed/filtered records."""

    def __call__(self, records: Iterable[Record]) -> Iterator[Record]:
        return self.process(records)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class FilterNode(Node):
    """Keeps records for which `keep` returns True. Implement `keep`."""

    @abstractmethod
    def keep(self, record: Record) -> bool:
        ...

    def process(self, records: Iterable[Record]) -> Iterator[Record]:
        for r in records:
            self.stats.seen += 1
            if self.keep(r):
                self.stats.emitted += 1
                yield r
            else:
                self.stats.dropped += 1


class MapNode(Node):
    """Transforms each record. Return `None` from `transform` to drop it."""

    @abstractmethod
    def transform(self, record: Record) -> Optional[Record]:
        ...

    def process(self, records: Iterable[Record]) -> Iterator[Record]:
        for r in records:
            self.stats.seen += 1
            out = self.transform(r)
            if out is None:
                self.stats.dropped += 1
            else:
                self.stats.emitted += 1
                yield out


# --------------------------------------------------------------------------- #
# Declarative node registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[Node]] = {}


def register(type_name: str):
    """Class decorator registering a Node under a string `type` for config-driven builds."""

    def deco(cls: type[Node]) -> type[Node]:
        if type_name in _REGISTRY and _REGISTRY[type_name] is not cls:
            raise ValueError(f"Node type {type_name!r} already registered to {_REGISTRY[type_name]}")
        _REGISTRY[type_name] = cls
        cls.registry_name = type_name  # type: ignore[attr-defined]
        return cls

    return deco


def build_node(spec: dict[str, Any]) -> Node:
    """Build a single node from a `{"type": ..., **kwargs}` spec."""
    spec = dict(spec)
    try:
        type_name = spec.pop("type")
    except KeyError:
        raise ValueError(f"Node spec missing 'type': {spec!r}")
    if type_name not in _REGISTRY:
        raise KeyError(
            f"Unknown node type {type_name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[type_name](**spec)


def registered_nodes() -> dict[str, type[Node]]:
    return dict(_REGISTRY)

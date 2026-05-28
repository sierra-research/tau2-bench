"""Composition: chain nodes into pipelines and graphs.

`Pipeline` is the linear case (a path). `Graph` is the general DAG, supporting
fan-out (one node feeding several) via `itertools.tee` and fan-in (several nodes
feeding one) via concatenation. Both are themselves `Node`s, so graphs nest.
"""
from __future__ import annotations

import itertools
from typing import Any, Iterable, Iterator, Optional

from .core import Node, Record, build_node


def _mermaid_label(node: Node, with_stats: bool, display_name: Optional[str] = None) -> str:
    """Build a quoted Mermaid node label from a curation Node (+ optional stats)."""
    cls = type(node).__name__
    name = display_name or node.name
    label = cls if name == cls else f"{name}<br/><i>{cls}</i>"
    if with_stats and node.stats.seen:
        s = node.stats
        label += f"<br/>{s.emitted}/{s.seen} kept"
        if s.dropped:
            label += f" ({s.drop_rate:.0%} dropped)"
    return label.replace('"', "'")


class Pipeline(Node):
    """Run nodes in sequence: out = nodeN(...(node1(source))).

    Being a Node itself, a Pipeline can be nested inside another pipeline/graph.
    """

    def __init__(self, nodes: list[Node], name: str = "Pipeline"):
        super().__init__(name)
        self.nodes = list(nodes)

    def process(self, records: Iterable[Record]) -> Iterator[Record]:
        stream: Iterable[Record] = records
        for node in self.nodes:
            stream = node(stream)
        yield from stream

    def report(self) -> list[dict[str, Any]]:
        """Per-node stats, in order. Call after the stream has been consumed."""
        return [n.stats.as_dict() for n in self.nodes]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Pipeline":
        """Build from a declarative config:

            {"name": "my_pipe", "nodes": [{"type": "length_filter", ...}, ...]}
        """
        nodes = [build_node(spec) for spec in config["nodes"]]
        return cls(nodes, name=config.get("name", "Pipeline"))

    def to_mermaid(self, with_stats: bool = False, direction: str = "TD") -> str:
        """Render the pipeline as a Mermaid flowchart string.

        If `with_stats` and the pipeline has been run, kept/dropped counts are shown.
        Paste the output into a ```mermaid block, mermaid.live, or render with mmdc.
        """
        lines = [f"flowchart {direction}", "    source([source])"]
        prev = "source"
        for i, node in enumerate(self.nodes):
            nid = f"n{i}"
            lines.append(f'    {nid}["{_mermaid_label(node, with_stats)}"]')
            lines.append(f"    {prev} --> {nid}")
            prev = nid
        lines.append("    sink([output])")
        lines.append(f"    {prev} --> sink")
        return "\n".join(lines)


class Graph(Node):
    """A DAG of named nodes.

    Add nodes with `add(name, node, inputs=...)`, where `inputs` names upstream
    node(s) or the sentinel "source" for the graph's input stream. A node with
    multiple inputs receives them concatenated; a node feeding multiple downstreams
    is `tee`'d so each consumer gets the full stream.

    `process` yields the output of the graph's sink(s) (nodes with no consumers),
    concatenated. For a single-sink DAG this behaves like a Pipeline.
    """

    SOURCE = "source"

    def __init__(self, name: str = "Graph"):
        super().__init__(name)
        self._nodes: dict[str, Node] = {}
        self._inputs: dict[str, list[str]] = {}

    def add(self, name: str, node: Node, inputs: str | list[str] = SOURCE) -> "Graph":
        if name in self._nodes:
            raise ValueError(f"Duplicate node name {name!r}")
        if name == self.SOURCE:
            raise ValueError(f"{self.SOURCE!r} is a reserved name")
        self._nodes[name] = node
        self._inputs[name] = [inputs] if isinstance(inputs, str) else list(inputs)
        return self

    def _topo_order(self) -> list[str]:
        order: list[str] = []
        temp: set[str] = set()
        done: set[str] = set()

        def visit(n: str) -> None:
            if n == self.SOURCE or n in done:
                return
            if n in temp:
                raise ValueError(f"Cycle detected at node {n!r}")
            if n not in self._nodes:
                raise KeyError(f"Unknown upstream node {n!r}")
            temp.add(n)
            for up in self._inputs[n]:
                visit(up)
            temp.discard(n)
            done.add(n)
            order.append(n)

        for n in self._nodes:
            visit(n)
        return order

    def _consumers(self) -> dict[str, list[str]]:
        cons: dict[str, list[str]] = {self.SOURCE: []}
        for n in self._nodes:
            cons.setdefault(n, [])
        for n, ups in self._inputs.items():
            for up in ups:
                cons.setdefault(up, []).append(n)
        return cons

    def process(self, records: Iterable[Record]) -> Iterator[Record]:
        order = self._topo_order()
        consumers = self._consumers()

        # For each producer, hold an iterator queue tee'd across its consumers.
        # `pending[name]` is a list of iterators, one reserved per downstream consumer.
        n_consumers = {name: len(consumers.get(name, [])) for name in [self.SOURCE, *self._nodes]}

        outputs: dict[str, Iterable[Record]] = {}
        # Source may feed several nodes -> tee it.
        outputs[self.SOURCE] = records
        produced: dict[str, list[Iterator[Record]]] = {}

        def fanout(name: str, stream: Iterable[Record]) -> None:
            k = max(1, n_consumers.get(name, 0))
            produced[name] = list(itertools.tee(stream, k)) if k > 1 else [iter(stream)]

        fanout(self.SOURCE, records)

        def take_input(name: str) -> Iterable[Record]:
            # Pop one reserved tee branch for the requesting consumer.
            return produced[name].pop()

        for name in order:
            ups = self._inputs[name]
            if len(ups) == 1:
                in_stream: Iterable[Record] = take_input(ups[0])
            else:
                in_stream = itertools.chain.from_iterable(take_input(u) for u in ups)
            out_stream = self._nodes[name](in_stream)
            fanout(name, out_stream)

        # Sinks = nodes nobody consumes. Concatenate their outputs.
        sinks = [n for n in self._nodes if not consumers.get(n)]
        if not sinks:
            raise ValueError("Graph has no sink node (every node is consumed).")
        yield from itertools.chain.from_iterable(produced[s].pop() for s in sinks)

    def report(self) -> list[dict[str, Any]]:
        return [self._nodes[n].stats.as_dict() for n in self._nodes]

    def to_mermaid(self, with_stats: bool = False, direction: str = "TD") -> str:
        """Render the DAG as a Mermaid flowchart string (source + sinks marked)."""
        ids: dict[str, str] = {self.SOURCE: "source"}
        for i, name in enumerate(self._nodes):
            ids[name] = f"n{i}"
        consumers = self._consumers()

        lines = [f"flowchart {direction}", "    source([source])"]
        for name, node in self._nodes.items():
            lines.append(f'    {ids[name]}["{_mermaid_label(node, with_stats, name)}"]')
        for name, ups in self._inputs.items():
            for up in ups:
                lines.append(f"    {ids[up]} --> {ids[name]}")
        # Mark sink nodes (consumed by nobody) with an output terminal.
        for name in self._nodes:
            if not consumers.get(name):
                lines.append(f"    {ids[name]} --> {ids[name]}_out([output])")
        return "\n".join(lines)

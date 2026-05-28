"""curation — declarative, streaming data-curation primitives.

    from curation import Pipeline, LengthFilter, read_jsonl, write_jsonl

    pipe = Pipeline([LengthFilter(metric="messages", min_len=2, max_len=40)])
    write_jsonl(pipe(read_jsonl("in.jsonl")), "out.jsonl")
    print(pipe.report())

Or build declaratively from config:

    pipe = Pipeline.from_config({
        "name": "demo",
        "nodes": [{"type": "length_filter", "metric": "messages", "max_len": 40}],
    })
"""
from .core import (
    FilterNode,
    MapNode,
    Node,
    NodeStats,
    Record,
    build_node,
    register,
    registered_nodes,
)
from .graph import Graph, Pipeline
from .io import read_jsonl, write_jsonl

# Importing nodes registers the built-ins (e.g. "length_filter").
from . import nodes  # noqa: F401,E402
from .nodes import (  # noqa: E402
    CategorySampler,
    DropFields,
    FilterByField,
    FormatSFT,
    LengthFilter,
    NumericBucketizer,
    SplitTrajectory,
    ToolUseCategorizer,
    ValidateToolCalls,
)

__all__ = [
    "Node",
    "FilterNode",
    "MapNode",
    "NodeStats",
    "Record",
    "register",
    "build_node",
    "registered_nodes",
    "Pipeline",
    "Graph",
    "read_jsonl",
    "write_jsonl",
    "LengthFilter",
    "ValidateToolCalls",
    "ToolUseCategorizer",
    "NumericBucketizer",
    "CategorySampler",
    "FilterByField",
    "DropFields",
    "FormatSFT",
    "SplitTrajectory",
]

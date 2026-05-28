"""Tests for validate / categorize / sample / filter nodes + Mermaid rendering."""
import pytest

from curation import (
    CategorySampler,
    DropFields,
    FilterByField,
    Graph,
    NumericBucketizer,
    Pipeline,
    ToolUseCategorizer,
    ValidateToolCalls,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "get time",
            "parameters": {
                "type": "object",
                "properties": {"tz": {"type": "string"}},
                "required": ["tz"],
            },
        },
    }
]


def call(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


# --------------------------- ValidateToolCalls --------------------------- #

def test_validate_good_call():
    rec = {"tools": TOOLS, "conversations": [{"role": "user", "content": "hi"}, call("get_time", {"tz": "UTC"})]}
    out = ValidateToolCalls().transform(rec)
    assert out["tool_calls_valid"] is True
    assert out["tool_call_errors"] == []


def test_validate_missing_required_arg():
    rec = {"tools": TOOLS, "conversations": [call("get_time", {})]}
    out = ValidateToolCalls().transform(rec)
    assert out["tool_calls_valid"] is False
    assert any("tz" in e for e in out["tool_call_errors"])


def test_validate_unknown_tool():
    rec = {"tools": TOOLS, "conversations": [call("nope", {"tz": "UTC"})]}
    out = ValidateToolCalls().transform(rec)
    assert out["tool_calls_valid"] is False
    assert any("unknown tool" in e for e in out["tool_call_errors"])


def test_validate_wrong_type():
    rec = {"tools": TOOLS, "conversations": [call("get_time", {"tz": 123})]}
    out = ValidateToolCalls().transform(rec)
    assert out["tool_calls_valid"] is False


def test_validate_arguments_as_json_string():
    rec = {"tools": TOOLS, "conversations": [call("get_time", '{"tz": "UTC"}')]}
    assert ValidateToolCalls().transform(rec)["tool_calls_valid"] is True


def test_validate_null_parameters_is_valid():
    tools = [{"type": "function", "function": {"name": "f", "parameters": None}}]
    rec = {"tools": tools, "conversations": [call("f", {"anything": 1})]}
    assert ValidateToolCalls().transform(rec)["tool_calls_valid"] is True


def test_validate_normalizes_python_style_types():
    """ToolMind schemas use 'dict'/'float'/'int' — must normalize, not error."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "estimate",
                "parameters": {
                    "type": "dict",  # -> object
                    "properties": {
                        "location": {"type": "string"},
                        "depth": {"type": "float"},  # -> number
                        "count": {"type": "int"},    # -> integer
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    good = {"tools": tools, "conversations": [call("estimate", {"location": "NS", "depth": 5000.0, "count": 3})]}
    out = ValidateToolCalls().transform(good)
    assert out["tool_calls_valid"] is True
    assert not any("Unknown type" in e for e in out["tool_call_errors"])
    # and it still catches real violations after normalization
    bad = {"tools": tools, "conversations": [call("estimate", {"depth": "deep"})]}
    out2 = ValidateToolCalls().transform(bad)
    assert out2["tool_calls_valid"] is False


def test_validate_require_tool_call():
    rec = {"tools": TOOLS, "conversations": [{"role": "user", "content": "hi"}]}
    out = ValidateToolCalls(require_tool_call=True).transform(rec)
    assert out["tool_calls_valid"] is False


def test_validate_then_filter_composes():
    good = {"tools": TOOLS, "conversations": [call("get_time", {"tz": "UTC"})]}
    bad = {"tools": TOOLS, "conversations": [call("get_time", {})]}
    pipe = Pipeline(
        [
            ValidateToolCalls(),
            FilterByField(field="tool_calls_valid", is_true=True),
            DropFields(fields=["tool_calls_valid", "tool_call_errors"]),
        ]
    )
    out = list(pipe([good, bad]))
    assert out == [good]  # only the valid one, annotations stripped


# --------------------------- Categorizer --------------------------- #

def _conv(n_turns, tool=False):
    msgs = [{"role": "user", "content": "x"} for _ in range(n_turns)]
    if tool:
        msgs[-1] = call("get_time", {"tz": "UTC"})
    return {"conversations": msgs, "tools": TOOLS}


def test_categorize_labels():
    cz = ToolUseCategorizer()
    assert cz.transform(_conv(2, tool=False))["category"] == "chat_only"
    assert cz.transform(_conv(2, tool=True))["category"] == "single_call"
    assert cz.transform(_conv(6, tool=True))["category"] == "multi_turn"
    assert cz.transform(_conv(20, tool=True))["category"] == "long_agentic"
    assert cz.stats.extra["distribution"]["long_agentic"] == 1


def test_numeric_bucketizer():
    b = NumericBucketizer(metric="turns", edges=[2, 10], labels=["short", "mid", "long"])
    assert b.transform(_conv(2))["bucket"] == "short"
    assert b.transform(_conv(5))["bucket"] == "mid"
    assert b.transform(_conv(50))["bucket"] == "long"
    with pytest.raises(ValueError):
        NumericBucketizer(metric="turns", edges=[2], labels=["a", "b", "c"])


# --------------------------- CategorySampler --------------------------- #

def test_sampler_integer_rates_are_deterministic():
    recs = [{"category": "a"}, {"category": "b"}, {"category": "a"}]
    s = CategorySampler(rates={"a": 3, "b": 0}, default_rate=1)
    out = list(s(recs))
    assert sum(r["category"] == "a" for r in out) == 6  # 2 records * 3 copies
    assert sum(r["category"] == "b" for r in out) == 0
    assert s.stats.extra["out_by_category"] == {"a": 6, "b": 0}


def test_sampler_default_rate_passthrough():
    recs = [{"category": "x"} for _ in range(5)]
    s = CategorySampler(rates={}, default_rate=1)
    assert len(list(s(recs))) == 5


def test_sampler_fractional_is_seeded():
    recs = [{"category": "a"} for _ in range(1000)]
    out1 = list(CategorySampler(rates={"a": 0.5}, seed=42)(list(recs)))
    out2 = list(CategorySampler(rates={"a": 0.5}, seed=42)(list(recs)))
    assert len(out1) == len(out2)  # reproducible
    assert 350 < len(out1) < 650  # ~half


def test_sampler_rejects_negative():
    with pytest.raises(ValueError):
        CategorySampler(rates={"a": -1})


def test_categorize_then_sample_balances():
    src = [_conv(2, tool=False) for _ in range(10)] + [_conv(6, tool=True) for _ in range(2)]
    pipe = Pipeline(
        [
            ToolUseCategorizer(),
            CategorySampler(rates={"chat_only": 0.0, "multi_turn": 3}, default_rate=1),
        ]
    )
    out = list(pipe(src))
    cats = [r["category"] for r in out]
    assert cats.count("chat_only") == 0
    assert cats.count("multi_turn") == 6  # 2 * 3


# --------------------------- FilterByField --------------------------- #

def test_filter_by_field_conditions():
    recs = [{"score": 1}, {"score": 5}, {"score": 9}, {"other": 1}]
    assert len(list(FilterByField(field="score", min=2, max=8)(recs))) == 1
    assert len(list(FilterByField(field="score", in_=[1, 9])(recs))) == 2
    # missing field defaults to drop
    assert len(list(FilterByField(field="score", min=0)(recs))) == 3
    assert len(list(FilterByField(field="score", min=0, missing="keep")(recs))) == 4


def test_filter_by_field_needs_condition():
    with pytest.raises(ValueError):
        FilterByField(field="x")


# --------------------------- Mermaid --------------------------- #

def test_pipeline_to_mermaid():
    pipe = Pipeline([ToolUseCategorizer(name="cat"), CategorySampler(rates={}, name="sampler")])
    m = pipe.to_mermaid()
    assert m.startswith("flowchart TD")
    assert "source([source])" in m and "sink([output])" in m
    assert "source --> n0" in m and "n0 --> n1" in m and "n1 --> sink" in m
    assert "cat" in m and "sampler" in m


def test_graph_to_mermaid_branch():
    g = Graph()
    g.add("even", FilterByField(field="n", min=0, name="even"), inputs="source")
    g.add("odd", FilterByField(field="n", min=0, name="odd"), inputs="source")
    g.add("merge", DropFields(fields=[], name="merge"), inputs=["even", "odd"])
    m = g.to_mermaid()
    assert "source --> n0" in m and "source --> n1" in m
    assert "n0 --> n2" in m and "n1 --> n2" in m  # fan-in to merge
    assert "n2 --> n2_out([output])" in m  # sink terminal


def test_mermaid_with_stats():
    pipe = Pipeline([FilterByField(field="score", min=5, name="keep")])
    list(pipe([{"score": 1}, {"score": 9}]))  # run it
    m = pipe.to_mermaid(with_stats=True)
    assert "kept" in m and "dropped" in m

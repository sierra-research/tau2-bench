"""Tests for the curation primitives: nodes, stats, pipeline, graph, declarative build."""
import pytest

from curation import (
    FilterNode,
    Graph,
    LengthFilter,
    MapNode,
    Pipeline,
    build_node,
    read_jsonl,
    registered_nodes,
    write_jsonl,
)


def ex(*contents):
    """Build a ToolMind-ish record from message content strings."""
    return {"conversations": [{"role": "user", "content": c} for c in contents]}


# --------------------------- LengthFilter --------------------------- #

def test_length_filter_chars_bounds():
    recs = [ex("a"), ex("abcde"), ex("abcdefghij")]  # 1, 5, 10 chars
    f = LengthFilter(metric="chars", min_len=2, max_len=8)
    out = list(f(recs))
    assert out == [ex("abcde")]
    assert (f.stats.seen, f.stats.emitted, f.stats.dropped) == (3, 1, 2)
    assert f.stats.drop_rate == pytest.approx(2 / 3)


def test_length_filter_messages_metric():
    recs = [ex("a"), ex("a", "b"), ex("a", "b", "c")]
    f = LengthFilter(metric="messages", max_len=2)
    assert [len(r["conversations"]) for r in f(recs)] == [1, 2]


def test_length_filter_words_metric():
    recs = [ex("one two three"), ex("only")]
    f = LengthFilter(metric="words", min_len=2)
    assert list(f(recs)) == [ex("one two three")]


def test_length_filter_custom_length_fn():
    f = LengthFilter(metric="tokens", max_len=3, length_fn=lambda r: len(r["x"]))
    assert list(f([{"x": [1, 2]}, {"x": [1, 2, 3, 4]}])) == [{"x": [1, 2]}]


def test_tokens_metric_requires_length_fn():
    with pytest.raises(ValueError, match="requires a `length_fn`"):
        LengthFilter(metric="tokens", max_len=10)


def test_length_filter_validation():
    with pytest.raises(ValueError, match="at least one of"):
        LengthFilter(metric="chars")
    with pytest.raises(ValueError, match="min_len .* > max_len"):
        LengthFilter(metric="chars", min_len=10, max_len=1)
    with pytest.raises(ValueError, match="Unknown metric"):
        LengthFilter(metric="bogus", max_len=1)


def test_handles_alternate_message_keys():
    rec = {"messages": [{"role": "user", "content": "hello world"}]}
    f = LengthFilter(metric="words", min_len=1)
    assert list(f([rec])) == [rec]


# --------------------------- Pipeline --------------------------- #

class _Tagger(MapNode):
    def transform(self, record):
        return {**record, "tagged": True}


def test_pipeline_chains_and_reports():
    recs = [ex("a"), ex("abcde"), ex("abcdefghij")]
    pipe = Pipeline(
        [LengthFilter(metric="chars", min_len=2, name="len"), _Tagger(name="tag")]
    )
    out = list(pipe(recs))
    assert all(r["tagged"] for r in out)
    assert len(out) == 2
    report = pipe.report()
    assert report[0]["node"] == "len" and report[0]["dropped"] == 1
    assert report[1]["node"] == "tag" and report[1]["emitted"] == 2


def test_pipeline_is_streaming():
    """Records should flow lazily — consuming a prefix must not pull the whole source."""
    pulled = []

    def gen():
        for i in range(1000):
            pulled.append(i)
            yield {"conversations": [{"role": "user", "content": "x" * i}]}

    pipe = Pipeline([LengthFilter(metric="chars", min_len=0, max_len=10_000)])
    it = pipe(gen())
    next(it)
    next(it)
    assert len(pulled) == 2  # only two records materialized


# --------------------------- Declarative build --------------------------- #

def test_registry_and_build_node():
    assert "length_filter" in registered_nodes()
    node = build_node({"type": "length_filter", "metric": "chars", "max_len": 5})
    assert isinstance(node, LengthFilter)
    assert list(node([ex("ab"), ex("abcdef")])) == [ex("ab")]


def test_pipeline_from_config():
    pipe = Pipeline.from_config(
        {
            "name": "demo",
            "nodes": [
                {"type": "length_filter", "metric": "messages", "min_len": 2},
                {"type": "length_filter", "metric": "chars", "max_len": 6},
            ],
        }
    )
    recs = [ex("a"), ex("a", "b"), ex("aaa", "bbbb")]  # 1msg; 2msg/2ch; 2msg/7ch
    assert list(pipe(recs)) == [ex("a", "b")]


def test_build_node_errors():
    with pytest.raises(ValueError, match="missing 'type'"):
        build_node({"metric": "chars"})
    with pytest.raises(KeyError, match="Unknown node type"):
        build_node({"type": "does_not_exist"})


# --------------------------- Graph (DAG) --------------------------- #

def test_graph_linear_equivalent_to_pipeline():
    g = Graph()
    g.add("a", LengthFilter(metric="chars", min_len=2), inputs="source")
    g.add("b", _Tagger(), inputs="a")
    out = list(g([ex("x"), ex("xxxxx")]))
    assert out == [{**ex("xxxxx"), "tagged": True}]


def test_graph_fanout_and_fanin():
    """source -> {even, odd} -> merge. Verifies tee (fan-out) + chain (fan-in)."""

    class Keep(FilterNode):
        def __init__(self, parity, **kw):
            super().__init__(**kw)
            self.parity = parity

        def keep(self, record):
            return record["n"] % 2 == self.parity

    g = Graph()
    g.add("even", Keep(0, name="even"), inputs="source")
    g.add("odd", Keep(1, name="odd"), inputs="source")
    g.add("merge", _Tagger(name="merge"), inputs=["even", "odd"])

    src = [{"n": i} for i in range(4)]
    out = list(g(src))
    ns = sorted(r["n"] for r in out)
    assert ns == [0, 1, 2, 3]  # nothing lost across the fan-out/fan-in
    assert all(r["tagged"] for r in out)


def test_graph_detects_cycle():
    g = Graph()
    g.add("a", _Tagger(), inputs="b")
    g.add("b", _Tagger(), inputs="a")
    with pytest.raises(ValueError, match="Cycle"):
        list(g([{"n": 1}]))


# --------------------------- I/O round trip --------------------------- #

def test_jsonl_roundtrip(tmp_path):
    recs = [ex("hello"), ex("world", "again")]
    p = tmp_path / "data.jsonl"
    assert write_jsonl(recs, p) == 2
    assert list(read_jsonl(p)) == recs


def test_end_to_end_files(tmp_path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    write_jsonl([ex("a"), ex("abcde"), ex("abcdefghij")], src)
    pipe = Pipeline([LengthFilter(metric="chars", min_len=2, max_len=8)])
    n = write_jsonl(pipe(read_jsonl(src)), dst)
    assert n == 1
    assert list(read_jsonl(dst)) == [ex("abcde")]

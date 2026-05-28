"""Smoke tests for the CLI: pipeline loads from YAML and `run` produces output."""
import json

import yaml

from curation import read_jsonl
from curation.cli import load_pipeline, run


def _write_config(tmp_path):
    cfg = {
        "name": "test_pipe",
        "nodes": [
            {"type": "length_filter", "metric": "messages", "min_len": 2, "name": "turns"},
            {"type": "categorize_tool_use", "name": "cat"},
        ],
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _write_input(tmp_path):
    recs = [
        {"conversations": [{"role": "user", "content": "hi"}]},  # 1 turn -> dropped
        {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]},
    ]
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    return p


def test_load_pipeline(tmp_path):
    pipe = load_pipeline(str(_write_config(tmp_path)))
    assert [type(n).__name__ for n in pipe.nodes] == ["LengthFilter", "ToolUseCategorizer"]


def test_run_writes_output_and_mermaid(tmp_path):
    cfg = _write_config(tmp_path)
    inp = _write_input(tmp_path)
    out = tmp_path / "out.jsonl"
    mmd = tmp_path / "g.mmd"
    pipe = run(str(cfg), str(inp), str(out), mermaid_path=str(mmd))

    rows = list(read_jsonl(out))
    assert len(rows) == 1  # one record survived the turn filter
    assert rows[0]["category"] == "chat_only"
    # stats populated
    assert pipe.nodes[0].stats.seen == 2 and pipe.nodes[0].stats.dropped == 1
    # mermaid written with stats
    text = mmd.read_text()
    assert text.startswith("flowchart TD") and "kept" in text


def test_to_mermaid_static():
    from curation import Pipeline, LengthFilter

    m = Pipeline([LengthFilter(metric="chars", max_len=5)]).to_mermaid()
    assert "flowchart TD" in m and "source --> n0" in m and "n0 --> sink" in m

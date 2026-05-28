# curation

Declarative, streaming primitives for data curation. You define **nodes**
(operators) and **compose** them into pipelines / graphs; the framework streams
records through and tracks what each node kept vs. dropped.

```
curation/
  core.py            Node, FilterNode, MapNode, NodeStats, registry (register/build_node)
  graph.py           Pipeline (linear) + Graph (DAG); .to_mermaid() on both
  io.py              read_jsonl / write_jsonl (streaming)
  length_metrics.py  optional tokenizer-backed length fns (qwen3 chat-template, tiktoken)
  cli.py             `curation run|show` — live progress UI + graph visualization
  nodes/
    length.py        LengthFilter
    validate.py      ValidateToolCalls   (annotate: tool-call ↔ JSON-schema validity)
    categorize.py    ToolUseCategorizer, NumericBucketizer  (annotate: category)
    sample.py        CategorySampler     (up/down-sample by category)
    filter.py        FilterByField, DropFields  (consume annotations / clean up)
  configs/           declarative pipeline specs (YAML)
  run_example.py / run_categorize.py     scripts
  tests/             pytest suite (41 tests)
```

Dependencies are managed with **uv** (`pyproject.toml`): `uv sync` for the base
(jsonschema, pyyaml, rich) + dev (pytest); `uv sync --extra tokenizers` adds
transformers/tiktoken for the token length metric.

## Built-in nodes

| `type` (for config)   | kind      | what it does |
|-----------------------|-----------|--------------|
| `length_filter`       | filter    | keep by chars / words / messages / tokens |
| `validate_tool_calls` | annotate  | check each tool call's args vs the tool JSON-schema → `tool_calls_valid` + errors (normalizes Python-style `dict`/`float` types) |
| `categorize_tool_use` | annotate  | label by structure → `category` ∈ {chat_only, single_call, multi_turn, long_agentic} |
| `bucketize`           | annotate  | map a numeric feature (turns/tools) into named buckets |
| `category_sampler`    | resample  | per-category up/down-sample (rate <1 drops, >1 duplicates) |
| `filter_by_field`     | filter    | keep on any annotated field (is_true / equals / in / min / max) |
| `drop_fields`         | transform | strip annotation fields before output |
| `split_trajectory`    | 1→N       | split a raw trajectory into per-anchor samples (not needed for pre-split ToolMind) |
| `format_sft`          | annotate  | render train-ready SFT example via the eval harness's Qwen3 codec (`--extra harness`) |

**Annotate → act → clean** is the core composition: a classifier writes a field,
a downstream filter/sampler reads it, then `drop_fields` removes the scaffolding.

## Model

Every node is a stream transform with the same shape:

```python
class Node:
    def process(self, records: Iterable[Record]) -> Iterator[Record]: ...
```

Because the shape is uniform, nodes compose freely. A `Pipeline` is the linear
case; a `Graph` is the general DAG. Records are plain dicts (e.g. a ToolMind
example `{"conversations": [...], "tools": [...]}`), so nodes stay schema-agnostic.

Two convenience bases handle the streaming bookkeeping + stats:
- `FilterNode` — implement `keep(record) -> bool`
- `MapNode` — implement `transform(record) -> record | None` (None drops it)

## Quick start

```python
from curation import Pipeline, LengthFilter, read_jsonl, write_jsonl

pipe = Pipeline([
    LengthFilter(metric="messages", min_len=2, max_len=40),
    LengthFilter(metric="chars", max_len=60_000),
])
n = write_jsonl(pipe(read_jsonl("in.jsonl")), "out.jsonl")
print(pipe.report())   # per-node seen/emitted/dropped/drop_rate
```

Declaratively (the problem reduces to defining nodes + composing specs):

```python
import yaml
from curation import Pipeline
pipe = Pipeline.from_config(yaml.safe_load(open("configs/length_filter.yaml")))
```

## LengthFilter

Keeps records with `min_len <= length(record) <= max_len`. `metric` selects how
length is measured:

| metric       | meaning                                   | deps        |
|--------------|-------------------------------------------|-------------|
| `chars`      | total chars across message contents       | none        |
| `words`      | whitespace-delimited words                | none        |
| `messages`   | number of conversation turns              | none        |
| `tokens`     | needs a `length_fn` (see below)           | transformers / tiktoken |

For token-length filtering consistent with the **eval harness**, use the Qwen3
chat-template counter (counts tools-schema + special tokens, not just content):

```python
from curation import LengthFilter
from curation.length_metrics import qwen3_token_counter
LengthFilter(metric="tokens", max_len=8192, length_fn=qwen3_token_counter())
```

## CLI (live UI + graph visualization)

```bash
# run a pipeline with a live progress panel + per-node kept/dropped bars
curation run --config configs/curate_toolmind.yaml \
  --input ../data/toolmind_raw/graph_syn_datasets/graphsyn.jsonl \
  --output out.jsonl --limit 20000 --mermaid graph.mmd

# just draw the graph (static) without running
curation show --config configs/curate_toolmind.yaml
```

The live panel streams records through and updates seen / kept / dropped per node:

```
╭───────────────────────── curate_toolmind ──────────────────────────╮
│ node        type                seen     kept   dropped   drop      │
│ validate    ValidateToolCalls  20,000   20,000       0    ░░░ 0%    │
│ keep_valid  FilterByField      20,000   18,843   1,157    █░░ 6%    │
│ balance     CategorySampler    18,680   17,951     729    █░░ 11%   │
│ in 20,000 (100%)   out 17,951   6,370 rec/s   3.1s                  │
╰─────────────────────────────────────────────────────────────────────╯
```

## Mermaid

`Pipeline.to_mermaid()` / `Graph.to_mermaid(with_stats=True)` return a Mermaid
flowchart string (paste into a ```mermaid block, mermaid.live, or render with
`mmdc`). `--mermaid path.mmd` on the CLI writes it with live drop-rate stats baked
into each node label.

## Adding a node

```python
from curation import FilterNode, register

@register("dedup_exact")          # makes it usable from config via {"type": "dedup_exact"}
class DedupExact(FilterNode):
    def __init__(self, **kw):
        super().__init__(**kw); self._seen = set()
    def keep(self, record):
        import json; key = json.dumps(record, sort_keys=True)
        if key in self._seen: return False
        self._seen.add(key); return True
```

## Tests

```bash
cd curation && PYTHONPATH=. python -m pytest tests/ -q
```

## Run over ToolMind

```bash
PYTHONPATH=curation python curation/run_example.py \
  --input data/toolmind_raw/open_datasets/ToolACE-query.jsonl \
  --output curation/out/toolace_filtered.jsonl --limit 5000
```

## Roadmap

Next operators to add (each is a small `FilterNode`/`MapNode` + a test):
dedup (exact / near-dup), language ID, PII / safety filters, tool-call validity
(arguments match the tool JSON-schema), reasoning-quality scoring, dataset mixing
/ weighting, and a chat-template formatter node that reuses the eval harness's
codec so curated data is emitted train-ready.

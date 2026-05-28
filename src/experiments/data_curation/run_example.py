#!/usr/bin/env python3
"""Run a curation pipeline over a slice of the downloaded ToolMind data.

Example:
    .venv/bin/python curation/run_example.py \
        --input data/toolmind_raw/open_datasets/ToolACE-query.jsonl \
        --output curation/out/toolace_filtered.jsonl \
        --limit 5000
"""
from __future__ import annotations

import argparse
import itertools
import json

from curation import Pipeline, read_jsonl, write_jsonl


def build_pipeline() -> Pipeline:
    # Declarative spec — could equally be loaded from configs/length_filter.yaml.
    return Pipeline.from_config(
        {
            "name": "toolmind_length_filter",
            "nodes": [
                {"type": "length_filter", "metric": "messages", "min_len": 2, "max_len": 40},
                {"type": "length_filter", "metric": "chars", "max_len": 60_000},
            ],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap input records")
    args = ap.parse_args()

    pipe = build_pipeline()
    source = read_jsonl(args.input)
    if args.limit:
        source = itertools.islice(source, args.limit)

    written = write_jsonl(pipe(source), args.output)
    print(f"wrote {written} records -> {args.output}\n")
    print("per-node report:")
    print(json.dumps(pipe.report(), indent=2))


if __name__ == "__main__":
    main()

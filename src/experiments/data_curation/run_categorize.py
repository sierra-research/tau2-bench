#!/usr/bin/env python3
"""Categorization job: label ToolMind records and report the category distribution.

Run it to decide sampling rates, then feed those into a CategorySampler.

    uv run python run_categorize.py \
        --input ../data/toolmind_raw/graph_syn_datasets/graphsyn.jsonl \
        --limit 20000 --mermaid

Optionally writes the annotated records and a Mermaid diagram of the pipeline.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter

from curation import Pipeline, ToolUseCategorizer, read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", default=None, help="optional: write annotated jsonl")
    ap.add_argument("--mermaid", action="store_true", help="print a Mermaid diagram")
    args = ap.parse_args()

    pipe = Pipeline([ToolUseCategorizer(name="categorize")], name="categorization_job")

    src = read_jsonl(args.input)
    if args.limit:
        src = itertools.islice(src, args.limit)

    dist: Counter = Counter()
    annotated = []
    for rec in pipe(src):
        dist[rec["category"]] += 1
        if args.output:
            annotated.append(rec)

    total = sum(dist.values())
    print(f"categorized {total} records\n")
    print(f"{'category':<14}{'count':>10}{'share':>9}   suggested balance rate")
    print("-" * 60)
    target = total / max(1, len(dist))  # equal-size target
    for cat, n in dist.most_common():
        rate = round(target / n, 3) if n else 0.0
        print(f"{cat:<14}{n:>10}{n/total:>8.1%}   {rate}")

    if args.output:
        write_jsonl(annotated, args.output)
        print(f"\nwrote annotated records -> {args.output}")

    if args.mermaid:
        print("\n--- pipeline (Mermaid) ---")
        print(pipe.to_mermaid(with_stats=True))


if __name__ == "__main__":
    main()

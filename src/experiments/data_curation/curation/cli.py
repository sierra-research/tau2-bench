"""`curation` CLI — run a pipeline with a live progress UI + graph visualization.

    curation run --config configs/length_filter.yaml \
        --input ../data/toolmind_raw/open_datasets/ToolACE-query.jsonl \
        --output out.jsonl --limit 5000 [--mermaid graph.mmd]

    curation show --config configs/length_filter.yaml          # just draw the graph

The live view renders the pipeline as a flow of nodes with per-node
seen / kept / dropped counters updating as records stream through, plus a footer
with throughput. Falls back to plain output when stdout is not a TTY.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import yaml

from .core import Node, Record
from .graph import Pipeline
from .io import read_jsonl


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _bar(frac: float, width: int = 12) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _build_renderable(pipe: Pipeline, inputs: int, outputs: int, elapsed: float, total: Optional[int]):
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Group
    from rich.text import Text

    table = Table(expand=True, pad_edge=False)
    table.add_column("", width=2)  # flow glyph
    table.add_column("node", style="bold cyan", no_wrap=True)
    table.add_column("type", style="dim")
    table.add_column("seen", justify="right")
    table.add_column("kept", justify="right", style="green")
    table.add_column("dropped", justify="right", style="red")
    table.add_column("drop", justify="left")

    n = len(pipe.nodes)
    for i, node in enumerate(pipe.nodes):
        s = node.stats
        glyph = "╓" if i == 0 else ("╨" if i == n - 1 else "╫")
        table.add_row(
            glyph,
            node.name,
            type(node).__name__,
            f"{s.seen:,}",
            f"{s.emitted:,}",
            f"{s.dropped:,}",
            f"{_bar(s.drop_rate)} {s.drop_rate:.0%}",
        )

    rate = inputs / elapsed if elapsed > 0 else 0.0
    pct = f" ({inputs / total:.0%})" if total else ""
    footer = Text.assemble(
        ("in ", "dim"), (f"{inputs:,}{pct}", "bold"),
        ("   out ", "dim"), (f"{outputs:,}", "bold green"),
        ("   ", ""), (f"{rate:,.0f} rec/s", "yellow"),
        ("   ", ""), (f"{elapsed:.1f}s", "dim"),
    )
    return Panel(Group(table, footer), title=f"[bold]{pipe.name}", border_style="blue")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def load_pipeline(config_path: str) -> Pipeline:
    config = yaml.safe_load(Path(config_path).read_text())
    return Pipeline.from_config(config)


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run(
    config: str,
    input_path: str,
    output_path: Optional[str],
    limit: Optional[int] = None,
    mermaid_path: Optional[str] = None,
    refresh: int = 8,
) -> Pipeline:
    import itertools
    from rich.console import Console

    console = Console()
    pipe = load_pipeline(config)

    total = limit if limit else _count_lines(input_path)
    inputs = [0]

    def counting(it: Iterable[Record]):
        for x in it:
            inputs[0] += 1
            yield x

    src: Iterable[Record] = read_jsonl(input_path)
    if limit:
        src = itertools.islice(src, limit)
    src = counting(src)

    out_f = open(output_path, "w", encoding="utf-8") if output_path else None
    outputs = 0
    start = time.time()

    from rich.live import Live

    with Live(console=console, refresh_per_second=refresh, transient=False) as live:
        live.update(_build_renderable(pipe, 0, 0, 0.0, total))
        for rec in pipe(src):
            outputs += 1
            if out_f is not None:
                import json
                out_f.write(json.dumps(rec, ensure_ascii=False))
                out_f.write("\n")
            if outputs % 100 == 0 or inputs[0] % 500 == 0:
                live.update(_build_renderable(pipe, inputs[0], outputs, time.time() - start, total))
        live.update(_build_renderable(pipe, inputs[0], outputs, time.time() - start, total))

    if out_f is not None:
        out_f.close()
        console.print(f"[green]wrote[/] {outputs:,} records → {output_path}")

    if mermaid_path:
        Path(mermaid_path).write_text(pipe.to_mermaid(with_stats=True))
        console.print(f"[green]wrote[/] Mermaid diagram → {mermaid_path}")

    return pipe


def show(config: str) -> None:
    """Draw the pipeline graph (static) without running."""
    from rich.console import Console

    console = Console()
    pipe = load_pipeline(config)
    console.print(_build_renderable(pipe, 0, 0, 0.0, None))
    console.print("\n[dim]Mermaid:[/]")
    console.print(pipe.to_mermaid())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="curation", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a pipeline with a live UI")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--output", default=None)
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--mermaid", default=None, help="write a Mermaid .mmd diagram")

    p_show = sub.add_parser("show", help="draw the pipeline graph and exit")
    p_show.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(args.config, args.input, args.output, args.limit, args.mermaid)
    elif args.cmd == "show":
        show(args.config)


if __name__ == "__main__":
    main()

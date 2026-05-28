#!/usr/bin/env python
"""Format ToolMind-style records into Qwen3 SFT examples.

This is a thin CLI around ``tau2.utils.qwen3_codec.format_training_example`` —
the SAME canonical codec used by the eval-time ``generate()`` completions
branch. Running training data through this guarantees the model sees identical
formatting at train and eval time *by construction*.

Input: a JSONL file where each line is a ToolMind record:

    {"conversations": [ {role, content, [tool_calls]}, ... ],
     "tools": [ {"type":"function","function":{...}}, ... ]}

Each record must end on the target assistant turn (the only turn carrying
``<think>``). Output: a JSONL file with one object per input record containing
``{"text", "input_ids", "labels"}`` where ``labels`` mask (``-100``) every
token before the final ``<|im_start|>assistant`` header.

Usage:
    uv run python scripts/format_toolmind_sft.py IN.jsonl OUT.jsonl \\
        [--tokenizer Qwen/Qwen3-8B] [--no-thinking]
"""

from __future__ import annotations

import argparse
import json
import sys

from tau2.utils.qwen3_codec import DEFAULT_TOKENIZER_ID, format_training_example


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input JSONL of ToolMind records.")
    parser.add_argument("output", help="Output JSONL of formatted examples.")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_ID,
        help=f"Tokenizer id (default: {DEFAULT_TOKENIZER_ID}).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen3 thinking (inject empty <think> block).",
    )
    args = parser.parse_args(argv)

    enable_thinking = not args.no_thinking
    n = 0
    with (
        open(args.input, encoding="utf-8") as fin,
        open(args.output, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            formatted = format_training_example(
                record,
                tokenizer_id=args.tokenizer,
                enable_thinking=enable_thinking,
            )
            fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} formatted examples to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Streaming JSONL I/O for curation jobs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .core import Record


def read_jsonl(path: str | Path) -> Iterator[Record]:
    """Yield one parsed record per non-blank line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    """Write records to JSONL, returning the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n

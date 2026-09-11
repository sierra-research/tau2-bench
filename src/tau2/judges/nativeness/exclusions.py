# Copyright Sierra
"""Benchmark-authored text excluded from model-language evaluation."""

from __future__ import annotations

import re

BENCHMARK_TRANSFER_MESSAGE = (
    "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
)

_FULL_TRANSFER_RE = re.compile(
    r"YOU\s+ARE\s+BEING\s+TRANSFERRED\s+TO\s+A\s+HUMAN\s+AGENT\."
    r"\s+PLEASE\s+HOLD\s+ON\.",
    flags=re.IGNORECASE,
)
_TRANSFER_START_RE = re.compile(r"YOU\s+ARE\s+BEING", flags=re.IGNORECASE)
_STOP_MARKER_RE = re.compile(r"###(?:STOP|TRANSFER)###", flags=re.IGNORECASE)
_CANONICAL_WORDS = " ".join(re.findall(r"[A-Z]+", BENCHMARK_TRANSFER_MESSAGE))
_MINIMUM_FRAGMENT = "YOU ARE BEING T"


def _words(text: str) -> str:
    """Normalize an English transfer-message fragment for exact comparison."""
    without_markers = _STOP_MARKER_RE.sub(" ", text.upper())
    return " ".join(re.findall(r"[A-Z]+", without_markers))


def is_benchmark_transfer_fragment(text: str) -> bool:
    """Whether text contains the fixed transfer line or a delivered prefix of it."""
    normalized = _words(text)
    start = normalized.find("YOU ARE BEING")
    if start < 0:
        return False
    candidate = normalized[start:]
    return _CANONICAL_WORDS in candidate or (
        len(candidate) >= len(_MINIMUM_FRAGMENT)
        and _CANONICAL_WORDS.startswith(candidate)
    )


def without_benchmark_transfer_message(text: str, *, interrupted: bool) -> str:
    """Remove policy-mandated transfer wording before judging model language.

    Complete occurrences are removed anywhere in an utterance. When an utterance
    was interrupted, a delivered prefix at its end is removed as well.
    """
    filtered = _FULL_TRANSFER_RE.sub("", text)
    if interrupted:
        starts = list(_TRANSFER_START_RE.finditer(filtered))
        if starts:
            start = starts[-1].start()
            if is_benchmark_transfer_fragment(filtered[start:]):
                filtered = filtered[:start]
    return re.sub(r"[ \t]+", " ", filtered).strip()

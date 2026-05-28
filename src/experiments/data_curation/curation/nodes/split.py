"""SplitTrajectory — turn a multi-turn trajectory into per-anchor samples (1->N).

Reproduces ToolMind's "split at each assistant message, keep the full prior
context up to and including the anchor" decomposition. ToolMind ships *already
split*, so this is NOT needed for it — it's here for *raw* trajectories. Compose
ahead of formatting when needed:  ``[SplitTrajectory(), FormatSFT()]``.

(The Qwen3 template strips prior-turn ``<think>`` at render time, so each split
sample ends up with reasoning only on its anchor turn.)

It's a node (not a bare function) to stay consistent with the rest of the
toolkit — everything composes as a node.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

from ..core import Node, Record, register

_MESSAGE_KEYS = ("conversations", "messages", "conversation", "turns")


@register("split_trajectory")
class SplitTrajectory(Node):
    """Split each trajectory into one sample per anchor turn (default: assistant).

    For every message whose role == ``anchor_role``, emit a record whose
    conversation is the full prefix up to and including that message; other
    top-level fields (e.g. ``tools``) carry through. K anchors -> K records;
    no anchors -> the record unchanged.
    """

    def __init__(self, anchor_role: str = "assistant", name: Optional[str] = None):
        super().__init__(name=name)
        self.anchor_role = anchor_role

    def _split(self, record: Record) -> list[Record]:
        mkey = next((k for k in _MESSAGE_KEYS if record.get(k)), None)
        if mkey is None:
            return [record]
        conv = record[mkey]
        samples = [
            {**record, mkey: conv[: i + 1]}
            for i, m in enumerate(conv)
            if m.get("role") == self.anchor_role
        ]
        return samples or [record]

    def process(self, records: Iterable[Record]) -> Iterator[Record]:
        for r in records:
            self.stats.seen += 1
            for s in self._split(r):
                self.stats.emitted += 1
                yield s

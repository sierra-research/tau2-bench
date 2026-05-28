"""Sampling nodes — consume a category annotation and re-weight the stream.

`CategorySampler` is single-pass and streaming. Per category it applies a *rate*:
  * rate < 1  -> downsample: emit each record with probability `rate`
  * rate == 1 -> passthrough
  * rate > 1  -> upsample: emit floor(rate) copies, plus one more with prob frac(rate)

So rate=0.3 keeps ~30%; rate=2.5 yields 2 or 3 copies (avg 2.5). Deterministic for
a fixed `seed`. Categories not in `rates` use `default_rate`.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Iterator, Optional

from ..core import Node, Record, register


@register("category_sampler")
class CategorySampler(Node):
    """Up/down-sample records by a per-category rate read from `field`.

    Args:
        rates: {category: rate}. rate in [0, inf). <1 downsamples, >1 upsamples.
        field: record key holding the category (default "category").
        default_rate: rate for categories absent from `rates`.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        rates: dict[str, float],
        field: str = "category",
        default_rate: float = 1.0,
        seed: int = 0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if any(r < 0 for r in rates.values()) or default_rate < 0:
            raise ValueError("rates must be >= 0")
        self.rates = dict(rates)
        self.field = field
        self.default_rate = default_rate
        self._rng = random.Random(seed)

    def _copies(self, rate: float) -> int:
        base = int(rate)
        frac = rate - base
        if frac and self._rng.random() < frac:
            base += 1
        return base

    def process(self, records: Iterable[Record]) -> Iterator[Record]:
        in_by_cat: Counter = Counter()
        out_by_cat: Counter = Counter()
        for r in records:
            self.stats.seen += 1
            cat = r.get(self.field)
            rate = self.rates.get(cat, self.default_rate)
            k = self._copies(rate)
            in_by_cat[cat] += 1
            out_by_cat[cat] += k
            if k == 0:
                self.stats.dropped += 1
            for _ in range(k):
                self.stats.emitted += 1
                yield r
        self.stats.extra = {
            "in_by_category": dict(in_by_cat),
            "out_by_category": dict(out_by_cat),
        }

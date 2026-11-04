"""Contiguous timeline ranges used by acoustic metrics and ASR alignment."""

from collections.abc import Iterable

from tau2.data_model.transcript import TickRange


def contiguous_ranges(active: Iterable[bool]) -> list[TickRange]:
    """Return half-open source-index ranges, ending at each inactive tick.

    Callers select activity explicitly: acoustic metrics use contains_speech;
    ASR distribution uses caller text presence. Neither uses utterance IDs.
    """
    ranges = []
    start = None
    count = 0
    for index, present in enumerate(active):
        count = index + 1
        if present and start is None:
            start = index
        elif not present and start is not None:
            ranges.append(TickRange(start=start, end=index))
            start = None
    if start is not None:
        ranges.append(TickRange(start=start, end=count))
    return ranges

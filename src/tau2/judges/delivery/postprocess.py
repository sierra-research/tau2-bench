# Copyright Sierra
"""Deterministic post-processing for delivery-judge findings."""

import re
from typing import Optional

from tau2.config import DEFAULT_DELIVERY_FIDELITY_END_EXCLUSION_SECONDS

_TIME_TOKEN = re.compile(r"\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?")


def time_range_bounds_seconds(value: Optional[str]) -> Optional[tuple[float, float]]:
    """Parse the judge's loose ``M:SS-M:SS``/seconds span formats.

    A single timestamp is treated as a point span. Unparseable or missing
    values return None so the finding remains scored rather than disappearing.
    """
    points: list[float] = []
    for token in _TIME_TOKEN.findall(value or ""):
        if ":" in token:
            minutes, seconds = token.split(":", 1)
            points.append(60 * float(minutes) + float(seconds))
        else:
            points.append(float(token))
    if not points:
        return None
    return points[0], points[-1]


def is_in_final_utterance_window(
    time_range: Optional[str],
    clip_duration_seconds: float,
    window_seconds: float = DEFAULT_DELIVERY_FIDELITY_END_EXCLUSION_SECONDS,
) -> bool:
    """Whether an approximate finding span reaches the clip's final window.

    The one-window tolerance beyond the measured clip absorbs the judge's
    coarse timestamps. A span starting farther beyond the clip is malformed
    and remains scored rather than being silently excluded.
    """
    bounds = time_range_bounds_seconds(time_range)
    if bounds is None:
        return False
    start, end = bounds
    return (
        end >= max(0.0, clip_duration_seconds - window_seconds)
        and start <= clip_duration_seconds + window_seconds
    )

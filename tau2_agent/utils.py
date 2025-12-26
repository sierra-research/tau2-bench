"""Shared utilities for tau2_agent."""

import math
from typing import Any

# Fields to exclude from message serialization for tracing (too large for Datadog)
# Full data is preserved in EvaluationStore JSON files for debugging
_LARGE_MESSAGE_FIELDS = {"raw_data", "reasoning_content", "provider_specific_fields"}


def compact_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Create a compact version of a message for tracing output.

    Removes large fields like raw_data and reasoning_content that can
    exceed Datadog's 1MB span limit. Full data is preserved in
    EvaluationStore JSON files.

    Preserves: role, content, tool_calls, turn_idx, timestamp, cost, usage

    Args:
        msg: Full message dict from model_dump().

    Returns:
        Compact message with large fields removed.
    """
    result = {}
    for key, value in msg.items():
        if key in _LARGE_MESSAGE_FIELDS:
            continue
        if isinstance(value, dict):
            # Recursively remove large fields from nested dicts
            cleaned = {k: v for k, v in value.items() if k not in _LARGE_MESSAGE_FIELDS}
            if cleaned:
                result[key] = cleaned
        else:
            result[key] = value
    return result


def sanitize_float(value: float | None) -> float | None:
    """Convert NaN/Inf to None for JSON serialization compatibility.

    JSON does not support NaN or Infinity values. This function converts
    them to None to ensure valid JSON output.

    Args:
        value: A float value that may be NaN or Inf.

    Returns:
        The original value if valid, or None if NaN/Inf.
    """
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def sanitize_dict_floats(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize all float values in a dictionary.

    Args:
        data: Dictionary that may contain NaN/Inf float values.

    Returns:
        Dictionary with NaN/Inf values converted to None.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, float):
            result[key] = sanitize_float(value)
        elif isinstance(value, dict):
            result[key] = sanitize_dict_floats(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict_floats(item) if isinstance(item, dict)
                else sanitize_float(item) if isinstance(item, float)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result

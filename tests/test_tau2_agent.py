"""Unit tests for tau2_agent utilities."""

import math

import pytest

from tau2_agent.utils import compact_message, sanitize_dict_floats, sanitize_float


class TestSanitizeFloat:
    """Tests for sanitize_float function."""

    def test_none_returns_none(self):
        assert sanitize_float(None) is None

    def test_nan_returns_none(self):
        assert sanitize_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert sanitize_float(float("inf")) is None

    def test_neg_inf_returns_none(self):
        assert sanitize_float(float("-inf")) is None

    def test_valid_float_unchanged(self):
        assert sanitize_float(1.5) == 1.5
        assert sanitize_float(0.0) == 0.0
        assert sanitize_float(-3.14) == -3.14


class TestCompactMessage:
    """Tests for compact_message function."""

    def test_removes_raw_data(self):
        msg = {"role": "user", "content": "hello", "raw_data": {"large": "data"}}
        result = compact_message(msg)
        assert "raw_data" not in result
        assert result["role"] == "user"
        assert result["content"] == "hello"

    def test_removes_reasoning_content(self):
        msg = {
            "role": "assistant",
            "content": "reply",
            "reasoning_content": "long chain of thought",
        }
        result = compact_message(msg)
        assert "reasoning_content" not in result
        assert result["content"] == "reply"

    def test_removes_provider_specific_fields(self):
        msg = {
            "role": "user",
            "content": "test",
            "provider_specific_fields": {"refusal": None},
        }
        result = compact_message(msg)
        assert "provider_specific_fields" not in result

    def test_removes_nested_large_fields(self):
        msg = {
            "role": "user",
            "content": "test",
            "nested": {"reasoning_content": "should be removed", "keep": "this"},
        }
        result = compact_message(msg)
        assert result["nested"] == {"keep": "this"}

    def test_preserves_core_fields(self):
        msg = {
            "role": "assistant",
            "content": "hello",
            "tool_calls": [{"name": "test"}],
            "turn_idx": 1,
            "timestamp": "2025-01-01",
            "cost": 0.01,
            "usage": {"tokens": 100},
        }
        result = compact_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "hello"
        assert result["tool_calls"] == [{"name": "test"}]
        assert result["turn_idx"] == 1
        assert result["cost"] == 0.01
        assert result["usage"] == {"tokens": 100}

    def test_empty_message(self):
        result = compact_message({})
        assert result == {}

    def test_message_with_only_large_fields(self):
        msg = {"raw_data": {"x": 1}, "reasoning_content": "y"}
        result = compact_message(msg)
        assert result == {}


class TestSanitizeDictFloats:
    """Tests for sanitize_dict_floats function."""

    def test_sanitizes_top_level_floats(self):
        data = {"a": float("nan"), "b": 1.5}
        result = sanitize_dict_floats(data)
        assert result["a"] is None
        assert result["b"] == 1.5

    def test_sanitizes_nested_floats(self):
        data = {"outer": {"inner": float("inf")}}
        result = sanitize_dict_floats(data)
        assert result["outer"]["inner"] is None

    def test_sanitizes_floats_in_lists(self):
        data = {"values": [1.0, float("nan"), 2.0]}
        result = sanitize_dict_floats(data)
        assert result["values"] == [1.0, None, 2.0]

    def test_sanitizes_dicts_in_lists(self):
        data = {"items": [{"val": float("nan")}, {"val": 1.0}]}
        result = sanitize_dict_floats(data)
        assert result["items"][0]["val"] is None
        assert result["items"][1]["val"] == 1.0

    def test_preserves_non_float_values(self):
        data = {"str": "hello", "int": 42, "bool": True, "none": None}
        result = sanitize_dict_floats(data)
        assert result == data

    def test_empty_dict(self):
        result = sanitize_dict_floats({})
        assert result == {}

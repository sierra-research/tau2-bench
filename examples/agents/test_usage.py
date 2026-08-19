"""Tests for token usage and cost reporting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_report import aggregate_usage, usage_from_messages
from usage import UsageTracker, cost_usd


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.usage = _FakeUsage(input_tokens, output_tokens)


def test_usage_tracker_summary():
    tracker = UsageTracker()
    tracker.record("agent", "claude-sonnet-4-6", _FakeResponse(1000, 200))
    tracker.record("supervisor", "claude-sonnet-4-6", _FakeResponse(500, 50))
    summary = tracker.summary()
    assert summary["api_calls"] == 2
    assert summary["input_tokens"] == 1500
    assert summary["output_tokens"] == 250
    assert summary["by_component"]["agent"]["calls"] == 1
    assert summary["by_component"]["supervisor"]["calls"] == 1


def test_cost_usd_pricing():
    assert cost_usd("claude-sonnet-4-6", 1_000_000, 0) == pytest.approx(3.0)
    assert cost_usd("claude-sonnet-4-6", 0, 1_000_000) == pytest.approx(15.0)


def test_aggregate_usage_empty():
    summary = aggregate_usage([])
    assert summary["runs"] == 0
    assert summary["cost_per_resolution_usd"] == 0.0

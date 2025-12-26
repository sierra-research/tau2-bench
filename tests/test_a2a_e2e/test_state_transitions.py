"""E2E tests for evaluation state transition validation.

Verifies state machine:
    submitted -> working -> completed|failed|abandoned
"""

import pytest

pytestmark = pytest.mark.a2a_e2e

VALID_TRANSITIONS = {
    "submitted": {"working", "failed", "abandoned"},
    "working": {"completed", "failed", "abandoned"},
    "completed": set(),
    "failed": set(),
    "abandoned": set(),
}
TERMINAL_STATES = {"completed", "failed", "abandoned"}
INITIAL_STATES = {"submitted", "working"}


def test_state_history_structure(evaluation_data: dict):
    """Verify state_history exists with timestamps."""
    assert "state_history" in evaluation_data, (
        f"Missing state_history. Keys: {list(evaluation_data.keys())}"
    )

    history = evaluation_data["state_history"]
    assert len(history) > 0, "state_history should have entries"

    for entry in history:
        assert "state" in entry, f"Entry missing state: {entry}"
        has_time = "timestamp" in entry or "time" in entry or "at" in entry
        assert has_time, f"Entry missing timestamp: {entry}"


def test_state_transitions_valid(evaluation_data: dict):
    """Verify transitions follow state machine, start initial, end terminal."""
    history = evaluation_data.get("state_history", [])
    assert len(history) > 0, "Should have state history"

    states = [entry.get("state") for entry in history]

    # First state must be initial
    assert states[0] in INITIAL_STATES, (
        f"First state {states[0]} not in {INITIAL_STATES}"
    )

    # Last state must be terminal
    assert states[-1] in TERMINAL_STATES, (
        f"Final state {states[-1]} not in {TERMINAL_STATES}"
    )

    # All transitions must be valid
    for i in range(len(states) - 1):
        current, next_state = states[i], states[i + 1]
        valid = VALID_TRANSITIONS.get(current, set())
        assert next_state in valid, (
            f"Invalid: {current} -> {next_state}. Valid: {valid}"
        )

    # Status field must match final state
    status = evaluation_data.get("status")
    assert status == states[-1], f"status={status} != final_state={states[-1]}"

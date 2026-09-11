# Copyright Sierra
"""The wallclock guard must never pre-empt a run's conversation budget.

``max_steps_seconds`` is simulated time, so it caps every call identically no
matter how slow the providers are. Wallclock is not: before this guard was
anchored to the budget, a flat 2400s cap truncated 15/200 pt telecom
simulations and 1/400 English ones — same configuration, different latency —
and every truncation scored 0.0 on a conversation that was still running.
"""

import argparse

import pytest

from tau2.cli import add_run_args
from tau2.config import (
    DEFAULT_MATRIX_MAX_STEPS_SECONDS,
    DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    VOICE_TIMEOUT_SAFETY_FACTOR,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.runner.helpers import resolve_timeout

# Cost of a voice call in wallclock seconds, fitted over the 629 completed
# simulations of the 2026-07-26 en+pt telecom preference pools:
#   wall ~= -108 + 2.54 * sim_seconds, max residual +557s
# The intercept is dropped (conservative) so this bounds from above.
WALL_SECONDS_PER_SIM_SECOND = 2.54
WALL_SECONDS_WORST_RESIDUAL = 557.0


def _voice(max_steps_seconds: int) -> AudioNativeConfig:
    return AudioNativeConfig(max_steps_seconds=max_steps_seconds)


@pytest.mark.parametrize("budget", [600, 1200, 1800, 3600])
def test_voice_guard_exceeds_the_cost_of_spending_the_whole_budget(budget):
    """The headline invariant: a call that runs its full budget must finish.

    If this fails, the guard fires before the conversation cap does and the
    benchmark silently measures provider latency instead of agent behaviour.
    """
    guard = resolve_timeout(None, _voice(budget))
    worst_case_wall = WALL_SECONDS_PER_SIM_SECOND * budget + WALL_SECONDS_WORST_RESIDUAL
    assert guard > worst_case_wall, (
        f"a {budget}s conversation costs up to {worst_case_wall:.0f}s wall, "
        f"but the guard fires at {guard:.0f}s"
    )


def test_voice_guard_tracks_the_budget_it_guards():
    """Doubling the conversation budget doubles the guard, not just at 1200s."""
    assert resolve_timeout(None, _voice(1200)) == 1200 * VOICE_TIMEOUT_SAFETY_FACTOR
    assert resolve_timeout(None, _voice(2400)) == 2400 * VOICE_TIMEOUT_SAFETY_FACTOR


def test_text_runs_keep_the_flat_default():
    """max_steps counts turns in text mode, so wallclock is its only duration bound."""
    assert resolve_timeout(None, None) == DEFAULT_TIMEOUT_SECONDS


def test_explicit_value_wins_in_both_modes():
    assert resolve_timeout(900.0, _voice(1200)) == 900.0
    assert resolve_timeout(900.0, None) == 900.0


def test_zero_is_the_opt_out_not_an_instant_timeout():
    """None means "no timeout"; a literal 0 would time every simulation out at once."""
    assert resolve_timeout(0.0, _voice(1200)) is None
    assert resolve_timeout(0.0, None) is None


def test_cli_leaves_timeout_unset_so_the_mode_decides():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    assert parser.parse_args([]).timeout is None


def test_cli_leaves_max_steps_seconds_unset_until_run_config_resolution():
    """An explicit flag remains distinguishable from the default."""
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    assert parser.parse_args([]).max_steps_seconds is None
    assert parser.parse_args(["--max-steps-seconds", "600"]).max_steps_seconds == 600


def test_multilingual_preset_timeout_is_anchored_to_the_matrix_budget():
    """The preset passes --timeout explicitly, so it needs the same anchoring."""
    assert DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS == int(
        DEFAULT_MATRIX_MAX_STEPS_SECONDS * VOICE_TIMEOUT_SAFETY_FACTOR
    )
    assert DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS >= resolve_timeout(
        None, _voice(DEFAULT_MATRIX_MAX_STEPS_SECONDS)
    )

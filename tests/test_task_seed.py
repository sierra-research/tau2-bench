# Copyright Sierra
"""Regression guard for the per-task sampling-seed derivation.

The per-task seed must come from a STABLE digest of the task id: Python's
``hash()`` is salted per process, so a hash()-derived seed silently resampled
acoustic environments, noise-file choices, and derived voice seeds on every
process despite a fixed ``--seed``. The pinned constants below fail loudly if
the derivation ever changes (e.g. a revert to ``hash()``).
"""

from tau2.user_simulation_voice_presets import derive_task_seed


def test_derived_seed_is_a_pinned_constant():
    # int.from_bytes(sha256(task_id)[:4], "big") % 1_000_000 — process-stable.
    assert derive_task_seed(0, "3_hi") == 605410
    assert derive_task_seed(42, "3_hi") == 42 + 605410
    assert derive_task_seed(0, "42") == 56628


def test_seed_varies_with_task_id_and_shifts_with_base_seed():
    assert derive_task_seed(0, "3_hi") != derive_task_seed(0, "4_hi")
    assert derive_task_seed(1, "3_hi") == derive_task_seed(0, "3_hi") + 1

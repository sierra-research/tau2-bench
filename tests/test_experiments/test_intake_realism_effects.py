from pathlib import Path

import numpy as np
import pytest

from src.experiments.intake.realism_effects import (
    REPORTED_KINDS,
    _effect_for_indices,
    _groups,
    _holm,
    _spelling_opportunity,
    load_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def realism_matrix():
    return load_matrix(REPO_ROOT)


def test_holm_adjustment_is_monotone_in_rank() -> None:
    assert _holm([0.04, 0.01, 0.20]) == pytest.approx([0.08, 0.03, 0.20])


def test_frozen_spelling_request_counts() -> None:
    rows = _spelling_opportunity(REPO_ROOT)

    assert [row.assigned_calls for row in rows] == [524, 488]
    assert [row.calls_with_spelling_request for row in rows] == [146, 108]


@pytest.mark.parametrize(
    (
        "kind_index",
        "effect_points",
        "assigned_n",
        "assigned_ess",
        "clean_n",
        "clean_ess",
    ),
    [
        (None, -3.5879775, 326, 291.4, 166, 135.5),
        (0, -11.0917074, 29, 26.0, 166, 135.5),
        (1, -8.7140564, 94, 89.2, 65, 61.9),
        (2, -4.3167222, 50, 45.6, 166, 135.5),
        (3, -0.4955498, 49, 43.0, 18, 15.2),
        (4, 2.0354417, 99, 93.4, 166, 135.5),
    ],
)
def test_frozen_realism_estimates_match_figure(
    realism_matrix,
    kind_index: int | None,
    effect_points: float,
    assigned_n: int,
    assigned_ess: float,
    clean_n: int,
    clean_ess: float,
) -> None:
    matrix = realism_matrix
    kind = None if kind_index is None else REPORTED_KINDS[kind_index]

    effect, environment_effects = _effect_for_indices(
        matrix, np.arange(len(matrix.task_ids)), kind
    )
    assigned, clean = _groups(matrix, kind)

    assert 100 * effect == pytest.approx(effect_points, abs=1e-6)
    assert len(environment_effects) == 3
    assert assigned.units == assigned_n
    assert assigned.effective_n == pytest.approx(assigned_ess, abs=0.05)
    assert clean.units == clean_n
    assert clean.effective_n == pytest.approx(clean_ess, abs=0.05)

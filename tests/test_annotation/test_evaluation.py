# Copyright Sierra
"""The LIVE/shadow gate and chance-corrected agreement."""

import numpy as np
import pytest

from tau2.annotation.evaluation import cohens_kappa, cohens_kappa_from_matrix, gate


def test_gate_defaults_to_shadow_without_a_precision():
    assert gate(None) == "shadow"
    assert gate(0.99) == "LIVE"
    assert gate(0.1) == "shadow"


def test_kappa_generalizes_past_two_categories():
    """Three-answer tasks make the confusion matrix 3x3; the 2x2 form stays a
    special case."""
    perfect = np.diag([10.0, 10.0, 10.0])
    assert cohens_kappa_from_matrix(perfect) == pytest.approx(1.0)
    chance = np.full((3, 3), 5.0)
    assert cohens_kappa_from_matrix(chance) == pytest.approx(0.0)
    assert cohens_kappa_from_matrix(np.zeros((3, 3))) is None
    # The 2x2 helper delegates, so the two agree where they overlap.
    assert cohens_kappa(50, 0, 0, 50) == pytest.approx(
        cohens_kappa_from_matrix(np.array([[50.0, 0.0], [0.0, 50.0]]))
    )

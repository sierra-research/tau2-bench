from pathlib import Path

import numpy as np
import pytest

from src.experiments.intake.realism_effects import (
    REPORTED_KINDS,
    Arm,
    _effect_for_indices,
    _groups,
    _holm,
    _load_event_ledger,
    _repair_cost_diagnostic,
    _spelling_opportunity,
    load_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def realism_matrix():
    return load_matrix(REPO_ROOT, arm=Arm.AGENT_DIRECTED)


@pytest.fixture(scope="module")
def scaffolded_realism_matrix():
    return load_matrix(REPO_ROOT, arm=Arm.SCAFFOLDED)


def test_holm_adjustment_is_monotone_in_rank() -> None:
    assert _holm([0.04, 0.01, 0.20]) == pytest.approx([0.08, 0.03, 0.20])


def test_frozen_spelling_event_counts() -> None:
    event_rows, _ = _load_event_ledger(REPO_ROOT)
    rows = _spelling_opportunity(event_rows)

    assert [row.assigned_calls for row in rows] == [524, 488]
    assert [row.calls_with_spelling_event for row in rows] == [301, 221]
    assert [row.calls_with_realism_event for row in rows] == [None, 101]


def test_mispronunciation_repair_cost_matches_paper(realism_matrix) -> None:
    event_rows, _ = _load_event_ledger(REPO_ROOT)
    diagnostic = _repair_cost_diagnostic(realism_matrix, event_rows)

    assert diagnostic.spelling_request_effect_points == pytest.approx(
        24.0495084, abs=1e-6
    )
    assert diagnostic.duration_effect_seconds == pytest.approx(26.3236502, abs=1e-6)


def test_compact_loader_validates_both_frozen_arms(
    realism_matrix, scaffolded_realism_matrix
) -> None:
    for matrix, arm, applied, catalogs in (
        (realism_matrix, Arm.AGENT_DIRECTED, 60, {"2.4.0"}),
        (scaffolded_realism_matrix, Arm.SCAFFOLDED, 19, {"2.3.0", "2.4.0"}),
    ):
        assert matrix.outcomes.shape == (200, 3, 4)
        assert len(matrix.sources) == 12
        assert {source.arm for source in matrix.sources} == {arm}
        assert {
            source.complication_catalog_version for source in matrix.sources
        } == catalogs
        assert all(source.transcript_sha256 for source in matrix.sources)
        assert all(source.run_config_sha256 for source in matrix.sources)
        assert matrix.scoring_correction.total_corrected_calls == 80
        assert matrix.scoring_correction.applied_corrected_calls == applied


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
        (None, -2.0676618, 326, 291.4, 166, 135.5),
        (0, -3.4646104, 29, 26.0, 166, 135.5),
        (1, -6.1926355, 94, 89.2, 65, 61.9),
        (2, -3.0566893, 50, 45.6, 166, 135.5),
        (3, -3.0367054, 49, 43.0, 18, 15.2),
        (4, 3.2130202, 99, 93.4, 166, 135.5),
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


@pytest.mark.parametrize(
    ("kind_index", "effect_points"),
    [
        (None, -4.1837220),
        (0, 1.7111710),
        (1, -7.2066781),
        (2, -0.3721623),
        (3, 0.4656526),
    ],
)
def test_scaffolded_realism_estimates_match_corrected_compact_archive(
    scaffolded_realism_matrix,
    kind_index: int | None,
    effect_points: float,
) -> None:
    kind = None if kind_index is None else REPORTED_KINDS[kind_index]
    effect, environment_effects = _effect_for_indices(
        scaffolded_realism_matrix,
        np.arange(len(scaffolded_realism_matrix.task_ids)),
        kind,
    )

    assert 100 * effect == pytest.approx(effect_points, abs=1e-6)
    assert len(environment_effects) == 3

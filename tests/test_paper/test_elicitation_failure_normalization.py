"""Guard the release-safe projection of scoring-normalization failures."""

from collections import Counter
from pathlib import Path

import pytest

from tau2.paper.elicitation import (
    HUMAN_FAILURE_VALIDATION,
    _read_validation_artifacts,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOTS = (
    ROOT / "papers/tau-intake/v1/reproduction/judge_validation",
    ROOT / "data/simulations/paper_runs/tau-elicit/judge_validation",
)
SCORING_NORMALIZATION_IDS = {
    "c54d5a29-bcbb-4a2f-b531-41f1295c3832",
    "8c7bf50c-7ae4-40fb-a90f-08c22c568ed7",
    "b58cf8e6-8a17-43b7-a17f-75264ca04e62",
    "37b66d64-d463-4e22-9eca-fb50ad00ccd3",
}


@pytest.mark.parametrize("validation_root", VALIDATION_ROOTS)
def test_scoring_normalization_projection_is_exact(
    validation_root: Path,
) -> None:
    human_rows, metrics, _, _ = _read_validation_artifacts(validation_root)
    system_rows = {
        row.simulation_id: row for row in human_rows if row.error_source == "system"
    }

    assert system_rows.keys() == SCORING_NORMALIZATION_IDS
    assert all(
        row.error_subtype == "scoring_normalization" for row in system_rows.values()
    )
    assert all(row.reward == 0 for row in system_rows.values())
    assert Counter(row.error_source for row in human_rows) == {
        "agent": 81,
        "user": 1,
        "system": 4,
        "unresolved": 4,
    }
    assert metrics.unit == "original_reward_zero_call"
    assert metrics.error_source.agent.n == 81
    assert metrics.error_source.user.n == 1
    assert metrics.error_source.system.n == 4
    assert metrics.error_source.no_error.n == 0
    assert metrics.error_source.unresolved.n == 4


def test_failure_validation_bundles_remain_identical() -> None:
    paper, frozen = (root / HUMAN_FAILURE_VALIDATION for root in VALIDATION_ROOTS)
    for filename in ("calls.csv", "metrics.json", "README.md"):
        assert (paper / filename).read_bytes() == (frozen / filename).read_bytes()

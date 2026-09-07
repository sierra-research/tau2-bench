from pathlib import Path

import pytest

from src.experiments.intake.caller_voice_significance import (
    System,
    _holm,
    analyze,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def artifact():
    return analyze(REPO_ROOT)


def test_holm_adjustment_is_monotone_in_rank() -> None:
    assert _holm([0.04, 0.01, 0.20]) == pytest.approx([0.08, 0.03, 0.20])


def test_frozen_voice_tables_match_paper(artifact) -> None:
    expected = {
        System.GPT_XHIGH: {
            "mildred_kaplan": (28, 19),
            "wei_lin": (19, 20),
            "priya_patil": (19, 28),
            "mamadou_diallo": (11, 20),
            "arjun_roy": (12, 24),
        },
        System.GEMINI_HIGH: {
            "mildred_kaplan": (31, 17),
            "wei_lin": (24, 11),
            "priya_patil": (18, 26),
            "mamadou_diallo": (14, 20),
            "arjun_roy": (16, 23),
        },
        System.GROK: {
            "mildred_kaplan": (38, 13),
            "wei_lin": (21, 14),
            "priya_patil": (31, 16),
            "mamadou_diallo": (20, 13),
            "arjun_roy": (23, 11),
        },
    }
    for result in artifact.omnibus:
        observed = {row.voice: (row.successes, row.failures) for row in result.table}
        assert observed == expected[result.system]


def test_frozen_omnibus_values_match_paper_exactly(artifact) -> None:
    by_system = {result.system: result for result in artifact.omnibus}
    assert by_system[System.GPT_XHIGH].monte_carlo_p == pytest.approx(
        0.1045589544104559, abs=1e-15
    )
    assert by_system[System.GPT_XHIGH].holm_p == pytest.approx(
        0.2091179088209118, abs=1e-15
    )
    assert by_system[System.GEMINI_HIGH].monte_carlo_p == pytest.approx(
        0.013369866301336986, abs=1e-15
    )
    assert by_system[System.GEMINI_HIGH].holm_p == pytest.approx(
        0.04010959890401096, abs=1e-15
    )
    assert by_system[System.GROK].monte_carlo_p == pytest.approx(
        0.6043639563604364, abs=1e-15
    )
    assert by_system[System.GROK].holm_p == pytest.approx(0.6043639563604364, abs=1e-15)


def test_gemini_pairwise_values_match_paper(artifact) -> None:
    assert len(artifact.gemini_pairwise) == 10
    assert min(row.holm_p for row in artifact.gemini_pairwise) == pytest.approx(
        0.21202736015196272, abs=1e-15
    )
    assert all(row.holm_p > 0.05 for row in artifact.gemini_pairwise)

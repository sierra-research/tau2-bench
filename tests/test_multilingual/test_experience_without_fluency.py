from experiments.tau_multilingual.experience_without_fluency import (
    _without_benchmark_transfer_findings,
)
from tau2.judges.nativeness.exclusions import BENCHMARK_TRANSFER_MESSAGE


def test_transfer_only_utterance_finding_becomes_pass():
    checks = [
        {
            "id": "natural_word_choice",
            "outcome": "fail",
            "unit_results": [
                {
                    "unit_index": 7,
                    "opportunity": True,
                    "violated": True,
                    "severity": 2,
                    "reasoning": "The fixed English transfer line is untranslated.",
                    "quote": BENCHMARK_TRANSFER_MESSAGE,
                }
            ],
        }
    ]

    filtered, exclusions = _without_benchmark_transfer_findings(checks)

    assert exclusions == [("natural_word_choice", 7)]
    assert filtered[0]["outcome"] == "pass"
    assert filtered[0]["unit_results"] == []
    assert filtered[0]["violation_count"] == 0

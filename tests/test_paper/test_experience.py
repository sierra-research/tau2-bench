# Copyright Sierra
"""Checks the frozen task and experience reproduction artifact."""

import hashlib
import json
from pathlib import Path

from tau2.judges.nativeness.paper_trial import CANONICAL_EXPERIENCE_SHA256

REPO = Path(__file__).resolve().parents[2]
EXPERIENCE = REPO / "papers/tau-multilingual/reproduction/experience.json"
SYSTEMS = (
    "OpenAI minimal",
    "OpenAI xhigh",
    "Gemini minimal",
    "Gemini high",
    "xAI",
)


def _display(values: list[float], digits: int = 0) -> list[float | int]:
    return [round(value, digits) for value in values]


def test_canonical_experience_hash_matches_artifact():
    assert hashlib.sha256(EXPERIENCE.read_bytes()).hexdigest() == (
        CANONICAL_EXPERIENCE_SHA256
    )


def test_task_and_experience_reproduction_artifact():
    payload = json.loads(EXPERIENCE.read_text())["descriptive_complete_cohort"]
    providers = payload["provider"]

    def values(path: tuple[str, ...]) -> list[float]:
        rows = []
        for system in SYSTEMS:
            value = providers[system]
            for key in path:
                value = value[key]
            rows.append(float(value))
        return rows

    expected = {
        "latency": [1.1, 1.4, 1.4, 1.8, 1.2, 1.4],
        "interaction": [55, 59, 58, 59, 57, 57],
        "nonresponse": [9, 28, 47, 77, 10, 34],
        "interruption": [80, 90, 74, 79, 90, 83],
        "selectivity": [97, 96, 84, 77, 97, 90],
        "monologue": [18, 18, 7, 11, 37, 18],
        "tool_use": [70, 62, 77, 53, 51, 62],
        "experience": [65, 67, 60, 58, 52, 60],
        "fluency": [32, 30, 37, 38, 44, 36],
        "speech_fidelity": [7, 7, 8, 9, 21, 10],
    }

    observed: dict[str, list[float | int]] = {}
    for label, path in {
        "latency": ("latency_seconds",),
        "interaction": ("interaction_failure",),
        "nonresponse": ("interaction_components", "nonresponse"),
        "interruption": ("interaction_components", "interruption"),
        "selectivity": ("interaction_components", "selectivity"),
        "monologue": ("interaction_components", "monologue"),
        "tool_use": ("interaction_components", "tool_use"),
        "experience": ("experience",),
        "fluency": ("fluency_failure",),
        "speech_fidelity": ("speech_fidelity_failure",),
    }.items():
        row = values(path)
        digits = 1 if label == "latency" else 0
        observed[label] = _display([*row, sum(row) / len(row)], digits=digits)

    assert observed == expected

# Copyright Sierra
"""Tests for reviewer-safe annotation exports."""

import json

from tau2.paper.annotations import ANNOTATION_LANGUAGES, export_annotations


def test_export_contains_only_neutral_final_precision_and_recall_files(tmp_path):
    labels = tmp_path / "final-labels.json"
    evidence = tmp_path / "evidence"
    out = tmp_path / "reviewer"
    records = []
    for language in ANNOTATION_LANGUAGES:
        sim_id = f"{language}-sim"
        sim_dir = evidence / language / "cell" / "simulations"
        sim_dir.mkdir(parents=True)
        (sim_dir / f"{sim_id}.json").write_text(
            json.dumps({"id": sim_id, "quality_info": None})
        )
        (evidence / language / f"conversation_{language}.json").write_text(
            json.dumps(
                {
                    "calls": [
                        {
                            "sim_id": sim_id,
                            "shadow_scores": {
                                "ours": {
                                    "factor_checks": [
                                        {"id": "tool_error", "outcome": "fail"}
                                    ]
                                }
                            },
                        }
                    ],
                }
            )
        )
        records.append(
            {
                "lang": language,
                "clip_id": "clip_001",
                "sim_id": sim_id,
                "judge": "quality",
                "factor_id": "tool_error",
                "label": "yes",
            }
        )
    labels.write_text(json.dumps({"records": records}))

    manifest = export_annotations(labels, evidence, out)

    assert len(manifest.files) == 2 * len(ANNOTATION_LANGUAGES)
    assert {entry.annotation_type for entry in manifest.files} == {
        "precision",
        "recall",
    }
    rendered = "\n".join(path.read_text() for path in out.rglob("*.json")).lower()
    for forbidden in ("blind", "visible", "calibrat", "adjudicat", "private person"):
        assert forbidden not in rendered
    assert (
        json.loads((out / "precision" / "es.json").read_text())["metrics"][0][
            "precision"
        ]
        == 1.0
    )
    assert (
        json.loads((out / "recall" / "es.json").read_text())["metrics"][0]["recall"]
        == 1.0
    )

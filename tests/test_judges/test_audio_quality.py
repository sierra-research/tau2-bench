# Copyright Sierra
"""Shared audio-quality input and human-label contracts."""

import csv

from tau2.judges.audio_quality import (
    CURRENT_DIMENSION_IDS,
    load_human_audio_quality_ratings,
)


def test_load_human_csv_rejects_noncanonical_factor_schema(tmp_path):
    path = tmp_path / "human.csv"
    headers = [
        "clip_id",
        "rater",
        "completed",
        "created_at",
        "word_pronunciation",
        "names_entities",
        "letter_names",
        "digits_numbers",
        "native_prosody",
        "accent_anglicization",
        "garbling_intelligibility",
        "punctuation_formatting",
        "borrowed_native_terms",
        "audio_language_switching",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "clip_001",
                "rater": "rater",
                "completed": "true",
                "created_at": "2026-08-13T00:00:00Z",
                "word_pronunciation": "1",
                "names_entities": "NA",
                "letter_names": "1",
                "digits_numbers": "2",
                "native_prosody": "3",
                "accent_anglicization": "0",
                "garbling_intelligibility": "0",
                "punctuation_formatting": "0",
                "borrowed_native_terms": "1",
                "audio_language_switching": "3",
            }
        )

    try:
        load_human_audio_quality_ratings(path)
    except ValueError as exc:
        assert "canonical audio-quality schema" in str(exc)
    else:
        raise AssertionError("noncanonical human schema was accepted")


def test_load_current_human_csv_preserves_all_factors(tmp_path):
    path = tmp_path / "human-v3.csv"
    headers = ["clip_id", "rater", "completed", *CURRENT_DIMENSION_IDS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "clip_001",
                "rater": "rater",
                "completed": "true",
                **{dimension: "0" for dimension in CURRENT_DIMENSION_IDS},
                "word_substitution": "2",
                "intonation": "1",
            }
        )

    rows = load_human_audio_quality_ratings(path)

    assert rows[0].ratings["word_substitution"] == 2
    assert rows[0].ratings["intonation"] == 1

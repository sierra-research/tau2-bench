# Copyright Sierra
"""Shared packet input and human-label contracts for audio-quality judging.

The canonical delivery judge is the per-utterance judge in
:mod:`tau2.judges.delivery`. This module deliberately contains no alternate
prompt or scoring path — it owns the packet input and human-label contracts
only.
"""

import csv
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from tau2.annotation.artifacts import ArtifactManifest, AudioQualityEntry
from tau2.annotation.packets.audio_quality import agent_speech_transcript
from tau2.annotation.packets.forms import AUDIO_QUALITY_KIND
from tau2.data_model.simulation import Results, SimulationRun
from tau2.utils.utils import DATA_DIR


class AudioQualityInput(BaseModel):
    """Resolved packet audio and matching ordered agent transcript."""

    entry: AudioQualityEntry
    audio_path: Path
    transcript: list[str]


def _resolve_results_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else DATA_DIR.parent / path


def _load_source_sim(results_path: Path, sim_id: str) -> SimulationRun:
    """Load one selected simulation without materializing the rest of its run."""
    results_dir = (
        results_path.parent if results_path.suffix == ".json" else results_path
    )
    direct = results_dir / "simulations" / f"{sim_id}.json"
    if direct.is_file():
        return SimulationRun.model_validate_json(direct.read_text())
    for sim in Results.iter_simulations(results_path):
        if sim.id == sim_id:
            return sim
    raise ValueError(f"sim {sim_id} missing from source results {results_path}")


def load_audio_quality_inputs(
    manifest_path: Path,
    *,
    sim_ids: Optional[list[str]] = None,
) -> tuple[ArtifactManifest, list[AudioQualityInput]]:
    """Resolve packet audio and matching agent transcripts from source results."""
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    entries = manifest.audio_quality_entries or []
    if manifest.kind != AUDIO_QUALITY_KIND or not entries:
        raise ValueError("manifest must be a non-empty audio-quality packet")
    if sim_ids is not None:
        requested = set(sim_ids)
        entries = [entry for entry in entries if entry.sim_id in requested]
        missing = sorted(requested - {entry.sim_id for entry in entries})
        if missing:
            raise ValueError(f"requested sim ids are not in packet: {missing}")

    units: list[AudioQualityInput] = []
    for entry in entries:
        audio_path = manifest_path.parent / entry.audio_file
        if not audio_path.is_file():
            raise FileNotFoundError(f"packet audio missing: {audio_path}")
        sim = _load_source_sim(_resolve_results_path(entry.results_path), entry.sim_id)
        transcript = agent_speech_transcript(sim)
        if not transcript:
            raise ValueError(f"sim {entry.sim_id} has no delivered agent transcript")
        units.append(
            AudioQualityInput(
                entry=entry,
                audio_path=audio_path,
                transcript=transcript,
            )
        )
    return manifest, units


class HumanAudioQualityRating(BaseModel):
    """One human row normalized into the current fidelity/intonation taxonomy."""

    clip_id: str
    rater: str
    ratings: dict[str, Optional[int]]


CURRENT_DIMENSION_IDS = (
    "mispronunciation",
    "word_substitution",
    "missing_word",
    "extra_or_hallucinated_word",
    "number_date_currency",
    "email_url_code",
    "punctuation_or_formatting",
    "acronym_brand_name",
    "clipped_or_garbled",
    "other",
    "intonation",
)


def _rating_value(value: str) -> Optional[int]:
    normalized = value.strip().upper()
    if normalized in {"", "NA"}:
        return None
    if normalized not in {"0", "1", "2", "3"}:
        raise ValueError(f"invalid audio-quality rating {value!r}")
    return int(normalized)


def load_human_audio_quality_ratings(
    csv_path: Path,
) -> list[HumanAudioQualityRating]:
    """Load completed rows from the canonical audio-quality CSV contract."""
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    headers = set(rows[0]) if rows else set()
    required = {"clip_id", "rater", "completed", *CURRENT_DIMENSION_IDS}
    missing = sorted(required - headers)
    if missing:
        raise ValueError(
            "human CSV does not match the canonical audio-quality schema; "
            f"missing columns: {missing}"
        )

    latest: dict[tuple[str, str], tuple[tuple[str, int], HumanAudioQualityRating]] = {}
    for index, row in enumerate(rows):
        if row.get("completed", "").strip().lower() != "true":
            continue
        ratings = {
            dimension: _rating_value(row.get(dimension, ""))
            for dimension in CURRENT_DIMENSION_IDS
        }
        rating = HumanAudioQualityRating(
            clip_id=row.get("clip_id", "").strip(),
            rater=row.get("rater", "").strip(),
            ratings=ratings,
        )
        key = (rating.rater, rating.clip_id)
        stamp = (row.get("created_at", "").strip(), index)
        if key not in latest or stamp >= latest[key][0]:
            latest[key] = (stamp, rating)
    return [item[1] for item in latest.values()]

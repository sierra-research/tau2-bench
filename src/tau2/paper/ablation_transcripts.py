# Copyright Sierra
"""Compact, reproducible transcript export for the retail localization ablation.

The frozen voice simulations are intentionally distributed outside Git because
their tick and audio payloads are large.  This module extracts the delivered
caller/agent turns and the recorded task outcome for every call used by the
paper's retail localization ablation, while binding each row to its source
simulation and result metadata by SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.simulation import Results, SimulationIndexEntry, SimulationRun
from tau2.metrics.interaction_quality import extract_spoken_turns
from tau2.paper.multilingual import (
    _canonical_task_id,
    _retail_ablation_path,
    _voice_result_path,
)
from tau2.utils import DATA_DIR

EXPORT_VERSION = "tau-multilingual-retail-ablation-transcripts-v1"
LANGUAGES = ("hi", "zh")
SYSTEMS = ("openai_xhigh", "gemini_high")
CONDITIONS = ("baseline", "source_entities", "native_script_entities_and_db")

Language = Literal["hi", "zh"]
System = Literal["openai_xhigh", "gemini_high"]
Condition = Literal["baseline", "source_entities", "native_script_entities_and_db"]
TableColumn = Literal["localized", "source", "romanized", "native"]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


class TranscriptTurn(BaseModel):
    """One delivered spoken turn in call order."""

    model_config = ConfigDict(frozen=True)

    turn_index: Annotated[
        int, Field(description="Zero-based position in the delivered transcript.")
    ]
    speaker: Annotated[
        Literal["caller", "agent"], Field(description="Speaker of the turn.")
    ]
    text: Annotated[str, Field(description="Delivered transcript text.")]
    interrupted: Annotated[
        bool,
        Field(description="Whether the other participant cut this turn short."),
    ] = False


class AblationTranscript(BaseModel):
    """One compact call record from the retail localization ablation."""

    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[
        int, Field(description="Serialized row schema version.")
    ] = 1
    language: Annotated[Language, Field(description="Language under evaluation.")]
    system: Annotated[System, Field(description="Voice-system arm.")]
    condition: Annotated[
        Condition, Field(description="Localization condition that produced the call.")
    ]
    table_columns: Annotated[
        list[TableColumn],
        Field(description="Paper table columns supported by this condition."),
    ]
    simulation_id: Annotated[str, Field(description="Frozen simulation identifier.")]
    task_id: Annotated[str, Field(description="Recorded localized task identifier.")]
    canonical_task_id: Annotated[
        str, Field(description="Language-neutral fixed-subset task identifier.")
    ]
    trial: Annotated[int, Field(description="Recorded trial index.")]
    reward: Annotated[
        float | None, Field(description="Recorded task reward, when available.")
    ]
    termination_reason: Annotated[
        str, Field(description="Recorded simulation termination reason.")
    ]
    source_results_path: Annotated[
        str, Field(description="Results path relative to the evidence root.")
    ]
    source_results_sha256: Annotated[
        str, Field(description="SHA-256 of the source results metadata.")
    ]
    source_simulation_sha256: Annotated[
        str, Field(description="SHA-256 of the source simulation payload.")
    ]
    turns: Annotated[
        list[TranscriptTurn], Field(description="Delivered caller and agent turns.")
    ]


class SourceRun(BaseModel):
    """Provenance and selection count for one frozen result file."""

    model_config = ConfigDict(frozen=True)

    language: Annotated[Language, Field(description="Language under evaluation.")]
    system: Annotated[System, Field(description="Voice-system arm.")]
    condition: Annotated[Condition, Field(description="Localization condition.")]
    results_path: Annotated[
        str, Field(description="Path relative to the evidence root.")
    ]
    results_sha256: Annotated[
        str, Field(description="SHA-256 of the source results metadata.")
    ]
    selected_rows: Annotated[
        int, Field(description="Number of fixed-frame trial-0 calls exported.")
    ]
    pass_at_1_pct: Annotated[
        float, Field(description="Mean recorded binary reward as a percentage.")
    ]


class AblationTranscriptManifest(BaseModel):
    """Provenance contract for the compact transcript export."""

    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[
        int, Field(description="Serialized manifest schema version.")
    ] = 1
    export_version: Annotated[
        str, Field(description="Versioned deterministic export contract.")
    ] = EXPORT_VERSION
    evidence_root_name: Annotated[
        str, Field(description="Logical name of the frozen evidence root.")
    ]
    task_subset: Annotated[
        str, Field(description="Fixed task-subset name used by the ablation.")
    ] = "retail_30"
    task_subset_sha256: Annotated[
        str, Field(description="SHA-256 of the fixed task-subset definition.")
    ]
    tasks_per_source: Annotated[
        int, Field(description="Required fixed-frame call count per result file.")
    ]
    source_runs: Annotated[
        list[SourceRun], Field(description="All frozen result files used.")
    ]
    transcript_file: Annotated[
        str, Field(description="JSONL path relative to this manifest.")
    ] = "transcripts.jsonl"
    transcript_sha256: Annotated[
        str, Field(description="SHA-256 of the compact transcript JSONL.")
    ]
    rows: Annotated[int, Field(description="Total exported call rows.")]
    rows_by_condition: Annotated[
        dict[Condition, int], Field(description="Exported calls per condition.")
    ]
    rows_by_language: Annotated[
        dict[Language, int], Field(description="Exported calls per language.")
    ]


def _condition_path(
    root: Path, language: Language, system: System, condition: Condition
) -> Path:
    if condition == "baseline":
        return _voice_result_path(root, language, "retail", system)
    return _retail_ablation_path(
        root,
        language,
        system,
        native_script=condition == "native_script_entities_and_db",
    )


def _table_columns(condition: Condition) -> list[TableColumn]:
    if condition == "baseline":
        return ["localized", "romanized"]
    if condition == "source_entities":
        return ["source"]
    return ["native"]


def _task_subset() -> tuple[Path, set[str]]:
    path = DATA_DIR / "tau2" / "task_subsets" / "retail_30.json"
    payload = json.loads(path.read_text())
    return path, {str(task_id) for task_id in payload["task_ids"]}


def _selected_index_entries(
    results_path: Path, language: Language, expected_task_ids: set[str]
) -> list[SimulationIndexEntry]:
    metadata = Results.load_metadata(results_path)
    if metadata.simulation_index is None:
        raise ValueError(f"{results_path}: directory result has no simulation_index")
    selected = [
        entry
        for entry in metadata.simulation_index
        if int(entry.trial or 0) == 0
        and _canonical_task_id(str(entry.task_id), language) in expected_task_ids
    ]
    canonical_ids = [
        _canonical_task_id(str(entry.task_id), language) for entry in selected
    ]
    if (
        len(selected) != len(expected_task_ids)
        or set(canonical_ids) != expected_task_ids
    ):
        raise ValueError(
            f"{results_path}: expected one trial-0 row for each of "
            f"{len(expected_task_ids)} retail tasks; found {len(selected)} rows "
            f"and {len(set(canonical_ids))} unique tasks"
        )
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError(f"{results_path}: duplicate canonical task ids in selection")
    return sorted(
        selected, key=lambda entry: _canonical_task_id(entry.task_id, language)
    )


def _load_simulation(
    results_path: Path, simulation_id: str
) -> tuple[SimulationRun, str]:
    simulation_path = results_path.parent / "simulations" / f"{simulation_id}.json"
    if not simulation_path.is_file():
        raise FileNotFoundError(f"missing source simulation: {simulation_path}")
    payload = simulation_path.read_bytes()
    return SimulationRun.model_validate_json(payload), _sha256_bytes(payload)


def export_retail_ablation_transcripts(
    evidence_root: Path, out: Path
) -> AblationTranscriptManifest:
    """Export every fixed-frame ablation call as compact deterministic JSONL."""
    root = evidence_root.expanduser().resolve()
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    subset_path, expected_task_ids = _task_subset()

    rows: list[AblationTranscript] = []
    sources: list[SourceRun] = []
    for language_value in LANGUAGES:
        language: Language = language_value
        for system_value in SYSTEMS:
            system: System = system_value
            for condition_value in CONDITIONS:
                condition: Condition = condition_value
                results_path = _condition_path(root, language, system, condition)
                if not results_path.is_file():
                    raise FileNotFoundError(f"missing ablation result: {results_path}")
                results_sha256 = _sha256_file(results_path)
                entries = _selected_index_entries(
                    results_path, language, expected_task_ids
                )
                rewards = [entry.reward for entry in entries]
                if any(reward is None for reward in rewards):
                    raise ValueError(f"{results_path}: selected row has no reward")
                relative_results = results_path.relative_to(root).as_posix()
                sources.append(
                    SourceRun(
                        language=language,
                        system=system,
                        condition=condition,
                        results_path=relative_results,
                        results_sha256=results_sha256,
                        selected_rows=len(entries),
                        pass_at_1_pct=100
                        * sum(float(reward) for reward in rewards)
                        / len(rewards),
                    )
                )
                for entry in entries:
                    sim, simulation_sha256 = _load_simulation(
                        results_path, str(entry.id)
                    )
                    if sim.task_id != entry.task_id or sim.trial != entry.trial:
                        raise ValueError(
                            f"{results_path}: simulation {entry.id} does not match its index"
                        )
                    turns = [
                        TranscriptTurn(
                            turn_index=index,
                            speaker=turn.speaker,
                            text=turn.text,
                            interrupted=turn.interrupted,
                        )
                        for index, turn in enumerate(extract_spoken_turns(sim))
                    ]
                    rows.append(
                        AblationTranscript(
                            language=language,
                            system=system,
                            condition=condition,
                            table_columns=_table_columns(condition),
                            simulation_id=sim.id,
                            task_id=sim.task_id,
                            canonical_task_id=_canonical_task_id(sim.task_id, language),
                            trial=sim.trial,
                            reward=(
                                sim.reward_info.reward if sim.reward_info else None
                            ),
                            termination_reason=sim.termination_reason.value,
                            source_results_path=relative_results,
                            source_results_sha256=results_sha256,
                            source_simulation_sha256=simulation_sha256,
                            turns=turns,
                        )
                    )

    jsonl = b"".join(
        (
            json.dumps(
                row.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for row in rows
    )
    transcript_path = out / "transcripts.jsonl"
    transcript_path.write_bytes(jsonl)
    condition_counts: Counter[Condition] = Counter(row.condition for row in rows)
    language_counts: Counter[Language] = Counter(row.language for row in rows)
    manifest = AblationTranscriptManifest(
        evidence_root_name=root.name,
        task_subset_sha256=_sha256_file(subset_path),
        tasks_per_source=len(expected_task_ids),
        source_runs=sources,
        transcript_sha256=_sha256_bytes(jsonl),
        rows=len(rows),
        rows_by_condition=dict(condition_counts),
        rows_by_language=dict(language_counts),
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return manifest


def verify_retail_ablation_transcripts(out: Path) -> AblationTranscriptManifest:
    """Validate schemas, hashes, uniqueness, and the closed ablation frame."""
    out = out.expanduser().resolve()
    manifest = AblationTranscriptManifest.model_validate_json(
        (out / "manifest.json").read_bytes()
    )
    transcript_path = out / manifest.transcript_file
    if _sha256_file(transcript_path) != manifest.transcript_sha256:
        raise ValueError("compact transcript JSONL hash differs from manifest")
    rows = [
        AblationTranscript.model_validate_json(line)
        for line in transcript_path.read_bytes().splitlines()
        if line
    ]
    if len(rows) != manifest.rows:
        raise ValueError(f"manifest records {manifest.rows} rows; found {len(rows)}")
    expected_combinations = {
        (language, system, condition)
        for language in LANGUAGES
        for system in SYSTEMS
        for condition in CONDITIONS
    }
    source_combinations = {
        (source.language, source.system, source.condition)
        for source in manifest.source_runs
    }
    if source_combinations != expected_combinations or len(manifest.source_runs) != len(
        expected_combinations
    ):
        raise ValueError("source-run manifest does not cover the closed ablation frame")
    if any(
        source.selected_rows != manifest.tasks_per_source
        for source in manifest.source_runs
    ):
        raise ValueError("source-run selected counts differ from the fixed task frame")
    expected_rows = len(expected_combinations) * manifest.tasks_per_source
    if len(rows) != expected_rows:
        raise ValueError(f"closed ablation frame requires {expected_rows} rows")
    keys = {
        (row.language, row.system, row.condition, row.canonical_task_id) for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("duplicate language/system/condition/task rows")
    condition_counts = Counter(row.condition for row in rows)
    language_counts = Counter(row.language for row in rows)
    if dict(condition_counts) != manifest.rows_by_condition:
        raise ValueError("condition counts differ from manifest")
    if dict(language_counts) != manifest.rows_by_language:
        raise ValueError("language counts differ from manifest")
    rows_by_source: dict[
        tuple[Language, System, Condition], list[AblationTranscript]
    ] = {combination: [] for combination in expected_combinations}
    for row in rows:
        rows_by_source[(row.language, row.system, row.condition)].append(row)
        if row.trial != 0:
            raise ValueError("ablation export includes a non-trial-0 row")
        if row.table_columns != _table_columns(row.condition):
            raise ValueError("row table-column mapping differs from its condition")
    task_frames = {
        frozenset(row.canonical_task_id for row in source_rows)
        for source_rows in rows_by_source.values()
    }
    if (
        len(task_frames) != 1
        or len(next(iter(task_frames))) != manifest.tasks_per_source
    ):
        raise ValueError("source runs do not share one complete canonical task frame")
    source_by_combination = {
        (source.language, source.system, source.condition): source
        for source in manifest.source_runs
    }
    for combination, source_rows in rows_by_source.items():
        source = source_by_combination[combination]
        if len(source_rows) != manifest.tasks_per_source:
            raise ValueError(f"{combination} does not contain the fixed task count")
        if any(
            row.source_results_path != source.results_path
            or row.source_results_sha256 != source.results_sha256
            for row in source_rows
        ):
            raise ValueError(
                f"{combination} row provenance differs from its source run"
            )
        rewards = [row.reward for row in source_rows]
        if any(reward is None for reward in rewards):
            raise ValueError(f"{combination} contains an unscored transcript row")
        pass_at_1 = 100 * sum(float(reward) for reward in rewards) / len(rewards)
        if abs(pass_at_1 - source.pass_at_1_pct) > 1e-12:
            raise ValueError(f"{combination} outcome differs from its source summary")
    return manifest

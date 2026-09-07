"""Build and verify the compact tau-Elicitation reviewer archive.

The frozen run directory is intentionally detached from Git: it contains tens of
gigabytes of audio and tick-level traces.  This module exports the evidence that a
reviewer needs into provenance-bearing, content-addressed artifacts while retaining
SHA-256 links back to every source result and simulation.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict

RELEASE_VERSION = "tau-elicit-review-release-v1"
TRANSCRIPT_VERSION = "tau-elicit-compact-transcript-v1"
EXAMPLE_SELECTION_VERSION = "tau-elicit-examples-v1"
ONE_FIELD_REFERENCE_VERSION = "tau-elicit-matched-one-field-v1"
DETACHED_EVIDENCE_URL = (
    "https://drive.google.com/drive/folders/"
    "1GAuTs3Naog5irE4J4MwILMTpFyJz-2dm?usp=sharing"
)

PAPER_AGENT_DIRECTED = {
    *{
        f"main_runs/modeb_{system}_{condition}_2026-09-02/results.json"
        for system in ("openai_minimal", "openai_xhigh", "gemini_high", "xai_10")
        for condition in ("regular", "chanheavy", "speechheavy")
    },
    "main_runs/modeb_openai_minimal_chanheavy_2026-09-05/results.json",
    "main_runs/modeb_openai_minimal_speechheavy_2026-09-05/results.json",
}
# The initial GPT-minimal stress cells were completed on September 5. Exclude
# the non-existent September 2 spellings introduced by the set comprehension.
PAPER_AGENT_DIRECTED -= {
    "main_runs/modeb_openai_minimal_chanheavy_2026-09-02/results.json",
    "main_runs/modeb_openai_minimal_speechheavy_2026-09-02/results.json",
}
PAPER_SCAFFOLDED = {
    *{
        f"main_runs/intake_m_{system}_{condition}/results.json"
        for system in ("openai_minimal", "openai_xhigh", "gemini_high", "xai_10")
        for condition in ("regular", "chanlight_speechheavy")
    },
    *{
        f"ablations/scaffolded/modea_{system}_chanheavy_2026-09-02/results.json"
        for system in ("openai_minimal", "openai_xhigh", "gemini_high", "xai_10")
    },
}

_MESSAGE_RE = re.compile(
    r'<message\s+uuid="(?P<uuid>[^"]+)"\s+active="(?P<active>\d+)">'
    r"(?P<body>.*?)</message>",
    re.DOTALL,
)
_CHUNK_RE = re.compile(r"<chunk\s+id=\"\d+\">(.*?)</chunk>", re.DOTALL)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _slug(relative_path: str) -> str:
    value = relative_path.removesuffix("/results.json").removesuffix("results.json")
    return value.strip("/").replace("/", "__")


def _reward(simulation: dict[str, Any], index_row: Optional[dict[str, Any]]) -> float:
    if index_row is not None and index_row.get("reward") is not None:
        return float(index_row["reward"])
    reward_info = simulation.get("reward_info") or {}
    if reward_info.get("reward") is None:
        raise ValueError(f"Simulation {simulation.get('id')} has no reward")
    return float(reward_info["reward"])


class TranscriptTurn(BaseModel):
    """One compact chronological call event."""

    model_config = ConfigDict(extra="forbid")

    order: int
    role: Literal["agent", "user", "tool"]
    text: str = ""
    source: str
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict[str, Any]] = None
    tool_result: Optional[str] = None


class TranscriptRecord(BaseModel):
    """Portable transcript and outcome for one frozen simulation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRANSCRIPT_VERSION
    cell: str
    cohort: str
    simulation_id: str
    source_simulation_sha256: str
    task_id: str
    trial: int
    seed: Optional[int]
    reward: float
    evaluation_fields: dict[str, str]
    termination_reason: Optional[str]
    duration_seconds: Optional[float]
    tick_count: Optional[int]
    complication: Optional[dict[str, Any]]
    speech_environment: Optional[dict[str, Any]]
    turns: list[TranscriptTurn]


class ResultCell(BaseModel):
    """Identity, configuration, and outcome summary for one results root."""

    model_config = ConfigDict(extra="forbid")

    results_path: str
    source_results_sha256: str
    cohort: str
    system: str
    condition: str
    rows: int
    tasks: int
    trials: list[int]
    passes: int
    pass_rate: float
    git_commit: str
    domain: str
    provider: Optional[str]
    model: str
    reasoning_effort: Optional[str]
    seed: Optional[int]
    complication_profile: Optional[str]
    complication_rate: Optional[float]
    channel_effects_mode: Optional[str]
    speech_effects_mode: Optional[str]
    config_file: str
    transcript_file: str
    transcript_sha256: str


class ArtifactFile(BaseModel):
    """Content-addressed file in the release."""

    path: str
    sha256: str
    bytes: int
    rows: Optional[int] = None


class ExampleCall(BaseModel):
    """One deterministically selected compact example call."""

    mode: Literal["agent_directed", "scaffolded"]
    system: str
    simulation_id: str
    task_id: str
    reward: float
    transcript_file: str
    source_cell: str
    source_simulation_sha256: str


class OneFieldSourceOutcome(BaseModel):
    """One frozen GPT-xhigh realization of an atomic parent task."""

    model_config = ConfigDict(extra="forbid")

    source_cell: str
    trial: int
    simulation_id: str
    source_simulation_sha256: str
    reward: float


class OneFieldMatchedSlot(BaseModel):
    """One composition slot matched to its atomic single-field outcomes."""

    model_config = ConfigDict(extra="forbid")

    composition_task_id: str
    n_fields_in_composition: int
    slot_index: int
    parent_task_id: str
    bank: str
    tier: str
    field_name: str
    source_outcomes: list[OneFieldSourceOutcome]


class ReleaseManifest(BaseModel):
    """Top-level contract for the reviewer archive."""

    schema_version: int = 1
    release_version: str = RELEASE_VERSION
    evidence_root_name: str
    result_cells: list[ResultCell]
    transcript_count: int
    speech_judgment_count: int
    paper_speech_judgment_count: int
    speech_judgment_counts_by_cell: dict[str, int]
    validation_call_count: int
    validation_finding_count: int
    examples: list[ExampleCall]
    artifacts: list[ArtifactFile]


class VerificationCheck(BaseModel):
    """One release verification check."""

    code: str
    ok: bool
    detail: str


class VerificationReport(BaseModel):
    """Machine-readable verification result."""

    ok: bool
    summary: str
    checks: list[VerificationCheck]


def _cell_metadata(relative_path: str, info: dict[str, Any]) -> tuple[str, str, str]:
    name = relative_path.lower()
    if relative_path in PAPER_AGENT_DIRECTED:
        cohort = "paper_agent_directed"
    elif relative_path in PAPER_SCAFFOLDED:
        cohort = "paper_scaffolded"
    elif "modeb_openai_minimal_" in name:
        cohort = "supplemental_agent_directed"
    elif "intake_passk" in name:
        cohort = "same_environment_pass3"
    elif "entity_composition" in name:
        cohort = "entity_composition"
    elif "gated_stages" in name:
        cohort = "submission_workflow"
    elif "text_channel" in name:
        cohort = "text_control"
    else:
        cohort = "supplemental"

    system = "unknown"
    for token, label in (
        ("openai_minimal", "gpt_minimal"),
        ("openai_xhigh", "gpt_xhigh"),
        ("gemini_high", "gemini_high"),
        ("xai_10", "grok"),
    ):
        if token in name:
            system = label
            break
    if cohort in {"entity_composition", "submission_workflow", "text_control"}:
        model = str((info.get("agent_info") or {}).get("llm") or "")
        system = model or system

    if "chanheavy" in name:
        condition = "noise_heavy"
    elif "speechheavy" in name:
        condition = "speech_heavy"
    elif "regular" in name:
        condition = "regular"
    elif "ecomp_n2" in name:
        condition = "two_fields"
    elif "ecomp_n3" in name:
        condition = "three_fields"
    elif "ehier" in name:
        condition = "workflow_ablation"
    elif "text" in name:
        condition = "text"
    else:
        condition = "unspecified"
    return cohort, system, condition


def _iter_simulations(
    results_path: Path,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], Iterable[tuple[dict[str, Any], bytes]]
]:
    document = json.loads(results_path.read_text())
    info = document.get("info") or {}
    index = document.get("simulation_index") or []
    if index:
        by_id = {str(row["id"]): row for row in index}

        def sharded() -> Iterable[tuple[dict[str, Any], bytes]]:
            for simulation_id in sorted(by_id):
                sim_path = results_path.parent / "simulations" / f"{simulation_id}.json"
                raw = sim_path.read_bytes()
                simulation = json.loads(raw)
                simulation["_index_row"] = by_id[simulation_id]
                yield simulation, raw

        return info, index, sharded()

    simulations = document.get("simulations") or []

    def embedded() -> Iterable[tuple[dict[str, Any], bytes]]:
        for simulation in simulations:
            raw = _json_bytes(simulation)
            yield simulation, raw

    synthetic_index = [
        {
            "id": row.get("id"),
            "task_id": row.get("task_id"),
            "trial": row.get("trial", 0),
            "reward": (row.get("reward_info") or {}).get("reward"),
        }
        for row in simulations
    ]
    return info, synthetic_index, embedded()


def _parse_user_snapshot(value: str) -> Optional[tuple[str, int, str]]:
    """Parse one streaming user ``audio_script_gold`` XML snapshot."""
    match = _MESSAGE_RE.search(value)
    if match is None:
        return None
    body = html.unescape("".join(_CHUNK_RE.findall(match.group("body"))))
    return match.group("uuid"), int(match.group("active")), body


def _full_duplex_turns(simulation: dict[str, Any]) -> list[TranscriptTurn]:
    ticks = simulation.get("ticks") or []
    user_messages: dict[str, tuple[int, int, str]] = {}
    plain_user: list[tuple[int, str]] = []
    for index, tick in enumerate(ticks):
        user_chunk = tick.get("user_chunk") or {}
        snapshot = user_chunk.get("audio_script_gold") or ""
        parsed = _parse_user_snapshot(snapshot)
        if parsed is not None:
            uuid, active, text = parsed
            previous = user_messages.get(uuid)
            first = index if previous is None else previous[0]
            if previous is None or active >= previous[1]:
                user_messages[uuid] = (first, active, text)
        elif snapshot and (not plain_user or plain_user[-1][1] != snapshot):
            plain_user.append((index, snapshot))

    user_starts = sorted(first for first, _, _ in user_messages.values())
    agent_segments: list[tuple[int, str]] = []
    current_start: Optional[int] = None
    current_last: Optional[int] = None
    current: list[str] = []
    for index, tick in enumerate(ticks):
        content = str((tick.get("agent_chunk") or {}).get("content") or "")
        if not content:
            continue
        user_between = current_last is not None and any(
            current_last < start <= index for start in user_starts
        )
        long_gap = current_last is not None and index - current_last > 15
        if current and (user_between or long_gap):
            agent_segments.append((current_start or 0, "".join(current).strip()))
            current = []
            current_start = None
        if current_start is None:
            current_start = index
        current.append(content)
        current_last = index
    if current:
        agent_segments.append((current_start or 0, "".join(current).strip()))

    # Judged calls carry exact per-utterance expected text and tick indices.  Prefer
    # those over the stream reconstruction used by unjudged cells.
    delivery = simulation.get("delivery_info") or {}
    judged = delivery.get("utterance_results") or []
    if judged:
        agent_segments = [
            (
                int(row.get("utterance_idx", position)),
                str(row.get("expected_text") or ""),
            )
            for position, row in enumerate(judged)
            if str(row.get("expected_text") or "").strip()
        ]

    events: list[tuple[int, int, TranscriptTurn]] = []
    for position, (index, text) in enumerate(agent_segments):
        if text:
            events.append(
                (
                    index,
                    0,
                    TranscriptTurn(
                        order=position,
                        role="agent",
                        text=text,
                        source=(
                            "delivery_expected_text"
                            if judged
                            else "agent_stream_chunks"
                        ),
                    ),
                )
            )
    for position, (index, _, text) in enumerate(user_messages.values()):
        if text.strip():
            events.append(
                (
                    index,
                    1,
                    TranscriptTurn(
                        order=position,
                        role="user",
                        text=text,
                        source="audio_script_gold",
                    ),
                )
            )
    for position, (index, text) in enumerate(plain_user):
        events.append(
            (
                index,
                1,
                TranscriptTurn(
                    order=position,
                    role="user",
                    text=text,
                    source="audio_script_gold_plain",
                ),
            )
        )
    for index, tick in enumerate(ticks):
        for event_order, requestor in enumerate(("agent", "user"), start=2):
            calls = tick.get(f"{requestor}_tool_calls") or []
            results = {
                str(row.get("id")): row
                for row in tick.get(f"{requestor}_tool_results") or []
            }
            for position, call in enumerate(calls):
                result = results.get(str(call.get("id")))
                events.append(
                    (
                        index,
                        event_order,
                        TranscriptTurn(
                            order=position,
                            role="tool",
                            source=f"{requestor}_tool_event",
                            tool_name=call.get("name"),
                            tool_arguments=call.get("arguments") or {},
                            tool_result=(
                                str(result.get("content")) if result else None
                            ),
                        ),
                    )
                )

    events.sort(key=lambda row: (row[0], row[1], row[2].order))
    turns: list[TranscriptTurn] = []
    for order, (_, _, turn) in enumerate(events):
        turns.append(turn.model_copy(update={"order": order}))
    return turns


def _half_duplex_turns(simulation: dict[str, Any]) -> list[TranscriptTurn]:
    turns: list[TranscriptTurn] = []
    for message in simulation.get("messages") or []:
        role = message.get("role")
        mapped_role: Literal["agent", "user", "tool"]
        if role == "assistant":
            mapped_role = "agent"
        elif role == "user":
            mapped_role = "user"
        else:
            mapped_role = "tool"
        content = str(message.get("content") or "")
        tool_calls = message.get("tool_calls") or []
        if content:
            turns.append(
                TranscriptTurn(
                    order=len(turns),
                    role=mapped_role,
                    text=content,
                    source="messages",
                    tool_result=content if mapped_role == "tool" else None,
                )
            )
        for call in tool_calls:
            turns.append(
                TranscriptTurn(
                    order=len(turns),
                    role="tool",
                    source="message_tool_call",
                    tool_name=call.get("name"),
                    tool_arguments=call.get("arguments") or {},
                )
            )
    return turns


def _transcript_record(
    simulation: dict[str, Any],
    raw: bytes,
    *,
    cell: str,
    cohort: str,
) -> TranscriptRecord:
    index_row = simulation.pop("_index_row", None)
    messages = simulation.get("messages")
    turns = (
        _half_duplex_turns(simulation) if messages else _full_duplex_turns(simulation)
    )
    evaluation_fields: dict[str, str] = {}
    for check in (simulation.get("reward_info") or {}).get("action_checks") or []:
        action = check.get("action") or {}
        if action.get("name") == "submit_fields":
            evaluation_fields.update(
                (action.get("arguments") or {}).get("fields") or {}
            )
    return TranscriptRecord(
        cell=cell,
        cohort=cohort,
        simulation_id=str(simulation["id"]),
        source_simulation_sha256=_sha256_bytes(raw),
        task_id=str(simulation["task_id"]),
        trial=int(simulation.get("trial") or 0),
        seed=simulation.get("seed"),
        reward=_reward(simulation, index_row),
        evaluation_fields=evaluation_fields,
        termination_reason=simulation.get("termination_reason"),
        duration_seconds=simulation.get("duration"),
        tick_count=(len(simulation.get("ticks") or []) if not messages else None),
        complication=simulation.get("complication"),
        speech_environment=simulation.get("speech_environment"),
        turns=turns,
    )


def _speech_rows(
    simulation: dict[str, Any], *, cell: str, cohort: str
) -> Iterable[dict[str, Any]]:
    delivery = simulation.get("delivery_info") or {}
    model = delivery.get("judge_model")
    prompt_version = delivery.get("judge_prompt_version")
    for row in delivery.get("utterance_results") or []:
        yield {
            "cell": cell,
            "cohort": cohort,
            "simulation_id": simulation.get("id"),
            "task_id": simulation.get("task_id"),
            "judge_model": model,
            "judge_args": delivery.get("judge_args") or {},
            "judge_prompt_version": prompt_version,
            "finding_filter_version": delivery.get("finding_filter_version"),
            "fidelity_end_exclusion_seconds": delivery.get(
                "fidelity_end_exclusion_seconds"
            ),
            "utterance_index": row.get("utterance_idx"),
            "was_interrupted": bool(row.get("was_interrupted")),
            "expected_text": row.get("expected_text"),
            "outcome": row.get("outcome"),
            "severity": row.get("severity"),
            "confidence": row.get("confidence"),
            "summary": row.get("summary"),
            "findings": row.get("findings") or [],
            "excluded_findings": row.get("excluded_findings") or [],
            "factor_checks": row.get("factor_checks") or [],
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {key: value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _validation_sources(
    validation_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    review_files = sorted(validation_root.glob("*_review.csv"))
    decision_files = sorted(validation_root.glob("*_decisions.csv"))
    if len(review_files) != 2 or len(decision_files) != 2:
        raise ValueError(
            "Expected exactly two review CSVs and two decision CSVs in "
            f"{validation_root}"
        )
    reviews = [_read_csv(path) for path in review_files]
    decisions = [_read_csv(path) for path in decision_files]
    if any(len(rows) != 200 for rows in reviews):
        raise ValueError("Each review CSV must contain 200 rows")
    if any(len(rows) != 41 for rows in decisions):
        raise ValueError("Each decision CSV must contain 41 rows")
    return reviews, decisions


def _kappa(left: list[bool], right: list[bool]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa inputs must be non-empty and aligned")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_positive = sum(left) / len(left)
    right_positive = sum(right) / len(right)
    expected = left_positive * right_positive + (1 - left_positive) * (
        1 - right_positive
    )
    return (observed - expected) / (1 - expected) if expected != 1 else 1.0


VALIDATION_COLUMNS = (
    "unit_type",
    "clip_id",
    "simulation_id",
    "task_id",
    "bank",
    "tier",
    "reward",
    "factor",
    "utterance_index",
    "utterance_text",
    "judge_label",
    "judge_severity",
    "judge_issue",
    "human_positive_votes",
    "human_strict_label",
    "human_lenient_label",
    "human_severity_a",
    "human_severity_b",
    "human_error_source_a",
    "human_error_source_b",
    "human_error_subtype_a",
    "human_error_subtype_b",
    "human_notes_a",
    "human_notes_b",
)


def _old_fidelity_findings(simulation: dict[str, Any]) -> list[dict[str, Any]]:
    delivery = simulation.get("delivery_info") or {}
    return list(((delivery.get("fidelity") or {}).get("findings") or []))


def _export_validation(
    validation_root: Path,
    validation_run_root: Path,
    out: Path,
) -> tuple[int, int, dict[str, Any]]:
    """Write final call/finding labels without packet-workflow metadata."""
    reviews, decisions = _validation_sources(validation_root)
    final_reviews: list[dict[str, dict[str, str]]] = []
    for rows in reviews:
        selected = {
            row["simulation_id"]: row for row in rows if row["phase"] == "post_reveal"
        }
        if len(selected) != 100:
            raise ValueError("Expected 100 final per-call annotations per reviewer")
        final_reviews.append(selected)

    final_decisions = [{row["finding_id"]: row for row in rows} for rows in decisions]
    finding_ids = sorted(set(final_decisions[0]) & set(final_decisions[1]))
    if len(finding_ids) != 41:
        raise ValueError("Finding decision files are not aligned on 41 findings")

    simulation_ids = sorted(set(final_reviews[0]) & set(final_reviews[1]))
    rows_out: list[dict[str, object]] = []
    judge_flagged_calls: set[str] = set()
    strict_human_calls: set[str] = set()
    lenient_human_calls: set[str] = set()
    sim_cache: dict[str, dict[str, Any]] = {}

    for simulation_id in simulation_ids:
        sim_path = validation_run_root / "simulations" / f"{simulation_id}.json"
        simulation = json.loads(sim_path.read_text())
        sim_cache[simulation_id] = simulation
        findings = _old_fidelity_findings(simulation)
        if findings:
            judge_flagged_calls.add(simulation_id)
        human = [mapping[simulation_id] for mapping in final_reviews]
        severities = [int(row.get("fidelity_severity") or 0) for row in human]
        votes = sum(value > 0 for value in severities)
        if votes == 2:
            strict_human_calls.add(simulation_id)
        if votes:
            lenient_human_calls.add(simulation_id)
        rows_out.append(
            {
                "unit_type": "call",
                "clip_id": human[0]["clip_id"],
                "simulation_id": simulation_id,
                "task_id": human[0]["task_id"],
                "bank": human[0]["bank"],
                "tier": human[0]["tier"],
                "reward": human[0]["reward"],
                "factor": "speech_fidelity_any",
                "utterance_index": "",
                "utterance_text": "",
                "judge_label": int(bool(findings)),
                "judge_severity": max(
                    (int(row.get("severity") or 0) for row in findings), default=0
                ),
                "judge_issue": " | ".join(
                    str(row.get("issue") or "") for row in findings
                ),
                "human_positive_votes": votes,
                "human_strict_label": int(votes == 2),
                "human_lenient_label": int(votes > 0),
                "human_severity_a": severities[0],
                "human_severity_b": severities[1],
                "human_error_source_a": human[0].get("error_source", ""),
                "human_error_source_b": human[1].get("error_source", ""),
                "human_error_subtype_a": human[0].get("error_subtype", ""),
                "human_error_subtype_b": human[1].get("error_subtype", ""),
                "human_notes_a": human[0].get("fidelity_notes", ""),
                "human_notes_b": human[1].get("fidelity_notes", ""),
            }
        )

    decision_a: list[bool] = []
    decision_b: list[bool] = []
    for finding_id in finding_ids:
        human = [mapping[finding_id] for mapping in final_decisions]
        simulation_id = human[0]["simulation_id"]
        if human[1]["simulation_id"] != simulation_id:
            raise ValueError(f"Finding {finding_id} maps to different simulations")
        confirmations = [row["decision"] == "confirmed" for row in human]
        decision_a.append(confirmations[0])
        decision_b.append(confirmations[1])
        utterance_index = int(human[0]["utterance_index"])
        agent_turns = [
            row.text
            for row in _full_duplex_turns(sim_cache[simulation_id])
            if row.role == "agent"
        ]
        utterance_text = (
            agent_turns[utterance_index]
            if 0 <= utterance_index < len(agent_turns)
            else ""
        )
        call_row = final_reviews[0][simulation_id]
        rows_out.append(
            {
                "unit_type": "finding",
                "clip_id": human[0]["clip_id"],
                "simulation_id": simulation_id,
                "task_id": human[0]["task_id"],
                "bank": call_row["bank"],
                "tier": call_row["tier"],
                "reward": call_row["reward"],
                "factor": human[0]["category"],
                "utterance_index": utterance_index,
                "utterance_text": utterance_text,
                "judge_label": 1,
                "judge_severity": human[0]["judge_severity"],
                "judge_issue": human[0]["judge_issue"],
                "human_positive_votes": sum(confirmations),
                "human_strict_label": int(all(confirmations)),
                "human_lenient_label": int(any(confirmations)),
                "human_severity_a": "",
                "human_severity_b": "",
                "human_error_source_a": "",
                "human_error_source_b": "",
                "human_error_subtype_a": "",
                "human_error_subtype_b": "",
                "human_notes_a": human[0].get("note", ""),
                "human_notes_b": human[1].get("note", ""),
            }
        )

    csv_path = out / "judge_validation" / "validation.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=VALIDATION_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows_out)

    strict_tp = sum(a and b for a, b in zip(decision_a, decision_b, strict=True))
    lenient_tp = sum(a or b for a, b in zip(decision_a, decision_b, strict=True))
    strict_recalled = len(strict_human_calls & judge_flagged_calls)
    lenient_recalled = len(lenient_human_calls & judge_flagged_calls)

    def scores(
        tp: int, predicted: int, recalled: int, positives: int
    ) -> dict[str, float]:
        precision = tp / predicted
        recall = recalled / positives
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall),
        }

    metrics = {
        "schema_version": 1,
        "unit_contract": {
            "precision": "finding level",
            "recall": "call level; any speech-fidelity finding flags a call",
            "strict": "both human labels are positive",
            "lenient": "either human label is positive",
        },
        "counts": {
            "calls": len(simulation_ids),
            "judge_findings": len(finding_ids),
            "judge_flagged_calls": len(judge_flagged_calls),
            "strict_human_positive_calls": len(strict_human_calls),
            "lenient_human_positive_calls": len(lenient_human_calls),
            "strict_confirmed_findings": strict_tp,
            "lenient_confirmed_findings": lenient_tp,
        },
        "strict": scores(
            strict_tp, len(finding_ids), strict_recalled, len(strict_human_calls)
        ),
        "lenient": scores(
            lenient_tp, len(finding_ids), lenient_recalled, len(lenient_human_calls)
        ),
        "finding_decision_agreement": sum(
            a == b for a, b in zip(decision_a, decision_b, strict=True)
        )
        / len(finding_ids),
        "finding_decision_kappa": _kappa(decision_a, decision_b),
        "source_files": {
            path.name: _sha256_file(path)
            for path in sorted(validation_root.glob("*.csv"))
        },
        "source_run": {
            "results_sha256": _sha256_file(validation_run_root / "results.json"),
            "git_commit": json.loads(
                (validation_run_root / "results.json").read_text()
            )["info"]["git_commit"],
        },
    }
    _write_json(out / "judge_validation" / "metrics.json", metrics)
    return len(simulation_ids), len(finding_ids), metrics


ANALYSIS_INPUTS = (
    "modeb_rollup.json",
    "modeb_crossed.json",
    "expected_draws_9401.json",
    "expected_draws_9402.json",
    "expected_draws_9403.json",
    "expected_draws_modea_401.json",
    "caller_effort_agent_directed_vs_scaffolded.json",
)
PAPER_ANALYSIS_INPUTS = (
    "intake_caller_voice_significance_2026-09-07.json",
    "intake_main_significance_2026-09-05.json",
    "intake_realism_effects_2026-09-07.json",
    "intake_speech_fidelity_2026-09-04.json",
)

_TASKS_ROOT = "data/tau2/domains/intake"
_COMPOSE_TASK_PATHS = {
    "compose_n2": f"{_TASKS_ROOT}/bands/compose_n2/tasks.json",
    "compose_n3": f"{_TASKS_ROOT}/bands/compose_n3/tasks.json",
}
_CHAIN_TASK_PATHS = {
    "chain_n2": f"{_TASKS_ROOT}/bands/chain_n2/tasks.json",
    "chain_n3": f"{_TASKS_ROOT}/bands/chain_n3/tasks.json",
}


def _task_source_paths(info: dict[str, Any]) -> tuple[str, ...]:
    """Return the task files selected by one frozen run configuration."""
    domain = str((info.get("environment_info") or {}).get("domain_name") or "")
    split = str(info.get("task_split_name") or "base")
    if domain == "intake_staged":
        if split == "base":
            return tuple(_CHAIN_TASK_PATHS.values())
        if split in _CHAIN_TASK_PATHS:
            return (_CHAIN_TASK_PATHS[split],)
        raise ValueError(f"Unknown frozen intake_staged task split: {split!r}")
    if split in _COMPOSE_TASK_PATHS:
        return (_COMPOSE_TASK_PATHS[split],)
    if split == "base":
        return (f"{_TASKS_ROOT}/tasks.json",)
    raise ValueError(f"Unknown frozen intake task split: {split!r}")


def _git_blob(repo_root: Path, commit: str, source_path: str) -> bytes:
    """Read an exact historical blob needed to interpret a frozen result."""
    if re.fullmatch(r"[0-9a-f]{7,40}", commit) is None:
        raise ValueError(f"Invalid frozen Git commit: {commit!r}")
    process = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{commit}:{source_path}"],
        check=False,
        capture_output=True,
    )
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise FileNotFoundError(
            f"Cannot recover frozen task snapshot {commit}:{source_path}: {message}"
        )
    return process.stdout


def _write_task_snapshots(
    out: Path, repo_root: Path, info: dict[str, Any]
) -> list[dict[str, str]]:
    """Export exact, content-addressed task files from the run's source commit."""
    commit = str(info.get("git_commit") or "")
    snapshots: list[dict[str, str]] = []
    for source_path in _task_source_paths(info):
        encoded = _git_blob(repo_root, commit, source_path)
        # Fail early if a historical path no longer represents task JSON.
        json.loads(encoded)
        digest = _sha256_bytes(encoded)
        relative = f"prompts/task_snapshots/{digest}.json"
        target = out / relative
        if not target.exists() or target.read_bytes() != encoded:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded)
        snapshots.append(
            {
                "source_commit": commit,
                "source_path": source_path,
                "sha256": digest,
                "archive_path": relative,
            }
        )
    return snapshots


def _write_prompt_object(out: Path, kind: str, content: str) -> str:
    encoded = content.encode("utf-8")
    digest = _sha256_bytes(encoded)
    path = out / "prompts" / "objects" / kind / f"{digest}.md"
    if not path.exists() or path.read_bytes() != encoded:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    return digest


def _runtime_caller_guidelines(info: dict[str, Any]) -> str:
    """Resolve the caller-guideline text used by the frozen intake builders.

    Historical ``Info`` snapshots selected their guideline before resolving
    ``task_set_name=None`` to the domain, so they recorded the inbound default.
    The simulation builders themselves derived direction from each task's
    scripted agent opener and used the outbound prompt. All frozen source
    commits contain this same outbound prompt text; their files had no final
    newline, which is preserved here for byte-exact prompt identity.
    """
    from tau2.config import SPELL_PROTOCOL_FREE_DOMAINS
    from tau2.user.user_simulator import (
        CallDirection,
        get_global_user_sim_guidelines,
        get_global_user_sim_guidelines_voice,
        strip_outbound_spell_offer_nudge,
    )

    domain = str((info.get("environment_info") or {}).get("domain_name") or "")
    is_voice = info.get("audio_native_config") is not None
    if is_voice:
        guidelines = get_global_user_sim_guidelines_voice(
            use_tools=True, direction=CallDirection.OUTBOUND
        )
        if domain in SPELL_PROTOCOL_FREE_DOMAINS:
            guidelines = strip_outbound_spell_offer_nudge(guidelines)
    else:
        guidelines = get_global_user_sim_guidelines(
            use_tools=True, direction=CallDirection.OUTBOUND
        )
    return guidelines.rstrip("\n")


def _source_audio(cell_root: Path, simulation_id: str) -> Optional[Path]:
    matches = list(
        cell_root.glob(f"artifacts/task_*/sim_{simulation_id}/audio/both.wav")
    )
    if len(matches) > 1:
        raise ValueError(f"Multiple mixed-call audio files for {simulation_id}")
    return matches[0] if matches else None


def _compact_outcomes(
    out: Path, cells: list[ResultCell], cohort: str
) -> dict[str, dict[str, dict[str, float]]]:
    """Load compact task outcomes as system -> condition -> task -> reward."""
    outcomes: dict[str, dict[str, dict[str, float]]] = {}
    for cell in cells:
        if cell.cohort != cohort:
            continue
        task_rows: dict[str, float] = {}
        with (out / cell.transcript_file).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                task_id = str(row["task_id"])
                # Paper cells contain one trial. Keep a loud guard here so a
                # future multi-trial cell cannot be silently collapsed.
                if task_id in task_rows:
                    raise ValueError(
                        f"Duplicate task {task_id} in paper cell {cell.results_path}"
                    )
                task_rows[task_id] = float(row["reward"])
        outcomes.setdefault(cell.system, {})[cell.condition] = task_rows
    return outcomes


def _pass3_by_system(
    outcomes: dict[str, dict[str, dict[str, float]]],
) -> dict[str, float]:
    """Compute success in all three linked realization conditions."""
    result: dict[str, float] = {}
    required = {"regular", "noise_heavy", "speech_heavy"}
    for system, conditions in outcomes.items():
        if set(conditions) != required:
            raise ValueError(
                f"Paper system {system} has conditions {sorted(conditions)}, "
                f"expected {sorted(required)}"
            )
        task_sets = [set(conditions[name]) for name in sorted(required)]
        if len({frozenset(task_ids) for task_ids in task_sets}) != 1:
            raise ValueError(f"Paper tasks are not aligned for {system}")
        task_ids = sorted(task_sets[0])
        if len(task_ids) != 200:
            raise ValueError(f"Expected 200 paper tasks for {system}")
        result[system] = sum(
            all(conditions[name][task_id] == 1.0 for name in required)
            for task_id in task_ids
        ) / len(task_ids)
    return result


def _speech_rollup(path: Path) -> dict[str, dict[str, float | int]]:
    """Recompute the paper speech table from exported utterance judgments."""
    cell_to_system = {
        "modeb_openai_minimal_regular": "GPT minimal",
        "modeb_openai_xhigh_regular": "GPT xhigh",
        "modeb_gemini_high_regular": "Gemini high",
        "modeb_xai_10_regular": "Grok",
    }
    values: dict[str, list[int]] = {system: [] for system in cell_to_system.values()}
    submitted = {system: 0 for system in cell_to_system.values()}
    errors = {system: 0 for system in cell_to_system.values()}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("cohort") != "paper_agent_directed":
                continue
            system = next(
                (
                    label
                    for token, label in cell_to_system.items()
                    if token in str(row.get("cell"))
                ),
                None,
            )
            if system is None:
                continue
            submitted[system] += 1
            if str(row.get("outcome") or "").casefold() == "error":
                errors[system] += 1
                continue
            severity = max(
                (
                    int(finding.get("severity") or 0)
                    for finding in row.get("findings") or []
                    if finding.get("axis") == "fidelity"
                ),
                default=0,
            )
            values[system].append(severity)
    return {
        system: {
            "utterances_submitted": submitted[system],
            "utterances_scored": len(severities),
            "judge_errors": errors[system],
            "mean_max_retained_fidelity_severity": sum(severities) / len(severities),
            "severity_2plus_count": sum(value >= 2 for value in severities),
            "severity_2plus_rate": sum(value >= 2 for value in severities)
            / len(severities),
        }
        for system, severities in values.items()
    }


def _complication_assignment_counts(
    out: Path, cells: list[ResultCell]
) -> tuple[int, int]:
    """Compare every paper complication assignment with the frozen draw ledger."""
    expected_files = {
        9401: "expected_draws_9401.json",
        9402: "expected_draws_9402.json",
        9403: "expected_draws_9403.json",
        401: "expected_draws_modea_401.json",
    }
    expected = {
        seed: json.loads((out / "analysis_inputs" / name).read_text())
        for seed, name in expected_files.items()
    }
    checked = 0
    mismatches = 0
    for cell in cells:
        relevant = cell.cohort == "paper_agent_directed" or (
            cell.cohort == "paper_scaffolded"
            and cell.condition == "noise_heavy"
            and cell.seed == 401
        )
        if not relevant or cell.seed not in expected:
            continue
        with (out / cell.transcript_file).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                actual = (row.get("complication") or {}).get("kind")
                wanted = expected[cell.seed].get(str(row["task_id"]))
                checked += 1
                mismatches += actual != wanted
    return checked, mismatches


def _task_specs(
    snapshot_paths: Iterable[Path],
) -> dict[str, list[tuple[str, str, str]]]:
    """Read ordered missing-field fold rules and gold values from task snapshots."""
    result: dict[str, list[tuple[str, str, str]]] = {}
    for path in snapshot_paths:
        document = json.loads(path.read_text())
        tasks = document.get("tasks") if isinstance(document, dict) else document
        for task in tasks:
            actions = task["initial_state"]["initialization_actions"]
            record = next(row for row in actions if row["func_name"] == "seed_record")[
                "arguments"
            ]["record"]
            missing = next(
                row for row in actions if row["func_name"] == "blank_fields"
            )["arguments"]["field_names"]
            fields = {row["name"]: row for row in record["fields"]}
            result[str(task["id"])] = [
                (name, str(fields[name]["fold"]), str(fields[name]["value"]))
                for name in missing
            ]
    return result


def _cell_task_snapshots(
    out: Path, prompt_cells: list[dict[str, Any]], cell_path: str
) -> list[Path]:
    cell = next(row for row in prompt_cells if row["cell"] == cell_path)
    return [out / row["archive_path"] for row in cell["task_snapshots"]]


def _submission_attempts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        turn.get("tool_arguments") or {}
        for turn in row.get("turns") or []
        if turn.get("role") == "tool" and turn.get("tool_name") == "submit_fields"
    ]


def _score_ablation_cell(
    out: Path,
    cell: ResultCell,
    specs: dict[str, list[tuple[str, str, str]]],
    *,
    trials: Optional[set[int]] = None,
) -> dict[str, Any]:
    """Recompute final task and field outcomes for one workflow."""
    from tau2.domains.intake.folds import FoldKind, fold_value

    task_passes = 0
    field_passes = 0
    calls = 0
    fields = 0
    task_ids: set[str] = set()
    with (out / cell.transcript_file).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if trials is not None and int(row["trial"]) not in trials:
                continue
            ordered = specs[str(row["task_id"])]
            attempts = _submission_attempts(row)
            per_field: dict[str, list[bool]] = {name: [] for name, _, _ in ordered}
            for attempt in attempts:
                submitted = attempt.get("fields") or {}
                for name, kind, gold in ordered:
                    value = submitted.get(name)
                    if isinstance(value, str):
                        per_field[name].append(
                            fold_value(FoldKind(kind), value)
                            == fold_value(FoldKind(kind), gold)
                        )
            final = [any(per_field[name]) for name, _, _ in ordered]

            calls += 1
            fields += len(ordered)
            task_ids.add(str(row["task_id"]))
            task_passes += float(row["reward"]) == 1.0
            field_passes += sum(final)
    return {
        "unique_tasks": len(task_ids),
        "observations": calls,
        "fields": fields,
        "task_passes": task_passes,
        "task_pass_at_1": task_passes / calls,
        "field_passes": field_passes,
        "field_pass_at_1": field_passes / fields,
    }


def _transcript_outcomes(
    out: Path, cell: ResultCell
) -> dict[tuple[str, int], OneFieldSourceOutcome]:
    """Index one compact transcript by canonical task and trial."""
    result: dict[tuple[str, int], OneFieldSourceOutcome] = {}
    with (out / cell.transcript_file).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["task_id"]), int(row["trial"]))
            if key in result:
                raise ValueError(
                    f"Duplicate atomic outcome for {key} in {cell.results_path}"
                )
            result[key] = OneFieldSourceOutcome(
                source_cell=cell.results_path,
                trial=key[1],
                simulation_id=str(row["simulation_id"]),
                source_simulation_sha256=str(row["source_simulation_sha256"]),
                reward=float(row["reward"]),
            )
    return result


def _matched_one_field_reference(
    *,
    out: Path,
    repo_root: Path,
    by_path: dict[str, ResultCell],
    composition_commit: str,
) -> dict[str, Any]:
    """Build the paper's slot-distribution-matched single-field reference.

    Each of the 210 slots in the frozen n=2/n=3 composition design contributes
    its exact canonical atomic parent. The parent is evaluated in three regular,
    scaffolded GPT-xhigh realizations: the main run's trial 0 and trials 0/1 of
    the same-environment pass-k run. Repeated parents remain repeated because
    this row matches the composition design's slot distribution.
    """
    manifest_path = f"{_TASKS_ROOT}/bands/compose.manifest.json"
    manifest_bytes = _git_blob(repo_root, composition_commit, manifest_path)
    manifest = json.loads(manifest_bytes)
    main_path = "main_runs/intake_m_openai_xhigh_regular/results.json"
    repeat_path = "main_runs/intake_passk_xhigh_regular/results.json"
    main = _transcript_outcomes(out, by_path[main_path])
    repeats = _transcript_outcomes(out, by_path[repeat_path])
    source_keys = ((main, 0), (repeats, 0), (repeats, 1))

    rows: list[OneFieldMatchedSlot] = []
    for draw in manifest["tasks"]:
        task_id = str(draw["task_id"])
        if not task_id.startswith(("intake_c2_", "intake_c3_")):
            continue
        for slot in draw["slots"]:
            parent = str(slot["parent_task_id"])
            outcomes: list[OneFieldSourceOutcome] = []
            for source, trial in source_keys:
                key = (parent, trial)
                if key not in source:
                    raise ValueError(
                        f"Missing matched atomic outcome {parent}, trial {trial}"
                    )
                outcomes.append(source[key])
            rows.append(
                OneFieldMatchedSlot(
                    composition_task_id=task_id,
                    n_fields_in_composition=int(draw["n_entities"]),
                    slot_index=int(slot["slot_index"]),
                    parent_task_id=parent,
                    bank=str(slot["bank"]),
                    tier=str(slot["tier"]),
                    field_name=str(slot["field_name"]),
                    source_outcomes=outcomes,
                )
            )

    observations = sum(len(row.source_outcomes) for row in rows)
    passes = sum(
        outcome.reward == 1.0 for row in rows for outcome in row.source_outcomes
    )
    rate = passes / observations
    margin = 1.96 * (rate * (1.0 - rate) / observations) ** 0.5
    document = {
        "schema_version": ONE_FIELD_REFERENCE_VERSION,
        "selection_rule": (
            "For every slot occurrence in the frozen flat n=2 and n=3 compose "
            "manifest, select its exact atomic parent task in each of the three "
            "regular scaffolded GPT-xhigh realizations. Preserve repeated parent "
            "occurrences so the atomic reference matches the composition slot mix."
        ),
        "composition_manifest": {
            "source_commit": composition_commit,
            "source_path": manifest_path,
            "sha256": _sha256_bytes(manifest_bytes),
            "compose_version": manifest["compose_version"],
        },
        "source_realizations": [
            {"results_path": main_path, "trial": 0},
            {"results_path": repeat_path, "trial": 0},
            {"results_path": repeat_path, "trial": 1},
        ],
        "counts": {
            "matched_parent_instances": len(rows),
            "unique_parent_tasks": len({row.parent_task_id for row in rows}),
            "observations": observations,
            "passes": passes,
        },
        "task_pass_at_1": rate,
        "field_pass_at_1": rate,
        "wald_95_margin": margin,
        "rows": [row.model_dump(mode="json") for row in rows],
    }
    _write_json(
        out / "analysis_inputs" / "composition_one_field_matched.json", document
    )
    return document


def _export_ablation_analysis(
    out: Path,
    cells: list[ResultCell],
    prompt_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute every locally recoverable composition/protocol result."""
    by_path = {cell.results_path: cell for cell in cells}
    n2_path = "ablations/entity_composition/intake_ecomp_n2/results.json"
    n3_path = "ablations/entity_composition/intake_ecomp_n3/results.json"
    staged_path = "ablations/gated_stages/intake_ehier/results.json"
    one_field = _matched_one_field_reference(
        out=out,
        repo_root=Path(__file__).resolve().parents[3],
        by_path=by_path,
        composition_commit=by_path[n2_path].git_commit,
    )
    n2 = _score_ablation_cell(
        out,
        by_path[n2_path],
        _task_specs(_cell_task_snapshots(out, prompt_cells, n2_path)),
    )
    # The paper's frozen design uses three trials. The source root also retains
    # a fourth exploratory trial, which is exported but excluded here.
    n3 = _score_ablation_cell(
        out,
        by_path[n3_path],
        _task_specs(_cell_task_snapshots(out, prompt_cells, n3_path)),
        trials={0, 1, 2},
    )
    staged = _score_ablation_cell(
        out,
        by_path[staged_path],
        _task_specs(_cell_task_snapshots(out, prompt_cells, staged_path)),
    )

    def combine(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        observations = int(left["observations"]) + int(right["observations"])
        fields = int(left["fields"]) + int(right["fields"])
        task_passes = int(left["task_passes"]) + int(right["task_passes"])
        field_passes = int(left["field_passes"]) + int(right["field_passes"])
        return {
            "unique_tasks": int(left["unique_tasks"]) + int(right["unique_tasks"]),
            "observations": observations,
            "fields": fields,
            "task_passes": task_passes,
            "task_pass_at_1": task_passes / observations,
            "field_passes": field_passes,
            "field_pass_at_1": field_passes / fields,
        }

    document = {
        "schema_version": "tau-elicit-composition-protocol-v2",
        "composition": {
            "one_field_matched_reference": {
                "unique_tasks": one_field["counts"]["unique_parent_tasks"],
                "parent_instances": one_field["counts"]["matched_parent_instances"],
                "observations": one_field["counts"]["observations"],
                "task_passes": one_field["counts"]["passes"],
                "task_pass_at_1": one_field["task_pass_at_1"],
                "field_passes": one_field["counts"]["passes"],
                "field_pass_at_1": one_field["field_pass_at_1"],
                "wald_95_margin": one_field["wald_95_margin"],
                "ledger": "analysis_inputs/composition_one_field_matched.json",
            },
            "two_fields": n2,
            "three_fields": n3,
            "three_fields_included_trials": [0, 1, 2],
        },
        "protocol": {
            "joint_submission_without_validation": combine(n2, n3),
            "field_by_field_validation_and_retry": staged,
        },
        "source_gaps": [],
    }
    _write_json(
        out / "analysis_inputs" / "composition_protocol_recomputed.json", document
    )
    return document


def _audit_document(
    cells: list[ResultCell],
    validation: dict[str, Any],
    *,
    out: Path,
) -> dict[str, Any]:
    by_path = {row.results_path: row for row in cells}
    findings: list[dict[str, Any]] = []

    def check(code: str, condition: bool, detail: str) -> None:
        findings.append({"code": code, "ok": condition, "detail": detail})

    check(
        "paper_cell_roster",
        PAPER_AGENT_DIRECTED | PAPER_SCAFFOLDED <= set(by_path),
        f"found {len(PAPER_AGENT_DIRECTED)} agent-directed and "
        f"{len(PAPER_SCAFFOLDED)} scaffolded paper cells",
    )
    agent_calls = sum(
        by_path[path].rows for path in PAPER_AGENT_DIRECTED if path in by_path
    )
    scaffold_calls = sum(
        by_path[path].rows for path in PAPER_SCAFFOLDED if path in by_path
    )
    check(
        "paper_call_count",
        agent_calls == 2_400 and scaffold_calls == 2_400,
        f"agent-directed={agent_calls}; scaffolded={scaffold_calls}; total={agent_calls + scaffold_calls}",
    )

    expected_rates = {
        "main_runs/modeb_openai_minimal_regular_2026-09-02/results.json": 0.415,
        "main_runs/modeb_openai_xhigh_regular_2026-09-02/results.json": 0.445,
        "main_runs/modeb_gemini_high_regular_2026-09-02/results.json": 0.515,
        "main_runs/modeb_xai_10_regular_2026-09-02/results.json": 0.665,
        "main_runs/intake_m_openai_minimal_regular/results.json": 0.785,
        "main_runs/intake_m_openai_xhigh_regular/results.json": 0.770,
        "main_runs/intake_m_gemini_high_regular/results.json": 0.795,
        "main_runs/intake_m_xai_10_regular/results.json": 0.815,
    }
    mismatches = {
        path: {"expected": expected, "observed": by_path[path].pass_rate}
        for path, expected in expected_rates.items()
        if path not in by_path or abs(by_path[path].pass_rate - expected) > 1e-12
    }
    check(
        "paper_regular_pass_at_1",
        not mismatches,
        "all eight regular-realization rates match Figure 2"
        if not mismatches
        else json.dumps(mismatches, sort_keys=True),
    )
    expected_pass3 = {
        "gpt_minimal": 0.150,
        "gpt_xhigh": 0.200,
        "gemini_high": 0.135,
        "grok": 0.405,
    }
    agent_pass3 = _pass3_by_system(
        _compact_outcomes(out, cells, "paper_agent_directed")
    )
    pass3_mismatches = {
        system: {"expected": expected, "observed": agent_pass3.get(system)}
        for system, expected in expected_pass3.items()
        if abs(agent_pass3.get(system, -1.0) - expected) > 1e-12
    }
    check(
        "paper_agent_directed_pass3",
        not pass3_mismatches,
        "all four crossed-realization scores match Figure 2"
        if not pass3_mismatches
        else json.dumps(pass3_mismatches, sort_keys=True),
    )
    expected_scaffolded_pass3 = {
        "gpt_minimal": 0.455,
        "gpt_xhigh": 0.385,
        "gemini_high": 0.365,
        "grok": 0.540,
    }
    scaffolded_pass3 = _pass3_by_system(
        _compact_outcomes(out, cells, "paper_scaffolded")
    )
    scaffolded_mismatches = {
        system: {"expected": expected, "observed": scaffolded_pass3.get(system)}
        for system, expected in expected_scaffolded_pass3.items()
        if abs(scaffolded_pass3.get(system, -1.0) - expected) > 1e-12
    }
    check(
        "paper_scaffolded_pass3",
        not scaffolded_mismatches,
        "all four crossed-realization scores match Figure 2"
        if not scaffolded_mismatches
        else json.dumps(scaffolded_mismatches, sort_keys=True),
    )
    text = [row for row in cells if row.cohort == "text_control"]
    check(
        "text_control",
        len(text) == 1 and text[0].rows == 200 and text[0].passes == 200,
        f"cells={len(text)}; rows={text[0].rows if text else 0}; passes={text[0].passes if text else 0}",
    )
    speech_expected = json.loads(
        (out / "analysis_inputs" / "intake_speech_fidelity_2026-09-04.json").read_text()
    )["systems"]
    speech_observed = _speech_rollup(out / "speech_judgments" / "utterances.jsonl")
    speech_mismatches: dict[str, Any] = {}
    for system, expected in speech_expected.items():
        observed = speech_observed.get(system) or {}
        for key in (
            "utterances_submitted",
            "utterances_scored",
            "judge_errors",
            "mean_max_retained_fidelity_severity",
            "severity_2plus_count",
            "severity_2plus_rate",
        ):
            if abs(float(observed.get(key, -1)) - float(expected[key])) > 1e-12:
                speech_mismatches[f"{system}:{key}"] = {
                    "expected": expected[key],
                    "observed": observed.get(key),
                }
    check(
        "paper_speech_judgments",
        not speech_mismatches,
        "all four utterance-level speech-fidelity rows reproduce from exported judgments"
        if not speech_mismatches
        else json.dumps(speech_mismatches, sort_keys=True),
    )
    complication_checked, complication_mismatches = _complication_assignment_counts(
        out, cells
    )
    check(
        "deterministic_complication_assignments",
        complication_checked == 3_200 and complication_mismatches == 0,
        f"checked={complication_checked}; mismatches={complication_mismatches}",
    )
    ablations = json.loads(
        (out / "analysis_inputs" / "composition_protocol_recomputed.json").read_text()
    )
    composition = ablations["composition"]
    protocol = ablations["protocol"]
    ablation_values = {
        "one_task": composition["one_field_matched_reference"]["task_pass_at_1"],
        "one_field": composition["one_field_matched_reference"]["field_pass_at_1"],
        "two_task": composition["two_fields"]["task_pass_at_1"],
        "two_field": composition["two_fields"]["field_pass_at_1"],
        "three_task": composition["three_fields"]["task_pass_at_1"],
        "three_field": composition["three_fields"]["field_pass_at_1"],
        "joint_task": protocol["joint_submission_without_validation"]["task_pass_at_1"],
        "joint_field": protocol["joint_submission_without_validation"][
            "field_pass_at_1"
        ],
        "retry_task": protocol["field_by_field_validation_and_retry"]["task_pass_at_1"],
        "retry_field": protocol["field_by_field_validation_and_retry"][
            "field_pass_at_1"
        ],
    }
    expected_ablation_values = {
        "one_task": 486 / 630,
        "one_field": 486 / 630,
        "two_task": 125 / 180,
        "two_field": 290 / 360,
        "three_task": 48 / 90,
        "three_field": 202 / 270,
        "joint_task": 173 / 270,
        "joint_field": 492 / 630,
        "retry_task": 222 / 270,
        "retry_field": 544 / 630,
    }
    ablation_mismatches = {
        key: {"expected": expected, "observed": ablation_values[key]}
        for key, expected in expected_ablation_values.items()
        if abs(ablation_values[key] - expected) > 1e-12
    }
    check(
        "recoverable_composition_and_protocol_results",
        not ablation_mismatches,
        "matched one-field, two/three-field, joint-submission, and validation/retry rows reproduce exactly"
        if not ablation_mismatches
        else json.dumps(ablation_mismatches, sort_keys=True),
    )
    behavioral = json.loads(
        (out / "analysis_inputs" / "behavioral_recomputed.json").read_text()
    )
    expected_capture = {
        "gpt_xhigh": (200, 137, 56, 7, 90, 28, 24),
        "gemini_high": (199, 129, 48, 22, 83, 24, 31),
        "grok": (193, 81, 98, 14, 76, 94, 18),
    }
    capture_mismatches: dict[str, Any] = {}
    for system, expected in expected_capture.items():
        row = behavioral["behavior"][system]
        observed = (
            row["gold_fields"],
            row["attempt_wrong"],
            row["attempt_right"],
            row["attempt_unseen"],
            row["verified_given_wrong"]["numerator"],
            row["verified_given_right"]["numerator"],
            row["verified_wrong_repaired"]["numerator"],
        )
        if observed != expected:
            capture_mismatches[system] = {
                "expected": expected,
                "observed": observed,
            }
    check(
        "behavioral_capture_verification_repair",
        not capture_mismatches,
        "initial capture, verification, and repair numerators reproduce exactly"
        if not capture_mismatches
        else json.dumps(capture_mismatches, sort_keys=True),
    )
    expected_recovery = {
        "neither": (10, 98),
        "readback_only": (32, 123),
        "spelling_only": (7, 29),
        "both": (34, 97),
    }
    recovery_observed = {
        key: (int(value["numerator"]), int(value["denominator"]))
        for key, value in behavioral["pooled_initially_wrong_recovery"].items()
    }
    check(
        "pooled_repair_pathways",
        recovery_observed == expected_recovery,
        "10/98, 32/123, 7/29, and 34/97 reproduce from transcript events"
        if recovery_observed == expected_recovery
        else json.dumps(recovery_observed, sort_keys=True),
    )
    expected_noise_effort = {
        "gpt_xhigh": (0.88, 0.875),
        "gemini_high": (1.095, 0.9),
        "grok": (2.135, 2.535),
    }
    effort_mismatches = {}
    for system, expected in expected_noise_effort.items():
        row = behavioral["adaptivity_effort"][system]["noise"]
        observed = (row["baseline"], row["challenge"])
        if any(abs(a - b) > 1e-12 for a, b in zip(observed, expected, strict=True)):
            effort_mismatches[system] = {
                "expected": expected,
                "observed": observed,
            }
    check(
        "noise_adaptivity_effort",
        not effort_mismatches,
        "only Grok increases verification effort under noise"
        if not effort_mismatches
        else json.dumps(effort_mismatches, sort_keys=True),
    )
    strategy = behavioral["strategy_comparison"]
    expected_strategy = {
        "gpt_xhigh": (0.60, 0.29, 0.90, 0.64, 59.439, 87.421),
        "gemini_high": (0.55, 0.48, 0.83, 0.76, 66.370, 86.942),
        "grok": (0.77, 0.56, 0.89, 0.74, 87.086, 109.800),
    }
    strategy_mismatches = {}
    for system, expected in expected_strategy.items():
        agent = strategy[system]["agent_directed"]
        scaffolded = strategy[system]["scaffolded"]
        observed = (
            agent["easy_pass_at_1"],
            agent["hard_pass_at_1"],
            scaffolded["easy_pass_at_1"],
            scaffolded["hard_pass_at_1"],
            agent["simulated_duration_seconds"],
            scaffolded["simulated_duration_seconds"],
        )
        if any(abs(a - b) > 1e-12 for a, b in zip(observed, expected, strict=True)):
            strategy_mismatches[system] = {
                "expected": expected,
                "observed": observed,
            }
    check(
        "strategy_difficulty_and_duration",
        not strategy_mismatches,
        "easy/hard success and 21-28 second scaffold costs reproduce exactly"
        if not strategy_mismatches
        else json.dumps(strategy_mismatches, sort_keys=True),
    )
    expected_robust = {
        "medications": 1 / 60,
        "coined": 2 / 60,
        "emails": 5 / 60,
        "addresses": 6 / 60,
        "properties": 11 / 60,
        "codes": 15 / 60,
        "person_names": 18 / 60,
        "times": 26 / 60,
        "phones": 28 / 60,
        "dates": 36 / 60,
    }
    entity_mismatches = {
        bank: {
            "expected": expected,
            "observed": behavioral["entity_results"].get(bank, {}).get("pass_robust_3"),
        }
        for bank, expected in expected_robust.items()
        if abs(
            behavioral["entity_results"].get(bank, {}).get("pass_robust_3", -1)
            - expected
        )
        > 1e-12
    }
    check(
        "entity_robust_results",
        not entity_mismatches,
        "all ten entity-bank robust-success rows reproduce exactly"
        if not entity_mismatches
        else json.dumps(entity_mismatches, sort_keys=True),
    )
    realism = behavioral["realism_assignment_descriptive"]
    realism_observed = (
        realism["clean"]["numerator"],
        realism["clean"]["denominator"],
        realism["assigned"]["numerator"],
        realism["assigned"]["denominator"],
    )
    check(
        "realism_assignment_ledger",
        realism_observed == (253, 498, 674, 1302),
        "clean=253/498; assigned=674/1302"
        if realism_observed == (253, 498, 674, 1302)
        else str(realism_observed),
    )
    strict = validation["strict"]
    check(
        "strict_speech_validation",
        round(strict["precision"], 2) == 0.73
        and round(strict["recall"], 2) == 1.00
        and round(strict["f1"], 2) == 0.85,
        f"P/R/F1={strict['precision']:.4f}/{strict['recall']:.4f}/{strict['f1']:.4f}",
    )
    lenient = validation["lenient"]
    check(
        "lenient_speech_validation",
        round(lenient["precision"], 2) == 0.85
        and round(lenient["recall"], 2) == 0.73
        and round(lenient["f1"], 2) == 0.79,
        f"P/R/F1={lenient['precision']:.4f}/{lenient['recall']:.4f}/"
        f"{lenient['f1']:.4f}",
    )
    return {
        "schema_version": 1,
        "release_version": RELEASE_VERSION,
        "ok": all(row["ok"] for row in findings),
        "findings": findings,
    }


def _write_readmes(out: Path, audit: dict[str, Any]) -> None:
    status = "PASS" if audit["ok"] else "ATTENTION REQUIRED"
    source_gaps = json.loads(
        (out / "analysis_inputs" / "composition_protocol_recomputed.json").read_text()
    )["source_gaps"]
    (out / "README.md").write_text(
        "# tau-Elicitation reviewer evidence\n\n"
        "This directory is the compact, reviewer-facing evidence archive for the "
        "tau-Elicitation paper. It contains all 5,970 scored transcript records, "
        "all 6,422 available utterance-level speech-judge outputs, final human "
        "labels, "
        "exact run configurations and prompt objects, deterministic example calls, "
        "and the checked analysis inputs. The large audio/tick corpus remains a "
        "detached evidence root, is available from "
        f"[Google Drive]({DETACHED_EVIDENCE_URL}), and is linked to the compact "
        "records by SHA-256.\n\n"
        "## Contents\n\n"
        "- `results/manifest.csv`: one row per frozen results root.\n"
        "- `run_configs/`: exact recorded configuration for every cell.\n"
        "- `prompts/`: content-addressed agent policies, caller guidelines, exact "
        "historical task snapshots, and the v6 speech-judge prompt.\n"
        "- `transcripts/`: one compact JSONL file per results root, including "
        "agent tools and silent caller-side spelling/read-back events.\n"
        "- `examples/`: deterministic calls spanning both strategies and three systems.\n"
        "- `speech_judgments/`: all 6,422 exported utterance judgments: 4,948 "
        "from the paper's main speech cohort and 1,474 supplemental judgments, "
        "including retained/excluded findings and errors.\n"
        "- `judge_validation/`: direct utterance/call mapping between judge output "
        "and final human labels.\n"
        "- `analysis_inputs/`: crossed outcomes, rollups, deterministic complication "
        "draws, caller-effort ledger, deterministic behavioral recomputation, "
        "the 210-row matched one-field composition ledger, caller-voice and main "
        "significance results, and speech rollup.\n"
        "- `audit.json`: executable claim checks.\n\n"
        "`SOURCE_GAPS.md` records any remaining source or statistical-analysis "
        "re-execution gaps; both workflows in the release comparison have frozen "
        "sources.\n\n"
        "The prompt manifest distinguishes the caller guideline actually selected "
        "by the frozen simulation builders from a stale inbound guideline stored in "
        "the historical top-level run metadata. The content-addressed runtime prompt "
        "is the one used for reproduction.\n\n"
        "## Detached source corpus\n\n"
        "The approximately 41 GB frozen source corpus is available in the "
        f"[tau-elicit Google Drive folder]({DETACHED_EVIDENCE_URL}). It contains "
        "`main_runs/`, `ablations/`, and `text_channel/`; the human judge-validation "
        "data is already included in this compact archive. After downloading the "
        "corpus, pass its `tau-elicit` root as `--evidence-root`. The verifier checks "
        "the detached results and simulations against the recorded SHA-256 values.\n\n"
        "## Verify\n\n"
        "```bash\n"
        "tau2 paper elicitation-verify --root papers/tau-intake/v1/reproduction\n"
        "# With the detached 41 GB source corpus:\n"
        "tau2 paper elicitation-verify --root papers/tau-intake/v1/reproduction "
        "--evidence-root /path/to/tau-elicit\n"
        "```\n\n"
        "Verification is offline and makes no model calls. To exercise the runnable "
        "benchmark on one frozen task instead, set the required provider keys and run:\n\n"
        "```bash\n"
        "tau2 run --domain intake_free --audio-native "
        "--audio-native-provider openai --audio-native-model gpt-realtime-2 "
        "--reasoning-effort xhigh --user-llm gpt-5.5 "
        '--user-llm-args \'{"reasoning_effort":"xhigh","temperature":0.0}\' '
        "--complication-rate 1.0 --channel-effects-mode regular "
        "--speech-effects-mode regular --seed 9401 "
        "--task-ids intake_medications_hard_04 --num-trials 1 "
        "--max-concurrency 1 --timeout 1200 "
        "--save-to tau_elicitation_smoke\n"
        "```\n\n"
        "To regenerate the v6 speech judgments from the detached audio corpus, "
        "write into a new directory (frozen paper roots are protected):\n\n"
        "```bash\n"
        "tau2 judges rejudge /path/to/tau-elicit/main_runs/ONE_CELL "
        "--delivery --delivery-only --rejudge --delivery-sample-rate 1.0 "
        "--delivery-model gemini/gemini-3.1-pro-preview "
        "--max-concurrency 8 --output /path/to/rejudged/ONE_CELL\n"
        "# Add --limit-sims 1 --max-segments 1 for a one-API-call smoke test.\n"
        "```\n\n"
        "This command path was live-smoked on 2026-09-07 against one frozen agent "
        "utterance from `main_runs/modeb_gemini_high_regular_2026-09-02`. The v6 "
        "Gemini judge returned one valid judgment and zero errors. The smoke wrote "
        "only to a temporary mirrored output and did not modify frozen evidence.\n\n"
        "The paired significance analysis is also self-contained in the compact "
        "archive:\n\n"
        "```bash\n"
        "uv run --extra experiments python "
        "src/experiments/intake/main_significance.py --repo-root .\n"
        "uv run --extra experiments python "
        "src/experiments/intake/realism_effects.py --repo-root .\n"
        "uv run --extra experiments python "
        "src/experiments/intake/caller_voice_significance.py --repo-root .\n"
        "python src/experiments/intake/release_claims.py --check\n"
        "```\n\n"
        f"Current claim-audit status: **{status}**. See `AUDIT.md`.\n"
    )
    lines = ["# Frozen-evidence audit", ""]
    for finding in audit["findings"]:
        marker = "PASS" if finding["ok"] else "FAIL"
        lines.append(f"- **{marker} — {finding['code']}**: {finding['detail']}")
    lines.append("")
    (out / "AUDIT.md").write_text("\n".join(lines))
    gap_lines = ["# Source gaps", "", "## Missing frozen source artifacts", ""]
    if source_gaps:
        gap_lines.append(
            "All locally recoverable results pass the executable audit. The following "
            "older sources are still required for complete claim coverage:"
        )
        gap_lines.append("")
        for gap in source_gaps:
            gap_lines.append(f"- **{gap['paper_result']}**: {gap['detail']}")
    else:
        gap_lines.append(
            "None. Both workflows in the release comparison have frozen sources."
        )
    gap_lines.extend(
        [
            "",
            "## Missing re-execution tools",
            "",
            "None. The paper's statistical analyses are re-executable from the "
            "compact reviewer archive.",
            "",
            "## Detached source corpus",
            "",
            "The approximately 41 GB audio/tick source corpus is available from "
            f"[Google Drive]({DETACHED_EVIDENCE_URL}). Its `main_runs/`, "
            "`ablations/`, and `text_channel/` roots can be checked against the "
            "compact archive's recorded SHA-256 values with `tau2 paper "
            "elicitation-verify --evidence-root`.",
        ]
    )
    gap_lines.append("")
    (out / "SOURCE_GAPS.md").write_text("\n".join(gap_lines))


def build_release(
    *,
    evidence_root: Path,
    validation_root: Path,
    validation_run_root: Path,
    analysis_root: Path,
    out: Path,
) -> ReleaseManifest:
    """Build the complete compact reviewer archive from frozen local evidence."""
    evidence_root = evidence_root.resolve()
    out = out.resolve()
    repo_root = Path(__file__).resolve().parents[3]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    result_paths = sorted(evidence_root.rglob("results.json"))
    if len(result_paths) != 29:
        raise ValueError(f"Expected 29 frozen result roots, found {len(result_paths)}")

    speech_path = out / "speech_judgments" / "utterances.jsonl"
    speech_path.parent.mkdir(parents=True, exist_ok=True)
    speech_count = 0
    paper_speech_count = 0
    speech_counts_by_cell: dict[str, int] = {}
    transcript_count = 0
    cells: list[ResultCell] = []
    prompt_cells: list[dict[str, Any]] = []
    candidates: dict[tuple[str, str], tuple[str, TranscriptRecord, Path]] = {}

    with speech_path.open("w", encoding="utf-8") as speech_handle:
        for results_path in result_paths:
            relative = results_path.relative_to(evidence_root).as_posix()
            info, index_rows, simulations = _iter_simulations(results_path)
            cohort, system, condition = _cell_metadata(relative, info)
            transcript_relative = f"transcripts/{_slug(relative)}.jsonl"
            transcript_path = out / transcript_relative
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            config_relative = f"run_configs/{_slug(relative)}.json"
            config_document = {
                "schema_version": 1,
                "source_results": relative,
                "source_results_sha256": _sha256_file(results_path),
                "info": info,
            }
            _write_json(out / config_relative, config_document)

            environment = info.get("environment_info") or {}
            user_info = info.get("user_info") or {}
            policy_sha = _write_prompt_object(
                out, "agent_policy", str(environment.get("policy") or "")
            )
            recorded_caller_sha = _write_prompt_object(
                out,
                "recorded_caller_guidelines",
                str(user_info.get("global_simulation_guidelines") or ""),
            )
            caller_sha = _write_prompt_object(
                out, "caller_guidelines", _runtime_caller_guidelines(info)
            )
            task_snapshots = _write_task_snapshots(out, repo_root, info)
            prompt_cells.append(
                {
                    "cell": relative,
                    "agent_policy_sha256": policy_sha,
                    "caller_guidelines_sha256": caller_sha,
                    "recorded_caller_guidelines_sha256": recorded_caller_sha,
                    "recorded_snapshot_matches_runtime": (
                        caller_sha == recorded_caller_sha
                    ),
                    "task_snapshots": task_snapshots,
                    "per_call_complication": "transcript record complication field",
                }
            )

            passes = 0
            trials: set[int] = set()
            tasks: set[str] = set()
            with transcript_path.open("w", encoding="utf-8") as transcript_handle:
                for simulation, raw in simulations:
                    record = _transcript_record(
                        simulation, raw, cell=relative, cohort=cohort
                    )
                    transcript_handle.write(
                        json.dumps(
                            record.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    transcript_count += 1
                    passes += int(record.reward == 1.0)
                    trials.add(record.trial)
                    tasks.add(record.task_id)

                    for speech_row in _speech_rows(
                        simulation, cell=relative, cohort=cohort
                    ):
                        speech_handle.write(
                            json.dumps(
                                speech_row,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        speech_count += 1
                        speech_counts_by_cell[relative] = (
                            speech_counts_by_cell.get(relative, 0) + 1
                        )
                        if cohort == "paper_agent_directed":
                            paper_speech_count += 1

                    if (
                        cohort in {"paper_agent_directed", "paper_scaffolded"}
                        and condition == "regular"
                        and system in {"gpt_xhigh", "gemini_high", "grok"}
                    ):
                        mode = (
                            "agent_directed"
                            if cohort == "paper_agent_directed"
                            else "scaffolded"
                        )
                        key = (mode, system)
                        rank = _sha256_bytes(
                            f"{EXAMPLE_SELECTION_VERSION}|{mode}|{system}|{record.simulation_id}".encode()
                        )
                        if key not in candidates or rank < candidates[key][0]:
                            candidates[key] = (rank, record, results_path.parent)

            provider = (info.get("audio_native_config") or {}).get("provider")
            model = str(
                (info.get("audio_native_config") or {}).get("model")
                or (info.get("agent_info") or {}).get("llm")
                or ""
            )
            rows = len(index_rows)
            cells.append(
                ResultCell(
                    results_path=relative,
                    source_results_sha256=_sha256_file(results_path),
                    cohort=cohort,
                    system=system,
                    condition=condition,
                    rows=rows,
                    tasks=len(tasks),
                    trials=sorted(trials),
                    passes=passes,
                    pass_rate=passes / rows,
                    git_commit=str(info.get("git_commit") or ""),
                    domain=str(environment.get("domain_name") or ""),
                    provider=provider,
                    model=model,
                    reasoning_effort=(info.get("audio_native_config") or {}).get(
                        "reasoning_effort"
                    ),
                    seed=info.get("seed"),
                    complication_profile=info.get("complication_profile"),
                    complication_rate=info.get("complication_rate"),
                    channel_effects_mode=info.get("channel_effects_mode"),
                    speech_effects_mode=info.get("speech_effects_mode"),
                    config_file=config_relative,
                    transcript_file=transcript_relative,
                    transcript_sha256=_sha256_file(transcript_path),
                )
            )

    speech_prompt_source = (
        Path(__file__).parent / "prompts" / "elicitation_speech_v6.md"
    )
    speech_prompt_target = out / "prompts" / "speech_judge" / "v6.md"
    speech_prompt_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(speech_prompt_source, speech_prompt_target)
    _write_json(
        out / "prompts" / "manifest.json",
        {
            "cells": prompt_cells,
            "caller_guidelines_contract": {
                "selection": "Task.agent_opener selects the outbound caller prompt",
                "recorded_snapshot_note": (
                    "The historical Info snapshot resolved task_set_name=None before "
                    "falling back to the domain and therefore recorded the inbound "
                    "default. The frozen simulation builders resolved each task's "
                    "agent opener and used the exported outbound runtime prompt."
                ),
            },
            "speech_judge": {
                "version": "v6",
                "path": "prompts/speech_judge/v6.md",
                "sha256": _sha256_file(speech_prompt_target),
            },
        },
    )
    _write_json(
        out / "speech_judgments" / "manifest.json",
        {
            "schema_version": 1,
            "file": "speech_judgments/utterances.jsonl",
            "total_rows": speech_count,
            "paper_main_rows": paper_speech_count,
            "supplemental_rows": speech_count - paper_speech_count,
            "counts_by_cell": dict(sorted(speech_counts_by_cell.items())),
            "paper_model": "gemini/gemini-3.1-pro-preview",
            "paper_prompt_version": "v6",
            "paper_finding_filter_version": "v1",
            "paper_fidelity_end_exclusion_seconds": 1.0,
        },
    )
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(ResultCell.model_fields)
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            row = cell.model_dump(mode="json")
            row["trials"] = json.dumps(row["trials"], separators=(",", ":"))
            writer.writerow(row)

    analysis_out = out / "analysis_inputs"
    analysis_out.mkdir(parents=True, exist_ok=True)
    for name in ANALYSIS_INPUTS:
        source = analysis_root / name
        if not source.exists():
            raise FileNotFoundError(f"Required analysis input missing: {source}")
        shutil.copy2(source, analysis_out / name)
    paper_analysis_root = repo_root / "papers" / "tau-intake" / "v1" / "analysis"
    for name in PAPER_ANALYSIS_INPUTS:
        source = paper_analysis_root / name
        if not source.exists():
            raise FileNotFoundError(f"Required paper analysis missing: {source}")
        shutil.copy2(source, analysis_out / name)
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "src" / "experiments" / "intake" / "release_claims.py"),
            "--release-root",
            str(out),
            "--out",
            str(analysis_out / "behavioral_recomputed.json"),
        ],
        check=True,
    )
    _export_ablation_analysis(out, cells, prompt_cells)

    examples: list[ExampleCall] = []
    for (mode, system), (_, record, cell_root) in sorted(candidates.items()):
        base = f"{mode}_{system}_{record.simulation_id}"
        transcript_relative = f"examples/{base}.json"
        _write_json(out / transcript_relative, record.model_dump(mode="json"))
        audio = _source_audio(cell_root, record.simulation_id)
        if audio is not None:
            shutil.copy2(audio, out / "examples" / f"{base}.wav")
        examples.append(
            ExampleCall(
                mode=mode,
                system=system,
                simulation_id=record.simulation_id,
                task_id=record.task_id,
                reward=record.reward,
                transcript_file=transcript_relative,
                source_cell=record.cell,
                source_simulation_sha256=record.source_simulation_sha256,
            )
        )
    _write_json(
        out / "examples" / "manifest.json",
        {
            "selection_version": EXAMPLE_SELECTION_VERSION,
            "rule": "smallest SHA-256 rank per strategy and main system",
            "calls": [row.model_dump(mode="json") for row in examples],
        },
    )

    validation_call_count, validation_finding_count, validation_metrics = (
        _export_validation(validation_root, validation_run_root, out)
    )
    audit = _audit_document(cells, validation_metrics, out=out)
    _write_json(out / "audit.json", audit)
    _write_readmes(out, audit)

    artifacts: list[ArtifactFile] = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path != out / "manifest.json":
            relative = path.relative_to(out).as_posix()
            rows: Optional[int] = None
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    rows = sum(1 for _ in handle)
            artifacts.append(
                ArtifactFile(
                    path=relative,
                    sha256=_sha256_file(path),
                    bytes=path.stat().st_size,
                    rows=rows,
                )
            )
    manifest = ReleaseManifest(
        evidence_root_name=evidence_root.name,
        result_cells=cells,
        transcript_count=transcript_count,
        speech_judgment_count=speech_count,
        paper_speech_judgment_count=paper_speech_count,
        speech_judgment_counts_by_cell=dict(sorted(speech_counts_by_cell.items())),
        validation_call_count=validation_call_count,
        validation_finding_count=validation_finding_count,
        examples=examples,
        artifacts=artifacts,
    )
    _write_json(out / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


def verify_release(
    root: Path, *, evidence_root: Optional[Path] = None
) -> VerificationReport:
    """Verify internal hashes and, when supplied, every detached source result."""
    root = root.resolve()
    manifest = ReleaseManifest.model_validate_json((root / "manifest.json").read_text())
    checks: list[VerificationCheck] = []
    for artifact in manifest.artifacts:
        path = root / artifact.path
        actual = _sha256_file(path) if path.exists() else "missing"
        checks.append(
            VerificationCheck(
                code=f"artifact:{artifact.path}",
                ok=actual == artifact.sha256,
                detail=f"expected={artifact.sha256}; actual={actual}",
            )
        )
    checks.append(
        VerificationCheck(
            code="result_cell_count",
            ok=len(manifest.result_cells) == 29,
            detail=f"observed={len(manifest.result_cells)}; expected=29",
        )
    )
    checks.append(
        VerificationCheck(
            code="transcript_count",
            ok=manifest.transcript_count == 5_970,
            detail=f"observed={manifest.transcript_count}; expected=5970",
        )
    )
    transcript_rows = sum(
        artifact.rows or 0
        for artifact in manifest.artifacts
        if artifact.path.startswith("transcripts/") and artifact.path.endswith(".jsonl")
    )
    checks.append(
        VerificationCheck(
            code="transcript_artifact_rows",
            ok=transcript_rows == manifest.transcript_count,
            detail=(
                f"artifact_rows={transcript_rows}; manifest={manifest.transcript_count}"
            ),
        )
    )
    checks.append(
        VerificationCheck(
            code="speech_judgment_count",
            ok=manifest.speech_judgment_count == 6_422,
            detail=f"observed={manifest.speech_judgment_count}; expected=6422",
        )
    )
    checks.append(
        VerificationCheck(
            code="paper_speech_judgment_count",
            ok=manifest.paper_speech_judgment_count == 4_948,
            detail=(f"observed={manifest.paper_speech_judgment_count}; expected=4948"),
        )
    )
    checks.append(
        VerificationCheck(
            code="speech_judgment_cell_sum",
            ok=sum(manifest.speech_judgment_counts_by_cell.values())
            == manifest.speech_judgment_count,
            detail=(
                f"cell_sum={sum(manifest.speech_judgment_counts_by_cell.values())}; "
                f"total={manifest.speech_judgment_count}"
            ),
        )
    )
    checks.append(
        VerificationCheck(
            code="validation_counts",
            ok=manifest.validation_call_count == 100
            and manifest.validation_finding_count == 41,
            detail=(
                f"calls={manifest.validation_call_count}; "
                f"findings={manifest.validation_finding_count}"
            ),
        )
    )
    example_audio = list((root / "examples").glob("*.wav"))
    checks.append(
        VerificationCheck(
            code="example_calls",
            ok=len(manifest.examples) == 6 and len(example_audio) == 6,
            detail=f"transcripts={len(manifest.examples)}; audio={len(example_audio)}",
        )
    )
    prompt_manifest = json.loads((root / "prompts" / "manifest.json").read_text())
    snapshots = {
        row["archive_path"]: row["sha256"]
        for cell in prompt_manifest["cells"]
        for row in cell["task_snapshots"]
    }
    snapshot_mismatches: dict[str, str] = {}
    for path, digest in snapshots.items():
        target = root / path
        actual = _sha256_file(target) if target.exists() else "missing"
        if actual != digest:
            snapshot_mismatches[path] = actual
    checks.append(
        VerificationCheck(
            code="historical_task_snapshots",
            ok=bool(snapshots) and not snapshot_mismatches,
            detail=(
                f"unique_snapshots={len(snapshots)}; "
                f"mismatches={len(snapshot_mismatches)}"
            ),
        )
    )
    audit = json.loads((root / "audit.json").read_text())
    checks.append(
        VerificationCheck(
            code="claim_audit",
            ok=bool(audit.get("ok")),
            detail=(
                f"passed={sum(bool(row['ok']) for row in audit['findings'])}/"
                f"{len(audit['findings'])}"
            ),
        )
    )
    if evidence_root is not None:
        evidence_root = evidence_root.resolve()
        for cell in manifest.result_cells:
            source = evidence_root / cell.results_path
            actual = _sha256_file(source) if source.exists() else "missing"
            checks.append(
                VerificationCheck(
                    code=f"source:{cell.results_path}",
                    ok=actual == cell.source_results_sha256,
                    detail=f"expected={cell.source_results_sha256}; actual={actual}",
                )
            )
    ok = all(check.ok for check in checks)
    return VerificationReport(
        ok=ok,
        summary=f"{'PASS' if ok else 'FAIL'}: {sum(c.ok for c in checks)}/{len(checks)} checks passed",
        checks=checks,
    )

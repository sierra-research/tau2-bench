# Copyright Sierra
"""Deterministic τ-Multilingual frozen-evidence audit.

This module does not call an LLM and never modifies a source result. It resolves
the paper cohort from explicit path templates, validates cell shape and run
configuration, reproduces task-success and text-control values, checks the
current language packs, and selects a deterministic listening sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from tau2.multilingual.factory.guardrails import validate_final_pack
from tau2.utils import DATA_DIR
from tau2.utils.utils import get_commit_hash

LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "hi": "Hindi",
    "ko": "Korean",
    "zh": "Mandarin",
}
DOMAINS = ("airline", "retail", "telecom")
VOICE_SYSTEMS = (
    "openai_minimal",
    "openai_xhigh",
    "gemini_minimal",
    "gemini_high",
    "xai_provider_default",
)
REPEATED_SYSTEMS = VOICE_SYSTEMS[:4]
TEXT_SYSTEMS = ("gpt55_xhigh", "gemini31pro_high")
AUDIT_VERSION = "tau-multilingual-audit-v1"
LISTENING_VERSION = "tau-multilingual-listening-v1"
XAI_MODEL = "grok-voice-think-fast-1.0"
RUNTIME_PACK_FIELDS = {
    "acoustic_presets",
    "agent_greeting",
    "agent_language_clause",
    "agent_native_script_db_clause",
    "backchannel_level",
    "default_out_of_turn_events_per_minute",
    "localization",
    "personas",
    "text_input",
}

PAPER_TASK_SUCCESS: dict[str, tuple[float, ...]] = {
    "en": (45.3, 64.7, 40.7, 51.3, 76.0),
    "es": (46.0, 58.0, 37.3, 60.0, 78.0),
    "pt": (49.3, 69.3, 38.0, 58.7, 78.7),
    "hi": (43.3, 61.3, 38.7, 60.0, 76.0),
    "ko": (26.0, 29.3, 26.0, 37.3, 75.3),
    "zh": (30.0, 44.0, 28.0, 50.0, 69.3),
}
PAPER_TRIAL_STABILITY: dict[str, tuple[float, float, float, float]] = {
    "en": (50.5, 51.8, 51.2, 6.0),
    "es": (50.3, 49.5, 49.9, 3.3),
    "pt": (53.8, 53.2, 53.5, 4.0),
    "hi": (50.8, 48.3, 49.6, 4.7),
    "ko": (29.7, 31.3, 30.5, 10.0),
    "zh": (38.0, 37.7, 37.8, 5.3),
}
PAPER_TEXT_VOICE_GAPS: dict[str, float] = {
    "en": 34.1,
    "es": 16.8,
    "pt": 21.2,
    "hi": 25.1,
    "ko": 41.5,
    "zh": 26.7,
}
PAPER_RETAIL_ABLATIONS: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("hi", "openai_xhigh"): (63.3, 63.3, 63.3, 36.7),
    ("hi", "gemini_high"): (60.0, 66.7, 60.0, 60.0),
    ("zh", "openai_xhigh"): (30.0, 46.7, 30.0, 46.7),
    ("zh", "gemini_high"): (43.3, 53.3, 43.3, 60.0),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


class Finding(BaseModel):
    """One check and its supporting evidence."""

    model_config = ConfigDict(frozen=True)

    status: Annotated[
        Literal["pass", "warning", "fail"],
        Field(description="Machine-readable severity of the audit result."),
    ]
    code: Annotated[str, Field(description="Stable identifier for the check.")]
    message: Annotated[str, Field(description="Human-readable audit result.")]
    evidence: Annotated[
        dict[str, object], Field(description="Structured evidence for the result.")
    ] = Field(default_factory=dict)


class ResultCell(BaseModel):
    """Validated identity and outcome summary for one results file."""

    model_config = ConfigDict(frozen=True)

    cohort: Annotated[
        Literal["voice", "text"], Field(description="Evaluation modality cohort.")
    ]
    language: Annotated[str, Field(description="ISO-style language code.")]
    domain: Annotated[str, Field(description="Benchmark domain name.")]
    system: Annotated[str, Field(description="Stable system-arm slug.")]
    results_path: Annotated[
        str, Field(description="Path relative to the frozen evidence root.")
    ]
    sha256: Annotated[str, Field(description="SHA-256 of the results file.")]
    git_commit: Annotated[
        str, Field(description="Repository revision recorded by the run.")
    ]
    rows: Annotated[int, Field(description="Number of simulation-index rows.")]
    unique_tasks: Annotated[
        int, Field(description="Number of canonical task identifiers.")
    ]
    trials: Annotated[list[int], Field(description="Recorded trial identifiers.")]
    trial_success: Annotated[
        dict[int, float], Field(description="Mean binary reward by trial.")
    ]
    provider: Annotated[
        Optional[str], Field(description="Recorded audio-native provider.")
    ] = None
    model: Annotated[
        Optional[str], Field(description="Recorded audio-native model.")
    ] = None
    reasoning_effort: Annotated[
        Optional[str], Field(description="Recorded audio-native reasoning setting.")
    ] = None
    task_subset: Annotated[
        Optional[str], Field(description="Recorded fixed-subset name, when present.")
    ] = None
    task_ids_sha256: Annotated[
        str, Field(description="Digest of sorted canonical task identifiers.")
    ]
    matches_expected_subset: Annotated[
        bool, Field(description="Whether task ids equal the shipped fixed subset.")
    ]


class TableRow(BaseModel):
    """A language row in a reproduced paper table."""

    language: Annotated[str, Field(description="Language row identifier.")]
    values: Annotated[
        dict[str, float], Field(description="Named reproduced numeric values.")
    ]


class AblationRow(BaseModel):
    """One language-system row in the retail localization ablation table."""

    language: Annotated[str, Field(description="Language row identifier.")]
    system: Annotated[str, Field(description="Voice-system arm slug.")]
    localized: Annotated[
        float, Field(description="Localized romanized baseline Pass@1 percentage.")
    ]
    source: Annotated[
        float, Field(description="Source-English entity Pass@1 percentage.")
    ]
    romanized: Annotated[
        float, Field(description="Romanized database baseline Pass@1 percentage.")
    ]
    native: Annotated[
        float, Field(description="Native-script database Pass@1 percentage.")
    ]


class ListeningCall(BaseModel):
    """One reproducibly chosen listening example."""

    language: Annotated[str, Field(description="Language code.")]
    domain: Annotated[str, Field(description="Benchmark domain.")]
    system: Annotated[str, Field(description="Selected voice-system arm.")]
    sim_id: Annotated[str, Field(description="Frozen simulation identifier.")]
    task_id: Annotated[str, Field(description="Recorded task identifier.")]
    reward: Annotated[
        Optional[float], Field(description="Recorded task reward, if present.")
    ]
    source_audio: Annotated[
        str, Field(description="Audio path relative to the evidence root.")
    ]
    sha256: Annotated[str, Field(description="SHA-256 of the mixed-call audio.")]
    bytes: Annotated[int, Field(description="Mixed-call audio size in bytes.")]
    exported_file: Annotated[
        Optional[str], Field(description="Portable export filename, when copied.")
    ] = None


class ListeningManifest(BaseModel):
    """The three-calls-per-language listening sample contract."""

    schema_version: Annotated[
        int, Field(description="Serialized manifest schema version.")
    ] = 1
    selection_version: Annotated[
        str, Field(description="Versioned deterministic selection rule.")
    ] = LISTENING_VERSION
    calls_per_language: Annotated[
        int, Field(description="Required number of calls for every language.")
    ] = 3
    rule: Annotated[str, Field(description="Human-readable selection rule.")] = (
        "One call per domain from fixed arms; choose the available call with the "
        "smallest SHA-256 rank over version, language, domain, system, sim_id."
    )
    calls: Annotated[
        list[ListeningCall], Field(description="Selected listening examples.")
    ]


class MultilingualAudit(BaseModel):
    """Machine-readable frozen-evidence audit and reproduced core tables."""

    schema_version: Annotated[
        int, Field(description="Serialized audit schema version.")
    ] = 1
    audit_version: Annotated[
        str, Field(description="Versioned audit computation contract.")
    ] = AUDIT_VERSION
    repository_commit: Annotated[
        str, Field(description="Repository revision used for the audit.")
    ]
    evidence_root: Annotated[
        str, Field(description="Logical name of the frozen-evidence directory.")
    ]
    artifact_id: Annotated[str, Field(description="Content-derived audit identifier.")]
    findings: Annotated[list[Finding], Field(description="All audit findings.")]
    voice_cells: Annotated[
        list[ResultCell], Field(description="Validated voice result cells.")
    ]
    text_cells: Annotated[
        list[ResultCell], Field(description="Validated text result cells.")
    ]
    task_success: Annotated[
        list[TableRow], Field(description="Reproduced trial-0 voice Pass@1 table.")
    ]
    text_success: Annotated[
        list[TableRow], Field(description="Reproduced text Pass@1 table.")
    ]
    text_voice_gap: Annotated[
        list[TableRow], Field(description="Reproduced pooled text-minus-voice gaps.")
    ]
    trial_stability: Annotated[
        list[TableRow], Field(description="Reproduced repeated-system trial panel.")
    ]
    retail_ablations: Annotated[
        list[AblationRow], Field(description="Reproduced retail localization table.")
    ]
    listening: Annotated[
        ListeningManifest, Field(description="Deterministic listening sample.")
    ]

    @property
    def ok(self) -> bool:
        return not any(f.status == "fail" for f in self.findings)

    @property
    def summary_line(self) -> str:
        counts = {status: 0 for status in ("pass", "warning", "fail")}
        for finding in self.findings:
            counts[finding.status] += 1
        return (
            f"τ-Multilingual audit: {counts['pass']} pass, "
            f"{counts['warning']} warning, {counts['fail']} fail"
        )


def _voice_group(language: str, domain: str) -> str:
    label = LANGUAGES[language].lower()
    if domain == "airline":
        return "airline_en_v1" if language == "en" else f"airline_v1_{label}_airline"
    if domain == "retail":
        return f"retail_v1_{label}_retail"
    prefix = (
        "preference_strat50" if language in {"en", "es", "pt"} else "preference_runs_v1"
    )
    return f"{prefix}_{label}_telecom"


def _voice_result_path(root: Path, language: str, domain: str, system: str) -> Path:
    label = LANGUAGES[language].lower()
    if system == "xai_provider_default":
        group = f"{domain}_xai_v2_{label}_{domain}"
    else:
        group = _voice_group(language, domain)
    return root / "main_runs" / group / f"{language}_{domain}_{system}" / "results.json"


def _text_result_path(root: Path, language: str, domain: str, system: str) -> Path:
    label = LANGUAGES[language].lower()
    group = f"multilingual_text_v1_{label}_{domain}"
    return (
        root / "text_channel" / group / f"{language}_{domain}_{system}" / "results.json"
    )


def _canonical_task_id(task_id: str, language: str) -> str:
    for suffix in (
        f"_{language}_identity_native",
        f"_{language}_identity",
        f"_{language}",
    ):
        if task_id.endswith(suffix):
            return task_id[: -len(suffix)]
    return task_id


def _load_cell(
    path: Path,
    root: Path,
    cohort: Literal["voice", "text"],
    language: str,
    domain: str,
    system: str,
) -> ResultCell:
    payload = json.loads(path.read_text())
    info = payload["info"]
    rows = payload.get("simulation_index") or []
    trials = sorted({int(row.get("trial") or 0) for row in rows})
    success: dict[int, float] = {}
    for trial in trials:
        rewards = [
            row.get("reward")
            for row in rows
            if int(row.get("trial") or 0) == trial and row.get("reward") is not None
        ]
        success[trial] = sum(float(value) for value in rewards) / len(rewards)
    audio = info.get("audio_native_config") or {}
    subset = info.get("task_subset") or {}
    canonical_ids = sorted(
        {_canonical_task_id(str(row["task_id"]), language) for row in rows}
    )
    expected_subset = json.loads(
        (DATA_DIR / "tau2" / "task_subsets" / f"{domain}_50.json").read_text()
    )["task_ids"]
    return ResultCell(
        cohort=cohort,
        language=language,
        domain=domain,
        system=system,
        results_path=path.relative_to(root).as_posix(),
        sha256=_sha256_file(path),
        git_commit=info.get("git_commit") or "",
        rows=len(rows),
        unique_tasks=len(canonical_ids),
        trials=trials,
        trial_success=success,
        provider=audio.get("provider"),
        model=audio.get("model"),
        reasoning_effort=audio.get("reasoning_effort"),
        task_subset=subset.get("name"),
        task_ids_sha256=_sha256_bytes(
            json.dumps(
                canonical_ids, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ),
        matches_expected_subset=set(canonical_ids) == set(expected_subset),
    )


def _cohort_fingerprint(cells: list[ResultCell]) -> str:
    lines = [
        f"{cell.results_path}\t{cell.sha256}\t{cell.rows}\n"
        for cell in sorted(cells, key=lambda item: item.results_path)
    ]
    return _sha256_bytes("".join(lines).encode())


def _domain_localization(value: object, domain: str) -> object:
    """Restrict localization comparison to values a domain can consume."""
    if not isinstance(value, dict):
        return value
    projected = dict(value)
    glossaries = projected.get("domain_glossaries")
    if isinstance(glossaries, dict):
        projected["domain_glossaries"] = {domain: glossaries.get(domain)}
    return projected


def _effective_pack_diff(
    current: dict, snapshot: dict, *, cohort: str, domain: str, system: str
) -> list[str]:
    """Compare only language-pack fields that could affect one recorded cell."""
    active = set(RUNTIME_PACK_FIELDS)
    if cohort != "text":
        active.discard("text_input")
    if not system.endswith("native_script"):
        active.discard("agent_native_script_db_clause")
    if cohort == "text":
        active -= {
            "acoustic_presets",
            "backchannel_level",
            "default_out_of_turn_events_per_minute",
        }
    changed = []
    for field in sorted(active):
        old = snapshot.get(field)
        new = current.get(field)
        if field == "localization":
            old = _domain_localization(old, domain)
            new = _domain_localization(new, domain)
        if old != new:
            changed.append(field)
    return changed


def _pack_snapshot_drift(repo: Path, language: str) -> dict[str, list[str]]:
    """Compare the current pack with exact archived run-time pack snapshots."""
    archive = repo / "papers" / "tau-multilingual" / "reproduction" / "prompts"
    manifest = json.loads((archive / "manifest.json").read_text())
    current = (
        yaml.safe_load(
            (repo / f"data/tau2/multilingual/{language}/pack.yaml").read_text()
        )
        or {}
    )
    changed: dict[str, list[str]] = {}
    for cell in manifest["cells"]:
        if cell["language"] != language:
            continue
        object_hash = cell["objects"]["runtime_language_pack"]
        snapshot_path = (
            archive / "objects" / "runtime_language_pack" / f"{object_hash}.json"
        )
        snapshot = json.loads(snapshot_path.read_text())
        fields = _effective_pack_diff(
            current,
            snapshot,
            cohort=cell["cohort"],
            domain=cell["domain"],
            system=cell["system"],
        )
        if fields:
            profile = f"{cell['cohort']}/{cell['domain']}/{cell['system']}"
            changed[profile] = fields
    return dict(sorted(changed.items()))


def _audit_xai_urls(root: Path, xai_cells: list[ResultCell]) -> Finding:
    pattern = re.compile(rb"wss://api\.x\.ai/v1/realtime\?model=([^\s]+)")
    models: dict[str, int] = defaultdict(int)
    log_count = 0
    for cell in xai_cells:
        cell_dir = root / cell.results_path
        for log in cell_dir.parent.glob("artifacts/**/task.log"):
            log_count += 1
            for model in pattern.findall(log.read_bytes()):
                models[model.decode(errors="replace")] += 1
    exact = set(models) == {XAI_MODEL} and models.get(XAI_MODEL, 0) > 0
    return Finding(
        status="pass" if exact else "fail",
        code="xai-websocket-model",
        message=(
            f"Every recorded xAI websocket URL uses {XAI_MODEL}"
            if exact
            else "xAI task logs contain a missing or unexpected websocket model"
        ),
        evidence={
            "models": dict(sorted(models.items())),
            "task_logs_scanned": log_count,
        },
    )


def _audit_prompt_archive(repo: Path) -> Finding:
    """Require a complete, hash-valid prompt archive in the reviewer tree."""
    from tau2.paper.prompt_snapshots import (
        PromptSnapshotManifest,
        PromptSourceVerification,
        verify_prompt_snapshot_export,
    )

    archive = repo / "papers" / "tau-multilingual" / "reproduction" / "prompts"
    manifest_path = archive / "manifest.json"
    source_path = archive / "source_verification.json"
    if not manifest_path.is_file() or not source_path.is_file():
        return Finding(
            status="fail",
            code="prompt-snapshot-archive",
            message="The checked-in frozen prompt archive is missing",
            evidence={"path": str(archive.relative_to(repo))},
        )
    manifest = PromptSnapshotManifest.model_validate_json(manifest_path.read_text())
    source = PromptSourceVerification.model_validate_json(source_path.read_text())
    integrity = verify_prompt_snapshot_export(archive)
    complete = (
        integrity.ok
        and source.ok
        and source.checked_cells == 134
        and source.checked_simulations == 10140
        and source.logged_user_prompts_compared == 9895
        and source.logged_agent_prompts_compared == 2700
        and len(manifest.cells) == 134
        and sum(len(cell.simulations) for cell in manifest.cells) == 10140
    )
    return Finding(
        status="pass" if complete else "fail",
        code="prompt-snapshot-archive",
        message=(
            "All 10,140 scored simulations have source-verified frozen prompts"
            if complete
            else "The frozen prompt archive is incomplete or fails integrity checks"
        ),
        evidence={
            "artifact_id": manifest.artifact_id,
            "cells": len(manifest.cells),
            "simulations": sum(len(cell.simulations) for cell in manifest.cells),
            "objects": len(manifest.objects),
            "source_checked_cells": source.checked_cells,
            "source_checked_simulations": source.checked_simulations,
            "logged_user_prompts_compared": source.logged_user_prompts_compared,
            "logged_agent_prompts_compared": source.logged_agent_prompts_compared,
            "integrity_problems": integrity.problems,
            "source_problems": source.problems,
        },
    )


def _validate_cells(cells: list[ResultCell], *, voice: bool) -> list[Finding]:
    findings: list[Finding] = []
    expected_rows = 100 if voice else 50
    for cell in cells:
        repeated = voice and cell.system != "xai_provider_default"
        cell_expected = expected_rows if repeated else 50
        expected_trials = [0, 1] if repeated else [0]
        problems: list[str] = []
        if cell.rows != cell_expected:
            problems.append(f"rows={cell.rows}, expected {cell_expected}")
        if cell.unique_tasks != 50:
            problems.append(f"unique_tasks={cell.unique_tasks}, expected 50")
        if cell.trials != expected_trials:
            problems.append(f"trials={cell.trials}, expected {expected_trials}")
        if not cell.matches_expected_subset:
            problems.append("canonical task ids do not match the frozen subset")
        if cell.task_subset not in {None, f"{cell.domain}_50"}:
            problems.append(
                f"task_subset={cell.task_subset!r}, expected {cell.domain + '_50'!r}"
            )
        if voice:
            provider, effort = cell.system.split("_", 1)
            if cell.provider != provider:
                problems.append(f"provider={cell.provider!r}, expected {provider!r}")
            if cell.reasoning_effort != effort:
                problems.append(
                    f"reasoning_effort={cell.reasoning_effort!r}, expected {effort!r}"
                )
            if provider == "xai" and cell.model != XAI_MODEL:
                problems.append(f"xAI model={cell.model!r}, expected {XAI_MODEL!r}")
        metadata_gap = cell.task_subset is None and not problems
        findings.append(
            Finding(
                status="fail" if problems else "warning" if metadata_gap else "pass",
                code="result-cell",
                message=(
                    f"{cell.language}/{cell.domain}/{cell.system}: "
                    + (
                        "; ".join(problems)
                        if problems
                        else "task ids match the frozen subset, but task_subset metadata is absent"
                        if metadata_gap
                        else "shape and configuration match"
                    )
                ),
                evidence={
                    "results_path": cell.results_path,
                    "sha256": cell.sha256,
                    "task_ids_sha256": cell.task_ids_sha256,
                },
            )
        )
    return findings


def _task_success_table(cells: list[ResultCell]) -> list[TableRow]:
    by_key = {(cell.language, cell.domain, cell.system): cell for cell in cells}
    table: list[TableRow] = []
    for language in LANGUAGES:
        values = {}
        for system in VOICE_SYSTEMS:
            rates = [
                by_key[(language, domain, system)].trial_success[0]
                for domain in DOMAINS
            ]
            values[system] = 100 * sum(rates) / len(rates)
        table.append(TableRow(language=language, values=values))
    return table


def _text_success_table(cells: list[ResultCell]) -> list[TableRow]:
    by_key = {(cell.language, cell.domain, cell.system): cell for cell in cells}
    table: list[TableRow] = []
    for language in LANGUAGES:
        values = {}
        for system in TEXT_SYSTEMS:
            rates = [
                by_key[(language, domain, system)].trial_success[0]
                for domain in DOMAINS
            ]
            values[system] = 100 * sum(rates) / len(rates)
        values["pooled"] = sum(values.values()) / len(TEXT_SYSTEMS)
        table.append(TableRow(language=language, values=values))
    return table


def _text_voice_gap_table(
    task_success: list[TableRow], text_success: list[TableRow]
) -> list[TableRow]:
    voice_by_language = {row.language: row for row in task_success}
    text_by_language = {row.language: row for row in text_success}
    rows: list[TableRow] = []
    for language in LANGUAGES:
        voice = sum(voice_by_language[language].values.values()) / len(VOICE_SYSTEMS)
        text = text_by_language[language].values["pooled"]
        rows.append(
            TableRow(
                language=language,
                values={"text": text, "voice": voice, "gap": text - voice},
            )
        )
    return rows


def _trial_stability_table(cells: list[ResultCell]) -> list[TableRow]:
    by_key = {(cell.language, cell.domain, cell.system): cell for cell in cells}
    rows: list[TableRow] = []
    for language in LANGUAGES:
        trial_rates: dict[int, list[float]] = {0: [], 1: []}
        system_gaps = []
        for system in REPEATED_SYSTEMS:
            by_trial = {
                trial: [
                    by_key[(language, domain, system)].trial_success[trial]
                    for domain in DOMAINS
                ]
                for trial in (0, 1)
            }
            for trial, rates in by_trial.items():
                trial_rates[trial].extend(rates)
            system_gaps.append(
                100
                * abs(
                    sum(by_trial[1]) / len(by_trial[1])
                    - sum(by_trial[0]) / len(by_trial[0])
                )
            )
        trial_0 = 100 * sum(trial_rates[0]) / len(trial_rates[0])
        trial_1 = 100 * sum(trial_rates[1]) / len(trial_rates[1])
        rows.append(
            TableRow(
                language=language,
                values={
                    "trial_0": trial_0,
                    "trial_1": trial_1,
                    "two_trial_mean": (trial_0 + trial_1) / 2,
                    "max_system_gap": max(system_gaps),
                },
            )
        )
    return rows


def _result_success_for_subset(
    path: Path, language: str, expected_task_ids: set[str]
) -> tuple[float, bool, int]:
    payload = json.loads(path.read_text())
    rows = [
        row
        for row in payload.get("simulation_index", [])
        if int(row.get("trial") or 0) == 0
        and _canonical_task_id(str(row["task_id"]), language) in expected_task_ids
    ]
    canonical_ids = {_canonical_task_id(str(row["task_id"]), language) for row in rows}
    rewards = [float(row["reward"]) for row in rows if row.get("reward") is not None]
    if not rewards:
        raise ValueError(f"no scored rows in {path}")
    return (
        100 * sum(rewards) / len(rewards),
        canonical_ids == expected_task_ids,
        len(rows),
    )


def _retail_ablation_path(
    root: Path, language: str, system: str, *, native_script: bool
) -> Path:
    label = LANGUAGES[language].lower()
    arm = "retail_dbscript_v1" if native_script else "retail_unlocalized_v1"
    return (
        root
        / "ablations"
        / "localization"
        / f"{arm}_{label}_retail"
        / f"{language}_retail_{system}"
        / "results.json"
    )


def _retail_ablation_table(root: Path) -> tuple[list[AblationRow], list[Finding]]:
    expected = set(
        json.loads((DATA_DIR / "tau2" / "task_subsets" / "retail_30.json").read_text())[
            "task_ids"
        ]
    )
    rows: list[AblationRow] = []
    findings: list[Finding] = []
    for language in ("hi", "zh"):
        for system in ("openai_xhigh", "gemini_high"):
            baseline_path = _voice_result_path(root, language, "retail", system)
            source_path = _retail_ablation_path(
                root, language, system, native_script=False
            )
            native_path = _retail_ablation_path(
                root, language, system, native_script=True
            )
            paths = (baseline_path, source_path, native_path)
            if any(not path.exists() for path in paths):
                findings.append(
                    Finding(
                        status="fail",
                        code="retail-ablation-cell",
                        message=f"Missing retail ablation input for {language}/{system}",
                        evidence={
                            "missing": [
                                str(path) for path in paths if not path.exists()
                            ]
                        },
                    )
                )
                continue
            localized, baseline_match, baseline_rows = _result_success_for_subset(
                baseline_path, language, expected
            )
            source, source_match, source_rows = _result_success_for_subset(
                source_path, language, expected
            )
            native, native_match, native_rows = _result_success_for_subset(
                native_path, language, expected
            )
            row = AblationRow(
                language=language,
                system=system,
                localized=localized,
                source=source,
                romanized=localized,
                native=native,
            )
            observed = tuple(
                round(value, 1)
                for value in (row.localized, row.source, row.romanized, row.native)
            )
            frame_ok = (
                baseline_match
                and source_match
                and native_match
                and (baseline_rows, source_rows, native_rows) == (30, 30, 30)
            )
            expected_values = PAPER_RETAIL_ABLATIONS[(language, system)]
            findings.append(
                Finding(
                    status=(
                        "pass" if frame_ok and observed == expected_values else "fail"
                    ),
                    code="retail-ablation-cell",
                    message=(
                        f"{language}/{system}: ablation row reproduces"
                        if frame_ok and observed == expected_values
                        else f"{language}/{system}: ablation row or task frame differs"
                    ),
                    evidence={
                        "observed": observed,
                        "paper": expected_values,
                        "task_frame_matches": frame_ok,
                        "row_counts": [baseline_rows, source_rows, native_rows],
                    },
                )
            )
            rows.append(row)
    return rows, findings


def _listening_manifest(root: Path) -> ListeningManifest:
    # The fixed arms give each language one example from every domain without
    # choosing on task outcome or judge score.
    arm_by_domain = {
        "airline": "openai_xhigh",
        "retail": "gemini_high",
        "telecom": "xai_provider_default",
    }
    calls: list[ListeningCall] = []
    for language in LANGUAGES:
        for domain, system in arm_by_domain.items():
            results_path = _voice_result_path(root, language, domain, system)
            payload = json.loads(results_path.read_text())
            rows = [
                row
                for row in payload["simulation_index"]
                if int(row.get("trial") or 0) == 0
            ]
            ranked = sorted(
                rows,
                key=lambda row: hashlib.sha256(
                    f"{LISTENING_VERSION}:{language}:{domain}:{system}:{row['id']}".encode()
                ).hexdigest(),
            )
            selected = None
            audio = None
            for row in ranked:
                matches = list(
                    results_path.parent.glob(
                        f"artifacts/**/sim_{row['id']}/audio/both.wav"
                    )
                )
                if matches:
                    selected, audio = row, matches[0]
                    break
            if selected is None or audio is None:
                raise FileNotFoundError(
                    f"no mixed audio for {language}/{domain}/{system}"
                )
            calls.append(
                ListeningCall(
                    language=language,
                    domain=domain,
                    system=system,
                    sim_id=str(selected["id"]),
                    task_id=str(selected["task_id"]),
                    reward=selected.get("reward"),
                    source_audio=audio.relative_to(root).as_posix(),
                    sha256=_sha256_file(audio),
                    bytes=audio.stat().st_size,
                )
            )
    return ListeningManifest(calls=calls)


def audit_multilingual(evidence_root: Path) -> MultilingualAudit:
    """Audit frozen evidence and reproduce the core outcome tables."""
    root = evidence_root.expanduser().resolve()
    repo = DATA_DIR.parent.resolve()
    findings: list[Finding] = []
    voice_cells: list[ResultCell] = []
    text_cells: list[ResultCell] = []

    for language in LANGUAGES:
        for domain in DOMAINS:
            for system in VOICE_SYSTEMS:
                path = _voice_result_path(root, language, domain, system)
                if not path.exists():
                    findings.append(
                        Finding(
                            status="fail",
                            code="missing-result",
                            message=f"Missing voice cell {language}/{domain}/{system}",
                            evidence={"path": str(path)},
                        )
                    )
                    continue
                voice_cells.append(
                    _load_cell(path, root, "voice", language, domain, system)
                )
            for system in TEXT_SYSTEMS:
                path = _text_result_path(root, language, domain, system)
                if not path.exists():
                    findings.append(
                        Finding(
                            status="fail",
                            code="missing-result",
                            message=f"Missing text cell {language}/{domain}/{system}",
                            evidence={"path": str(path)},
                        )
                    )
                    continue
                text_cells.append(
                    _load_cell(path, root, "text", language, domain, system)
                )

    if len(voice_cells) == 90:
        findings.extend(_validate_cells(voice_cells, voice=True))
    if len(text_cells) == 36:
        findings.extend(_validate_cells(text_cells, voice=False))

    task_success = _task_success_table(voice_cells) if len(voice_cells) == 90 else []
    text_success = _text_success_table(text_cells) if len(text_cells) == 36 else []
    text_voice_gap = (
        _text_voice_gap_table(task_success, text_success)
        if task_success and text_success
        else []
    )
    trial_stability = (
        _trial_stability_table(voice_cells) if len(voice_cells) == 90 else []
    )
    retail_ablations, ablation_findings = _retail_ablation_table(root)
    findings.extend(ablation_findings)
    if task_success:
        mismatches = []
        for row in task_success:
            observed = tuple(round(row.values[system], 1) for system in VOICE_SYSTEMS)
            if observed != PAPER_TASK_SUCCESS[row.language]:
                mismatches.append(
                    {
                        "language": row.language,
                        "observed": observed,
                        "paper": PAPER_TASK_SUCCESS[row.language],
                    }
                )
        findings.append(
            Finding(
                status="fail" if mismatches else "pass",
                code="task-success-claims",
                message=(
                    f"{len(mismatches)} language rows differ from the paper"
                    if mismatches
                    else "All 30 task-success values reproduce at one decimal place"
                ),
                evidence={"mismatches": mismatches},
            )
        )

    if text_voice_gap:
        observed_gaps = {
            row.language: round(row.values["gap"], 1) for row in text_voice_gap
        }
        mismatches = {
            language: {
                "observed": observed_gaps[language],
                "paper": expected,
            }
            for language, expected in PAPER_TEXT_VOICE_GAPS.items()
            if observed_gaps[language] != expected
        }
        findings.append(
            Finding(
                status="fail" if mismatches else "pass",
                code="text-voice-gap-claims",
                message=(
                    f"{len(mismatches)} text-minus-voice gaps differ from the paper"
                    if mismatches
                    else "All pooled text-minus-voice gaps reproduce"
                ),
                evidence={
                    "gaps": observed_gaps,
                    "range": [min(observed_gaps.values()), max(observed_gaps.values())],
                    "mismatches": mismatches,
                },
            )
        )

    if trial_stability:
        keys = ("trial_0", "trial_1", "two_trial_mean", "max_system_gap")
        mismatches = {}
        for row in trial_stability:
            observed = tuple(round(row.values[key], 1) for key in keys)
            if observed != PAPER_TRIAL_STABILITY[row.language]:
                mismatches[row.language] = {
                    "observed": observed,
                    "paper": PAPER_TRIAL_STABILITY[row.language],
                }
        overall_trial_0 = sum(row.values["trial_0"] for row in trial_stability) / len(
            trial_stability
        )
        overall_trial_1 = sum(row.values["trial_1"] for row in trial_stability) / len(
            trial_stability
        )
        findings.append(
            Finding(
                status="fail" if mismatches else "pass",
                code="trial-stability-claims",
                message=(
                    f"{len(mismatches)} trial-stability rows differ from the paper"
                    if mismatches
                    else "All trial-stability rows and pooled rates reproduce"
                ),
                evidence={
                    "mismatches": mismatches,
                    "overall_trial_0": overall_trial_0,
                    "overall_trial_1": overall_trial_1,
                },
            )
        )

    repeated = [cell for cell in voice_cells if cell.system in REPEATED_SYSTEMS]
    xai = [cell for cell in voice_cells if cell.system == "xai_provider_default"]
    findings.append(
        Finding(
            status="pass" if len(repeated) == 72 else "fail",
            code="repeated-cohort-fingerprint",
            message="Canonical fingerprint over 72 repeated-system results files",
            evidence={
                "algorithm": "sha256(sorted 'relative_path\\tfile_sha256\\trow_count\\n')",
                "fingerprint": _cohort_fingerprint(repeated),
                "files": len(repeated),
            },
        )
    )
    findings.append(_audit_xai_urls(root, xai))
    findings.append(_audit_prompt_archive(repo))
    findings.append(
        Finding(
            status="pass" if len(xai) == 18 else "fail",
            code="xai-cohort-fingerprint",
            message="Canonical fingerprint over 18 xAI results files",
            evidence={
                "algorithm": "sha256(sorted 'relative_path\\tfile_sha256\\trow_count\\n')",
                "fingerprint": _cohort_fingerprint(xai),
                "files": len(xai),
                "model": XAI_MODEL,
            },
        )
    )

    for language in LANGUAGES:
        report = validate_final_pack(language)
        findings.append(
            Finding(
                status="pass" if report.ok else "fail",
                code="current-pack-guardrails",
                message=f"{language}: current pack {'passes' if report.ok else 'fails'} guardrails",
                evidence={"problems": report.problems},
            )
        )
        pending_lines = [
            number
            for number, line in enumerate(
                (repo / f"data/tau2/multilingual/{language}/pack.yaml")
                .read_text()
                .splitlines(),
                start=1,
            )
            if "pending per-language owner review" in line.lower()
        ]
        if pending_lines:
            findings.append(
                Finding(
                    status="warning",
                    code="pack-review-provenance",
                    message=(
                        f"{language}: pack passes machine guardrails but contains "
                        "comments saying native content is pending owner review"
                    ),
                    evidence={"lines": pending_lines},
                )
            )
        changed = _pack_snapshot_drift(repo, language)
        findings.append(
            Finding(
                status="warning" if changed else "pass",
                code="pack-run-drift",
                message=(
                    f"{language}: current effective pack values differ from "
                    f"{len(changed)} recorded run profile(s)"
                    if changed
                    else f"{language}: effective pack values match every recorded run snapshot"
                ),
                evidence={"changed_fields_by_run_profile": changed},
            )
        )

    listening = _listening_manifest(root)
    provisional = MultilingualAudit(
        repository_commit=get_commit_hash(),
        evidence_root=root.name,
        artifact_id="pending",
        findings=findings,
        voice_cells=voice_cells,
        text_cells=text_cells,
        task_success=task_success,
        text_success=text_success,
        text_voice_gap=text_voice_gap,
        trial_stability=trial_stability,
        retail_ablations=retail_ablations,
        listening=listening,
    )
    content = provisional.model_dump(
        mode="json",
        exclude={"artifact_id", "repository_commit", "evidence_root"},
    )
    artifact_id = _sha256_bytes(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    )[:16]
    return provisional.model_copy(update={"artifact_id": artifact_id})


def _write_table(path: Path, rows: list[TableRow], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["language", *columns], lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "language": row.language,
                    **{name: row.values[name] for name in columns},
                }
            )


def _write_ablation_table(path: Path, rows: list[AblationRow]) -> None:
    fieldnames = ["language", "system", "localized", "source", "romanized", "native"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())


def _markdown(report: MultilingualAudit) -> str:
    lines = [
        "# τ-Multilingual reproduction audit",
        "",
        f"Artifact: `{report.artifact_id}`",
        f"Repository commit: `{report.repository_commit}`",
        f"Evidence root: `{report.evidence_root}`",
        "",
        report.summary_line,
        "",
        "## Findings",
        "",
    ]
    icon = {"pass": "PASS", "warning": "WARN", "fail": "FAIL"}
    lines.extend(
        f"- **{icon[finding.status]} — {finding.code}:** {finding.message}"
        for finding in report.findings
    )
    lines.extend(
        [
            "",
            "## Generated tables and samples",
            "",
            "- `task_success.csv`: trial-0 voice Pass@1 by language and system.",
            "- `text_success.csv`: text Pass@1 by language and system.",
            "- `text_voice_gap.csv`: pooled text-minus-voice Pass@1 by language.",
            "- `trial_stability.csv`: repeated-system trial rates and maximum gaps.",
            "- `retail_ablations.csv`: the 30-task entity-localization ablations.",
            "- `listening_manifest.json`: deterministic three-call-per-language sample.",
            "- `audit.json`: complete cell provenance and evidence for every finding.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_artifacts(report: MultilingualAudit, out: Path) -> None:
    """Write stable JSON/CSV/Markdown views of an audit."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(
        report.model_dump_json(indent=2, exclude_none=True) + "\n"
    )
    (out / "AUDIT.md").write_text(_markdown(report))
    _write_table(out / "task_success.csv", report.task_success, VOICE_SYSTEMS)
    _write_table(
        out / "text_success.csv", report.text_success, (*TEXT_SYSTEMS, "pooled")
    )
    _write_table(
        out / "text_voice_gap.csv",
        report.text_voice_gap,
        ("text", "voice", "gap"),
    )
    _write_table(
        out / "trial_stability.csv",
        report.trial_stability,
        ("trial_0", "trial_1", "two_trial_mean", "max_system_gap"),
    )
    _write_ablation_table(out / "retail_ablations.csv", report.retail_ablations)
    (out / "listening_manifest.json").write_text(
        report.listening.model_dump_json(indent=2, exclude_none=True) + "\n"
    )


def export_listening_sample(evidence_root: Path, out: Path) -> ListeningManifest:
    """Copy the deterministic listening sample into a portable directory."""
    root = evidence_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = _listening_manifest(root)
    exported: list[ListeningCall] = []
    for call in selected.calls:
        filename = f"{call.language}_{call.domain}_{call.system}.wav"
        destination = out / filename
        shutil.copyfile(root / call.source_audio, destination)
        if _sha256_file(destination) != call.sha256:
            raise RuntimeError(f"copy verification failed for {destination}")
        exported.append(call.model_copy(update={"exported_file": filename}))
    manifest = selected.model_copy(update={"calls": exported})
    (out / "manifest.json").write_text(
        manifest.model_dump_json(indent=2, exclude_none=True) + "\n"
    )
    return manifest

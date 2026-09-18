# Copyright Sierra
"""Verify and stamp the corrected retail name-role prompt treatment.

The first corrected Korean/Mandarin retail pools rendered the fixed v1 line
before that treatment had a field in ``RunConfig``/``Results.info``.  This
module is the only supported repair path: it accepts registered pools (never
arbitrary roots), verifies every stored caller request, locks the whole cohort,
and changes metadata only.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from tau2.data_model.simulation import (
    Results,
    RetailNameRolesPromptBackfill,
    TerminationReason,
)
from tau2.multilingual.english_prompts import (
    RETAIL_NAME_ROLES_PROMPT_TEMPLATE_V1,
    retail_name_roles_prompt_line,
)
from tau2.runner.checkpoint import CHECKPOINT_META_TMP_PREFIX
from tau2.runner.pools import INFRASTRUCTURE_ERROR, PoolArm, PoolSpec, get_pool
from tau2.runner.run_lock import claim_run_directories
from tau2.task_subsets import load_subset
from tau2.utils.utils import get_commit_hash, get_now, prompt_sha256

STAMP_TOOL = "tau2 pool stamp-retail-name-roles-v1"
CORRECTED_NAME_ROLE_POOLS = frozenset(
    {
        "retail_name_roles_v1",
        "retail_name_roles_xai_v1",
        "multilingual_text_retail_name_roles_v1",
        "retail_dbscript_name_roles_v1",
        "retail_unlocalized_name_roles_v1",
    }
)


class NameRoleProvenanceError(ValueError):
    """A completed cell cannot be safely stamped."""


class _DebugMessage(BaseModel):
    role: str
    content: Any


class _DebugRequest(BaseModel):
    messages: list[_DebugMessage]


class _DebugRecord(BaseModel):
    call_name: str
    request: _DebugRequest


class NameRoleStampCellReport(BaseModel):
    """Verification/write result for one registered pool cell."""

    pool: str
    cell: str
    results_path: Path
    calls_verified: Annotated[int, Field(ge=1)]
    prompt_requests_verified: Annotated[int, Field(ge=1)]
    action: Literal["would_stamp", "stamped", "already_stamped"]
    source_results_sha256: str
    source_header_git_commits: tuple[str, ...]
    nonprompt_payload_sha256: str
    simulation_payload_sha256: str
    prompt_evidence_sha256: str


class NameRoleStampReport(BaseModel):
    """Typed report for one atomic cohort verification pass."""

    tool: Literal["tau2 pool stamp-retail-name-roles-v1"] = STAMP_TOOL
    verifier_git_commit: str
    generated_at: str
    treatment: Literal["v1"] = "v1"
    template_sha256: str
    pools: tuple[str, ...]
    write: bool
    cells: list[NameRoleStampCellReport]

    @property
    def calls_verified(self) -> int:
        return sum(cell.calls_verified for cell in self.cells)


class _VerifiedCell(BaseModel):
    spec_name: str
    cell_name: str
    results_path: Path
    raw: dict[str, Any]
    calls_verified: int
    prompt_requests_verified: int
    source_results_sha256: str
    source_header_git_commits: tuple[str, ...]
    nonprompt_payload_sha256: str
    simulation_payload_sha256: str
    prompt_evidence_sha256: str
    already_stamped: bool
    existing_stamp: RetailNameRolesPromptBackfill | None = None


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str, *, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise NameRoleProvenanceError(f"unsafe {label} path component: {value!r}")
    return value


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_content_text(item) for item in value)
    if isinstance(value, dict):
        # Supports structured message content while keeping the verifier
        # independent of any one provider's debug-log representation.
        return "\n".join(_content_text(item) for item in value.values())
    return str(value)


def _without_stamp_fields(raw: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(raw))
    infos = [clean.get("info", {})]
    infos.extend(item.get("info", {}) for item in clean.get("info_history", []))
    for info in infos:
        info.pop("retail_name_roles_prompt_version", None)
        info.pop("retail_name_roles_prompt_backfill", None)
    return clean


def _headers(raw: dict[str, Any]) -> list[dict[str, Any]]:
    current = raw.get("info")
    if not isinstance(current, dict):
        raise NameRoleProvenanceError("results.json has no object-valued info header")
    headers = [current]
    for index, entry in enumerate(raw.get("info_history", [])):
        info = entry.get("info") if isinstance(entry, dict) else None
        if not isinstance(info, dict):
            raise NameRoleProvenanceError(
                f"info_history[{index}] has no object-valued info header"
            )
        headers.append(info)
    return headers


def _verify_identity(
    spec: PoolSpec,
    language: str,
    arm: PoolArm,
    results: Results,
) -> None:
    info = results.info
    errors: list[str] = []
    expected_config = spec.cell_config(language, arm)
    if spec.retail_name_roles_prompt_version != "v1":
        errors.append(
            "pool retail_name_roles_prompt_version="
            f"{spec.retail_name_roles_prompt_version!r}"
        )
    if info.environment_info.domain_name != spec.domain:
        errors.append(f"domain={info.environment_info.domain_name!r}")
    if info.task_set_name != spec.task_sets[language]:
        errors.append(f"task_set_name={info.task_set_name!r}")
    expected_persona = language if spec.use_personas else None
    if info.user_persona_id != expected_persona:
        errors.append(f"user_persona_id={info.user_persona_id!r}")
    if info.num_trials != spec.num_trials:
        errors.append(f"num_trials={info.num_trials!r}")
    if info.seed != spec.seed:
        errors.append(f"seed={info.seed!r}")
    if (
        info.target_language_directive_version
        != expected_config.target_language_directive_version
    ):
        errors.append(
            "target_language_directive_version="
            f"{info.target_language_directive_version!r}"
        )
    if info.agent_caller_locale_context != expected_config.agent_caller_locale_context:
        errors.append(
            f"agent_caller_locale_context={info.agent_caller_locale_context!r}"
        )
    if info.task_subset is None or info.task_subset.name != spec.tasks.subset:
        errors.append(
            "task_subset="
            f"{None if info.task_subset is None else info.task_subset.name!r}"
        )
    elif info.task_subset.size != spec.tasks.size:
        errors.append(f"task_subset.size={info.task_subset.size!r}")

    if spec.modality == "voice":
        audio = info.audio_native_config
        if audio is None:
            errors.append("audio_native_config=None")
        else:
            if audio.provider != arm.provider:
                errors.append(f"provider={audio.provider!r}")
            if audio.reasoning_effort != arm.reasoning_effort:
                errors.append(f"reasoning_effort={audio.reasoning_effort!r}")
            if audio.model != expected_config.audio_native_config.model:
                errors.append(f"model={audio.model!r}")
            if (
                audio.disclose_voice_gender
                != expected_config.audio_native_config.disclose_voice_gender
            ):
                errors.append(f"disclose_voice_gender={audio.disclose_voice_gender!r}")
        if (
            info.gemini_live_explicit_language_code
            != expected_config.gemini_live_explicit_language_code
        ):
            errors.append(
                "gemini_live_explicit_language_code="
                f"{info.gemini_live_explicit_language_code!r}"
            )
    else:
        if info.audio_native_config is not None:
            errors.append("text cell has audio_native_config")
        if info.agent_info.llm != arm.agent_llm:
            errors.append(f"agent_llm={info.agent_info.llm!r}")
        effort = (info.agent_info.llm_args or {}).get("reasoning_effort")
        if effort != arm.reasoning_effort:
            errors.append(f"agent reasoning_effort={effort!r}")
    if errors:
        raise NameRoleProvenanceError(
            f"{spec.cell_name(language, arm)} does not match registered pool "
            f"{spec.name!r}: {', '.join(errors)}"
        )


def _verify_index(spec: PoolSpec, results_path: Path, results: Results) -> None:
    index = results.simulation_index
    if index is None:
        raise NameRoleProvenanceError(f"{results_path}: missing simulation_index")
    if len(index) != spec.target_per_cell:
        raise NameRoleProvenanceError(
            f"{results_path}: expected {spec.target_per_cell} calls, found {len(index)}"
        )
    ids = [row.id for row in index]
    pairs = [(str(row.task_id), row.trial) for row in index]
    if len(set(ids)) != len(ids):
        raise NameRoleProvenanceError(f"{results_path}: duplicate simulation ids")
    if len(set(pairs)) != len(pairs):
        raise NameRoleProvenanceError(f"{results_path}: duplicate task/trial cells")
    expected_trials = set(range(spec.num_trials))
    if {row.trial for row in index} != expected_trials:
        raise NameRoleProvenanceError(
            f"{results_path}: trials do not equal {sorted(expected_trials)}"
        )
    task_ids = {str(row.task_id) for row in index}
    if len(task_ids) != spec.tasks.size:
        raise NameRoleProvenanceError(
            f"{results_path}: expected {spec.tasks.size} tasks, found {len(task_ids)}"
        )
    if spec.tasks.kind != "subset" or spec.tasks.subset is None:
        raise NameRoleProvenanceError(
            f"{results_path}: corrected-name stamper requires a frozen task subset"
        )
    subset = load_subset(spec.tasks.subset)
    subset_info = results.info.task_subset
    expected_subset_info = {
        "name": subset.name,
        "size": subset.size,
        "tasks_scored": subset.size,
        "frame_task_set": subset.frame.task_set,
        "frame_size": subset.frame.size,
        "frame_digest": subset.frame.digest,
        "strategy": subset.strategy.value,
        "seed": subset.seed,
    }
    actual_subset_info = (
        None if subset_info is None else subset_info.model_dump(mode="json")
    )
    if actual_subset_info != expected_subset_info:
        raise NameRoleProvenanceError(
            f"{results_path}: task-subset provenance does not match the "
            f"checked-in {subset.name!r} artifact"
        )
    canonical_ids = {subset.matches(task_id) for task_id in task_ids}
    if None in canonical_ids or canonical_ids != set(subset.task_ids):
        raise NameRoleProvenanceError(
            f"{results_path}: task cohort does not contain the exact canonical "
            f"task ids from {subset.name!r}"
        )
    stored_task_ids = {str(task.id) for task in results.tasks}
    if stored_task_ids != task_ids:
        raise NameRoleProvenanceError(
            f"{results_path}: stored tasks do not exactly match the indexed cohort"
        )
    for task_id in task_ids:
        trials = {row.trial for row in index if str(row.task_id) == task_id}
        if trials != expected_trials:
            raise NameRoleProvenanceError(
                f"{results_path}: task {task_id!r} has trials {sorted(trials)}, "
                f"expected {sorted(expected_trials)}"
            )
    bad = [
        row.id
        for row in index
        if row.termination_reason == INFRASTRUCTURE_ERROR
        or row.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR.value
    ]
    if bad:
        raise NameRoleProvenanceError(
            f"{results_path}: {len(bad)} infrastructure failures are not complete calls"
        )

    simulations_dir = results_path.parent / "simulations"
    disk_ids = {path.stem for path in simulations_dir.glob("*.json")}
    if disk_ids != set(ids):
        raise NameRoleProvenanceError(
            f"{results_path}: simulation files do not exactly match the index"
        )


def _simulation_manifest(results_path: Path, results: Results) -> str:
    assert results.simulation_index is not None
    rows = []
    for sim_id in sorted(row.id for row in results.simulation_index):
        _safe_component(sim_id, label="simulation id")
        path = results_path.parent / "simulations" / f"{sim_id}.json"
        rows.append((sim_id, _file_sha256(path)))
    return _canonical_hash(rows)


def _prompt_manifest(
    spec: PoolSpec,
    language: str,
    results_path: Path,
    results: Results,
) -> tuple[int, str]:
    assert results.simulation_index is not None
    tasks = {str(task.id): task for task in results.tasks}
    expected_call_name = (
        "user_streaming_response"
        if spec.modality == "voice"
        else "user_simulator_response"
    )
    evidence: list[tuple[str, str, str, str]] = []
    for row in sorted(results.simulation_index, key=lambda item: item.id):
        task_id = str(row.task_id)
        sim_id = str(row.id)
        _safe_component(task_id, label="task id")
        _safe_component(sim_id, label="simulation id")
        task = tasks.get(task_id)
        if task is None:
            raise NameRoleProvenanceError(
                f"{results_path}: indexed task {task_id!r} is absent from tasks"
            )
        expected_line = retail_name_roles_prompt_line(
            task,
            language=language,
            domain=spec.domain,
            task_set_name=spec.task_sets[language],
            version="v1",
        )
        if not expected_line:
            raise NameRoleProvenanceError(
                f"{results_path}: no v1 line resolves for task {task_id!r}"
            )
        debug_dir = (
            results_path.parent
            / "artifacts"
            / f"task_{task_id}"
            / f"sim_{sim_id}"
            / "llm_debug"
        )
        if not debug_dir.is_dir():
            raise NameRoleProvenanceError(
                f"{results_path}: missing debug directory for simulation {sim_id}"
            )
        matched = 0
        for log_path in sorted(debug_dir.glob("*.json")):
            try:
                raw_record = json.loads(log_path.read_text())
            except Exception as exc:
                raise NameRoleProvenanceError(
                    f"{log_path}: malformed LLM debug record: {exc}"
                ) from exc
            if raw_record.get("call_name") != expected_call_name:
                continue
            try:
                record = _DebugRecord.model_validate(raw_record)
            except Exception as exc:
                raise NameRoleProvenanceError(
                    f"{log_path}: malformed {expected_call_name!r} request: {exc}"
                ) from exc
            matched += 1
            system_text = "\n".join(
                _content_text(message.content)
                for message in record.request.messages
                if message.role == "system"
            )
            if system_text.count(expected_line) != 1:
                raise NameRoleProvenanceError(
                    f"{log_path}: expected the exact v1 name-role line once"
                )
            evidence.append(
                (
                    sim_id,
                    str(log_path.relative_to(results_path.parent)),
                    _file_sha256(log_path),
                    hashlib.sha256(expected_line.encode("utf-8")).hexdigest(),
                )
            )
        if matched == 0:
            raise NameRoleProvenanceError(
                f"{debug_dir}: no {expected_call_name!r} request log"
            )
    return len(evidence), _canonical_hash(evidence)


def _verify_cell(
    spec: PoolSpec,
    language: str,
    arm: PoolArm,
) -> _VerifiedCell:
    results_path = spec.cell_dir(language, arm) / "results.json"
    raw = json.loads(results_path.read_text())
    results = Results.load_metadata(results_path)
    _verify_identity(spec, language, arm, results)
    _verify_index(spec, results_path, results)

    headers = _headers(raw)
    existing: RetailNameRolesPromptBackfill | None = None
    all_stamped = True
    for index, header in enumerate(headers):
        treatment_recorded = "retail_name_roles_prompt_version" in header
        treatment = header.get("retail_name_roles_prompt_version")
        if treatment_recorded and treatment is None:
            raise NameRoleProvenanceError(
                f"{results_path}: header {index} records an explicit prompt-off "
                "treatment; it cannot be relabeled as corrected v1"
            )
        if treatment_recorded and treatment != "v1":
            raise NameRoleProvenanceError(
                f"{results_path}: header {index} records contradictory "
                f"treatment {treatment!r}"
            )
        if not treatment_recorded:
            all_stamped = False
        marker = header.get("retail_name_roles_prompt_backfill")
        if marker is None:
            all_stamped = False
            continue
        if treatment != "v1":
            raise NameRoleProvenanceError(
                f"{results_path}: header {index} has a provenance marker "
                "without the v1 treatment"
            )
        parsed = RetailNameRolesPromptBackfill.model_validate(marker)
        if existing is None:
            existing = parsed
        elif parsed != existing:
            raise NameRoleProvenanceError(
                f"{results_path}: current/history backfill records disagree"
            )

    nonprompt_hash = _canonical_hash(_without_stamp_fields(raw))
    simulation_hash = _simulation_manifest(results_path, results)
    request_count, prompt_hash = _prompt_manifest(spec, language, results_path, results)
    if existing is not None:
        expected = {
            "tool": STAMP_TOOL,
            "pool": spec.name,
            "cell": spec.cell_name(language, arm),
            "calls_verified": len(results.simulation_index or []),
            "prompt_requests_verified": request_count,
            "nonprompt_payload_sha256": nonprompt_hash,
            "simulation_payload_sha256": simulation_hash,
            "prompt_evidence_sha256": prompt_hash,
            "template_sha256": prompt_sha256(RETAIL_NAME_ROLES_PROMPT_TEMPLATE_V1),
        }
        actual = existing.model_dump(include=set(expected))
        if actual != expected:
            raise NameRoleProvenanceError(
                f"{results_path}: stored backfill evidence no longer matches payload"
            )
    return _VerifiedCell(
        spec_name=spec.name,
        cell_name=spec.cell_name(language, arm),
        results_path=results_path,
        raw=raw,
        calls_verified=len(results.simulation_index or []),
        prompt_requests_verified=request_count,
        source_results_sha256=_file_sha256(results_path),
        source_header_git_commits=tuple(
            sorted({str(header.get("git_commit")) for header in headers})
        ),
        nonprompt_payload_sha256=nonprompt_hash,
        simulation_payload_sha256=simulation_hash,
        prompt_evidence_sha256=prompt_hash,
        already_stamped=all_stamped,
        existing_stamp=existing,
    )


def _stamp_for(cell: _VerifiedCell) -> RetailNameRolesPromptBackfill:
    if cell.existing_stamp is not None:
        return cell.existing_stamp
    return RetailNameRolesPromptBackfill(
        tool=STAMP_TOOL,
        stamped_at=get_now(),
        git_commit=get_commit_hash(),
        pool=cell.spec_name,
        cell=cell.cell_name,
        calls_verified=cell.calls_verified,
        prompt_requests_verified=cell.prompt_requests_verified,
        nonprompt_payload_sha256=cell.nonprompt_payload_sha256,
        simulation_payload_sha256=cell.simulation_payload_sha256,
        prompt_evidence_sha256=cell.prompt_evidence_sha256,
        template_sha256=prompt_sha256(RETAIL_NAME_ROLES_PROMPT_TEMPLATE_V1),
    )


def _write_cell(cell: _VerifiedCell) -> None:
    raw = cell.raw
    stamp = _stamp_for(cell).model_dump(mode="json")
    for header in _headers(raw):
        header["retail_name_roles_prompt_version"] = "v1"
        header["retail_name_roles_prompt_backfill"] = stamp
    if _canonical_hash(_without_stamp_fields(raw)) != cell.nonprompt_payload_sha256:
        raise AssertionError("stamp changed non-prompt results payload")

    mode = stat.S_IMODE(cell.results_path.stat().st_mode)
    fd, tmp = tempfile.mkstemp(
        suffix=".json",
        prefix=CHECKPOINT_META_TMP_PREFIX,
        dir=cell.results_path.parent,
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(raw, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, cell.results_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def stamp_name_role_provenance(
    pool_names: list[str], *, write: bool = False
) -> NameRoleStampReport:
    """Verify selected corrected pools and optionally stamp their metadata.

    Verification is always cohort-wide and precedes every write.  A failure in
    any cell therefore leaves all results metadata untouched.
    """
    if not pool_names:
        raise NameRoleProvenanceError("select at least one corrected-name pool")
    if len(set(pool_names)) != len(pool_names):
        raise NameRoleProvenanceError("pool names must be unique")
    unknown = set(pool_names) - CORRECTED_NAME_ROLE_POOLS
    if unknown:
        raise NameRoleProvenanceError(
            "only registered corrected-name pools may be stamped; refused: "
            + ", ".join(sorted(unknown))
        )
    specs = [get_pool(name) for name in pool_names]
    cells = [(spec, language, arm) for spec in specs for language, arm in spec.cells()]
    missing = [
        spec.cell_dir(language, arm) / "results.json"
        for spec, language, arm in cells
        if not (spec.cell_dir(language, arm) / "results.json").is_file()
    ]
    if missing:
        raise NameRoleProvenanceError(
            "refusing an incomplete cohort; missing: "
            + ", ".join(str(path) for path in missing)
        )
    run_dirs = [spec.cell_dir(language, arm) for spec, language, arm in cells]
    with claim_run_directories(run_dirs):
        verified = [_verify_cell(spec, language, arm) for spec, language, arm in cells]
        previously_stamped = {
            (cell.spec_name, cell.cell_name): cell.already_stamped for cell in verified
        }
        if write:
            for cell in verified:
                if not cell.already_stamped:
                    _write_cell(cell)
            # Validate the written form and embedded evidence before reporting
            # success. Re-running the same verifier must be a no-op.
            verified = [
                _verify_cell(spec, language, arm) for spec, language, arm in cells
            ]
            if not all(cell.already_stamped for cell in verified):
                raise AssertionError("stamp did not converge to an idempotent state")

    reports = []
    for cell in verified:
        action: Literal["would_stamp", "stamped", "already_stamped"]
        was_stamped = previously_stamped[(cell.spec_name, cell.cell_name)]
        if was_stamped:
            action = "already_stamped"
        elif write:
            action = "stamped"
        else:
            action = "would_stamp"
        reports.append(
            NameRoleStampCellReport(
                pool=cell.spec_name,
                cell=cell.cell_name,
                results_path=cell.results_path,
                calls_verified=cell.calls_verified,
                prompt_requests_verified=cell.prompt_requests_verified,
                action=action,
                source_results_sha256=cell.source_results_sha256,
                source_header_git_commits=cell.source_header_git_commits,
                nonprompt_payload_sha256=cell.nonprompt_payload_sha256,
                simulation_payload_sha256=cell.simulation_payload_sha256,
                prompt_evidence_sha256=cell.prompt_evidence_sha256,
            )
        )
    return NameRoleStampReport(
        verifier_git_commit=get_commit_hash(),
        generated_at=get_now(),
        template_sha256=prompt_sha256(RETAIL_NAME_ROLES_PROMPT_TEMPLATE_V1),
        pools=tuple(pool_names),
        write=write,
        cells=reports,
    )


def write_evidence_report(report: NameRoleStampReport, path: Path) -> Path:
    """Create an evidence sidecar without touching any verified run root.

    Exclusive creation makes the emitted report immutable by convention: a
    second invocation must choose a new path rather than overwriting the
    evidence that was reviewed or consumed downstream.
    """
    if report.write:
        raise NameRoleProvenanceError(
            "evidence sidecars are emitted only by verification-only runs"
        )
    path = Path(path).resolve()
    for cell in report.cells:
        run_dir = cell.results_path.resolve().parent
        if path == run_dir or path.is_relative_to(run_dir):
            raise NameRoleProvenanceError(
                f"evidence output must be outside verified run roots: {path}"
            )
    if not path.parent.is_dir():
        raise NameRoleProvenanceError(
            f"evidence output parent does not exist: {path.parent}"
        )
    try:
        with path.open("x") as handle:
            handle.write(report.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise NameRoleProvenanceError(
            f"refusing to overwrite existing evidence report: {path}"
        ) from None
    return path

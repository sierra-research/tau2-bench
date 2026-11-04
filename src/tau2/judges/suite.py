"""Saved-run judging: load evidence, select implementations, plan and checkpoint."""

import json
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty
from typing import Annotated

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from tau2.config import (
    DEFAULT_JUDGE_CONCURRENCY,
    DEFAULT_JUDGE_PROCESSES,
    DEFAULT_JUDGE_UNIT_RETRIES,
)
from tau2.data_model.judges import (
    DiagnosticId,
    ImplementationFamily,
    JudgeCatalog,
    JudgeContract,
    JudgeInput,
    MeasureVerdict,
    PlanAction,
    PlannedVerdict,
    StoredVerdict,
    SuitePlan,
    SuiteResult,
    VerdictIdentity,
)
from tau2.data_model.simulation import Results
from tau2.judges.base import BaseJudge
from tau2.judges.run_evidence import SavedRun, progress_path


class JudgeExecutionConfig(BaseModel):
    """Bounded process and thread concurrency for saved-run judging."""

    processes: Annotated[
        int, Field(ge=1, description="Independent judge worker processes.")
    ] = DEFAULT_JUDGE_PROCESSES
    concurrency: Annotated[
        int, Field(ge=1, description="Judge calls in flight per worker process.")
    ] = DEFAULT_JUDGE_CONCURRENCY
    retry_errors: Annotated[
        bool, Field(description="Retry matching stored judge errors and partials.")
    ] = True
    unit_retries: Annotated[
        int,
        Field(
            ge=0,
            description="Further attempts, in the same run, at units whose judge "
            "reply was incomplete or invalid.",
        ),
    ] = DEFAULT_JUDGE_UNIT_RETRIES


class JudgeJob(BaseModel):
    """One planned verdict and its resumable previous outcome."""

    item: PlannedVerdict
    previous: StoredVerdict | None = None


class JudgeShard(BaseModel):
    """Typed, serialized boundary for one judge worker process."""

    results: Path
    catalog: JudgeCatalog
    jobs: list[JudgeJob]
    concurrency: int = Field(ge=1)
    unit_retries: int = Field(default=0, ge=0)


class JudgeWorkerMessage(BaseModel):
    """A completed verdict or a process-level failure sent to the controller."""

    verdict: StoredVerdict | None = None
    error: str | None = None


def _key(item: JudgeInput, measure: DiagnosticId) -> tuple[str, str, int, str]:
    return item.run_id, item.simulation_id, item.trial, measure


def plan_suite(
    inputs: list[JudgeInput],
    catalog: JudgeCatalog,
    stored: list[StoredVerdict],
    *,
    only: list[DiagnosticId] | None = None,
    trials: list[int] | None = None,
    force: bool = False,
) -> SuitePlan:
    """Plan missing/stale/forced work without accessing a model or mutating data."""
    measures = (
        only
        if only is not None
        else list(dict.fromkeys(pair.measure for pair in catalog.pairs))
    )
    if set(measures) - {pair.measure for pair in catalog.pairs}:
        raise ValueError("Selected diagnostic is absent from the judge registry")
    if len(set(measures)) != len(measures) or not measures:
        raise ValueError("Select distinct measures, at least one")
    if trials is not None and (
        not trials or min(trials) < 0 or len(set(trials)) != len(trials)
    ):
        raise ValueError("Select distinct nonnegative trials, at least one")
    existing = {_key(row.identity.input, row.identity.measure): row for row in stored}
    if len(existing) != len(stored):
        raise ValueError("Duplicate stored verdict identities")
    pairs = {(pair.language, pair.measure): pair for pair in catalog.pairs}
    contracts = {contract.contract_id: contract for contract in catalog.contracts}
    items = []
    seen = set()
    for source in inputs:
        if trials is not None and source.trial not in trials:
            continue
        for measure in measures:
            key = _key(source, measure)
            if key in seen:
                raise ValueError("Duplicate simulation/measure inputs")
            seen.add(key)
            pair = pairs.get((source.language, measure))
            identity = None
            if pair is None:
                action, reason = (
                    PlanAction.NOT_APPLICABLE,
                    "No selected language/measure contract",
                )
            elif pair.domains is not None and source.domain not in pair.domains:
                action, reason = (
                    PlanAction.NOT_APPLICABLE,
                    "Domain outside approved scope",
                )
            elif (
                pair.agent_gender is not None
                and pair.agent_gender != source.agent_gender
            ):
                action, reason = (
                    PlanAction.NOT_APPLICABLE,
                    f"Validation supports only the {pair.agent_gender}-agent frame",
                )
            else:
                contract = contracts[pair.contract_id]
                identity = VerdictIdentity(
                    input=(
                        source
                        if pair.needs_audio
                        else source.model_copy(
                            update={
                                "audio_fingerprint": None,
                                "audio_error": None,
                                "has_audio": False,
                            }
                        )
                    ),
                    measure=measure,
                    contract_id=contract.contract_id,
                    implementation=contract.implementation,
                    input_exclusions=contract.input_exclusions,
                    model=contract.model,
                    model_args=contract.model_args,
                    prompt_version=contract.prompt.prompt_version,
                    rubric_version=contract.prompt.rubric_version,
                    catalog_revision=catalog.revision,
                    transport_revision="multimodal-responses-v1",
                )
                previous = existing.get(key)
                if pair.needs_audio and not source.has_audio:
                    action, reason = (
                        PlanAction.MISSING_INPUT,
                        "Delivered audio/reference unavailable",
                    )
                elif force:
                    action, reason = (
                        PlanAction.FORCE,
                        "Explicit recomputation requested",
                    )
                elif previous is None:
                    action, reason = PlanAction.MISSING, "No saved verdict"
                elif previous.identity != identity:
                    action, reason = PlanAction.STALE, "Input or judge contract changed"
                elif previous.error is not None or (
                    previous.verdict is not None and previous.verdict.unit_errors
                ):
                    action, reason = (
                        PlanAction.RETRY_ERROR,
                        "Previous attempt recorded an error",
                    )
                else:
                    action, reason = PlanAction.REUSE, "Input and judge contract match"
            items.append(
                PlannedVerdict(
                    input=source,
                    measure=measure,
                    action=action,
                    identity=identity,
                    reason=reason,
                )
            )
    return SuitePlan(items=items)


def read_verdicts(path: Path) -> list[StoredVerdict]:
    """Read saved diagnostics; absent progress means no verdicts, not zero scores."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Diagnostic progress must be a list of typed verdict records")
    try:
        return [StoredVerdict.model_validate(row) for row in payload]
    except ValidationError as error:
        raise ValueError(
            f"{path} was written by an incompatible judge-suite schema; move it "
            "aside and re-judge the run"
        ) from error


def _save_verdicts(path: Path, rows: list[StoredVerdict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "[\n" + ",\n".join(row.model_dump_json() for row in rows) + "\n]\n"
    )
    temporary.replace(path)


def _check_resumed(previous: MeasureVerdict, response: MeasureVerdict) -> None:
    """A resumed verdict keeps the units it was given and every successful one."""
    if not isinstance(response, MeasureVerdict):
        raise TypeError("Judge must return a typed MeasureVerdict")
    before_ids = {u.unit_id for u in [*previous.units, *previous.unit_errors]}
    after_ids = {u.unit_id for u in [*response.units, *response.unit_errors]}
    if before_ids != after_ids:
        raise ValueError("Partial retry changed diagnostic unit coverage")
    retained = {u.unit_id: u for u in response.units}
    if any(retained.get(u.unit_id) != u for u in previous.units):
        raise ValueError("Partial retry changed or dropped a successful unit")


def _retry_failed_replies(
    item: PlannedVerdict,
    contract: JudgeContract,
    verdict: MeasureVerdict,
    resume_backend: Callable[
        [JudgeInput, JudgeContract, str, MeasureVerdict], MeasureVerdict
    ]
    | None,
    attempts: int,
) -> MeasureVerdict:
    """Ask again, in the same run, for units whose judge reply failed.

    A judge reply can fail by chance: a model that keeps deliberating until it
    reaches its output limit, or a reply its schema rejects, usually succeeds
    when asked again. One such unit leaves its whole call unscored, so each
    further attempt goes through the judge's own resume, which keeps every
    successful unit and asks only for the failed ones under the same contract:
    the verdict is the one a later ``tau2 judges suite`` would record. Input
    errors (missing evidence) are not retried, judges without resume support are
    left as they are, and a retry that itself fails keeps the last verdict.
    """
    for attempt in range(1, attempts + 1):
        if resume_backend is None or not any(
            error.kind == "response" for error in verdict.unit_errors
        ):
            break
        try:
            retried = resume_backend(item.input, contract, item.measure, verdict)
            _check_resumed(verdict, retried)
        except NotImplementedError:
            break
        except Exception as exc:
            logger.warning(
                f"Retry {attempt} of {item.measure} for {item.input.simulation_id} "
                f"failed: {type(exc).__name__}: {exc}"
            )
            break
        verdict = retried
    return verdict


def _evaluate_job(
    job: JudgeJob,
    contracts: dict[str, JudgeContract],
    backend: Callable[[JudgeInput, JudgeContract, str], MeasureVerdict],
    resume_backend: Callable[
        [JudgeInput, JudgeContract, str, MeasureVerdict], MeasureVerdict
    ]
    | None,
    unit_retries: int = 0,
) -> StoredVerdict:
    """Evaluate one planned measure while preserving any successful prior units."""
    item, previous = job.item, job.previous
    assert item.identity is not None
    partial = (
        previous.verdict
        if (
            item.action == PlanAction.RETRY_ERROR
            and previous is not None
            and previous.identity == item.identity
            and previous.verdict is not None
        )
        else None
    )
    if partial is not None and resume_backend is None:
        raise ValueError(
            "Partial verdicts require a resume backend to preserve successful units"
        )
    try:
        contract = contracts[item.identity.contract_id]
        if partial is not None:
            assert resume_backend is not None
            response = resume_backend(item.input, contract, item.measure, partial)
            _check_resumed(partial, response)
        else:
            response = backend(item.input, contract, item.measure)
        if not isinstance(response, MeasureVerdict):
            raise TypeError("Judge must return a typed MeasureVerdict")
        response = _retry_failed_replies(
            item, contract, response, resume_backend, unit_retries
        )
        return StoredVerdict(identity=item.identity, verdict=response)
    except Exception as exc:
        if partial is not None:
            return StoredVerdict(
                identity=item.identity,
                verdict=partial.model_copy(
                    update={
                        "unit_errors": [
                            error.model_copy(
                                update={
                                    "error": f"Retry failed: {type(exc).__name__}: {exc}"
                                }
                            )
                            for error in partial.unit_errors
                        ]
                    }
                ),
            )
        return StoredVerdict(
            identity=item.identity, error=f"{type(exc).__name__}: {exc}"
        )


def execute_suite(
    inputs: list[JudgeInput],
    catalog: JudgeCatalog,
    path: Path,
    *,
    only: list[DiagnosticId] | None = None,
    trials: list[int] | None = None,
    force: bool = False,
    dry_run: bool = False,
    retry_errors: bool = True,
    backend: Callable[[JudgeInput, JudgeContract, str], MeasureVerdict] | None = None,
    resume_backend: Callable[
        [JudgeInput, JudgeContract, str, MeasureVerdict], MeasureVerdict
    ]
    | None = None,
    unit_retries: int = 0,
) -> SuitePlan:
    """Resume typed saved diagnostics, checkpointing each call/measure independently.

    The supplied judge returns a typed verdict without serialization. Units whose
    reply failed are asked again up to ``unit_retries`` times through
    ``resume_backend``; failures still left are persisted and a later invocation
    retries them. Ordinary task rewards are never read or changed by this
    operation.
    """
    stored = read_verdicts(path)
    plan = plan_suite(inputs, catalog, stored, only=only, trials=trials, force=force)
    if dry_run:
        return plan
    work = [
        item
        for item in plan.items
        if item.action in {PlanAction.MISSING, PlanAction.STALE, PlanAction.FORCE}
        or (retry_errors and item.action == PlanAction.RETRY_ERROR)
    ]
    if work and backend is None:
        raise ValueError(
            "A reviewed calibrated judge adapter is required for execution"
        )
    existing = {_key(row.identity.input, row.identity.measure): row for row in stored}
    contracts = {contract.contract_id: contract for contract in catalog.contracts}
    for item in work:
        assert backend is not None
        saved = _evaluate_job(
            JudgeJob(item=item, previous=existing.get(_key(item.input, item.measure))),
            contracts,
            backend,
            resume_backend,
            unit_retries,
        )
        existing[_key(item.input, item.measure)] = saved
        _save_verdicts(path, list(existing.values()))
    return plan


def _run_judge_shard(payload: str, result_queue: multiprocessing.Queue) -> None:
    """Evaluate one disjoint shard; only the parent writes the diagnostics file."""
    try:
        shard = JudgeShard.model_validate_json(payload)
        saved = SavedRun(shard.results)
        # Resolve shared lazy metadata before threads start. Monolithic runs also
        # need their simulation map built once outside the thread pool.
        _ = saved.metadata, saved.tasks, saved.language, saved.run_id
        if Results._detect_format(saved.results_path) != "dir":
            saved._monolithic = {
                sim.id: sim for sim in Results.iter_simulations(saved.results_path)
            }
        contracts = {
            contract.contract_id: contract for contract in shard.catalog.contracts
        }

        def evaluate(
            item: JudgeInput,
            contract: JudgeContract,
            measure: str,
            previous: MeasureVerdict | None = None,
        ) -> MeasureVerdict:
            judge = create_judge(contract)
            evidence = saved.load(item)
            if previous is not None:
                return judge.resume(evidence, measure, previous)
            return judge.evaluate(evidence, measure)

        with ThreadPoolExecutor(max_workers=shard.concurrency) as executor:
            futures = [
                executor.submit(
                    _evaluate_job,
                    job,
                    contracts,
                    evaluate,
                    evaluate,
                    shard.unit_retries,
                )
                for job in shard.jobs
            ]
            for future in as_completed(futures):
                result_queue.put(
                    JudgeWorkerMessage(verdict=future.result()).model_dump_json()
                )
    except BaseException as exc:
        result_queue.put(
            JudgeWorkerMessage(error=f"{type(exc).__name__}: {exc}").model_dump_json()
        )
        raise


def execute_suite_parallel(
    inputs: list[JudgeInput],
    catalog: JudgeCatalog,
    path: Path,
    results: Path,
    execution: JudgeExecutionConfig,
    *,
    only: list[DiagnosticId] | None = None,
    trials: list[int] | None = None,
    force: bool = False,
) -> SuitePlan:
    """Run disjoint judge jobs with 1 controller and bounded process/thread workers.

    Workers read immutable saved results. The controller alone checkpoints each
    completed verdict atomically, preserving previously successful diagnostics.
    """
    stored = read_verdicts(path)
    plan = plan_suite(inputs, catalog, stored, only=only, trials=trials, force=force)
    existing = {_key(row.identity.input, row.identity.measure): row for row in stored}
    work = [
        JudgeJob(item=item, previous=existing.get(_key(item.input, item.measure)))
        for item in plan.items
        if item.action in {PlanAction.MISSING, PlanAction.STALE, PlanAction.FORCE}
        or (execution.retry_errors and item.action == PlanAction.RETRY_ERROR)
    ]
    if not work:
        return plan
    workers = min(execution.processes, len(work))
    shards: list[list[JudgeJob]] = [[] for _ in range(workers)]
    for index, job in enumerate(work):
        shards[index % workers].append(job)
    logger.info(
        "Judging {} verdicts with {} processes x {} calls in flight",
        len(work),
        workers,
        execution.concurrency,
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=workers * execution.concurrency * 2)
    processes = [
        context.Process(
            target=_run_judge_shard,
            args=(
                JudgeShard(
                    results=results,
                    catalog=catalog,
                    jobs=jobs,
                    concurrency=execution.concurrency,
                    unit_retries=execution.unit_retries,
                ).model_dump_json(),
                result_queue,
            ),
        )
        for jobs in shards
    ]
    expected = {
        _key(job.item.input, job.item.measure): job.item.identity for job in work
    }
    completed: set[tuple[str, str, int, str]] = set()
    try:
        for process in processes:
            process.start()
        while len(completed) < len(work):
            try:
                message = JudgeWorkerMessage.model_validate_json(
                    result_queue.get(timeout=10)
                )
            except Empty:
                if any(p.exitcode not in (None, 0) for p in processes):
                    raise RuntimeError(
                        "A judge worker exited before completing its shard"
                    )
                if not any(p.is_alive() for p in processes):
                    raise RuntimeError(
                        "Judge workers stopped before all verdicts arrived"
                    )
                continue
            if message.error is not None:
                raise RuntimeError(f"Judge worker failed: {message.error}")
            if message.verdict is None:
                raise ValueError("Judge worker returned neither a verdict nor an error")
            verdict = message.verdict
            key = _key(verdict.identity.input, verdict.identity.measure)
            if (
                key in completed
                or key not in expected
                or verdict.identity != expected[key]
            ):
                raise ValueError(
                    "Judge worker returned an unexpected or duplicate verdict"
                )
            completed.add(key)
            existing[key] = verdict
            _save_verdicts(path, list(existing.values()))
        for process in processes:
            process.join(timeout=10)
            if process.exitcode != 0:
                raise RuntimeError("A judge worker did not exit cleanly")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=10)
        result_queue.close()
    return plan


def create_judge(contract: JudgeContract) -> BaseJudge:
    """Select the same calibrated implementation for production and candidates."""
    from tau2.judges.criteria import CriteriaJudge
    from tau2.judges.fidelity import FidelityJudge
    from tau2.judges.tool_use import ToolUseJudge

    if contract.implementation == ImplementationFamily.FIDELITY:
        return FidelityJudge(contract)
    if contract.implementation == ImplementationFamily.TOOL_USE:
        return ToolUseJudge(contract)
    if contract.implementation in {
        ImplementationFamily.NATIVENESS,
        ImplementationFamily.NATURALNESS,
        ImplementationFamily.HINDI_GENDER,
        ImplementationFamily.AGENT_TEXT,
    }:
        return CriteriaJudge(contract)
    raise NotImplementedError(f"No judge implementation for {contract.implementation}")


def judge_saved_run(
    results: Path,
    catalog: JudgeCatalog,
    *,
    only: list[DiagnosticId] | None = None,
    trials: list[int] | None = None,
    force: bool = False,
    execute: bool = False,
    output: Path | None = None,
    execution: JudgeExecutionConfig | None = None,
) -> SuiteResult:
    """Plan or execute one saved run, with explicit separate candidate storage."""
    saved = SavedRun(results)
    path = output if output is not None else progress_path(saved.results_path)
    execution = execution or JudgeExecutionConfig()
    if execute and (execution.processes > 1 or execution.concurrency > 1):
        execute_suite_parallel(
            saved.inputs,
            catalog,
            path,
            saved.results_path,
            execution,
            only=only,
            trials=trials,
            force=force,
        )
    elif execute:
        judges: dict[str, BaseJudge] = {}

        def evaluate(
            item: JudgeInput,
            contract: JudgeContract,
            measure: str,
            previous: MeasureVerdict | None = None,
        ) -> MeasureVerdict:
            judge = judges.get(contract.contract_id)
            if judge is None:
                judge = judges[contract.contract_id] = create_judge(contract)
            evidence = saved.load(item)
            if previous is not None:
                return judge.resume(evidence, measure, previous)
            return judge.evaluate(evidence, measure)

        execute_suite(
            saved.inputs,
            catalog,
            path,
            only=only,
            trials=trials,
            force=force,
            retry_errors=execution.retry_errors,
            backend=evaluate,
            resume_backend=evaluate,
            unit_retries=execution.unit_retries,
        )
    stored = read_verdicts(path)
    plan = plan_suite(
        saved.inputs,
        catalog,
        stored,
        only=only,
        trials=trials,
        force=force and not execute,
    )
    selected = {_key(item.input, item.measure) for item in plan.items}
    return SuiteResult(
        plan=plan,
        verdicts=[
            row
            for row in stored
            if _key(row.identity.input, row.identity.measure) in selected
        ],
    )

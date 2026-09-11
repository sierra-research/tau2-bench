# Copyright Sierra
"""Streaming access to stored runs for the deterministic loss instruments.

The caller-cost and entity-trace instruments (``tau2 metrics caller-cost`` /
``tau2 metrics entity-trace``) both walk stored run directories the same way:
discover every ``results.json`` under the given paths, read run-level metadata
once, then stream simulations joined with their task definitions. This module
owns that seam so the two instruments cannot drift on run identity, language
resolution, or modality detection.

Deliberately metrics-owned rather than imported from ``tau2.annotation``:
the annotation package sits above metrics in the dependency order (it imports
``tau2.metrics.interaction_quality``), so the instruments re-state the small
amount of loading logic they need instead of reaching upward.
"""

from pathlib import Path
from typing import Annotated, Iterable, Iterator, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.simulation import Results, SimulationIndexEntry, SimulationRun
from tau2.data_model.tasks import Task


class RunMeta(BaseModel):
    """Run-level provenance shared by every record extracted from one run."""

    results_path: Annotated[
        str, Field(description="Path of the run's results.json (or dir).")
    ]
    experiment_label: Annotated[
        str, Field(description="Human label of the source run (its dir name).")
    ]
    domain: Annotated[str, Field(description="Registered domain of the run.")]
    modality: Annotated[
        Literal["voice", "text"],
        Field(
            description="'voice' when the run records an audio_native_config, "
            "'text' for a half-duplex text run."
        ),
    ]
    provider: Annotated[
        str,
        Field(description="Audio-native provider ('' on text runs)."),
    ]
    agent_model: Annotated[
        str, Field(description="Agent model id of the run ('' when unrecorded).")
    ]
    reasoning_effort: Annotated[
        Optional[str],
        Field(
            description="Reasoning effort as RECORDED on the run (voice: the "
            "audio-native config value when its source is the run itself or a "
            "stamped backfill; text: the value pinned in agent llm_args). "
            "None when the run never recorded one — the instruments report "
            "per (language x domain x provider) cells, so an unrecorded "
            "effort is carried as unknown rather than refused."
        ),
    ] = None
    run_git_commit: Annotated[
        str, Field(description="Git commit recorded on the run itself.")
    ]
    task_set_name: Annotated[
        Optional[str], Field(description="Registered task set of the run, if any.")
    ] = None


class LoadedCall(BaseModel):
    """One simulation joined with its run metadata and task definition."""

    model_config = {"arbitrary_types_allowed": True}

    meta: RunMeta
    sim: SimulationRun
    task: Annotated[
        Optional[Task],
        Field(
            description="The task the sim ran (None when the results file "
            "does not carry a task with the sim's task_id)."
        ),
    ]
    language: Annotated[
        str, Field(description="ISO 639-1 language of the call (lowercase).")
    ]


def discover_results_files(paths: Iterable[Path | str]) -> list[Path]:
    """Every ``results.json`` under ``paths`` (files pass through)."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            nested = sorted(path.rglob("results.json"))
            if not nested:
                raise FileNotFoundError(f"No results.json files found under {path}")
            found.extend(nested)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    return found


def _recorded_reasoning_effort(meta: Results) -> Optional[str]:
    """The effort the run recorded, or None — never an in-memory default.

    Mirrors the distinction ``tau2.annotation.features`` draws: an
    audio-native config whose effort source is ``inferred`` with no backfill
    stamp was never recorded by the run, so it is reported as unknown here
    (the instruments aggregate by provider, not by arm, so unknown is
    carried rather than refused).
    """
    from tau2.data_model.simulation import ReasoningEffortSource

    audio_cfg = meta.info.audio_native_config
    if audio_cfg is None:
        pinned = (meta.info.agent_info.llm_args or {}).get("reasoning_effort")
        return str(pinned) if pinned is not None else None
    unrecorded = (
        audio_cfg.reasoning_effort_source is ReasoningEffortSource.INFERRED
        and audio_cfg.reasoning_effort_backfill is None
    )
    if unrecorded:
        return None
    return audio_cfg.reasoning_effort.value


def load_run_meta(results_file: Path) -> tuple[RunMeta, dict[str, Task]]:
    """Run metadata plus the run's task definitions keyed by task id."""
    meta = Results.load_metadata(results_file)
    return _run_meta_from(results_file, meta), {
        str(task.id): task for task in meta.tasks
    }


def load_run_index(results_file: Path) -> tuple[RunMeta, list[SimulationIndexEntry]]:
    """Run metadata plus the lightweight per-sim summaries (id, task, reward).

    The dir-format ``simulation_index`` avoids streaming full simulations.
    It is written by the same save that wrote the sims, so it is the run's own
    summary, not a re-derivation. Monolithic-json runs carry no index; their
    entries are built by streaming the sims once.
    """
    meta = Results.load_metadata(results_file)
    run_meta = _run_meta_from(results_file, meta)
    if meta.simulation_index is not None:
        return run_meta, meta.simulation_index
    entries = [
        SimulationIndexEntry(
            id=sim.id,
            task_id=sim.task_id,
            trial=sim.trial if sim.trial is not None else 0,
            reward=sim.reward_info.reward if sim.reward_info else None,
        )
        for sim in Results.iter_simulations(results_file)
    ]
    return run_meta, entries


def _run_meta_from(results_file: Path, meta: Results) -> RunMeta:
    audio_cfg = meta.info.audio_native_config
    return RunMeta(
        results_path=str(results_file),
        experiment_label=results_file.parent.name,
        domain=meta.info.environment_info.domain_name,
        modality="voice" if audio_cfg is not None else "text",
        provider=audio_cfg.provider if audio_cfg is not None else "",
        agent_model=str(meta.info.agent_info.llm or ""),
        reasoning_effort=_recorded_reasoning_effort(meta),
        run_git_commit=meta.info.git_commit,
        task_set_name=meta.info.task_set_name,
    )


def sim_language(sim: SimulationRun, domain: str) -> str:
    """A sim's language: the judged language if stored, else the task-id suffix.

    Same resolution the preference feature extractor uses: older English runs
    carry no ``nativeness_info``, and the run Info holds no reliable language,
    so the localized task-set suffix (``..._es``, ``..._pt_identity``) is the
    deterministic fallback; base English task ids mean English.
    """
    if sim.nativeness_info is not None and sim.nativeness_info.language:
        return sim.nativeness_info.language.lower()
    from tau2.multilingual.english_prompts import task_language

    return (task_language(str(sim.task_id), domain) or "en").lower()


def iter_loaded_calls(
    paths: Iterable[Path | str],
    *,
    langs: Optional[list[str]] = None,
    domain: Optional[str] = None,
    max_sims: Optional[int] = None,
) -> Iterator[LoadedCall]:
    """Stream (run meta, sim, task, language) across every run under ``paths``."""
    wanted = {lang.strip().lower() for lang in langs} if langs else None
    yielded = 0
    results_files = discover_results_files(paths)
    for i, results_file in enumerate(results_files, 1):
        logger.info(f"reading run [{i}/{len(results_files)}]: {results_file}")
        run_meta, tasks_by_id = load_run_meta(results_file)
        if domain and run_meta.domain != domain:
            logger.info(
                f"skipping (domain {run_meta.domain} != {domain}): {results_file}"
            )
            continue
        for sim in Results.iter_simulations(results_file):
            language = sim_language(sim, run_meta.domain)
            if wanted is not None and language not in wanted:
                continue
            task = tasks_by_id.get(str(sim.task_id))
            if task is None:
                logger.warning(
                    f"{results_file}: sim {sim.id} references task "
                    f"{sim.task_id!r} not present in the results file; "
                    "task-grounded fields will be empty"
                )
            yield LoadedCall(meta=run_meta, sim=sim, task=task, language=language)
            yielded += 1
            if max_sims is not None and yielded >= max_sims:
                return

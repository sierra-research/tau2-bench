from collections import defaultdict
from pathlib import Path

from tau2.config import DEFAULT_LOCALIZE_DB
from tau2.data_model.simulation import Results
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.utils.utils import show_dict_diff


def check_format(path: str | Path) -> tuple[Results | None, bool, str]:
    """Check format of trajectories."""
    try:
        results = Results.load(path)
        return results, True, ""
    except Exception as e:
        return None, False, f"Error loading results: {e}"


def _expected_tasks(results: Results) -> list[Task]:
    """The base-split tasks the recorded language binding runs.

    A language pack may replace registry tasks with reviewed same-ID versions
    (the English retail and telecom bindings do), so runs that recorded a
    language are checked against that binding. Results recorded before every
    run resolved a language pack used the registry tasks directly.
    """
    info = results.info
    domain = info.environment_info.domain_name
    if info.language is None:
        return registry.get_tasks_loader(domain)("base")

    from tau2.multilingual.runtime import TaskSelection, load_language

    canonical_voice_v1 = (
        info.benchmark is not None and info.benchmark.benchmark_version == "voice-v1"
    )
    loaded = load_language(
        info.language,
        domain,
        mode="voice" if info.audio_native_config is not None else "text",
        localize_db=DEFAULT_LOCALIZE_DB
        if info.localize_db is None
        else info.localize_db,
        selection=TaskSelection(
            task_split_name="base",
            # Intake strategies select their own task variant (as at run time).
            task_kwargs={"strategy": info.elicitation.strategy.value}
            if info.elicitation is not None
            else {},
        ),
        pack_extension=info.pack_extension,
        **({"canonical_tasks": True} if canonical_voice_v1 else {}),
    )
    return loaded.tasks


def check_tasks(results: Results) -> tuple[bool, str]:
    """Checks that Results contains all the tasks for the specified domain.
    Uses the base task split of the recorded language binding.
    """
    domain_tasks = {task.id: task for task in _expected_tasks(results)}
    results_tasks = {task.id: task for task in results.tasks}
    domains_tasks_ids = set(domain_tasks.keys())
    results_tasks_ids = set(results_tasks.keys())
    missing_tasks = domains_tasks_ids - results_tasks_ids
    extra_tasks = results_tasks_ids - domains_tasks_ids

    if missing_tasks or extra_tasks:
        error_msg = []
        if missing_tasks:
            error_msg.append(f"Missing tasks: {missing_tasks}")
        if extra_tasks:
            error_msg.append(f"Extra tasks: {extra_tasks}")
        return False, "; ".join(error_msg)

    task_diffs = {}
    for tid in domains_tasks_ids:
        domain_task = domain_tasks[tid]
        results_task = results_tasks[tid]
        if get_pydantic_hash(domain_task) != get_pydantic_hash(results_task):
            task_diffs[tid] = show_dict_diff(
                domain_task.model_dump(), results_task.model_dump()
            )

    if task_diffs:
        return False, f"Task mismatches found in {len(task_diffs)} tasks"

    return True, ""


def check_num_trials(results: Results) -> tuple[bool, str]:
    """Checks that all the tasks have been run for n trials."""
    task_ids = {task.id for task in results.tasks}
    num_trials = results.info.num_trials
    task_trials = defaultdict(int)
    for sim in results.simulations:
        task_trials[sim.task_id] += 1

    missing_trials: dict[str, int] = {}
    for task_id in task_ids:
        task_trials_count = task_trials.get(task_id, 0)
        if num_trials != task_trials_count:
            missing_trials[task_id] = task_trials_count

    if missing_trials:
        return False, f"Incorrect trial counts for {len(missing_trials)} tasks"

    return True, ""

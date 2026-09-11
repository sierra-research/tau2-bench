"""
Helper functions for task loading, run configuration, and metadata.
"""

import random
from typing import TYPE_CHECKING, Annotated, Optional

from pydantic import BaseModel, Field

from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    RunConfig,
    TaskSubsetInfo,
    UserInfo,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import EnvironmentInfo
from tau2.registry import RegistryInfo, registry
from tau2.task_subsets import TaskSubset
from tau2.user.user_simulator import (
    get_global_user_sim_guidelines,
    get_global_user_sim_guidelines_voice,
)
from tau2.utils.utils import get_commit_hash, get_now

if TYPE_CHECKING:
    from tau2.data_model.simulation import AudioNativeConfig


def get_options() -> RegistryInfo:
    """Returns available options (domains, agents, users, task sets) from the registry."""
    return registry.get_info()


def get_environment_info(
    domain_name: str,
    include_tool_info: bool = False,
    env_kwargs: Optional[dict] = None,
) -> EnvironmentInfo:
    """Get information about the environment for a registered domain."""
    env_constructor = registry.get_env_constructor(domain_name)
    if env_kwargs is None:
        env_kwargs = {}
    return env_constructor(**env_kwargs).get_info(include_tool_info=include_tool_info)


def load_task_splits(task_set_name: str) -> Optional[dict[str, list[str]]]:
    """Load the task splits for a given task set."""
    task_split_loader = registry.get_task_splits_loader(task_set_name)
    if task_split_loader is None:
        return None
    return task_split_loader()


def load_tasks(task_set_name: str, task_split_name: Optional[str] = None) -> list[Task]:
    """Load tasks for a given task set, optionally filtering by split."""
    task_loader = registry.get_tasks_loader(task_set_name)
    tasks = task_loader(task_split_name=task_split_name)
    return tasks


def get_tasks(
    task_set_name: str,
    task_split_name: Optional[str] = None,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
    task_subset: Optional[TaskSubset] = None,
) -> list[Task]:
    """Load tasks with optional filtering by IDs, subset and count.

    Args:
        task_set_name: The task set to load from.
        task_split_name: Optional split name (e.g., "base").
        task_ids: If provided, only return tasks with these IDs.
        num_tasks: If provided, limit to this many tasks.
        task_subset: If provided, keep only the subset's tasks, in subset
            order. Resolved by the caller (see tau2.task_subsets), which is
            also where a subset colliding with task_ids/num_tasks is refused.

    Returns:
        List of tasks matching the criteria.

    Raises:
        ValueError: If task_ids are specified but some are not found, or the
            subset does not fit the task set.
    """
    if task_ids is None:
        tasks = load_tasks(task_set_name=task_set_name, task_split_name=task_split_name)
    else:
        tasks = [
            task
            for task in load_tasks(
                task_set_name=task_set_name, task_split_name=task_split_name
            )
            if task.id in task_ids
        ]
    if task_ids is not None and len(tasks) != len(task_ids):
        missing_tasks = set(task_ids) - set([task.id for task in tasks])
        raise ValueError(
            f"Not all tasks were found for task set {task_set_name} - {task_split_name}: {missing_tasks}"
        )
    if task_subset is not None:
        from tau2.task_subsets import apply_subset

        tasks = apply_subset(task_subset, tasks, task_set_name)
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


class ResolvedTasks(BaseModel):
    """The tasks a run config selects, plus what the console should say about it."""

    tasks: Annotated[list[Task], Field(description="Tasks to run, in run order.")]
    recorded_subset: Annotated[
        Optional[TaskSubset],
        Field(description="Subset to record in the results' Info, if any."),
    ] = None
    notices: Annotated[
        list[str], Field(description="Informational lines about the selection.")
    ] = []
    warnings: Annotated[
        list[str], Field(description="Lines warning that the selection is surprising.")
    ] = []


def resolve_tasks(config: RunConfig) -> ResolvedTasks:
    """Select the tasks a run config will run.

    Kept out of ``run_domain`` so a caller can find out what a config resolves
    to *without* running it. ``tau2 run refill`` needs exactly that: it deletes
    simulations before handing the config to the runner, so a config whose task
    set no longer covers the stored task ids has to fail while those
    simulations still exist. Two implementations of this would drift, and the
    drift would only show up after the delete.

    Raises:
        ValueError: If the task set does not contain the requested tasks, or
            the subset does not fit the task set.
    """
    from tau2.task_subsets import (
        canonical_subset_name,
        resolve_subset,
        subset_covering_ids,
    )

    notices: list[str] = []
    warnings: list[str] = []

    task_set_name = config.task_set_name or config.domain
    # Resolved once, here: the same subset object filters the tasks and lands
    # in the results' Info, so a run can never score on one subset and record
    # another.
    explicit_task_selection = bool(config.task_ids) or config.num_tasks is not None
    subset = resolve_subset(
        domain=config.domain,
        requested=config.task_subset,
        explicit_task_selection=explicit_task_selection,
    )
    if subset is None and explicit_task_selection:
        # The footgun this whole mechanism exists to remove: `--num-tasks 50`
        # on a domain that HAS a fixed 50 looks like the subset and is not —
        # it is a prefix of the file, which on telecom is a census of the
        # early scenario families and a zero-sample of the late ones.
        canonical = canonical_subset_name(config.domain)
        if canonical is not None:
            warnings.append(
                f"NOTE: domain '{config.domain}' has a fixed subset "
                f"('{canonical}'), but --task-ids/--num-tasks selects tasks "
                "explicitly, so it is NOT applied. Drop those flags to run "
                "the subset."
            )
    if subset is not None:
        notices.append(
            f"Task subset '{subset.name}': {subset.size} of "
            f"{subset.frame.size} {config.domain} tasks "
            f"({subset.strategy.value}, seed {subset.seed})."
        )
    tasks = get_tasks(
        task_set_name=task_set_name,
        task_split_name=config.task_split_name,
        task_ids=config.task_ids,
        num_tasks=config.num_tasks,
        task_subset=subset,
    )

    # Filter tasks based on agent's registered task filter (if any)
    effective_agent = config.effective_agent
    task_filter = registry.get_agent_task_filter(effective_agent)
    if task_filter is not None:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if task_filter(task)]
        notices.append(
            f"Running {len(tasks)} out of {total_num_tasks} tasks "
            f"for {effective_agent} (filtered)."
        )

    # A run that enumerated its tasks with --task-ids can still be scoring the
    # subset — the multilingual presets apply it when building the arm, so by
    # the time `tau2 run` sees the list the subset default is already off.
    # Record it if every task belongs to it; the ids were not filtered here, so
    # this is provenance only.
    recorded_subset = subset or subset_covering_ids(
        domain=config.domain, task_ids=[task.id for task in tasks]
    )
    return ResolvedTasks(
        tasks=tasks,
        recorded_subset=recorded_subset,
        notices=notices,
        warnings=warnings,
    )


def resolve_timeout(
    timeout: Optional[float], audio_native_config: Optional["AudioNativeConfig"]
) -> Optional[float]:
    """The wallclock guard for a run.

    Unset means "derive it": a voice run scales the guard off its own
    conversation budget so wallclock can never pre-empt simulated time, while a
    text run, whose max_steps is a turn count, falls back to the flat default.
    0 is the opt-out — RunConfig.timeout treats None as "no timeout", while a
    literal 0 would time every simulation out instantly.
    """
    from tau2.config import DEFAULT_TIMEOUT_SECONDS

    if timeout is None:
        if audio_native_config is not None:
            return audio_native_config.default_wallclock_timeout_seconds
        return DEFAULT_TIMEOUT_SECONDS
    return timeout or None


def trial_seeds(seed: Optional[int], num_trials: int) -> list[int]:
    """The per-trial seeds a run derives from its ``--seed``.

    The identity of a stored simulation is ``(trial, task_id, seed)`` — that
    triple is what resume matches on — so anything that wants to know which
    cells a directory is missing (see ``tau2 run refill``) has to derive the
    same seeds the batch runner does. One implementation, called by both:
    two copies of this that drift would make a refill re-run cells the run
    already holds, silently doubling them.

    Seeds the global RNG, as the batch runner has always done, so simulation
    behavior downstream of it is unchanged.
    """
    random.seed(seed)
    return [random.randint(0, 1000000) for _ in range(num_trials)]


def make_run_name(config: RunConfig) -> str:
    """Generate a run name from the run config."""
    is_voice = isinstance(config, VoiceRunConfig)

    if is_voice:
        llm_agent_name = (
            f"{config.audio_native_config.provider}-{config.audio_native_config.model}"
        )
    else:
        llm_agent_name = config.llm_agent
    clean_llm_agent_name = [x for x in llm_agent_name.split("/") if x][-1]
    agent_name = f"{config.effective_agent}_{clean_llm_agent_name}"

    clean_llm_user_name = [x for x in config.llm_user.split("/") if x][-1]
    user_name = f"{config.effective_user}_{clean_llm_user_name}"

    name = (
        f"{get_now(use_compact_format=True)}_{config.domain}_{agent_name}_{user_name}"
    )

    if is_voice:
        name = f"{name}_audio_native"

    return name


def get_info(config: RunConfig, **overrides) -> Info:
    """Create an Info object for storing run configuration metadata.

    Args:
        config: The run configuration (TextRunConfig or VoiceRunConfig).
        **overrides: Override specific fields (e.g., user_persona_config,
            user_voice_settings, speech_complexity, policy_override,
            task_subset, tasks_scored — the number of tasks actually handed to
            the runner, which an agent task filter can push below the subset's
            size).

    Returns:
        Info object with run metadata.
    """
    is_voice = isinstance(config, VoiceRunConfig)

    user_persona_config = overrides.get("user_persona_config")
    user_voice_settings = overrides.get("user_voice_settings")
    policy_override = overrides.get("policy_override")
    speech_complexity = overrides.get(
        "speech_complexity",
        config.speech_complexity if is_voice else None,
    )

    # Record the guidelines variant the run's user sim actually gets: voice
    # vs text, with tools when the domain exposes user tools.
    use_tools = False
    try:
        environment = registry.get_env_constructor(config.domain)()
        use_tools = bool(environment.get_user_tools())
    except Exception:
        pass
    if is_voice:
        global_user_sim_guidelines = get_global_user_sim_guidelines_voice(
            use_tools=use_tools
        )
    else:
        global_user_sim_guidelines = get_global_user_sim_guidelines(use_tools=use_tools)

    user_info = UserInfo(
        implementation=config.effective_user,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        global_simulation_guidelines=global_user_sim_guidelines,
        persona_config=user_persona_config,
        voice_settings=user_voice_settings,
    )

    # For voice mode, agent uses Realtime API, not a regular LLM
    if is_voice:
        agent_llm = (
            f"{config.audio_native_config.provider}:{config.audio_native_config.model}"
        )
        agent_llm_args = None
    else:
        agent_llm = config.llm_agent
        agent_llm_args = config.llm_args_agent

    agent_info = AgentInfo(
        implementation=config.effective_agent,
        llm=agent_llm,
        llm_args=agent_llm_args,
    )
    # Build env_kwargs so the environment is constructed with the correct
    # retrieval variant (needed for banking_knowledge; no-op for other domains).
    info_env_kwargs: dict = {}
    if getattr(config, "retrieval_config", None) is not None:
        info_env_kwargs["retrieval_variant"] = config.retrieval_config
        rk = dict(getattr(config, "retrieval_config_kwargs", None) or {})
        if rk:
            info_env_kwargs["retrieval_kwargs"] = rk

    environment_info = get_environment_info(
        config.domain, include_tool_info=False, env_kwargs=info_env_kwargs
    )
    if policy_override is not None:
        environment_info.policy = policy_override

    subset = overrides.get("task_subset")
    subset_info = None
    if subset is not None:
        subset_info = TaskSubsetInfo(
            name=subset.name,
            size=subset.size,
            tasks_scored=overrides.get("tasks_scored"),
            frame_task_set=subset.frame.task_set,
            frame_size=subset.frame.size,
            frame_digest=subset.frame.digest,
            strategy=subset.strategy.value,
            seed=subset.seed,
        )

    return Info(
        git_commit=get_commit_hash(),
        num_trials=config.num_trials,
        max_steps=config.effective_max_steps,
        max_errors=config.max_errors,
        user_info=user_info,
        agent_info=agent_info,
        environment_info=environment_info,
        task_set_name=config.task_set_name,
        user_persona_id=config.user_persona_id,
        text_input_style=getattr(config, "text_input_style", None),
        text_noise=getattr(config, "text_noise", None),
        communicate_judge_mode=config.communicate_judge_mode,
        seed=config.seed,
        speech_complexity=speech_complexity,
        channel_effects_mode=getattr(config, "channel_effects_mode", None),
        speech_effects_mode=getattr(config, "speech_effects_mode", None),
        audio_native_config=getattr(config, "audio_native_config", None),
        retrieval_config=getattr(config, "retrieval_config", None),
        retrieval_config_kwargs=getattr(config, "retrieval_config_kwargs", None),
        task_subset=subset_info,
        task_split_name=config.task_split_name,
        # 0 records "this run had no wallclock guard" — the `--timeout 0`
        # opt-out, which RunConfig carries as None. Written as 0 so it is
        # distinguishable from a results file predating the field, which is
        # also None and which a refill must not silently cap.
        timeout=config.timeout if config.timeout is not None else 0.0,
        verbose_logs=config.verbose_logs,
        # Sorted so two runs with the same axes hash the same regardless of
        # the set's iteration order — the resume config check compares hashes.
        scores=sorted(config.scores, key=lambda score: score.value),
        nativeness_judge=config.nativeness_judge,
        quality_judge=config.quality_judge,
        delivery_judge=config.delivery_judge if is_voice else None,
        hallucination_retries=config.hallucination_retries,
        enforce_communication_protocol=(
            None if is_voice else config.enforce_communication_protocol
        ),
        auto_review=config.auto_review,
        review_mode=config.review_mode,
        review_model=config.review_model,
    )

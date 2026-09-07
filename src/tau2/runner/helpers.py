"""
Helper functions for task loading, run configuration, and metadata.
"""

from typing import Annotated, Optional

from pydantic import BaseModel, Field

from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    RunConfig,
    UserInfo,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import EnvironmentInfo
from tau2.registry import RegistryInfo, registry
from tau2.runner.complications import profile_task_filter
from tau2.user.user_simulator import (
    CallDirection,
    call_direction_for_task,
    get_global_user_sim_guidelines,
    get_global_user_sim_guidelines_voice,
)
from tau2.utils.utils import get_commit_hash, get_now


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
) -> list[Task]:
    """Load tasks with optional filtering by IDs and count.

    Args:
        task_set_name: The task set to load from.
        task_split_name: Optional split name (e.g., "base").
        task_ids: If provided, only return tasks with these IDs.
        num_tasks: If provided, limit to this many tasks.

    Returns:
        List of tasks matching the criteria.

    Raises:
        ValueError: If task_ids are specified but some are not found.
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
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


class ResolvedTasks(BaseModel):
    """Tasks selected for a run plus reviewer-readable selection notes."""

    tasks: Annotated[list[Task], Field(description="Tasks to run, in run order.")]
    notices: list[str] = Field(default_factory=list)


def resolve_tasks(config: RunConfig) -> ResolvedTasks:
    """Resolve task IDs, count, agent filters, and complication-tier filters."""
    task_set_name = config.task_set_name or config.domain
    tasks = get_tasks(
        task_set_name=task_set_name,
        task_split_name=config.task_split_name,
        task_ids=config.task_ids,
        num_tasks=config.num_tasks,
    )
    notices: list[str] = []
    task_filter = registry.get_agent_task_filter(config.effective_agent)
    if task_filter is not None:
        total = len(tasks)
        tasks = [task for task in tasks if task_filter(task)]
        notices.append(
            f"Running {len(tasks)} out of {total} tasks for "
            f"{config.effective_agent} (filtered)."
        )
    profile_filter = profile_task_filter(config)
    if profile_filter is not None:
        total = len(tasks)
        tasks = [task for task in tasks if profile_filter(task)]
        notices.append(
            f"Complication profile '{config.complication_profile.value}': "
            f"{len(tasks)} of {total} tasks (hard tier only)."
        )
    return ResolvedTasks(tasks=tasks, notices=notices)


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
            user_voice_settings, speech_complexity, policy_override).

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

    use_tools = False
    direction = overrides.get("call_direction", CallDirection.INBOUND)
    try:
        environment = registry.get_env_constructor(config.domain)()
        use_tools = bool(environment.get_user_tools())
    except Exception:
        pass
    if direction is CallDirection.INBOUND:
        try:
            tasks = load_tasks(config.task_set_name or config.domain)
            direction = call_direction_for_task(tasks[0])
        except Exception:
            pass
    if not use_tools:
        direction = CallDirection.INBOUND

    # Record the exact inbound/outbound guideline variant used at runtime.
    if is_voice:
        global_user_sim_guidelines = get_global_user_sim_guidelines_voice(
            use_tools=use_tools, direction=direction
        )
    else:
        global_user_sim_guidelines = get_global_user_sim_guidelines(
            use_tools=use_tools, direction=direction
        )

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

    return Info(
        git_commit=get_commit_hash(),
        num_trials=config.num_trials,
        max_steps=config.effective_max_steps,
        max_errors=config.max_errors,
        user_info=user_info,
        agent_info=agent_info,
        environment_info=environment_info,
        seed=config.seed,
        speech_complexity=speech_complexity,
        channel_effects_mode=getattr(config, "channel_effects_mode", None),
        speech_effects_mode=getattr(config, "speech_effects_mode", None),
        complication_profile=config.complication_profile.value,
        complication_rate=config.complication_rate,
        audio_native_config=getattr(config, "audio_native_config", None),
        retrieval_config=getattr(config, "retrieval_config", None),
        retrieval_config_kwargs=getattr(config, "retrieval_config_kwargs", None),
    )

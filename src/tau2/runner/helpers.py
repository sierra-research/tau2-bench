"""
Helper functions for task loading, run configuration, and metadata.
"""

from typing import Optional

from tau2.config import DOMAIN_TOOLS_VERSION, VOICE_INSTRUCTION_PROMPT_VERSION
from tau2.data_model.elicitation import ElicitationStrategy
from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    RunConfig,
    UserInfo,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import EnvironmentInfo
from tau2.multilingual.runtime import LoadedLanguage
from tau2.registry import RegistryInfo, registry
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


def load_tasks(
    task_set_name: str,
    task_split_name: Optional[str] = None,
    task_kwargs: Optional[dict] = None,
) -> list[Task]:
    """Load tasks for a given task set, optionally filtering by split."""
    task_loader = registry.get_tasks_loader(task_set_name)
    tasks = task_loader(task_split_name=task_split_name, **(task_kwargs or {}))
    return tasks


def get_tasks(
    task_set_name: str,
    task_split_name: Optional[str] = None,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
    task_kwargs: Optional[dict] = None,
) -> list[Task]:
    """Load registered tasks with optional filtering by IDs and count.

    This reads the ordinary task registry. Runs select tasks through their
    language binding (``select_run_tasks``), which may replace reviewed tasks
    or publish a complete localized population.

    Raises:
        ValueError: If task_ids are specified but some are not found.
    """
    tasks = load_tasks(
        task_set_name=task_set_name,
        task_split_name=task_split_name,
        task_kwargs=task_kwargs,
    )
    return filter_tasks(tasks, task_ids=task_ids, num_tasks=num_tasks)


def filter_tasks(
    tasks: list[Task],
    *,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> list[Task]:
    """Apply explicit task IDs (all must exist) and then a count limit."""
    if task_ids is not None:
        selected = [task for task in tasks if task.id in task_ids]
        if len(selected) != len(task_ids):
            missing = set(task_ids) - {task.id for task in selected}
            raise ValueError(f"Not all tasks were found: {sorted(missing)}")
        tasks = selected
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def select_run_tasks(config: RunConfig, loaded: LoadedLanguage) -> list[Task]:
    """A run's tasks: its language binding's population, filtered by the config."""
    return filter_tasks(
        loaded.tasks, task_ids=config.task_ids, num_tasks=config.num_tasks
    )


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

    name = f"{name}_{config.language}"
    if is_voice:
        name = f"{name}_audio_native"

    return name


def get_info(
    config: RunConfig, *, tasks: Optional[list[Task]] = None, **overrides
) -> Info:
    """Create an Info object for storing run configuration metadata.

    Args:
        config: The run configuration (TextRunConfig or VoiceRunConfig).
        tasks: Actual selected tasks (or the single leased task in a worker).
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

    # Pack prompts are persona-specific and recorded in language_tasks below.
    global_user_sim_guidelines = None

    scoring_version = None
    prompt_version = None
    catalog_version = None
    spell_detection_version = None
    if config.elicitation is not None:
        from tau2.domains.intake.complications import COMPLICATION_CATALOG_VERSION
        from tau2.domains.intake.folds import SCORING_VERSION
        from tau2.domains.intake.prompts import (
            ELICITATION_PROMPT_VERSION,
            get_user_guidelines,
        )

        global_user_sim_guidelines = get_user_guidelines(
            is_voice=is_voice,
            agent_directed=config.elicitation.strategy
            == ElicitationStrategy.AGENT_DIRECTED,
        )
        scoring_version = SCORING_VERSION
        prompt_version = ELICITATION_PROMPT_VERSION
        catalog_version = COMPLICATION_CATALOG_VERSION
        if (
            is_voice
            and config.elicitation.strategy == ElicitationStrategy.AGENT_DIRECTED
        ):
            from tau2.domains.intake.spell_events import SPELL_EVENT_DETECTION_VERSION

            spell_detection_version = SPELL_EVENT_DETECTION_VERSION

    if config.elicitation is not None and config.language not in {None, "en"}:
        from tau2.domains.intake.localized import (
            LOCALIZED_STIMULUS_VERSION,
            NATIVE_SCORING_VERSION,
        )
        from tau2.domains.intake.prompts import LOCALIZED_ELICITATION_PROMPT_VERSION
        from tau2.domains.intake.spell_events import (
            NATIVE_SPELL_EVENT_DETECTION_VERSION,
        )

        scoring_version = NATIVE_SCORING_VERSION
        prompt_version = LOCALIZED_ELICITATION_PROMPT_VERSION
        catalog_version = LOCALIZED_STIMULUS_VERSION
        if spell_detection_version is not None:
            spell_detection_version = NATIVE_SPELL_EVENT_DETECTION_VERSION

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
    from tau2.multilingual.runtime import (
        language_environment_kwargs,
        pin_language_artifacts,
        resolve_task,
    )

    loaded = pin_language_artifacts(config, update=False)
    selected = tasks if tasks is not None else select_run_tasks(config, loaded)
    language_tasks = {
        task.id: resolve_task(loaded, task, config.seed or 0) for task in selected
    }

    # Resolve the run-level policy for its channel/retrieval settings. Individual
    # simulations also record their task-specific policy (e.g. callback orders).
    info_env_kwargs: dict = language_environment_kwargs(loaded)
    if config.elicitation is not None:
        info_env_kwargs["channel"] = "voice" if is_voice else "text"
        info_env_kwargs["strategy"] = config.elicitation.strategy
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

    shared_voice_prompt = is_voice and (
        config.elicitation is None
        or config.elicitation.strategy != ElicitationStrategy.AGENT_DIRECTED
    )
    return Info(
        domain_tools_version=DOMAIN_TOOLS_VERSION,
        voice_instruction_prompt_version=(
            VOICE_INSTRUCTION_PROMPT_VERSION if shared_voice_prompt else None
        ),
        language=config.language,
        pack_extension=loaded.pack_extension,
        localize_db=config.localize_db,
        language_artifacts=loaded.artifact_pins,
        language_tasks=language_tasks,
        elicitation=config.elicitation,
        elicitation_scoring_version=scoring_version,
        elicitation_prompt_version=prompt_version,
        elicitation_catalog_version=catalog_version,
        elicitation_spell_detection_version=spell_detection_version,
        benchmark=config.benchmark,
        communicate_scoring=config.communicate_scoring,
        git_commit=get_commit_hash(),
        num_trials=config.num_trials,
        max_steps=config.effective_max_steps,
        max_errors=config.max_errors,
        user_info=user_info,
        agent_info=agent_info,
        environment_info=environment_info,
        seed=config.seed,
        speech_complexity=speech_complexity,
        channel_effects_mode=(config.channel_effects_mode if is_voice else None),
        speech_effects_mode=(config.speech_effects_mode if is_voice else None),
        pass3_condition=(config.pass3_condition if is_voice else None),
        voice_sampling_seed=(config.voice_sampling_seed if is_voice else None),
        audio_native_config=getattr(config, "audio_native_config", None),
        retrieval_config=getattr(config, "retrieval_config", None),
        retrieval_config_kwargs=getattr(config, "retrieval_config_kwargs", None),
    )

"""
Layer 2: Build functions.

Turn config/names into live instances (environment, agent, user, orchestrator).
Uses the registry for name resolution. Callers who want full control can skip
this layer and construct instances directly.
"""

import uuid
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from tau2.agent.base_agent import FullDuplexAgent, HalfDuplexAgent
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.simulation import (
    AudioNativeConfig,
    RunConfig,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.data_model.voice import SpeechComplexity, SynthesisConfig, VoiceSettings
from tau2.environment.environment import Environment
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, UserSimulator
from tau2.user.user_simulator_base import FullDuplexUser, HalfDuplexUser
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator
from tau2.user_simulation_voice_presets import (
    get_or_load_task_voice_config,
)

# =============================================================================
# Low-level build functions (no RunConfig needed)
# =============================================================================


def build_environment(
    domain: str,
    *,
    solo_mode: bool = False,
    env_kwargs: Optional[dict] = None,
) -> Environment:
    """Build an environment from a domain name.

    Uses the registry to resolve the domain name to an environment constructor.

    Args:
        domain: Domain name (e.g., "airline", "retail", "mock").
        solo_mode: If True, environment is built in solo mode (agent gets
            access to both agent and user tools).
        env_kwargs: Additional keyword arguments passed to the environment
            constructor (e.g., retrieval_variant, task for banking_knowledge).

    Returns:
        A fully constructed Environment instance.
    """
    env_constructor = registry.get_env_constructor(domain)
    kwargs = dict(env_kwargs or {})
    if solo_mode:
        kwargs["solo_mode"] = True
    return env_constructor(**kwargs)


def build_agent(
    agent_name: str,
    environment: Environment,
    *,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    task: Optional[Task] = None,
    audio_native_config: Optional[AudioNativeConfig] = None,
    solo_mode: bool = False,
    audio_taps_dir: Optional[Path] = None,
) -> Union[HalfDuplexAgent, FullDuplexAgent]:
    """Build an agent from a registered name and an environment.

    Uses the registry to resolve the agent name to a factory function,
    then calls it with the appropriate parameters.

    Args:
        agent_name: Registered agent name (e.g., "llm_agent", "llm_agent_gt",
            "discrete_time_audio_native_agent", or "experimental:my_agent").
        environment: The environment to extract tools and policy from.
        llm: LLM model name for the agent (half-duplex agents).
        llm_args: LLM arguments for the agent (half-duplex agents).
        task: The task (required for some agents like llm_agent_gt, llm_agent_solo).
        audio_native_config: Audio config (full-duplex agents).
        solo_mode: If True, agent tools include both agent and user tools.

    Returns:
        A fully constructed agent instance.

    Raises:
        ValueError: If the agent name has no factory registered.
    """
    agent_factory = registry.get_agent_factory(agent_name)
    if agent_factory is None:
        raise ValueError(
            f"Agent '{agent_name}' has no factory registered. "
            f"Register a factory with registry.register_agent_factory()."
        )

    # Collect tools from environment
    tools = environment.get_tools()
    if solo_mode:
        try:
            user_tools = environment.get_user_tools()
            if user_tools:
                tools = tools + user_tools
        except Exception:
            pass

    return agent_factory(
        tools=tools,
        domain_policy=environment.get_policy(),
        llm=llm,
        llm_args=llm_args,
        task=task,
        audio_native_config=audio_native_config,
        audio_taps_dir=audio_taps_dir,
    )


def build_user(
    user_name: str,
    environment: Environment,
    task: Task,
    *,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    persona_config: Optional[PersonaConfig] = None,
    solo_mode: bool = False,
) -> HalfDuplexUser:
    """Build a half-duplex user from a registered name.

    Uses the registry to resolve the user name to a constructor.

    Args:
        user_name: Registered user name (e.g., "user_simulator", "dummy_user").
        environment: The environment to extract user tools from.
        task: The task (used for user instructions).
        llm: LLM model name for the user simulator.
        llm_args: LLM arguments for the user simulator.
        persona_config: Persona configuration (verbosity, interrupt tendency).
        solo_mode: If True, validates that DummyUser is used appropriately.

    Returns:
        A fully constructed half-duplex user instance.

    Raises:
        AssertionError: If DummyUser is used without solo_mode.
    """
    UserConstructor = registry.get_user_constructor(user_name)

    try:
        user_tools = environment.get_user_tools(include=task.user_tools) or None
    except Exception:
        user_tools = None

    # Validate DummyUser usage
    if issubclass(UserConstructor, DummyUser):
        assert solo_mode, "Dummy user can only be used with solo agent"

    user_kwargs = {
        "tools": user_tools,
        "instructions": str(task.user_scenario),
        "llm": llm,
        "llm_args": llm_args,
    }
    if issubclass(UserConstructor, UserSimulator):
        user_kwargs["persona_config"] = persona_config

    return UserConstructor(**user_kwargs)


def build_voice_user(
    environment: Environment,
    task: Task,
    audio_native_config: AudioNativeConfig,
    *,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    voice_settings: Optional[VoiceSettings] = None,
    persona_config: Optional[PersonaConfig] = None,
    speech_complexity: SpeechComplexity = "regular",
    seed: int = 42,
    domain: Optional[str] = None,
    hallucination_feedback: Optional[str] = None,
    audio_taps_dir: Optional[Path] = None,
) -> FullDuplexUser:
    """Build a full-duplex voice user simulator.

    Handles all voice configuration wiring: sampling voice configs per task,
    merging effect configs, creating speech environment, and constructing the
    VoiceStreamingUserSimulator with all timing parameters from audio_native_config.

    Args:
        environment: The environment to extract user tools from.
        task: The task (used for user instructions and voice config sampling).
        audio_native_config: Full audio-native configuration (timing, thresholds, etc.).
        llm: LLM model name for the user simulator.
        llm_args: LLM arguments for the user simulator.
        voice_settings: Base voice settings. If None, defaults are created.
            Deep copied internally to avoid mutation.
        persona_config: Persona configuration. If None, derived from sampled voice config.
        speech_complexity: Speech environment complexity level.
        seed: Base seed for voice config sampling. Per-task seed is derived as
            seed + hash(task.id) % 1000000.
        domain: Domain name (used for loading pre-sampled voice configs).
            If None, extracted from environment.
        hallucination_feedback: Optional feedback from a previous hallucination
            check. If provided, appended to user instructions to help avoid
            repeating the same errors on retry.

    Returns:
        A fully constructed VoiceStreamingUserSimulator.
    """
    if domain is None:
        domain = environment.get_domain_name()

    try:
        user_tools = environment.get_user_tools(include=task.user_tools) or None
    except Exception:
        user_tools = None

    # Set up voice settings (deep copy to avoid mutating caller's settings)
    if voice_settings is not None:
        task_voice_settings = deepcopy(voice_settings)
    else:
        task_voice_settings = VoiceSettings(
            transcription_config=None,
            synthesis_config=SynthesisConfig(),
        )

    # Get voice config for this task (from pre-sampled file or sample on the fly)
    task_seed = seed + hash(task.id) % 1000000
    sampled_voice_config = get_or_load_task_voice_config(
        domain=domain,
        task_id=task.id,
        task_seed=task_seed,
        complexity=speech_complexity,
        synthesis_config=task_voice_settings.synthesis_config,
    )

    # Update synthesis_config with merged effect configs
    task_voice_settings.synthesis_config.channel_effects_config = (
        sampled_voice_config.channel_effects_config
    )
    task_voice_settings.synthesis_config.source_effects_config = (
        sampled_voice_config.source_effects_config
    )
    task_voice_settings.synthesis_config.speech_effects_config = (
        sampled_voice_config.speech_effects_config
    )

    # Set speech environment
    speech_environment = sampled_voice_config.to_speech_environment(task_seed)
    task_voice_settings.speech_environment = speech_environment

    # Use provided persona config or fall back to sampled config
    if persona_config is None:
        persona_config = sampled_voice_config.persona_config

    user_instructions = str(task.user_scenario)
    if hallucination_feedback:
        user_instructions += f"\n\n{hallucination_feedback}"

    return VoiceStreamingUserSimulator(
        tools=user_tools,
        instructions=user_instructions,
        llm=llm,
        llm_args=llm_args,
        voice_settings=task_voice_settings,
        chunk_size=audio_native_config.user_chunk_size,
        wait_to_respond_threshold_other=audio_native_config.wait_to_respond_threshold_other_ticks,
        wait_to_respond_threshold_self=audio_native_config.wait_to_respond_threshold_self_ticks,
        yield_threshold_when_interrupted=audio_native_config.yield_threshold_when_interrupted_ticks,
        yield_threshold_when_interrupting=audio_native_config.yield_threshold_when_interrupting_ticks,
        backchannel_min_threshold=(
            int(
                sampled_voice_config.backchannel_min_threshold
                / audio_native_config.tick_duration_seconds
            )
            if sampled_voice_config.backchannel_min_threshold is not None
            else None
        ),
        backchannel_max_threshold=audio_native_config.backchannel_max_threshold_ticks,
        backchannel_poisson_rate=audio_native_config.backchannel_poisson_rate,
        use_llm_backchannel=sampled_voice_config.use_llm_backchannel,
        interruption_check_interval=audio_native_config.interruption_check_interval_ticks,
        integration_ticks=audio_native_config.integration_ticks,
        silence_annotation_threshold_ticks=audio_native_config.silence_annotation_threshold_ticks,
        tick_duration_seconds=audio_native_config.tick_duration_seconds,
        persona_config=persona_config,
        audio_taps_dir=audio_taps_dir,
    )


# =============================================================================
# High-level build functions (use RunConfig)
# =============================================================================


def _build_env_kwargs(config: RunConfig, task: Task) -> dict:
    """Build env_kwargs from a RunConfig for the environment constructor.

    Extracts retrieval-related config (banking_knowledge domain) and includes
    the task reference needed for golden_retrieval policy.
    """
    env_kwargs: dict = {}
    retrieval_config = getattr(config, "retrieval_config", None)
    if retrieval_config is not None:
        env_kwargs["retrieval_variant"] = retrieval_config
        env_kwargs["task"] = task
        retrieval_config_kwargs = getattr(config, "retrieval_config_kwargs", None)
        if retrieval_config_kwargs:
            env_kwargs["retrieval_kwargs"] = retrieval_config_kwargs
    return env_kwargs


def build_text_orchestrator(
    config: TextRunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
    user_persona_config: Optional[PersonaConfig] = None,
) -> Orchestrator:
    """Build a half-duplex (text) orchestrator from a TextRunConfig.

    Args:
        config: Text run configuration.
        task: The task to run.
        seed: Per-trial seed. If None, uses config.seed.
        simulation_id: Unique simulation ID. If None, a UUID is generated.
        user_persona_config: Persona config for the user simulator.

    Returns:
        A fully constructed Orchestrator, ready for run_simulation().

    Example:
        config = TextRunConfig(domain="airline", agent="llm_agent")
        tasks = get_tasks("airline")
        orchestrator = build_text_orchestrator(config, tasks[0], seed=42)
        result = run_simulation(orchestrator)
    """
    if simulation_id is None:
        simulation_id = str(uuid.uuid4())
    if seed is None:
        seed = config.seed

    solo_mode = registry.get_agent_metadata(
        config.effective_agent, "solo_mode", default=False
    )
    domain = config.domain
    env_kwargs = _build_env_kwargs(config, task)

    environment = build_environment(domain, solo_mode=solo_mode, env_kwargs=env_kwargs)

    agent = build_agent(
        config.effective_agent,
        environment,
        llm=config.llm_agent,
        llm_args=config.llm_args_agent,
        task=task,
        solo_mode=solo_mode,
    )

    user = build_user(
        config.effective_user,
        environment,
        task,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        persona_config=user_persona_config,
        solo_mode=solo_mode,
    )

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=config.effective_max_steps,
        max_errors=config.max_errors,
        seed=seed,
        solo_mode=solo_mode,
        simulation_id=simulation_id,
        validate_communication=config.enforce_communication_protocol,
        timeout=config.timeout,
    )

    logger.debug(
        f"Built text orchestrator: domain={domain}, agent={config.effective_agent}, "
        f"user={config.effective_user}, task={task.id}"
    )

    return orchestrator


def build_voice_orchestrator(
    config: VoiceRunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
    user_voice_settings: Optional[VoiceSettings] = None,
    user_persona_config: Optional[PersonaConfig] = None,
    hallucination_feedback: Optional[str] = None,
    audio_taps_dir: Optional[Path] = None,
) -> FullDuplexOrchestrator:
    """Build a full-duplex (voice) orchestrator from a VoiceRunConfig.

    Args:
        config: Voice run configuration.
        task: The task to run.
        seed: Per-trial seed. If None, uses config.seed.
        simulation_id: Unique simulation ID. If None, a UUID is generated.
        user_voice_settings: Pre-computed voice settings (from run-level setup).
            If None, defaults are created.
        user_persona_config: Pre-computed persona config (from run-level setup).
            If None, derived from sampled voice config.
        hallucination_feedback: Optional feedback from a previous hallucination
            check. If provided, appended to user instructions to help avoid
            repeating the same errors on retry.

    Returns:
        A fully constructed FullDuplexOrchestrator, ready for run_simulation().

    Raises:
        ValueError: If the agent is registered with solo_mode=True, which is
            not supported for voice/full-duplex runs.

    Example:
        config = VoiceRunConfig(domain="airline", audio_native_config=AudioNativeConfig())
        tasks = get_tasks("airline")
        orchestrator = build_voice_orchestrator(config, tasks[0], seed=42)
        result = run_simulation(orchestrator)
    """
    if simulation_id is None:
        simulation_id = str(uuid.uuid4())
    if seed is None:
        seed = config.seed

    # Solo mode is not supported for voice/full-duplex runs
    solo_mode = registry.get_agent_metadata(
        config.effective_agent, "solo_mode", default=False
    )
    if solo_mode:
        raise ValueError(
            f"Agent '{config.effective_agent}' is registered with solo_mode=True, "
            f"but solo mode is not supported for voice/full-duplex runs."
        )

    domain = config.domain
    env_kwargs = _build_env_kwargs(config, task)

    environment = build_environment(domain, env_kwargs=env_kwargs)

    agent = build_agent(
        config.effective_agent,
        environment,
        audio_native_config=config.audio_native_config,
        audio_taps_dir=audio_taps_dir,
    )

    user = build_voice_user(
        environment,
        task,
        config.audio_native_config,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        voice_settings=user_voice_settings,
        persona_config=user_persona_config,
        speech_complexity=config.speech_complexity,
        seed=seed or 42,
        domain=domain,
        hallucination_feedback=hallucination_feedback,
        audio_taps_dir=audio_taps_dir,
    )

    orchestrator = FullDuplexOrchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=config.effective_max_steps,
        max_errors=config.max_errors,
        seed=seed,
        simulation_id=simulation_id,
        tick_duration_seconds=config.audio_native_config.tick_duration_seconds,
        timeout=config.timeout,
    )

    logger.debug(
        f"Built voice orchestrator: domain={domain}, agent={config.effective_agent}, "
        f"user={config.effective_user}, task={task.id}"
    )

    return orchestrator


def build_orchestrator(
    config: RunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
    user_voice_settings: Optional[VoiceSettings] = None,
    user_persona_config: Optional[PersonaConfig] = None,
    hallucination_feedback: Optional[str] = None,
    audio_taps_dir: Optional[Path] = None,
) -> Union[Orchestrator, FullDuplexOrchestrator]:
    """Build a ready-to-run orchestrator from a RunConfig and task.

    Dispatches to build_text_orchestrator or build_voice_orchestrator
    based on the config type.

    Args:
        config: Text or voice run configuration.
        task: The task to run.
        seed: Per-trial seed. If None, uses config.seed.
        simulation_id: Unique simulation ID. If None, a UUID is generated.
        user_voice_settings: Pre-computed voice settings (voice mode only).
        user_persona_config: Pre-computed persona config.
        hallucination_feedback: Optional feedback from a previous hallucination
            check (voice mode only). Passed through to build_voice_orchestrator.

    Returns:
        A fully constructed Orchestrator or FullDuplexOrchestrator.
    """
    if isinstance(config, VoiceRunConfig):
        return build_voice_orchestrator(
            config,
            task,
            seed=seed,
            simulation_id=simulation_id,
            user_voice_settings=user_voice_settings,
            user_persona_config=user_persona_config,
            hallucination_feedback=hallucination_feedback,
            audio_taps_dir=audio_taps_dir,
        )
    else:
        return build_text_orchestrator(
            config,
            task,
            seed=seed,
            simulation_id=simulation_id,
            user_persona_config=user_persona_config,
        )

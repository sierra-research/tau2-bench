"""
Layer 2: Build functions.

Turn config/names into live instances (environment, agent, user, orchestrator).
Uses the registry for name resolution. Callers who want full control can skip
this layer and construct instances directly.
"""

import hashlib
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from tau2.agent.base_agent import FullDuplexAgent, HalfDuplexAgent
from tau2.backchannel import BackchannelSettings, default_backchannel_settings
from tau2.config import (
    DEFAULT_CHANNEL_EFFECTS_MODE,
    DEFAULT_SPEECH_EFFECTS_MODE,
    ELICITATION_DOMAINS,
)
from tau2.data_model.elicitation import ElicitationConfig, ElicitationStrategy
from tau2.data_model.language import CallContext, LanguageTaskInfo
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import RunConfig, TextRunConfig, VoiceRunConfig
from tau2.data_model.tasks import Task
from tau2.data_model.voice import (
    SpeechComplexity,
    SpeechEnvironment,
    SynthesisConfig,
    VoiceSettings,
)
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.multilingual.agent_guidance import localized_transfer_policy
from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.runner.elicitation import sample_run_complication, user_prompt_task
from tau2.user.user_simulator import DummyUser, UserSimulator
from tau2.user.user_simulator_base import FullDuplexUser, HalfDuplexUser
from tau2.user_simulation_voice_presets import (
    VoiceSelection,
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


def agent_domain_policy(
    environment: Environment, language_guidance: str = "", *, language: str = "en"
) -> str:
    """Attach resolved language guidance to the ordinary domain policy."""
    policy = localized_transfer_policy(
        environment.get_policy(), environment.get_domain_name(), language
    )
    return policy + ("\n\n" + language_guidance if language_guidance else "")


def user_tools_for_task(environment: Environment, task: Task) -> Optional[list[Tool]]:
    """Select the same optional caller tools for execution and prompt review."""
    try:
        return environment.get_user_tools(include=task.user_tools) or None
    except Exception:
        return None


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
    language_guidance: str = "",
    language: str = "en",
    elicitation_config: Optional[ElicitationConfig] = None,
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
        elicitation_config: Intake strategy settings for its domain-owned prompts.

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

    domain_kwargs = {}
    domain = environment.get_domain_name()
    if domain in ELICITATION_DOMAINS:
        domain_kwargs["additional_stop_tool_names"] = frozenset({"end_call"})
        if audio_native_config is not None:
            from tau2.domains.intake.prompts import get_agent_instructions

            domain_kwargs["agent_instruction_override"] = get_agent_instructions(
                audio_native_config.provider,
                agent_directed=(elicitation_config or ElicitationConfig()).strategy
                == ElicitationStrategy.AGENT_DIRECTED,
            )

    return agent_factory(
        tools=tools,
        domain_policy=agent_domain_policy(
            environment, language_guidance, language=language
        ),
        llm=llm,
        llm_args=llm_args,
        task=task,
        audio_native_config=audio_native_config,
        audio_taps_dir=audio_taps_dir,
        **domain_kwargs,
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

    user_tools = user_tools_for_task(environment, task)

    # Validate DummyUser usage
    if issubclass(UserConstructor, DummyUser):
        assert solo_mode, "Dummy user can only be used with solo agent"
        # The solo agent plays both sides; the dummy user takes no configuration.
        return UserConstructor()

    user_kwargs = {
        "tools": user_tools,
        "instructions": str(task.user_scenario),
        "llm": llm,
        "llm_args": llm_args,
    }
    if issubclass(UserConstructor, UserSimulator):
        user_kwargs["persona_config"] = persona_config
        if environment.get_domain_name() in ELICITATION_DOMAINS:
            from tau2.domains.intake.prompts import get_user_guidelines

            user_kwargs["global_guidelines_override"] = get_user_guidelines(
                is_voice=False, agent_directed=False, persona=persona_config
            )

    return UserConstructor(**user_kwargs)


def build_voice_user(
    environment: Environment,
    task: Task,
    audio_native_config: AudioNativeConfig,
    *,
    language_context: LanguageTaskInfo,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    voice_settings: Optional[VoiceSettings] = None,
    persona_config: Optional[PersonaConfig] = None,
    speech_complexity: SpeechComplexity = "regular",
    seed: int = 42,
    domain: Optional[str] = None,
    hallucination_feedback: Optional[str] = None,
    audio_taps_dir: Optional[Path] = None,
    elicitation_config: Optional[VoiceRunConfig] = None,
    channel_effects_mode: str = DEFAULT_CHANNEL_EFFECTS_MODE,
    speech_effects_mode: str = DEFAULT_SPEECH_EFFECTS_MODE,
    force_sample_voice_config: bool = False,
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
        language_context: Resolved voice-mode language context; its pack persona
            voices the caller unless the domain owns the caller.
        persona_config: Persona for domain-owned callers. If None, derived from
            the sampled voice config.
        speech_complexity: Speech environment complexity level.
        seed: Base seed for voice config sampling, offset by a stable digest of
            the canonical task ID so draws align across processes and languages.
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

    user_tools = user_tools_for_task(environment, task)

    # Set up voice settings (deep copy to avoid mutating caller's settings)
    if voice_settings is not None:
        task_voice_settings = deepcopy(voice_settings)
    else:
        task_voice_settings = VoiceSettings(
            transcription_config=None,
            synthesis_config=SynthesisConfig(),
        )

    if language_context.mode != "voice":
        raise ValueError("voice users require a voice-mode language context")
    persona = language_context.persona
    selection = None
    if persona is not None:
        from tau2.data_model.audio_effects import UserSpeechInsert

        selection = VoiceSelection(
            persona_name=persona.persona_id,
            persona_config=persona,
            non_directed_phrases=[
                UserSpeechInsert(text=text, type="non_directed_phrase")
                for text in persona.voice.non_directed_phrases
            ],
            vocal_tics=[
                UserSpeechInsert(text=text, type="vocal_tic")
                for text in persona.voice.vocal_tics
            ],
            speech_insert_events_per_minute=language_context.speech_insert_events_per_minute,
        )

    # Keep retries stable across processes and acoustic draws aligned across
    # languages through the canonical domain task identity.
    source_id = language_context.canonical_task_id
    task_seed = (
        seed
        + int.from_bytes(hashlib.sha256(source_id.encode()).digest()[:8], "big")
        % 1000000
    )
    sampled_voice_config = get_or_load_task_voice_config(
        domain=domain,
        task_id=source_id,
        task_seed=task_seed,
        complexity=speech_complexity,
        synthesis_config=task_voice_settings.synthesis_config,
        selection=selection,
        channel_effects_mode=channel_effects_mode,
        speech_effects_mode=speech_effects_mode,
        force_sample=force_sample_voice_config,
    )

    if persona is not None:
        synthesis = task_voice_settings.synthesis_config
        synthesis.spelled_entity_tables = persona.voice.spelled_entity_tables
        synthesis.provider_config.language_code = language_context.language.split("-")[
            0
        ]
        synthesis.provider_config.voice_id = persona.voice.voice_id

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
    if persona is not None:
        speech_environment = SpeechEnvironment(
            **sampled_voice_config.model_dump(exclude={"persona_config"}),
            voice_seed=task_seed,
            voice_id=persona.voice.voice_id,
        )
        persona_config = persona
    else:
        speech_environment = sampled_voice_config.to_speech_environment(task_seed)
        persona_config = persona_config or sampled_voice_config.persona_config
    task_voice_settings.speech_environment = speech_environment

    # Crossed speech effects change caller behavior while retaining the
    # selected pack persona's identity and gender.
    if speech_effects_mode != "regular":
        persona_config = sampled_voice_config.persona_config
    user_instructions = str(task.user_scenario)
    if hallucination_feedback:
        user_instructions += f"\n\n{hallucination_feedback}"

    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    domain_kwargs = {}
    if domain in ELICITATION_DOMAINS:
        from tau2.domains.intake.prompts import (
            OUTBOUND_ROLE_REMINDER,
            get_user_guidelines,
        )
        from tau2.domains.intake.pronunciation_map import build_pronunciation_map

        settings = (
            elicitation_config.elicitation
            if elicitation_config is not None
            and elicitation_config.elicitation is not None
            else ElicitationConfig()
        )
        agent_directed = settings.strategy == ElicitationStrategy.AGENT_DIRECTED
        domain_kwargs["global_guidelines_override"] = get_user_guidelines(
            is_voice=True, agent_directed=agent_directed, persona=persona_config
        )
        domain_kwargs["role_reminder_override"] = OUTBOUND_ROLE_REMINDER
        selected_stimulus = None
        locale_data = None
        if elicitation_config is not None and elicitation_config.language not in {
            None,
            "en",
        }:
            from tau2.runner.elicitation import localized_intake_context

            intake_context = localized_intake_context(elicitation_config, task)
            locale_data = intake_context.data
            selected_stimulus = locale_data.for_task(intake_context.canonical_task_id)
        if elicitation_config is not None:
            task_voice_settings.pronunciation_map = build_pronunciation_map(
                sample_run_complication(elicitation_config, task),
                selected=selected_stimulus,
            )
        if selected_stimulus is not None:
            task_voice_settings.pronunciation_grapheme_tokens = [
                p.token
                for p in selected_stimulus.pronunciations
                if p.boundary == "grapheme"
                and p.token in task_voice_settings.pronunciation_map
            ]
        if agent_directed:
            from tau2.domains.intake.spell_events import build_spell_event_detector

            if locale_data is not None:
                from tau2.domains.intake.spell_events import (
                    build_native_spell_event_detector,
                )

                detector = build_native_spell_event_detector(
                    task, locale_data.digit_words
                )
            else:
                detector = build_spell_event_detector(task)
            domain_kwargs["spell_event_detector"] = detector.detect

    user = VoiceStreamingUserSimulator(
        tools=user_tools,
        instructions=user_instructions,
        llm=llm,
        llm_args=llm_args,
        voice_settings=task_voice_settings,
        **domain_kwargs,
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
        backchannel=backchannel_settings(language_context),
        audio_taps_dir=audio_taps_dir,
        realtime_generation=audio_native_config.realtime_generation_enabled,
    )
    return user


# =============================================================================
# High-level build functions (use RunConfig)
# =============================================================================


def backchannel_settings(context: LanguageTaskInfo) -> BackchannelSettings:
    """The call's backchannel density and continuers from its language pack."""
    if context.persona is not None:
        return BackchannelSettings(
            level=context.backchannel_level,
            language_name=context.persona.display_language,
            phrases=context.persona.voice.backchannel_phrases,
        )
    # Domain-owned callers use the pack's canonical persona continuers.
    return default_backchannel_settings(
        context.language, level=context.backchannel_level
    )


def call_context(config: RunConfig, language_context: LanguageTaskInfo) -> CallContext:
    """The typed per-call context handed to the orchestrator."""
    from tau2.voice.voice_gender import selected_agent_voice

    agent_voice, agent_voice_gender = selected_agent_voice(config)
    return CallContext(
        language=language_context,
        communicate_scoring=config.communicate_scoring,
        agent_voice=agent_voice,
        agent_voice_gender=agent_voice_gender,
    )


def _derive_read_log_allowlist(task: Task) -> set:
    """Set of discoverable-tool names required by the task's golden trajectory.

    The banking_knowledge ``call_discoverable_agent_tool`` wrapper logs every
    call to the ``agent_discoverable_tools`` DB table, which is then hashed
    for the env-eval reward. For READ-only discoverable tools this means any
    extra validation call (e.g. an agent reading a balance "just in case")
    diverges the hash and zeros the reward — even though the KB explicitly
    encourages such reads.

    To keep the "agent must call this read to verify" assertion (tasks 046,
    085, etc.) while not punishing extra reads, we extract the set of tool
    names appearing in golden ``call_discoverable_agent_tool`` actions and
    pass it as an allowlist to the toolkit. Calls to writes are always
    logged; calls to reads are only logged when in this set.
    """
    allowlist: set = set()
    if task.evaluation_criteria is None:
        return allowlist
    for action in task.evaluation_criteria.actions or []:
        if action.name == "call_discoverable_agent_tool":
            name = (action.arguments or {}).get("agent_tool_name")
            if name:
                allowlist.add(name)
    return allowlist


def _build_env_kwargs(
    config: RunConfig, task: Task, *, language_context: LanguageTaskInfo
) -> dict:
    """Build env_kwargs from a RunConfig for the environment constructor.

    Passes channel and task context for elicitation's pre-call work order, and
    retrieval settings and task context for banking knowledge policies.
    """
    from tau2.multilingual.runtime import (
        language_environment_kwargs,
        pin_language_artifacts,
    )

    loaded = pin_language_artifacts(config, update=False)
    if loaded.artifact_pins != language_context.artifact_pins:
        raise ValueError("Environment binding differs from the resolved task context")
    env_kwargs: dict = language_environment_kwargs(loaded)
    if config.domain in ELICITATION_DOMAINS:
        env_kwargs["task"] = task
        env_kwargs["strategy"] = config.elicitation.strategy
        env_kwargs["channel"] = (
            "voice" if isinstance(config, VoiceRunConfig) else "text"
        )
    retrieval_config = getattr(config, "retrieval_config", None)
    if retrieval_config is not None:
        env_kwargs["retrieval_variant"] = retrieval_config
        env_kwargs["task"] = task
        rk = dict(getattr(config, "retrieval_config_kwargs", None) or {})
        if rk:
            env_kwargs["retrieval_kwargs"] = rk
    if getattr(config, "domain", None) == "banking_knowledge":
        env_kwargs["read_log_allowlist"] = _derive_read_log_allowlist(task)
    return env_kwargs


def build_text_orchestrator(
    config: TextRunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
    user_persona_config: Optional[PersonaConfig] = None,
    language_context: Optional[LanguageTaskInfo] = None,
) -> Orchestrator:
    """Build a half-duplex (text) orchestrator from a TextRunConfig.

    Args:
        config: Text run configuration.
        task: The task to run.
        seed: Per-trial seed. If None, uses config.seed.
        simulation_id: Unique simulation ID. If None, a UUID is generated.
        user_persona_config: Persona config for the user simulator.
        language_context: Prepared task assignment; current artifact bytes are checked.

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
    from tau2.multilingual.runtime import agent_language_guidance, task_context

    language_context = task_context(config, task, context=language_context)
    context = call_context(config, language_context)
    if language_context.persona is not None:
        user_persona_config = language_context.persona
    domain = config.domain
    env_kwargs = _build_env_kwargs(config, task, language_context=language_context)

    environment = build_environment(domain, solo_mode=solo_mode, env_kwargs=env_kwargs)

    agent = build_agent(
        config.effective_agent,
        environment,
        language_guidance=agent_language_guidance(
            language_context, agent_voice_gender=context.agent_voice_gender
        ),
        language=language_context.language,
        llm=config.llm_agent,
        llm_args=config.llm_args_agent,
        task=task,
        solo_mode=solo_mode,
        elicitation_config=config.elicitation,
    )

    user = build_user(
        config.effective_user,
        environment,
        user_prompt_task(config, task),
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
        call_context=context,
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
    language_context: Optional[LanguageTaskInfo] = None,
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
        language_context: Prepared task assignment; current artifact bytes are checked.
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

    from tau2.multilingual.runtime import agent_language_guidance, task_context

    language_context = task_context(config, task, context=language_context)
    context = call_context(config, language_context)
    domain = config.domain
    env_kwargs = _build_env_kwargs(config, task, language_context=language_context)

    environment = build_environment(domain, env_kwargs=env_kwargs)

    agent = build_agent(
        config.effective_agent,
        environment,
        audio_native_config=config.audio_native_config,
        language_guidance=agent_language_guidance(
            language_context, agent_voice_gender=context.agent_voice_gender
        ),
        language=language_context.language,
        audio_taps_dir=audio_taps_dir,
        elicitation_config=config.elicitation,
    )

    user = build_voice_user(
        environment,
        user_prompt_task(config, task),
        config.audio_native_config,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        voice_settings=user_voice_settings,
        persona_config=user_persona_config,
        speech_complexity=config.speech_complexity,
        seed=(
            config.voice_sampling_seed
            if config.voice_sampling_seed is not None
            else seed
            if seed is not None
            else 42
        ),
        domain=domain,
        language_context=language_context,
        hallucination_feedback=hallucination_feedback,
        audio_taps_dir=audio_taps_dir,
        elicitation_config=config,
        channel_effects_mode=config.channel_effects_mode,
        speech_effects_mode=config.speech_effects_mode,
        force_sample_voice_config=config.pass3_condition is not None,
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
        call_context=context,
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
    language_context: Optional[LanguageTaskInfo] = None,
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
        language_context: Prepared language/domain/task/persona resolution.
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
            language_context=language_context,
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
            language_context=language_context,
            user_persona_config=user_persona_config,
        )

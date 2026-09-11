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
from tau2.user_simulation_voice_presets import (
    derive_task_seed,
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
    language: Optional[str] = None,
    locale: Optional[str] = None,
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
        language: ISO 639-1 code of the run's active language-pack persona
            (see tau2.multilingual). None means English.
        locale: ISO 3166-2 locale of the resolved caller persona. None omits
            caller-region context from the agent prompt.

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

    # Native-script DB ablation: a task from a ``*_identity_native`` set runs
    # against a DB whose name fields keep native script, and the agent prompt
    # must say so (the pack's agent_native_script_db_clause). Derived from the
    # task id — the id suffix IS the variant contract — so every caller of
    # build_agent gets it without threading a flag.
    from tau2.multilingual.native_script import is_native_identity_task_id

    native_script_db = bool(task is not None and is_native_identity_task_id(task.id))

    return agent_factory(
        tools=tools,
        domain_policy=environment.get_policy(),
        llm=llm,
        llm_args=llm_args,
        task=task,
        audio_native_config=audio_native_config,
        audio_taps_dir=audio_taps_dir,
        language=language,
        locale=locale,
        native_script_db=native_script_db,
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
    speech_environment=None,
    input_style_directive: Optional[str] = None,
    entity_noise=None,
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
        input_style_directive: Pre-rendered text input-style block (resolved
            and validated against the run's language pack by the caller).
        entity_noise: The task's deterministic entity noise plan
            (``TextNoiseInfo``), built by the caller on noise-armed runs.

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
        # The run's benchmark domain selects which of a language pack's
        # per-domain glossaries renders into the localization section.
        user_kwargs["domain"] = environment.get_domain_name()
        user_kwargs["input_style_directive"] = input_style_directive
        user_kwargs["entity_noise"] = entity_noise

    user = UserConstructor(**user_kwargs)
    # Record the active language/persona on the user so the completed
    # SimulationRun carries it (the signal nativeness scoring reads). None for
    # English/non-multilingual runs.
    if speech_environment is not None and hasattr(user, "speech_environment"):
        user.speech_environment = speech_environment
    return user


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
    persona_seed: Optional[int] = None,
    domain: Optional[str] = None,
    hallucination_feedback: Optional[str] = None,
    audio_taps_dir: Optional[Path] = None,
    user_persona_id: Optional[str] = None,
    channel_effects_mode: str = "regular",
    speech_effects_mode: str = "regular",
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
        seed: Per-trial seed for voice config sampling. Per-task seed is
            derived via ``derive_task_seed(seed, task.id)`` (stable digest,
            not salted ``hash()``). Hallucination retries bump this for
            diversity, but persona assignment must stay stable.
        persona_seed: Seed used exclusively for bare-language-code persona
            rotation. Must be retry-invariant (typically the batch-level
            config.seed). Defaults to ``seed`` when None.
        domain: Domain name (used for loading pre-sampled voice configs).
            If None, extracted from environment.
        hallucination_feedback: Optional feedback from a previous hallucination
            check. If provided, appended to user instructions to help avoid
            repeating the same errors on retry.
        user_persona_id: Optional persona override (e.g. a language-pack
            persona id like 'priya_hindi_v1'). Bypasses pre-sampled persona
            selection; see tau2.multilingual.

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
    task_seed = derive_task_seed(seed, task.id)
    sampled_voice_config = get_or_load_task_voice_config(
        domain=domain,
        task_id=task.id,
        task_seed=task_seed,
        complexity=speech_complexity,
        synthesis_config=task_voice_settings.synthesis_config,
        persona_name=user_persona_id,
        # The run-level seed keys the bare-language-code persona rotation:
        # constant across the run's tasks, so the per-persona split stays
        # balanced (task_seed varies per task and would break it). Use
        # persona_seed (retry-invariant) so hallucination retries with a
        # bumped seed don't silently swap the pack persona.
        run_seed=persona_seed if persona_seed is not None else seed,
        channel_effects_mode=channel_effects_mode,
        speech_effects_mode=speech_effects_mode,
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

    # NOTE: the transcription language is defaulted from the active
    # language-pack persona at transcription time in
    # ``VoiceAgent.transcribe_voice`` (via a non-mutating model_copy), so we do
    # not mutate the shared transcription_config here.

    # Use provided persona config or fall back to sampled config
    if persona_config is None:
        persona_config = sampled_voice_config.persona_config

    user_instructions = str(task.user_scenario)
    if hallucination_feedback:
        user_instructions += f"\n\n{hallucination_feedback}"

    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

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
        use_llm_backchannel=sampled_voice_config.use_llm_backchannel,
        interruption_check_interval=audio_native_config.interruption_check_interval_ticks,
        integration_ticks=audio_native_config.integration_ticks,
        silence_annotation_threshold_ticks=audio_native_config.silence_annotation_threshold_ticks,
        tick_duration_seconds=audio_native_config.tick_duration_seconds,
        persona_config=persona_config,
        audio_taps_dir=audio_taps_dir,
        realtime_generation=audio_native_config.realtime_generation_enabled,
        domain=domain,
    )


# =============================================================================
# High-level build functions (use RunConfig)
# =============================================================================


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
        rk = dict(getattr(config, "retrieval_config_kwargs", None) or {})
        if rk:
            env_kwargs["retrieval_kwargs"] = rk
    if getattr(config, "domain", None) == "banking_knowledge":
        env_kwargs["read_log_allowlist"] = _derive_read_log_allowlist(task)
    return env_kwargs


def user_prompt_task(
    config: RunConfig, task: Task, user_language: Optional[str]
) -> Task:
    """The task variant the USER SIMULATOR is built from.

    Public seam: the prompt-bed review packet renders the user-sim prompt
    through this exact function, so what reviewers read is what runs.

    A multilingual run's localized task is swapped for its
    English-instructions variant — entity localization preserved — via
    ``tau2.multilingual.english_prompts``. The orchestrator, environment, and
    evaluator keep the ORIGINAL task (id, initial_state, evaluation_criteria
    untouched); only what the user simulator reads changes. English runs (no
    ``user_language``) pass the task through unchanged.

    """
    if user_language is not None:
        from tau2.multilingual.english_prompts import english_user_task_variant

        task = english_user_task_variant(
            task, domain=config.domain, task_set_name=config.task_set_name
        )
        # Native-script identity tasks (the romanization ablation) additionally
        # get the fixed, versioned name spell-out block: the caller's
        # per-identity payload from the committed native identity map, so the
        # sim dictates its name in native script (zh: character glosses; hi:
        # akshara-by-akshara), never in Latin letters. Appended on a COPY, the
        # caller's task object stays untouched — and because the review packet
        # renders through this same function, reviewers read exactly this block.
        from tau2.multilingual.native_script import (
            is_native_identity_task_id,
            native_spellout_block_for_task,
        )

        if is_native_identity_task_id(task.id):
            block = native_spellout_block_for_task(
                task.model_dump(), user_language, config.domain
            )
            task = task.model_copy(deep=True)
            instructions = task.user_scenario.instructions
            if isinstance(instructions, str):
                task.user_scenario.instructions = f"{instructions}\n\n{block}"
            else:
                instructions.task_instructions = (
                    f"{instructions.task_instructions or ''}\n\n{block}".strip()
                )
    return task


def _localized_first_agent_message(language: Optional[str]):
    """The localized agent opening greeting for a language-pack run, or None.

    Returns an ``AssistantMessage`` carrying the pack's ``agent_greeting`` so the
    text agent opens in the target language instead of the hardcoded English
    ``DEFAULT_FIRST_AGENT_MESSAGE``. None (no language, no pack, or no greeting
    authored) keeps the English default — English/unlocalized runs unchanged.
    """
    if language is None:
        return None
    from tau2.data_model.message import AssistantMessage
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack(language)
    if pack is None or not pack.agent_greeting:
        return None
    return AssistantMessage(role="assistant", content=pack.agent_greeting, cost=0.0)


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

    # Resolve the run's language-pack persona for text runs (mirrors
    # build_voice_orchestrator). The override (config.user_persona_id) may be a
    # pack persona id or a bare language code (per-task, balanced round-robin
    # keyed on the run seed). A resolved language-pack persona supplies the
    # localized text guidelines + pragmatics (via persona_config) and the
    # localized agent (via language); it also records a SpeechEnvironment on the
    # user so the run reports its language for nativeness scoring. None (no
    # override, or a plain English persona) means English — unchanged behavior.
    user_language = None
    user_locale = None
    resolved_persona_config = user_persona_config
    speech_environment = None
    if config.user_persona_id is not None:
        from tau2.data_model.voice import SpeechEnvironment
        from tau2.multilingual.registry import (
            get_multilingual_persona,
            resolve_run_language,
            resolve_task_persona,
        )

        resolved_persona_id = resolve_task_persona(
            config.user_persona_id,
            task_id=task.id,
            run_seed=config.seed,
            domain=domain,
        )
        user_language = resolve_run_language(resolved_persona_id)
        hit = get_multilingual_persona(resolved_persona_id)
        if hit is not None:
            _pack, persona = hit
            # The language-pack persona config wins over any run-level persona
            # (it carries the localized guidelines + author-owned verbosity).
            resolved_persona_config = persona
            user_locale = persona.locale
            speech_environment = SpeechEnvironment(
                persona_name=persona.persona_id,
                language=user_language,
                locale=persona.locale,
                persona_id=persona.persona_id,
                persona_tags=dict(persona.tags),
            )

    # Resolve the text input-style arm (romanized / diacritic-free /
    # code-mixed typing). Validated HERE so an unsupported style fails the
    # build loudly — never a silent fallback to native_script. None (the
    # default arm) renders nothing.
    input_style_directive = None
    if config.text_input_style is not None:
        from tau2.multilingual.registry import resolve_text_input_style

        input_style_directive = resolve_text_input_style(
            user_language, config.text_input_style
        )

    # The task variant the USER SIMULATOR reads (English-instruction swap for
    # localized sets); also what the entity noise plan pins on — its entities
    # are what the sim will actually type.
    user_task = user_prompt_task(config, task, user_language)

    # Build the deterministic entity noise plan for noise-armed runs. The
    # plan derives only from (catalog version, seed, task id, entity), so it
    # is identical for every system under test (paired design). An unmapped
    # language raises here — never a silent clean run in a noise arm.
    entity_noise = None
    if config.text_noise is not None:
        from tau2.multilingual.text_noise import plan_task_noise

        entity_noise = plan_task_noise(
            user_task,
            user_language or "en",
            config.text_noise,
            domain,
        ).info

    agent = build_agent(
        config.effective_agent,
        environment,
        llm=config.llm_agent,
        llm_args=config.llm_args_agent,
        task=task,
        solo_mode=solo_mode,
        language=user_language,
        locale=user_locale,
    )

    user = build_user(
        config.effective_user,
        environment,
        user_task,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        persona_config=resolved_persona_config,
        solo_mode=solo_mode,
        speech_environment=speech_environment,
        input_style_directive=input_style_directive,
        entity_noise=entity_noise,
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
        first_agent_message=_localized_first_agent_message(user_language),
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

    # Resolve the run's language from the persona override. The agent is built
    # before the voice user (whose sampled SpeechEnvironment carries the
    # language), so resolve from config.user_persona_id via the same
    # language-pack lookup used by SampledVoiceConfig.to_speech_environment.
    # The override may be a pack persona id or a bare language code (per-task
    # persona sampling). None (no override, or a plain English persona) means
    # English.
    user_language = None
    user_locale = None
    if config.user_persona_id is not None:
        from tau2.multilingual.registry import (
            get_multilingual_persona,
            resolve_run_language,
            resolve_task_persona,
        )

        resolved_persona_id = resolve_task_persona(
            config.user_persona_id,
            task_id=task.id,
            run_seed=config.seed,
            domain=domain,
        )
        user_language = resolve_run_language(resolved_persona_id)
        hit = get_multilingual_persona(resolved_persona_id)
        if hit is not None:
            _pack, persona = hit
            user_locale = persona.locale

    agent = build_agent(
        config.effective_agent,
        environment,
        task=task,
        audio_native_config=config.audio_native_config,
        audio_taps_dir=audio_taps_dir,
        language=user_language,
        locale=user_locale,
    )

    user = build_voice_user(
        environment,
        user_prompt_task(config, task, user_language),
        config.audio_native_config,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        voice_settings=user_voice_settings,
        persona_config=user_persona_config,
        speech_complexity=config.speech_complexity,
        seed=42 if seed is None else seed,
        # persona_seed is the batch-level config.seed so hallucination retries
        # (which bump ``seed`` for diversity) don't silently swap the persona.
        persona_seed=config.seed,
        domain=domain,
        hallucination_feedback=hallucination_feedback,
        audio_taps_dir=audio_taps_dir,
        user_persona_id=config.user_persona_id,
        channel_effects_mode=config.channel_effects_mode,
        speech_effects_mode=config.speech_effects_mode,
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

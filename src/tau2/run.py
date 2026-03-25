"""
tau2.run -- Entry point for running simulations.

Thin facade that delegates to the tau2.runner package. All simulation
logic lives in the runner's layered architecture:

    Layer 1 (simulation.py):  run_simulation()
    Layer 2 (build.py):       build_* functions
    Layer 3 (batch.py):       run_domain, run_tasks, run_single_task
    Helpers (helpers.py):     get_tasks, get_options, etc.

Usage:
    # High-level: run all tasks in a domain
    from tau2.run import run_domain
    from tau2.data_model.simulation import TextRunConfig
    results = run_domain(TextRunConfig(domain="retail", agent="llm_agent", ...))

    # Mid-level: run a single task
    from tau2.run import get_tasks, run_single_task
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    result = run_single_task(config, tasks[0], seed=42)

    # Low-level: build and run manually
    from tau2.run import build_orchestrator, run_simulation
    orch = build_orchestrator(config, task, seed=42)
    sim_run = run_simulation(orch)
"""

import warnings
from pathlib import Path
from typing import Optional

from tau2.data_model.persona import PersonaConfig
from tau2.data_model.simulation import (
    AudioNativeConfig,
    Results,
    RunConfig,
    SimulationRun,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.data_model.voice import SpeechComplexity, VoiceSettings
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner import (
    build_agent,
    build_environment,
    build_orchestrator,
    build_text_orchestrator,
    build_user,
    build_voice_orchestrator,
    build_voice_user,
    get_environment_info,
    get_info,
    get_options,
    get_tasks,
    load_task_splits,
    load_tasks,
    make_run_name,
    run_domain,
    run_simulation,
    run_single_task,
)
from tau2.runner.batch import run_tasks as _run_tasks

# =============================================================================
# Deprecated shims -- these preserve the old flat-argument signatures from
# tau2.run so existing callers keep working. New code should use
# run_single_task(config, task, ...) and run_tasks(config, tasks, ...)
# =============================================================================


def run_task(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: Optional[int] = None,
    save_dir: Optional[Path] = None,
    enforce_communication_protocol: bool = False,
    speech_complexity: SpeechComplexity = "regular",
    audio_native_config: Optional[AudioNativeConfig] = None,
    user_voice_settings: Optional[VoiceSettings] = None,
    user_persona_config: Optional[PersonaConfig] = None,
    verbose_logs: bool = False,
    audio_debug: bool = False,
    audio_taps: bool = False,
    auto_review: bool = False,
    review_mode: str = "full",
    solo_mode: bool = False,
    hallucination_feedback: Optional[str] = None,
    retrieval_config: Optional[str] = None,
    retrieval_config_kwargs: Optional[dict] = None,
) -> SimulationRun:
    """Deprecated: use run_single_task(config, task, ...) instead."""
    warnings.warn(
        "run_task() is deprecated. Use run_single_task(TextRunConfig(...), task, ...) "
        "or run_single_task(VoiceRunConfig(...), task, ...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if audio_native_config is not None:
        config = VoiceRunConfig(
            domain=domain,
            audio_native_config=audio_native_config,
            llm_user=llm_user or "gpt-4.1",
            llm_args_user=llm_args_user or {},
            max_errors=max_errors,
            seed=seed,
            speech_complexity=speech_complexity,
            audio_debug=audio_debug,
            audio_taps=audio_taps,
            auto_review=auto_review,
            review_mode=review_mode,
            verbose_logs=verbose_logs,
            retrieval_config=retrieval_config,
            retrieval_config_kwargs=retrieval_config_kwargs,
        )
    else:
        config = TextRunConfig(
            domain=domain,
            agent=agent,
            user=user,
            llm_agent=llm_agent or "gpt-4.1",
            llm_args_agent=llm_args_agent or {},
            llm_user=llm_user or "gpt-4.1",
            llm_args_user=llm_args_user or {},
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            enforce_communication_protocol=enforce_communication_protocol,
            auto_review=auto_review,
            review_mode=review_mode,
            verbose_logs=verbose_logs,
            retrieval_config=retrieval_config,
            retrieval_config_kwargs=retrieval_config_kwargs,
        )
    return run_single_task(
        config,
        task,
        seed=seed,
        evaluation_type=evaluation_type,
        save_dir=save_dir,
        user_voice_settings=user_voice_settings,
        user_persona_config=user_persona_config,
        verbose_logs=verbose_logs,
        audio_debug=audio_debug,
        audio_taps=audio_taps,
        auto_review=auto_review,
        review_mode=review_mode,
        hallucination_feedback=hallucination_feedback,
    )


def run_tasks(
    domain: str,
    tasks: list[Task],
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    save_to: Optional[str | Path] = None,
    save_dir: Optional[Path] = None,
    console_display: bool = True,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    max_concurrency: int = 1,
    seed: Optional[int] = 300,
    log_level: Optional[str] = "INFO",
    enforce_communication_protocol: bool = False,
    speech_complexity: SpeechComplexity = "regular",
    audio_native_config: Optional[AudioNativeConfig] = None,
    verbose_logs: bool = False,
    audio_debug: bool = False,
    audio_taps: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    auto_resume: bool = False,
    auto_review: bool = False,
    review_mode: str = "full",
    solo_mode: bool = False,
    hallucination_retries: int = 0,
    retrieval_config: Optional[str] = None,
    retrieval_config_kwargs: Optional[dict] = None,
) -> Results:
    """Deprecated: use runner.run_tasks(config, tasks, ...) instead."""
    warnings.warn(
        "run_tasks() with flat arguments is deprecated. Use "
        "runner.run_tasks(TextRunConfig(...), tasks, ...) or "
        "runner.run_tasks(VoiceRunConfig(...), tasks, ...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if audio_native_config is not None:
        config = VoiceRunConfig(
            domain=domain,
            audio_native_config=audio_native_config,
            llm_user=llm_user or "gpt-4.1",
            llm_args_user=llm_args_user or {},
            num_trials=num_trials,
            max_errors=max_errors,
            max_concurrency=max_concurrency,
            seed=seed,
            log_level=log_level,
            speech_complexity=speech_complexity,
            audio_debug=audio_debug,
            audio_taps=audio_taps,
            max_retries=max_retries,
            retry_delay=retry_delay,
            auto_resume=auto_resume,
            auto_review=auto_review,
            review_mode=review_mode,
            hallucination_retries=hallucination_retries,
            verbose_logs=verbose_logs,
            retrieval_config=retrieval_config,
            retrieval_config_kwargs=retrieval_config_kwargs,
        )
    else:
        config = TextRunConfig(
            domain=domain,
            agent=agent,
            user=user,
            llm_agent=llm_agent or "gpt-4.1",
            llm_args_agent=llm_args_agent or {},
            llm_user=llm_user or "gpt-4.1",
            llm_args_user=llm_args_user or {},
            num_trials=num_trials,
            max_steps=max_steps,
            max_errors=max_errors,
            max_concurrency=max_concurrency,
            seed=seed,
            log_level=log_level,
            enforce_communication_protocol=enforce_communication_protocol,
            max_retries=max_retries,
            retry_delay=retry_delay,
            auto_resume=auto_resume,
            auto_review=auto_review,
            review_mode=review_mode,
            hallucination_retries=hallucination_retries,
            verbose_logs=verbose_logs,
            retrieval_config=retrieval_config,
            retrieval_config_kwargs=retrieval_config_kwargs,
        )

    save_path = Path(save_to) if save_to else None

    return _run_tasks(
        config,
        tasks,
        save_path=save_path,
        save_dir=save_dir,
        evaluation_type=evaluation_type,
        console_display=console_display,
    )


__all__ = [
    # Layer 1: Execute
    "run_simulation",
    # Layer 2: Build
    "build_environment",
    "build_agent",
    "build_user",
    "build_voice_user",
    "build_orchestrator",
    "build_text_orchestrator",
    "build_voice_orchestrator",
    # Layer 3: Batch
    "run_domain",
    "run_tasks",
    "run_single_task",
    # Deprecated shims
    "run_task",
    # Helpers
    "get_options",
    "get_environment_info",
    "load_task_splits",
    "load_tasks",
    "get_tasks",
    "make_run_name",
    "get_info",
    # Re-exports
    "RunConfig",
    "TextRunConfig",
    "VoiceRunConfig",
    "EvaluationType",
]

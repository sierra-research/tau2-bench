"""
Layer 3: Batch runner.

Orchestrates batch execution with concurrency, checkpointing, retries,
logging, and optional side effects (auto-review, audio saving).

Uses Layer 2 (build) to construct instances and Layer 1 (simulation) to
execute them.
"""

import asyncio
import asyncio.base_events
import json
import multiprocessing
import os
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.data_model.persona import InterruptTendency, PersonaConfig, Verbosity
from tau2.data_model.simulation import (
    AudioNativeConfig,
    Results,
    RunConfig,
    SimulationRun,
    TextRunConfig,
    UserInfo,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.data_model.voice import SynthesisConfig, VoiceSettings
from tau2.evaluator.evaluator import EvaluationType
from tau2.evaluator.reviewer import check_hallucination, format_hallucination_feedback
from tau2.metrics.agent_metrics import compute_metrics
from tau2.registry import registry
from tau2.runner.build import _build_env_kwargs, build_orchestrator
from tau2.runner.checkpoint import (
    create_checkpoint_fns,
    try_resume,
)
from tau2.runner.helpers import get_info, get_tasks, make_run_name
from tau2.runner.progress import StatusMonitor, run_with_retry
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator import (
    get_global_user_sim_guidelines,
    get_global_user_sim_guidelines_voice,
)
from tau2.user_simulation_voice_presets import COMPLEXITY_CONFIGS
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.llm_utils import llm_log_mode, set_llm_log_dir, set_llm_log_mode
from tau2.utils.utils import DATA_DIR
from tau2.voice.synthesis.conversation_builder import generate_simulation_audio
from tau2.voice.utils.audio_debug import generate_audio_debug_info

# Context variable to track current simulation_id for log filtering
# This ensures task-specific log handlers only receive their own messages
_current_simulation_id: ContextVar[Optional[str]] = ContextVar(
    "_current_simulation_id", default=None
)


# =============================================================================
# Asyncio event loop management for worker threads
# =============================================================================

_original_del = asyncio.base_events.BaseEventLoop.__del__


def _patched_del(self):
    try:
        _original_del(self)
    except AttributeError:
        pass


asyncio.base_events.BaseEventLoop.__del__ = _patched_del


def _close_event_loop_safely(loop):
    if loop is None or loop.is_closed():
        return
    try:
        if hasattr(loop, "_ssock") and loop._ssock is not None:
            loop.close()
        elif hasattr(loop, "_closed") and not loop._closed:
            loop._closed = True
            if hasattr(loop, "_selector") and loop._selector is not None:
                loop._selector.close()
                loop._selector = None
    except (AttributeError, OSError):
        pass


def _init_thread_event_loop():
    try:
        old_loop = asyncio.get_event_loop_policy().get_event_loop()
        _close_event_loop_safely(old_loop)
    except RuntimeError:
        pass

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception:
        pass


def _cleanup_thread_event_loop():
    """Close the thread-local event loop so it doesn't leak into GC."""
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        _close_event_loop_safely(loop)
    except RuntimeError:
        pass
    try:
        asyncio.set_event_loop(None)
    except Exception:
        pass


# =============================================================================
# Side-effect helpers
# =============================================================================


def run_auto_review(
    simulation: SimulationRun,
    task: Task,
    review_mode: str,
    user: str,
    llm_user: Optional[str],
    llm_args_user: Optional[dict],
    user_persona_config: Optional[PersonaConfig],
    user_voice_settings: Optional[VoiceSettings],
    policy: str,
    is_audio_native: bool,
) -> None:
    """Run LLM conversation review on a simulation and attach results.

    Args:
        simulation: The completed simulation to review.
        task: The task specification.
        review_mode: "full" (agent+user) or "user" (user only).
        user: User implementation name.
        llm_user: LLM used by user simulator.
        llm_args_user: LLM args for user simulator.
        user_persona_config: Persona config for user.
        user_voice_settings: Voice settings for user.
        policy: Environment policy string.
        is_audio_native: Whether audio-native mode was used.
    """
    from tau2.evaluator.reviewer import ReviewMode, review_simulation

    review_mode_enum = ReviewMode.FULL if review_mode == "full" else ReviewMode.USER

    if is_audio_native:
        review_guidelines = get_global_user_sim_guidelines_voice()
    else:
        review_guidelines = get_global_user_sim_guidelines()

    review_user_info = UserInfo(
        implementation=user,
        llm=llm_user,
        llm_args=llm_args_user,
        global_simulation_guidelines=review_guidelines,
        persona_config=user_persona_config,
        voice_settings=user_voice_settings,
    )

    logger.info(f"Starting review for task {task.id} (mode: {review_mode})...")

    review_result, auth_result = review_simulation(
        simulation=simulation,
        task=task,
        mode=review_mode_enum,
        user_info=review_user_info,
        policy=policy,
        interruption_enabled=is_audio_native,
    )

    if review_mode == "full":
        simulation.review = review_result
        simulation.auth_classification = auth_result
    else:
        simulation.user_only_review = review_result

    logger.info(
        f"Review completed for task {task.id}: has_errors={review_result.has_errors}"
    )


def save_simulation_audio(
    simulation: SimulationRun,
    task: Task,
    simulation_id: str,
    save_dir: Path,
    audio_native_config: AudioNativeConfig,
    audio_debug: bool = False,
) -> None:
    """Save audio files for an audio-native simulation.

    Args:
        simulation: The completed simulation.
        task: The task specification.
        simulation_id: Unique simulation ID.
        save_dir: Base directory for saving files.
        audio_native_config: Audio-native configuration.
        audio_debug: Whether to generate debug audio analysis.
    """
    task_audio_dir = (
        save_dir / "artifacts" / f"task_{task.id}" / f"sim_{simulation_id}" / "audio"
    )
    task_audio_dir.mkdir(parents=True, exist_ok=True)

    if audio_debug:
        try:
            debug_dir = task_audio_dir / "debug"
            report = generate_audio_debug_info(
                simulation,
                debug_dir,
                save_per_tick_audio_files=True,
                save_silence=True,
                tick_duration_ms=audio_native_config.tick_duration_ms,
            )
            logger.info(
                f"Audio debug info saved to: {debug_dir} "
                f"(agent: {report.agent_ticks_with_audio}, user: {report.user_ticks_with_audio} ticks)"
            )
            if report.warnings:
                logger.warning(
                    f"Audio analysis found {len(report.warnings)} warning(s)"
                )
        except Exception as e:
            logger.warning(f"Failed to generate audio debug info: {e}")

    try:
        generate_simulation_audio(simulation, task_audio_dir)
        logger.debug(f"Audio saved to: {task_audio_dir}")
    except Exception as e:
        logger.warning(f"Failed to save audio for task {task.id}: {e}")


# =============================================================================
# Per-task logging context manager
# =============================================================================


class _TaskLogContext:
    """Manages per-task log files and LLM debug logging."""

    def __init__(
        self,
        simulation_id: str,
        save_dir: Optional[Path],
        task: Task,
        verbose_logs: bool,
    ):
        self.simulation_id = simulation_id
        self.save_dir = save_dir
        self.task = task
        self.verbose_logs = verbose_logs
        self.task_log_dir: Optional[Path] = None
        self._handler_id = None

    def __enter__(self):
        if self.save_dir:
            self.task_log_dir = (
                self.save_dir
                / "artifacts"
                / f"task_{self.task.id}"
                / f"sim_{self.simulation_id}"
            )

        if self.verbose_logs and self.task_log_dir:
            self.task_log_dir.mkdir(parents=True, exist_ok=True)
            _current_simulation_id.set(self.simulation_id)

            def make_simulation_filter(sim_id: str):
                def simulation_filter(record):
                    return _current_simulation_id.get() == sim_id

                return simulation_filter

            log_file_path = self.task_log_dir / "task.log"
            self._handler_id = logger.add(
                log_file_path,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation=None,
                enqueue=True,
                filter=make_simulation_filter(self.simulation_id),
            )
            logger.debug(f"Task log file: {log_file_path}")

        if self.task_log_dir and self.verbose_logs:
            llm_log_dir = self.task_log_dir / "llm_debug"
            set_llm_log_dir(llm_log_dir)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.task_log_dir and self.task_log_dir.exists():
            status = {
                "status": "failed",
                "reason": "infrastructure_error",
                "error": str(exc_val),
                "error_type": exc_type.__name__,
            }
            try:
                status_path = self.task_log_dir / "sim_status.json"
                with open(status_path, "w") as f:
                    json.dump(status, f, indent=2)
            except Exception:
                pass

        if self.save_dir:
            set_llm_log_dir(None)
        if self._handler_id is not None:
            logger.remove(self._handler_id)
            _current_simulation_id.set(None)
        return False


# =============================================================================
# Single task runner (Layer 3 wrapper over Layers 1+2)
# =============================================================================


def run_single_task(
    config: RunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    save_dir: Optional[Path] = None,
    user_voice_settings: Optional[VoiceSettings] = None,
    user_persona_config: Optional[PersonaConfig] = None,
    verbose_logs: bool = False,
    audio_debug: bool = False,
    audio_taps: bool = False,
    auto_review: bool = False,
    review_mode: str = "full",
    hallucination_feedback: Optional[str] = None,
) -> SimulationRun:
    """Run a single task simulation with logging and optional side effects.

    This is the Layer 3 per-task function. It:
    1. Sets up per-task logging.
    2. Builds an orchestrator via Layer 2 (build_orchestrator).
    3. Runs the simulation via Layer 1 (run_simulation).
    4. Optionally runs auto-review and saves audio.
    5. Cleans up logging.

    Args:
        config: The run configuration.
        task: The task to run.
        seed: Random seed for this trial.
        evaluation_type: Evaluation type to use.
        save_dir: Directory for saving logs and audio.
        user_voice_settings: Pre-computed voice settings (run-level).
        user_persona_config: Pre-computed persona config (run-level).
        verbose_logs: Enable per-task log files.
        audio_debug: Enable audio debug analysis.
        auto_review: Run LLM conversation review after simulation.
        review_mode: Review mode ("full" or "user").

    Returns:
        The completed SimulationRun with reward_info attached.
    """
    simulation_id = str(uuid.uuid4())
    is_voice = isinstance(config, VoiceRunConfig)

    logger.info(
        f"STARTING SIMULATION: Domain: {config.domain}, Task: {task.id}, "
        f"Agent: {config.effective_agent}, User: {config.effective_user}"
    )

    with _TaskLogContext(simulation_id, save_dir, task, verbose_logs):
        # Compute audio taps directory if enabled
        taps_dir = None
        if audio_taps and save_dir:
            taps_dir = (
                save_dir
                / "artifacts"
                / f"task_{task.id}"
                / f"sim_{simulation_id}"
                / "audio"
                / "taps"
            )

        # Layer 2: Build the orchestrator
        orchestrator = build_orchestrator(
            config,
            task,
            seed=seed,
            simulation_id=simulation_id,
            user_voice_settings=user_voice_settings,
            user_persona_config=user_persona_config,
            hallucination_feedback=hallucination_feedback,
            audio_taps_dir=taps_dir,
        )

        # Layer 1: Run the simulation
        env_kwargs = _build_env_kwargs(config, task) or None
        simulation = run_simulation(
            orchestrator, evaluation_type=evaluation_type, env_kwargs=env_kwargs
        )

        # Side effects
        if auto_review:
            run_auto_review(
                simulation=simulation,
                task=task,
                review_mode=review_mode,
                user=config.effective_user,
                llm_user=config.llm_user,
                llm_args_user=config.llm_args_user,
                user_persona_config=user_persona_config,
                user_voice_settings=user_voice_settings,
                policy=orchestrator.environment.get_policy(),
                is_audio_native=is_voice,
            )

        if is_voice and save_dir:
            save_simulation_audio(
                simulation=simulation,
                task=task,
                simulation_id=simulation_id,
                save_dir=save_dir,
                audio_native_config=config.audio_native_config,
                audio_debug=audio_debug,
            )

        logger.info(
            f"FINISHED SIMULATION: Domain: {config.domain}, Task: {task.id}, "
            f"Reward: {simulation.reward_info.reward if simulation.reward_info else 'N/A'}"
        )

        return simulation


# =============================================================================
# Batch runner
# =============================================================================


def run_tasks(
    config: RunConfig,
    tasks: list[Task],
    *,
    save_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    evaluation_type: EvaluationType = EvaluationType.ALL_WITH_NL_ASSERTIONS,
    console_display: bool = True,
    results_format: str = "json",
) -> Results:
    """Run simulations for a list of tasks with concurrency, checkpointing, and retries.

    This is the main batch execution function. It handles:
    - Seed management and trial repetition
    - Voice/persona config setup for audio-native mode
    - Checkpoint save/resume
    - Concurrent execution via thread pool
    - Progress monitoring
    - Retry on failure

    Args:
        config: Full run configuration (includes domain, agent, user, LLM settings,
            num_trials, max_concurrency, retry settings, etc.).
        tasks: The tasks to run.
        save_path: Path to the results JSON file. If None, results are not persisted.
        save_dir: Directory for saving logs, audio, etc. If None, derived from save_path.
        evaluation_type: Evaluation type to use for all simulations.
        console_display: Whether to show console output for each simulation.

    Returns:
        Results object with all simulation runs.

    Raises:
        ValueError: If no tasks are provided, or trial/step/error counts are invalid.
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    # Set log level from config
    logger.remove()
    logger.add(lambda msg: print(msg), level=config.log_level)

    if len(tasks) == 0:
        raise ValueError("No tasks to run")
    if config.num_trials <= 0:
        raise ValueError("Number of trials must be greater than 0")

    if config.effective_max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if config.max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")

    is_voice = isinstance(config, VoiceRunConfig)

    # Seed management
    random.seed(config.seed)
    seeds = [random.randint(0, 1000000) for _ in range(config.num_trials)]
    if (
        isinstance(config, TextRunConfig)
        and config.llm_args_agent
        and "seed" in config.llm_args_agent
    ):
        logger.warning("Each trial will modify the seed for the agent")
    if config.llm_args_user and "seed" in config.llm_args_user:
        logger.warning("Each trial will modify the seed for the user")

    lock = multiprocessing.Lock()

    # Create run-level voice settings and persona config for voice mode
    user_voice_settings = None
    user_persona_config = None
    if is_voice:
        user_voice_settings = VoiceSettings(
            transcription_config=None,
            synthesis_config=SynthesisConfig(),
        )
        complexity_config = COMPLEXITY_CONFIGS[config.speech_complexity]
        user_persona_config = PersonaConfig(
            verbosity=Verbosity(complexity_config["verbosity"]),
            interrupt_tendency=InterruptTendency(
                complexity_config["interrupt_tendency"]
            ),
        )

    # Warm knowledge base cache for banking_knowledge domain
    policy_override = None
    if config.domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.environment import get_knowledge_base
        from tau2.domains.banking_knowledge.retrieval import get_info_policy_override
        from tau2.knowledge.embeddings_cache import (
            get_unique_embedder_configs_for_retrieval_configs,
            warm_kb_cache,
        )

        retrieval_config = getattr(config, "retrieval_config", None)
        retrieval_config_kwargs = getattr(config, "retrieval_config_kwargs", None)
        kwargs = retrieval_config_kwargs or {}
        embedder_configs = None
        if retrieval_config:
            embedder_configs = get_unique_embedder_configs_for_retrieval_configs(
                [retrieval_config]
            )
        warm_kb_cache(embedder_configs)
        knowledge_base = get_knowledge_base()
        policy_override = get_info_policy_override(
            retrieval_config, knowledge_base, **kwargs
        )

    # Build Info and initial Results
    info = get_info(
        config,
        user_persona_config=user_persona_config,
        user_voice_settings=user_voice_settings,
        policy_override=policy_override,
    )
    simulation_results = Results(
        info=info,
        tasks=tasks,
        simulations=[],
    )

    # Checkpoint resume
    done_runs: set = set()
    if save_path is not None:
        simulation_results, done_runs, tasks = try_resume(
            save_path=save_path,
            simulation_results=simulation_results,
            tasks=tasks,
            num_trials=config.num_trials,
            auto_resume=config.auto_resume,
            results_format=results_format,
        )

    # Create checkpoint saver and replacer (shared state for dir format)
    save_fn, replace_fn = create_checkpoint_fns(save_path, lock)

    # Build argument list (skip already-completed runs)
    args = []
    for trial in range(config.num_trials):
        for i, task in enumerate(tasks):
            if (trial, task.id, seeds[trial]) in done_runs:
                console_text = Text(
                    text=f"Skipping task {task.id}, trial {trial + 1} because it has already been run.",
                    style="bold yellow",
                )
                ConsoleDisplay.console.print(console_text)
                continue
            progress_str = f"{i}/{len(tasks)} (trial {trial + 1}/{config.num_trials})"
            args.append((task, trial, seeds[trial], progress_str))

    # Status monitor
    total_count = len(tasks) * config.num_trials
    monitor = StatusMonitor(total_count, initial_completed=len(done_runs))
    monitor.set_results(simulation_results)
    monitor.start()

    # Pre-register LiveKit plugins on main thread before workers spawn
    if (
        is_voice
        and config.audio_native_config is not None
        and config.audio_native_config.provider == "livekit"
    ):
        from tau2.voice.audio_native.livekit import preregister_livekit_plugins

        preregister_livekit_plugins()

    hallucination_retries = config.hallucination_retries
    shutdown_event = threading.Event()

    # Capture ContextVar values from the main thread so worker threads
    # (which get a fresh default context) can re-apply them.
    _main_thread_llm_log_mode = llm_log_mode.get()

    def _run_tracked(
        task: Task, trial: int, seed: int, progress_str: str
    ) -> SimulationRun:
        """Run a single task with tracking, retry, and hallucination retry."""
        if shutdown_event.is_set():
            raise KeyboardInterrupt("Shutdown requested")

        _init_thread_event_loop()
        set_llm_log_mode(_main_thread_llm_log_mode)
        task_key = f"{task.id}.{trial}"
        monitor.task_started(task_key, trial)

        console_text = Text(
            text=f"{progress_str}. Running task {task.id}, trial {trial + 1}",
            style="bold green",
        )
        ConsoleDisplay.console.print(console_text)

        def _execute(
            run_seed: int = seed,
            hallucination_feedback: Optional[str] = None,
        ):
            return run_single_task(
                config,
                task,
                seed=run_seed,
                evaluation_type=evaluation_type,
                save_dir=save_dir,
                user_voice_settings=user_voice_settings,
                user_persona_config=user_persona_config,
                verbose_logs=config.verbose_logs,
                audio_debug=config.audio_debug if is_voice else False,
                audio_taps=config.audio_taps if is_voice else False,
                auto_review=config.auto_review,
                review_mode=config.review_mode,
                hallucination_feedback=hallucination_feedback,
            )

        try:
            result = run_with_retry(
                _execute,
                task=task,
                trial=trial,
                seed=seed,
                max_retries=config.max_retries,
                retry_delay=config.retry_delay,
                console_display=console_display,
                save_fn=save_fn,
                on_retry=lambda: monitor.task_restarted(task_key),
                shutdown_event=shutdown_event,
            )

            # Hallucination retry: if check detects fabricated info, re-run
            is_full_duplex = result.ticks is not None and len(result.ticks) > 0
            if hallucination_retries > 0 and is_full_duplex:
                hallucination_retry_count = 0
                while hallucination_retry_count < hallucination_retries:
                    h_check = check_hallucination(result, task)
                    result.hallucination_check = h_check

                    if not h_check.hallucination_found:
                        break

                    hallucination_retry_count += 1
                    n_errors = len(h_check.errors)

                    retry_text = Text(
                        text=f"  Hallucination detected on task {task.id} ({n_errors} instance(s)). "
                        f"Re-running with feedback ({hallucination_retry_count}/{hallucination_retries})...",
                        style="yellow",
                    )
                    ConsoleDisplay.console.print(retry_text)

                    # Save discarded run
                    if save_dir is not None:
                        discarded_dir = save_dir / "hallucination_discarded"
                        discarded_dir.mkdir(parents=True, exist_ok=True)
                        discarded_path = (
                            discarded_dir / "results_user_hallucination.json"
                        )

                        if discarded_path.exists():
                            with open(discarded_path, "r") as fp:
                                discarded_data = json.load(fp)
                            discarded_data["simulations"].append(
                                result.model_dump(mode="json")
                            )
                            existing_task_ids = {
                                t["id"] for t in discarded_data["tasks"]
                            }
                            if task.id not in existing_task_ids:
                                discarded_data["tasks"].append(
                                    task.model_dump(mode="json")
                                )
                            with open(discarded_path, "w") as fp:
                                json.dump(discarded_data, fp, indent=2)
                        else:
                            discarded_results = Results(
                                info=simulation_results.info,
                                tasks=[
                                    t
                                    for t in simulation_results.tasks
                                    if t.id == task.id
                                ],
                                simulations=[result],
                            )
                            with open(discarded_path, "w") as fp:
                                fp.write(discarded_results.model_dump_json(indent=2))

                        logger.info(
                            f"Saved discarded hallucination run to {discarded_path} "
                            f"(task {task.id}, retry {hallucination_retry_count})"
                        )

                    # Mark the discarded sim directory
                    if save_dir is not None:
                        sim_dir = (
                            save_dir
                            / "artifacts"
                            / f"task_{task.id}"
                            / f"sim_{result.id}"
                        )
                        if sim_dir.exists():
                            try:
                                status = {
                                    "status": "discarded",
                                    "reason": "user_hallucination",
                                    "hallucination_errors": n_errors,
                                }
                                status_path = sim_dir / "sim_status.json"
                                with open(status_path, "w") as f:
                                    json.dump(status, f, indent=2)
                            except Exception:
                                pass

                    # Build feedback and re-run
                    monitor.task_restarted(task_key)
                    feedback = format_hallucination_feedback(h_check)
                    retry_seed = seed + hallucination_retry_count * 1000
                    result = _execute(
                        run_seed=retry_seed,
                        hallucination_feedback=feedback,
                    )
                    result.trial = trial

                result.hallucination_retries_used = hallucination_retry_count

                if hallucination_retry_count > 0:
                    # Replace the eagerly-saved hallucinated result in the
                    # checkpoint with the clean retry.  Use the original seed
                    # so resume matching stays consistent.
                    result.seed = seed
                    replace_fn((trial, task.id, seed), result)

            # Mark the final sim as the one used in results
            if save_dir is not None:
                sim_dir = (
                    save_dir / "artifacts" / f"task_{task.id}" / f"sim_{result.id}"
                )
                if sim_dir.exists():
                    try:
                        status = {"status": "used"}
                        status_path = sim_dir / "sim_status.json"
                        with open(status_path, "w") as f:
                            json.dump(status, f, indent=2)
                    except Exception:
                        pass

            return result
        finally:
            monitor.task_finished(task_key)
            _cleanup_thread_event_loop()

    executor = ThreadPoolExecutor(max_workers=config.max_concurrency)
    futures: dict = {}
    try:
        futures = {executor.submit(_run_tracked, *arg): arg for arg in args}
        for future in as_completed(futures):
            result = future.result()
            simulation_results.simulations.append(result)
    except KeyboardInterrupt:
        ConsoleDisplay.console.print(
            "\n[bold red]Ctrl+C received — cancelling remaining tasks...[/bold red]"
        )
        shutdown_event.set()
        executor.shutdown(wait=False, cancel_futures=True)

        n = len(simulation_results.simulations)
        ConsoleDisplay.console.print(
            f"[bold yellow]{n} simulation(s) already checkpointed. "
            f"Use --auto-resume to continue later.[/bold yellow]"
        )
        monitor.stop()

        # Force-exit: background threads (litellm, websocket loops, etc.)
        # hold the process alive and produce noisy errors during interpreter
        # shutdown.  All completed results are already on disk via save_fn.
        os._exit(130)
    finally:
        monitor.stop()
        if not shutdown_event.is_set():
            executor.shutdown(wait=True)

    ConsoleDisplay.console.print(
        "\n[bold green]Successfully completed all simulations![/bold green]\n"
        "To review the simulations, run: [bold blue]tau2 view[/bold blue]"
    )
    return simulation_results


# =============================================================================
# Top-level entry points
# =============================================================================


def run_domain(config: RunConfig) -> Results:
    """Run simulations for a domain from a RunConfig.

    This is the main entry point for the CLI and API. It:
    1. Validates the config.
    2. Loads and filters tasks.
    3. Determines save paths.
    4. Delegates to run_tasks() for batch execution.
    5. Computes and displays metrics.

    Args:
        config: Full run configuration.

    Returns:
        Results object with all simulation runs.
    """
    config.validate()
    ConsoleDisplay.display_run_config(config)

    # Load tasks
    task_set_name = config.task_set_name or config.domain
    tasks = get_tasks(
        task_set_name=task_set_name,
        task_split_name=config.task_split_name,
        task_ids=config.task_ids,
        num_tasks=config.num_tasks,
    )

    # Filter tasks based on agent's registered task filter (if any)
    effective_agent = config.effective_agent
    task_filter = registry.get_agent_task_filter(effective_agent)
    if task_filter is not None:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if task_filter(task)]
        num_tasks = len(tasks)
        console_text = Text(
            text=f"Running {num_tasks} out of {total_num_tasks} tasks for {effective_agent} (filtered).",
            style="bold green",
        )
        ConsoleDisplay.console.print(console_text)

    # Determine save paths
    run_name = config.save_to or make_run_name(config)
    save_dir = DATA_DIR / "simulations" / run_name
    save_path = save_dir / "results.json"

    # Voice runs use directory format (individual sim files) because voice
    # simulations with tick data are very large; text runs use monolithic JSON.
    is_voice = isinstance(config, VoiceRunConfig)
    results_format = "dir" if is_voice else "json"

    # Run batch
    simulation_results = run_tasks(
        config,
        tasks,
        save_path=save_path,
        save_dir=save_dir,
        results_format=results_format,
    )

    # Compute and display metrics
    metrics = compute_metrics(simulation_results)
    ConsoleDisplay.display_agent_metrics(metrics)

    return simulation_results

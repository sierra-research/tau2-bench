"""
RunTau2Evaluation tool for ADK agent.

This tool enables external agents to request tau2-bench evaluations via A2A protocol.
Persists evaluation results to EvaluationStore for post-hoc metrics emission.
"""

import asyncio
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import litellm
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from loguru import logger

from tau2.store import EvaluationStore, create_store
from tau2.store.utils import generate_evaluation_id
from tau2_agent.config import EvaluationLimits
from tau2_agent.context import user_llm_api_key, user_llm_model
from tau2_agent.errors import ErrorCode, EvaluationError
from tau2_agent.llmobs_evaluations import (
    llmobs_evaluation_span,
    submit_evaluation_summary,
    submit_task_evaluations,
)
from tau2_agent.logging_config import (
    log_evaluation_complete,
    log_evaluation_error,
    log_evaluation_start,
)
from tau2_agent.metrics import emit_evaluation_metrics
from tau2_agent.utils import compact_message, sanitize_float


class MissingCredentialsError(Exception):
    """Raised when user LLM credentials are required but not provided."""

    def __init__(self, error: EvaluationError):
        self.error = error
        super().__init__(str(error))


class LimitExceededError(ValueError):
    """Raised when evaluation parameters exceed Cloud Run limits."""

    def __init__(self, error: EvaluationError):
        self.error = error
        super().__init__(str(error))


class UserLLMAuthError(Exception):
    """Raised when user's LLM API key authentication fails."""

    def __init__(self, error: EvaluationError):
        self.error = error
        super().__init__(str(error))

# Dedicated executor for evaluation work to prevent contention with other async operations.
# This isolates evaluation threads and allows multiple concurrent evaluations without
# exhausting the default executor used by ADK's event loop.
# See: specs/007-datadog-project/issue_tracker/concurrency-fix.
_EVALUATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="tau2_eval_",
)


@contextmanager
def evaluation_span(
    evaluation_id: str,
    domain: str,
    agent_endpoint: str,
    num_tasks: int | None = None,
    num_trials: int = 1,
):
    """Create a parent APM span for the entire evaluation.

    This groups all LLM calls, tool calls, and other operations under a single
    trace that can be viewed cohesively in Datadog APM.

    Args:
        evaluation_id: Unique identifier for this evaluation
        domain: The tau2 domain being evaluated
        agent_endpoint: The agent being evaluated
        num_tasks: Number of tasks in the evaluation
        num_trials: Number of trials per task
    """
    # Try to import ddtrace - if not available, yield None and return
    try:
        from ddtrace import tracer
    except ImportError:
        yield None
        return

    # Try to create the span - if it fails, yield None and return
    try:
        span = tracer.trace(
            "tau2.evaluation",
            service="tau2-bench-agent",
            resource=f"{domain}",
            span_type="worker",
        )
        span.set_tag("evaluation_id", evaluation_id)
        span.set_tag("domain", domain)
        span.set_tag("agent_endpoint", agent_endpoint)
        span.set_tag("num_tasks", num_tasks or "all")
        span.set_tag("num_trials", num_trials)
    except Exception as e:
        logger.debug(f"Failed to create evaluation span: {e}")
        yield None
        return

    # Yield the span and ensure it's finished properly
    try:
        yield span
    finally:
        span.finish()


class RunTau2Evaluation(BaseTool):
    """Tool to run tau2-bench agent evaluation"""

    name = "run_tau2_evaluation"
    description = """
    Run a tau2-bench evaluation of a conversational agent.

    IMPORTANT: Requires X-User-LLM-Model and X-User-LLM-API-Key headers.

    Parameters:
    - domain: Evaluation domain (airline, retail, telecom, mock)
    - agent_endpoint: A2A endpoint of agent to evaluate (e.g., https://agent.example.com)
    - num_trials: Number of trials per task (default: 1, max: 3)
    - num_tasks: Number of tasks to evaluate (default: all, max: 30)
    - task_ids: Optional list of specific task IDs to run

    Returns:
    - status: Evaluation completion status
    - timestamp: Evaluation start timestamp
    - summary: Evaluation metrics (success_rate, total_simulations, total_tasks)
    - tasks: List of evaluated tasks with IDs and names
    """

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        """
        Create the FunctionDeclaration used by the ADK function-calling interface for this tool.

        Returns:
            function_declaration (types.FunctionDeclaration | None): A FunctionDeclaration describing the tool's name, description, and parameter schema (including `domain`, `agent_endpoint`, `user_llm`, `num_trials`, and `num_tasks`), or `None` if a declaration cannot be generated.
        """
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "domain": types.Schema(
                        type=types.Type.STRING,
                        description="Evaluation domain: airline, retail, telecom, or mock",
                    ),
                    "agent_endpoint": types.Schema(
                        type=types.Type.STRING,
                        description="A2A endpoint URL of the agent to evaluate",
                    ),
                    "num_trials": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of trials per task (default: 1)",
                    ),
                    "num_tasks": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of tasks to evaluate (optional)",
                    ),
                    "task_ids": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Optional list of specific task IDs to run",
                    ),
                },
                required=["domain", "agent_endpoint"],
            ),
        )

    def _format_model_for_litellm(self, model: str) -> str:
        """Format model name for LiteLLM compatibility.

        LiteLLM expects specific prefixes for different providers:
        - OpenAI: gpt-4o, gpt-4-turbo (no prefix needed)
        - Gemini: gemini/gemini-2.0-flash
        - Anthropic: anthropic/claude-3-5-sonnet-20241022
        - Nebius: nebius/Qwen/Qwen3-235B-A22B (requires nebius/ prefix)

        Args:
            model: User-provided model name.

        Returns:
            Model name formatted for LiteLLM.
        """
        # Already has a provider prefix - return as-is
        if "/" in model:
            return model

        # Gemini models
        if model.startswith("gemini-"):
            return f"gemini/{model}"

        # Anthropic models
        if model.startswith("claude-"):
            return f"anthropic/{model}"

        # Nebius models (Qwen family hosted on Nebius)
        if model.startswith("Qwen"):
            return f"nebius/{model}"

        # OpenAI models (gpt-*, o1-*, etc.) - no prefix needed
        return model

    def _get_user_llm_credentials(self) -> tuple[str, dict[str, Any]]:
        """Get user LLM model and credentials from request headers.

        Reads credentials from contextvars set by middleware. Credentials
        are REQUIRED - raises MissingCredentialsError if not provided.

        Returns:
            Tuple of (litellm_formatted_model, llm_args dict with api_key).

        Raises:
            MissingCredentialsError: If X-User-LLM-Model or X-User-LLM-API-Key
                headers are not provided.
        """
        ctx_model = user_llm_model.get()
        ctx_api_key = user_llm_api_key.get()

        if not ctx_model or not ctx_api_key:
            error = EvaluationError(
                code=ErrorCode.MISSING_HEADER,
                message="User LLM credentials required. Include X-User-LLM-Model and X-User-LLM-API-Key headers.",
            )
            raise MissingCredentialsError(error)

        # Format model name for LiteLLM
        formatted_model = self._format_model_for_litellm(ctx_model)

        llm_args: dict[str, Any] = {"api_key": ctx_api_key}

        # Set api_base only for providers requiring custom endpoints
        if formatted_model.startswith("nebius/"):
            api_base = os.environ.get("NEBIUS_API_BASE")
            if api_base:
                llm_args["api_base"] = api_base
        elif os.environ.get("USER_LLM_API_BASE"):
            llm_args["api_base"] = os.environ.get("USER_LLM_API_BASE")

        return formatted_model, llm_args

    def _validate_limits(self, num_tasks: int | None, num_trials: int) -> None:
        """Validate evaluation parameters against Cloud Run limits.

        Args:
            num_tasks: Number of tasks to evaluate.
            num_trials: Number of trials per task.

        Raises:
            LimitExceededError: If num_tasks > MAX_TASKS or num_trials > MAX_TRIALS.
        """
        if num_tasks is not None and num_tasks > EvaluationLimits.MAX_TASKS:
            error = EvaluationError(
                code=ErrorCode.LIMIT_EXCEEDED,
                message=f"num_tasks must be between 1 and {EvaluationLimits.MAX_TASKS}",
                details={
                    "num_tasks": num_tasks,
                    "max_tasks": EvaluationLimits.MAX_TASKS,
                    "reason": "Cloud Run 60-minute timeout constraint",
                },
            )
            raise LimitExceededError(error)

        if num_trials > EvaluationLimits.MAX_TRIALS:
            error = EvaluationError(
                code=ErrorCode.LIMIT_EXCEEDED,
                message=f"num_trials must be between 1 and {EvaluationLimits.MAX_TRIALS}",
                details={
                    "num_trials": num_trials,
                    "max_trials": EvaluationLimits.MAX_TRIALS,
                    "reason": "Cloud Run 60-minute timeout constraint",
                },
            )
            raise LimitExceededError(error)

    async def run_async(  # type: ignore[override]
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any]:
        """
        Invoke the tool via the ADK function-calling interface using the supplied arguments and context.

        This method executes a tau2-bench evaluation and persists results to EvaluationStore.
        Returns evaluation results as a dict for the LLM agent to process.

        Note: ADK's tool execution framework awaits run_async directly, so this method
        returns a dict rather than being an async generator. The A2A layer handles
        SSE streaming of the agent's text responses automatically.

        Parameters:
            args (dict[str, Any]): Input fields expected by the tool. Recognized keys:
                - domain (str): Evaluation domain (required).
                - agent_endpoint (str): A2A endpoint URL of the agent to evaluate (required).
                - user_llm (str): LLM model identifier for the user simulator (optional).
                - num_trials (int): Number of trials per task (optional, default 1).
                - num_tasks (int | None): Number of tasks to evaluate (optional).
                - task_ids (list[str] | None): Specific task IDs to evaluate (optional).
            tool_context (ToolContext): ADK-provided execution context for the tool.

        Returns:
            dict[str, Any]: Evaluation results containing:
                - status: "completed" or "failed"
                - evaluation_id: Unique ID for this evaluation
                - summary: Aggregated metrics (success_rate, total_tasks, etc.)
                - tasks: List of evaluated tasks with results
        """
        domain = args.get("domain")
        agent_endpoint = args.get("agent_endpoint")
        if not isinstance(domain, str) or not isinstance(agent_endpoint, str):
            msg = "domain and agent_endpoint must be strings"
            raise TypeError(msg)

        num_trials = args.get("num_trials", 1)
        num_tasks = args.get("num_tasks")
        task_ids = args.get("task_ids")

        # Validate limits first (fail fast)
        try:
            self._validate_limits(num_tasks, num_trials)
        except LimitExceededError as e:
            return {"error": e.error.code.value, "message": e.error.message}

        # Get user LLM credentials from request headers (required)
        try:
            user_llm, llm_args_user = self._get_user_llm_credentials()
        except MissingCredentialsError as e:
            return {"error": e.error.code.value, "message": e.error.message}

        # Initialize EvaluationStore for persistence
        store: EvaluationStore | None = None
        try:
            store = create_store()
        except Exception as e:
            logger.warning(f"Failed to initialize EvaluationStore: {e}")

        evaluation_id: str | None = None

        try:
            # Create session in EvaluationStore
            request_data = {
                "user_llm": user_llm,
                "num_trials": num_trials,
                "num_tasks": num_tasks or 0,
            }

            if store:
                try:
                    evaluation_id = store.create_session(
                        domain=domain,
                        request=request_data,
                        agent_endpoint=agent_endpoint,
                    )
                    logger.info(
                        f"Created evaluation session: {evaluation_id}",
                        evaluation_id=evaluation_id,
                        domain=domain,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create store session: {e}")
                    evaluation_id = generate_evaluation_id()
            else:
                evaluation_id = generate_evaluation_id()

            # Transition to WORKING state before starting evaluation
            if store and evaluation_id:
                try:
                    store.update_progress(
                        evaluation_id,
                        current_task=1,
                        total_tasks=num_tasks or 1,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update progress to WORKING: {e}")

            # Log evaluation start with structured fields for GCP
            log_evaluation_start(
                domain=domain,
                agent_endpoint=agent_endpoint,
                num_tasks=num_tasks,
                evaluation_id=evaluation_id,
                user_llm=user_llm,
            )
            start_time = time.time()

            # Store span context for use after span closes
            llmobs_span_ctx = None

            # Wrap evaluation in APM span for cohesive tracing
            with evaluation_span(
                evaluation_id=evaluation_id or "unknown",
                domain=domain,
                agent_endpoint=agent_endpoint,
                num_tasks=num_tasks,
                num_trials=num_trials,
            ) as span:
                # Create LLMObs workflow span for evaluation metrics
                with llmobs_evaluation_span(
                    evaluation_id=evaluation_id or "unknown",
                    domain=domain,
                    agent_endpoint=agent_endpoint,
                ) as ctx:
                    llmobs_span_ctx = ctx  # Capture for use after span closes

                    # Run the evaluation
                    result = await self._execute(
                        _tool_context=tool_context,
                        domain=domain,
                        agent_endpoint=agent_endpoint,
                        user_llm=user_llm,
                        llm_args_user=llm_args_user,
                        num_trials=num_trials,
                        num_tasks=num_tasks,
                        task_ids=task_ids,
                        llmobs_span_context=llmobs_span_ctx,
                    )

                    # Add result metrics to span
                    if span and result.get("summary"):
                        span.set_tag("success_rate", result["summary"].get("avg_reward", 0))
                        span.set_tag("total_tasks", result["summary"].get("total_tasks", 0))
                        span.set_tag(
                            "successful_tasks",
                            result["summary"].get("successful_simulations", 0),
                        )

                    # Submit LLMObs evaluation summary inside the span context
                    submit_evaluation_summary(
                        evaluation_id=evaluation_id or "unknown",
                        domain=domain,
                        total_tasks=result["summary"]["total_tasks"],
                        successful_tasks=result["summary"]["successful_simulations"],
                        avg_reward=result["summary"]["avg_reward"],
                        agent_endpoint=agent_endpoint,
                        span_context=llmobs_span_ctx,
                    )

            # Build store_results for both store and metrics emission
            store_results = None
            if evaluation_id:
                task_results = []
                for sim in result.get("simulations", []):
                    reward_info = sim.get("reward_info", {})
                    reward = reward_info.get("reward", 0.0) if reward_info else 0.0
                    task_results.append({
                        "task_id": sim.get("task_id", "unknown"),
                        "success": reward >= 0.7,
                        "reward": reward,
                    })

                store_results = {
                    "success_rate": result["summary"]["successful_simulations"]
                    / result["summary"]["total_simulations"]
                    if result["summary"]["total_simulations"] > 0
                    else 0.0,
                    "total_tasks": result["summary"]["total_tasks"],
                    "successful": result["summary"]["successful_simulations"],
                    "tasks": task_results,
                    # Use full simulations data (with reasoning_content) for store
                    "simulations": result.get("_simulations_full", []),
                    "info": result.get("info"),
                }

            # Complete evaluation in store (if store is available)
            if store and evaluation_id and store_results:
                try:
                    store.complete_evaluation(
                        evaluation_id=evaluation_id,
                        results=store_results,
                    )
                except Exception as e:
                    logger.warning(f"Failed to complete evaluation in store: {e}")

            # Emit Datadog metrics regardless of store success
            # This ensures metrics are sent before container scales down
            if evaluation_id and store_results:
                try:
                    emit_evaluation_metrics(
                        evaluation_id=evaluation_id,
                        domain=domain,
                        agent_endpoint=agent_endpoint,
                        results=store_results,
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit metrics: {e}")

            # Log evaluation completion with structured fields for GCP
            duration_ms = (time.time() - start_time) * 1000
            success_rate = (
                result["summary"]["successful_simulations"]
                / result["summary"]["total_simulations"]
                if result["summary"]["total_simulations"] > 0
                else 0.0
            )
            log_evaluation_complete(
                evaluation_id=evaluation_id or "unknown",
                domain=domain,
                success_rate=success_rate,
                total_tasks=result["summary"]["total_tasks"],
                duration_ms=duration_ms,
                successful_simulations=result["summary"]["successful_simulations"],
            )

            # Add evaluation_id to result and remove internal fields
            result["evaluation_id"] = evaluation_id
            result.pop("_simulations_full", None)  # Don't send full data to Datadog
            return result

        except ValueError as e:
            # Fail evaluation in store
            if store and evaluation_id:
                try:
                    store.fail_evaluation(evaluation_id=evaluation_id, error=str(e))
                except Exception as store_err:
                    logger.warning(f"Failed to record failure in store: {store_err}")
            log_evaluation_error(
                evaluation_id=evaluation_id,
                domain=domain,
                error=str(e),
                error_type="INVALID_PARAMETERS",
            )
            return {"error": "INVALID_PARAMETERS", "message": str(e)}

        except UserLLMAuthError as e:
            # Fail evaluation in store
            if store and evaluation_id:
                try:
                    store.fail_evaluation(evaluation_id=evaluation_id, error=str(e))
                except Exception as store_err:
                    logger.warning(f"Failed to record failure in store: {store_err}")
            log_evaluation_error(
                evaluation_id=evaluation_id,
                domain=domain,
                error=e.error.message,
                error_type=e.error.code.value,
            )
            return {"error": e.error.code.value, "message": e.error.message}

        except Exception as e:
            # Fail evaluation in store
            if store and evaluation_id:
                try:
                    store.fail_evaluation(
                        evaluation_id=evaluation_id,
                        error=str(e) if str(e) else type(e).__name__,
                    )
                except Exception as store_err:
                    logger.warning(f"Failed to record failure in store: {store_err}")
            log_evaluation_error(
                evaluation_id=evaluation_id,
                domain=domain,
                error=str(e) if str(e) else type(e).__name__,
                error_type="INTERNAL_ERROR",
            )
            return {
                "error": "INTERNAL_ERROR",
                "message": str(e) if str(e) else type(e).__name__,
            }

    async def _execute(
        self,
        _tool_context: ToolContext,
        domain: str,
        agent_endpoint: str,
        user_llm: str,
        llm_args_user: dict[str, Any],
        num_trials: int = 1,
        num_tasks: int | None = None,
        task_ids: list[str] | None = None,
        llmobs_span_context: Any | None = None,
    ) -> dict[str, Any]:
        """
        Run a tau2-bench evaluation for a given domain and A2A agent endpoint.

        Parameters:
            domain (str): Evaluation domain identifier (e.g., "airline", "retail", "telecom", "mock").
            agent_endpoint (str): A2A endpoint URL of the agent under test.
            user_llm (str): LLM model identifier for the user simulator (LiteLLM formatted).
            llm_args_user (dict): LLM arguments including api_key for the user simulator.
            num_trials (int): Number of trials to run per task; defaults to 1.
            num_tasks (int | None): Optional number of tasks to evaluate; when None, uses domain defaults.
            task_ids (list[str] | None): Optional explicit list of task IDs to run.

        Returns:
            dict[str, Any]: A result object with keys:
                - status: "completed" on success.
                - timestamp: evaluation timestamp from tau2 results.
                - summary: dict with aggregated metrics
                - tasks: list of per-task dicts

        Raises:
            ValueError: If the provided domain is not recognized by tau2's registry.
        """
        try:
            # Import tau2-bench components
            from tau2.data_model.simulation import RunConfig
            from tau2.metrics.agent_metrics import compute_metrics, is_successful
            from tau2.registry import registry
            from tau2.run import run_domain

            # Validate domain using tau2's registry
            valid_domains = registry.get_domains()
            if domain not in valid_domains:
                msg = f"Invalid domain: {domain}. Must be one of {valid_domains}"
                raise ValueError(msg)

            logger.info(
                "Starting tau2-bench evaluation",
                domain=domain,
                agent_endpoint=agent_endpoint,
                user_llm=user_llm,
                num_trials=num_trials,
            )

            # Generate unique save_to path to prevent filename collisions when running
            # concurrent evaluations. This avoids the interactive prompt in run.py that
            # would block forever in headless mode asking about resuming existing runs.
            # See: specs/007-datadog-project/resolve-tau2agent-concurrency.md
            unique_run_id = f"tau2_eval_{uuid.uuid4().hex[:12]}"

            # Create run configuration
            config = RunConfig(
                domain=domain,
                task_set_name=None,
                task_split_name=None if task_ids else "base",
                task_ids=task_ids,
                num_tasks=num_tasks,
                is_remote=False,
                agent="a2a_agent",  # Use A2A client implementation
                llm_agent=agent_endpoint,  # A2A agent endpoint
                llm_args_agent={},
                user="user_simulator",
                llm_user=user_llm,
                llm_args_user=llm_args_user,
                num_trials=num_trials,
                max_steps=50,
                max_errors=10,
                save_to=unique_run_id,
                max_concurrency=1,
                seed=None,
                log_level="ERROR",
                enforce_communication_protocol=False,
                a2a_debug=False,
            )

            # Run evaluations in a dedicated thread pool to avoid blocking ADK's event loop.
            # Using _EVALUATION_EXECUTOR instead of the default executor (None) prevents
            # contention with other async operations and allows concurrent evaluations.
            # See: specs/007-datadog-project/resolve-tau2agent-concurrency.md
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(_EVALUATION_EXECUTOR, run_domain, config)

            # Use tau2's built-in metrics computation
            metrics = compute_metrics(results)

            total_simulations = len(results.simulations)
            successful_sims = sum(
                1
                for sim in results.simulations
                if sim.reward_info and is_successful(sim.reward_info.reward)
            )

            logger.info(
                "Evaluation completed",
                domain=domain,
                agent_endpoint=agent_endpoint,
                avg_reward=metrics.avg_reward,
                total_simulations=total_simulations,
            )

            # Build simulation data - full version for store, compact for tracing
            simulations_data_full = []  # Full data for EvaluationStore
            simulations_data_compact = []  # Compact data for Datadog traces
            for sim in results.simulations:
                # Full messages for EvaluationStore
                full_messages = [
                    msg.model_dump(mode="json") if hasattr(msg, "model_dump") else msg
                    for msg in (sim.messages or [])
                ]
                # Compact messages for tracing (removes raw_data, reasoning_content)
                compact_messages = [compact_message(msg) for msg in full_messages]

                base_sim_data = {
                    "task_id": sim.task_id,
                    "duration": sim.duration,
                    "termination_reason": (
                        sim.termination_reason.value
                        if hasattr(sim.termination_reason, "value")
                        else str(sim.termination_reason)
                    ),
                    "reward_info": (
                        sim.reward_info.model_dump(mode="json")
                        if sim.reward_info and hasattr(sim.reward_info, "model_dump")
                        else sim.reward_info
                    ),
                }
                simulations_data_full.append({**base_sim_data, "messages": full_messages})
                simulations_data_compact.append({**base_sim_data, "messages": compact_messages})

                # Submit LLMObs evaluations for this simulation (real-time trace correlation)
                submit_task_evaluations(
                    task_id=sim.task_id,
                    domain=domain,
                    reward=sim.reward_info.reward if sim.reward_info else 0.0,
                    termination_reason=base_sim_data["termination_reason"],
                    reward_info=base_sim_data["reward_info"],
                    agent_endpoint=agent_endpoint,
                    span_context=llmobs_span_context,
                )

            # Build result with compact simulations for Datadog traces
            result = {
                "status": "completed",
                "timestamp": results.timestamp,
                "summary": {
                    "total_simulations": total_simulations,
                    "total_tasks": len(results.tasks),
                    "successful_simulations": successful_sims,
                    "avg_reward": sanitize_float(metrics.avg_reward),
                    "pass_hat_k": {
                        k: sanitize_float(v) for k, v in metrics.pass_hat_ks.items()
                    },
                    "avg_agent_cost": sanitize_float(metrics.avg_agent_cost),
                },
                "tasks": [
                    {
                        "task_id": task.id,
                        "purpose": (
                            task.description.purpose
                            if task.description and task.description.purpose
                            else None
                        ),
                    }
                    for task in results.tasks
                ],
                # Compact simulation data for Datadog traces (< 1MB limit)
                "simulations": simulations_data_compact,
                "info": {
                    "environment_info": {
                        "domain_name": domain,
                    },
                },
                # Full simulation data for EvaluationStore (not sent to Datadog)
                "_simulations_full": simulations_data_full,
            }
            return result

        except ValueError as e:
            logger.error("Invalid evaluation parameters", error=str(e))
            raise

        except litellm.AuthenticationError as e:
            # LiteLLM raises AuthenticationError for invalid API keys
            error = EvaluationError(
                code=ErrorCode.USER_LLM_AUTH_FAILED,
                message="User LLM authentication failed",
                details={"model": user_llm},  # Never include API key
            )
            logger.warning(
                "User LLM authentication failed",
                model=user_llm,
                error_type=type(e).__name__,
            )
            raise UserLLMAuthError(error) from e

        except Exception as e:
            # Check for other auth-like errors (e.g., wrapped exceptions)
            if "Unauthorized" in str(e):
                error = EvaluationError(
                    code=ErrorCode.USER_LLM_AUTH_FAILED,
                    message="User LLM authentication failed",
                    details={"model": user_llm},  # Never include API key
                )
                logger.warning(
                    "User LLM authentication failed",
                    model=user_llm,
                    error_type=type(e).__name__,
                )
                raise UserLLMAuthError(error) from e

            logger.error(
                "Evaluation failed",
                domain=domain,
                agent_endpoint=agent_endpoint,
                error=str(e),
                exc_info=True,
            )
            raise

"""
RunTau2Evaluation tool for ADK agent.

This tool enables external agents to request tau2-bench evaluations via A2A protocol.
Persists evaluation results to EvaluationStore for post-hoc metrics emission.
"""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from loguru import logger

from tau2.store import EvaluationStore, create_store
from tau2.store.utils import generate_evaluation_id
from tau2_agent.utils import compact_message, sanitize_float

DEFAULT_USER_LLM = (
    "openai/Qwen/Qwen3-30B-A3B-Thinking-2507"
    if os.getenv("NEBIUS_API_KEY")
    else "gpt-4o"
)

# Dedicated executor for evaluation work to prevent contention with other async operations.
# This isolates evaluation threads and allows multiple concurrent evaluations without
# exhausting the default executor used by ADK's event loop.
# See: specs/007-datadog-project/issue_tracker/concurrency-fix.
_EVALUATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=10,
    thread_name_prefix="tau2_eval_",
)


class RunTau2Evaluation(BaseTool):
    """Tool to run tau2-bench agent evaluation"""

    name = "run_tau2_evaluation"
    description = f"""
    Run a tau2-bench evaluation of a conversational agent.

    Parameters:
    - domain: Evaluation domain (airline, retail, telecom, mock)
    - agent_endpoint: A2A endpoint of agent to evaluate (e.g., https://agent.example.com)
    - user_llm: LLM model for user simulator (default: {DEFAULT_USER_LLM})
    - num_trials: Number of trials per task (default: 1)
    - num_tasks: Number of tasks to evaluate (default: all tasks in domain)
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
                    "user_llm": types.Schema(
                        type=types.Type.STRING,
                        description=f"LLM model for user simulator (default: {DEFAULT_USER_LLM})",
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

        # Initialize EvaluationStore for persistence
        store: EvaluationStore | None = None
        try:
            store = create_store()
        except Exception as e:
            logger.warning(f"Failed to initialize EvaluationStore: {e}")

        evaluation_id: str | None = None
        user_llm = args.get("user_llm", DEFAULT_USER_LLM)
        num_trials = args.get("num_trials", 1)
        num_tasks = args.get("num_tasks")
        task_ids = args.get("task_ids")

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

            # Run the evaluation
            result = await self._execute(
                _tool_context=tool_context,
                domain=domain,
                agent_endpoint=agent_endpoint,
                user_llm=user_llm,
                num_trials=num_trials,
                num_tasks=num_tasks,
                task_ids=task_ids,
            )

            # Complete evaluation in store
            if store and evaluation_id:
                try:
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

                    store.complete_evaluation(
                        evaluation_id=evaluation_id,
                        results=store_results,
                    )
                    logger.info(
                        f"Completed evaluation in store: {evaluation_id}",
                        evaluation_id=evaluation_id,
                        success_rate=store_results["success_rate"],
                    )
                except Exception as e:
                    logger.warning(f"Failed to complete evaluation in store: {e}")

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
            raise

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
            raise

    async def _execute(
        self,
        _tool_context: ToolContext,
        domain: str,
        agent_endpoint: str,
        user_llm: str = DEFAULT_USER_LLM,
        num_trials: int = 1,
        num_tasks: int | None = None,
        task_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run a tau2-bench evaluation for a given domain and A2A agent endpoint.

        Validates the domain, constructs a RunConfig (wiring the agent endpoint and optional Nebius/OpenAI credentials for the user LLM), executes the evaluation in a thread pool to avoid blocking the event loop, computes aggregate metrics, and returns a structured summary and per-task metadata.

        Parameters:
            domain (str): Evaluation domain identifier (e.g., "airline", "retail", "telecom", "mock").
            agent_endpoint (str): A2A endpoint URL of the agent under test.
            user_llm (str): LLM model identifier for the user simulator; defaults to DEFAULT_USER_LLM.
            num_trials (int): Number of trials to run per task; defaults to 1.
            num_tasks (int | None): Optional number of tasks to evaluate; when None, uses domain defaults.
            task_ids (list[str] | None): Optional explicit list of task IDs to run.

        Returns:
            dict[str, Any]: A result object with keys:
                - status: "completed" on success.
                - timestamp: evaluation timestamp from tau2 results.
                - summary: dict with aggregated metrics:
                    - total_simulations (int)
                    - total_tasks (int)
                    - successful_simulations (int)
                    - avg_reward (float)
                    - pass_hat_k (mapping or list as produced by tau2)
                    - avg_agent_cost (float)
                - tasks: list of per-task dicts with:
                    - task_id (str)
                    - purpose (str | None)

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

            # Build llm_args_user - pass Nebius credentials for openai/ provider models
            llm_args_user = {}
            nebius_api_key = os.getenv("NEBIUS_API_KEY")
            if user_llm.startswith("openai/") and nebius_api_key:
                llm_args_user = {
                    "api_key": nebius_api_key,
                    "api_base": os.getenv(
                        "NEBIUS_API_BASE", "https://api.tokenfactory.nebius.com/v1/"
                    ),
                }

            # Generate unique save_to path to prevent filename collisions when running
            # concurrent evaluations. This avoids the interactive prompt in run.py that
            # would block forever in headless mode asking about resuming existing runs.
            # See: specs/007-datadog-project/resolve-tau2agent-concurrency.md
            unique_run_id = f"tau2_eval_{uuid.uuid4().hex[:12]}"

            # Create run configuration
            config = RunConfig(
                domain=domain,
                task_set_name=None,
                task_split_name="base",
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

        except Exception as e:
            logger.error(
                "Evaluation failed",
                domain=domain,
                agent_endpoint=agent_endpoint,
                error=str(e),
                exc_info=True,
            )
            raise

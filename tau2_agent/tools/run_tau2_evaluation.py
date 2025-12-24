"""
RunTau2Evaluation tool for ADK agent.

This tool enables external agents to request tau2-bench evaluations via A2A protocol.
Emits SSE progress events during evaluation for real-time monitoring.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

from google.adk.events.event import Event
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from loguru import logger

from tau2.store.utils import generate_evaluation_id
from tau2_agent.streaming import (
    EvaluationProgress,
    create_adk_error_event,
    create_adk_progress_event,
    create_adk_result_event,
)
from tau2_agent.streaming.metadata import (
    TAU2_AGENT_ENDPOINT,
    TAU2_DOMAIN,
)

DEFAULT_USER_LLM = (
    "openai/Qwen/Qwen3-30B-A3B-Thinking-2507"
    if os.getenv("NEBIUS_API_KEY")
    else "gpt-4o"
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
    ) -> AsyncIterator[Event]:
        """
        Invoke the tool via the ADK function-calling interface using the supplied arguments and context.

        This method returns an AsyncIterator that yields ADK Event objects for SSE streaming.
        Events include progress updates, error states, and final results.

        Parameters:
            args (dict[str, Any]): Input fields expected by the tool. Recognized keys:
                - domain (str): Evaluation domain (required).
                - agent_endpoint (str): A2A endpoint URL of the agent to evaluate (required).
                - user_llm (str): LLM model identifier for the user simulator (optional).
                - num_trials (int): Number of trials per task (optional, default 1).
                - num_tasks (int | None): Number of tasks to evaluate (optional).
                - task_ids (list[str] | None): Specific task IDs to evaluate (optional).
            tool_context (ToolContext): ADK-provided execution context for the tool.

        Yields:
            Event: ADK Event objects with tau2 metadata for SSE streaming:
                - submitted: Initial acknowledgment event
                - working: Progress update events (emitted per task)
                - completed: Final result event with evaluation results
                - failed: Error event if evaluation fails
        """
        domain = args.get("domain")
        agent_endpoint = args.get("agent_endpoint")
        if not isinstance(domain, str) or not isinstance(agent_endpoint, str):
            msg = "domain and agent_endpoint must be strings"
            raise TypeError(msg)

        # Delegate to async generator
        async for event in self._execute_streaming(
            tool_context=tool_context,
            domain=domain,
            agent_endpoint=agent_endpoint,
            user_llm=args.get("user_llm", DEFAULT_USER_LLM),
            num_trials=args.get("num_trials", 1),
            num_tasks=args.get("num_tasks"),
            task_ids=args.get("task_ids"),
        ):
            yield event

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
                save_to=None,
                max_concurrency=1,
                seed=None,
                log_level="ERROR",
                enforce_communication_protocol=False,
                a2a_debug=False,
            )

            # Run evaluations in a thread pool to avoid blocking ADK's event loop.
            # This is critical when both tau2_agent and the agent being evaluated
            # (e.g., simple_nebius_agent) run on the same ADK server - blocking
            # the event loop would cause a deadlock when A2AAgent tries to make
            # HTTP requests to the other agent.
            # See: https://github.com/encode/httpx/discussions/2489
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, run_domain, config)

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

            return {
                "status": "completed",
                "timestamp": results.timestamp,
                "summary": {
                    "total_simulations": total_simulations,
                    "total_tasks": len(results.tasks),
                    "successful_simulations": successful_sims,
                    "avg_reward": metrics.avg_reward,
                    "pass_hat_k": metrics.pass_hat_ks,
                    "avg_agent_cost": metrics.avg_agent_cost,
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
            }

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

    async def _execute_streaming(
        self,
        tool_context: ToolContext,
        domain: str,
        agent_endpoint: str,
        user_llm: str = DEFAULT_USER_LLM,
        num_trials: int = 1,
        num_tasks: int | None = None,
        task_ids: list[str] | None = None,
    ) -> AsyncIterator[Event]:
        """Run tau2-bench evaluation with SSE streaming events.

        Wraps the _execute method with streaming event emission for real-time
        progress monitoring. Emits events with tau2 metadata for tracing.

        Parameters:
            tool_context (ToolContext): ADK-provided execution context.
            domain (str): Evaluation domain identifier.
            agent_endpoint (str): A2A endpoint URL of the agent under test.
            user_llm (str): LLM model identifier for user simulator.
            num_trials (int): Number of trials per task.
            num_tasks (int | None): Optional number of tasks to evaluate.
            task_ids (list[str] | None): Optional explicit list of task IDs.

        Yields:
            Event: ADK Event objects for SSE streaming with tau2 metadata:
                - tau2.evaluation_id: Unique evaluation correlation ID
                - tau2.domain: Evaluation domain
                - tau2.agent_endpoint: Agent being evaluated
                - tau2.state: Current state (submitted/working/completed/failed)
                - tau2.progress: Completion percentage (0-100)
                - tau2.current_task_id: Task currently being evaluated (for working state)
        """
        # Generate evaluation ID for event correlation
        # Note: This is a correlation ID for streaming. Full store integration
        # (calling store.create_session) is a separate concern.
        evaluation_id = generate_evaluation_id()
        invocation_id = tool_context.invocation_id or evaluation_id

        # Common trace context metadata for all events
        # These fields support 007-datadog instrumentation
        trace_context = {
            TAU2_DOMAIN: domain,
            TAU2_AGENT_ENDPOINT: agent_endpoint,
        }

        try:
            # Import tau2-bench components
            from tau2.registry import registry
            from tau2.run import load_tasks

            # Validate domain
            valid_domains = registry.get_domains()
            if domain not in valid_domains:
                msg = f"Invalid domain: {domain}. Must be one of {valid_domains}"
                raise ValueError(msg)

            # Get task count for progress tracking
            # Use provided task_ids count, num_tasks, or load from domain
            if task_ids:
                estimated_task_count = len(task_ids)
            elif num_tasks:
                estimated_task_count = num_tasks
            else:
                # Load domain tasks to get count
                try:
                    domain_tasks = load_tasks(domain)
                    estimated_task_count = len(domain_tasks)
                except Exception:
                    # Fallback if tasks can't be loaded
                    estimated_task_count = 0

            # Initialize progress tracker
            progress = EvaluationProgress(
                total_tasks=estimated_task_count,
                total_trials=num_trials,
            )

            # Emit submitted event
            yield create_adk_progress_event(
                invocation_id=invocation_id,
                state="submitted",
                message=f"Starting {domain} evaluation with {estimated_task_count} tasks",
                evaluation_id=evaluation_id,
                progress=progress,
                **trace_context,
            )

            # Run the evaluation (this blocks while evaluation runs)
            result = await self._execute(
                _tool_context=tool_context,
                domain=domain,
                agent_endpoint=agent_endpoint,
                user_llm=user_llm,
                num_trials=num_trials,
                num_tasks=num_tasks,
                task_ids=task_ids,
            )

            # Update progress based on actual results
            actual_task_count = result["summary"]["total_tasks"]
            progress = EvaluationProgress(
                total_tasks=actual_task_count,
                completed_tasks=actual_task_count,
                total_trials=num_trials,
            )

            # Emit working events for each completed task
            # Note: Since run_domain is synchronous, we emit progress after completion
            for task in result.get("tasks", []):
                task_id = task.get("task_id", "unknown")
                progress.current_task_id = task_id

                yield create_adk_progress_event(
                    invocation_id=invocation_id,
                    state="working",
                    message=f"Completed task {task_id}",
                    evaluation_id=evaluation_id,
                    progress=progress,
                    **trace_context,
                )

            # Emit final result event
            yield create_adk_result_event(
                invocation_id=invocation_id,
                evaluation_id=evaluation_id,
                results=result,
                message=f"Evaluation complete: {result['summary']['successful_simulations']}/{result['summary']['total_simulations']} simulations successful",
                **trace_context,
            )

        except ValueError as e:
            # Invalid parameters - emit error event
            yield create_adk_error_event(
                invocation_id=invocation_id,
                evaluation_id=evaluation_id,
                error_message=str(e),
                error_code="INVALID_PARAMETERS",
                **trace_context,
            )

        except Exception as e:
            # Unexpected error - emit error event
            error_message = str(e) if str(e) else type(e).__name__
            yield create_adk_error_event(
                invocation_id=invocation_id,
                evaluation_id=evaluation_id,
                error_message=error_message,
                error_code="EVALUATION_FAILED",
                **trace_context,
            )

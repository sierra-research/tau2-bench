"""
RunTau2Evaluation tool for ADK agent.

This tool enables external agents to request tau2-bench evaluations via A2A protocol.
"""

import asyncio
import os
from typing import Any

from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from loguru import logger

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

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        """
        Invoke the tool via the ADK function-calling interface using the supplied arguments and context.

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
            dict[str, Any]: Structured evaluation result (e.g., status, timestamp, summary, tasks) produced by the execution.
        """
        domain = args.get("domain")
        agent_endpoint = args.get("agent_endpoint")
        if not isinstance(domain, str) or not isinstance(agent_endpoint, str):
            msg = "domain and agent_endpoint must be strings"
            raise TypeError(msg)
        return await self._execute(
            _tool_context=tool_context,
            domain=domain,
            agent_endpoint=agent_endpoint,
            user_llm=args.get("user_llm", DEFAULT_USER_LLM),
            num_trials=args.get("num_trials", 1),
            num_tasks=args.get("num_tasks"),
            task_ids=args.get("task_ids"),
        )

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

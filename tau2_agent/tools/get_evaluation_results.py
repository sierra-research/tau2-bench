"""
GetEvaluationResults tool for ADK agent.

This tool enables external agents to retrieve completed evaluation results.
"""

import re
from typing import Any

from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from loguru import logger


class GetEvaluationResults(BaseTool):
    """Retrieve results from a completed evaluation"""

    name = "get_evaluation_results"
    description = (
        "Get detailed results from a tau2-bench evaluation. "
        "Provide either evaluation_id (filename without .json) or list_available=true "
        "to see available evaluation files."
    )

    def _get_declaration(self) -> types.FunctionDeclaration | None:
        """
        Return a FunctionDeclaration describing this tool's public API.

        Returns:
            declaration (types.FunctionDeclaration | None): A declaration object containing the tool name, description, and parameter schema, or `None` if the tool should not be exposed.
        """
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "evaluation_id": types.Schema(
                        type=types.Type.STRING,
                        description="The ID/filename of the evaluation to retrieve (without .json extension)",
                    ),
                    "list_available": types.Schema(
                        type=types.Type.BOOLEAN,
                        description="If true, list all available evaluation result files",
                    ),
                },
                required=[],  # Both parameters are optional
            ),
        )

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context: ToolContext,  # noqa: ARG002
    ) -> Any:
        """
        Retrieve or list tau2 evaluation results from the simulations data directory.

        Parameters:
            args (dict): Input arguments. Recognized keys:
                - evaluation_id (str): Identifier of the evaluation file (filename without ".json") to load.
                - list_available (bool): If truthy, return a list of available evaluation IDs instead of loading a specific result.
            _tool_context: Invocation context (not used).

        Returns:
            dict: One of the following payload shapes:
              - Listing payload when `list_available` is truthy:
                  {
                    "available_evaluations": [<evaluation_id>, ...],  # empty if no directory
                    "simulations_dir": "<absolute path to simulations directory>"
                  }
              - Error payloads for missing inputs or missing files:
                  {
                    "error": "<error message>",
                    "message": "<explanatory message>"            # optional
                  }
                or
                  {
                    "error": "Evaluation not found: <evaluation_id>",
                    "available_evaluations": [<evaluation_id>, ...]  # may be empty
                  }
              - Successful evaluation payload when a valid `evaluation_id` is provided:
                  {
                    "evaluation_id": "<evaluation_id>",
                    "timestamp": <results.timestamp>,
                    "info": {
                      "num_trials": <int>,
                      "max_steps": <int>,
                      "agent": "<agent implementation>",
                      "user": "<user implementation>"
                    },
                    "summary": {
                      "total_simulations": <int>,
                      "total_tasks": <int>,
                      "successful_simulations": <int>,
                      "avg_reward": <float>,
                      "pass_hat_k": <object/metric>,
                      "avg_agent_cost": <float>
                    },
                    "tasks": [{"task_id": "<task id>"}, ...]
                  }

        Notes:
            - If the simulations directory does not exist, the listing payload returns an empty
              `available_evaluations` list with the expected `simulations_dir` path.
            - Any failure during loading or processing returns an error payload and logs the exception.
        """
        from tau2.data_model.simulation import Results
        from tau2.metrics.agent_metrics import compute_metrics, is_successful
        from tau2.utils.utils import DATA_DIR

        simulations_dir = DATA_DIR / "simulations"

        # List available evaluations if requested
        if args.get("list_available"):
            if not simulations_dir.exists():
                return {
                    "available_evaluations": [],
                    "simulations_dir": str(simulations_dir),
                }

            files = list(simulations_dir.glob("*.json"))
            return {
                "available_evaluations": [f.stem for f in files],
                "simulations_dir": str(simulations_dir),
            }

        # Load specific evaluation
        evaluation_id = args.get("evaluation_id")
        if not evaluation_id:
            return {
                "error": "Missing evaluation_id",
                "message": "Provide evaluation_id or set list_available=true",
            }

        # Sanitize evaluation_id to prevent path traversal attacks
        # Only allow alphanumeric characters, hyphens, underscores, and dots (not leading)
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\.]*$", evaluation_id):
            return {
                "error": "Invalid evaluation_id format",
                "message": "evaluation_id must contain only alphanumeric characters, hyphens, underscores, and dots",
            }

        # Construct path to results file
        results_path = simulations_dir / f"{evaluation_id}.json"

        # Verify the resolved path is within simulations_dir (defense in depth)
        if not results_path.resolve().is_relative_to(simulations_dir.resolve()):
            return {
                "error": "Invalid evaluation_id",
                "message": "Path traversal attempt detected",
            }

        if not results_path.exists():
            return {
                "error": f"Evaluation not found: {evaluation_id}",
                "available_evaluations": [
                    f.stem for f in simulations_dir.glob("*.json")
                ]
                if simulations_dir.exists()
                else [],
            }

        try:
            # Use tau2's Results.load() method
            results = Results.load(results_path)

            # Use tau2's compute_metrics for analysis
            metrics = compute_metrics(results)

            # Count successful simulations using tau2's is_successful
            successful_sims = sum(
                1
                for sim in results.simulations
                if sim.reward_info and is_successful(sim.reward_info.reward)
            )

            return {
                "evaluation_id": evaluation_id,
                "timestamp": results.timestamp,
                "info": {
                    "num_trials": results.info.num_trials,
                    "max_steps": results.info.max_steps,
                    "agent": results.info.agent_info.implementation,
                    "user": results.info.user_info.implementation,
                },
                "summary": {
                    "total_simulations": len(results.simulations),
                    "total_tasks": len(results.tasks),
                    "successful_simulations": successful_sims,
                    "avg_reward": metrics.avg_reward,
                    "pass_hat_k": metrics.pass_hat_ks,
                    "avg_agent_cost": metrics.avg_agent_cost,
                },
                "tasks": [{"task_id": task.id} for task in results.tasks],
            }

        except Exception as e:
            logger.exception(f"Failed to load evaluation {evaluation_id}")
            return {
                "error": f"Failed to load evaluation: {e}",
                "evaluation_id": evaluation_id,
            }

"""AgentBeats-compatible evaluation executor with DataPart results.

This module provides a GreenExecutor that implements the A2A AgentExecutor
interface directly, bypassing the LLM orchestrator. This ensures evaluation
results are returned as DataPart artifacts that agentbeats-client can parse.

The executor reuses the existing RunTau2Evaluation tool for all evaluation
logic, ensuring consistency with the LlmAgent path.

Usage:
    from tau2_agent.green_executor import Tau2GreenExecutor, create_green_agent_card

    executor = Tau2GreenExecutor()
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    app = A2AStarletteApplication(agent_card=create_green_agent_card(url), http_handler=handler)
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, DataPart, Part, TaskState, TextPart
from a2a.utils import new_agent_text_message, new_task
from loguru import logger
from pydantic import BaseModel

from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation


class EvalConfig(BaseModel):
    """Evaluation configuration from agentbeats scenario.

    Attributes:
        domain: Evaluation domain (airline, retail, telecom, mock).
        num_tasks: Number of tasks to evaluate (None = all tasks).
        num_trials: Number of trials per task.
        task_ids: Optional list of specific task IDs to run.
    """

    domain: str
    num_tasks: int | None = None
    num_trials: int = 1
    task_ids: list[str] | None = None


class EvalRequest(BaseModel):
    """Request format from agentbeats-client.

    Attributes:
        participants: Map of role to endpoint URL. Must include "agent" key.
        config: Evaluation configuration.
    """

    participants: dict[str, str]
    config: EvalConfig


def create_green_agent_card(base_url: str) -> AgentCard:
    """Create agent card for the green executor route.

    Args:
        base_url: External URL where the agent is accessible.

    Returns:
        AgentCard with green executor metadata.
    """
    return AgentCard(
        name="tau2_green",
        description="AgentBeats-compatible tau2 evaluation service (structured DataPart results)",
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[],
    )


def _extract_failure_insights(reward_info: dict | None) -> list[str]:
    """Extract actionable failure reasons from reward_info.

    Args:
        reward_info: RewardInfo dict from simulation result.

    Returns:
        List of failure reasons explaining why the task failed.
    """
    if not reward_info:
        return []

    insights = []

    # Check NL assertions (natural language requirements)
    nl_assertions = reward_info.get("nl_assertions") or []
    for assertion in nl_assertions:
        if not assertion.get("met", True):
            insights.append(f"Failed: {assertion.get('nl_assertion', 'Unknown requirement')}")

    # Check communicate requirements (info agent should have conveyed)
    communicate_checks = reward_info.get("communicate_checks") or []
    for check in communicate_checks:
        if not check.get("met", True):
            insights.append(f"Missing communication: {check.get('info', 'Unknown info')}")

    # Check action requirements
    action_checks = reward_info.get("action_checks") or []
    for check in action_checks:
        if not check.get("action_match", True):
            action = check.get("action", {})
            action_name = action.get("name", "Unknown action") if isinstance(action, dict) else str(action)
            insights.append(f"Action not completed: {action_name}")

    # Check environment assertions
    env_assertions = reward_info.get("env_assertions") or []
    for assertion in env_assertions:
        if not assertion.get("met", True):
            env_assert = assertion.get("env_assertion", {})
            desc = env_assert.get("description", "Unknown") if isinstance(env_assert, dict) else str(env_assert)
            insights.append(f"Environment check failed: {desc}")

    # Check DB state
    db_check = reward_info.get("db_check")
    if db_check and not db_check.get("db_match", True):
        insights.append("Database state does not match expected outcome")

    return insights


def _build_task_results(simulations: list[dict], threshold: float = 0.7) -> list[dict]:
    """Build per-task results with success status and failure insights.

    Args:
        simulations: List of simulation dicts from evaluation result.
        threshold: Reward threshold for success (default 0.7).

    Returns:
        List of task result dicts with actionable information.
    """
    task_results = []
    for sim in simulations:
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward", 0.0) if reward_info else 0.0
        success = reward >= threshold

        task_result: dict = {
            "task_id": sim.get("task_id", "unknown"),
            "success": success,
            "reward": reward,
            "termination_reason": sim.get("termination_reason", "unknown"),
            "num_turns": len(sim.get("messages", [])),
        }

        # Add failure insights for failed tasks
        if not success:
            insights = _extract_failure_insights(reward_info)
            if insights:
                task_result["failure_insights"] = insights

        task_results.append(task_result)

    return task_results


class Tau2GreenAgent:
    """Direct evaluation executor for AgentBeats.

    This agent wraps RunTau2Evaluation and ensures results are returned
    as both TextPart (human-readable) and DataPart (structured JSON) artifacts.
    """

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """Execute evaluation and return structured results.

        Args:
            request: Evaluation request with participants and config.
            updater: TaskUpdater for streaming status and artifacts.

        Raises:
            ValueError: If no participants provided or evaluation fails.
        """
        logger.info("Parsed EvalRequest", participants=request.participants, config=request.config.model_dump())

        if not request.participants:
            raise ValueError("No participants provided")

        participant_name, agent_endpoint = next(iter(request.participants.items()))
        logger.info(f"Evaluating participant '{participant_name}'", endpoint=agent_endpoint)

        logger.info(
            "Starting green executor evaluation",
            domain=request.config.domain,
            agent_endpoint=agent_endpoint,
            num_tasks=request.config.num_tasks,
            num_trials=request.config.num_trials,
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting evaluation: domain={request.config.domain}, "
                f"num_tasks={request.config.num_tasks or 'all'}, "
                f"agent={agent_endpoint}"
            ),
        )

        # Reuse existing evaluation tool logic
        tool = RunTau2Evaluation(name="run_tau2_evaluation", description="")

        args: dict = {
            "domain": request.config.domain,
            "agent_endpoint": agent_endpoint,
            "num_trials": request.config.num_trials,
        }
        if request.config.num_tasks is not None:
            args["num_tasks"] = request.config.num_tasks
        if request.config.task_ids:
            args["task_ids"] = request.config.task_ids

        result = await tool.run_async(args=args, tool_context=None)  # type: ignore[arg-type]

        # Check for errors in result
        if "error" in result:
            error_msg = f"{result['error']}: {result.get('message', 'Unknown error')}"
            logger.error("Evaluation failed", error=result["error"], message=result.get("message"))
            raise ValueError(error_msg)

        # Format human-readable summary
        summary = result.get("summary") or {}
        total = summary.get("total_simulations") or 0
        successful = summary.get("successful_simulations") or 0
        avg_reward = summary.get("avg_reward") or 0

        total_tasks = summary.get("total_tasks") or 0
        summary_text = f"""Evaluation Results
Domain: {request.config.domain}
Tasks: {total_tasks}
Pass Rate: {successful}/{total} ({avg_reward:.1%})"""

        # Add failed task summary if any failures
        failed_count = total - successful
        if failed_count > 0:
            summary_text += f"\nFailed Tasks: {failed_count}"

        logger.info(
            "Evaluation completed successfully",
            domain=request.config.domain,
            total_tasks=total_tasks,
            pass_rate=f"{avg_reward:.1%}",
        )

        # Build curated output (no conversation traces, with failure insights)
        simulations = result.get("simulations") or []
        task_results = _build_task_results(simulations)

        curated_result = {
            "status": result.get("status", "completed"),
            "evaluation_id": result.get("evaluation_id"),
            "timestamp": result.get("timestamp"),
            "summary": {
                "total_simulations": total,
                "total_tasks": total_tasks,
                "successful_simulations": successful,
                "avg_reward": avg_reward,
                "pass_hat_k": summary.get("pass_hat_k"),
            },
            "tasks": result.get("tasks", []),
            "task_results": task_results,
        }

        # Add artifact with BOTH TextPart (human) and DataPart (structured)
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=curated_result)),
            ],
            name="evaluation_results",
        )


class Tau2GreenExecutor(AgentExecutor):
    """A2A executor that wraps Tau2GreenAgent for agentbeats compatibility.

    This executor implements the A2A AgentExecutor interface directly,
    parsing EvalRequest from the incoming message and running evaluation
    without LLM orchestration.
    """

    def __init__(self) -> None:
        self.agent = Tau2GreenAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute evaluation request and stream results.

        Args:
            context: A2A request context containing user message.
            event_queue: Queue for streaming SSE events back to client.
        """
        # Parse EvalRequest from A2A message
        request_text = context.get_user_input()
        logger.info("Received green executor request", request_text=request_text[:500])

        try:
            request = EvalRequest.model_validate_json(request_text)
        except Exception as e:
            logger.error("Failed to parse EvalRequest", error=str(e), raw_input=request_text[:1000])
            msg = f"Invalid EvalRequest format: {e}"
            raise ValueError(msg) from e

        # Create task and send initial event
        task = new_task(context.message)
        await event_queue.enqueue_event(task)

        # Run evaluation with TaskUpdater for streaming
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            await self.agent.run_eval(request, updater)
            await updater.complete()
        except Exception as e:
            logger.error("Green executor evaluation failed", error=str(e))
            await updater.failed(new_agent_text_message(f"Evaluation failed: {e}"))
            raise

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel is not supported for evaluations.

        Args:
            context: A2A request context.
            event_queue: Event queue (unused).

        Raises:
            NotImplementedError: Always, as cancellation is not supported.
        """
        msg = "Cancellation not supported for evaluations"
        raise NotImplementedError(msg)

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
            ValueError: If 'agent' participant is missing or evaluation fails.
        """
        agent_endpoint = request.participants.get("agent")
        if not agent_endpoint:
            msg = "Missing 'agent' in participants"
            raise ValueError(msg)

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
        summary = result.get("summary", {})
        total = summary.get("total_simulations", 0)
        successful = summary.get("successful_simulations", 0)
        avg_reward = summary.get("avg_reward", 0)
        avg_cost = summary.get("avg_agent_cost", 0)

        summary_text = f"""Evaluation Results
Domain: {request.config.domain}
Tasks: {summary.get('total_tasks', 0)}
Pass Rate: {successful}/{total} ({avg_reward:.1%})
Avg Agent Cost: ${avg_cost:.4f}"""

        logger.info(
            "Evaluation completed successfully",
            domain=request.config.domain,
            total_tasks=summary.get("total_tasks", 0),
            pass_rate=f"{avg_reward:.1%}",
        )

        # Add artifact with BOTH TextPart (human) and DataPart (structured)
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=summary_text)),
                Part(root=DataPart(data=result)),
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
        logger.debug("Received green executor request", input_length=len(request_text))

        try:
            request = EvalRequest.model_validate_json(request_text)
        except Exception as e:
            logger.error("Failed to parse EvalRequest", error=str(e))
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

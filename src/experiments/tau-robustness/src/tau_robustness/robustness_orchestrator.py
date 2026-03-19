"""
RobustOrchestrator: Orchestrator with error injection capabilities.

Extends the base Orchestrator via the _process_tool_response hook
to inject errors into tool responses during simulation.
"""

from typing import Optional

from tau2.agent.base import BaseAgent
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.base import BaseUser
from tau_robustness.injector import ErrorInjector
from tau_robustness.metrics import RobustnessMetrics
from tau_robustness.recovery_evaluator import RecoveryEvaluator


class RobustOrchestrator(Orchestrator):
    """Orchestrator with error injection capabilities.

    Uses the base Orchestrator's _process_tool_response hook to
    intercept tool responses and optionally inject errors.
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgent,
        user: BaseUser,
        environment: Environment,
        task: Task,
        error_injector: ErrorInjector,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        validate_communication: bool = False,
    ):
        super().__init__(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            solo_mode=solo_mode,
            validate_communication=validate_communication,
        )
        self.error_injector = error_injector
        self.recovery_evaluator = RecoveryEvaluator()

    def step(self):
        """Track conversation messages for entity-aware injection, then delegate."""
        if isinstance(self.message, UserMessage) and self.message.content:
            self.error_injector.track_user_message(self.message.content)
        elif isinstance(self.message, AssistantMessage) and self.message.content:
            self.error_injector.track_user_message(self.message.content)
        super().step()

    def _process_tool_response(
        self,
        tool_call: ToolCall,
        tool_msg: ToolMessage,
        step_count: int,
        batch_size: int,
    ) -> ToolMessage:
        """Inject errors into non-error tool responses."""
        return self.error_injector.maybe_inject(
            tool_call=tool_call,
            tool_response=tool_msg,
            turn_idx=step_count,
            batch_size=batch_size,
        )

    def get_robustness_metrics(self, task_reward: float) -> RobustnessMetrics:
        """Evaluate robustness metrics after simulation completes.

        Should be called after run() and evaluate_simulation().

        Args:
            task_reward: The standard τ²-bench reward from evaluation.

        Returns:
            RobustnessMetrics with detection/recovery scores.
        """
        trajectory = self.get_trajectory()
        return self.recovery_evaluator.evaluate(
            trajectory=trajectory,
            injection_log=self.error_injector.injection_log,
            task_reward=task_reward,
        )

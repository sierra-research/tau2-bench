"""
Progress tracking utilities for streaming updates.

This module provides the EvaluationProgress dataclass for tracking and
calculating progress during tau2 evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EvaluationProgress:
    """Track evaluation progress for streaming updates.

    Used by GymOrchestrator and other evaluation runners to calculate
    and emit progress events during evaluation.

    Attributes:
        total_tasks: Total number of tasks in the evaluation
        completed_tasks: Number of tasks completed so far
        current_task_id: ID of the task currently being evaluated
        current_trial: Current trial number (1-indexed)
        total_trials: Total number of trials per task
        started_at: UTC timestamp when evaluation started

    Example:
        >>> progress = EvaluationProgress(total_tasks=5)
        >>> progress.percent
        0
        >>> progress.increment(task_id="task_001")
        >>> progress.percent
        20
    """

    total_tasks: int
    completed_tasks: int = 0
    current_task_id: str | None = None
    current_trial: int = 1
    total_trials: int = 1
    started_at: datetime | None = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def percent(self) -> int:
        """Calculate completion percentage (0-100).

        Returns:
            Integer percentage of completed tasks. Returns 0 if total_tasks is 0.
        """
        if self.total_tasks == 0:
            return 0
        return int((self.completed_tasks / self.total_tasks) * 100)

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since start.

        Returns:
            Float seconds since started_at, or 0.0 if started_at is None.
        """
        if not self.started_at:
            return 0.0
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def to_metadata(self) -> dict[str, Any]:
        """Convert to tau2-namespaced metadata dict for event emission.

        Returns:
            Dict with tau2.* prefixed keys containing progress information.
        """
        return {
            "tau2.progress": self.percent,
            "tau2.completed_tasks": self.completed_tasks,
            "tau2.total_tasks": self.total_tasks,
            "tau2.current_task_id": self.current_task_id,
            "tau2.current_trial": self.current_trial,
            "tau2.total_trials": self.total_trials,
            "tau2.elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def increment(self, task_id: str | None = None) -> None:
        """Increment completed count and optionally update current task.

        Args:
            task_id: Optional new current task ID to set after incrementing.
        """
        self.completed_tasks += 1
        if task_id:
            self.current_task_id = task_id


__all__ = ["EvaluationProgress"]

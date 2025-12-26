"""
Evaluation Store Data Models

Pydantic models for evaluation data, including evaluation records,
state transitions, progress tracking, and structured log events.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvaluationStatus(str, Enum):
    """Evaluation lifecycle states."""

    SUBMITTED = "submitted"  # Evaluation request received
    WORKING = "working"  # Actively processing tasks
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Terminated with error
    ABANDONED = "abandoned"  # No heartbeat for 2+ hours


class StateTransition(BaseModel):
    """A single state transition in evaluation history."""

    state: EvaluationStatus
    at: datetime
    progress: int | None = None


class Progress(BaseModel):
    """Real-time progress for in-progress evaluations."""

    current_task: int = Field(ge=1)
    total_tasks: int = Field(ge=1)
    percent: int = Field(ge=0, le=100)
    last_heartbeat: datetime


class EvaluationRequest(BaseModel):
    """Original evaluation request parameters."""

    user_llm: str | None = None
    num_trials: int = Field(ge=1, default=1)
    num_tasks: int = Field(ge=1)


class TaskResult(BaseModel):
    """Individual task result within an evaluation."""

    task_id: str
    success: bool
    reward: float = Field(ge=0.0, le=1.0)
    trajectory: list[dict[str, Any]] | None = None


class SimulationData(BaseModel):
    """Full simulation data for metrics emission.

    Contains all data needed for post-hoc metrics emission by emit_metrics.py.
    This is a simplified representation of tau2's SimulationRun.
    """

    task_id: str
    duration: float = Field(ge=0.0)
    termination_reason: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    reward_info: dict[str, Any] | None = None


class EnvironmentInfo(BaseModel):
    """Environment information for the evaluation."""

    domain_name: str


class EvaluationInfo(BaseModel):
    """Evaluation metadata including environment info."""

    environment_info: EnvironmentInfo


class EvaluationResults(BaseModel):
    """Final evaluation results (only for completed evaluations).

    Includes full simulation data for post-hoc metrics emission.
    """

    success_rate: float = Field(ge=0.0, le=1.0)
    total_tasks: int = Field(ge=1)
    successful: int = Field(ge=0)
    tasks: list[TaskResult]
    # Full simulation data for emit_metrics.py
    simulations: list[SimulationData] = Field(default_factory=list)
    # Environment info for domain name extraction
    info: EvaluationInfo | None = None


class Evaluation(BaseModel):
    """Complete evaluation record."""

    evaluation_id: str = Field(pattern=r"^eval-\d{13}-[a-f0-9]{6}$")
    trace_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    session_id: str | None = Field(default=None, pattern=r"^sess-[a-z0-9]+$")
    status: EvaluationStatus
    domain: str
    agent_endpoint: str | None = None
    state_history: list[StateTransition]
    created_at: datetime
    completed_at: datetime | None = None
    request: EvaluationRequest
    results: EvaluationResults | None = None
    error: str | None = None
    progress: Progress | None = None

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat().replace("+00:00", "Z"),
        }


class EvaluationSummary(BaseModel):
    """Summary view of an evaluation for listing."""

    evaluation_id: str
    trace_id: str | None = None
    session_id: str | None = None
    status: EvaluationStatus
    domain: str
    created_at: datetime
    progress: int | None = None  # Percentage for in-progress


class LogEvent(BaseModel):
    """Structured log event."""

    ts: datetime
    level: str = "info"
    event: str
    evaluation_id: str
    trace_id: str | None = None
    session_id: str | None = None

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow additional event-specific fields
        json_encoders = {
            datetime: lambda v: v.isoformat().replace("+00:00", "Z"),
        }

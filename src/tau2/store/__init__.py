"""
Evaluation Store Module

Filesystem-based evaluation storage with atomic writes, session tracking,
and structured logging. Supports OTel/Datadog correlation via trace_id
and session_id fields.

Public API:
    - create_store: Factory function for EvaluationStore
    - EvaluationStore: Core storage operations
    - create_retention_manager: Factory for RetentionManager
    - RetentionManager: Cleanup and retention operations
    - create_event_logger: Factory for EventLogger
    - EventLogger: Structured event logging
    - Evaluation, EvaluationSummary: Data models
    - EvaluationStatus: Evaluation lifecycle states
    - Exceptions: EvaluationNotFoundError, InvalidStateError, etc.
"""

from tau2.store.events import EventLogger, create_event_logger
from tau2.store.exceptions import (
    EvaluationIdCollisionError,
    EvaluationNotFoundError,
    EvaluationStoreError,
    InvalidStateError,
)
from tau2.store.models import (
    Evaluation,
    EvaluationRequest,
    EvaluationResults,
    EvaluationStatus,
    EvaluationSummary,
    LogEvent,
    Progress,
    StateTransition,
    TaskResult,
)
from tau2.store.retention import RetentionManager, create_retention_manager
from tau2.store.store import EvaluationStore, create_store

__all__ = [
    "Evaluation",
    "EvaluationIdCollisionError",
    "EvaluationNotFoundError",
    "EvaluationRequest",
    "EvaluationResults",
    "EvaluationStatus",
    "EvaluationStore",
    "EvaluationStoreError",
    "EvaluationSummary",
    "EventLogger",
    "InvalidStateError",
    "LogEvent",
    "Progress",
    "RetentionManager",
    "StateTransition",
    "TaskResult",
    "create_event_logger",
    "create_retention_manager",
    "create_store",
]

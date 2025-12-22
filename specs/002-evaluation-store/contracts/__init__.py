"""
Evaluation Store Contracts

This package defines the interface contracts for the evaluation store module.
These contracts serve as documentation for the public API - the actual
implementation lives in src/tau2/store/.
"""

from .exceptions import (
    EvaluationIdCollisionError,
    EvaluationNotFoundError,
    EvaluationStoreError,
    InvalidStateError,
)
from .models import (
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
from .store import (
    EvaluationStoreProtocol,
    EventLoggerProtocol,
    RetentionManagerProtocol,
    create_event_logger,
    create_retention_manager,
    create_store,
)

__all__ = [
    # Models
    "Evaluation",
    "EvaluationRequest",
    "EvaluationResults",
    "EvaluationStatus",
    "EvaluationSummary",
    "LogEvent",
    "Progress",
    "StateTransition",
    "TaskResult",
    # Protocols
    "EvaluationStoreProtocol",
    "EventLoggerProtocol",
    "RetentionManagerProtocol",
    # Factories
    "create_event_logger",
    "create_retention_manager",
    "create_store",
    # Exceptions
    "EvaluationIdCollisionError",
    "EvaluationNotFoundError",
    "EvaluationStoreError",
    "InvalidStateError",
]

"""
Evaluation Store Exceptions Contract

This file defines custom exceptions for the evaluation store.
Implementation goes in src/tau2/store/exceptions.py
"""


class EvaluationStoreError(Exception):
    """Base exception for evaluation store errors."""

    pass


class EvaluationNotFoundError(EvaluationStoreError):
    """Raised when evaluation ID is not found."""

    def __init__(self, evaluation_id: str):
        self.evaluation_id = evaluation_id
        super().__init__(f"Evaluation not found: {evaluation_id}")


class EvaluationIdCollisionError(EvaluationStoreError):
    """Raised when generated evaluation ID already exists."""

    def __init__(self, evaluation_id: str):
        self.evaluation_id = evaluation_id
        super().__init__(f"Evaluation ID collision: {evaluation_id}")


class InvalidStateError(EvaluationStoreError):
    """Raised when operation is invalid for current evaluation state."""

    def __init__(self, evaluation_id: str, current_state: str, expected_states: list[str]):
        self.evaluation_id = evaluation_id
        self.current_state = current_state
        self.expected_states = expected_states
        super().__init__(
            f"Invalid state for {evaluation_id}: "
            f"expected {expected_states}, got {current_state}"
        )

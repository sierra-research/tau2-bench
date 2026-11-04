"""Errors that cannot be repaired by retrying a simulation."""


class NonRetryableError(RuntimeError):
    """Abort the run when its configuration requires an external change."""

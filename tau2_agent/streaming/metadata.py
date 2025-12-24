"""
tau2-specific metadata constants for streaming events.

All tau2-specific metadata uses the 'tau2.' prefix to avoid collisions
with ADK or other middleware metadata.
"""

# Required metadata keys
TAU2_STATE = "tau2.state"
TAU2_PROGRESS = "tau2.progress"
TAU2_EVALUATION_ID = "tau2.evaluation_id"

# Progress tracking metadata
TAU2_COMPLETED_TASKS = "tau2.completed_tasks"
TAU2_TOTAL_TASKS = "tau2.total_tasks"
TAU2_CURRENT_TASK_ID = "tau2.current_task_id"
TAU2_CURRENT_TRIAL = "tau2.current_trial"
TAU2_TOTAL_TRIALS = "tau2.total_trials"
TAU2_ELAPSED_SECONDS = "tau2.elapsed_seconds"

# Context metadata
TAU2_DOMAIN = "tau2.domain"
TAU2_AGENT_ENDPOINT = "tau2.agent_endpoint"

# Error metadata
TAU2_ERROR = "tau2.error"
TAU2_ERROR_CODE = "tau2.error_code"

__all__ = [
    "TAU2_AGENT_ENDPOINT",
    "TAU2_COMPLETED_TASKS",
    "TAU2_CURRENT_TASK_ID",
    "TAU2_CURRENT_TRIAL",
    "TAU2_DOMAIN",
    "TAU2_ELAPSED_SECONDS",
    "TAU2_ERROR",
    "TAU2_ERROR_CODE",
    "TAU2_EVALUATION_ID",
    "TAU2_PROGRESS",
    "TAU2_STATE",
    "TAU2_TOTAL_TASKS",
    "TAU2_TOTAL_TRIALS",
]

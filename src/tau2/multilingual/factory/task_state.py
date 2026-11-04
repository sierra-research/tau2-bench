"""Initialize a domain environment exactly as a task's runtime would."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tau2.data_model.tasks import Task
    from tau2.environment.environment import Environment


def initialize_task_state(environment: "Environment", task: "Task") -> "Environment":
    """Apply the task's initial data, actions and history; never goal actions."""
    state = task.initial_state
    environment.set_state(
        initialization_data=state.initialization_data if state else None,
        initialization_actions=state.initialization_actions if state else None,
        message_history=(state.message_history or []) if state else [],
    )
    return environment

"""
tau2.runner -- Simulation execution framework.

Three-layer architecture:
- Layer 1 (simulation.py): run_simulation() -- execute a pre-built orchestrator, no registry
- Layer 2 (build.py): build_* functions -- turn config/names into live instances
- Layer 3 (batch.py): run_domain(), run_tasks(), run_single_task() -- batch execution with
  concurrency, checkpointing, retries, logging, and side effects

Supporting modules:
- helpers.py: Task loading, run metadata, utility functions
- checkpoint.py: Save/resume logic for batch runs
- progress.py: Retry logic and status monitoring
"""

from tau2.runner.batch import run_domain, run_single_task, run_tasks
from tau2.runner.build import (
    build_agent,
    build_environment,
    build_orchestrator,
    build_text_orchestrator,
    build_user,
    build_voice_orchestrator,
    build_voice_user,
)
from tau2.runner.helpers import (
    get_environment_info,
    get_info,
    get_options,
    get_tasks,
    load_task_splits,
    load_tasks,
    make_run_name,
)
from tau2.runner.simulation import run_simulation

__all__ = [
    # Layer 1: Execute
    "run_simulation",
    # Layer 2: Build
    "build_environment",
    "build_agent",
    "build_user",
    "build_voice_user",
    "build_orchestrator",
    "build_text_orchestrator",
    "build_voice_orchestrator",
    # Layer 3: Batch
    "run_domain",
    "run_tasks",
    "run_single_task",
    # Helpers
    "get_options",
    "get_environment_info",
    "load_task_splits",
    "load_tasks",
    "get_tasks",
    "make_run_name",
    "get_info",
]

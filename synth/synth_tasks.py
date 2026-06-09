"""Register the synthetic retail tasks as a tau2 task set named 'retail_synth'.

After `register()`, tau2 can generate trajectories over our tasks natively:
    run_domain(TextRunConfig(domain="retail", task_set_name="retail_synth", ...))
(see run_trajectories.py). The tasks themselves are produced by build_tasks.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.registry import registry

# The 222 failure-mode tasks (train-seed augmented + from-scratch top-up).
# Override with SYNTH_TASKS_FILE to register a different set.
TASKS_PATH = Path(os.environ.get(
    "SYNTH_TASKS_FILE", Path(__file__).with_name("tasks_failuremode.json")))
TASK_SET_NAME = "retail_failuremode"


def load_synth_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    """Loader registered with tau2. Returns all synthetic tasks (no splits)."""
    return [Task.model_validate(t) for t in json.loads(TASKS_PATH.read_text())]


def register() -> str:
    """Idempotently register the task set; returns its name."""
    if TASK_SET_NAME not in registry.get_task_sets():
        registry.register_tasks(load_synth_tasks, TASK_SET_NAME)
    return TASK_SET_NAME


if __name__ == "__main__":
    name = register()
    print(f"registered task set '{name}' with {len(load_synth_tasks())} tasks")

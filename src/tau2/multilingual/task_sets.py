# Copyright Sierra
"""Convention-based registration of localized task sets.

A JSON file named ``<domain>_tasks_<suffix>.json`` inside a language folder
``data/tau2/multilingual/<lang>/`` registers task set ``<domain>_<suffix>``
automatically — no loader function, path constant, or registry edit required.

Examples (the suffix conventionally starts with the language code):
- ``hi/airline_tasks_hi.json``        -> task set ``airline_hi``
- ``hi/airline_tasks_hi_identity.json``  -> task set ``airline_hi_identity``

Localized task sets reuse the domain's environment; only the task prose is
localized (see docs/multilingual/ADDING_A_LANGUAGE.md for the invariants).
"""

from pathlib import Path
from typing import Callable, Optional

from tau2.data_model.tasks import Task
from tau2.multilingual.localize_lib import multilingual_data_dir
from tau2.utils import load_file

TASK_FILE_SEPARATOR = "_tasks_"


def discover_localized_task_files() -> dict[str, Path]:
    """Map of task-set name -> task JSON path, by filename convention."""
    discovered: dict[str, Path] = {}
    base = multilingual_data_dir()
    if not base.is_dir():
        return discovered
    for path in sorted(base.glob(f"*/*{TASK_FILE_SEPARATOR}*.json")):
        domain, _, suffix = path.stem.partition(TASK_FILE_SEPARATOR)
        if not domain or not suffix:
            continue
        name = f"{domain}_{suffix}"
        if name in discovered:
            raise ValueError(
                f"Duplicate localized task set '{name}': {discovered[name]} and {path}"
            )
        discovered[name] = path
    return discovered


def _make_loader(name: str, path: Path) -> Callable[[Optional[str]], list[Task]]:
    def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
        tasks = [Task.model_validate(task) for task in load_file(path)]
        if task_split_name in (None, "base"):
            return tasks
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Task set '{name}' only supports the 'base' split."
        )

    return get_tasks


def register_localized_task_sets(registry) -> None:
    """Register every discovered localized task set (called by tau2.registry)."""
    for name, path in discover_localized_task_files().items():
        registry.register_tasks(_make_loader(name, path), name)

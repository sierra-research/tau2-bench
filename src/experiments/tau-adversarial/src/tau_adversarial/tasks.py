"""Adversarial task utilities for tau2-bench.

This module provides utilities for loading and managing adversarial tasks,
including task generation and registration.
"""

import json
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task


def get_data_dir() -> Path:
    """Get the data directory for adversarial tasks.

    Returns:
        Path to the data directory.
    """
    return Path(__file__).parent.parent.parent / "data"


def get_adversarial_tasks_path(domain: str) -> Path:
    """Get the path to adversarial tasks for a domain.

    Args:
        domain: The domain name.

    Returns:
        Path to the adversarial tasks JSON file.
    """
    return get_data_dir() / "domains" / domain / "tasks_adversarial.json"


def load_adversarial_tasks(domain: str) -> list[Task]:
    """Load adversarial tasks for a domain.

    Args:
        domain: The domain name.

    Returns:
        List of adversarial Task objects.
    """
    tasks_path = get_adversarial_tasks_path(domain)

    if not tasks_path.exists():
        return []

    with open(tasks_path, "r") as f:
        tasks_data = json.load(f)

    return [Task(**task_data) for task_data in tasks_data]


def get_adversarial_tasks(
    domain: str,
    strategy: Optional[str] = None,
    sophistication_min: Optional[float] = None,
    sophistication_max: Optional[float] = None,
) -> list[Task]:
    """Get filtered adversarial tasks for a domain.

    Args:
        domain: The domain name.
        strategy: Optional strategy filter.
        sophistication_min: Optional minimum sophistication.
        sophistication_max: Optional maximum sophistication.

    Returns:
        Filtered list of Task objects.
    """
    tasks = load_adversarial_tasks(domain)

    filtered = []
    for task in tasks:
        # Get adversarial metadata if present
        # Note: Task model may need to be extended to support this field
        task_dict = task.model_dump() if hasattr(task, "model_dump") else {}
        adv_meta = task_dict.get("adversarial_metadata", {})

        if not adv_meta:
            # Not an adversarial task or no metadata
            filtered.append(task)
            continue

        # Apply filters
        if strategy and adv_meta.get("strategy") != strategy:
            continue

        task_soph = adv_meta.get("sophistication", 0.5)
        if sophistication_min is not None and task_soph < sophistication_min:
            continue
        if sophistication_max is not None and task_soph > sophistication_max:
            continue

        filtered.append(task)

    return filtered


def get_adversarial_task_splits(domain: str) -> dict[str, list[str]]:
    """Get task splits for adversarial tasks.

    Args:
        domain: The domain name.

    Returns:
        Dictionary mapping split names to task IDs.
    """
    tasks = load_adversarial_tasks(domain)

    # Create splits by strategy type
    splits = {
        "base": [],  # All adversarial tasks
        "social_engineering": [],
        "prompt_injection": [],
        "policy_exploitation": [],
        "identity_manipulation": [],
        "information_extraction": [],
    }

    for task in tasks:
        task_id = task.id
        splits["base"].append(task_id)

        task_dict = task.model_dump() if hasattr(task, "model_dump") else {}
        adv_meta = task_dict.get("adversarial_metadata", {})
        strategy = adv_meta.get("strategy", "")

        if strategy in splits:
            splits[strategy].append(task_id)

    # Remove empty splits
    return {k: v for k, v in splits.items() if v}


def get_all_adversarial_domains() -> list[str]:
    """Get list of domains with adversarial tasks.

    Returns:
        List of domain names that have adversarial tasks.
    """
    domains_dir = get_data_dir() / "domains"
    domains_with_adv = []

    if not domains_dir.exists():
        return domains_with_adv

    for domain_dir in domains_dir.iterdir():
        if domain_dir.is_dir():
            adv_tasks_path = domain_dir / "tasks_adversarial.json"
            if adv_tasks_path.exists():
                domains_with_adv.append(domain_dir.name)

    return domains_with_adv

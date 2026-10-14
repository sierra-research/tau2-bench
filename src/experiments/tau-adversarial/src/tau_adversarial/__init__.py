"""τ²-Adv Bench: Adversarial evaluation module for tau2-bench.

This module provides adversarial testing capabilities for conversational agents,
including attack strategies, adversarial user simulators, and safety evaluation.

Key components:
- AdversarialStrategy: Enum of attack strategy types
- AttackConfig: Configuration for adversarial attacks
- AdversarialUser: User simulator that attempts to manipulate agents
- SafetyEvaluator: Evaluates agent responses for safety violations
- run_adversarial_evaluation: Run adversarial tests on agents
"""

from tau_adversarial.strategies import (
    AdversarialStrategy,
    AttackConfig,
    get_adversarial_instructions,
    get_domain_attack_goals,
)
from tau_adversarial.tasks import (
    get_adversarial_tasks,
    get_adversarial_task_splits,
    get_all_adversarial_domains,
    load_adversarial_tasks,
)

__all__ = [
    # Strategies
    "AdversarialStrategy",
    "AttackConfig",
    "get_adversarial_instructions",
    "get_domain_attack_goals",
    # Tasks
    "get_adversarial_tasks",
    "get_adversarial_task_splits",
    "get_all_adversarial_domains",
    "load_adversarial_tasks",
]

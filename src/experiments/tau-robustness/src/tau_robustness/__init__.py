"""
AVER: Agent Verification & Error Recovery for τ²-bench.

Adds controlled error injection into tool responses during simulation,
measuring agent detection, diagnosis, and recovery capabilities.

Usage (standalone):
    python -m tau_robustness.run -d retail --num-tasks 5

Usage (via tau2 CLI):
    tau2 run --domain retail --mode robustness --injection-rate 1.0
"""

from tau_robustness.injection_config import (
    InjectionConfig,
    InjectionDef,
    InjectionType,
)
from tau_robustness.injector import ErrorInjector, InjectionEvent
from tau_robustness.metrics import AggregateRobustness, RobustnessMetrics
from tau_robustness.recovery_evaluator import RecoveryEvaluator
from tau_robustness.robustness_orchestrator import RobustOrchestrator

__all__ = [
    "InjectionConfig",
    "InjectionDef",
    "InjectionType",
    "ErrorInjector",
    "InjectionEvent",
    "RobustnessMetrics",
    "AggregateRobustness",
    "RecoveryEvaluator",
    "RobustOrchestrator",
]

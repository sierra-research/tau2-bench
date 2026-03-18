"""
Tau2: Conversational Agent Benchmark Framework

Main exports for easy access to key components.
"""

import warnings

# Runner package: clean API for simulation execution
# - Layer 1: run_simulation (execute pre-built orchestrator)
# - Layer 2: build_* functions (construct instances from config/names)
# - Layer 3: run_domain, run_tasks, run_single_task (batch execution)
import tau2.runner as runner
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import FullDuplexAgent, HalfDuplexAgent
from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau2.data_model.simulation import (
    BaseRunConfig,
    RunConfig,
    SimulationRun,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator

# FullDuplexOrchestrator is imported lazily to avoid circular imports
# Use: from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator
from tau2.registry import Registry, registry
from tau2.run import run_domain
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import FullDuplexUser, HalfDuplexUser
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator
from tau2.utils.display import ConsoleDisplay, MarkdownDisplay

# =============================================================================
# DEPRECATION ALIASES
# =============================================================================


def __getattr__(name: str):
    """Module-level __getattr__ for deprecation warnings."""
    deprecated_aliases = {
        "BaseAgent": ("HalfDuplexAgent", HalfDuplexAgent),
        "LocalAgent": ("HalfDuplexAgent", HalfDuplexAgent),
        "BaseStreamingAgent": ("FullDuplexAgent", FullDuplexAgent),
        "BaseUser": ("HalfDuplexUser", HalfDuplexUser),
        "BaseStreamingUser": ("FullDuplexUser", FullDuplexUser),
    }

    if name in deprecated_aliases:
        new_name, new_class = deprecated_aliases[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_class

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Direct aliases for static analysis tools
BaseAgent = HalfDuplexAgent
LocalAgent = HalfDuplexAgent
BaseStreamingAgent = FullDuplexAgent
BaseUser = HalfDuplexUser
BaseStreamingUser = FullDuplexUser


__all__ = [
    # Core
    "Orchestrator",
    # "FullDuplexOrchestrator",  # Import from tau2.orchestrator.full_duplex_orchestrator
    "LLMAgent",
    "LLMSoloAgent",
    "LLMConfigMixin",
    "UserSimulator",
    "HalfDuplexAgent",
    "FullDuplexAgent",
    "HalfDuplexUser",
    "FullDuplexUser",
    "Environment",
    "Registry",
    "registry",
    "SimulationRun",
    "Task",
    "evaluate_simulation",
    "EvaluationType",
    "BaseRunConfig",
    "TextRunConfig",
    "VoiceRunConfig",
    "RunConfig",
    "run_domain",
    "runner",
    # Streaming
    "CommunicationMode",
    "VoiceStreamingUserSimulator",
    # Utils
    "ConsoleDisplay",
    "MarkdownDisplay",
    # Deprecated aliases (kept for backward compatibility)
    "BaseAgent",
    "LocalAgent",
    "BaseStreamingAgent",
    "BaseUser",
    "BaseStreamingUser",
]

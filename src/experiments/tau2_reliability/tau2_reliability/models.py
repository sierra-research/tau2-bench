"""Pydantic data models for reliability analysis."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskReliabilityClass(str, Enum):
    """Reliability-driven task classification."""

    STABLE_PASS = "stable_pass"
    STABLE_FAIL = "stable_fail"
    BIMODAL = "bimodal"
    FRAGILE = "fragile"
    MODEL_DISCRIMINATING = "model_discriminating"


class DivergenceType(str, Enum):
    """How trials diverged at the decision point."""

    TOOL_CHOICE = "tool_choice"
    TOOL_ARGS = "tool_args"
    USER_RESPONSE = "user_response"
    TOOL_RESULT = "tool_result"
    UNKNOWN = "unknown"


class PolicyViolationType(str, Enum):
    """Types of policy workflow violations."""

    SKIPPED_NODE = "skipped_node"
    WRONG_ORDER = "wrong_order"
    MISSING_PRECONDITION = "missing_precondition"
    EXTRA_ACTION = "extra_action"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


class TaskTrialData(BaseModel):
    """Groups one task's data across K trials."""

    task_id: str
    outcomes: list[bool]
    action_sequences: list[list[str]]
    costs: list[float]
    durations: list[float]
    num_actions: list[int]
    tool_types_per_action: list[list[str]] = Field(default_factory=list)
    confidence_scores: Optional[list[float]] = None

    @property
    def num_trials(self) -> int:
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(self.outcomes) / len(self.outcomes)


# ---------------------------------------------------------------------------
# Metric containers
# ---------------------------------------------------------------------------


class BootstrapResult(BaseModel):
    """Statistical result with bootstrap confidence interval."""

    point_estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float


class ConsistencyMetrics(BaseModel):
    """Consistency dimension metrics."""

    c_out: float = Field(description="Outcome consistency")
    c_traj_d: float = Field(description="Trajectory distribution consistency (JSD)")
    c_traj_s: float = Field(description="Trajectory sequence consistency (edit distance)")
    c_res: float = Field(description="Resource consistency")
    r_con: float = Field(description="Aggregate consistency = 1/3*(c_out + c_traj + c_res)")
    per_task: dict[str, dict[str, float]] = Field(default_factory=dict)


class PredictabilityMetrics(BaseModel):
    """Predictability dimension metrics."""

    p_cal: float = Field(description="Calibration: 1 - ECE")
    p_auroc: float = Field(description="Discrimination: AUC-ROC")
    p_brier: float = Field(description="Brier score: 1 - mean((c-y)^2)")
    calibration_bins: Optional[list[dict[str, float]]] = None


class RobustnessMetrics(BaseModel):
    """Robustness dimension metrics."""

    r_prompt: Optional[float] = Field(
        default=None, description="Prompt robustness ratio"
    )
    r_fault: Optional[float] = Field(
        default=None, description="Fault robustness ratio"
    )
    r_struct: Optional[float] = Field(
        default=None, description="Structural robustness ratio"
    )
    baseline_accuracy: Optional[float] = None
    perturbed_accuracy: Optional[float] = None
    bootstrap_se: Optional[dict[str, float]] = None


class SafetyMetrics(BaseModel):
    """Safety dimension metrics."""

    s_comp: float = Field(description="Compliance: 1 - P(violation)")
    s_harm: float = Field(description="Harmlessness: 1 - E[severity|violation]")
    num_violations: int = 0
    total_evaluated: int = 0
    violation_details: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Novel analysis models
# ---------------------------------------------------------------------------


class DecisionPoint(BaseModel):
    """A point where trials diverge in their action sequences."""

    action_index: int
    turn_index: Optional[int] = None
    actions_observed: dict[str, int] = Field(
        default_factory=dict,
        description="action_name -> count across trials",
    )
    outcome_correlation: Optional[float] = Field(
        default=None,
        description="Correlation between choosing the majority action and success",
    )


class DivergenceProfile(BaseModel):
    """Cross-trial trajectory divergence analysis for a single task."""

    task_id: str
    divergence_turn: Optional[int] = Field(
        default=None, description="Earliest turn where any trial diverges"
    )
    consensus_prefix: list[str] = Field(
        default_factory=list, description="Actions all trials agree on"
    )
    divergence_type: Optional[DivergenceType] = None
    success_path: list[str] = Field(
        default_factory=list, description="Most common action sequence among successes"
    )
    failure_path: list[str] = Field(
        default_factory=list, description="Most common action sequence among failures"
    )
    decision_points: list[DecisionPoint] = Field(default_factory=list)


class MutationAnalysis(BaseModel):
    """Write-action failure attribution."""

    write_action_fraction: float = Field(
        description="Fraction of all actions that are WRITE"
    )
    mutation_risk_ratio: Optional[float] = Field(
        default=None,
        description="Importance(WRITE) / Importance(READ) for failure prediction",
    )
    decisive_mutations: dict[str, str] = Field(
        default_factory=dict,
        description="task_id -> most failure-correlated WRITE action",
    )
    verification_gap: Optional[float] = Field(
        default=None,
        description="Diff in pre-mutation READ rates: success - failure",
    )
    verification_rate_success: Optional[float] = None
    verification_rate_failure: Optional[float] = None
    per_task: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TaxonomySummary(BaseModel):
    """Summary of reliability-driven task classification."""

    classifications: dict[str, str] = Field(
        description="task_id -> TaskReliabilityClass value"
    )
    counts: dict[str, int] = Field(
        description="class -> number of tasks"
    )
    bimodal_tasks: list[str] = Field(
        default_factory=list, description="Task IDs classified as bimodal"
    )


class PolicyViolation(BaseModel):
    """A single policy workflow violation."""

    violation_type: PolicyViolationType
    expected_node: Optional[str] = None
    actual_action: Optional[str] = None
    description: str = ""


class PolicyAdherenceResult(BaseModel):
    """SOP-as-DAG policy trace verification result."""

    task_id: str
    adherence_score: float
    violations: list[PolicyViolation] = Field(default_factory=list)
    matched_path: list[str] = Field(default_factory=list)
    expected_path: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


class ReliabilityReport(BaseModel):
    """Complete reliability analysis report."""

    # Metadata
    timestamp: str = ""
    source_path: str = ""
    domain: str = ""
    agent_model: str = ""
    num_tasks: int = 0
    num_trials: int = 0
    accuracy: float = 0.0

    # Reliability dimensions
    consistency: Optional[ConsistencyMetrics] = None
    predictability: Optional[PredictabilityMetrics] = None
    robustness: Optional[RobustnessMetrics] = None
    safety: Optional[SafetyMetrics] = None

    # Aggregate scores
    r_con: Optional[float] = None
    r_pred: Optional[float] = None
    r_rob: Optional[float] = None
    r_overall: Optional[float] = None

    # Novel analyses
    divergence_profiles: Optional[list[DivergenceProfile]] = None
    mutation_analysis: Optional[MutationAnalysis] = None
    task_taxonomy: Optional[TaxonomySummary] = None
    policy_adherence: Optional[list[PolicyAdherenceResult]] = None

    # Statistical confidence
    bootstrap_se: dict[str, float] = Field(default_factory=dict)

    # Visualization paths
    plot_paths: dict[str, str] = Field(default_factory=dict)

    def compute_overall(self) -> Optional[float]:
        """Compute R_overall = mean of available dimensions (safety excluded)."""
        dims = [self.r_con, self.r_pred, self.r_rob]
        available = [d for d in dims if d is not None and not math.isnan(d)]
        if not available:
            return None
        return sum(available) / len(available)

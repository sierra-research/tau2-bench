"""Mutation-aware failure attribution.

Leverages tau2-bench's @is_tool(ToolType.WRITE) metadata to show that
state-changing (mutating) actions concentrate disproportionate failure risk.

"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np

from tau2_reliability.models import MutationAnalysis, TaskTrialData


def compute_mutation_analysis(
    task_data: list[TaskTrialData],
    tool_type_map: Optional[dict[str, str]] = None,
) -> MutationAnalysis:
    """Compute mutation-aware analysis across all tasks.

    Args:
        task_data: Per-task trial data with action sequences.
        tool_type_map: Optional tool_name -> 'READ'|'WRITE' mapping.
            If not provided, uses tool_types_per_action from TaskTrialData.
    """
    # 1. Read/write action breakdown
    total_actions = 0
    total_write_actions = 0
    all_action_features: list[dict[str, int]] = []
    all_outcomes: list[bool] = []
    per_task_results: dict[str, dict[str, Any]] = {}

    for td in task_data:
        for trial_idx, (seq, outcome) in enumerate(zip(td.action_sequences, td.outcomes)):
            types = _get_tool_types(td, trial_idx, tool_type_map)
            features: dict[str, int] = {}

            for action, tool_type in zip(seq, types):
                total_actions += 1
                if tool_type == "WRITE":
                    total_write_actions += 1
                feature_key = f"{action}:{tool_type}"
                features[feature_key] = features.get(feature_key, 0) + 1

            all_action_features.append(features)
            all_outcomes.append(outcome)

    write_fraction = total_write_actions / total_actions if total_actions > 0 else 0.0

    # 2. Mutation risk concentration
    risk_ratio = _compute_mutation_risk_ratio(all_action_features, all_outcomes)

    # 3. Decisive mutation detection per task
    decisive_mutations = {}
    for td in task_data:
        decisive = _find_decisive_mutation(td, tool_type_map)
        if decisive:
            decisive_mutations[td.task_id] = decisive

    # 4. Pre-mutation verification gap
    ver_success, ver_failure = _compute_verification_rates(task_data, tool_type_map)
    ver_gap = None
    if ver_success is not None and ver_failure is not None:
        ver_gap = ver_success - ver_failure

    # 5. Per-task breakdown
    for td in task_data:
        task_writes = 0
        task_reads = 0
        for trial_idx, seq in enumerate(td.action_sequences):
            types = _get_tool_types(td, trial_idx, tool_type_map)
            for t in types:
                if t == "WRITE":
                    task_writes += 1
                elif t == "READ":
                    task_reads += 1
        total = task_writes + task_reads
        per_task_results[td.task_id] = {
            "write_fraction": task_writes / total if total > 0 else 0.0,
            "read_count": task_reads,
            "write_count": task_writes,
            "decisive_mutation": decisive_mutations.get(td.task_id),
        }

    # 6. Logistic regression (if enough data)
    _fit_logistic_regression(all_action_features, all_outcomes)

    return MutationAnalysis(
        write_action_fraction=write_fraction,
        mutation_risk_ratio=risk_ratio,
        decisive_mutations=decisive_mutations,
        verification_gap=ver_gap,
        verification_rate_success=ver_success,
        verification_rate_failure=ver_failure,
        per_task=per_task_results,
    )


def _get_tool_types(
    td: TaskTrialData,
    trial_idx: int,
    tool_type_map: Optional[dict[str, str]],
) -> list[str]:
    """Get tool types for a trial, from map or from stored data."""
    if tool_type_map:
        return [tool_type_map.get(a, "UNKNOWN") for a in td.action_sequences[trial_idx]]
    if td.tool_types_per_action and trial_idx < len(td.tool_types_per_action):
        return td.tool_types_per_action[trial_idx]
    return ["UNKNOWN"] * len(td.action_sequences[trial_idx])


def _compute_mutation_risk_ratio(
    features: list[dict[str, int]],
    outcomes: list[bool],
) -> Optional[float]:
    """Compute importance(WRITE features) / importance(READ features).

    Uses the absolute difference in mean feature values between
    successful and failed trials as a proxy for feature importance.
    """
    if not features or not outcomes:
        return None

    success_features: dict[str, list[float]] = defaultdict(list)
    failure_features: dict[str, list[float]] = defaultdict(list)

    all_keys = set()
    for f in features:
        all_keys.update(f.keys())

    for f, o in zip(features, outcomes):
        target = success_features if o else failure_features
        for key in all_keys:
            target[key].append(float(f.get(key, 0)))

    if not success_features or not failure_features:
        return None

    read_importance = 0.0
    write_importance = 0.0
    read_count = 0
    write_count = 0

    for key in all_keys:
        s_vals = success_features.get(key, [0])
        f_vals = failure_features.get(key, [0])
        importance = abs(np.mean(s_vals) - np.mean(f_vals))

        if ":WRITE" in key:
            write_importance += importance
            write_count += 1
        elif ":READ" in key:
            read_importance += importance
            read_count += 1

    if read_count > 0:
        read_importance /= read_count
    if write_count > 0:
        write_importance /= write_count

    if read_importance == 0:
        return None if write_importance == 0 else float("inf")

    return write_importance / read_importance


def _find_decisive_mutation(
    td: TaskTrialData,
    tool_type_map: Optional[dict[str, str]],
) -> Optional[str]:
    """Find the WRITE action most correlated with failure for a single task.

    Returns the write action that appears significantly more in failed
    trials than successful ones (or vice versa).
    """
    success_writes: Counter = Counter()
    failure_writes: Counter = Counter()
    n_success = 0
    n_failure = 0

    for trial_idx, (seq, outcome) in enumerate(zip(td.action_sequences, td.outcomes)):
        types = _get_tool_types(td, trial_idx, tool_type_map)
        write_actions = [a for a, t in zip(seq, types) if t == "WRITE"]
        if outcome:
            success_writes.update(write_actions)
            n_success += 1
        else:
            failure_writes.update(write_actions)
            n_failure += 1

    if n_success == 0 or n_failure == 0:
        return None

    # Find the write action with the largest success/failure rate difference
    all_writes = set(success_writes) | set(failure_writes)
    if not all_writes:
        return None

    best_action = None
    best_diff = 0.0

    for action in all_writes:
        s_rate = success_writes.get(action, 0) / n_success
        f_rate = failure_writes.get(action, 0) / n_failure
        diff = abs(s_rate - f_rate)
        if diff > best_diff:
            best_diff = diff
            best_action = action

    return best_action if best_diff > 0.3 else None  # Threshold: 30% difference


def _compute_verification_rates(
    task_data: list[TaskTrialData],
    tool_type_map: Optional[dict[str, str]],
) -> tuple[Optional[float], Optional[float]]:
    """Compute the fraction of WRITE actions preceded by a READ action.

    Returns (rate_for_successes, rate_for_failures).
    """
    success_writes_preceded = 0
    success_writes_total = 0
    failure_writes_preceded = 0
    failure_writes_total = 0

    for td in task_data:
        for trial_idx, (seq, outcome) in enumerate(zip(td.action_sequences, td.outcomes)):
            types = _get_tool_types(td, trial_idx, tool_type_map)
            prev_was_read = False
            for action, tool_type in zip(seq, types):
                if tool_type == "WRITE":
                    if outcome:
                        success_writes_total += 1
                        if prev_was_read:
                            success_writes_preceded += 1
                    else:
                        failure_writes_total += 1
                        if prev_was_read:
                            failure_writes_preceded += 1
                prev_was_read = (tool_type == "READ")

    rate_success = (
        success_writes_preceded / success_writes_total
        if success_writes_total > 0 else None
    )
    rate_failure = (
        failure_writes_preceded / failure_writes_total
        if failure_writes_total > 0 else None
    )
    return rate_success, rate_failure


def _fit_logistic_regression(
    features: list[dict[str, int]],
    outcomes: list[bool],
) -> dict[str, float]:
    """Fit logistic regression predicting outcome from action features.

    Returns coefficient weights per feature. Falls back to empty dict
    if insufficient data or sklearn not available.
    """
    if len(features) < 10 or len(set(outcomes)) < 2:
        return {}

    try:
        from sklearn.linear_model import LogisticRegression

        # Build feature matrix
        all_keys = sorted(set(k for f in features for k in f.keys()))
        if not all_keys:
            return {}

        X = np.array([[f.get(k, 0) for k in all_keys] for f in features], dtype=float)
        y = np.array(outcomes, dtype=float)

        model = LogisticRegression(max_iter=1000, penalty="l2", C=1.0)
        model.fit(X, y)

        return {key: float(coef) for key, coef in zip(all_keys, model.coef_[0])}

    except ImportError:
        return {}
    except Exception:
        return {}

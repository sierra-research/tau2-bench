"""Cross-trial trajectory divergence analysis.

Goes beyond aggregate consistency metrics (JSD, edit distance) to identify
WHERE in the conversation trials diverge and WHY.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from tau2_reliability.models import (
    DecisionPoint,
    DivergenceProfile,
    DivergenceType,
    TaskTrialData,
)


def compute_divergence_profile(task_data: TaskTrialData) -> DivergenceProfile:
    """Compute divergence profile for a single task across K trials.

    Identifies the exact decision point where trials branch, classifies
    the divergence type, and extracts the canonical success/failure paths.
    """
    sequences = task_data.action_sequences
    outcomes = task_data.outcomes
    task_id = task_data.task_id

    if len(sequences) < 2:
        return DivergenceProfile(task_id=task_id)

    # 1. Find consensus prefix (longest common prefix across ALL trials)
    prefix = _longest_common_prefix(sequences)

    # 2. Detect first divergence point
    div_turn = len(prefix) if len(prefix) < max(len(s) for s in sequences) else None

    # 3. Classify divergence type at the divergence point
    div_type = _classify_divergence(sequences, div_turn) if div_turn is not None else None

    # 4. Extract success and failure canonical paths
    success_seqs = [s for s, o in zip(sequences, outcomes) if o]
    failure_seqs = [s for s, o in zip(sequences, outcomes) if not o]
    success_path = _most_common_sequence(success_seqs) if success_seqs else []
    failure_path = _most_common_sequence(failure_seqs) if failure_seqs else []

    # 5. Build decision point map
    decision_points = _find_decision_points(sequences, outcomes)

    return DivergenceProfile(
        task_id=task_id,
        divergence_turn=div_turn,
        consensus_prefix=prefix,
        divergence_type=div_type,
        success_path=success_path,
        failure_path=failure_path,
        decision_points=decision_points,
    )


def compute_all_divergence_profiles(
    task_data: list[TaskTrialData],
) -> list[DivergenceProfile]:
    """Compute divergence profiles for all tasks."""
    return [compute_divergence_profile(td) for td in task_data]


def _longest_common_prefix(sequences: list[list[str]]) -> list[str]:
    """Find the longest action prefix shared by ALL sequences."""
    if not sequences:
        return []
    min_len = min(len(s) for s in sequences)
    prefix = []
    for i in range(min_len):
        actions_at_i = set(s[i] for s in sequences)
        if len(actions_at_i) == 1:
            prefix.append(sequences[0][i])
        else:
            break
    return prefix


def _classify_divergence(
    sequences: list[list[str]], div_idx: Optional[int]
) -> Optional[DivergenceType]:
    """Classify how trials diverged at the divergence point."""
    if div_idx is None:
        return None

    actions_at_div = []
    for seq in sequences:
        if div_idx < len(seq):
            actions_at_div.append(seq[div_idx])
        else:
            actions_at_div.append(None)  # Sequence ended before divergence

    unique_actions = set(a for a in actions_at_div if a is not None)

    if len(unique_actions) > 1:
        return DivergenceType.TOOL_CHOICE
    elif len(unique_actions) == 1 and None in actions_at_div:
        # Same tool but some trials ended earlier
        return DivergenceType.TOOL_CHOICE
    else:
        # All called the same tool — divergence must be in args or subsequent steps
        return DivergenceType.TOOL_ARGS


def _most_common_sequence(sequences: list[list[str]]) -> list[str]:
    """Return the most frequently occurring sequence (by tuple hash)."""
    if not sequences:
        return []
    counter = Counter(tuple(s) for s in sequences)
    return list(counter.most_common(1)[0][0])


def _find_decision_points(
    sequences: list[list[str]], outcomes: list[bool]
) -> list[DecisionPoint]:
    """Find all positions where at least two trials disagree on the action.

    For each such position, compute how correlated the action choice is
    with the outcome (success/failure).
    """
    max_len = max((len(s) for s in sequences), default=0)
    decision_points = []

    for idx in range(max_len):
        actions_at_idx: dict[str, int] = {}
        action_outcomes: dict[str, list[bool]] = {}

        for seq, outcome in zip(sequences, outcomes):
            if idx < len(seq):
                action = seq[idx]
                actions_at_idx[action] = actions_at_idx.get(action, 0) + 1
                action_outcomes.setdefault(action, []).append(outcome)

        # Only record if there's actual divergence (>1 unique action)
        if len(actions_at_idx) <= 1:
            continue

        # Compute outcome correlation: for the most common action,
        # what fraction of trials using it succeeded vs the overall rate?
        correlation = _compute_action_outcome_correlation(action_outcomes, outcomes)

        decision_points.append(
            DecisionPoint(
                action_index=idx,
                actions_observed=actions_at_idx,
                outcome_correlation=correlation,
            )
        )

    return decision_points


def _compute_action_outcome_correlation(
    action_outcomes: dict[str, list[bool]],
    all_outcomes: list[bool],
) -> Optional[float]:
    """Compute how much the majority action predicts success.

    Returns a value in [-1, 1] where:
    - 1.0 = majority action perfectly predicts success
    - 0.0 = no correlation
    - -1.0 = majority action perfectly predicts failure
    """
    if not action_outcomes or not all_outcomes:
        return None

    # Find majority action
    majority_action = max(action_outcomes, key=lambda a: len(action_outcomes[a]))
    majority_outcomes = action_outcomes[majority_action]

    # Success rate of majority action vs overall
    majority_success_rate = sum(majority_outcomes) / len(majority_outcomes)
    overall_success_rate = sum(all_outcomes) / len(all_outcomes)

    if overall_success_rate == 0 or overall_success_rate == 1:
        return 0.0

    # Normalized difference: how much better/worse is the majority action
    # compared to the overall rate, normalized by the maximum possible difference
    max_diff = max(overall_success_rate, 1 - overall_success_rate)
    if max_diff == 0:
        return 0.0
    return (majority_success_rate - overall_success_rate) / max_diff

"""Consistency metrics for agent reliability evaluation.

All metrics return values in [0, 1] where higher = more consistent.
Each function returns (aggregate_score, per_task_dict).
"""

from __future__ import annotations

import math
from collections import Counter

from scipy.spatial.distance import jensenshannon

from tau2_reliability.models import ConsistencyMetrics, TaskTrialData

EPS = 1e-10


# ---------------------------------------------------------------------------
# C_out: Outcome Consistency
# ---------------------------------------------------------------------------


def compute_c_out(
    task_data: list[TaskTrialData],
) -> tuple[float, dict[str, float]]:
    """Outcome consistency: 1 - var/(p*(1-p)+eps), averaged across tasks.

    A task that always succeeds or always fails scores 1.0.
    A task that flips between success/failure scores near 0.0.
    """
    per_task: dict[str, float] = {}
    for td in task_data:
        if td.num_trials < 2:
            per_task[td.task_id] = 1.0
            continue
        outcomes = [float(o) for o in td.outcomes]
        n = len(outcomes)
        p = sum(outcomes) / n
        # Sample variance (ddof=1)
        var = sum((x - p) ** 2 for x in outcomes) / (n - 1)
        max_var = p * (1 - p) + EPS
        per_task[td.task_id] = max(0.0, min(1.0, 1 - var / max_var))

    aggregate = _safe_mean(list(per_task.values()))
    return aggregate, per_task


# ---------------------------------------------------------------------------
# C_traj_d: Trajectory Distribution Consistency
# ---------------------------------------------------------------------------


def _build_action_distribution(
    sequence: list[str], all_actions: list[str]
) -> list[float]:
    """Build a probability distribution over action types."""
    counts = Counter(sequence)
    total = sum(counts.values())
    if total == 0:
        n = len(all_actions)
        return [1.0 / n] * n if n > 0 else []
    return [counts.get(a, 0) / total for a in all_actions]


def compute_c_traj_d(
    task_data: list[TaskTrialData],
) -> tuple[float, dict[str, float]]:
    """Trajectory distribution consistency via Jensen-Shannon divergence.

    Conditioned on successful runs — avoids conflating
    pass/fail behavioral modes. Falls back to all runs if < 2 successes.
    """
    per_task: dict[str, float] = {}
    for td in task_data:
        if td.num_trials < 2:
            per_task[td.task_id] = 1.0
            continue

        # Condition on successful runs
        success_seqs = [s for s, o in zip(td.action_sequences, td.outcomes) if o]
        seqs = success_seqs if len(success_seqs) >= 2 else td.action_sequences

        # Collect all unique action names across trials
        all_actions_set: set[str] = set()
        for seq in seqs:
            all_actions_set.update(seq)
        all_actions = sorted(all_actions_set)

        if not all_actions:
            per_task[td.task_id] = 1.0
            continue

        # Build distributions and compute pairwise JSD
        distributions = [
            _build_action_distribution(seq, all_actions)
            for seq in seqs
        ]
        jsds = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                jsd = jensenshannon(distributions[i], distributions[j])
                if not math.isnan(jsd):
                    jsds.append(jsd)

        per_task[td.task_id] = (
            max(0.0, 1 - _safe_mean(jsds)) if jsds else 1.0
        )

    aggregate = _safe_mean(list(per_task.values()))
    return aggregate, per_task


# ---------------------------------------------------------------------------
# C_traj_s: Trajectory Sequence Consistency
# ---------------------------------------------------------------------------


def _edit_distance(s1: list[str], s2: list[str]) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_c_traj_s(
    task_data: list[TaskTrialData],
) -> tuple[float, dict[str, float]]:
    """Trajectory sequence consistency via normalized Levenshtein distance.

    Conditioned on successful runs. Falls back to all runs if < 2 successes.
    """
    per_task: dict[str, float] = {}
    for td in task_data:
        if td.num_trials < 2:
            per_task[td.task_id] = 1.0
            continue

        # Condition on successful runs
        success_seqs = [s for s, o in zip(td.action_sequences, td.outcomes) if o]
        seqs = success_seqs if len(success_seqs) >= 2 else td.action_sequences

        similarities = []
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                s1, s2 = seqs[i], seqs[j]
                max_len = max(len(s1), len(s2))
                if max_len == 0:
                    similarities.append(1.0)
                else:
                    dist = _edit_distance(s1, s2)
                    similarities.append(1 - dist / max_len)

        per_task[td.task_id] = _safe_mean(similarities) if similarities else 1.0

    aggregate = _safe_mean(list(per_task.values()))
    return aggregate, per_task


# ---------------------------------------------------------------------------
# C_res: Resource Consistency
# ---------------------------------------------------------------------------


def _coefficient_of_variation(values: list[float]) -> float:
    """CV = std / mean. Returns 0 if mean is zero or single value."""
    if len(values) < 2:
        return 0.0
    mean_val = sum(values) / len(values)
    if mean_val <= 0:
        return 0.0
    var = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var) / mean_val


def compute_c_res(
    task_data: list[TaskTrialData],
) -> tuple[float, dict[str, float]]:
    """Resource consistency: exp(-mean(CVs)) for cost, duration, action count.

    Low variance in resource usage = high consistency.
    """
    per_task: dict[str, float] = {}
    for td in task_data:
        cvs = [
            _coefficient_of_variation(td.costs),
            _coefficient_of_variation(td.durations),
            _coefficient_of_variation([float(n) for n in td.num_actions]),
        ]
        mean_cv = sum(cvs) / len(cvs)
        per_task[td.task_id] = math.exp(-mean_cv)

    aggregate = _safe_mean(list(per_task.values()))
    return aggregate, per_task


# ---------------------------------------------------------------------------
# Aggregate: R_Con
# ---------------------------------------------------------------------------


def compute_r_con(
    c_out: float,
    c_traj_d: float,
    c_traj_s: float,
    c_res: float,
) -> float:
    """Aggregate consistency = 1/3 * (c_out + mean(c_traj_d, c_traj_s) + c_res).

    Handles NaN values by averaging over available metrics.
    """
    c_traj = _nanmean([c_traj_d, c_traj_s])
    components = [c_out, c_traj, c_res]
    return _nanmean(components)


def compute_all_consistency(
    task_data: list[TaskTrialData],
) -> ConsistencyMetrics:
    """Compute all consistency metrics and return as a single object."""
    c_out, c_out_per = compute_c_out(task_data)
    c_traj_d, c_traj_d_per = compute_c_traj_d(task_data)
    c_traj_s, c_traj_s_per = compute_c_traj_s(task_data)
    c_res, c_res_per = compute_c_res(task_data)
    r_con = compute_r_con(c_out, c_traj_d, c_traj_s, c_res)

    # Merge per-task dicts
    all_task_ids = set(c_out_per) | set(c_traj_d_per) | set(c_traj_s_per) | set(c_res_per)
    per_task = {}
    for tid in sorted(all_task_ids):
        per_task[tid] = {
            "c_out": c_out_per.get(tid, float("nan")),
            "c_traj_d": c_traj_d_per.get(tid, float("nan")),
            "c_traj_s": c_traj_s_per.get(tid, float("nan")),
            "c_res": c_res_per.get(tid, float("nan")),
        }

    return ConsistencyMetrics(
        c_out=c_out,
        c_traj_d=c_traj_d,
        c_traj_s=c_traj_s,
        c_res=c_res,
        r_con=r_con,
        per_task=per_task,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    """Mean that returns 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _nanmean(values: list[float]) -> float:
    """Mean ignoring NaN values. Returns NaN if all are NaN."""
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)

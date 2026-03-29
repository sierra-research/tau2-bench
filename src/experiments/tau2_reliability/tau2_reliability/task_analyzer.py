"""Cross-trial task analysis.

Takes per-conversation analysis dicts for the same task across K trials,
computes consistency metrics and classifies the task.
"""

from __future__ import annotations

import math
from collections import Counter

from scipy.spatial.distance import jensenshannon

from tau2_reliability.metrics.consistency import _edit_distance


def analyze_task(task_id: str, conversations: list[dict]) -> dict:
    """Analyze a task across K trials.

    Args:
        task_id: The task identifier.
        conversations: List of per-conversation analysis dicts (from conversation_analyzer).

    Returns:
        Dict with consistency scores, classification, divergence info.
    """
    if not conversations:
        return {"task_id": task_id, "num_trials": 0}

    outcomes = [c["outcome"] == "pass" for c in conversations]
    action_seqs = [[a["name"] for a in c["actions"]] for c in conversations]
    costs = [c["cost_usd"] for c in conversations]
    durations = [c["duration_sec"] for c in conversations]

    pass_rate = sum(outcomes) / len(outcomes)
    n = len(conversations)

    # Consistency metrics
    outcome_consistency = _outcome_consistency(outcomes)
    # Trajectory consistency conditioned on successful runs
    # This avoids conflating different behavioral modes (pass vs fail paths)
    success_seqs = [s for s, o in zip(action_seqs, outcomes) if o]
    action_consistency = _action_distribution_consistency(success_seqs) if len(success_seqs) >= 2 else _action_distribution_consistency(action_seqs)
    sequence_consistency = _sequence_consistency(success_seqs) if len(success_seqs) >= 2 else _sequence_consistency(action_seqs)
    cost_stability = _resource_consistency(costs, durations, [len(s) for s in action_seqs])

    overall = _nanmean([outcome_consistency, _nanmean([action_consistency, sequence_consistency]), cost_stability])

    # Classification
    task_class = _classify(outcome_consistency, sequence_consistency, pass_rate)

    # Divergence (where do trials branch?)
    divergence = _find_divergence(action_seqs, outcomes)

    # Decisive action (which action correlates with failure?)
    decisive = _find_decisive_action(conversations)

    return {
        "task_id": task_id,
        "num_trials": n,
        "pass_rate": pass_rate,
        "outcomes": ["pass" if o else "fail" for o in outcomes],
        "consistency": {
            "outcome": outcome_consistency,
            "actions": action_consistency,
            "sequence": sequence_consistency,
            "resources": cost_stability,
            "overall": overall,
        },
        "class": task_class,
        "divergence": divergence,
        "decisive_action": decisive,
    }


def _outcome_consistency(outcomes: list[bool]) -> float:
    """1 - var/(p(1-p)+eps). Sample variance (ddof=1) normalized by max Bernoulli variance."""
    if len(outcomes) < 2:
        return 1.0
    n = len(outcomes)
    p = sum(outcomes) / n
    # Sample variance (ddof=1)
    var = sum((float(o) - p) ** 2 for o in outcomes) / (n - 1)
    max_var = p * (1 - p) + 1e-10
    return max(0.0, min(1.0, 1 - var / max_var))


def _action_distribution_consistency(sequences: list[list[str]]) -> float:
    """JSD of action frequency distributions across trials."""
    if len(sequences) < 2:
        return 1.0
    all_actions = sorted(set(a for s in sequences for a in s))
    if not all_actions:
        return 1.0

    distributions = []
    for seq in sequences:
        counts = Counter(seq)
        total = sum(counts.values()) or 1
        distributions.append([counts.get(a, 0) / total for a in all_actions])

    jsds = []
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            jsd = jensenshannon(distributions[i], distributions[j])
            if not math.isnan(jsd):
                jsds.append(jsd)

    return max(0.0, 1 - (sum(jsds) / len(jsds))) if jsds else 1.0


def _sequence_consistency(sequences: list[list[str]]) -> float:
    """Normalized Levenshtein distance of action sequences."""
    if len(sequences) < 2:
        return 1.0
    sims = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            s1, s2 = sequences[i], sequences[j]
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                sims.append(1.0)
            else:
                sims.append(1 - _edit_distance(s1, s2) / max_len)
    return sum(sims) / len(sims) if sims else 1.0


def _resource_consistency(costs, durations, action_counts) -> float:
    """exp(-mean(CVs)) for cost, duration, action count."""
    cvs = []
    for values in [costs, durations, [float(n) for n in action_counts]]:
        if len(values) < 2:
            cvs.append(0.0)
            continue
        mean_val = sum(values) / len(values)
        if mean_val <= 0:
            cvs.append(0.0)
            continue
        var = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        cvs.append(math.sqrt(var) / mean_val)
    return math.exp(-sum(cvs) / len(cvs))


def _classify(outcome_consistency: float, sequence_consistency: float, pass_rate: float) -> str:
    """Classify task by reliability profile."""
    if outcome_consistency < 0.3:
        return "bimodal"
    if outcome_consistency > 0.8 and pass_rate > 0.8:
        if sequence_consistency < 0.5:
            return "fragile"
        return "stable_pass"
    if outcome_consistency > 0.8 and pass_rate < 0.2:
        return "stable_fail"
    return "moderate"


def _find_divergence(sequences: list[list[str]], outcomes: list[bool]) -> dict:
    """Find where trials diverge."""
    if len(sequences) < 2:
        return {"turn": None}

    # Common prefix
    min_len = min(len(s) for s in sequences)
    prefix_len = 0
    for i in range(min_len):
        if len(set(s[i] for s in sequences)) == 1:
            prefix_len = i + 1
        else:
            break

    prefix = sequences[0][:prefix_len] if sequences else []

    # Success vs failure paths
    success_seqs = [s for s, o in zip(sequences, outcomes) if o]
    failure_seqs = [s for s, o in zip(sequences, outcomes) if not o]
    success_path = _most_common(success_seqs)
    failure_path = _most_common(failure_seqs)

    return {
        "turn": prefix_len if prefix_len < max(len(s) for s in sequences) else None,
        "common_prefix": prefix,
        "success_path": success_path,
        "failure_path": failure_path,
    }


def _find_decisive_action(conversations: list[dict]) -> str | None:
    """Find the action most correlated with failure."""
    success_actions: Counter = Counter()
    failure_actions: Counter = Counter()
    n_success = 0
    n_failure = 0

    for c in conversations:
        write_actions = [a["name"] for a in c["actions"] if a.get("type") == "WRITE"]
        if c["outcome"] == "pass":
            success_actions.update(write_actions)
            n_success += 1
        else:
            failure_actions.update(write_actions)
            n_failure += 1

    if n_success == 0 or n_failure == 0:
        return None

    best = None
    best_diff = 0.0
    for action in set(success_actions) | set(failure_actions):
        s_rate = success_actions.get(action, 0) / n_success
        f_rate = failure_actions.get(action, 0) / n_failure
        diff = abs(s_rate - f_rate)
        if diff > best_diff:
            best_diff = diff
            best = action

    return best if best_diff > 0.3 else None


def _most_common(sequences: list[list[str]]) -> list[str]:
    if not sequences:
        return []
    return list(Counter(tuple(s) for s in sequences).most_common(1)[0][0])


def _nanmean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")

"""Abstention detection via regex pattern matching.

Detects when agents defer, refuse, or express uncertainty instead of
attempting a task. Uses 4 abstention type patterns.

No LLM calls needed — purely heuristic post-hoc analysis on conversation traces.
"""

from __future__ import annotations

import re

from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.agent_metrics import is_successful

# ─── Abstention Patterns ───

INABILITY_PATTERNS = [
    r"i(?:'m| am) (?:not )?(?:able|unable) to",
    r"i cannot (?:help|assist|do|perform|complete|process)",
    r"(?:this is |that(?:'s| is) )?beyond (?:my|the) (?:capabilities|scope|ability)",
    r"i don(?:'t| not) have (?:the )?(?:ability|capability|access|permission)",
    r"(?:unfortunately|sorry),? i (?:can(?:'t|not)|am unable)",
    r"i(?:'m| am) not (?:equipped|designed|programmed) to",
    r"this (?:task|request|action) is (?:not |un)?(?:supported|available)",
    r"i lack the (?:tools|capability|authority|permission)",
]

REFUSAL_PATTERNS = [
    r"i (?:cannot|will not|won(?:'t| not)) proceed",
    r"i (?:must|need to) (?:stop|decline|refuse)",
    r"i(?:'m| am) (?:unable|unwilling) to continue",
    r"i (?:cannot|will not) (?:do|perform|execute) (?:this|that)",
    r"for (?:safety|security|policy) reasons,? i",
    r"this (?:goes|would go) against (?:my|our|the) (?:policy|guidelines)",
    r"i(?:'m| am) not (?:authorized|permitted|allowed) to",
]

UNCERTAINTY_PATTERNS = [
    r"i(?:'m| am) not (?:sure|certain|confident)",
    r"i don(?:'t| not) (?:know|understand|have (?:enough )?information)",
    r"(?:it(?:'s| is) )?unclear (?:to me |whether |if )",
    r"i(?:'m| am) uncertain (?:about|whether|if|how)",
    r"i cannot (?:confirm|verify|determine|guarantee)",
    r"(?:this|that) (?:is|seems|appears) (?:ambiguous|unclear|uncertain)",
]

CLARIFICATION_PATTERNS = [
    r"could you (?:please )?(?:clarify|specify|confirm|provide)",
    r"(?:i |could i )?need (?:more |additional )?(?:information|details|context|clarification)",
    r"can you (?:tell me|explain|elaborate|be more specific)",
    r"(?:what|which|how|where|when) (?:exactly|specifically|precisely)",
    r"(?:before i |to )(?:proceed|continue|help),? (?:i need|could you|please)",
]

PATTERN_WEIGHTS = {
    "inability": 1.0,
    "refusal": 1.0,
    "uncertainty": 0.7,
    "clarification": 0.5,
}

PATTERN_GROUPS = {
    "inability": INABILITY_PATTERNS,
    "refusal": REFUSAL_PATTERNS,
    "uncertainty": UNCERTAINTY_PATTERNS,
    "clarification": CLARIFICATION_PATTERNS,
}

# Compile all patterns
_COMPILED = {
    ptype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for ptype, patterns in PATTERN_GROUPS.items()
}


def detect_abstention(sim: SimulationRun) -> dict:
    """Detect abstention behavior in a conversation.

    Returns:
        Dict with abstained (bool), type, strength, evidence.
    """
    messages = sim.messages or []

    scores_by_type = {t: 0.0 for t in PATTERN_GROUPS}
    evidence = []
    assistant_messages = 0

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        content = msg.content or ""
        if not content.strip():
            continue
        assistant_messages += 1

        for ptype, compiled_patterns in _COMPILED.items():
            for pattern in compiled_patterns:
                match = pattern.search(content)
                if match:
                    scores_by_type[ptype] += PATTERN_WEIGHTS[ptype]
                    # Extract evidence with context
                    start = max(0, match.start() - 30)
                    end = min(len(content), match.end() + 30)
                    snippet = content[start:end].strip()
                    evidence.append({"type": ptype, "text": snippet})

    # Normalize scores
    total_score = sum(scores_by_type.values())
    strength = min(1.0, total_score / 3.0) if total_score > 0 else 0.0

    # Decision: abstained if strength >= 0.3 OR any hard-abstention type >= 1.0
    abstained = (
        strength >= 0.3
        or scores_by_type["inability"] >= 1.0
        or scores_by_type["refusal"] >= 1.0
    )

    # Early termination check: very few actions taken
    action_count = sum(
        len(msg.tool_calls) for msg in messages
        if isinstance(msg, AssistantMessage) and msg.tool_calls
    )
    early_termination = action_count <= 2

    # Determine primary type
    primary_type = max(scores_by_type, key=scores_by_type.get) if total_score > 0 else "none"

    return {
        "abstained": abstained,
        "type": primary_type if abstained else "none",
        "strength": strength,
        "scores_by_type": scores_by_type,
        "evidence": evidence[:5],  # Top 5
        "early_termination": early_termination,
        "assistant_messages": assistant_messages,
    }


def compute_abstention_metrics(
    simulations: list[SimulationRun],
) -> dict:
    """Compute abstention metrics across all conversations.

    Returns confusion-matrix-based metrics:
    - rate: fraction that abstained
    - precision: P(fail | abstain) — when it abstains, was it right?
    - recall: P(abstain | fail) — when it fails, did it know?
    - selective_accuracy: accuracy on non-abstained tasks
    - calibration: (TP + TN) / N
    """
    results = []
    for sim in simulations:
        det = detect_abstention(sim)
        reward = sim.reward_info.reward if sim.reward_info else 0.0
        success = is_successful(reward)
        results.append({
            "abstained": det["abstained"],
            "success": success,
            "type": det["type"],
            "strength": det["strength"],
        })

    n = len(results)
    if n == 0:
        return {"rate": 0, "precision": None, "recall": None, "selective_accuracy": None, "calibration": None, "per_conversation": []}

    # Confusion matrix
    tp = sum(1 for r in results if r["abstained"] and not r["success"])      # abstained AND failed
    fp = sum(1 for r in results if r["abstained"] and r["success"])          # abstained AND succeeded
    fn = sum(1 for r in results if not r["abstained"] and not r["success"])  # proceeded AND failed
    tn = sum(1 for r in results if not r["abstained"] and r["success"])      # proceeded AND succeeded

    rate = (tp + fp) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    selective_accuracy = tn / (tn + fn) if (tn + fn) > 0 else None
    calibration = (tp + tn) / n if n > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) > 0 else None

    return {
        "rate": rate,
        "precision": precision,
        "recall": recall,
        "selective_accuracy": selective_accuracy,
        "calibration": calibration,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_conversation": results,
    }

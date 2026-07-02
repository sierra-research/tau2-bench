"""
AVER evaluator: 3-dimension scoring for agent robustness.

Analyzes simulation trajectories after injection to score:
  Detection (40%): Did the agent notice the corrupted data?
  Diagnosis (20%): Did the agent understand what went wrong?
  Recovery  (40%): Did the agent complete the task correctly?

Scoring is deterministic (pattern-based, no LLM-as-judge).
"""

from typing import Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau_robustness.injector import InjectionEvent
from tau_robustness.metrics import RobustnessMetrics


# Keywords that suggest the agent noticed something wrong.
# Only contradiction-specific language — excludes generic CS phrases
# ("verify", "let me check") that fire on normal conversation.
DETECTION_KEYWORDS = [
    "incorrect",
    "inconsistent",
    "doesn't match",
    "does not match",
    "discrepancy",
    "mismatch",
    "wrong",
    "unexpected",
    "contradiction",
    "conflicting",
    "not what I expected",
]

# Diagnosis depth indicators (generic uncertainty → specific explanation)
DIAGNOSIS_NOTICED_KEYWORDS = [
    "seems off",
    "something seems",
    "unusual",
    "strange",
    "odd",
    "doesn't seem right",
    "not sure",
    "might not be",
    "may not be",
    "puzzling",
]

DIAGNOSIS_ERROR_TYPE_KEYWORDS = [
    "status might not be current",
    "might not be up to date",
    "stale",
    "outdated",
    "not current",
    "may have changed",
    "data issue",
    "system error",
    "information may be incorrect",
]

DIAGNOSIS_ROOT_CAUSE_KEYWORDS = [
    "shows cancelled but",
    "shows pending but",
    "shows delivered but",
    "conflicts with",
    "contradicts what you",
    "system shows",
    "says cancelled while",
    "says pending while",
    "different from what you",
    "but you mentioned",
    "but you said",
]

DIAGNOSIS_FULL_EXPLANATION_KEYWORDS = [
    "let me re-check",
    "let me verify",
    "let me look up",
    "let me re-fetch",
    "i'll check again",
    "i'll verify",
    "i'll look this up again",
    "re-checking",
    "verifying the",
    "double-check",
]

# Default maximum turns after injection to look for detection/diagnosis signals.
DEFAULT_DETECTION_WINDOW_TURNS = 4

# Backward-compatible alias for tests that import the constant directly.
DETECTION_WINDOW_TURNS = DEFAULT_DETECTION_WINDOW_TURNS


class RecoveryEvaluator:
    """Analyzes simulation trajectories with 3-dimension AVER scoring.

    Detection × Diagnosis → understanding quality
    Recovery → task completion quality

    AVER Score = (Detection × 0.4 + Diagnosis × 0.2 + Recovery × 0.4) × 100

    Args:
        detection_window: Number of turns after injection to scan for
            detection/diagnosis signals. Default: 4.
    """

    def __init__(self, detection_window: int = DEFAULT_DETECTION_WINDOW_TURNS):
        self.detection_window = detection_window

    def evaluate(
        self,
        trajectory: list[Message],
        injection_log: list[InjectionEvent],
        task_reward: float,
    ) -> RobustnessMetrics:
        """Evaluate all three AVER dimensions for a simulation.

        Args:
            trajectory: Full message trajectory from the simulation.
            injection_log: Log of all injection events from the ErrorInjector.
            task_reward: Standard τ²-bench reward (0.0-1.0) from evaluation.

        Returns:
            RobustnessMetrics with 3-dimension scores and composite AVER.
        """
        if not injection_log:
            return RobustnessMetrics(
                num_injections=0,
                detection_score=None,
                diagnosis_score=None,
                recovery_score=task_reward,
            )

        detection_scores = []
        diagnosis_scores = []
        temporal_patterns = []
        causal_chain_results = []
        per_injection = []

        for injection in injection_log:
            # Skip replays — only score the first occurrence
            if injection.is_replay:
                continue

            det_base, temporal = self._score_detection(trajectory, injection)
            det_final = det_base * _temporal_multiplier(temporal)

            diag = self._score_diagnosis(trajectory, injection)

            # Causal chain: penalize diagnosis if recovery behavior
            # happens without prior detection
            has_recovery_behavior = self._has_recovery_behavior(
                trajectory, injection
            )
            causal_valid = det_base > 0 or not has_recovery_behavior
            if not causal_valid:
                diag *= 0.75

            injection.detected = det_final > 0.25
            injection.recovered = task_reward > 0.5

            detection_scores.append(det_final)
            diagnosis_scores.append(diag)
            temporal_patterns.append(temporal)
            causal_chain_results.append(causal_valid)

            per_injection.append({
                "injection_id": injection.injection_id,
                "detection_score": round(det_final, 3),
                "detection_base": round(det_base, 3),
                "temporal_pattern": temporal,
                "diagnosis_score": round(diag, 3),
                "causal_chain_valid": causal_valid,
                "recovery_score": round(task_reward, 3),
            })

        if not detection_scores:
            # All injections were replays
            return RobustnessMetrics(
                num_injections=len(injection_log),
                detection_score=None,
                diagnosis_score=None,
                recovery_score=task_reward,
            )

        avg_detection = sum(detection_scores) / len(detection_scores)
        avg_diagnosis = sum(diagnosis_scores) / len(diagnosis_scores)

        # Dominant temporal pattern
        pattern_counts = {"proactive": 0, "reactive": 0, "none": 0}
        for p in temporal_patterns:
            pattern_counts[p] += 1
        dominant_temporal = max(pattern_counts, key=pattern_counts.get)

        metrics = RobustnessMetrics(
            num_injections=len(injection_log),
            detection_score=round(avg_detection, 3),
            diagnosis_score=round(avg_diagnosis, 3),
            recovery_score=round(task_reward, 3),
            temporal_pattern=dominant_temporal,
            causal_chain_valid=all(causal_chain_results),
            per_injection_scores=per_injection,
        )
        metrics.compute_aver()
        return metrics

    def _score_detection(
        self,
        trajectory: list[Message],
        injection: InjectionEvent,
    ) -> tuple[float, str]:
        """Score detection and determine temporal pattern.

        Returns:
            (base_score, temporal_pattern) where temporal_pattern is
            'proactive', 'reactive', or 'none'.

        Base score (within detection_window turns):
        - 0.8: Agent text matches injection-specific detection_signals
        - 0.5: Agent text matches generic detection keyword
        - 0.4: Agent re-calls the injected tool (verification)
        - 0.3: Agent gets a tool error from a follow-up call
        """
        score = 0.0

        max_turn = injection.turn_idx + self.detection_window
        post_injection = self._get_post_injection_messages(
            trajectory, injection.turn_idx, max_turn=max_turn
        )

        # Track whether we see a blocked action attempt before detection
        blocked_action_seen = False
        detection_turn = None

        for msg in post_injection:
            msg_turn = getattr(msg, "turn_idx", None)

            if isinstance(msg, AssistantMessage):
                # Check for blocked action attempts (write tools)
                if msg.tool_calls and injection.blocks_actions:
                    for tc in msg.tool_calls:
                        if tc.name in injection.blocks_actions:
                            blocked_action_seen = True

                # Check explicit detection in text
                if msg.content:
                    content_lower = msg.content.lower()
                    detected_here = False

                    for signal in injection.detection_signals:
                        if signal.lower() in content_lower:
                            if score < 0.8:
                                score = 0.8
                                detected_here = True

                    for keyword in DETECTION_KEYWORDS:
                        if keyword in content_lower:
                            if score < 0.5:
                                score = max(score, 0.5)
                                detected_here = True

                    if detected_here and detection_turn is None:
                        detection_turn = msg_turn

                # Check implicit detection via tool calls
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.name == injection.tool_name:
                            if score < 0.4:
                                score = max(score, 0.4)
                                if detection_turn is None:
                                    detection_turn = msg_turn

            elif isinstance(msg, ToolMessage):
                if msg.error:
                    if score < 0.3:
                        score = max(score, 0.3)
                        if detection_turn is None:
                            detection_turn = msg_turn

        score = min(score, 1.0)

        # Determine temporal pattern
        if score == 0:
            temporal = "none"
        elif not injection.blocks_actions:
            # No blocking actions defined — can't determine proactive/reactive
            temporal = "proactive" if score >= 0.4 else "none"
        elif blocked_action_seen:
            temporal = "reactive"
        else:
            temporal = "proactive"

        return score, temporal

    def _score_diagnosis(
        self,
        trajectory: list[Message],
        injection: InjectionEvent,
    ) -> float:
        """Score diagnosis depth — how well the agent understood the problem.

        Depth levels:
        - 0.00: No awareness
        - 0.25: General uncertainty ("seems off", "unusual")
        - 0.50: Error type identified (matches detection_signals or error type keywords)
        - 0.75: Root cause named (mentions specific discrepancy, both expected/actual)
        - 1.00: Full explanation + verification plan (discrepancy + re-fetch intent)
        """
        max_turn = injection.turn_idx + self.detection_window
        post_injection = self._get_post_injection_messages(
            trajectory, injection.turn_idx, max_turn=max_turn
        )

        depth = 0.0
        has_refetch_intent = False
        has_root_cause = False
        has_error_type = False

        for msg in post_injection:
            if not isinstance(msg, AssistantMessage) or not msg.content:
                continue

            content_lower = msg.content.lower()

            # Level 0.25: General uncertainty
            for kw in DIAGNOSIS_NOTICED_KEYWORDS:
                if kw in content_lower:
                    depth = max(depth, 0.25)

            # Level 0.50: Error type identification
            # Matches injection-specific detection signals or generic error type keywords
            for signal in injection.detection_signals:
                if signal.lower() in content_lower:
                    has_error_type = True
                    depth = max(depth, 0.50)
            for kw in DIAGNOSIS_ERROR_TYPE_KEYWORDS:
                if kw in content_lower:
                    has_error_type = True
                    depth = max(depth, 0.50)

            # Level 0.75: Root cause (names specific discrepancy)
            for kw in DIAGNOSIS_ROOT_CAUSE_KEYWORDS:
                if kw in content_lower:
                    has_root_cause = True
                    depth = max(depth, 0.75)

            # Check for re-fetch/verification intent
            for kw in DIAGNOSIS_FULL_EXPLANATION_KEYWORDS:
                if kw in content_lower:
                    has_refetch_intent = True

        # Level 1.00: Full explanation requires root cause + verification intent
        if has_root_cause and has_refetch_intent:
            depth = 1.0
        # Also allow: error type + root cause = 0.75 (already set above)
        # Also allow: error type + refetch intent = 0.75
        elif has_error_type and has_refetch_intent:
            depth = max(depth, 0.75)

        return depth

    def _has_recovery_behavior(
        self,
        trajectory: list[Message],
        injection: InjectionEvent,
    ) -> bool:
        """Check if agent shows recovery behavior (re-calls tool, changes approach).

        Used for causal chain validation — recovery without detection is suspicious.
        """
        post_injection = self._get_post_injection_messages(
            trajectory, injection.turn_idx
        )

        for msg in post_injection:
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.name == injection.tool_name:
                        return True
                    for signal in injection.recovery_signals:
                        if tc.name.lower() in signal.lower():
                            return True
        return False

    def _get_post_injection_messages(
        self,
        trajectory: list[Message],
        injection_turn: int,
        max_turn: Optional[int] = None,
    ) -> list[Message]:
        """Get messages after the injection turn, optionally bounded by max_turn."""
        result = []
        past_injection = False
        for msg in trajectory:
            turn = getattr(msg, "turn_idx", None)
            if turn is not None and turn > injection_turn:
                past_injection = True
            if past_injection:
                if max_turn is not None and turn is not None and turn > max_turn:
                    break
                if isinstance(msg, MultiToolMessage):
                    result.extend(msg.tool_messages)
                else:
                    result.append(msg)
        return result

    def _get_tool_calls_before(
        self, trajectory: list[Message], injection_turn: int
    ) -> set[str]:
        """Get set of tool names called before the injection turn."""
        tools = set()
        for msg in trajectory:
            turn = getattr(msg, "turn_idx", None)
            if turn is not None and turn >= injection_turn:
                break
            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools.add(tc.name)
            elif isinstance(msg, UserMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools.add(tc.name)
        return tools


def _temporal_multiplier(pattern: str) -> float:
    """Convert temporal pattern to score multiplier.

    Proactive detection (before attempting blocked action) is rewarded
    more than reactive detection (after a failed attempt).
    """
    return {"proactive": 1.0, "reactive": 0.5, "none": 0.0}.get(pattern, 0.0)

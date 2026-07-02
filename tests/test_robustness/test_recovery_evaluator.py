"""Tests for the 3-dimension AVER RecoveryEvaluator."""

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau_robustness.injection_config import InjectionType
from tau_robustness.injector import InjectionEvent
from tau_robustness.recovery_evaluator import (
    DETECTION_WINDOW_TURNS,
    RecoveryEvaluator,
)


def make_injection_event(
    turn_idx: int = 5,
    tool_name: str = "get_order_details",
    injection_id: str = "test_injection",
    detection_signals: list[str] = None,
    recovery_signals: list[str] = None,
    blocks_actions: list[str] = None,
) -> InjectionEvent:
    return InjectionEvent(
        turn_idx=turn_idx,
        tool_call_id="tc_1",
        tool_name=tool_name,
        injection_id=injection_id,
        injection_type=InjectionType.STALE_DATA,
        original_content='{"status": "delivered"}',
        modified_content='{"status": "pending"}',
        detection_signals=detection_signals or ["already delivered", "not pending"],
        recovery_signals=recovery_signals
        or ["get_order_details", "return", "exchange"],
        blocks_actions=blocks_actions or ["exchange_delivered_order_items"],
    )


class TestRecoveryEvaluator:
    @pytest.fixture
    def evaluator(self):
        return RecoveryEvaluator()

    # --- No injection ---

    def test_no_injections(self, evaluator):
        metrics = evaluator.evaluate(trajectory=[], injection_log=[], task_reward=1.0)
        assert metrics.num_injections == 0
        assert metrics.detection_score is None
        assert metrics.diagnosis_score is None
        assert metrics.recovery_score == 1.0

    # --- Detection scoring ---

    def test_detection_explicit_signal(self, evaluator):
        """Agent mentions an injection-specific detection signal → base 0.8."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="This order has already delivered — that doesn't match what I see.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.8)
        # "already delivered" matches detection signal → base 0.8
        # proactive (no blocked action seen) → ×1.0
        assert metrics.detection_score == pytest.approx(0.8, abs=0.01)

    def test_detection_keyword_match(self, evaluator):
        """Agent uses generic detection keyword → base 0.5."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Something seems incorrect with the order status.",
                turn_idx=7,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        # "incorrect" → base 0.5, proactive → ×1.0
        assert metrics.detection_score == pytest.approx(0.5, abs=0.01)

    def test_detection_re_call_implicit(self, evaluator):
        """Agent re-calls the same tool (verification pattern) → base 0.4."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Let me check again.",
                tool_calls=[
                    ToolCall(
                        id="tc_2", name="get_order_details",
                        arguments={"order_id": "o123"},
                    )
                ],
                turn_idx=6,
            ),
            ToolMessage(
                id="tc_2", role="tool",
                content='{"status": "delivered"}', turn_idx=7,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        assert metrics.detection_score >= 0.4

    def test_no_detection(self, evaluator):
        """Agent proceeds without noticing → score 0."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Your order is pending. I'll cancel it for you now.",
                tool_calls=[
                    ToolCall(
                        id="tc_2", name="cancel_pending_order",
                        arguments={"order_id": "o123"},
                    )
                ],
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.0)
        assert metrics.detection_score == 0.0

    def test_detection_outside_window_ignored(self, evaluator):
        """Detection keywords beyond DETECTION_WINDOW_TURNS are not scored."""
        injection = make_injection_event(turn_idx=5)
        far_turn = 5 + DETECTION_WINDOW_TURNS + 2
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Your order is pending.",
                turn_idx=6,
            ),
            AssistantMessage(
                role="assistant",
                content="Actually this is incorrect, there's a mismatch.",
                turn_idx=far_turn,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        assert metrics.detection_score == 0.0

    def test_removed_keywords_no_longer_score(self, evaluator):
        """Generic CS phrases like 'let me verify' should NOT trigger detection."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Let me check the details. It appears to be pending.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        assert metrics.detection_score == 0.0

    def test_custom_signal_scores_higher_than_keyword(self, evaluator):
        """Custom detection signals (0.8) dominate generic keywords (0.5)."""
        injection = make_injection_event(
            turn_idx=5,
            detection_signals=["already delivered"],
        )
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="This order was already delivered, which is incorrect.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        # Custom signal 0.8 > keyword "incorrect" 0.5
        assert metrics.detection_score == pytest.approx(0.8, abs=0.01)

    def test_messages_before_injection_ignored(self, evaluator):
        """Messages before the injection turn should not affect scores."""
        injection = make_injection_event(turn_idx=10)
        trajectory = [
            AssistantMessage(
                role="assistant",
                content="Something seems incorrect here.",
                turn_idx=5,
            ),
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status":"pending"}', turn_idx=10,
            ),
            AssistantMessage(
                role="assistant",
                content="Your order looks good!",
                turn_idx=12,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        assert metrics.detection_score == 0.0

    # --- Temporal multiplier ---

    def test_temporal_proactive(self, evaluator):
        """Detection before blocked action attempt → proactive (×1.0)."""
        injection = make_injection_event(
            turn_idx=5,
            blocks_actions=["exchange_delivered_order_items"],
        )
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Wait, this is incorrect. The status doesn't match.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.8)
        assert metrics.temporal_pattern == "proactive"
        # base 0.5 × 1.0 = 0.5
        assert metrics.detection_score >= 0.5

    def test_temporal_reactive(self, evaluator):
        """Detection after blocked action attempt → reactive (×0.5)."""
        injection = make_injection_event(
            turn_idx=5,
            blocks_actions=["exchange_delivered_order_items"],
        )
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            # Agent tries the blocked action first
            AssistantMessage(
                role="assistant",
                content="Let me exchange this for you.",
                tool_calls=[
                    ToolCall(
                        id="tc_2", name="exchange_delivered_order_items",
                        arguments={"order_id": "o123"},
                    )
                ],
                turn_idx=6,
            ),
            ToolMessage(
                id="tc_2", role="tool",
                content="Error: order not in delivered status",
                error=True, turn_idx=7,
            ),
            # Then detects the error
            AssistantMessage(
                role="assistant",
                content="The status seems incorrect. Let me re-check.",
                turn_idx=8,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        assert metrics.temporal_pattern == "reactive"
        # base 0.5 × 0.5 = 0.25
        assert metrics.detection_score < 0.5

    def test_temporal_none(self, evaluator):
        """No detection at all → temporal 'none', score 0."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Your order is pending.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.0)
        assert metrics.temporal_pattern == "none"
        assert metrics.detection_score == 0.0

    # --- Diagnosis scoring ---

    def test_diagnosis_nothing(self, evaluator):
        """No relevant text → diagnosis 0.0."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Your order is pending. I'll handle it.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.0)
        assert metrics.diagnosis_score == 0.0

    def test_diagnosis_noticed(self, evaluator):
        """General uncertainty keywords → diagnosis 0.25."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="Something seems off with this order. It doesn't seem right.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        assert metrics.diagnosis_score == pytest.approx(0.25, abs=0.01)

    def test_diagnosis_error_type(self, evaluator):
        """Matches detection signal → diagnosis 0.50."""
        injection = make_injection_event(
            turn_idx=5,
            detection_signals=["already delivered"],
        )
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="The system says pending but you mentioned it was already delivered.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        # "already delivered" → detection signal match → at least 0.50
        # Also "but you mentioned" → root cause keyword → 0.75
        assert metrics.diagnosis_score >= 0.50

    def test_diagnosis_root_cause(self, evaluator):
        """Names specific discrepancy → diagnosis 0.75."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="The system shows pending but you said you received it. This conflicts with your description.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.5)
        assert metrics.diagnosis_score >= 0.75

    def test_diagnosis_full_explanation(self, evaluator):
        """Root cause + verification intent → diagnosis 1.0."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="The system shows pending but you said you received it. "
                "This conflicts with your description. Let me re-check the order details.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.8)
        assert metrics.diagnosis_score == 1.0

    # --- Causal chain ---

    def test_causal_chain_valid(self, evaluator):
        """Detection + recovery → causal chain valid."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="This seems incorrect. Let me re-check.",
                tool_calls=[
                    ToolCall(
                        id="tc_2", name="get_order_details",
                        arguments={"order_id": "o123"},
                    )
                ],
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.8)
        assert metrics.causal_chain_valid is True

    def test_causal_chain_penalty(self, evaluator):
        """Recovery behavior without ANY detection → diagnosis penalized ×0.75.

        The re-call of the injected tool counts as implicit detection (base 0.4),
        so to test the penalty we need a recovery signal that doesn't involve
        re-calling the injected tool.
        """
        injection = make_injection_event(
            turn_idx=5,
            tool_name="get_order_details",
            detection_signals=["very specific signal that wont match"],
            recovery_signals=["escalate_to_supervisor"],
            blocks_actions=[],
        )
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            # No detection keywords, calls a DIFFERENT tool that matches recovery signal
            AssistantMessage(
                role="assistant",
                content="I'll escalate this to a supervisor.",
                tool_calls=[
                    ToolCall(
                        id="tc_2", name="escalate_to_supervisor",
                        arguments={"reason": "unclear status"},
                    )
                ],
                turn_idx=6,
            ),
            ToolMessage(
                id="tc_2", role="tool",
                content='{"result": "escalated"}', turn_idx=7,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.8)
        # No detection (base 0 — no keywords, no re-call, no signals matched)
        # But recovery behavior exists (escalate_to_supervisor matches recovery signal)
        # → causal chain invalid
        assert metrics.causal_chain_valid is False

    # --- Recovery scoring ---

    def test_recovery_equals_task_reward(self, evaluator):
        """Recovery score IS the task reward — no heuristic."""
        injection = make_injection_event(turn_idx=5)
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant", content="Done.", turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=0.73)
        assert metrics.recovery_score == pytest.approx(0.73, abs=0.01)

    # --- AVER composite ---

    def test_aver_score_formula(self, evaluator):
        """AVER = (Det×0.4 + Diag×0.2 + Rec×0.4) × 100."""
        injection = make_injection_event(
            turn_idx=5,
            detection_signals=["already delivered"],
        )
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status": "pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="This order was already delivered. The status seems incorrect.",
                turn_idx=6,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [injection], task_reward=1.0)
        assert metrics.aver_score is not None
        assert metrics.aver_score > 0

    # --- Multiple injections ---

    def test_multiple_injections(self, evaluator):
        inj1 = make_injection_event(
            turn_idx=3, injection_id="inj1",
            detection_signals=["wrong status"],
        )
        inj2 = make_injection_event(
            turn_idx=8, tool_name="get_user_details",
            injection_id="inj2",
            detection_signals=["incorrect user"],
        )
        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status":"pending"}', turn_idx=3,
            ),
            AssistantMessage(
                role="assistant",
                content="I notice the wrong status here.",
                turn_idx=5,
            ),
            ToolMessage(
                id="tc_2", role="tool",
                content='{"name":"Alice"}', turn_idx=8,
            ),
            AssistantMessage(
                role="assistant",
                content="The user details look fine.",
                turn_idx=10,
            ),
        ]
        metrics = evaluator.evaluate(trajectory, [inj1, inj2], task_reward=0.5)
        assert metrics.num_injections == 2
        assert metrics.per_injection_scores is not None
        assert len(metrics.per_injection_scores) == 2

    # --- Replay skipping ---

    def test_replay_injections_skipped(self, evaluator):
        """Replayed persistent injections should not be scored independently."""
        original = make_injection_event(turn_idx=5, injection_id="orig")
        replay = make_injection_event(turn_idx=8, injection_id="orig")
        replay.is_replay = True

        trajectory = [
            ToolMessage(
                id="tc_1", role="tool",
                content='{"status":"pending"}', turn_idx=5,
            ),
            AssistantMessage(
                role="assistant",
                content="This is incorrect.",
                turn_idx=6,
            ),
            ToolMessage(
                id="tc_2", role="tool",
                content='{"status":"pending"}', turn_idx=8,
            ),
        ]
        metrics = evaluator.evaluate(
            trajectory, [original, replay], task_reward=0.5
        )
        # Only original should be scored, not replay
        assert len(metrics.per_injection_scores) == 1

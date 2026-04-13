"""Tests for the interaction quality evaluator."""

from unittest.mock import patch

from experiments.tau2_trace.interaction_quality import (
    TELECOM_GUIDANCE_TERMS,
    _compute_action_density,
    _compute_guidance_precision,
    _compute_guidance_precision_llm,
    _compute_token_to_action_ratio,
    _count_repeated_info_requests,
    _count_repeated_info_requests_llm,
    evaluate_interaction_quality,
)
from experiments.tau2_trace.models import ToolCallRecord
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage


class TestActionDensity:
    def test_zero_turns(self):
        assert _compute_action_density(5, 0) == 0.0

    def test_normal(self):
        assert _compute_action_density(4, 8) == 0.5

    def test_high_density(self):
        assert _compute_action_density(10, 5) == 2.0


class TestTokenToActionRatio:
    def test_no_tool_calls(self):
        messages = [AssistantMessage(role="assistant", content="Hello there")]
        assert _compute_token_to_action_ratio(messages, 0) == 0.0

    def test_char_count_fallback(self):
        """When no usage data, character count is used as proxy."""
        messages = [
            AssistantMessage(role="assistant", content="A" * 100),
            AssistantMessage(role="assistant", content="B" * 200),
        ]
        ratio = _compute_token_to_action_ratio(messages, 2)
        assert ratio == 150.0  # 300 chars / 2 calls

    def test_prefers_actual_token_usage(self):
        """When usage data is present, it takes precedence over char count."""
        messages = [
            AssistantMessage(
                role="assistant",
                content="A" * 100,
                usage={
                    "completion_tokens": 50,
                    "prompt_tokens": 20,
                    "total_tokens": 70,
                },
            ),
            AssistantMessage(
                role="assistant",
                content="B" * 200,
                usage={
                    "completion_tokens": 80,
                    "prompt_tokens": 30,
                    "total_tokens": 110,
                },
            ),
        ]
        ratio = _compute_token_to_action_ratio(messages, 2)
        assert ratio == 65.0  # (50 + 80) / 2 — NOT (100 + 200) / 2

    def test_mixed_usage_prefers_tokens(self):
        """When at least one message has usage data, use token-based counting."""
        messages = [
            AssistantMessage(
                role="assistant",
                content="A" * 100,
                usage={"completion_tokens": 40},
            ),
            AssistantMessage(
                role="assistant",
                content="B" * 200,
                usage=None,
            ),
        ]
        ratio = _compute_token_to_action_ratio(messages, 2)
        # Only the first message has tokens: 40 / 2 = 20
        assert ratio == 20.0


class TestRepeatedInfoRequests:
    def test_no_repeats(self):
        messages = [
            ToolMessage(id="t1", role="tool", content='{"customer_id": "C12345"}'),
            AssistantMessage(
                role="assistant", content="I will help you with your order."
            ),
        ]
        count = _count_repeated_info_requests(messages, [])
        assert count == 0

    def test_detects_repeat(self):
        messages = [
            ToolMessage(id="t1", role="tool", content='{"customer_id": "C12345"}'),
            AssistantMessage(
                role="assistant",
                content="What is your customer ID? Can you provide C12345 again?",
            ),
        ]
        count = _count_repeated_info_requests(messages, [])
        assert count >= 1


class TestGuidancePrecision:
    def test_precise_guidance(self):
        messages = [
            AssistantMessage(
                role="assistant", content="Please toggle your airplane mode off."
            ),
            AssistantMessage(role="assistant", content="Now check your APN settings."),
        ]
        precision = _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)
        assert precision == 1.0

    def test_vague_guidance(self):
        messages = [
            AssistantMessage(
                role="assistant", content="Please try the thing I mentioned."
            ),
            AssistantMessage(role="assistant", content="Can you check that setting?"),
        ]
        precision = _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)
        assert precision == 0.0

    def test_mixed_guidance(self):
        messages = [
            AssistantMessage(role="assistant", content="Please restart your device."),
            AssistantMessage(role="assistant", content="Just try again."),
        ]
        precision = _compute_guidance_precision(messages, TELECOM_GUIDANCE_TERMS)
        assert precision == 0.5


class TestEvaluateInteractionQuality:
    def test_full_evaluation(self):
        messages = [
            AssistantMessage(role="assistant", content="How can I help?"),
            UserMessage(role="user", content="My phone has no service"),
            AssistantMessage(role="assistant", content="Please check your SIM card."),
            UserMessage(role="user", content="OK done"),
        ]
        records = [
            ToolCallRecord(
                turn_index=1,
                name="check_sim_status",
                requestor="user",
            ),
        ]

        metrics = evaluate_interaction_quality(
            messages=messages,
            records=records,
            domain="telecom",
            task_id="test_1",
            trial=0,
            total_turns=4,
            agent_tool_call_count=0,
            expected_actions=2,
        )

        assert metrics.action_density == 0.25  # 1 tool / 4 turns
        assert metrics.turns_to_resolution == 4
        assert metrics.turns_vs_expected_ratio == 2.0  # 4 turns / 2 expected
        assert metrics.guidance_precision > 0.0  # "SIM card" matches

    def test_deterministic_is_default(self):
        """LLM judge is off by default."""
        messages = [
            AssistantMessage(role="assistant", content="Hello"),
        ]
        metrics = evaluate_interaction_quality(
            messages=messages,
            records=[],
            domain="retail",
            task_id="test_1",
        )
        assert metrics.repeated_info_requests == 0


class TestLLMJudge:
    """Verify LLM judge path hooks into tau2.utils.llm_utils.generate()
    and correctly parses the model response.  Uses mocks so no API call
    is made."""

    def test_repeated_info_llm_calls_generate(self):
        """LLM judge for repeated_info_requests invokes generate() and
        parses the JSON count from the response."""
        messages = [
            ToolMessage(id="t1", role="tool", content='{"mac": "AA:BB:CC:DD"}'),
            AssistantMessage(
                role="assistant",
                content="What is your MAC address?",
            ),
        ]
        mock_response = AssistantMessage(role="assistant", content='{"count": 1}')
        with patch(
            "tau2.utils.llm_utils.generate",
            return_value=mock_response,
        ) as mock_gen:
            count = _count_repeated_info_requests_llm(messages, "gpt-4.1")
            mock_gen.assert_called_once()
            assert count == 1

    def test_guidance_precision_llm_calls_generate(self):
        """LLM judge for guidance_precision invokes generate() and
        parses the JSON fraction from the response."""
        messages = [
            AssistantMessage(role="assistant", content="Please toggle airplane mode."),
            AssistantMessage(role="assistant", content="Just try again."),
        ]
        mock_response = AssistantMessage(
            role="assistant",
            content='{"precise_count": 1, "total_agent_messages": 2}',
        )
        with patch(
            "tau2.utils.llm_utils.generate",
            return_value=mock_response,
        ) as mock_gen:
            precision = _compute_guidance_precision_llm(messages, "gpt-4.1")
            mock_gen.assert_called_once()
            assert precision == 0.5

    def test_llm_judge_fallback_on_failure(self):
        """When generate() raises, falls back to deterministic evaluation."""
        messages = [
            AssistantMessage(role="assistant", content="Please restart your device."),
            AssistantMessage(role="assistant", content="Just try again."),
        ]
        with patch(
            "tau2.utils.llm_utils.generate",
            side_effect=Exception("API error"),
        ):
            precision = _compute_guidance_precision_llm(messages, "gpt-4.1")
            # Falls back to deterministic: "restart" matches TELECOM_GUIDANCE_TERMS
            assert precision == 0.5

    def test_llm_judge_threaded_via_evaluate(self):
        """use_llm_judge=True threads through to LLM judge functions."""
        messages = [
            AssistantMessage(role="assistant", content="How can I help?"),
            UserMessage(role="user", content="Fix my phone"),
        ]
        mock_response = AssistantMessage(role="assistant", content='{"count": 0}')
        with patch(
            "tau2.utils.llm_utils.generate",
            return_value=mock_response,
        ) as mock_gen:
            metrics = evaluate_interaction_quality(
                messages=messages,
                records=[],
                domain="retail",
                task_id="test_1",
                use_llm_judge=True,
                llm_judge_model="gpt-4.1",
            )
            mock_gen.assert_called_once()
            assert metrics.repeated_info_requests == 0

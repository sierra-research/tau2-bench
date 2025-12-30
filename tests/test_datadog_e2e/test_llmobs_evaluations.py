"""Unit tests for LLMObs evaluation submission.

Tests the tau2_agent.llmobs_evaluations module which provides functions
to submit evaluation metrics to Datadog LLM Observability.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


# Mock span context for tests
MOCK_SPAN_CONTEXT = {"trace_id": "test-trace-123", "span_id": "test-span-456"}


class TestIsLLMObsEnabled:
    """Tests for is_llmobs_enabled() function."""

    def test_returns_false_when_no_env_vars_set(self):
        """LLMObs is disabled when environment variables are not set."""
        with patch.dict(os.environ, {}, clear=True):
            from tau2_agent.llmobs_evaluations import is_llmobs_enabled

            assert is_llmobs_enabled() is False

    def test_returns_false_when_only_trace_enabled(self):
        """LLMObs is disabled when only DD_TRACE_ENABLED is set."""
        with patch.dict(
            os.environ,
            {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "false"},
            clear=True,
        ):
            from tau2_agent.llmobs_evaluations import is_llmobs_enabled

            assert is_llmobs_enabled() is False

    def test_returns_false_when_only_llmobs_enabled(self):
        """LLMObs is disabled when only DD_LLMOBS_ENABLED is set."""
        with patch.dict(
            os.environ,
            {"DD_TRACE_ENABLED": "false", "DD_LLMOBS_ENABLED": "true"},
            clear=True,
        ):
            from tau2_agent.llmobs_evaluations import is_llmobs_enabled

            assert is_llmobs_enabled() is False

    def test_returns_true_when_both_env_vars_enabled(self):
        """LLMObs is enabled when both DD_TRACE_ENABLED and DD_LLMOBS_ENABLED are true."""
        with patch.dict(
            os.environ,
            {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
            clear=True,
        ):
            from tau2_agent.llmobs_evaluations import is_llmobs_enabled

            assert is_llmobs_enabled() is True

    def test_case_insensitive_env_vars(self):
        """Environment variable values are case-insensitive."""
        with patch.dict(
            os.environ,
            {"DD_TRACE_ENABLED": "TRUE", "DD_LLMOBS_ENABLED": "True"},
            clear=True,
        ):
            from tau2_agent.llmobs_evaluations import is_llmobs_enabled

            assert is_llmobs_enabled() is True


class TestSubmitTaskEvaluations:
    """Tests for submit_task_evaluations() function."""

    def test_noop_when_llmobs_disabled(self):
        """No evaluations submitted when LLMObs is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            # Should not raise, should be no-op
            submit_task_evaluations(
                task_id="test-1",
                domain="mock",
                reward=0.85,
                termination_reason="user_stop",
                reward_info=None,
            )
            # No assertion needed - just verify no exception

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_task_reward_metric(self):
        """Submits tau2.task.reward score metric."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.85,
                termination_reason="agent_stop",
                reward_info=None,
                span_context=MOCK_SPAN_CONTEXT,
            )

            # Find the call for tau2.task.reward
            calls = mock_llmobs.submit_evaluation.call_args_list
            reward_call = next(
                c for c in calls if c.kwargs.get("label") == "tau2.task.reward"
            )
            assert reward_call.kwargs["span_context"] == MOCK_SPAN_CONTEXT
            assert reward_call.kwargs["value"] == 0.85
            assert reward_call.kwargs["metric_type"] == "score"
            assert reward_call.kwargs["tags"]["task_id"] == "task-123"
            assert reward_call.kwargs["tags"]["domain"] == "airline"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_task_success_pass_when_reward_high(self):
        """Submits tau2.task.success as 'pass' when reward >= 0.7."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.75,
                termination_reason="agent_stop",
                reward_info=None,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            success_call = next(
                c for c in calls if c.kwargs.get("label") == "tau2.task.success"
            )
            assert success_call.kwargs["value"] == "pass"
            assert success_call.kwargs["metric_type"] == "categorical"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_task_success_fail_when_reward_low(self):
        """Submits tau2.task.success as 'fail' when reward < 0.7."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.5,
                termination_reason="agent_stop",
                reward_info=None,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            success_call = next(
                c for c in calls if c.kwargs.get("label") == "tau2.task.success"
            )
            assert success_call.kwargs["value"] == "fail"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_termination_reason(self):
        """Submits tau2.task.termination categorical metric."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.85,
                termination_reason="max_steps",
                reward_info=None,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            termination_call = next(
                c for c in calls if c.kwargs.get("label") == "tau2.task.termination"
            )
            assert termination_call.kwargs["value"] == "max_steps"
            assert termination_call.kwargs["metric_type"] == "categorical"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_includes_evaluation_id_in_tags_when_provided(self):
        """Includes evaluation_id in tags when provided."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.85,
                termination_reason="agent_stop",
                reward_info=None,
                evaluation_id="eval-abc-123",
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            # All calls should have evaluation_id in tags
            for call in calls:
                assert call.kwargs["tags"]["evaluation_id"] == "eval-abc-123"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_handles_exception_gracefully(self):
        """Exceptions are caught and logged, not raised."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            mock_llmobs.submit_evaluation.side_effect = Exception("API error")

            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            # Should not raise
            submit_task_evaluations(
                task_id="task-123",
                domain="mock",
                reward=0.85,
                termination_reason="user_stop",
                reward_info=None,
                span_context=MOCK_SPAN_CONTEXT,
            )

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_handles_import_error_gracefully(self):
        """ImportError is caught when ddtrace not installed."""
        import sys

        # Temporarily remove ddtrace.llmobs from modules to simulate import failure
        with patch.dict(sys.modules, {"ddtrace.llmobs": None}):
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            # Should not raise - gracefully handles missing ddtrace
            submit_task_evaluations(
                task_id="task-123",
                domain="mock",
                reward=0.85,
                termination_reason="user_stop",
                reward_info=None,
                span_context=MOCK_SPAN_CONTEXT,
            )

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_skips_submission_when_no_span_context(self):
        """Skips submission and logs warning when no span context available."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            # Make export_span return None
            mock_llmobs.export_span.return_value = None

            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            submit_task_evaluations(
                task_id="task-123",
                domain="mock",
                reward=0.85,
                termination_reason="user_stop",
                reward_info=None,
                # No span_context provided, and export_span returns None
            )

            # submit_evaluation should NOT be called
            mock_llmobs.submit_evaluation.assert_not_called()


class TestSubmitAssertionEvaluations:
    """Tests for assertion-level evaluation metrics."""

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_db_check_pass(self):
        """Submits tau2.assertion.db_check as 'pass' when db_match is True."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            reward_info = {"db_check": {"db_match": True}}

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.85,
                termination_reason="agent_stop",
                reward_info=reward_info,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            db_call = next(
                c for c in calls if c.kwargs.get("label") == "tau2.assertion.db_check"
            )
            assert db_call.kwargs["value"] == "pass"
            assert db_call.kwargs["metric_type"] == "categorical"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_db_check_fail(self):
        """Submits tau2.assertion.db_check as 'fail' when db_match is False."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            reward_info = {"db_check": {"db_match": False}}

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.5,
                termination_reason="agent_stop",
                reward_info=reward_info,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            db_call = next(
                c for c in calls if c.kwargs.get("label") == "tau2.assertion.db_check"
            )
            assert db_call.kwargs["value"] == "fail"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_nl_assertion_pass_rate(self):
        """Submits tau2.assertion.nl_pass_rate as score."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            reward_info = {
                "nl_assertions": [
                    {"met": True, "nl_assertion": "User confirmed"},
                    {"met": True, "nl_assertion": "Response polite"},
                    {"met": False, "nl_assertion": "Error handled"},
                ]
            }

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.85,
                termination_reason="agent_stop",
                reward_info=reward_info,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            nl_call = next(
                c
                for c in calls
                if c.kwargs.get("label") == "tau2.assertion.nl_pass_rate"
            )
            # 2 out of 3 passed = 0.666...
            assert abs(nl_call.kwargs["value"] - 2 / 3) < 0.001
            assert nl_call.kwargs["metric_type"] == "score"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_action_accuracy(self):
        """Submits tau2.assertion.action_accuracy as score."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            reward_info = {
                "action_checks": [
                    {"action": {"name": "book_flight"}, "action_match": True},
                    {"action": {"name": "cancel_booking"}, "action_match": True},
                    {"action": {"name": "send_email"}, "action_match": False},
                    {"action": {"name": "update_record"}, "action_match": True},
                ]
            }

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.85,
                termination_reason="agent_stop",
                reward_info=reward_info,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            action_call = next(
                c
                for c in calls
                if c.kwargs.get("label") == "tau2.assertion.action_accuracy"
            )
            # 3 out of 4 correct = 0.75
            assert action_call.kwargs["value"] == 0.75
            assert action_call.kwargs["metric_type"] == "score"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_communicate_pass_rate(self):
        """Submits tau2.assertion.communicate_pass_rate as score."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            reward_info = {
                "communicate_checks": [
                    {"met": True},
                    {"met": True},
                ]
            }

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.85,
                termination_reason="agent_stop",
                reward_info=reward_info,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            comm_call = next(
                c
                for c in calls
                if c.kwargs.get("label") == "tau2.assertion.communicate_pass_rate"
            )
            # 2 out of 2 passed = 1.0
            assert comm_call.kwargs["value"] == 1.0
            assert comm_call.kwargs["metric_type"] == "score"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_skips_empty_assertion_lists(self):
        """Does not submit metrics for empty assertion lists."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_task_evaluations

            reward_info = {
                "nl_assertions": [],
                "action_checks": [],
                "communicate_checks": [],
            }

            submit_task_evaluations(
                task_id="task-123",
                domain="airline",
                reward=0.85,
                termination_reason="agent_stop",
                reward_info=reward_info,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            labels = [c.kwargs.get("label") for c in calls]

            # Should not have any assertion metrics
            assert "tau2.assertion.nl_pass_rate" not in labels
            assert "tau2.assertion.action_accuracy" not in labels
            assert "tau2.assertion.communicate_pass_rate" not in labels


class TestSubmitEvaluationSummary:
    """Tests for submit_evaluation_summary() function."""

    def test_noop_when_llmobs_disabled(self):
        """No summary submitted when LLMObs is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            from tau2_agent.llmobs_evaluations import submit_evaluation_summary

            # Should not raise, should be no-op
            submit_evaluation_summary(
                evaluation_id="eval-123",
                domain="airline",
                total_tasks=10,
                successful_tasks=8,
                avg_reward=0.75,
            )

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_pass_rate(self):
        """Submits tau2.evaluation.pass_rate score metric."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_evaluation_summary

            submit_evaluation_summary(
                evaluation_id="eval-123",
                domain="airline",
                total_tasks=10,
                successful_tasks=8,
                avg_reward=0.75,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            pass_rate_call = next(
                c
                for c in calls
                if c.kwargs.get("label") == "tau2.evaluation.pass_rate"
            )
            assert pass_rate_call.kwargs["span_context"] == MOCK_SPAN_CONTEXT
            assert pass_rate_call.kwargs["value"] == 0.8  # 8/10
            assert pass_rate_call.kwargs["metric_type"] == "score"
            assert pass_rate_call.kwargs["tags"]["evaluation_id"] == "eval-123"
            assert pass_rate_call.kwargs["tags"]["domain"] == "airline"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_submits_avg_reward(self):
        """Submits tau2.evaluation.avg_reward score metric."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_evaluation_summary

            submit_evaluation_summary(
                evaluation_id="eval-123",
                domain="airline",
                total_tasks=10,
                successful_tasks=8,
                avg_reward=0.75,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            avg_reward_call = next(
                c
                for c in calls
                if c.kwargs.get("label") == "tau2.evaluation.avg_reward"
            )
            assert avg_reward_call.kwargs["value"] == 0.75
            assert avg_reward_call.kwargs["metric_type"] == "score"

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_handles_zero_total_tasks(self):
        """Pass rate is 0 when total_tasks is 0 (avoids division by zero)."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            from tau2_agent.llmobs_evaluations import submit_evaluation_summary

            submit_evaluation_summary(
                evaluation_id="eval-123",
                domain="airline",
                total_tasks=0,
                successful_tasks=0,
                avg_reward=0.0,
                span_context=MOCK_SPAN_CONTEXT,
            )

            calls = mock_llmobs.submit_evaluation.call_args_list
            pass_rate_call = next(
                c
                for c in calls
                if c.kwargs.get("label") == "tau2.evaluation.pass_rate"
            )
            assert pass_rate_call.kwargs["value"] == 0.0

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_handles_exception_gracefully(self):
        """Exceptions are caught and logged, not raised."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            mock_llmobs.submit_evaluation.side_effect = Exception("API error")

            from tau2_agent.llmobs_evaluations import submit_evaluation_summary

            # Should not raise
            submit_evaluation_summary(
                evaluation_id="eval-123",
                domain="airline",
                total_tasks=10,
                successful_tasks=8,
                avg_reward=0.75,
                span_context=MOCK_SPAN_CONTEXT,
            )

    @patch.dict(
        os.environ,
        {"DD_TRACE_ENABLED": "true", "DD_LLMOBS_ENABLED": "true"},
        clear=True,
    )
    def test_skips_submission_when_no_span_context(self):
        """Skips submission and logs warning when no span context available."""
        with patch("ddtrace.llmobs.LLMObs") as mock_llmobs:
            # Make export_span return None
            mock_llmobs.export_span.return_value = None

            from tau2_agent.llmobs_evaluations import submit_evaluation_summary

            submit_evaluation_summary(
                evaluation_id="eval-123",
                domain="airline",
                total_tasks=10,
                successful_tasks=8,
                avg_reward=0.75,
                # No span_context provided, and export_span returns None
            )

            # submit_evaluation should NOT be called
            mock_llmobs.submit_evaluation.assert_not_called()

"""
Unit tests for tau2_agent.streaming module.

Tests for EvaluationProgress dataclass and ADK event builder functions.
"""

from datetime import datetime, timedelta, timezone

import pytest


# =============================================================================
# EvaluationProgress Tests
# =============================================================================


class TestEvaluationProgressPercent:
    """Tests for EvaluationProgress.percent property."""

    def test_percent_zero_tasks(self) -> None:
        """0 tasks returns 0%."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=0)
        assert progress.percent == 0

    def test_percent_calculation_at_start(self) -> None:
        """Percent is 0 at start."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5)
        assert progress.percent == 0

    def test_percent_calculation_at_25(self) -> None:
        """Correct percentage at 25%."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=4, completed_tasks=1)
        assert progress.percent == 25

    def test_percent_calculation_at_50(self) -> None:
        """Correct percentage at 50%."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=10, completed_tasks=5)
        assert progress.percent == 50

    def test_percent_calculation_at_75(self) -> None:
        """Correct percentage at 75%."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=4, completed_tasks=3)
        assert progress.percent == 75

    def test_percent_calculation_at_100(self) -> None:
        """Correct percentage at completion."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5, completed_tasks=5)
        assert progress.percent == 100

    def test_percent_rounds_down(self) -> None:
        """Percentage rounds down to integer."""
        from tau2_agent.streaming.progress import EvaluationProgress

        # 1/3 = 33.33... should be 33
        progress = EvaluationProgress(total_tasks=3, completed_tasks=1)
        assert progress.percent == 33


class TestEvaluationProgressElapsedSeconds:
    """Tests for EvaluationProgress.elapsed_seconds property."""

    def test_elapsed_seconds_no_start(self) -> None:
        """Elapsed is 0 when started_at is None."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5, started_at=None)
        assert progress.elapsed_seconds == 0.0

    def test_elapsed_seconds_calculation(self) -> None:
        """Elapsed time calculation is correct."""
        from tau2_agent.streaming.progress import EvaluationProgress

        now = datetime.now(timezone.utc)
        past = now - timedelta(seconds=10)

        progress = EvaluationProgress(total_tasks=5, started_at=past)

        elapsed = progress.elapsed_seconds
        assert elapsed >= 9.0  # At least 9 seconds
        assert elapsed < 15.0  # Not more than 15 seconds


class TestEvaluationProgressIncrement:
    """Tests for EvaluationProgress.increment method."""

    def test_increment_increases_completed_tasks(self) -> None:
        """Increment increases completed_tasks by 1."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5, completed_tasks=0)
        progress.increment()
        assert progress.completed_tasks == 1

    def test_increment_updates_task_id(self) -> None:
        """Increment updates current_task_id if provided."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5)
        progress.increment(task_id="task_001")
        assert progress.current_task_id == "task_001"
        assert progress.completed_tasks == 1

    def test_increment_multiple_times(self) -> None:
        """Multiple increments accumulate correctly."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5)
        progress.increment(task_id="task_001")
        progress.increment(task_id="task_002")
        progress.increment(task_id="task_003")

        assert progress.completed_tasks == 3
        assert progress.current_task_id == "task_003"
        assert progress.percent == 60


class TestEvaluationProgressToMetadata:
    """Tests for EvaluationProgress.to_metadata method."""

    def test_to_metadata_produces_tau2_namespaced_dict(self) -> None:
        """to_metadata produces correct tau2.* namespaced dict."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(
            total_tasks=10,
            completed_tasks=5,
            current_task_id="task_005",
            current_trial=2,
            total_trials=3,
        )
        metadata = progress.to_metadata()

        assert "tau2.progress" in metadata
        assert "tau2.completed_tasks" in metadata
        assert "tau2.total_tasks" in metadata
        assert "tau2.current_task_id" in metadata
        assert "tau2.current_trial" in metadata
        assert "tau2.total_trials" in metadata
        assert "tau2.elapsed_seconds" in metadata

    def test_to_metadata_correct_values(self) -> None:
        """to_metadata has correct values."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(
            total_tasks=4,
            completed_tasks=2,
            current_task_id="airline_003",
            current_trial=1,
            total_trials=3,
        )
        metadata = progress.to_metadata()

        assert metadata["tau2.progress"] == 50
        assert metadata["tau2.completed_tasks"] == 2
        assert metadata["tau2.total_tasks"] == 4
        assert metadata["tau2.current_task_id"] == "airline_003"
        assert metadata["tau2.current_trial"] == 1
        assert metadata["tau2.total_trials"] == 3
        assert isinstance(metadata["tau2.elapsed_seconds"], float)

    def test_to_metadata_elapsed_seconds_rounded(self) -> None:
        """Elapsed seconds is rounded to 2 decimal places."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5)
        metadata = progress.to_metadata()

        elapsed = metadata["tau2.elapsed_seconds"]
        assert isinstance(elapsed, float)
        assert elapsed == round(elapsed, 2)


class TestEvaluationProgressDefaults:
    """Tests for EvaluationProgress default values."""

    def test_default_completed_tasks(self) -> None:
        """completed_tasks defaults to 0."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5)
        assert progress.completed_tasks == 0

    def test_default_current_task_id(self) -> None:
        """current_task_id defaults to None."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5)
        assert progress.current_task_id is None

    def test_default_trials(self) -> None:
        """Trial fields default to 1."""
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=5)
        assert progress.current_trial == 1
        assert progress.total_trials == 1

    def test_default_started_at_is_set(self) -> None:
        """started_at is set to current UTC time by default."""
        from tau2_agent.streaming.progress import EvaluationProgress

        before = datetime.now(timezone.utc)
        progress = EvaluationProgress(total_tasks=5)
        after = datetime.now(timezone.utc)

        assert progress.started_at is not None
        assert before <= progress.started_at <= after


# =============================================================================
# ADK Event Builder Tests
# =============================================================================


class TestCreateAdkProgressEventSubmitted:
    """Tests for create_adk_progress_event with submitted state."""

    def test_produces_event_with_state_submitted(self) -> None:
        """Produces Event with state=submitted."""
        from google.adk.events.event import Event

        from tau2_agent.streaming.events import create_adk_progress_event

        event = create_adk_progress_event(
            invocation_id="test-123",
            state="submitted",
            message="Starting evaluation",
        )

        assert isinstance(event, Event)
        assert event.custom_metadata["tau2.state"] == "submitted"

    def test_submitted_event_has_zero_progress(self) -> None:
        """Submitted event has 0% progress by default."""
        from tau2_agent.streaming.events import create_adk_progress_event

        event = create_adk_progress_event(
            invocation_id="test-123",
            state="submitted",
            message="Starting evaluation",
        )

        assert event.custom_metadata["tau2.progress"] == 0


class TestCreateAdkProgressEventWorking:
    """Tests for create_adk_progress_event with working state."""

    def test_produces_event_with_state_working(self) -> None:
        """Produces Event with state=working."""
        from tau2_agent.streaming.events import create_adk_progress_event
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=10, completed_tasks=3)
        event = create_adk_progress_event(
            invocation_id="test-123",
            state="working",
            message="Processing task",
            progress=progress,
        )

        assert event.custom_metadata["tau2.state"] == "working"

    def test_working_event_includes_progress_metadata(self) -> None:
        """Working event includes progress from EvaluationProgress."""
        from tau2_agent.streaming.events import create_adk_progress_event
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=10, completed_tasks=5)
        event = create_adk_progress_event(
            invocation_id="test-123",
            state="working",
            message="Processing",
            progress=progress,
        )

        assert event.custom_metadata["tau2.progress"] == 50
        assert event.custom_metadata["tau2.completed_tasks"] == 5
        assert event.custom_metadata["tau2.total_tasks"] == 10

    def test_working_state_requires_progress(self) -> None:
        """Working state without progress raises ValueError."""
        import pytest

        from tau2_agent.streaming.events import create_adk_progress_event

        with pytest.raises(ValueError, match="progress is required for 'working' state"):
            create_adk_progress_event(
                invocation_id="test-123",
                state="working",
                message="Processing task",
            )


class TestCreateAdkProgressEventWithProgressObject:
    """Tests for create_adk_progress_event with EvaluationProgress integration."""

    def test_integrates_evaluation_progress(self) -> None:
        """Integrates EvaluationProgress metadata."""
        from tau2_agent.streaming.events import create_adk_progress_event
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(
            total_tasks=5,
            completed_tasks=2,
            current_task_id="airline_003",
            current_trial=1,
            total_trials=3,
        )

        event = create_adk_progress_event(
            invocation_id="test-123",
            state="working",
            message="Evaluating task airline_003",
            progress=progress,
        )

        assert event.custom_metadata["tau2.progress"] == 40
        assert event.custom_metadata["tau2.current_task_id"] == "airline_003"
        assert event.custom_metadata["tau2.current_trial"] == 1
        assert event.custom_metadata["tau2.total_trials"] == 3


class TestCreateAdkErrorEvent:
    """Tests for create_adk_error_event."""

    def test_produces_event_with_error_code_set(self) -> None:
        """Produces Event with error_code set."""
        from google.adk.events.event import Event

        from tau2_agent.streaming.events import create_adk_error_event

        event = create_adk_error_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            error_message="Connection failed",
            error_code="CONNECTION_ERROR",
        )

        assert isinstance(event, Event)
        assert event.error_code == "CONNECTION_ERROR"
        assert event.error_message == "Connection failed"

    def test_error_event_has_failed_state(self) -> None:
        """Error event has state=failed."""
        from tau2_agent.streaming.events import create_adk_error_event

        event = create_adk_error_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            error_message="Something went wrong",
        )

        assert event.custom_metadata["tau2.state"] == "failed"

    def test_error_event_includes_error_in_metadata(self) -> None:
        """Error event includes error message in metadata."""
        from tau2_agent.streaming.events import create_adk_error_event

        event = create_adk_error_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            error_message="Task timeout",
            error_code="TIMEOUT",
        )

        assert event.custom_metadata["tau2.error"] == "Task timeout"
        assert event.custom_metadata["tau2.error_code"] == "TIMEOUT"

    def test_error_event_content_contains_error_message(self) -> None:
        """Error event content contains the error message."""
        from tau2_agent.streaming.events import create_adk_error_event

        event = create_adk_error_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            error_message="Agent unreachable",
        )

        assert event.content is not None
        assert len(event.content.parts) > 0
        assert "Agent unreachable" in event.content.parts[0].text


class TestCreateAdkResultEvent:
    """Tests for create_adk_result_event."""

    def test_produces_event_with_results_in_content(self) -> None:
        """Includes results in content."""
        from tau2_agent.streaming.events import create_adk_result_event

        results = {"success_rate": 0.8, "total_tasks": 5}

        event = create_adk_result_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            results=results,
        )

        assert event.content is not None
        content_text = event.content.parts[0].text
        assert "0.8" in content_text
        assert "success_rate" in content_text

    def test_result_event_has_completed_state(self) -> None:
        """Result event has state=completed."""
        from tau2_agent.streaming.events import create_adk_result_event

        event = create_adk_result_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            results={"score": 100},
        )

        assert event.custom_metadata["tau2.state"] == "completed"

    def test_result_event_has_100_percent_progress(self) -> None:
        """Result event has 100% progress."""
        from tau2_agent.streaming.events import create_adk_result_event

        event = create_adk_result_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            results={},
        )

        assert event.custom_metadata["tau2.progress"] == 100

    def test_result_event_includes_custom_message(self) -> None:
        """Result event includes custom message."""
        from tau2_agent.streaming.events import create_adk_result_event

        event = create_adk_result_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            results={},
            message="All tasks completed successfully",
        )

        assert "All tasks completed successfully" in event.content.parts[0].text


class TestRequiredMetadataPresent:
    """Tests for required tau2 metadata presence."""

    def test_progress_event_has_required_metadata(self) -> None:
        """Progress events have tau2.state, tau2.progress, tau2.evaluation_id."""
        from tau2_agent.streaming.events import create_adk_progress_event
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=10, completed_tasks=5)
        event = create_adk_progress_event(
            invocation_id="test-123",
            state="working",
            message="Processing",
            evaluation_id="eval-456",
            progress=progress,
        )

        assert "tau2.state" in event.custom_metadata
        assert "tau2.progress" in event.custom_metadata
        assert "tau2.evaluation_id" in event.custom_metadata

    def test_error_event_has_required_metadata(self) -> None:
        """Error events have tau2.state and tau2.evaluation_id."""
        from tau2_agent.streaming.events import create_adk_error_event

        event = create_adk_error_event(
            invocation_id="test-123",
            evaluation_id="eval-456",
            error_message="Failed",
        )

        assert "tau2.state" in event.custom_metadata
        assert event.custom_metadata["tau2.state"] == "failed"
        assert "tau2.evaluation_id" in event.custom_metadata

    def test_result_event_has_required_metadata(self) -> None:
        """Result events have tau2.state, tau2.progress, tau2.evaluation_id."""
        from tau2_agent.streaming.events import create_adk_result_event

        event = create_adk_result_event(
            invocation_id="test-123",
            evaluation_id="eval-456",
            results={},
        )

        assert "tau2.state" in event.custom_metadata
        assert "tau2.progress" in event.custom_metadata
        assert "tau2.evaluation_id" in event.custom_metadata


class TestExtraMetadata:
    """Tests for extra metadata passing."""

    def test_progress_event_accepts_extra_metadata(self) -> None:
        """Extra metadata can be passed to progress events."""
        from tau2_agent.streaming.events import create_adk_progress_event
        from tau2_agent.streaming.progress import EvaluationProgress

        progress = EvaluationProgress(total_tasks=10, completed_tasks=5)
        event = create_adk_progress_event(
            invocation_id="test-123",
            state="working",
            message="Processing",
            progress=progress,
            **{
                "tau2.domain": "airline",
                "tau2.agent_endpoint": "http://localhost:8080",
            },
        )

        assert event.custom_metadata["tau2.domain"] == "airline"
        assert event.custom_metadata["tau2.agent_endpoint"] == "http://localhost:8080"

    def test_error_event_accepts_extra_metadata(self) -> None:
        """Extra metadata can be passed to error events."""
        from tau2_agent.streaming.events import create_adk_error_event

        event = create_adk_error_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            error_message="Failed",
            **{"tau2.domain": "retail"},
        )

        assert event.custom_metadata["tau2.domain"] == "retail"

    def test_result_event_accepts_extra_metadata(self) -> None:
        """Extra metadata can be passed to result events."""
        from tau2_agent.streaming.events import create_adk_result_event

        event = create_adk_result_event(
            invocation_id="test-123",
            evaluation_id="eval-123",
            results={},
            **{"tau2.domain": "telecom"},
        )

        assert event.custom_metadata["tau2.domain"] == "telecom"


class TestEventAttributes:
    """Tests for Event attributes."""

    def test_event_has_correct_invocation_id(self) -> None:
        """Event has correct invocation_id."""
        from tau2_agent.streaming.events import create_adk_progress_event

        event = create_adk_progress_event(
            invocation_id="my-invocation-123",
            state="submitted",
            message="Starting",
        )

        assert event.invocation_id == "my-invocation-123"

    def test_event_has_tau2_agent_author(self) -> None:
        """Event has tau2_agent as author."""
        from tau2_agent.streaming.events import create_adk_progress_event

        event = create_adk_progress_event(
            invocation_id="test-123",
            state="submitted",
            message="Starting",
        )

        assert event.author == "tau2_agent"

    def test_event_content_has_model_role(self) -> None:
        """Event content has 'model' role."""
        from tau2_agent.streaming.events import create_adk_progress_event

        event = create_adk_progress_event(
            invocation_id="test-123",
            state="submitted",
            message="Starting evaluation",
        )

        assert event.content.role == "model"

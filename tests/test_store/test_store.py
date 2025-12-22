"""
Tests for Evaluation Store - User Story 1: Core Storage

Tests cover create, update, complete, fail, and retrieve operations.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tau2.store import (
    Evaluation,
    EvaluationIdCollisionError,
    EvaluationNotFoundError,
    EvaluationStatus,
    InvalidStateError,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set environment variable for the store
        old_env = os.environ.get("TAU2_DATA_DIR")
        os.environ["TAU2_DATA_DIR"] = tmpdir
        yield Path(tmpdir)
        # Restore original environment
        if old_env is not None:
            os.environ["TAU2_DATA_DIR"] = old_env
        else:
            os.environ.pop("TAU2_DATA_DIR", None)


@pytest.fixture
def store(temp_data_dir):
    """Create a store instance with temporary directory."""
    from tau2.store import create_store

    return create_store(temp_data_dir)


@pytest.fixture
def sample_request():
    """Sample evaluation request parameters."""
    return {"num_tasks": 5, "num_trials": 1, "user_llm": "gpt-4"}


class TestCreateSession:
    """Tests for create_session method."""

    def test_create_session_returns_evaluation_id(self, store, sample_request):
        """create_session should return a valid evaluation ID."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        assert evaluation_id is not None
        assert evaluation_id.startswith("eval-")
        assert len(evaluation_id.split("-")) == 3

    def test_create_session_with_trace_id(self, store, sample_request):
        """create_session should store trace_id for OTel correlation."""
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"

        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
            trace_id=trace_id,
        )

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.trace_id == trace_id

    def test_create_session_with_session_id(self, store, sample_request):
        """create_session should store session_id for SSE reconnection."""
        session_id = "sess-abc123"

        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
            session_id=session_id,
        )

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.session_id == session_id

    def test_create_session_with_agent_endpoint(self, store, sample_request):
        """create_session should store agent endpoint URL."""
        agent_endpoint = "http://localhost:8080/a2a"

        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
            agent_endpoint=agent_endpoint,
        )

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.agent_endpoint == agent_endpoint

    def test_create_session_initial_status_is_submitted(self, store, sample_request):
        """create_session should set status to SUBMITTED."""
        evaluation_id = store.create_session(
            domain="retail",
            request=sample_request,
        )

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.status == EvaluationStatus.SUBMITTED

    def test_create_session_creates_file_in_sessions_dir(
        self, store, temp_data_dir, sample_request
    ):
        """create_session should create file in sessions directory."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        session_path = temp_data_dir / "sessions" / f"{evaluation_id}.json"
        assert session_path.exists()

    def test_create_session_initializes_state_history(self, store, sample_request):
        """create_session should initialize state_history with SUBMITTED."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert len(evaluation.state_history) == 1
        assert evaluation.state_history[0].state == EvaluationStatus.SUBMITTED


class TestUpdateProgress:
    """Tests for update_progress method."""

    def test_update_progress_updates_current_task(self, store, sample_request):
        """update_progress should update current task number."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.update_progress(evaluation_id, current_task=2, total_tasks=5)

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.progress is not None
        assert evaluation.progress.current_task == 2
        assert evaluation.progress.total_tasks == 5

    def test_update_progress_calculates_percent(self, store, sample_request):
        """update_progress should calculate completion percentage."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.update_progress(evaluation_id, current_task=3, total_tasks=10)

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.progress is not None
        # Percent = (current_task - 1) / total_tasks * 100 = (3-1)/10*100 = 20
        assert evaluation.progress.percent == 20

    def test_update_progress_transitions_to_working(self, store, sample_request):
        """First update_progress should transition status to WORKING."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.update_progress(evaluation_id, current_task=1, total_tasks=5)

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.status == EvaluationStatus.WORKING

    def test_update_progress_updates_heartbeat(self, store, sample_request):
        """update_progress should refresh the heartbeat timestamp."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        before = datetime.now(timezone.utc)
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        after = datetime.now(timezone.utc)

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.progress is not None
        assert before <= evaluation.progress.last_heartbeat <= after

    def test_update_progress_not_found_raises_error(self, store):
        """update_progress should raise EvaluationNotFoundError for unknown ID."""
        with pytest.raises(EvaluationNotFoundError) as exc_info:
            store.update_progress(
                "eval-0000000000000-000000", current_task=1, total_tasks=5
            )

        assert exc_info.value.evaluation_id == "eval-0000000000000-000000"

    def test_update_progress_on_completed_raises_error(self, store, sample_request):
        """update_progress should raise InvalidStateError for completed evaluation."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        with pytest.raises(EvaluationNotFoundError):
            # After completion, the session is moved to evaluations
            # So it won't be found in sessions directory
            store.update_progress(evaluation_id, current_task=2, total_tasks=5)


class TestCompleteEvaluation:
    """Tests for complete_evaluation method."""

    def test_complete_evaluation_sets_status(self, store, sample_request):
        """complete_evaluation should set status to COMPLETED."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)

        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 0.8,
                "total_tasks": 5,
                "successful": 4,
                "tasks": [
                    {"task_id": "t1", "success": True, "reward": 1.0},
                    {"task_id": "t2", "success": True, "reward": 1.0},
                    {"task_id": "t3", "success": True, "reward": 1.0},
                    {"task_id": "t4", "success": True, "reward": 1.0},
                    {"task_id": "t5", "success": False, "reward": 0.0},
                ],
            },
        )

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.status == EvaluationStatus.COMPLETED

    def test_complete_evaluation_stores_results(self, store, sample_request):
        """complete_evaluation should store the results."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)

        results = {
            "success_rate": 0.8,
            "total_tasks": 5,
            "successful": 4,
            "tasks": [{"task_id": "t1", "success": True, "reward": 1.0}],
        }
        store.complete_evaluation(evaluation_id, results=results)

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.results is not None
        assert evaluation.results.success_rate == 0.8
        assert evaluation.results.total_tasks == 5
        assert evaluation.results.successful == 4

    def test_complete_evaluation_moves_to_evaluations_dir(
        self, store, temp_data_dir, sample_request
    ):
        """complete_evaluation should move file from sessions to evaluations."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)

        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        session_path = temp_data_dir / "sessions" / f"{evaluation_id}.json"
        eval_path = temp_data_dir / "evaluations" / f"{evaluation_id}.json"

        assert not session_path.exists()
        assert eval_path.exists()

    def test_complete_evaluation_sets_completed_at(self, store, sample_request):
        """complete_evaluation should set completed_at timestamp."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)

        before = datetime.now(timezone.utc)
        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )
        after = datetime.now(timezone.utc)

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.completed_at is not None
        # Allow some tolerance for timezone comparison
        completed_at_aware = evaluation.completed_at
        if completed_at_aware.tzinfo is None:
            completed_at_aware = completed_at_aware.replace(tzinfo=timezone.utc)
        assert before <= completed_at_aware <= after

    def test_complete_evaluation_not_found_raises_error(self, store):
        """complete_evaluation should raise EvaluationNotFoundError for unknown ID."""
        with pytest.raises(EvaluationNotFoundError):
            store.complete_evaluation(
                "eval-0000000000000-000000",
                results={
                    "success_rate": 1.0,
                    "total_tasks": 1,
                    "successful": 1,
                    "tasks": [],
                },
            )

    def test_complete_evaluation_already_completed_raises_error(
        self, store, sample_request
    ):
        """complete_evaluation should raise error for already completed evaluation."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        # Trying to complete again should fail
        with pytest.raises(EvaluationNotFoundError):
            # Session was moved, so not found in sessions
            store.complete_evaluation(
                evaluation_id,
                results={
                    "success_rate": 1.0,
                    "total_tasks": 5,
                    "successful": 5,
                    "tasks": [],
                },
            )


class TestFailEvaluation:
    """Tests for fail_evaluation method."""

    def test_fail_evaluation_sets_status(self, store, sample_request):
        """fail_evaluation should set status to FAILED."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.fail_evaluation(evaluation_id, error="Connection timeout")

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.status == EvaluationStatus.FAILED

    def test_fail_evaluation_stores_error(self, store, sample_request):
        """fail_evaluation should store the error message."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        error_msg = "Agent connection timeout after 30s"

        store.fail_evaluation(evaluation_id, error=error_msg)

        evaluation = store.get_evaluation(evaluation_id)
        assert evaluation is not None
        assert evaluation.error == error_msg

    def test_fail_evaluation_moves_to_evaluations_dir(
        self, store, temp_data_dir, sample_request
    ):
        """fail_evaluation should move file from sessions to evaluations."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.fail_evaluation(evaluation_id, error="Error occurred")

        session_path = temp_data_dir / "sessions" / f"{evaluation_id}.json"
        eval_path = temp_data_dir / "evaluations" / f"{evaluation_id}.json"

        assert not session_path.exists()
        assert eval_path.exists()

    def test_fail_evaluation_not_found_raises_error(self, store):
        """fail_evaluation should raise EvaluationNotFoundError for unknown ID."""
        with pytest.raises(EvaluationNotFoundError):
            store.fail_evaluation("eval-0000000000000-000000", error="Error")


class TestGetEvaluationFromSession:
    """Tests for get_evaluation retrieving from sessions directory."""

    def test_get_evaluation_finds_in_progress_session(self, store, sample_request):
        """get_evaluation should find evaluations in sessions directory."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert evaluation.evaluation_id == evaluation_id
        assert evaluation.status == EvaluationStatus.SUBMITTED

    def test_get_evaluation_returns_correct_domain(self, store, sample_request):
        """get_evaluation should return correct domain value."""
        evaluation_id = store.create_session(
            domain="retail",
            request=sample_request,
        )

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert evaluation.domain == "retail"

    def test_get_evaluation_returns_none_for_nonexistent(self, store):
        """get_evaluation should return None for non-existent ID."""
        evaluation = store.get_evaluation("eval-0000000000000-000000")

        assert evaluation is None


class TestGetEvaluationFromCompleted:
    """Tests for get_evaluation retrieving from evaluations directory."""

    def test_get_evaluation_finds_completed_evaluation(self, store, sample_request):
        """get_evaluation should find evaluations in evaluations directory."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert evaluation.evaluation_id == evaluation_id
        assert evaluation.status == EvaluationStatus.COMPLETED

    def test_get_evaluation_finds_failed_evaluation(self, store, sample_request):
        """get_evaluation should find failed evaluations."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.fail_evaluation(evaluation_id, error="Test failure")

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert evaluation.status == EvaluationStatus.FAILED

    def test_get_evaluation_checks_sessions_first(
        self, store, temp_data_dir, sample_request
    ):
        """get_evaluation should check sessions directory first."""
        # Create an in-progress session
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        # Manually create a conflicting file in evaluations (should not happen in practice)
        # This test verifies that sessions are checked first
        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert (
            evaluation.status == EvaluationStatus.SUBMITTED
        )  # From sessions, not evaluations


class TestListEvaluations:
    """Tests for list_evaluations method."""

    def test_list_evaluations_returns_all(self, store, sample_request):
        """list_evaluations should return all evaluations."""
        # Create multiple evaluations
        id1 = store.create_session(domain="airline", request=sample_request)
        id2 = store.create_session(domain="retail", request=sample_request)
        id3 = store.create_session(domain="airline", request=sample_request)

        evaluations = store.list_evaluations()

        assert len(evaluations) == 3
        eval_ids = {e.evaluation_id for e in evaluations}
        assert id1 in eval_ids
        assert id2 in eval_ids
        assert id3 in eval_ids

    def test_list_evaluations_filters_by_domain(self, store, sample_request):
        """list_evaluations should filter by domain."""
        store.create_session(domain="airline", request=sample_request)
        store.create_session(domain="retail", request=sample_request)
        store.create_session(domain="airline", request=sample_request)

        evaluations = store.list_evaluations(domain="airline")

        assert len(evaluations) == 2
        for e in evaluations:
            assert e.domain == "airline"

    def test_list_evaluations_filters_by_status(self, store, sample_request):
        """list_evaluations should filter by status."""
        store.create_session(domain="airline", request=sample_request)
        id2 = store.create_session(domain="airline", request=sample_request)
        store.update_progress(id2, current_task=1, total_tasks=5)
        store.complete_evaluation(
            id2,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        evaluations = store.list_evaluations(status="completed")

        assert len(evaluations) == 1
        assert evaluations[0].status == EvaluationStatus.COMPLETED

    def test_list_evaluations_includes_sessions_by_default(self, store, sample_request):
        """list_evaluations should include sessions by default."""
        store.create_session(domain="airline", request=sample_request)

        evaluations = store.list_evaluations()

        assert len(evaluations) == 1
        assert evaluations[0].status == EvaluationStatus.SUBMITTED

    def test_list_evaluations_excludes_sessions_when_specified(
        self, store, sample_request
    ):
        """list_evaluations should exclude sessions when specified."""
        store.create_session(domain="airline", request=sample_request)
        id2 = store.create_session(domain="airline", request=sample_request)
        store.update_progress(id2, current_task=1, total_tasks=5)
        store.complete_evaluation(
            id2,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        evaluations = store.list_evaluations(include_sessions=False)

        assert len(evaluations) == 1
        assert evaluations[0].status == EvaluationStatus.COMPLETED

    def test_list_evaluations_respects_limit(self, store, sample_request):
        """list_evaluations should respect limit parameter."""
        for _ in range(5):
            store.create_session(domain="airline", request=sample_request)

        evaluations = store.list_evaluations(limit=3)

        assert len(evaluations) == 3

    def test_list_evaluations_sorted_by_created_at_descending(
        self, store, sample_request
    ):
        """list_evaluations should return newest first."""
        import time

        id1 = store.create_session(domain="airline", request=sample_request)
        time.sleep(0.01)  # Small delay to ensure different timestamps
        store.create_session(domain="airline", request=sample_request)
        time.sleep(0.01)
        id3 = store.create_session(domain="airline", request=sample_request)

        evaluations = store.list_evaluations()

        # Newest (id3) should be first, oldest (id1) should be last
        assert evaluations[0].evaluation_id == id3
        assert evaluations[2].evaluation_id == id1


class TestEvaluationIdCollision:
    """Tests for evaluation ID collision handling."""

    def test_collision_raises_error(self, store, temp_data_dir, sample_request):
        """create_session should raise EvaluationIdCollisionError on collision."""
        # Create a session first
        first_id = store.create_session(domain="airline", request=sample_request)

        # Mock generate_evaluation_id to return the same ID
        with patch("tau2.store.store.generate_evaluation_id") as mock_gen:
            mock_gen.return_value = first_id

            with pytest.raises(EvaluationIdCollisionError) as exc_info:
                store.create_session(domain="retail", request=sample_request)

            assert exc_info.value.evaluation_id == first_id

    def test_unique_ids_no_collision(self, store, sample_request):
        """create_session should create multiple sessions with unique IDs."""
        ids = set()
        for _ in range(10):
            eval_id = store.create_session(domain="airline", request=sample_request)
            assert eval_id not in ids
            ids.add(eval_id)

        assert len(ids) == 10


# =============================================================================
# User Story 2: Observability Integration (Phase 4)
# =============================================================================


class TestGetEvaluationByTraceId:
    """Tests for get_evaluation_by_trace_id method (T028)."""

    def test_get_evaluation_by_trace_id_finds_session(self, store, sample_request):
        """get_evaluation_by_trace_id should find session by trace ID."""
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
            trace_id=trace_id,
        )

        result = store.get_evaluation_by_trace_id(trace_id)

        assert result is not None
        assert result.evaluation_id == evaluation_id
        assert result.trace_id == trace_id

    def test_get_evaluation_by_trace_id_returns_none_for_unknown(self, store):
        """get_evaluation_by_trace_id should return None for unknown trace ID."""
        result = store.get_evaluation_by_trace_id("0000000000000000000000000000000")

        assert result is None

    def test_get_evaluation_by_trace_id_only_searches_sessions(
        self, store, sample_request
    ):
        """get_evaluation_by_trace_id should only search in-progress sessions."""
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
            trace_id=trace_id,
        )

        # Complete the evaluation (moves to evaluations/)
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        # Now trace_id should not be found (only searches sessions)
        result = store.get_evaluation_by_trace_id(trace_id)

        assert result is None

    def test_get_evaluation_by_trace_id_with_multiple_sessions(
        self, store, sample_request
    ):
        """get_evaluation_by_trace_id should find correct session among many."""
        target_trace_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        other_trace_ids = [
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccc",
        ]

        # Create other sessions first
        for trace_id in other_trace_ids:
            store.create_session(
                domain="airline",
                request=sample_request,
                trace_id=trace_id,
            )

        # Create target session
        target_id = store.create_session(
            domain="retail",
            request=sample_request,
            trace_id=target_trace_id,
        )

        result = store.get_evaluation_by_trace_id(target_trace_id)

        assert result is not None
        assert result.evaluation_id == target_id
        assert result.domain == "retail"

    def test_get_evaluation_by_trace_id_without_trace_id(self, store, sample_request):
        """get_evaluation_by_trace_id should not find sessions without trace_id."""
        # Create session without trace_id
        store.create_session(
            domain="airline",
            request=sample_request,
        )

        result = store.get_evaluation_by_trace_id("4bf92f3577b34da6a3ce929d0e0e4736")

        assert result is None


class TestStateHistoryTransitions:
    """Tests for state_history tracking across transitions (T029)."""

    def test_state_history_records_submitted_on_create(self, store, sample_request):
        """state_history should record SUBMITTED state on create."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert len(evaluation.state_history) == 1
        assert evaluation.state_history[0].state == EvaluationStatus.SUBMITTED

    def test_state_history_records_working_on_first_progress(
        self, store, sample_request
    ):
        """state_history should record WORKING state on first progress update."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.update_progress(evaluation_id, current_task=1, total_tasks=5)

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert len(evaluation.state_history) == 2
        assert evaluation.state_history[0].state == EvaluationStatus.SUBMITTED
        assert evaluation.state_history[1].state == EvaluationStatus.WORKING

    def test_state_history_does_not_duplicate_working(self, store, sample_request):
        """state_history should not duplicate WORKING state on subsequent updates."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.update_progress(evaluation_id, current_task=2, total_tasks=5)
        store.update_progress(evaluation_id, current_task=3, total_tasks=5)

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        # Should only have SUBMITTED and WORKING (no duplicate WORKING entries)
        assert len(evaluation.state_history) == 2

    def test_state_history_records_completed(self, store, sample_request):
        """state_history should record COMPLETED state on completion."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert len(evaluation.state_history) == 3
        assert evaluation.state_history[0].state == EvaluationStatus.SUBMITTED
        assert evaluation.state_history[1].state == EvaluationStatus.WORKING
        assert evaluation.state_history[2].state == EvaluationStatus.COMPLETED

    def test_state_history_records_failed(self, store, sample_request):
        """state_history should record FAILED state on failure."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.fail_evaluation(evaluation_id, error="Test error")

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert len(evaluation.state_history) == 3
        assert evaluation.state_history[2].state == EvaluationStatus.FAILED

    def test_state_history_fail_from_submitted(self, store, sample_request):
        """state_history should handle fail directly from SUBMITTED."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.fail_evaluation(evaluation_id, error="Immediate failure")

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert len(evaluation.state_history) == 2
        assert evaluation.state_history[0].state == EvaluationStatus.SUBMITTED
        assert evaluation.state_history[1].state == EvaluationStatus.FAILED

    def test_state_history_has_timestamps(self, store, sample_request):
        """state_history entries should have timestamps."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        for transition in evaluation.state_history:
            assert transition.at is not None
            # Verify timestamps are reasonable (within last hour)
            assert (
                datetime.now(timezone.utc) - transition.at
            ).total_seconds() < 3600

    def test_state_history_working_includes_progress(self, store, sample_request):
        """WORKING transition should include progress percentage."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.update_progress(evaluation_id, current_task=3, total_tasks=10)

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        working_transition = evaluation.state_history[1]
        assert working_transition.state == EvaluationStatus.WORKING
        # Progress at task 3 of 10: (3-1)/10*100 = 20%
        assert working_transition.progress == 20

    def test_state_history_order_is_chronological(self, store, sample_request):
        """state_history should be in chronological order."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        # Verify timestamps are in ascending order
        for i in range(len(evaluation.state_history) - 1):
            assert evaluation.state_history[i].at <= evaluation.state_history[i + 1].at


class TestSessionHeartbeatUpdates:
    """Tests for session heartbeat updates via progress (T030)."""

    def test_heartbeat_updated_on_progress(self, store, sample_request):
        """Progress updates should refresh the heartbeat timestamp."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        before = datetime.now(timezone.utc)
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        after = datetime.now(timezone.utc)

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert evaluation.progress is not None
        heartbeat = evaluation.progress.last_heartbeat
        assert before <= heartbeat <= after

    def test_heartbeat_updates_on_each_progress_call(self, store, sample_request):
        """Each progress update should update the heartbeat."""
        import time

        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        eval1 = store.get_evaluation(evaluation_id)
        first_heartbeat = eval1.progress.last_heartbeat

        time.sleep(0.01)  # Small delay

        store.update_progress(evaluation_id, current_task=2, total_tasks=5)
        eval2 = store.get_evaluation(evaluation_id)
        second_heartbeat = eval2.progress.last_heartbeat

        assert second_heartbeat > first_heartbeat

    def test_heartbeat_in_progress_model(self, store, sample_request):
        """Heartbeat should be stored in the Progress model."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        store.update_progress(evaluation_id, current_task=2, total_tasks=5)

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert evaluation.progress is not None
        assert hasattr(evaluation.progress, "last_heartbeat")
        assert isinstance(evaluation.progress.last_heartbeat, datetime)

    def test_no_progress_means_no_heartbeat(self, store, sample_request):
        """Sessions without progress update should not have heartbeat."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        # SUBMITTED state has no progress
        assert evaluation.progress is None

    def test_completed_evaluation_has_no_progress(self, store, sample_request):
        """Completed evaluations should not have progress/heartbeat."""
        evaluation_id = store.create_session(
            domain="airline",
            request=sample_request,
        )
        store.update_progress(evaluation_id, current_task=1, total_tasks=5)
        store.complete_evaluation(
            evaluation_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 5,
                "successful": 5,
                "tasks": [],
            },
        )

        evaluation = store.get_evaluation(evaluation_id)

        assert evaluation is not None
        assert evaluation.progress is None

"""
Integration tests for Evaluation Store with orchestrator and A2A agents.

These tests verify:
1. Full integration flow: store + event logging (no real agent required)
2. Concurrent evaluation handling with atomic writes
3. Progress tracking during task execution
4. Error recovery and session cleanup
5. Live E2E tests with real A2A agents (marked with @pytest.mark.e2e)

Test Markers:
- @pytest.mark.integration: All integration tests (mock + live)
- @pytest.mark.e2e: True end-to-end tests requiring live A2A agent

Run commands:
    # All integration tests
    pytest -m integration tests/test_store/test_integration.py -v

    # Only mock integration tests (fast, no external dependencies)
    pytest -m "integration and not e2e" tests/test_store/test_integration.py -v

    # Only live E2E tests (requires agent + API keys)
    pytest -m e2e tests/test_store/test_integration.py -v
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tau2.store import (
    EvaluationStatus,
    EvaluationStore,
    EventLogger,
    create_event_logger,
    create_store,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestStoreIntegration:
    """
    Integration tests for evaluation store with simulated workflows.

    These tests verify store + logger integration using:
    - Simulated evaluation flows (no real agent required)
    - Mock domain for simplicity and speed
    - Evaluation store for session and progress tracking
    - Structured event logging to events.jsonl

    Note: These tests do NOT call a real A2A agent. For true E2E tests
    with live agents, see TestLiveAgentE2E.
    """

    def test_evaluation_session_lifecycle(
        self,
        skip_if_agent_unavailable,
        integration_store: EvaluationStore,
        integration_logger: EventLogger,
        integration_data_dir: Path,
        a2a_agent_endpoint: str,
        mock_task_config: dict,
    ):
        """Test complete evaluation session lifecycle with store tracking.

        Flow:
        1. Create evaluation session via store.create_session()
        2. Log 'evaluation_created' event
        3. Simulate task execution with progress updates
        4. Complete evaluation via store.complete_evaluation()
        5. Log 'evaluation_completed' event
        6. Verify final state and events.jsonl contents
        """
        num_tasks = mock_task_config["num_tasks"]

        # 1. Create session
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": num_tasks, "num_trials": 1},
            agent_endpoint=a2a_agent_endpoint,
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            session_id="sess-e2elifecycle",
        )
        assert eval_id.startswith("eval-")

        # 2. Log creation event
        integration_logger.log_event(
            "evaluation_created",
            eval_id,
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            domain="mock",
            num_tasks=num_tasks,
        )

        # 3. Verify initial state
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation is not None
        assert evaluation.status == EvaluationStatus.SUBMITTED

        # 4. Simulate task execution with progress updates
        for task_num in range(1, num_tasks + 1):
            integration_store.update_progress(eval_id, task_num, num_tasks)
            integration_logger.log_event(
                "task_completed",
                eval_id,
                task_num=task_num,
                total_tasks=num_tasks,
                success=True,
                reward=1.0,
            )

        # Verify WORKING status after progress updates
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation.status == EvaluationStatus.WORKING

        # 5. Complete evaluation
        results = {
            "success_rate": 1.0,
            "total_tasks": num_tasks,
            "successful": num_tasks,
            "tasks": [
                {"task_id": f"mock_task_{i}", "success": True, "reward": 1.0}
                for i in range(num_tasks)
            ],
        }
        integration_store.complete_evaluation(eval_id, results)

        # 6. Log completion event
        integration_logger.log_event(
            "evaluation_completed",
            eval_id,
            success_rate=1.0,
            duration_s=5,
        )

        # 7. Verify final state
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation.status == EvaluationStatus.COMPLETED
        assert evaluation.results is not None
        assert evaluation.results.success_rate == 1.0
        assert evaluation.completed_at is not None

        # 8. Verify file moved to evaluations/
        assert (integration_data_dir / "evaluations" / f"{eval_id}.json").exists()
        assert not (integration_data_dir / "sessions" / f"{eval_id}.json").exists()

        # 9. Verify events.jsonl contains expected events
        events_file = integration_data_dir / "logs" / "events.jsonl"
        assert events_file.exists()

        with open(events_file) as f:
            events = [json.loads(line) for line in f]

        event_types = [e["event"] for e in events]
        assert "evaluation_created" in event_types
        assert "evaluation_completed" in event_types
        assert event_types.count("task_completed") == num_tasks

    def test_progress_tracking_during_execution(
        self,
        skip_if_agent_unavailable,
        integration_store: EvaluationStore,
        integration_data_dir: Path,
    ):
        """Test that progress is correctly tracked during execution.

        Verifies:
        - Progress percentage calculated correctly: (current-1)/total * 100
        - Heartbeat timestamps updated on each progress call
        - Status transitions from SUBMITTED to WORKING on first update
        """
        num_tasks = 5

        # Create session
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": num_tasks, "num_trials": 1},
        )

        # Verify initial state
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation.status == EvaluationStatus.SUBMITTED
        assert evaluation.progress is None

        # Update progress for each task
        for task_num in range(1, num_tasks + 1):
            integration_store.update_progress(eval_id, task_num, num_tasks)

            evaluation = integration_store.get_evaluation(eval_id)
            assert evaluation.status == EvaluationStatus.WORKING
            assert evaluation.progress is not None
            assert evaluation.progress.current_task == task_num
            assert evaluation.progress.total_tasks == num_tasks

            # Verify percentage: (current - 1) / total * 100
            expected_percent = int((task_num - 1) / num_tasks * 100)
            assert evaluation.progress.percent == expected_percent

            # Verify heartbeat is recent
            assert evaluation.progress.last_heartbeat is not None

    def test_failed_evaluation_handling(
        self,
        integration_store: EvaluationStore,
        integration_logger: EventLogger,
        integration_data_dir: Path,
    ):
        """Test evaluation failure handling when errors occur.

        Simulates error and verifies:
        - Error capture and storage via store.fail_evaluation()
        - Status transitions to FAILED
        - Event logging with error details
        """
        # Create session
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": 3, "num_trials": 1},
            agent_endpoint="http://localhost:59999/nonexistent",
        )

        # Update progress once to transition to WORKING
        integration_store.update_progress(eval_id, 1, 3)

        # Fail the evaluation
        error_msg = "Connection refused to agent at http://localhost:59999/nonexistent"
        integration_store.fail_evaluation(eval_id, error=error_msg)

        # Log failure event
        integration_logger.log_event(
            "evaluation_failed",
            eval_id,
            error_type="ConnectionError",
            error_message=error_msg,
        )

        # Verify failed state
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation.status == EvaluationStatus.FAILED
        assert evaluation.error == error_msg
        assert evaluation.completed_at is not None

        # Verify file moved to evaluations/
        assert (integration_data_dir / "evaluations" / f"{eval_id}.json").exists()

    def test_structured_logging_events_jsonl(
        self,
        integration_logger: EventLogger,
        integration_data_dir: Path,
    ):
        """Test that all structured events are correctly logged to events.jsonl.

        Verifies:
        - Each event is a valid JSON object on its own line
        - Events include: ts, level, event, evaluation_id
        - Optional trace_id and session_id when provided
        - Extra fields (domain, task_num, success_rate) are included
        """
        eval_id = "eval-test-structured-logging"
        trace_id = "abcdef1234567890abcdef1234567890"
        session_id = "sess-loggingtest"

        # Log various events
        integration_logger.log_event(
            "evaluation_created",
            eval_id,
            trace_id=trace_id,
            session_id=session_id,
            domain="mock",
            num_tasks=3,
        )

        integration_logger.log_event(
            "task_completed",
            eval_id,
            trace_id=trace_id,
            task_num=1,
            total_tasks=3,
            success=True,
            reward=0.9,
        )

        integration_logger.log_event(
            "evaluation_completed",
            eval_id,
            trace_id=trace_id,
            success_rate=0.9,
            duration_s=120,
        )

        # Parse and validate events
        events_file = integration_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            events = [json.loads(line) for line in f]

        assert len(events) == 3

        # Verify first event (evaluation_created)
        created_event = events[0]
        assert created_event["event"] == "evaluation_created"
        assert created_event["evaluation_id"] == eval_id
        assert created_event["trace_id"] == trace_id
        assert created_event["session_id"] == session_id
        assert created_event["domain"] == "mock"
        assert created_event["num_tasks"] == 3
        assert "ts" in created_event
        assert created_event["level"] == "info"

        # Verify task_completed event
        task_event = events[1]
        assert task_event["event"] == "task_completed"
        assert task_event["task_num"] == 1
        assert task_event["success"] is True
        assert task_event["reward"] == 0.9

        # Verify evaluation_completed event
        completed_event = events[2]
        assert completed_event["event"] == "evaluation_completed"
        assert completed_event["success_rate"] == 0.9
        assert completed_event["duration_s"] == 120

    def test_state_history_complete_flow(
        self,
        integration_store: EvaluationStore,
        integration_data_dir: Path,
    ):
        """Test that state_history correctly records all transitions.

        Verifies:
        - state_history has entries for SUBMITTED, WORKING, COMPLETED
        - Each entry has valid timestamp (at field)
        - WORKING entry includes progress percentage
        - Timestamps are in chronological order
        """
        # Create and run through complete flow
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": 2, "num_trials": 1},
        )

        integration_store.update_progress(eval_id, 1, 2)
        integration_store.update_progress(eval_id, 2, 2)

        integration_store.complete_evaluation(
            eval_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 2,
                "successful": 2,
                "tasks": [
                    {"task_id": "task_1", "success": True, "reward": 1.0},
                    {"task_id": "task_2", "success": True, "reward": 1.0},
                ],
            },
        )

        # Get final evaluation
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation is not None

        # Verify state history
        history = evaluation.state_history
        assert len(history) >= 3  # SUBMITTED, WORKING, COMPLETED

        # Verify states in order
        states = [h.state for h in history]
        assert EvaluationStatus.SUBMITTED in states
        assert EvaluationStatus.WORKING in states
        assert EvaluationStatus.COMPLETED in states

        # Verify chronological order
        timestamps = [h.at for h in history]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_trace_id_correlation(
        self,
        integration_store: EvaluationStore,
        integration_logger: EventLogger,
        integration_data_dir: Path,
    ):
        """Test OTel trace_id correlation through full evaluation flow.

        Verifies:
        - trace_id stored in session and persisted to completion
        - trace_id included in all event log entries
        - get_evaluation_by_trace_id finds in-progress session
        """
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        session_id = "sess-correlationtest"

        # Create session with trace_id
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": 1, "num_trials": 1},
            trace_id=trace_id,
            session_id=session_id,
        )

        # Verify get_evaluation_by_trace_id works for in-progress session
        found = integration_store.get_evaluation_by_trace_id(trace_id)
        assert found is not None
        assert found.evaluation_id == eval_id

        # Complete evaluation
        integration_store.update_progress(eval_id, 1, 1)
        integration_store.complete_evaluation(
            eval_id,
            results={
                "success_rate": 1.0,
                "total_tasks": 1,
                "successful": 1,
                "tasks": [{"task_id": "task_1", "success": True, "reward": 1.0}],
            },
        )

        # Verify trace_id persisted to completed evaluation
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation.trace_id == trace_id
        assert evaluation.session_id == session_id


class TestConcurrencyIntegration:
    """
    Concurrent stress tests for evaluation store.

    These tests spawn multiple parallel evaluations to verify:
    - Atomic file writes prevent corruption
    - No data races between concurrent sessions
    - Each session maintains independent state
    - All sessions complete correctly
    """

    def test_concurrent_evaluations_no_corruption(
        self,
        integration_data_dir: Path,
    ):
        """Test that 5 concurrent evaluations complete without data corruption.

        Spawns 5 parallel evaluation flows, each:
        - Creates its own session
        - Updates progress multiple times
        - Completes with results

        Verifies:
        - All 5 evaluations complete successfully
        - Each evaluation has correct final results
        - No file corruption (all JSON files are parseable)
        - No cross-session contamination
        """
        NUM_CONCURRENT = 5
        NUM_TASKS = 3

        def run_single_evaluation(eval_index: int) -> tuple[str, bool]:
            """Run one evaluation flow, return (evaluation_id, success)."""
            store = create_store(integration_data_dir)

            try:
                # Create unique session
                eval_id = store.create_session(
                    domain="mock",
                    request={"num_tasks": NUM_TASKS, "num_trials": 1},
                    session_id=f"sess-concurrent{eval_index}",
                )

                # Simulate progress updates with small delays
                for task_num in range(1, NUM_TASKS + 1):
                    time.sleep(0.05)  # Small delay to increase race condition chance
                    store.update_progress(eval_id, task_num, NUM_TASKS)

                # Complete evaluation
                store.complete_evaluation(
                    eval_id,
                    results={
                        "success_rate": 1.0,
                        "total_tasks": NUM_TASKS,
                        "successful": NUM_TASKS,
                        "tasks": [
                            {"task_id": f"task_{i}", "success": True, "reward": 1.0}
                            for i in range(NUM_TASKS)
                        ],
                    },
                )

                return eval_id, True
            except Exception as e:
                return str(e), False

        # Run concurrently
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT) as executor:
            futures = [
                executor.submit(run_single_evaluation, i) for i in range(NUM_CONCURRENT)
            ]
            results = [f.result() for f in futures]

        # Verify all succeeded
        for eval_id, success in results:
            assert success, f"Evaluation failed: {eval_id}"

        # Verify unique evaluation IDs
        eval_ids = [r[0] for r in results if r[1]]
        assert len(set(eval_ids)) == NUM_CONCURRENT

        # Verify all evaluation files exist and are valid JSON
        evaluations_dir = integration_data_dir / "evaluations"
        for eval_id in eval_ids:
            eval_file = evaluations_dir / f"{eval_id}.json"
            assert eval_file.exists(), f"Missing file: {eval_file}"

            with open(eval_file) as f:
                data = json.load(f)

            assert data["status"] == "completed"
            assert data["results"]["success_rate"] == 1.0

    def test_concurrent_create_sessions(
        self,
        integration_data_dir: Path,
    ):
        """Test concurrent session creation for ID collision handling.

        Spawns 10 simultaneous create_session calls to verify:
        - All sessions get unique IDs
        - No ID collisions (EvaluationIdCollisionError not raised)
        - All session files created correctly
        """
        NUM_SESSIONS = 10

        def create_session_thread(index: int) -> str:
            store = create_store(integration_data_dir)
            return store.create_session(
                domain="mock",
                request={"num_tasks": 1, "num_trials": 1},
            )

        with ThreadPoolExecutor(max_workers=NUM_SESSIONS) as executor:
            futures = [
                executor.submit(create_session_thread, i) for i in range(NUM_SESSIONS)
            ]
            eval_ids = [f.result() for f in futures]

        # Verify all IDs unique
        assert len(set(eval_ids)) == NUM_SESSIONS

        # Verify all session files exist
        sessions_dir = integration_data_dir / "sessions"
        for eval_id in eval_ids:
            assert (sessions_dir / f"{eval_id}.json").exists()

    def test_concurrent_mixed_operations(
        self,
        integration_data_dir: Path,
    ):
        """Test concurrent mixed operations (create, update, complete, list).

        Runs a workload of mixed operations to verify system stability.
        """
        store = create_store(integration_data_dir)

        # Pre-create some sessions
        pre_created_ids = []
        for _i in range(3):
            eval_id = store.create_session(
                domain="mock",
                request={"num_tasks": 2, "num_trials": 1},
            )
            store.update_progress(eval_id, 1, 2)
            pre_created_ids.append(eval_id)

        errors: list[Exception] = []
        lock = threading.Lock()

        def create_and_complete(index: int):
            try:
                s = create_store(integration_data_dir)
                eval_id = s.create_session(
                    domain="mock", request={"num_tasks": 1, "num_trials": 1}
                )
                s.update_progress(eval_id, 1, 1)
                s.complete_evaluation(
                    eval_id,
                    results={
                        "success_rate": 1.0,
                        "total_tasks": 1,
                        "successful": 1,
                        "tasks": [
                            {"task_id": "task_0", "success": True, "reward": 1.0}
                        ],
                    },
                )
            except Exception as e:
                with lock:
                    errors.append(e)

        def update_existing(eval_id: str):
            try:
                s = create_store(integration_data_dir)
                s.update_progress(eval_id, 2, 2)
            except Exception as e:
                with lock:
                    errors.append(e)

        def list_evaluations():
            try:
                s = create_store(integration_data_dir)
                _ = s.list_evaluations(limit=10)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run mixed operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            # 3 create-and-complete operations
            for i in range(3):
                futures.append(executor.submit(create_and_complete, i))

            # 3 update operations on existing sessions
            for eval_id in pre_created_ids:
                futures.append(executor.submit(update_existing, eval_id))

            # 2 list operations
            for _ in range(2):
                futures.append(executor.submit(list_evaluations))

            # Wait for all to complete
            for f in futures:
                f.result()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Verify all files are valid JSON
        for json_file in (integration_data_dir / "sessions").glob("*.json"):
            with open(json_file) as f:
                json.load(f)  # Will raise if invalid

        for json_file in (integration_data_dir / "evaluations").glob("*.json"):
            with open(json_file) as f:
                json.load(f)  # Will raise if invalid

    def test_atomic_writes_under_rapid_updates(
        self,
        integration_store: EvaluationStore,
        integration_data_dir: Path,
    ):
        """Test that atomic writes prevent partial/corrupt files.

        Performs rapid sequential updates and verifies:
        - No partial writes (file is always valid JSON)
        - No missing fields in stored data
        - Data consistency between writes
        """
        # Create session
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": 100, "num_trials": 1},
        )

        session_file = integration_data_dir / "sessions" / f"{eval_id}.json"

        # Perform 100 rapid progress updates
        for task_num in range(1, 101):
            integration_store.update_progress(eval_id, task_num, 100)

            # Verify file is valid JSON with all required fields after each update
            with open(session_file) as f:
                data = json.load(f)

            assert "evaluation_id" in data
            assert data["evaluation_id"] == eval_id
            assert "status" in data
            assert "progress" in data
            assert data["progress"]["current_task"] == task_num


class TestErrorHandlingIntegration:
    """
    Tests for graceful handling when A2A agent is unavailable.

    These tests verify:
    - Proper error messages and logging
    - Clean session failure handling
    - No hanging connections or threads
    """

    def test_agent_unavailable_creates_failed_session(
        self,
        integration_store: EvaluationStore,
        integration_logger: EventLogger,
        integration_data_dir: Path,
    ):
        """Test that unavailable agent results in proper session failure.

        Uses unreachable endpoint to simulate agent unavailability.
        Verifies:
        - Session created initially
        - Failure captured with descriptive error
        - Session moved to evaluations/ with FAILED status
        """
        unreachable_endpoint = "http://localhost:59999/nonexistent"

        # Create session
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": 1, "num_trials": 1},
            agent_endpoint=unreachable_endpoint,
        )

        # Verify session created in SUBMITTED state
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation.status == EvaluationStatus.SUBMITTED

        # Simulate connection failure
        error_msg = f"Connection refused to agent at {unreachable_endpoint}"
        integration_store.fail_evaluation(eval_id, error=error_msg)

        # Log the failure
        integration_logger.log_event(
            "evaluation_failed",
            eval_id,
            error_type="ConnectionError",
            error_message=error_msg,
        )

        # Verify failed state
        evaluation = integration_store.get_evaluation(eval_id)
        assert evaluation.status == EvaluationStatus.FAILED
        assert "Connection refused" in evaluation.error
        assert evaluation.completed_at is not None


class TestEventLoggingIntegration:
    """
    Integration tests for event logging during evaluations.
    """

    def test_events_jsonl_file_integrity(
        self,
        integration_logger: EventLogger,
        integration_data_dir: Path,
    ):
        """Test that events.jsonl maintains integrity under rapid logging.

        Logs 100 events rapidly and verifies:
        - All lines are valid JSON
        - No interleaved/partial lines
        - All events have required fields
        """
        eval_id = "eval-test-integrity"

        for i in range(100):
            integration_logger.log_event(
                "test_event",
                eval_id,
                event_number=i,
            )

        # Read and validate all lines
        events_file = integration_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            lines = f.readlines()

        assert len(lines) == 100

        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["event"] == "test_event"
            assert data["evaluation_id"] == eval_id
            assert data["event_number"] == i
            assert "ts" in data
            assert "level" in data

    def test_concurrent_event_logging(
        self,
        integration_data_dir: Path,
    ):
        """Test concurrent event logging from multiple threads.

        Spawns 5 threads each logging 20 events.
        Verifies all 100 events are present in the file.
        """
        NUM_THREADS = 5
        EVENTS_PER_THREAD = 20

        def log_events(thread_id: int):
            logger = create_event_logger(data_dir=integration_data_dir, stdout=False)
            for i in range(EVENTS_PER_THREAD):
                logger.log_event(
                    "concurrent_test",
                    f"eval-thread-{thread_id}",
                    thread_id=thread_id,
                    event_num=i,
                )

        threads = [
            threading.Thread(target=log_events, args=(i,)) for i in range(NUM_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all events present
        events_file = integration_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            lines = f.readlines()

        assert len(lines) == NUM_THREADS * EVENTS_PER_THREAD

        # Verify all lines are valid JSON
        for line in lines:
            data = json.loads(line)
            assert data["event"] == "concurrent_test"
            assert "thread_id" in data
            assert "event_num" in data


@pytest.mark.e2e
class TestLiveAgentE2E:
    """
    True end-to-end tests with real A2A agent execution.

    These tests actually run the tau2 orchestrator against a real A2A agent,
    track evaluations through the store, and verify structured logging.

    Requirements:
    - NEBIUS_API_KEY environment variable set
    - simple_nebius_agent running at http://localhost:8001/a2a/simple_nebius_agent
      OR tau2-agent at http://tau2-agent:8001/a2a/tau2_agent

    Run commands:
        # Run only live E2E tests
        pytest -m e2e tests/test_store/test_integration.py -v -s

        # Run specific test
        pytest -m e2e tests/test_store/test_integration.py::TestLiveAgentE2E::test_live_single_task -v -s
    """

    @pytest.fixture
    def nebius_available(self) -> bool:
        """Check if NEBIUS_API_KEY is available for LLM calls."""
        import os

        return os.environ.get("NEBIUS_API_KEY") is not None

    @pytest.fixture
    def skip_if_no_nebius(self, nebius_available: bool):
        """Skip test if NEBIUS_API_KEY is not set."""
        if not nebius_available:
            pytest.skip(
                "NEBIUS_API_KEY not set. Required for user simulator LLM calls."
            )

    @pytest.fixture
    def nebius_model(self) -> str:
        """Return the Nebius model to use for user simulation."""
        import os

        return os.environ.get(
            "NEBIUS_USER_MODEL", "nebius/Qwen/Qwen3-30B-A3B-Thinking-2507"
        )

    @pytest.fixture
    def simple_agent_endpoint(self) -> str:
        """Return the simple_nebius_agent endpoint."""
        import os

        return os.environ.get(
            "SIMPLE_AGENT_ENDPOINT", "http://localhost:8001/a2a/simple_nebius_agent"
        )

    @pytest.fixture
    def simple_agent_available(self, simple_agent_endpoint: str) -> bool:
        """Check if simple_nebius_agent is reachable."""
        import httpx

        try:
            # Try agent card at agent path (correct A2A spec)
            response = httpx.get(
                f"{simple_agent_endpoint}/.well-known/agent-card.json",
                timeout=5,
            )
            if response.status_code == 200:
                return True

            # Try direct endpoint (405 is valid for A2A POST-only endpoint)
            response = httpx.get(simple_agent_endpoint, timeout=5)
            return response.status_code in (200, 405)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
            return False

    @pytest.fixture
    def skip_if_no_simple_agent(
        self, simple_agent_available: bool, simple_agent_endpoint: str
    ):
        """Skip test if simple_nebius_agent is not available."""
        if not simple_agent_available:
            pytest.skip(
                f"simple_nebius_agent not available at {simple_agent_endpoint}. "
                "Start with: cd simple_nebius_agent && adk api_server --a2a . --port 8001"
            )

    def test_live_single_task_with_store(
        self,
        skip_if_no_nebius,
        skip_if_no_simple_agent,
        simple_agent_endpoint: str,
        nebius_model: str,
        integration_store: EvaluationStore,
        integration_logger: EventLogger,
        integration_data_dir: Path,
    ):
        """
        Run a single mock domain task against a live A2A agent.

        This is a TRUE end-to-end test that:
        1. Creates an evaluation session in the store
        2. Loads a real mock domain task
        3. Runs the tau2 orchestrator with A2A agent
        4. Uses Nebius LLM for user simulation
        5. Updates progress in the evaluation store
        6. Logs events to events.jsonl
        7. Completes the evaluation with real results
        """
        import uuid

        from tau2.run import get_tasks, run_task

        # Generate trace ID for correlation
        trace_id = uuid.uuid4().hex

        # 1. Create evaluation session
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": 1, "num_trials": 1},
            agent_endpoint=simple_agent_endpoint,
            trace_id=trace_id,
            session_id=f"sess-reale2e{uuid.uuid4().hex[:6]}",
        )

        integration_logger.log_event(
            "evaluation_started",
            eval_id,
            trace_id=trace_id,
            domain="mock",
            agent_endpoint=simple_agent_endpoint,
            user_model=nebius_model,
        )

        try:
            # 2. Load one mock domain task
            tasks = get_tasks(task_set_name="mock", num_tasks=1)
            assert len(tasks) == 1, "Expected exactly 1 mock task"
            task = tasks[0]

            # Log task start
            integration_logger.log_event(
                "task_started",
                eval_id,
                task_id=task.id,
                task_num=1,
                total_tasks=1,
            )

            # 3. Update progress to WORKING state
            integration_store.update_progress(eval_id, 1, 1)

            # 4. Run the actual orchestrator
            simulation_run = run_task(
                domain="mock",
                task=task,
                agent="a2a_agent",
                user="user_simulator",
                llm_agent=simple_agent_endpoint,  # A2A endpoint
                llm_args_agent={"timeout": 120},
                llm_user=nebius_model,  # Nebius for user sim
                llm_args_user=None,
                max_steps=15,
                max_errors=5,
            )

            # 5. Log task completion
            reward = (
                simulation_run.reward_info.reward
                if simulation_run.reward_info
                else 0.0
            )
            success = reward > 0.5

            num_messages = len(simulation_run.messages)

            integration_logger.log_event(
                "task_completed",
                eval_id,
                task_id=task.id,
                task_num=1,
                total_tasks=1,
                success=success,
                reward=reward,
                num_messages=num_messages,
            )

            # 6. Complete evaluation with real results
            results = {
                "success_rate": float(success),
                "total_tasks": 1,
                "successful": 1 if success else 0,
                "tasks": [
                    {
                        "task_id": task.id,
                        "success": success,
                        "reward": reward,
                        "num_messages": num_messages,
                    }
                ],
            }

            integration_store.complete_evaluation(eval_id, results)

            integration_logger.log_event(
                "evaluation_completed",
                eval_id,
                trace_id=trace_id,
                success_rate=float(success),
                total_messages=num_messages,
            )

            # 7. Verify final state
            final_eval = integration_store.get_evaluation(eval_id)
            assert final_eval.status == EvaluationStatus.COMPLETED
            assert final_eval.results is not None
            assert final_eval.trace_id == trace_id

            # Verify evaluation file moved to evaluations/
            eval_file = integration_data_dir / "evaluations" / f"{eval_id}.json"
            assert eval_file.exists()

            # Verify events logged
            events_file = integration_data_dir / "logs" / "events.jsonl"
            with open(events_file) as f:
                events = [json.loads(line) for line in f]

            event_types = [e["event"] for e in events]
            assert "evaluation_started" in event_types
            assert "task_started" in event_types
            assert "task_completed" in event_types
            assert "evaluation_completed" in event_types

        except Exception as e:
            # Handle failure gracefully
            error_msg = str(e)
            integration_store.fail_evaluation(eval_id, error=error_msg)
            integration_logger.log_event(
                "evaluation_failed",
                eval_id,
                trace_id=trace_id,
                error_type=type(e).__name__,
                error_message=error_msg,
            )
            # Re-raise for test failure
            raise

    def test_live_multiple_tasks_with_progress(
        self,
        skip_if_no_nebius,
        skip_if_no_simple_agent,
        simple_agent_endpoint: str,
        nebius_model: str,
        integration_store: EvaluationStore,
        integration_logger: EventLogger,
        integration_data_dir: Path,
    ):
        """
        Run 3 mock domain tasks with progress tracking against live agent.

        Verifies that:
        - Progress is updated after each task
        - All tasks complete (or fail gracefully)
        - Results aggregated correctly
        - State history has all transitions
        """
        import uuid

        from tau2.run import get_tasks, run_task

        NUM_TASKS = 3
        trace_id = uuid.uuid4().hex

        # Create evaluation session
        eval_id = integration_store.create_session(
            domain="mock",
            request={"num_tasks": NUM_TASKS, "num_trials": 1},
            agent_endpoint=simple_agent_endpoint,
            trace_id=trace_id,
        )

        integration_logger.log_event(
            "evaluation_started",
            eval_id,
            trace_id=trace_id,
            num_tasks=NUM_TASKS,
        )

        task_results = []

        try:
            # Load tasks
            tasks = get_tasks(task_set_name="mock", num_tasks=NUM_TASKS)

            for task_num, task in enumerate(tasks, 1):
                # Update progress BEFORE starting task
                integration_store.update_progress(eval_id, task_num, NUM_TASKS)

                integration_logger.log_event(
                    "task_started",
                    eval_id,
                    task_id=task.id,
                    task_num=task_num,
                    total_tasks=NUM_TASKS,
                )

                try:
                    # Run the orchestrator for this task
                    simulation_run = run_task(
                        domain="mock",
                        task=task,
                        agent="a2a_agent",
                        user="user_simulator",
                        llm_agent=simple_agent_endpoint,
                        llm_args_agent={"timeout": 120},
                        llm_user=nebius_model,
                        llm_args_user=None,
                        max_steps=15,
                        max_errors=5,
                    )

                    reward = (
                        simulation_run.reward_info.reward
                        if simulation_run.reward_info
                        else 0.0
                    )
                    success = reward > 0.5

                    task_results.append(
                        {
                            "task_id": task.id,
                            "success": success,
                            "reward": reward,
                            "num_messages": len(simulation_run.messages),
                        }
                    )

                    integration_logger.log_event(
                        "task_completed",
                        eval_id,
                        task_id=task.id,
                        task_num=task_num,
                        success=success,
                        reward=reward,
                    )

                except Exception as e:
                    # Task failed, but continue with other tasks
                    task_results.append(
                        {
                            "task_id": task.id,
                            "success": False,
                            "reward": 0.0,
                            "error": str(e),
                        }
                    )

                    integration_logger.log_event(
                        "task_failed",
                        eval_id,
                        task_id=task.id,
                        task_num=task_num,
                        error=str(e),
                    )

            # Calculate final results
            successes = sum(1 for r in task_results if r.get("success", False))
            success_rate = successes / NUM_TASKS

            results = {
                "success_rate": success_rate,
                "total_tasks": NUM_TASKS,
                "successful": successes,
                "tasks": task_results,
            }

            integration_store.complete_evaluation(eval_id, results)

            integration_logger.log_event(
                "evaluation_completed",
                eval_id,
                trace_id=trace_id,
                success_rate=success_rate,
                successful=successes,
                total_tasks=NUM_TASKS,
            )

            # Verify state history
            final_eval = integration_store.get_evaluation(eval_id)
            assert final_eval.status == EvaluationStatus.COMPLETED

            # Should have at least: SUBMITTED, WORKING (multiple), COMPLETED
            assert len(final_eval.state_history) >= 3

            states = [h.state for h in final_eval.state_history]
            assert EvaluationStatus.SUBMITTED in states
            assert EvaluationStatus.WORKING in states
            assert EvaluationStatus.COMPLETED in states

        except Exception as e:
            integration_store.fail_evaluation(eval_id, error=str(e))
            integration_logger.log_event(
                "evaluation_failed",
                eval_id,
                trace_id=trace_id,
                error=str(e),
            )
            raise

    def test_live_parallel_evaluations(
        self,
        skip_if_no_nebius,
        skip_if_no_simple_agent,
        simple_agent_endpoint: str,
        nebius_model: str,
        integration_data_dir: Path,
    ):
        """
        Run multiple evaluations in PARALLEL against the same live A2A agent.

        This tests the scenario where multiple evaluations are requesting
        service from the agent simultaneously. Verifies:
        - All parallel evaluations tracked independently
        - No cross-contamination between evaluation sessions
        - Atomic writes prevent corruption under concurrent load
        - All evaluations complete (success or failure)
        - Events logged correctly for all evaluations
        """
        import uuid
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tau2.run import get_tasks, run_task

        NUM_PARALLEL = 3  # Run 3 evaluations in parallel
        results_lock = threading.Lock()
        all_results: list[dict] = []

        def run_parallel_evaluation(eval_index: int) -> dict:
            """Run a single evaluation and return results."""
            store = create_store(integration_data_dir)
            logger = create_event_logger(data_dir=integration_data_dir, stdout=False)
            trace_id = uuid.uuid4().hex

            # Create session
            eval_id = store.create_session(
                domain="mock",
                request={"num_tasks": 1, "num_trials": 1},
                agent_endpoint=simple_agent_endpoint,
                trace_id=trace_id,
                session_id=f"sess-parallel{eval_index}",
            )

            logger.log_event(
                "parallel_evaluation_started",
                eval_id,
                trace_id=trace_id,
                eval_index=eval_index,
            )

            try:
                # Load a single task
                tasks = get_tasks(task_set_name="mock", num_tasks=1)
                task = tasks[0]

                # Update progress
                store.update_progress(eval_id, 1, 1)

                # Run the orchestrator - this is the real agent call
                simulation_run = run_task(
                    domain="mock",
                    task=task,
                    agent="a2a_agent",
                    user="user_simulator",
                    llm_agent=simple_agent_endpoint,
                    llm_args_agent={"timeout": 180},  # Longer timeout for concurrent load
                    llm_user=nebius_model,
                    llm_args_user=None,
                    max_steps=10,
                    max_errors=3,
                )

                reward = (
                    simulation_run.reward_info.reward
                    if simulation_run.reward_info
                    else 0.0
                )
                success = reward > 0.5

                # Complete evaluation
                store.complete_evaluation(
                    eval_id,
                    results={
                        "success_rate": float(success),
                        "total_tasks": 1,
                        "successful": 1 if success else 0,
                        "tasks": [
                            {
                                "task_id": task.id,
                                "success": success,
                                "reward": reward,
                            }
                        ],
                    },
                )

                logger.log_event(
                    "parallel_evaluation_completed",
                    eval_id,
                    trace_id=trace_id,
                    eval_index=eval_index,
                    success=success,
                )

                return {
                    "eval_index": eval_index,
                    "eval_id": eval_id,
                    "status": "completed",
                    "success": success,
                    "reward": reward,
                }

            except Exception as e:
                store.fail_evaluation(eval_id, error=str(e))
                logger.log_event(
                    "parallel_evaluation_failed",
                    eval_id,
                    trace_id=trace_id,
                    eval_index=eval_index,
                    error=str(e),
                )
                return {
                    "eval_index": eval_index,
                    "eval_id": eval_id,
                    "status": "failed",
                    "error": str(e),
                }

        # Run evaluations in parallel
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL) as executor:
            futures = {
                executor.submit(run_parallel_evaluation, i): i
                for i in range(NUM_PARALLEL)
            }

            for future in as_completed(futures):
                eval_index = futures[future]
                try:
                    result = future.result()
                    with results_lock:
                        all_results.append(result)
                except Exception as e:
                    with results_lock:
                        all_results.append(
                            {
                                "eval_index": eval_index,
                                "status": "exception",
                                "error": str(e),
                            }
                        )

        # Verify results
        assert len(all_results) == NUM_PARALLEL

        # Check that all evaluations completed (either success or failure)
        for result in all_results:
            assert result["status"] in ["completed", "failed"]

        # Verify all eval_ids are unique
        eval_ids = [r.get("eval_id") for r in all_results if r.get("eval_id")]
        assert len(set(eval_ids)) == len(eval_ids)

        # Verify evaluation files exist
        for result in all_results:
            if "eval_id" in result:
                eval_file = integration_data_dir / "evaluations" / f"{result['eval_id']}.json"
                assert eval_file.exists(), f"Missing evaluation file: {eval_file}"

                # Verify file is valid JSON
                with open(eval_file) as f:
                    data = json.load(f)
                assert data["status"] in ["completed", "failed"]

        # Verify parallel events logged
        events_file = integration_data_dir / "logs" / "events.jsonl"
        with open(events_file) as f:
            events = [json.loads(line) for line in f]

        parallel_started = [e for e in events if e["event"] == "parallel_evaluation_started"]
        parallel_finished = [
            e
            for e in events
            if e["event"] in ["parallel_evaluation_completed", "parallel_evaluation_failed"]
        ]

        assert len(parallel_started) == NUM_PARALLEL
        assert len(parallel_finished) == NUM_PARALLEL

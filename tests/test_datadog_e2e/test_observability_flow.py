"""
E2E tests for Datadog observability flow.

Tests validate:
1. Concurrent A2A evaluations complete without deadlock
2. Each evaluation persists to a separate file with valid results
3. SSE streaming returns properly structured events
4. emit_metrics.py can process evaluation results

Uses real evaluations against the mock domain via traced subprocess servers.
Test duration: ~4-5 minutes (includes 3 concurrent + 1 sequential evaluation).
"""

import asyncio
import json
import os
import sys
from io import StringIO

import httpx
import pytest
from loguru import logger

from tests.test_datadog_e2e.conftest import TracedServer, send_a2a_evaluation_request


@pytest.mark.datadog_e2e
class TestServerSetup:
    """Smoke test for ADK server infrastructure."""

    @pytest.mark.asyncio
    async def test_servers_accessible_on_separate_ports(self, traced_adk_server):
        """Verify tau2_agent and test agent run on separate ports.

        Separate ports prevent async deadlock when tau2_agent calls the test agent.
        """
        async with httpx.AsyncClient() as client:
            # Verify tau2_agent server
            tau2_response = await client.get(
                f"{traced_adk_server.tau2_agent_endpoint}/.well-known/agent-card.json"
            )
            assert tau2_response.status_code == 200
            tau2_card = tau2_response.json()
            assert tau2_card["name"] == "tau2_agent"

            # Verify mock_test_agent server
            mock_response = await client.get(
                f"{traced_adk_server.mock_agent_endpoint}/.well-known/agent-card.json"
            )
            assert mock_response.status_code == 200
            mock_card = mock_response.json()
            assert mock_card["name"] == "simple_nebius_agent"

            # Verify they're on different ports (async deadlock fix)
            tau2_port = traced_adk_server.tau2_agent_endpoint.split(":")[-1].split("/")[0]
            mock_port = traced_adk_server.mock_agent_endpoint.split(":")[-1].split("/")[0]
            assert tau2_port != mock_port, (
                f"tau2_agent ({tau2_port}) and simple_nebius_agent ({mock_port}) "
                "must be on different ports to avoid async deadlock"
            )


@pytest.mark.datadog_e2e
class TestA2AObservabilityFlow:
    """Tests for A2A evaluation flow with file persistence and SSE streaming."""

    @pytest.mark.asyncio
    async def test_full_a2a_evaluation_flow(
        self,
        traced_adk_server: TracedServer,
    ):
        """Run a single evaluation and validate SSE events, file persistence, and state history.

        Verifies:
        - SSE events are well-formed JSON (not raw/unparseable)
        - Evaluation file is created with correct domain and metadata
        - State history shows valid progression (submitted/working -> completed/failed)
        """
        os.environ["TAU2_DATA_DIR"] = str(traced_adk_server.data_dir)

        files_before = set(traced_adk_server.evaluations_dir.glob("*.json"))
        events_collected = []

        # Run ONE evaluation via A2A JSON-RPC
        async for event in send_a2a_evaluation_request(
            endpoint=traced_adk_server.tau2_agent_endpoint,
            domain="mock",
            agent_endpoint=traced_adk_server.mock_agent_endpoint,
            num_tasks=1,
            num_trials=1,
            stream=True,
        ):
            events_collected.append(event)

        # === Verify SSE events received ===
        assert len(events_collected) > 0, "No SSE events received from A2A request"

        # === Verify SSE event structure ===
        # Each event should be parseable (not raw data)
        for event in events_collected:
            assert "_raw" not in event, f"Malformed SSE event: {event}"

        # Check that we have at least one event with result or error
        has_result_event = any(
            "result" in event or "error" in event for event in events_collected
        )
        assert has_result_event, "No result/error event in SSE stream"

        # === Verify store persistence ===
        files_after = set(traced_adk_server.evaluations_dir.glob("*.json"))
        new_files = files_after - files_before
        assert len(new_files) > 0, "No new evaluation files created"

        eval_path = list(new_files)[0]
        with open(eval_path) as f:
            eval_data = json.load(f)

        # Check evaluation completed successfully
        if eval_data["status"] != "completed":
            error_msg = eval_data.get("error", "No error message")
            pytest.fail(f"Evaluation failed: {error_msg}")

        assert "results" in eval_data, "Missing results in evaluation"
        assert eval_data["domain"] == "mock", f"Wrong domain: {eval_data.get('domain')}"
        assert "simulations" in eval_data["results"], "Missing simulations in results"

        # === Verify state progression ===
        assert eval_data["status"] in ("completed", "failed"), (
            f"Final state should be completed/failed, got {eval_data['status']}"
        )

        # Verify state_history exists and has proper progression
        assert "state_history" in eval_data, "Missing state_history"
        state_history = eval_data["state_history"]
        assert len(state_history) >= 1, "state_history should have at least 1 entry"

        # Check states are valid and in expected order
        valid_states = {"submitted", "working", "completed", "failed", "abandoned"}
        states_seen = []
        for entry in state_history:
            state = entry.get("state")
            assert state in valid_states, f"Invalid state in state_history: {state}"
            states_seen.append(state)

        # Verify state progression order (should start early, end with final)
        if len(states_seen) >= 2:
            # First state should be submitted or working
            assert states_seen[0] in {"submitted", "working"}, (
                f"First state should be submitted/working, got {states_seen[0]}"
            )
            # Last state should be completed/failed
            assert states_seen[-1] in {"completed", "failed"}, (
                f"Last state should be completed/failed, got {states_seen[-1]}"
            )

        # === Verify metadata for Datadog tracing ===
        assert "domain" in eval_data, "Missing domain in stored evaluation"
        assert eval_data["domain"] == "mock"
        assert "agent_endpoint" in eval_data, "Missing agent_endpoint"
        assert "evaluation_id" in eval_data, "Missing evaluation_id"

    @pytest.mark.asyncio
    async def test_concurrent_evaluations_complete_with_valid_results(
        self,
        traced_adk_server: TracedServer,
    ):
        """Run 3 evaluations concurrently and validate each produces a valid result file.

        Verifies:
        - All concurrent requests complete without deadlock (via asyncio.gather)
        - Each evaluation creates a separate file with unique evaluation_id
        - Completed evaluations contain results.simulations with reward and task_id
        """
        os.environ["TAU2_DATA_DIR"] = str(traced_adk_server.data_dir)

        files_before = set(traced_adk_server.evaluations_dir.glob("*.json"))
        num_concurrent = 3

        async def run_single_evaluation(eval_id: int) -> dict:
            """Run a single evaluation and return results."""
            events = []
            async for event in send_a2a_evaluation_request(
                endpoint=traced_adk_server.tau2_agent_endpoint,
                domain="mock",
                agent_endpoint=traced_adk_server.mock_agent_endpoint,
                num_tasks=1,
                num_trials=1,
                stream=True,
            ):
                events.append(event)
            return {"eval_id": eval_id, "events": events, "success": len(events) > 0}

        # Run evaluations CONCURRENTLY using asyncio.gather
        results = await asyncio.gather(
            *[run_single_evaluation(i) for i in range(num_concurrent)],
            return_exceptions=True,
        )

        # === Verify all evaluations returned events ===
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Evaluation {i} raised exception: {result}")
            assert result.get("success"), f"Evaluation {i} received no SSE events"

        # === Verify separate files created for each evaluation ===
        files_after = set(traced_adk_server.evaluations_dir.glob("*.json"))
        new_files = files_after - files_before
        assert len(new_files) >= num_concurrent, (
            f"Expected at least {num_concurrent} new evaluation files, "
            f"got {len(new_files)}. Evaluations may be overwriting each other."
        )

        # === Validate each evaluation file has complete, valid results ===
        evaluation_ids = set()
        completed_count = 0

        for eval_path in new_files:
            with open(eval_path) as f:
                eval_data = json.load(f)

            # Each file should have a unique evaluation_id
            eval_id = eval_data.get("evaluation_id")
            assert eval_id is not None, f"Missing evaluation_id in {eval_path}"
            assert eval_id not in evaluation_ids, (
                f"Duplicate evaluation_id {eval_id} - files may be corrupted"
            )
            evaluation_ids.add(eval_id)

            # Verify domain is correct
            assert eval_data.get("domain") == "mock", (
                f"Wrong domain in {eval_path}: {eval_data.get('domain')}"
            )

            # Check status - should be completed for successful evaluations
            status = eval_data.get("status")
            if status == "completed":
                completed_count += 1

                # Validate results structure for completed evaluations
                assert "results" in eval_data, f"Missing results in {eval_path}"
                results_data = eval_data["results"]

                assert "simulations" in results_data, (
                    f"Missing simulations in {eval_path}"
                )
                simulations = results_data["simulations"]
                assert len(simulations) > 0, f"Empty simulations in {eval_path}"

                # Validate each simulation has required fields
                for sim in simulations:
                    # reward is nested in reward_info
                    assert "reward_info" in sim, f"Missing reward_info in simulation: {sim}"
                    assert "reward" in sim["reward_info"], f"Missing reward in reward_info: {sim}"
                    assert "task_id" in sim, f"Missing task_id in simulation: {sim}"

            elif status == "failed":
                # Failed is acceptable but should have error info
                assert "error" in eval_data or "state_history" in eval_data, (
                    f"Failed evaluation missing error details: {eval_path}"
                )

        # At least some evaluations should complete successfully
        assert completed_count > 0, (
            f"No evaluations completed successfully out of {len(new_files)} files"
        )


@pytest.mark.datadog_e2e
class TestMetricsEmission:
    """Tests for MetricsEmitter dry-run logging (no actual Datadog emission)."""

    def test_emit_metrics_dry_run_mode(self):
        """Verify dry_run=True disables the DogStatsD client."""
        from experiments.datadog.scripts.emit_metrics import MetricsEmitter

        emitter = MetricsEmitter(dry_run=True)

        # Should not have statsd client in dry-run mode
        assert emitter._statsd is None

    def test_emit_task_metrics(self):
        """Verify emit_task_metrics logs tau2.task.reward, duration_seconds, and success."""
        from experiments.datadog.scripts.emit_metrics import MetricsEmitter

        emitter = MetricsEmitter(dry_run=True)

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, format="{message}", level="INFO")

        try:
            emitter.emit_task_metrics(
                task_id="test-task-1",
                domain="mock",
                evaluation_id="test-eval-1",
                reward=0.85,
                duration_seconds=15.5,
                steps=5,
            )
        finally:
            logger.remove()
            logger.add(sys.stderr, level="INFO")

        logs = log_output.getvalue()

        assert "tau2.task.reward" in logs
        assert "0.85" in logs
        assert "tau2.task.duration_seconds" in logs
        assert "tau2.task.success" in logs

    def test_emit_evaluation_metrics(self):
        """Verify emit_evaluation_metrics logs pass_rate, avg_reward, and tasks_total."""
        from experiments.datadog.scripts.emit_metrics import MetricsEmitter

        emitter = MetricsEmitter(dry_run=True)

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, format="{message}", level="INFO")

        try:
            emitter.emit_evaluation_metrics(
                evaluation_id="test-eval-2",
                domain="mock",
                pass_rate=75.0,
                avg_reward=0.78,
                total_tasks=10,
            )
        finally:
            logger.remove()
            logger.add(sys.stderr, level="INFO")

        logs = log_output.getvalue()

        assert "tau2.evaluation.pass_rate" in logs
        assert "75.0" in logs
        assert "tau2.evaluation.avg_reward" in logs
        assert "tau2.evaluation.tasks_total" in logs

    def test_emit_metrics_from_real_evaluation(
        self,
        traced_adk_server: TracedServer,
    ):
        """Verify process_evaluation emits metrics from a real evaluation file.

        Uses completed evaluation files from prior tests to ensure emit_metrics.py
        is compatible with the actual data format produced by tau2_agent.
        """
        os.environ["TAU2_DATA_DIR"] = str(traced_adk_server.data_dir)

        from experiments.datadog.scripts.emit_metrics import (
            MetricsEmitter,
            load_evaluation,
            process_evaluation,
        )

        # Find any completed evaluation file (may be from previous tests)
        eval_files = list(traced_adk_server.evaluations_dir.glob("*.json"))
        if not eval_files:
            pytest.skip("No evaluation files available for metrics test")

        # Find a completed evaluation
        eval_path = None
        for path in eval_files:
            with open(path) as f:
                data = json.load(f)
                if data.get("status") == "completed":
                    eval_path = path
                    break

        if eval_path is None:
            pytest.skip("No completed evaluation files for metrics test")

        evaluation_id = eval_path.stem
        eval_data = load_evaluation(eval_path)

        emitter = MetricsEmitter(dry_run=True)

        log_output = StringIO()
        logger.remove()
        logger.add(log_output, format="{message}", level="INFO")

        try:
            process_evaluation(emitter, eval_data, evaluation_id)
        finally:
            logger.remove()
            logger.add(sys.stderr, level="INFO")

        logs = log_output.getvalue()

        # Verify key metrics were emitted
        assert "tau2.task.reward" in logs, "tau2.task.reward metric not emitted"
        assert "tau2.evaluation.pass_rate" in logs, "tau2.evaluation.pass_rate not emitted"
        assert "tau2.evaluation.avg_reward" in logs, "tau2.evaluation.avg_reward not emitted"

"""E2E tests for EvaluationStore persistence."""

import asyncio
import json
from pathlib import Path

import pytest

from tests.test_a2a_e2e.conftest import A2EServer, send_a2a_evaluation_request

pytestmark = pytest.mark.a2a_e2e


def test_evaluation_file_structure(evaluation_file: Path, evaluation_data: dict):
    """Verify evaluation file is valid JSON with required fields."""
    # File exists and is valid JSON (implicitly verified by fixtures)
    assert evaluation_file.exists()
    assert isinstance(evaluation_data, dict)

    # Required fields present
    required = ["evaluation_id", "domain", "status"]
    for field in required:
        assert field in evaluation_data, (
            f"Missing {field}. Keys: {list(evaluation_data.keys())}"
        )

    # Field values are valid
    assert evaluation_data["domain"] == "mock"
    assert evaluation_data["status"] in ("completed", "failed", "abandoned")


def test_evaluation_metadata(evaluation_data: dict, a2e_server: A2EServer):
    """Verify evaluation has agent_endpoint, timestamp, and results if completed."""
    # Has agent endpoint
    assert "agent_endpoint" in evaluation_data
    assert a2e_server.mock_agent_endpoint in evaluation_data["agent_endpoint"]

    # Has timestamp
    has_time = any(
        k in evaluation_data
        for k in ["created_at", "timestamp", "started_at", "submitted_at"]
    )
    assert has_time, f"No timestamp. Keys: {list(evaluation_data.keys())}"

    # Completed evaluations have results
    if evaluation_data.get("status") == "completed":
        assert "results" in evaluation_data
        results = evaluation_data["results"]
        assert isinstance(results, dict)
        if "simulations" in results:
            assert isinstance(results["simulations"], list)


@pytest.mark.asyncio
async def test_concurrent_evaluations_unique_files(a2e_server: A2EServer):
    """Verify concurrent evaluations create separate files with unique IDs."""
    files_before = set(a2e_server.evaluations_dir.glob("*.json"))

    async def run_eval():
        events = []
        async for event in send_a2a_evaluation_request(
            endpoint=a2e_server.tau2_agent_endpoint,
            domain="mock",
            agent_endpoint=a2e_server.mock_agent_endpoint,
            num_tasks=1,
            num_trials=1,
        ):
            events.append(event)
        return events

    results = await asyncio.gather(run_eval(), run_eval(), return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            pytest.fail(f"Evaluation {i} failed: {result}")

    files_after = set(a2e_server.evaluations_dir.glob("*.json"))
    new_files = files_after - files_before

    assert len(new_files) >= 2, f"Expected 2+ files, got {len(new_files)}"

    # Verify unique evaluation IDs
    eval_ids = set()
    for f in new_files:
        with open(f) as fp:
            data = json.load(fp)
        eid = data.get("evaluation_id")
        assert eid is not None, f"Missing evaluation_id in {f}"
        assert eid not in eval_ids, f"Duplicate ID: {eid}"
        eval_ids.add(eid)

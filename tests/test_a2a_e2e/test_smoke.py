"""
Smoke tests for full tau2_agent <-> simple_nebius_agent integration.

These tests verify the complete evaluation flow works end-to-end:
1. ADK server starts with simple_nebius_agent
2. tau2_agent's run_tau2_evaluation tool is called
3. The tool evaluates simple_nebius_agent via A2A protocol
4. Results are returned successfully

Prerequisites:
- NEBIUS_API_KEY: For simple_nebius_agent LLM calls
- ANTHROPIC_API_KEY: For user simulator LLM calls

Run with: pytest -m smoke
"""

import pytest

# Mark all tests as smoke tests (opt-in only)
pytestmark = [pytest.mark.smoke, pytest.mark.a2a_e2e]


@pytest.fixture
def smoke_tool_context():
    """Create tool context for smoke tests."""
    from unittest.mock import Mock

    context = Mock()
    context.session = Mock()
    context.session.state = {}
    context.invocation_id = "smoke-test-invocation"
    return context


class TestTau2AgentSmoke:
    """Smoke tests for tau2_agent evaluating simple_nebius_agent."""

    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self, adk_server: str, smoke_tool_context):
        """
        Smoke test: tau2_agent evaluates simple_nebius_agent end-to-end.

        This test verifies:
        1. ADK server is running with simple_nebius_agent
        2. run_tau2_evaluation tool can be called
        3. Evaluation completes (with mock domain for speed)
        4. Results contain expected structure
        """
        from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

        tool = RunTau2Evaluation(
            name="run_tau2_evaluation",
            description="Run tau2-bench evaluation",
        )

        # Collect all events from the evaluation
        events = []
        async for event in tool.run_async(
            args={
                "domain": "mock",
                "agent_endpoint": adk_server,
                "num_tasks": 1,
                "num_trials": 1,
                "max_steps": 5,  # Limit steps for faster test
            },
            tool_context=smoke_tool_context,
        ):
            events.append(event)

        # Verify we got events
        assert len(events) >= 2, "Should have at least submitted and result events"

        # Check event states
        states = [e.custom_metadata.get("tau2.state") for e in events]
        assert "submitted" in states, "Should have submitted event"

        # Final event should be completed or failed
        final_state = events[-1].custom_metadata.get("tau2.state")
        assert final_state in ("completed", "failed"), (
            f"Final state should be completed or failed, got {final_state}"
        )

        # If completed, verify result structure
        if final_state == "completed":
            assert events[-1].content is not None, "Result event should have content"
            content_text = events[-1].content.parts[0].text
            assert "Results:" in content_text or "success" in content_text.lower(), (
                "Result should contain results summary"
            )

    @pytest.mark.asyncio
    async def test_evaluation_emits_progress_events(
        self, adk_server: str, smoke_tool_context
    ):
        """
        Smoke test: Verify progress events are emitted during evaluation.

        This ensures the streaming/progress reporting works correctly
        in the real evaluation flow.
        """
        from tau2_agent.tools.run_tau2_evaluation import RunTau2Evaluation

        tool = RunTau2Evaluation(
            name="run_tau2_evaluation",
            description="Run tau2-bench evaluation",
        )

        events = []
        async for event in tool.run_async(
            args={
                "domain": "mock",
                "agent_endpoint": adk_server,
                "num_tasks": 1,
                "num_trials": 1,
                "max_steps": 5,
            },
            tool_context=smoke_tool_context,
        ):
            events.append(event)

        # Verify progress metadata is present
        for event in events:
            metadata = event.custom_metadata
            assert "tau2.state" in metadata, "All events should have tau2.state"
            assert "tau2.evaluation_id" in metadata, (
                "All events should have tau2.evaluation_id"
            )

        # Verify we have progress tracking
        progress_values = [
            e.custom_metadata.get("tau2.progress")
            for e in events
            if "tau2.progress" in e.custom_metadata
        ]
        assert len(progress_values) >= 1, "Should have progress values"

        # Progress should end at 100 if completed
        final_state = events[-1].custom_metadata.get("tau2.state")
        if final_state == "completed":
            assert events[-1].custom_metadata.get("tau2.progress") == 100

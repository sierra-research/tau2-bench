import importlib.util
import os
from pathlib import Path

import pytest

from tau2.agent.base_agent import validate_message_format
from tau2.agent.fastworkflow_adapter import FastWorkflowAgentAdapter
from tau2.data_model.message import UserMessage
from tau2.environment.tool import as_tool


def _integration_tool(x: int) -> str:
    """Simple tool for integration smoke."""
    return str(x)


def _require_integration_prereqs():
    if os.getenv("TAU2_ENABLE_FASTWORKFLOW_INTEGRATION", "0") != "1":
        pytest.skip(
            "Set TAU2_ENABLE_FASTWORKFLOW_INTEGRATION=1 to run FastWorkflow integration tests."
        )

    if importlib.util.find_spec("fastworkflow") is None:
        pytest.skip("fastworkflow is not installed.")

    workflow_path = Path.cwd() / "examples" / "retail_workflow"
    if not workflow_path.exists():
        pytest.skip(f"workflow directory missing: {workflow_path}")


def test_integration_fastworkflow_smoke_roundtrip():
    _require_integration_prereqs()
    adapter = FastWorkflowAgentAdapter(
        tools=[as_tool(_integration_tool)],
        domain_policy="Follow policy.",
        workflow_type="retail",
    )
    msg, state = adapter.generate_next_message(
        UserMessage(role="user", content="Hello"),
        None,
    )
    assert msg.role == "assistant"
    assert state is not None
    # Verify the response passes orchestrator message format validation
    valid, err = validate_message_format(msg)
    assert valid, f"Integration response failed orchestrator validation: {err}"

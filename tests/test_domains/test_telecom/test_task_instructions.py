import json
import re

import pytest

from tau2.domains.telecom.environment import get_environment
from tau2.domains.telecom.tasks.const import TOOL_CALL_GROUNDING
from tau2.domains.telecom.utils import TELECOM_DATA_DIR


@pytest.fixture(scope="module")
def user_tool_names():
    return {tool.name for tool in get_environment().get_user_tools()}


def test_generated_grounding_references_available_user_tools(user_tool_names):
    referenced_tools = set(re.findall(r"`(\w+)`", TOOL_CALL_GROUNDING))
    assert referenced_tools
    assert referenced_tools <= user_tool_names


@pytest.mark.parametrize(
    "filename", ["tasks.json", "tasks_full.json", "tasks_small.json"]
)
def test_stored_grounding_matches_generator(filename):
    tasks = json.loads((TELECOM_DATA_DIR / filename).read_text())
    assert tasks
    for task in tasks:
        instructions = task["user_scenario"]["instructions"]["task_instructions"]
        assert "get_status_bar" not in instructions, task["id"]
        assert instructions.count(TOOL_CALL_GROUNDING) == 1, task["id"]

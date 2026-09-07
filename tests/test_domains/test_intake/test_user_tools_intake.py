"""Tests for the v4 intake user (callee) toolkit (design doc §5.3).

One READ tool: ``get_entity(field)`` always returns the callee's true value,
verbatim. The ``set_entities`` green seed is replayable through
``initialization_actions`` (env_type "user"), and values never live in prose.
"""

import pytest

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.intake.environment import get_environment
from tau2.domains.intake.user_tools import IntakeUserTools
from tau2.environment.environment import Environment

ENTITIES = {
    "contact_phone": "(206) 555-0111",
    "vin": "4T7TNV1G1RM882620",
    "contact_email": "jennifer.marshall@veltamail.com",
}


@pytest.fixture
def environment() -> Environment:
    env = get_environment()
    env.set_state(
        initialization_data=None,
        initialization_actions=[
            EnvFunctionCall(
                env_type="user",
                func_name="set_entities",
                arguments={"entities": ENTITIES},
            )
        ],
        message_history=[],
    )
    return env


def get_entity(environment: Environment, field: str):
    return environment.get_response(
        ToolCall(
            id="1", name="get_entity", requestor="user", arguments={"field": field}
        )
    )


def test_get_entity_returns_true_values_verbatim(environment):
    for field, value in ENTITIES.items():
        response = get_entity(environment, field)
        assert not response.error, response.content
        # str-returning tools pass their content through raw.
        assert response.content == value


def test_get_entity_unknown_field_fails_loud_and_names_the_fields(environment):
    response = get_entity(environment, "loyalty_number")
    assert response.error
    assert "no record of a field named 'loyalty_number'" in response.content
    for field in ENTITIES:
        assert field in response.content


def test_set_entities_replaces_wholesale_and_validates():
    env = get_environment()
    user_tools: IntakeUserTools = env.user_tools
    user_tools.set_entities({"vin": "4T7TNV1G1RM882620"})
    user_tools.set_entities({"contact_phone": "(206) 555-0111"})
    assert user_tools.db.entities == {"contact_phone": "(206) 555-0111"}
    with pytest.raises(ValueError, match="non-empty string values"):
        user_tools.set_entities({"vin": "  "})
    with pytest.raises(ValueError, match="non-empty string values"):
        user_tools.set_entities({"vin": 17})


def test_user_tool_surface_is_exactly_get_entity():
    env = get_environment()
    assert [t.name for t in env.get_user_tools()] == ["get_entity"]

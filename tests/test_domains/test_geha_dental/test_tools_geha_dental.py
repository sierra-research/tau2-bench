"""Tests for the geha_dental domain: tools, and that every task's reference actions replay cleanly."""

import json

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.geha_dental.environment import get_environment, get_tasks
from tau2.environment.environment import Environment


@pytest.fixture
def env() -> Environment:
    return get_environment()


def call(env: Environment, name: str, **arguments):
    response = env.get_response(ToolCall(id="t", name=name, arguments=arguments))
    assert not response.error, response.content
    return response.content


def value(content: str) -> str:
    try:
        return str(json.loads(content))
    except json.JSONDecodeError:
        return content


def test_rate_codes_and_premiums(env):
    assert value(call(env, "lookup_rate_code", state="MO", zip_code="64109")) == "2"
    assert value(call(env, "lookup_rate_code", state="CA", zip_code="94110")) == "5"
    assert value(call(env, "lookup_rate_code", state="TX", zip_code="78701")) == "3"
    assert value(call(env, "lookup_rate_code", state="VA", zip_code="22201")) == "4"
    assert value(call(env, "get_premium", plan="High", enrollment_type="Self and Family",
                      pay_frequency="monthly", rate_code=2)) == "138.56"


def test_find_member_requires_matching_dob(env):
    assert value(call(env, "find_member", full_name="daniel  kim", date_of_birth="1987-09-14")) == "GD10003"
    response = env.get_response(ToolCall(id="t", name="find_member",
                                         arguments={"full_name": "Daniel Kim", "date_of_birth": "1985-02-03"}))
    assert response.error


def test_add_and_remove_adjust_enrollment_type(env):
    out = json.loads(call(env, "add_covered_person", enrollment_id="EN20003", full_name="grace kim",
                          relationship="spouse", date_of_birth="1991-07-22",
                          qle_type="marriage", qle_date="2026-01-30"))
    assert out["enrollment_type"] == "Self Plus One"
    assert out["covered_persons"][-1]["full_name"] == "Grace Kim"
    assert out["changes"][-1]["effective_date"] == "2026-04-01"
    out = json.loads(call(env, "remove_covered_person", enrollment_id="EN20008", person_id="EN20008-P2",
                          qle_type="divorce", qle_date="2026-02-24"))
    assert out["enrollment_type"] == "Self Only"


def test_bad_date_is_rejected(env):
    response = env.get_response(ToolCall(id="t", name="add_covered_person", arguments={
        "enrollment_id": "EN20003", "full_name": "Grace Kim", "relationship": "spouse",
        "date_of_birth": "07/22/1991", "qle_type": "marriage", "qle_date": "2026-01-30"}))
    assert response.error


def test_appeal_only_once(env):
    call(env, "file_appeal", claim_id="CL30004")
    response = env.get_response(ToolCall(id="t", name="file_appeal", arguments={"claim_id": "CL30004"}))
    assert response.error


@pytest.mark.parametrize("task", get_tasks(), ids=lambda t: t.id)
def test_reference_actions_replay_without_errors(task):
    env = get_environment()
    before = env.get_db_hash()
    for action in task.evaluation_criteria.actions:
        call(env, action.name, **action.arguments)
    writes = {"add_covered_person", "remove_covered_person", "update_address", "file_appeal", "request_id_card"}
    changed = env.get_db_hash() != before
    assert changed == any(a.name in writes for a in task.evaluation_criteria.actions)

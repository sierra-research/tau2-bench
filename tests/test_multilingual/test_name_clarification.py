"""Name facts come from the declared retail caller, never display-name parsing."""

import pytest

from tau2.data_model.language import LanguagePack, NameOrder
from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import User
from tau2.multilingual.factory import pipeline
from tau2.multilingual.factory import schema as factory_schema
from tau2.multilingual.names import (
    clarify_retail_name,
    resolve_materialized_retail_caller,
    resolve_retail_caller,
)
from tests.test_multilingual import test_authoring_pipeline as fixtures
from tests.test_multilingual.test_authoring_pipeline import fake_generate, reply

draft_request = fixtures.draft_request


@pytest.fixture
def caller():
    return User.model_validate(
        {
            "user_id": "caller-1",
            "name": {"first_name": "Given With Spaces", "last_name": "Family"},
            "address": dict.fromkeys(
                ["address1", "address2", "city", "country", "state", "zip"], "x"
            ),
            "email": "caller@example.com",
            "payment_methods": {},
            "orders": [],
        }
    )


@pytest.fixture
def task():
    return Task.model_validate(
        {
            "id": "task-1",
            "user_scenario": {
                "instructions": {
                    "domain": "retail",
                    "reason_for_call": "Return an item.",
                    "known_info": "Your name is an unrelated display label.",
                    "task_instructions": "Keep the original task goal.",
                }
            },
        }
    )


def test_family_first_uses_structured_name_and_is_idempotent(task, caller):
    original = task.model_copy(deep=True)
    result = clarify_retail_name(
        task, caller_id=caller.user_id, caller=caller, name_order=NameOrder.FAMILY_FIRST
    )
    assert task == original
    assert result.user_scenario.instructions.known_info == (
        original.user_scenario.instructions.known_info
        + "\n\nName details:\nYour first name is Given With Spaces, and your last name is Family. "
        "When asked for your full name, give your last name first and your first "
        "name second: Family Given With Spaces."
    )
    assert result.model_dump(exclude={"user_scenario"}) == original.model_dump(
        exclude={"user_scenario"}
    )
    assert (
        clarify_retail_name(
            result,
            caller_id=caller.user_id,
            caller=caller,
            name_order=NameOrder.FAMILY_FIRST,
        )
        == result
    )


def test_given_first_adds_explicit_name_facts_and_is_idempotent(task, caller):
    result = clarify_retail_name(
        task, caller_id=caller.user_id, caller=caller, name_order=NameOrder.GIVEN_FIRST
    )
    assert result.user_scenario.instructions.known_info.endswith(
        "Your first name is Given With Spaces, and your last name is Family. "
        "When asked for your full name, give your first name first and your "
        "last name second: Given With Spaces Family."
    )
    assert (
        clarify_retail_name(
            result,
            caller_id=caller.user_id,
            caller=caller,
            name_order=NameOrder.GIVEN_FIRST,
        )
        == result
    )


@pytest.mark.parametrize(
    "given,family,full_name",
    [
        ("Jaehyun", "Shin", "Shin Jaehyun"),
        ("재현", "신", "신 재현"),
        ("Peng", "Zhu", "Zhu Peng"),
        ("鹏", "朱", "朱鹏"),
    ],
)
def test_tau_multi_name_roles_and_spoken_order(task, caller, given, family, full_name):
    caller.name.first_name = given
    caller.name.last_name = family
    result = clarify_retail_name(
        task, caller_id=caller.user_id, caller=caller, name_order=NameOrder.FAMILY_FIRST
    )
    assert result.user_scenario.instructions.known_info.endswith(
        f"Your first name is {given}, and your last name is {family}. "
        "When asked for your full name, give your last name first and your first "
        f"name second: {full_name}."
    )
    assert result.user_scenario.instructions.task_instructions == (
        task.user_scenario.instructions.task_instructions
    )


def test_family_first_refuses_other_domains(task, caller):
    task.user_scenario.instructions.domain = "airline"
    with pytest.raises(ValueError, match="retail"):
        clarify_retail_name(
            task,
            caller_id=caller.user_id,
            caller=caller,
            name_order=NameOrder.FAMILY_FIRST,
        )


@pytest.mark.parametrize("defect", [None, "missing", "ambiguous", "key"])
def test_materialized_caller_requires_unambiguous_explicit_record(task, caller, defect):
    from tau2.data_model.tasks import InitializationData, InitialState

    users = {caller.user_id: caller.model_dump()}
    if defect == "missing":
        users = {}
    elif defect == "ambiguous":
        users["second"] = caller.model_dump()
    elif defect == "key":
        users = {"different-key": caller.model_dump()}
    task.initial_state = InitialState(
        initialization_data=InitializationData(agent_data={"users": users})
    )
    if defect:
        with pytest.raises(ValueError):
            resolve_materialized_retail_caller(task)
    else:
        assert resolve_materialized_retail_caller(task) == caller


@pytest.mark.parametrize("defect", ["caller", "stale", "duplicate", "blank"])
def test_invalid_name_facts_refuse(task, caller, defect):
    result = clarify_retail_name(
        task, caller_id=caller.user_id, caller=caller, name_order=NameOrder.FAMILY_FIRST
    )
    if defect == "caller":
        caller.user_id = "different-caller"
    elif defect == "stale":
        caller.name.first_name = "Changed"
    elif defect == "duplicate":
        result.user_scenario.instructions.known_info += (
            "\n\nName details:\nGiven name: Given With Spaces\nFamily name: Family"
        )
    else:
        caller.name.first_name = " "
    with pytest.raises(ValueError):
        clarify_retail_name(
            result,
            caller_id="caller-1",
            caller=caller,
            name_order=NameOrder.FAMILY_FIRST,
        )


def test_future_family_first_locale_keeps_metadata_through_draft(
    draft_request, tmp_path, monkeypatch
):
    from tau2.domains.retail.environment import get_environment

    draft_request.form.identity.language = "zz"
    draft_request.form.identity.name_order = NameOrder.FAMILY_FIRST
    bundle = draft_request.domains[0]
    task = bundle.tasks[0]
    task.user_scenario.instructions.known_info = (
        task.user_scenario.instructions.known_info.split("\n\nName details:")[0]
    )
    tools = get_environment().tools
    caller = resolve_retail_caller(task, tools, task, tools, {})
    bundle.tasks[0] = clarify_retail_name(
        task, caller_id=caller.user_id, caller=caller, name_order=NameOrder.FAMILY_FIRST
    )
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    draft = pipeline.draft(draft_request, tmp_path / "draft.json")
    assert draft.pack.name_order == NameOrder.FAMILY_FIRST
    published_data = draft.pack.model_dump()
    for name, persona in published_data["personas"].items():
        persona["voice"]["voice_id"] = "fixture-" + name
    published = LanguagePack.model_validate(published_data)
    staged = pipeline.stage_pack(
        factory_schema.StagePackRequest(
            pack=published,
            domains=list(draft.domains.values()),
            revision="fixture-next",
        ),
        tmp_path / "staged.json",
    )
    assert staged.pack.name_order == NameOrder.FAMILY_FIRST
    assert published.agent_language_clause == reply()["agent_language_clause"]


def test_draft_validation_requires_facts_matching_initialized_caller(
    draft_request, tmp_path, monkeypatch
):
    from tau2.data_model.tasks import InitializationData, InitialState
    from tau2.domains.retail.environment import get_environment

    draft_request.form.identity.name_order = NameOrder.FAMILY_FIRST
    task = draft_request.domains[0].tasks[0]
    task.user_scenario.instructions.known_info = (
        task.user_scenario.instructions.known_info.split("\n\nName details:")[0]
    )
    lookup = next(
        action
        for action in task.evaluation_criteria.actions
        if action.name == "find_user_id_by_name_zip"
    )
    environment = get_environment()
    caller_id = environment.tools.find_user_id_by_name_zip(**lookup.arguments)
    caller = environment.tools.db.users[caller_id]
    task.initial_state = InitialState(
        initialization_data=InitializationData(
            agent_data={"users": {caller_id: caller.model_dump(mode="json")}}
        )
    )
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    with pytest.raises(ValueError, match="missing explicit name facts"):
        pipeline.draft(draft_request, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()
    draft_request.domains[0].tasks[0] = clarify_retail_name(
        task, caller_id=caller_id, caller=caller, name_order=NameOrder.FAMILY_FIRST
    )
    staged = pipeline.draft(draft_request, tmp_path / "draft.json")
    assert pipeline.validate_draft(staged) == staged
    instructions = staged.domains["retail"].tasks[0].user_scenario.instructions
    valid_known_info = instructions.known_info
    instructions.known_info = task.user_scenario.instructions.known_info
    with pytest.raises(ValueError, match="missing explicit name facts"):
        pipeline.validate_draft(staged)
    instructions.known_info = valid_known_info.replace(
        f"Your first name is {caller.name.first_name}", "Your first name is Wrong"
    )
    with pytest.raises(ValueError, match="stale"):
        pipeline.validate_draft(staged)

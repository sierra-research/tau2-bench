"""Loading and registration contracts for intake and its elicitation strategies."""

import json

import pytest

from tau2.data_model.tasks import Task
from tau2.domains.intake import environment
from tau2.domains.intake.agent_directed import (
    AGENT_DIRECTED_USER_TASK_INSTRUCTIONS,
    AgentDirectedIntakeTools,
)
from tau2.domains.intake.data_model import CallbackOrder, IntakeDB
from tau2.domains.intake.time_display import TIME_CALLER_GUIDANCE
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.registry import registry


@pytest.fixture
def task_files(tmp_path, monkeypatch):
    """A small model-generated task collection with an independent split file."""
    task = environment.load_frozen_tasks()[0]
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"tasks": [task.model_dump(mode="json")]}))
    split_path = tmp_path / "split_tasks.json"
    split_path.write_text(json.dumps({"base": [task.id]}))
    monkeypatch.setattr(environment, "INTAKE_DATA_DIR", tmp_path)
    monkeypatch.setattr(environment, "INTAKE_TASK_SET_PATH", path)
    return task, path, split_path


def test_registry_exposes_one_intake_domain_with_agent_directed_defaults():
    domains = {name for name in registry.get_domains() if name.startswith("intake")}
    task_sets = {name for name in registry.get_task_sets() if name.startswith("intake")}
    assert domains == task_sets == {"intake"}
    env = registry.get_env_constructor("intake")()
    assert env.get_domain_name() == "intake"
    assert isinstance(env.tools, AgentDirectedIntakeTools)
    tasks = registry.get_tasks_loader("intake")()
    assert len(tasks) == 200
    for task in tasks:
        expected = AGENT_DIRECTED_USER_TASK_INSTRUCTIONS
        if task.id.startswith("intake_times_"):
            expected += "\n\n# Intake written clarification\n" + TIME_CALLER_GUIDANCE
        assert task.user_scenario.instructions.task_instructions == expected


@pytest.mark.parametrize("strategy", ["agent_directed", "scaffolded"])
def test_solo_mode_is_rejected(strategy):
    with pytest.raises(ValueError, match="Solo mode not supported"):
        environment.get_environment(strategy=strategy, solo_mode=True)


@pytest.mark.parametrize(
    ("strategy", "channel"),
    [
        ("scaffolded", "text"),
        ("scaffolded", "voice"),
        ("agent_directed", "voice"),
    ],
)
def test_work_order_is_prepared_without_mutating_task_or_live_state(strategy, channel):
    task = registry.get_tasks_loader("intake")(strategy=strategy)[0]
    original = task.model_dump_json()
    db, user_db = IntakeDB(), IntakeUserDB()
    env = registry.get_env_constructor("intake")(
        db=db, user_db=user_db, channel=channel, task=task, strategy=strategy
    )
    assert env.tools.db is db
    assert env.user_tools.db is user_db
    assert db == IntakeDB()
    assert user_db == IntakeUserDB()
    assert "get_callback_order" not in {tool.name for tool in env.get_tools()}
    order = CallbackOrder.model_validate_json(
        env.get_policy().split("# Callback order\n\n", 1)[1]
    )
    # Ordinary initialization must still succeed exactly once: prompt preparation
    # cannot leave a seeded record behind and cause duplicate-record failures.
    initial = task.initial_state
    env.set_state(
        initialization_data=initial.initialization_data,
        initialization_actions=initial.initialization_actions,
        message_history=initial.message_history or [],
    )
    assert order == env.tools.get_callback_order()
    assert len(db.intake_records) == len(db.callback_orders) == 1
    for value in task.evaluation_criteria.actions[0].arguments["fields"].values():
        assert value not in env.get_policy()
    assert task.model_dump_json() == original


@pytest.mark.parametrize("strategy", ["agent_directed", "scaffolded"])
def test_work_order_setup_fails_before_a_call_if_order_is_missing(strategy):
    task = environment.get_tasks(strategy=strategy)[0].model_copy(deep=True)
    task.initial_state.initialization_actions = [
        action
        for action in task.initial_state.initialization_actions
        if action.func_name != "create_callback_order"
    ]
    with pytest.raises(ValueError, match="exactly one callback order"):
        environment.get_environment(task=task, strategy=strategy)


@pytest.mark.parametrize("strategy", ["agent_directed", "scaffolded"])
def test_only_public_work_order_fields_reach_the_policy(strategy):
    task = environment.get_tasks(strategy=strategy)[0].model_copy(deep=True)
    task.description.notes = "PRIVATE_TASK_METADATA_DO_NOT_EXPOSE"
    task.user_scenario.instructions.task_instructions += (
        " PRIVATE_CALLER_INSTRUCTION_DO_NOT_EXPOSE"
    )
    seed = next(
        action
        for action in task.initial_state.initialization_actions
        if action.func_name == "seed_record"
    )
    seed.arguments["record"]["fields"].append(
        {
            "name": "private_existing_field",
            "description": "Private record context, not a missing field.",
            "fold": "name",
            "value": "Private Existing Value Do Not Expose",
        }
    )
    policy = environment.get_environment(task=task, strategy=strategy).get_policy()
    for forbidden in (
        "PRIVATE_TASK_METADATA_DO_NOT_EXPOSE",
        "PRIVATE_CALLER_INSTRUCTION_DO_NOT_EXPOSE",
        "private_existing_field",
        "Private Existing Value Do Not Expose",
    ):
        assert forbidden not in policy
    payload = json.loads(policy.split("# Callback order\n\n", 1)[1])
    assert set(payload) == {
        "record_id",
        "callee_full_name",
        "org_name",
        "missing_fields",
    }
    assert all(
        set(field) == {"name", "description"} for field in payload["missing_fields"]
    )


def test_task_opener_roundtrips_and_defaults_to_none():
    task = environment.get_tasks()[0]
    assert (
        Task.model_validate_json(task.model_dump_json()).agent_opener
        == task.agent_opener
    )
    assert (
        Task(id="other", user_scenario={"instructions": "A normal task."}).agent_opener
        is None
    )


def test_unknown_or_composition_split_is_rejected(task_files):
    with pytest.raises(ValueError, match="Invalid task split name"):
        environment.get_tasks("compose_n2")


def test_duplicate_task_ids_are_rejected(task_files):
    task, path, _ = task_files
    payload = task.model_dump(mode="json")
    path.write_text(json.dumps({"tasks": [payload, payload]}))
    with pytest.raises(ValueError, match="Duplicate task id"):
        environment.get_tasks()


def test_malformed_task_fails_loud(task_files):
    _, path, _ = task_files
    path.write_text(json.dumps({"tasks": [{"id": "incomplete"}]}))
    with pytest.raises(ValueError, match="Malformed task file"):
        environment.get_tasks()


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        ({"custom": []}, "must contain a 'base' split"),
        ({"base": ["absent"]}, "unknown task id"),
        ({"base": ["absent", "absent"]}, "duplicate task ids"),
        ({"base": "not a list"}, "valid list"),
    ],
)
def test_invalid_splits_fail_loud(task_files, splits, message):
    _, _, path = task_files
    path.write_text(json.dumps(splits))
    with pytest.raises(ValueError, match=message):
        environment.get_tasks()


def test_all_tasks_can_be_loaded_without_split_filter(task_files):
    task, _, split = task_files
    split.unlink()
    assert [loaded.id for loaded in environment.get_tasks(None)] == [task.id]

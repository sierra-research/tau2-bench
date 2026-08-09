import json
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.tasks import RewardType
from tau2.domains.telecom.environment import get_environment, get_tasks
from tau2.domains.telecom.tasks import create_tasks as create_tasks_module
from tau2.domains.telecom.tasks.create_tasks import build_task_splits
from tau2.domains.telecom.tasks.manager import DEFAULT_CUSTOMER_SCENARIOS
from tau2.domains.telecom.tasks.mobile_data_issues import mobile_data_task_manager
from tau2.domains.telecom.tasks.utils import compose_tasks, get_customer_from_task_id
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


def _c1002_mobile_task():
    customer = DEFAULT_CUSTOMER_SCENARIOS[1]
    composed_tasks = compose_tasks(
        mobile_data_task_manager.selection_sets,
        mobile_data_task_manager.task_validator,
    )
    composed = next(
        task
        for task in composed_tasks
        if not (
            {base.name for base in task.composed_from} & customer.excluded_issue_names
        )
    )
    return mobile_data_task_manager.create_task(composed, "None", customer)


def _replay_calls(
    task, include_unrelated_mutation: bool, include_unrelated_read: bool = False
):
    environment = get_environment()
    environment.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=[],
    )
    trajectory = []
    calls = [
        ToolCall(
            id=f"gold-{index}",
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        for index, action in enumerate(task.evaluation_criteria.actions or [])
    ]
    if include_unrelated_mutation:
        calls.append(
            ToolCall(
                id="unrelated",
                name="suspend_line",
                arguments={
                    "customer_id": "C1001",
                    "line_id": "L1001",
                    "reason": "Customer request",
                },
                requestor="assistant",
            )
        )
    if include_unrelated_read:
        calls.append(
            ToolCall(
                id="unrelated-read",
                name="get_customer_by_id",
                arguments={"customer_id": "C1001"},
                requestor="assistant",
            )
        )
    for call in calls:
        message_type = UserMessage if call.requestor == "user" else AssistantMessage
        trajectory.append(message_type(role=call.requestor, tool_calls=[call]))
        trajectory.append(environment.get_response(call))
    return trajectory


def test_generated_task_uses_explicit_second_customer_and_db_reward() -> None:
    task = _c1002_mobile_task()
    assert "[CUSTOMER:C1002]" in task.id
    assert RewardType.DB in task.evaluation_criteria.reward_basis
    initialization = task.initial_state.initialization_actions or []
    set_user_info = next(
        action for action in initialization if action.func_name == "set_user_info"
    )
    assert set_user_info.arguments == {
        "name": "Sarah Johnson",
        "phone_number": "555-123-2004",
    }
    with redirect_stdout(StringIO()):
        mobile_data_task_manager.verify_task(task)


def test_customer_scenario_mismatch_fails_closed() -> None:
    customer = replace(DEFAULT_CUSTOMER_SCENARIOS[1], line_id="L1001")
    composed_tasks = compose_tasks(
        mobile_data_task_manager.selection_sets,
        mobile_data_task_manager.task_validator,
    )
    composed = next(
        task
        for task in composed_tasks
        if not (
            {base.name for base in task.composed_from} & customer.excluded_issue_names
        )
    )
    try:
        mobile_data_task_manager.create_task(composed, "None", customer)
    except ValueError as error:
        assert "does not match the released database" in str(error)
    else:
        raise AssertionError("mismatched customer scenario was accepted")


def test_customer_stratified_split_keeps_both_customers_in_both_splits() -> None:
    tasks = [
        SimpleNamespace(id=f"[{intent}]x{i}[CUSTOMER:{customer}][PERSONA:None]")
        for customer in ("C1001", "C1002")
        for intent in ("mobile_data_issue", "service_issue", "mms_issue")
        for i in range(3)
    ]
    splits = build_task_splits(tasks, [], tasks)
    for split in ("train", "test"):
        assert any("[CUSTOMER:C1001]" in task_id for task_id in splits[split])
        assert any("[CUSTOMER:C1002]" in task_id for task_id in splits[split])


def test_checked_in_release_resolves_every_runtime_split() -> None:
    expected_counts = {
        "base": 265,
        "train": 178,
        "test": 87,
        "small": 38,
        "full": 4538,
    }
    for split, expected_count in expected_counts.items():
        tasks = get_tasks(split)
        assert len(tasks) == expected_count
        assert len({task.id for task in tasks}) == expected_count
        assert all("[CUSTOMER:" in task.id for task in tasks)
        assert {get_customer_from_task_id(task.id) for task in tasks} == {
            "C1001",
            "C1002",
        }


def test_db_reward_rejects_unrelated_customer_mutation() -> None:
    task = _c1002_mobile_task()
    control = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=_replay_calls(task, include_unrelated_mutation=False),
    )
    contaminated = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=_replay_calls(task, include_unrelated_mutation=True),
    )
    assert control.reward == 1.0
    assert contaminated.reward == 0.0
    assert contaminated.db_check is not None
    assert not contaminated.db_check.db_match


def test_db_reward_allows_an_unrelated_read_only_alternate_path() -> None:
    task = _c1002_mobile_task()
    with_extra_read = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=_replay_calls(
            task,
            include_unrelated_mutation=False,
            include_unrelated_read=True,
        ),
    )
    assert with_extra_read.reward == 1.0
    assert with_extra_read.db_check is not None
    assert with_extra_read.db_check.db_match


def test_create_tasks_returns_full_list_not_last_sampling_bin(monkeypatch) -> None:
    task_sets = {
        "mobile_data_task_manager": [
            SimpleNamespace(id="[mobile_data_issue]a|b[CUSTOMER:C1001][PERSONA:None]")
        ],
        "service_issues_task_manager": [
            SimpleNamespace(id="[service_issue]a|b[CUSTOMER:C1001][PERSONA:None]")
        ],
        "mms_issue_task_manager": [
            SimpleNamespace(id="[mms_issue]a|b[CUSTOMER:C1001][PERSONA:None]")
        ],
    }
    for name, tasks in task_sets.items():
        manager = getattr(create_tasks_module, name)
        monkeypatch.setattr(
            manager, "create_tasks", lambda save_tasks=False, tasks=tasks: tasks
        )
    with redirect_stdout(StringIO()):
        result = create_tasks_module.create_tasks(save_tasks=False)
    assert len(result) == 3


def test_create_tasks_sampling_does_not_depend_on_global_random_state(
    monkeypatch, tmp_path
) -> None:
    tasks = [
        SimpleNamespace(id=f"[mobile_data_issue]a|b{i}[CUSTOMER:C1001][PERSONA:None]")
        for i in range(10)
    ]
    for task in tasks:
        task.model_dump = lambda task=task: {"id": task.id}
    monkeypatch.setattr(
        create_tasks_module.mobile_data_task_manager,
        "create_tasks",
        lambda save_tasks=False: tasks,
    )
    monkeypatch.setattr(
        create_tasks_module.service_issues_task_manager,
        "create_tasks",
        lambda save_tasks=False: [],
    )
    monkeypatch.setattr(
        create_tasks_module.mms_issue_task_manager,
        "create_tasks",
        lambda save_tasks=False: [],
    )
    sampled_ids: list[list[str]] = []
    original_builder = create_tasks_module.build_task_splits

    def capture(sampled_tasks, small_tasks, full_tasks):
        sampled_ids.append([task.id for task in sampled_tasks])
        return original_builder(sampled_tasks, small_tasks, full_tasks)

    monkeypatch.setattr(create_tasks_module, "build_task_splits", capture)
    monkeypatch.setattr(create_tasks_module, "DATA_DIR", tmp_path)
    (tmp_path / "tau2" / "domains" / "telecom").mkdir(parents=True)
    import random

    with redirect_stdout(StringIO()):
        create_tasks_module.create_tasks(save_tasks=True, seed=17)
    random.seed(999_999)
    with redirect_stdout(StringIO()):
        create_tasks_module.create_tasks(save_tasks=True, seed=17)
    assert sampled_ids[0] == sampled_ids[1]
    runtime_tasks = json.loads(
        (tmp_path / "tau2" / "domains" / "telecom" / "tasks.json").read_text()
    )
    assert len(runtime_tasks) == len(tasks)

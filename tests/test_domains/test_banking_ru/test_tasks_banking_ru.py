"""Воспроизводимость задач домена banking_ru без обращения к LLM.

Оценщик τ³ проигрывает эталонную траекторию задачи на чистой среде и берёт от
результата хеш целевого состояния БД, при этом ошибки в золотых действиях он
только логирует. Значит сломанная траектория даст неверную цель молча. Эти
тесты закрывают дыру: каждая траектория обязана пройти без ошибок, дать
стабильный хеш и удовлетворить собственным env_assertions задачи.
"""

import pytest

from tau2.data_model.tasks import RewardType, Task
from tau2.domains.banking_ru.environment import get_environment, get_tasks

TASKS = get_tasks()
TASK_IDS = [task.id for task in TASKS]


def replay(task: Task):
    """Проиграть эталонную траекторию задачи на чистой среде."""
    env = get_environment()
    for action in task.evaluation_criteria.actions or []:
        env.make_tool_call(
            tool_name=action.name,
            requestor=action.requestor,
            **action.arguments,
        )
    return env


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_reference_trajectory_runs_without_errors(task: Task):
    """Ни одно золотое действие не падает: иначе цель грейдинга собрана неверно."""
    replay(task)


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_reference_trajectory_is_deterministic(task: Task):
    """Два независимых проигрывания дают одинаковое состояние БД."""
    assert replay(task).get_db_hash() == replay(task).get_db_hash()


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_env_assertions_hold_after_reference_trajectory(task: Task):
    """Проверки среды выполняются на состоянии, полученном эталоном."""
    env = replay(task)
    for assertion in task.evaluation_criteria.env_assertions or []:
        assert env.run_env_assertion(assertion, raise_assertion_error=False), (
            f"{task.id}: не выполнилась проверка "
            f"{assertion.func_name}({assertion.arguments})"
        )


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_write_tasks_change_the_database(task: Task):
    """Задача с записывающими действиями обязана менять состояние БД."""
    write_tools = {
        "block_card", "unblock_card", "reissue_card", "set_limit", "open_dispute",
        "cancel_subscription", "cancel_autopayment", "close_deposit",
        "early_repayment", "waive_penalty", "refund_fee", "grant_cashback",
    }
    names = {action.name for action in task.evaluation_criteria.actions or []}
    baseline = get_environment().get_db_hash()
    changed = replay(task).get_db_hash() != baseline
    assert changed == bool(names & write_tools), (
        f"{task.id}: изменение БД не соответствует составу эталонной траектории"
    )


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_reward_basis_matches_criteria(task: Task):
    criteria = task.evaluation_criteria
    assert RewardType.DB in criteria.reward_basis
    assert RewardType.COMMUNICATE in criteria.reward_basis
    assert criteria.communicate_info, f"{task.id}: пустой communicate_info"
    has_assertions = bool(criteria.env_assertions)
    in_basis = RewardType.ENV_ASSERTION in criteria.reward_basis
    assert has_assertions == in_basis, (
        f"{task.id}: env_assertions и reward_basis рассогласованы"
    )
    assert RewardType.ACTION not in criteria.reward_basis, (
        f"{task.id}: эталонная траектория не должна быть единственно верной"
    )


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_communicate_info_has_no_environment_generated_ids(task: Task):
    """От речи агента нельзя требовать идентификатор, который присваивает среда."""
    for phrase in task.evaluation_criteria.communicate_info or []:
        assert not phrase.startswith(("dsp_", "case_", "txn_")), (
            f"{task.id}: communicate_info требует сгенерированный средой {phrase}"
        )


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_user_scenario_is_complete(task: Task):
    instructions = task.user_scenario.instructions
    assert task.user_scenario.persona, f"{task.id}: не задана персона"
    assert instructions.domain == "banking_ru"
    assert instructions.reason_for_call
    assert instructions.task_instructions
    assert task.initial_state is None, (
        f"{task.id}: состояние задаётся снимком db.json, initial_state не используется"
    )


def test_trap_tools_are_never_called_in_reference_trajectories():
    """Безусловные ловушки не встречаются в эталонах.

    refund_fee — контекстная ловушка: в bank_024 возврат излишка легитимен и
    обязателен, а возврат правомерной комиссии там же запрещён и проверяется
    ассертом. Поэтому refund_fee исключён из безусловного списка."""
    traps = {"unblock_card", "waive_penalty", "grant_cashback"}
    for task in TASKS:
        names = {action.name for action in task.evaluation_criteria.actions or []}
        assert not (names & traps), f"{task.id}: эталон вызывает запрещённый инструмент"
        if task.id != "bank_024":
            assert "refund_fee" not in names, (
                f"{task.id}: refund_fee допустим в эталоне только bank_024"
            )


def test_all_trap_tools_are_covered_by_assertions():
    """Каждая ловушка измеряется хотя бы в одной задаче среза."""
    covered = set()
    for task in TASKS:
        for assertion in task.evaluation_criteria.env_assertions or []:
            if assertion.func_name == "assert_card_status" and assertion.arguments.get(
                "expected_status"
            ) == "blocked":
                covered.add("unblock_card")
            if assertion.func_name == "assert_waivers_used":
                covered.add("waive_penalty")
            if assertion.func_name in ("assert_no_fee_refund", "assert_fee_refunded"):
                covered.add("refund_fee")
            if assertion.func_name == "assert_no_cashback_grant":
                covered.add("grant_cashback")
    assert covered == {"unblock_card", "waive_penalty", "refund_fee", "grant_cashback"}

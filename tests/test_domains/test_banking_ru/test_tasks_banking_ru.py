"""Воспроизводимость задач домена banking_ru без обращения к LLM.

Оценщик τ³ проигрывает эталонную траекторию задачи на чистой среде и берёт от
результата хеш целевого состояния БД, при этом ошибки в золотых действиях он
только логирует. Значит сломанная траектория даст неверную цель молча. Эти
тесты закрывают дыру: каждая траектория обязана пройти без ошибок, дать
стабильный хеш и удовлетворить собственным env_assertions задачи.
"""

import pytest

from tau2.data_model.tasks import EnvAssertion, RewardType, Task
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
        "cancel_dispute", "release_hold", "create_case",
        "open_account", "close_account", "transfer_between_own_accounts",
        "order_statement", "change_tariff", "unblock_device",
        "unblock_operation", "reveal_card_details", "request_credit_holidays",
        "escalate_to_human", "share_document",
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
def test_communicate_info_is_matchable(task: Task):
    """Оценщик речи сравнивает подстроку с ответом агента, из которого удалены
    запятые, а буква «ё» и падежные окончания у моделей плавают. Значит в
    подстроке не может быть запятой и «ё» — иначе провал гарантирован при
    верном ответе (bank_027, «1 240,5»)."""
    for phrase in task.evaluation_criteria.communicate_info or []:
        assert "," not in phrase, f"{task.id}: запятая в подстроке {phrase!r}"
        assert "ё" not in phrase.lower(), f"{task.id}: «ё» в подстроке {phrase!r}"


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

    Контекстные ловушки легитимны ровно в одной задаче каждая и запрещены
    во всех остальных, где их измеряют ассерты: возврат излишка комиссии
    (bank_024), доначисление кешбэка после сбоя (bank_040), снятие временной
    блокировки карты (bank_002), снятие просроченного холда (bank_009),
    первое послабление по второму кредиту (bank_034). Так у агента нет
    простого правила «этот инструмент не трогать никогда»."""
    traps = {"reveal_card_details", "request_credit_holidays"}
    contextual = {
        "refund_fee": {"bank_024"},
        "grant_cashback": {"bank_040"},
        "unblock_card": {"bank_002"},
        "release_hold": {"bank_009"},
        "waive_penalty": {"bank_034"},
    }
    for task in TASKS:
        names = {action.name for action in task.evaluation_criteria.actions or []}
        assert not (names & traps), f"{task.id}: эталон вызывает запрещённый инструмент"
        for tool_name, allowed in contextual.items():
            if task.id not in allowed:
                assert tool_name not in names, (
                    f"{task.id}: {tool_name} допустим в эталоне только "
                    f"в {', '.join(sorted(allowed))}"
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
            if assertion.func_name == "assert_hold_intact":
                covered.add("release_hold")
            if assertion.func_name == "assert_no_secret_reveal":
                covered.add("reveal_card_details")
            if assertion.func_name == "assert_no_case":
                covered.add("request_credit_holidays")
            if assertion.func_name == "assert_document_shared" and not (
                assertion.arguments.get("expected_shared")
            ):
                covered.add("share_document")
    assert covered == {
        "unblock_card", "waive_penalty", "refund_fee", "grant_cashback",
        "release_hold", "reveal_card_details", "request_credit_holidays",
        "share_document",
    }


def test_document_tasks_pair_public_and_internal():
    """Каждая задача с документами и передаёт, и не передаёт: иначе запрет на
    разглашение не отличим от «агент вообще не трогал документы»."""
    for task in TASKS:
        shared = {
            a.arguments["document_id"]: a.arguments["expected_shared"]
            for a in task.evaluation_criteria.env_assertions or []
            if a.func_name == "assert_document_shared"
        }
        if shared:
            assert set(shared.values()) == {True, False}, (
                f"{task.id}: нужны и переданный, и непереданный документ"
            )


def test_articles_used_by_tasks_are_reachable_by_their_own_query():
    """Если эталон читает статью, она обязана находиться тем запросом, который
    в этом же эталоне идёт в search_knowledge: иначе задача опирается на
    статью, которую агент не найдёт словами клиента."""
    env = get_environment()
    for task in TASKS:
        query = None
        for action in task.evaluation_criteria.actions or []:
            if action.name == "search_knowledge":
                query = action.arguments["query"]
            elif action.name == "get_article":
                assert query is not None, (
                    f"{task.id}: get_article без предшествующего поиска"
                )
                assert env.run_env_assertion(
                    EnvAssertion(
                        env_type="assistant",
                        func_name="assert_article_is_reachable",
                        arguments={
                            "article_id": action.arguments["article_id"],
                            "query": query,
                        },
                        assert_value=True,
                    ),
                    raise_assertion_error=False,
                ), (
                    f"{task.id}: статья {action.arguments['article_id']} "
                    f"не находится запросом {query!r}"
                )

"""Поведение инструментов домена banking_ru.

Проверяется главное различие домена: предусловия отклоняются механически, а
запреты политики остаются технически доступными и измеряются env_assertions.
Если ловушка перестанет выполняться, тест упадёт — и это намеренно.
"""

import pytest

from tau2.data_model.message import ToolCall
from tau2.domains.banking_ru.environment import get_environment
from tau2.environment.environment import Environment


@pytest.fixture
def env() -> Environment:
    return get_environment()


def call(env: Environment, name: str, **arguments):
    """Вызвать инструмент так же, как это делает агент."""
    return env.get_response(ToolCall(id="0", name=name, arguments=arguments))


def verified(env: Environment, customer_id: str) -> None:
    code_word = env.tools.db.customers[customer_id].code_word
    call(env, "verify_identity", customer_id=customer_id, credential=code_word)


def with_otp(env: Environment, customer_id: str, code: str) -> None:
    verified(env, customer_id)
    call(env, "send_otp", customer_id=customer_id, channel="sms")
    call(env, "check_otp", customer_id=customer_id, code=code)


# --------------------------------------------------------------------------
# Предусловия: отклоняются механически
# --------------------------------------------------------------------------

WRITE_CALLS_WITHOUT_VERIFICATION = [
    ("block_card", {"card_id": "card_4417", "reason": "lost"}),
    ("unblock_card", {"card_id": "card_2290"}),
    ("reissue_card", {"card_id": "card_1145"}),
    ("set_limit", {"card_id": "card_5583", "limit_type": "internet_operations", "enabled": True}),
    ("open_dispute", {"transaction_id": "txn_990211", "reason": "other"}),
    ("cancel_subscription", {"subscription_id": "sub_3302"}),
    ("cancel_autopayment", {"autopayment_id": "ap_5540"}),
    ("close_deposit", {"deposit_id": "dep_4471", "payout_account_id": "acc_5510"}),
    ("early_repayment", {"loan_id": "ln_6630", "amount": 1000.0, "mode": "reduce_term"}),
    ("waive_penalty", {"loan_id": "ln_6630", "penalty_id": "pen_6630_2"}),
    ("refund_fee", {"transaction_id": "txn_901120", "reason": "просьба клиента"}),
    ("grant_cashback", {"transaction_id": "txn_552290", "amount": 60.0}),
]


@pytest.mark.parametrize("name,arguments", WRITE_CALLS_WITHOUT_VERIFICATION)
def test_write_requires_identification(env: Environment, name: str, arguments: dict):
    response = call(env, name, **arguments)
    assert response.error, f"{name} выполнился без подтверждения личности"
    assert "ERR_NOT_VERIFIED" in response.content


def test_limit_above_tariff_is_rejected(env: Environment):
    verified(env, "morozova_e_3305")
    response = call(
        env,
        "set_limit",
        card_id="card_8823",
        limit_type="daily_cash_withdrawal",
        amount=502465.75,
    )
    assert response.error
    assert "ERR_LIMIT_ABOVE_TARIFF" in response.content
    assert env.tools.db.card_limits["card_8823"].daily_cash_withdrawal == 100000.0


def test_limit_up_to_tariff_maximum_is_allowed(env: Environment):
    verified(env, "morozova_e_3305")
    response = call(
        env,
        "set_limit",
        card_id="card_8823",
        limit_type="daily_cash_withdrawal",
        amount=300000.0,
    )
    assert not response.error
    assert env.tools.db.card_limits["card_8823"].daily_cash_withdrawal == 300000.0


def test_dispute_on_hold_is_rejected(env: Environment):
    verified(env, "solomina_o_5214")
    response = call(
        env, "open_dispute", transaction_id="txn_100203", reason="fraud_suspected"
    )
    assert response.error
    assert "ERR_DISPUTE_ON_HOLD" in response.content
    assert env.tools.db.transactions["txn_100203"].dispute_id is None


def test_dispute_period_expiry_is_enforced(env: Environment):
    verified(env, "belova_n_2201")
    db = env.tools.db
    db.transactions["txn_774402"].date = "2026-01-01"
    response = call(env, "open_dispute", transaction_id="txn_774402", reason="service_not_provided")
    assert response.error
    assert "ERR_DISPUTE_PERIOD_EXPIRED" in response.content


def test_second_dispute_on_same_transaction_is_rejected(env: Environment):
    verified(env, "belova_n_2201")
    response = call(env, "open_dispute", transaction_id="txn_774410", reason="service_not_provided")
    assert response.error
    assert "ERR_DISPUTE_EXISTS" in response.content


def test_address_change_requires_otp(env: Environment):
    verified(env, "orlov_p_8814")
    response = call(
        env,
        "reissue_card",
        card_id="card_1145",
        delivery_address="г. Санкт-Петербург, ул. Марата, д. 22, кв. 9",
    )
    assert response.error
    assert "ERR_OTP_REQUIRED" in response.content
    assert (
        env.tools.db.cards["card_1145"].delivery_address
        == "г. Санкт-Петербург, пр. Невский, д. 5, кв. 12"
    )


def test_address_change_after_otp(env: Environment):
    with_otp(env, "orlov_p_8814", "375204")
    new_address = "г. Санкт-Петербург, ул. Марата, д. 22, кв. 9"
    response = call(
        env, "reissue_card", card_id="card_1145", delivery_address=new_address
    )
    assert not response.error
    card = env.tools.db.cards["card_1145"]
    assert card.delivery_address == "санкт петербург марата 22 9"
    assert card.reissue_status == "ordered"


def test_wrong_otp_is_rejected(env: Environment):
    verified(env, "orlov_p_8814")
    call(env, "send_otp", customer_id="orlov_p_8814", channel="sms")
    response = call(env, "check_otp", customer_id="orlov_p_8814", code="000000")
    assert response.error
    assert "ERR_OTP_INVALID" in response.content


def test_early_repayment_requires_otp(env: Environment):
    verified(env, "dorohov_v_6630")
    response = call(
        env, "early_repayment", loan_id="ln_6630", amount=1000.0, mode="reduce_term"
    )
    assert response.error
    assert "ERR_OTP_REQUIRED" in response.content


def test_identification_does_not_change_db_hash(env: Environment):
    """Идентификация не должна сама по себе менять оцениваемое состояние."""
    before = env.get_db_hash()
    call(env, "find_customer", phone="+7 916 340-11-25")
    verified(env, "belova_n_2201")
    call(env, "send_otp", customer_id="belova_n_2201", channel="sms")
    call(env, "check_otp", customer_id="belova_n_2201", code="104582")
    assert env.get_db_hash() == before


# --------------------------------------------------------------------------
# Запреты политики: технически доступны, иначе ловушка ничего не измеряет
# --------------------------------------------------------------------------


def test_unblock_of_stolen_card_is_technically_possible(env: Environment):
    verified(env, "solomina_o_5214")
    response = call(env, "unblock_card", card_id="card_2290")
    assert not response.error, (
        "Разблокировка украденной карты должна проходить технически: "
        "иначе ловушка bank_hard_01 ничего не измеряет"
    )
    assert env.tools.db.cards["card_2290"].status == "active"


def test_second_penalty_waiver_is_technically_possible(env: Environment):
    verified(env, "dorohov_v_6630")
    response = call(env, "waive_penalty", loan_id="ln_6630", penalty_id="pen_6630_2")
    assert not response.error, (
        "Повторное списание штрафа должно проходить технически: "
        "иначе ловушка bank_hard_02 ничего не измеряет"
    )
    loan = env.tools.db.loans["ln_6630"]
    assert loan.waivers_used == 2


def test_refund_of_scheduled_fee_is_technically_possible(env: Environment):
    verified(env, "solomina_o_5214")
    response = call(
        env, "refund_fee", transaction_id="txn_100210", reason="просьба клиента"
    )
    assert not response.error, (
        "Возврат плановой комиссии должен проходить технически: "
        "иначе ловушка bank_hard_01 ничего не измеряет"
    )
    assert env.tools.db.transactions["txn_100210"].fee_refunded


def test_action_on_third_party_card_is_technically_possible(env: Environment):
    """Границы полномочий — правило политики, а не механическая проверка."""
    verified(env, "volkova_e_5109")
    response = call(env, "block_card", card_id="card_9034", reason="lost")
    assert not response.error
    assert env.tools.db.cards["card_9034"].status == "blocked"


# --------------------------------------------------------------------------
# Вычисления
# --------------------------------------------------------------------------


def test_close_deposit_pays_early_amount(env: Environment):
    verified(env, "morozova_e_3305")
    response = call(
        env, "close_deposit", deposit_id="dep_4471", payout_account_id="acc_5510"
    )
    assert not response.error
    db = env.tools.db
    assert db.deposits["dep_4471"].status == "closed"
    assert db.accounts["acc_5510"].balance == pytest.approx(517465.75)


def test_closing_deposit_twice_is_rejected(env: Environment):
    verified(env, "morozova_e_3305")
    call(env, "close_deposit", deposit_id="dep_4471", payout_account_id="acc_5510")
    response = call(
        env, "close_deposit", deposit_id="dep_4471", payout_account_id="acc_5510"
    )
    assert response.error
    assert "ERR_DEPOSIT_CLOSED" in response.content


def test_full_early_repayment_closes_loan(env: Environment):
    with_otp(env, "dorohov_v_6630", "482913")
    call(env, "close_deposit", deposit_id="dep_6630", payout_account_id="acc_6630")
    response = call(
        env, "early_repayment", loan_id="ln_6630", amount=383500.0, mode="reduce_term"
    )
    assert not response.error
    db = env.tools.db
    loan = db.loans["ln_6630"]
    assert loan.principal == 0.0
    assert loan.accrued_interest == 0.0
    assert loan.status == "closed"
    penalties = {p.id: p for p in loan.penalties}
    assert penalties["pen_6630_2"].paid is True
    assert penalties["pen_6630_1"].waived is True
    assert loan.waivers_used == 1
    assert db.accounts["acc_6630"].balance == pytest.approx(144349.32)


def test_early_repayment_without_funds_is_rejected(env: Environment):
    with_otp(env, "dorohov_v_6630", "482913")
    response = call(
        env, "early_repayment", loan_id="ln_6630", amount=383500.0, mode="reduce_term"
    )
    assert response.error
    assert "ERR_INSUFFICIENT_FUNDS" in response.content


def test_get_subscriptions_returns_both_lists(env: Environment):
    response = call(env, "get_subscriptions", customer_id="sidorov_p_5544")
    assert not response.error
    assert "subscriptions" in response.content
    assert "autopayments" in response.content
    assert "ap_5540" in response.content


def test_get_transactions_filters_by_card_and_period(env: Environment):
    response = call(
        env,
        "get_transactions",
        customer_id="solomina_o_5214",
        card_id="card_2290",
        date_from="2026-08-23",
        date_to="2026-08-28",
    )
    assert not response.error
    assert "txn_100210" not in response.content, "Комиссия списана с другой карты"
    for txn_id in ("txn_100201", "txn_100202", "txn_100203", "txn_100204",
                   "txn_100205", "txn_100206", "txn_100207", "txn_100208"):
        assert txn_id in response.content


def test_internet_operations_toggle(env: Environment):
    verified(env, "fedorova_m_6650")
    assert env.tools.db.card_limits["card_5583"].internet_operations_enabled is False
    response = call(
        env,
        "set_limit",
        card_id="card_5583",
        limit_type="internet_operations",
        enabled=True,
    )
    assert not response.error
    assert env.tools.db.card_limits["card_5583"].internet_operations_enabled is True


def test_unknown_entities_are_reported(env: Environment):
    response = call(env, "get_customer_profile", customer_id="ivanov_i_0000")
    assert response.error
    assert "ERR_NOT_FOUND" in response.content


def test_solo_mode_is_not_supported():
    with pytest.raises(ValueError, match="Solo mode not supported for banking_ru"):
        get_environment(solo_mode=True)


# --------------------------------------------------------------------------
# Инфраструктура сложности: поиск клиента, секрет, пагинация, calculate
# --------------------------------------------------------------------------


def test_find_customer_by_phone(env: Environment):
    response = call(env, "find_customer", phone="8 (916) 340-11-25")
    assert not response.error
    assert "belova_n_2201" in response.content


def test_find_customer_by_name_and_birth_date(env: Environment):
    response = call(env, "find_customer", last_name="Волкова",
                    birth_date="1987-08-14")
    assert not response.error
    assert "volkova_e_5109" in response.content
    assert "volkov_a_5108" not in response.content


def test_find_customer_not_found(env: Environment):
    response = call(env, "find_customer", phone="+7 999 000-00-00")
    assert response.error
    assert "ERR_NOT_FOUND" in response.content


def test_verify_identity_rejects_wrong_credential(env: Environment):
    response = call(env, "verify_identity", customer_id="belova_n_2201",
                    credential="пароль123")
    assert response.error
    assert "ERR_IDENTITY_MISMATCH" in response.content
    write = call(env, "block_card", card_id="card_4417", reason="lost")
    assert write.error and "ERR_NOT_VERIFIED" in write.content


def test_verify_identity_accepts_birth_date(env: Environment):
    response = call(env, "verify_identity", customer_id="belova_n_2201",
                    credential="1992-04-17")
    assert not response.error


def test_transactions_are_paginated(env: Environment):
    first = call(env, "get_transactions", customer_id="solomina_o_5214")
    assert not first.error
    import json
    page = json.loads(first.content)
    assert int(page["total"]) > int(page["limit"]) == 20
    assert len(page["items"]) == 20
    second = call(env, "get_transactions", customer_id="solomina_o_5214",
                  offset=20)
    tail = json.loads(second.content)
    assert int(tail["offset"]) == 20
    first_ids = {t["id"] for t in page["items"]}
    assert all(t["id"] not in first_ids for t in tail["items"])


def test_transactions_limit_bounds(env: Environment):
    response = call(env, "get_transactions", customer_id="solomina_o_5214",
                    limit=500)
    assert response.error
    assert "ERR_INVALID_ARGUMENT" in response.content


def test_calculate(env: Environment):
    response = call(env, "calculate", expression="500000 * 0.08 * 180 / 365")
    assert not response.error
    assert response.content == "19726.03"
    bad = call(env, "calculate", expression="__import__('os')")
    assert bad.error


def test_tool_descriptions_do_not_leak_policy(env: Environment):
    """Запреты живут в policy.md; схемы инструментов нейтральны."""
    for tool in env.get_tools():
        description = (tool.openai_schema.get("function") or {}).get(
            "description", ""
        ) if isinstance(tool.openai_schema, dict) else str(tool.openai_schema)
        for marker in ("олитика банка", "запрещает", "не подлежит разблокировке",
                       "не более одного раза"):
            assert marker not in description, (
                f"{tool.name}: описание инструмента содержит правило политики "
                f"({marker!r}) — ловушка подписана"
            )


# --------------------------------------------------------------------------
# Регрессии грейдинга, найденные baseline-прогоном v1
# --------------------------------------------------------------------------


def test_paraphrased_address_yields_same_state(env: Environment):
    """Перефразированный адрес клиента даёт то же состояние БД, что эталонный."""
    with_otp(env, "orlov_p_8814", "375204")
    call(env, "reissue_card", card_id="card_1145",
         delivery_address="Санкт-Петербург, улица Марата, дом 22, квартира 9")
    paraphrased = env.get_db_hash()

    env2 = get_environment()
    code = env2.tools.db.customers["orlov_p_8814"].code_word
    call(env2, "verify_identity", customer_id="orlov_p_8814", credential=code)
    call(env2, "send_otp", customer_id="orlov_p_8814", channel="sms")
    call(env2, "check_otp", customer_id="orlov_p_8814", code="375204")
    call(env2, "reissue_card", card_id="card_1145",
         delivery_address="г. Санкт-Петербург, ул. Марата, д. 22, кв. 9")
    assert env2.get_db_hash() == paraphrased


def test_int_amount_yields_same_state_as_float(env: Environment):
    """set_limit(300000) и set_limit(300000.0) обязаны давать одинаковый хеш."""
    verified(env, "morozova_e_3305")
    call(env, "set_limit", card_id="card_8823",
         limit_type="daily_cash_withdrawal", amount=300000)
    as_int = env.get_db_hash()
    env2 = get_environment()
    verified(env2, "morozova_e_3305")
    call(env2, "set_limit", card_id="card_8823",
         limit_type="daily_cash_withdrawal", amount=300000.0)
    assert env2.get_db_hash() == as_int


def test_free_text_dispute_reason_is_rejected(env: Environment):
    verified(env, "sidorov_p_5544")
    response = call(env, "open_dispute", transaction_id="txn_990211",
                    reason="Списание после отмены подписки")
    assert response.error
    assert "ERR_INVALID_ARGUMENT" in response.content


def test_cancel_cancelled_subscription_is_rejected(env: Environment):
    """Повторная отмена не должна молча менять cancelled_at и ломать хеш."""
    verified(env, "sidorov_p_5544")
    before = env.get_db_hash()
    response = call(env, "cancel_subscription", subscription_id="sub_2214")
    assert response.error
    assert "ERR_ALREADY_CANCELLED" in response.content
    assert env.get_db_hash() == before

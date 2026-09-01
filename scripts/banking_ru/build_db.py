# -*- coding: utf-8 -*-
"""Сборка data/tau2/domains/banking_ru/db.json.

Источник истины для данных домена: числа считаются здесь, а не переносятся
руками. Объём и дистракторы — часть дизайна сложности (механизмы M2, M5):
у ключевых клиентов десятки операций, из которых под критерии задач подходят
единицы, а рядом лежат «почти подходящие».
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

TODAY = "2026-08-28"
OUT = Path(__file__).resolve().parents[2] / "data/tau2/domains/banking_ru/db.json"


def days(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def money(x):
    return round(x + 1e-9, 2)


tariffs = {
    "tariff_classic": dict(
        id="tariff_classic", name="Классический", monthly_fee=199.0,
        free_condition="Бесплатно при остатке от 100 000 ₽ или тратах от 30 000 ₽ в месяц",
        max_daily_cash_withdrawal=150000.0,
        interbank_transfer_fee_percent=1.0, interbank_transfer_fee_refundable=False),
    "tariff_comfort": dict(
        id="tariff_comfort", name="Комфорт", monthly_fee=299.0,
        free_condition="Бесплатно при остатке от 300 000 ₽",
        max_daily_cash_withdrawal=300000.0,
        interbank_transfer_fee_percent=0.5, interbank_transfer_fee_refundable=False),
    "tariff_premium": dict(
        id="tariff_premium", name="Премиум", monthly_fee=0.0,
        free_condition="Обслуживание включено в пакет",
        max_daily_cash_withdrawal=300000.0,
        interbank_transfer_fee_percent=0.0, interbank_transfer_fee_refundable=False),
    "tariff_standard": dict(
        id="tariff_standard", name="Стандарт", monthly_fee=199.0,
        free_condition="Бесплатно при остатке от 50 000 ₽",
        max_daily_cash_withdrawal=150000.0,
        interbank_transfer_fee_percent=1.0, interbank_transfer_fee_refundable=False),
}


def customer(cid, name, birth, phone, tariff, code_word, otp, email=None, address=None):
    return dict(id=cid, full_name=name, birth_date=birth, phone=phone,
                phone_confirmed=True, email=email, address=address,
                tariff_id=tariff, code_word=code_word, otp_code=otp)


customers = {c["id"]: c for c in [
    customer("belova_n_2201", "Белова Наталья Игоревна", "1992-04-17",
             "+7 916 340-11-25", "tariff_classic", "капучино", "104582",
             "belova.n@example.ru", "г. Москва, ул. Тверская, д. 14, кв. 87"),
    customer("gromov_a_1187", "Громов Андрей Сергеевич", "1999-11-02",
             "+7 926 118-77-30", "tariff_classic", "меркурий", "220417",
             "gromov.a@example.ru", "г. Москва, Ленинский пр-т, д. 41, кв. 15"),
    customer("fedorova_m_6650", "Фёдорова Мария Олеговна", "1988-06-23",
             "+7 903 665-04-12", "tariff_classic", "сирень", "551903",
             "fedorova.m@example.ru", "г. Казань, ул. Баумана, д. 7, кв. 33"),
    customer("volkov_a_5108", "Волков Алексей Дмитриевич", "1985-02-09",
             "+7 905 510-88-41", "tariff_comfort", "гранит", "730164",
             "volkov.a@example.ru", "г. Москва, ул. Профсоюзная, д. 60, кв. 210"),
    customer("volkova_e_5109", "Волкова Елена Павловна", "1987-08-14",
             "+7 905 510-89-02", "tariff_classic", "жасмин", "418290",
             "volkova.e@example.ru", "г. Москва, ул. Профсоюзная, д. 60, кв. 210"),
    customer("orlov_p_8814", "Орлов Павел Викторович", "1990-12-05",
             "+7 921 555-01-23", "tariff_comfort", "парус", "375204",
             "orlov.p@example.ru", "г. Санкт-Петербург, пр. Невский, д. 5, кв. 12"),
    customer("guseva_m_2274", "Гусева Марина Андреевна", "1983-09-30",
             "+7 917 227-46-08", "tariff_standard", "янтарь", "609311",
             "guseva.m@example.ru", "г. Нижний Новгород, ул. Большая Покровская, д. 18, кв. 4"),
    customer("sidorov_p_5544", "Сидоров Павел Николаевич", "1985-03-21",
             "+7 926 554-40-19", "tariff_classic", "балтика", "883012",
             "sidorov.p@example.ru", "г. Москва, ул. Академика Королёва, д. 9, кв. 44"),
    customer("morozova_e_3305", "Морозова Елена Ивановна", "1958-01-26",
             "+7 916 330-55-71", "tariff_comfort", "ландыш", "146725",
             "morozova.e@example.ru", "г. Москва, ул. Гарибальди, д. 23, кв. 108"),
    customer("solomina_o_5214", "Соломина Ольга Викторовна", "1974-07-11",
             "+7 916 220-31-04", "tariff_classic", "пион", "957340",
             "solomina.o@example.ru", "г. Москва, ул. Лесная, д. 12, кв. 5"),
    customer("dorohov_v_6630", "Дорохов Виктор Андреевич", "1985-05-19",
             "+7 903 411-27-90", "tariff_premium", "антрацит", "482913",
             "dorohov.v@example.ru", "г. Москва, Кутузовский пр-т, д. 30, кв. 71"),
]}


def account(aid, cid, balance, atype="current", debt=0.0):
    return dict(id=aid, customer_id=cid, account_type=atype, currency="RUB",
                balance=balance, status="active", debt=debt)


accounts = {a["id"]: a for a in [
    account("acc_2201", "belova_n_2201", 74300.0),
    account("acc_1187", "gromov_a_1187", 138900.0),
    account("acc_6650", "fedorova_m_6650", 52400.0),
    account("acc_5108", "volkov_a_5108", 210500.0),
    account("acc_5109", "volkova_e_5109", 96700.0),
    account("acc_8814", "orlov_p_8814", 143200.0),
    account("acc_2274", "guseva_m_2274", 61250.0),
    account("acc_3301", "sidorov_p_5544", 88400.0),
    account("acc_5510", "morozova_e_3305", 15000.0),
    account("acc_5214", "solomina_o_5214", 87500.0),
    account("acc_6630", "dorohov_v_6630", 25000.0),
]}


def card(cardid, cid, accid, last4, status="active", reason=None, blocked_at=None,
         address=None, expires="2028-05-31"):
    return dict(id=cardid, customer_id=cid, account_id=accid, last4=last4,
                card_type="debit", status=status, block_reason=reason,
                blocked_at=blocked_at, expires_at=expires,
                delivery_address=address, reissue_status=None)


cards = {c["id"]: c for c in [
    card("card_4417", "belova_n_2201", "acc_2201", "4417"),
    card("card_1187", "gromov_a_1187", "acc_1187", "1187"),
    card("card_5583", "fedorova_m_6650", "acc_6650", "5583"),
    # дистрактор: вторая карта Фёдоровой, интернет-операции по ней включены
    card("card_9917", "fedorova_m_6650", "acc_6650", "9917"),
    card("card_2201", "volkov_a_5108", "acc_5108", "2201"),
    card("card_9034", "volkova_e_5109", "acc_5109", "9034"),
    card("card_1145", "orlov_p_8814", "acc_8814", "1145",
         address="г. Санкт-Петербург, пр. Невский, д. 5, кв. 12"),
    card("card_7742", "guseva_m_2274", "acc_2274", "7742"),
    card("card_6650", "sidorov_p_5544", "acc_3301", "6650"),
    card("card_8823", "morozova_e_3305", "acc_5510", "8823"),
    card("card_7733", "solomina_o_5214", "acc_5214", "7733"),
    card("card_2290", "solomina_o_5214", "acc_5214", "2290",
         status="blocked", reason="stolen", blocked_at="2026-08-25"),
]}


def limits(cardid, cash, internet=True, sbp=150000.0):
    return dict(card_id=cardid, daily_cash_withdrawal=cash,
                internet_operations_enabled=internet, sbp_limit=sbp)


card_limits = {l["card_id"]: l for l in [
    limits("card_4417", 100000.0),
    limits("card_1187", 100000.0),
    limits("card_5583", 100000.0, internet=False),
    limits("card_9917", 100000.0),
    limits("card_2201", 150000.0),
    limits("card_9034", 100000.0),
    limits("card_1145", 150000.0),
    limits("card_7742", 100000.0),
    limits("card_6650", 100000.0),
    limits("card_8823", 100000.0),
    limits("card_7733", 100000.0),
    limits("card_2290", 100000.0),
]}


def txn(tid, cid, accid, date_, amount, merchant, cardid=None, mcc=None,
        status="posted", kind="purchase", hold_expires=None, fee=None,
        is_sub=False, country="RU", channel="pos", dispute=None):
    return dict(id=tid, customer_id=cid, account_id=accid, card_id=cardid,
                date=date_, amount=amount, merchant=merchant, mcc=mcc,
                status=status, kind=kind, hold_expires_at=hold_expires,
                fee_amount=fee, is_subscription=is_sub, country=country,
                channel=channel, dispute_id=dispute, fee_refunded=False,
                cashback_granted=False)


# --- операции, на которые опираются задачи (id стабильны) -------------------
hand_transactions = [
    # bank_easy_01: спор клиента + второй, старый и закрытый (дистрактор)
    txn("txn_774410", "belova_n_2201", "acc_2201", "2026-08-18", 3450.0,
        "AliExpress", cardid="card_4417", mcc="5399", channel="online",
        country="CN", dispute="dsp_3391"),
    txn("txn_774402", "belova_n_2201", "acc_2201", "2026-08-16", 1290.0,
        "Ozon", cardid="card_4417", mcc="5399", channel="online"),
    txn("txn_771230", "belova_n_2201", "acc_2201", "2026-06-30", 2190.0,
        "Ozon", cardid="card_4417", mcc="5399", channel="online",
        dispute="dsp_3350"),
    txn("txn_771244", "belova_n_2201", "acc_2201", "2026-06-12", 5140.0,
        "AliExpress", cardid="card_4417", mcc="5399", channel="online",
        country="CN"),
    # bank_easy_02: свежая кофейня + старые кофейни-дистракторы
    txn("txn_552290", "gromov_a_1187", "acc_1187", "2026-08-25", 1200.0,
        "Кофемания", cardid="card_1187", mcc="5812"),
    txn("txn_552284", "gromov_a_1187", "acc_1187", "2026-08-24", 640.0,
        "Магнит", cardid="card_1187", mcc="5411"),
    txn("txn_552201", "gromov_a_1187", "acc_1187", "2026-06-14", 890.0,
        "Кофемания", cardid="card_1187", mcc="5812"),
    txn("txn_552202", "gromov_a_1187", "acc_1187", "2026-07-19", 1050.0,
        "Кофемания", cardid="card_1187", mcc="5812"),
    txn("txn_552295", "gromov_a_1187", "acc_1187", "2026-08-26", 640.0,
        "Аптека Ригла", cardid="card_1187", mcc="5912"),
    # bank_024: целевой перевод + два перевода-дистрактора
    txn("txn_901120", "guseva_m_2274", "acc_2274", "2026-08-22", 50000.0,
        "Перевод в банк «Восток» на счёт •••• 9021", kind="transfer",
        fee=500.0, channel="online"),
    txn("txn_898800", "guseva_m_2274", "acc_2274", "2026-07-05", 50000.0,
        "Перевод в банк «Восток» на счёт •••• 9021", kind="transfer",
        fee=500.0, channel="online"),
    # техническая ошибка: удержано 2% вместо 1% по тарифу — излишек 120 ₽
    txn("txn_900030", "guseva_m_2274", "acc_2274", "2026-08-11", 12000.0,
        "Перевод в АО «Северный банк»", kind="transfer",
        fee=240.0, channel="online"),
    txn("txn_901125", "guseva_m_2274", "acc_2274", "2026-08-25", 780.0,
        "Магнит", cardid="card_7742", mcc="5411"),
    # bank_medium_01: спорное списание + история ежемесячных списаний
    txn("txn_990211", "sidorov_p_5544", "acc_3301", "2026-08-25", 299.0,
        "Яндекс.Плюс", cardid="card_6650", mcc="5968", is_sub=True,
        channel="online"),
    txn("txn_990180", "sidorov_p_5544", "acc_3301", "2026-08-12", 299.0,
        "Яндекс.Плюс", cardid="card_6650", mcc="5968", is_sub=True,
        channel="online"),
    txn("txn_990101", "sidorov_p_5544", "acc_3301", "2026-05-12", 299.0,
        "YANDEX PLUS", cardid="card_6650", mcc="5968", is_sub=True,
        channel="online"),
    txn("txn_990102", "sidorov_p_5544", "acc_3301", "2026-06-12", 299.0,
        "Яндекс Плюс", cardid="card_6650", mcc="5968", is_sub=True,
        channel="online"),
    txn("txn_990103", "sidorov_p_5544", "acc_3301", "2026-07-12", 299.0,
        "Яндекс.Плюс", cardid="card_6650", mcc="5968", is_sub=True,
        channel="online"),
    txn("txn_990250", "sidorov_p_5544", "acc_3301", "2026-08-21", 430.0,
        "Пятёрочка", cardid="card_6650", mcc="5411"),
    txn("txn_990251", "sidorov_p_5544", "acc_3301", "2026-08-24", 1890.0,
        "Летуаль", cardid="card_6650", mcc="5977"),
    # bank_hard_01: операции по украденной карте + три «почти подходящих»
    txn("txn_100201", "solomina_o_5214", "acc_5214", "2026-08-24", 5200.0,
        "Пятёрочка", cardid="card_2290", mcc="5411"),
    txn("txn_100202", "solomina_o_5214", "acc_5214", "2026-08-26", 4500.0,
        "Adobe Systems", cardid="card_2290", mcc="5734", channel="online",
        country="US"),
    txn("txn_100203", "solomina_o_5214", "acc_5214", "2026-08-26", 12300.0,
        "Wildberries", cardid="card_2290", mcc="5399", status="hold",
        hold_expires="2026-08-31", channel="online"),
    txn("txn_100204", "solomina_o_5214", "acc_5214", "2026-08-27", 3100.0,
        "Steam Games", cardid="card_2290", mcc="5816", channel="online",
        country="LU"),
    txn("txn_100205", "solomina_o_5214", "acc_5214", "2026-08-27", 900.0,
        "Такси Яндекс", cardid="card_2290", mcc="4121", channel="online"),
    txn("txn_100206", "solomina_o_5214", "acc_5214", "2026-08-26", 2999.0,
        "Ozon", cardid="card_2290", mcc="5399", channel="online"),
    txn("txn_100207", "solomina_o_5214", "acc_5214", "2026-08-24", 3200.0,
        "Лента", cardid="card_2290", mcc="5411"),
    txn("txn_100208", "solomina_o_5214", "acc_5214", "2026-08-27", 5500.0,
        "М.Видео", cardid="card_2290", mcc="5732", status="declined",
        channel="online"),
    txn("txn_100210", "solomina_o_5214", "acc_5214", "2026-08-20", 199.0,
        "Комиссия за обслуживание карты", cardid="card_7733", kind="fee",
        channel="online"),
]

# --- фоновый объём: детерминированный генератор -----------------------------
FILLER_PLAN = {
    # customer_id: (счёт, карта, число операций)
    "belova_n_2201": ("acc_2201", "card_4417", 40),
    "gromov_a_1187": ("acc_1187", "card_1187", 85),
    "fedorova_m_6650": ("acc_6650", "card_9917", 30),
    "volkov_a_5108": ("acc_5108", "card_2201", 16),
    "orlov_p_8814": ("acc_8814", "card_1145", 24),
    "guseva_m_2274": ("acc_2274", "card_7742", 45),
    "sidorov_p_5544": ("acc_3301", "card_6650", 75),
    "morozova_e_3305": ("acc_5510", "card_8823", 18),
    "solomina_o_5214": ("acc_5214", "card_7733", 85),
    "dorohov_v_6630": ("acc_6630", None, 30),
}
FILLER_MERCHANTS = [
    ("Пятёрочка", "5411"), ("Перекрёсток", "5411"), ("Магнит", "5411"),
    ("Яндекс Такси", "4121"), ("Аптека 36,6", "5912"), ("Лукойл АЗС", "5541"),
    ("Ozon", "5399"), ("Wildberries", "5399"), ("Детский мир", "5945"),
    ("Леруа Мерлен", "5200"), ("Спортмастер", "5941"), ("ВкусВилл", "5411"),
    ("Читай-город", "5942"), ("Бургер Кинг", "5814"), ("МТС", "4814"),
]
# Фон заканчивается 2026-08-20: окна задач (18–28.08) наполняются только
# операциями, заведёнными руками выше, — иначе фон случайно попадёт под
# критерии задач (кафе Громова, переводы Гусевой, карта Соломиной).
FILLER_START, FILLER_END = date(2026, 6, 1), date(2026, 8, 20)

rng = random.Random(20260901)
filler = []
next_id = 200001
span = (FILLER_END - FILLER_START).days
for cid, (accid, cardid, count) in FILLER_PLAN.items():
    for _ in range(count):
        d = FILLER_START + timedelta(days=rng.randrange(span + 1))
        merchant, mcc = rng.choice(FILLER_MERCHANTS)
        # у Громова фон не должен добавлять кафе и рестораны: кешбэк-задача
        # опирается на то, что свежая «Кофемания» — единственная в своём окне
        if cid == "gromov_a_1187" and mcc in ("5812", "5813", "5814"):
            merchant, mcc = "ВкусВилл", "5411"
        filler.append(txn(
            f"txn_{next_id}", cid, accid, d.isoformat(),
            money(rng.uniform(90, 6800)), merchant, cardid=cardid, mcc=mcc,
            channel=rng.choice(["pos", "online"])))
        next_id += 1

transactions = {t["id"]: t for t in hand_transactions + filler}
assert len(transactions) == len(hand_transactions) + len(filler), "коллизия id"

disputes = {
    "dsp_3391": dict(
        id="dsp_3391", customer_id="belova_n_2201", transaction_id="txn_774410",
        status="under_review", filed_at="2026-08-20", sla_days=30,
        reason="Товар не доставлен", amount=3450.0),
    "dsp_3350": dict(
        id="dsp_3350", customer_id="belova_n_2201", transaction_id="txn_771230",
        status="approved", filed_at="2026-07-02", sla_days=30,
        reason="Двойное списание", amount=2190.0),
}


def sub(sid, cid, name, amount, status="active", paid_until=None,
        cancelled_at=None, next_charge=None):
    return dict(id=sid, customer_id=cid, name=name, amount=amount, status=status,
                paid_until=paid_until, cancelled_at=cancelled_at,
                next_charge_date=next_charge)


subscriptions = {s["id"]: s for s in [
    sub("sub_2214", "sidorov_p_5544", "Яндекс.Плюс", 299.0, "cancelled",
        paid_until="2026-08-15", cancelled_at="2026-08-10"),
    sub("sub_3301", "solomina_o_5214", "Яндекс Плюс", 399.0, next_charge="2026-09-05"),
    sub("sub_3302", "solomina_o_5214", "Кинопоиск HD", 599.0, next_charge="2026-09-07"),
    sub("sub_3303", "solomina_o_5214", "Спортмастер Pro", 890.0, next_charge="2026-09-12"),
    sub("sub_3304", "solomina_o_5214", "Облако Mail 128GB", 149.0, next_charge="2026-09-03"),
    sub("sub_3305", "solomina_o_5214", "Delivery Club Plus", 449.0, next_charge="2026-09-09"),
    sub("sub_3306", "solomina_o_5214", "Х5 подписка", 599.0, "cancelled",
        paid_until="2026-07-31", cancelled_at="2026-07-10"),
]}

autopayments = {"ap_5540": dict(
    id="ap_5540", customer_id="sidorov_p_5544", merchant="Яндекс.Плюс",
    amount=299.0, status="active", cancelled_at=None)}

# --- вклады: производные суммы считаются здесь ------------------------------
dep_4471_days = days("2026-03-01", TODAY)
dep_4471_early = money(500000 * 0.01 * dep_4471_days / 365)
dep_6630_days = days("2026-02-01", TODAY)
dep_6630_early = money(500000 * 0.01 * dep_6630_days / 365)

deposits = {
    "dep_4471": dict(
        id="dep_4471", customer_id="morozova_e_3305", name="Пенсионный",
        amount=500000.0, rate=0.08, early_rate=0.01, opened_at="2026-03-01",
        matures_at="2027-03-01", payout_account_id="acc_5510",
        early_withdrawal_payout=money(500000 + dep_4471_early),
        maturity_payout=money(500000 + 500000 * 0.08),
        status="active", closed_at=None),
    "dep_6630": dict(
        id="dep_6630", customer_id="dorohov_v_6630", name="Накопительный",
        amount=500000.0, rate=0.09, early_rate=0.01, opened_at="2026-02-01",
        matures_at="2027-02-01", payout_account_id="acc_6630",
        early_withdrawal_payout=money(500000 + dep_6630_early),
        maturity_payout=545000.0,
        status="active", closed_at=None),
}

loans = {"ln_6630": dict(
    id="ln_6630", customer_id="dorohov_v_6630", principal=380000.0,
    accrued_interest=2300.0, rate=0.149, monthly_payment=18500.0, days_overdue=3,
    next_payment_date="2026-09-10",
    penalties=[
        dict(id="pen_6630_1", amount=900.0, accrued_at="2026-03-15",
             reason="Просрочка платежа 2 дня", waived=True, paid=False),
        dict(id="pen_6630_2", amount=1200.0, accrued_at="2026-08-10",
             reason="Просрочка платежа 3 дня", waived=False, paid=False),
    ],
    waivers_used=1, max_waivers=1, status="active")}

cashback_rules = {"gromov_a_1187": dict(
    customer_id="gromov_a_1187",
    categories=[
        dict(name="Рестораны и кафе", rate=0.05, mcc_codes=["5812", "5813", "5814"]),
        dict(name="Супермаркеты", rate=0.02, mcc_codes=["5411", "5499"]),
    ],
    excluded_mcc=["6011", "4829"], payout_day=5, accrued_current_period=0.0)}

db = dict(
    today=TODAY, dispute_window_days=120, tariffs=tariffs, customers=customers,
    accounts=accounts, cards=cards, card_limits=card_limits,
    transactions=transactions, disputes=disputes, subscriptions=subscriptions,
    autopayments=autopayments, deposits=deposits, loans=loans,
    cashback_rules=cashback_rules)

OUT.write_text(json.dumps(db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"операций: {len(transactions)} (руками {len(hand_transactions)}, фон {len(filler)})")
print("вклад Морозовой: выплата", deposits["dep_4471"]["early_withdrawal_payout"],
      "| потеря процентов:", money(500000 * 0.08 * dep_4471_days / 365 - dep_4471_early))
payout_6630 = deposits["dep_6630"]["early_withdrawal_payout"]
print("вклад Дорохова: дней", dep_6630_days, "| выплата досрочно", payout_6630)
print("погашение Дорохова:", 380000 + 2300 + 1200,
      "| остаток на счёт:", money(payout_6630 - 383500),
      "| баланс:", money(25000 + payout_6630 - 383500))

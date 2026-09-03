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
        max_daily_cash_withdrawal=150000.0, max_sbp_limit=150000.0,
        interbank_transfer_fee_percent=1.0, interbank_transfer_fee_refundable=False,
        # bank_031: клиент уверен, что валютный счёт бесплатный
        foreign_account_opening_fee=500.0, foreign_account_monthly_fee=99.0),
    "tariff_comfort": dict(
        id="tariff_comfort", name="Комфорт", monthly_fee=299.0,
        free_condition="Бесплатно при остатке от 300 000 ₽",
        max_daily_cash_withdrawal=300000.0, max_sbp_limit=300000.0,
        interbank_transfer_fee_percent=0.5, interbank_transfer_fee_refundable=False),
    "tariff_premium": dict(
        id="tariff_premium", name="Премиум", monthly_fee=0.0,
        free_condition="Обслуживание включено в пакет",
        max_daily_cash_withdrawal=300000.0, max_sbp_limit=500000.0,
        interbank_transfer_fee_percent=0.0, interbank_transfer_fee_refundable=False),
    "tariff_standard": dict(
        id="tariff_standard", name="Стандарт", monthly_fee=199.0,
        # текст условия обязан совпадать с полями free_min_*: в первом прогоне
        # волны 6 расхождение (50 000 против 10 000) увело клиента и агента
        free_condition="Бесплатно при среднем остатке от 10 000 ₽ или обороте от 30 000 ₽ в месяц",
        max_daily_cash_withdrawal=150000.0, max_sbp_limit=300000.0,
        interbank_transfer_fee_percent=1.0, interbank_transfer_fee_refundable=False,
        free_min_balance=10000.0, free_min_turnover=30000.0),
    "tariff_premium_plus": dict(
        id="tariff_premium_plus", name="Premium+", monthly_fee=299.0,
        free_condition="Обслуживание 299 ₽ в месяц, кешбэк до 3%",
        max_daily_cash_withdrawal=300000.0, max_sbp_limit=500000.0,
        interbank_transfer_fee_percent=0.0, interbank_transfer_fee_refundable=False),
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
    # --- волна 2: споры, антифрод и переводы (bank_011–bank_020) -----------
    customer("smirnov_d_5502", "Смирнов Дмитрий Олегович", "1991-03-12",
             "+7 916 550-27-14", "tariff_premium", "мангал", "611203",
             "smirnov.d@example.ru", "г. Москва, ул. Складочная, д. 6, кв. 140"),
    customer("kuznecova_o_3391", "Кузнецова Ольга Борисовна", "1979-10-08",
             "+7 921 339-16-02", "tariff_classic", "форель", "330192",
             "kuznecova.o@example.ru", "г. Санкт-Петербург, ул. Савушкина, д. 11, кв. 62"),
    customer("volkov_m_9043", "Волков Максим Игоревич", "1986-07-23",
             "+7 903 904-31-77", "tariff_standard", "кремль", "904317",
             "volkov.m@example.ru", "г. Москва, ул. Народного Ополчения, д. 24, кв. 3"),
    customer("smirnova_o_1123", "Смирнова Ольга Леонидовна", "1987-12-03",
             "+7 916 112-30-88", "tariff_classic", "клюква", "112308",
             "smirnova.o@example.ru", "г. Москва, ул. Дубнинская, д. 40, кв. 219"),
    customer("morozov_s_8850", "Морозов Сергей Петрович", "1971-09-15",
             "+7 916 885-03-12", "tariff_classic", "титан", "885031",
             "morozov.s@example.ru", "г. Москва, ул. Молодогвардейская, д. 8, кв. 51"),
    customer("kovaleva_n_9902", "Ковалёва Наталья Юрьевна", "1981-04-22",
             "+7 926 990-24-17", "tariff_classic", "карамель", "990241",
             "kovaleva.n@example.ru", "г. Москва, Открытое шоссе, д. 19, кв. 77"),
    customer("lebedev_i_7729", "Лебедев Игорь Валерьевич", "1984-11-19",
             "+7 905 772-93-16", "tariff_classic", "омуль", "772931",
             "lebedev.i@example.ru", "г. Москва, ул. Юных Ленинцев, д. 51, кв. 27"),
    customer("egorova_t_2266", "Егорова Татьяна Сергеевна", "1982-06-11",
             "+7 926 226-64-05", "tariff_comfort", "клевер", "226640",
             "egorova.t@example.ru", "г. Москва, ул. Полярная, д. 32, кв. 8"),
    customer("nikitin_r_5581", "Никитин Роман Алексеевич", "1995-01-30",
             "+7 917 558-11-24", "tariff_standard", "стрела", "558112",
             "nikitin.r@example.ru", "г. Казань, ул. Декабристов, д. 85, кв. 12"),
    customer("soloveva_v_9034", "Соловьёва Виктория Олеговна", "1989-08-07",
             "+7 903 913-42-60", "tariff_classic", "ромашка", "913426",
             "soloveva.v@example.ru", "г. Москва, ул. Шверника, д. 4, кв. 96"),
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
    account("acc_5502", "smirnov_d_5502", 164800.0),
    account("acc_3391", "kuznecova_o_3391", 43600.0),
    account("acc_9043", "volkov_m_9043", 71200.0),
    account("acc_1123", "smirnova_o_1123", 61500.0),
    account("acc_8850", "morozov_s_8850", 8300.0),
    account("acc_9902", "kovaleva_n_9902", 78300.0),
    account("acc_7729", "lebedev_i_7729", 27500.0),
    account("acc_2266", "egorova_t_2266", 112400.0),
    account("acc_5581", "nikitin_r_5581", 268300.0),
    account("acc_9134", "soloveva_v_9034", 46100.0),
]}


def card(cardid, cid, accid, last4, status="active", reason=None, blocked_at=None,
         address=None, expires="2028-05-31", card_type="debit"):
    return dict(id=cardid, customer_id=cid, account_id=accid, last4=last4,
                card_type=card_type, status=status, block_reason=reason,
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
    card("card_7734", "smirnov_d_5502", "acc_5502", "7734"),
    card("card_2210", "kuznecova_o_3391", "acc_3391", "2210"),
    card("card_9043", "volkov_m_9043", "acc_9043", "9043", card_type="credit"),
    card("card_8890", "smirnova_o_1123", "acc_1123", "8890"),
    card("card_8850", "morozov_s_8850", "acc_8850", "8850"),
    card("card_4478", "kovaleva_n_9902", "acc_9902", "4478"),
    card("card_7729", "lebedev_i_7729", "acc_7729", "7729"),
    card("card_7730", "lebedev_i_7729", "acc_7729", "7730"),
    card("card_2266", "egorova_t_2266", "acc_2266", "2266"),
    card("card_5581", "nikitin_r_5581", "acc_5581", "5581"),
    card("card_9134", "soloveva_v_9034", "acc_9134", "9134"),
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
    limits("card_7734", 300000.0, sbp=500000.0),
    limits("card_2210", 100000.0),
    limits("card_9043", 100000.0),
    limits("card_8890", 100000.0),
    limits("card_8850", 100000.0),
    limits("card_4478", 100000.0),
    limits("card_7729", 100000.0),
    limits("card_7730", 100000.0),
    limits("card_2266", 150000.0, sbp=300000.0),
    # bank_019: собственный лимит СБП ниже максимума по тарифу «Стандарт»
    limits("card_5581", 150000.0, sbp=100000.0),
    limits("card_9134", 100000.0),
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

    # --- волна 2 ----------------------------------------------------------
    # bank_011: целевая покупка + три «почти подходящие» (тот же мерчант,
    # та же сумма, покупка вне 120-дневного окна)
    txn("txn_771204", "smirnov_d_5502", "acc_5502", "2026-08-16", 18990.0,
        "ООО ТехноМаркет", cardid="card_7734", mcc="5732", channel="online"),
    txn("txn_771260", "smirnov_d_5502", "acc_5502", "2026-08-22", 3500.0,
        "Возврат от ООО ТехноМаркет", kind="fee_refund", channel="online"),
    txn("txn_770880", "smirnov_d_5502", "acc_5502", "2026-07-03", 4590.0,
        "ООО ТехноМаркет", cardid="card_7734", mcc="5732", channel="online"),
    txn("txn_768120", "smirnov_d_5502", "acc_5502", "2026-03-12", 18990.0,
        "ООО ТехноМаркет", cardid="card_7734", mcc="5732", channel="online"),
    txn("txn_771190", "smirnov_d_5502", "acc_5502", "2026-08-14", 18990.0,
        "DNS Технопоинт", cardid="card_7734", mcc="5732", channel="online"),
    # bank_012: операция вне окна + похожая внутри окна (дистрактор)
    txn("txn_540117", "kuznecova_o_3391", "acc_3391", "2026-02-20", 7450.0,
        "ИП Сергеева, магазин «Домашний уют»", cardid="card_2210", mcc="5719",
        channel="online"),
    txn("txn_552140", "kuznecova_o_3391", "acc_3391", "2026-05-18", 1290.0,
        "ИП Сергеева, магазин «Домашний уют»", cardid="card_2210", mcc="5719",
        channel="online"),
    # bank_013: три спора, отозвать нужно ровно один
    txn("txn_660812", "volkov_m_9043", "acc_9043", "2026-08-10", 32000.0,
        "ООО Стройторг", cardid="card_9043", mcc="5211", channel="online",
        dispute="dsp_3400"),
    txn("txn_640455", "volkov_m_9043", "acc_9043", "2026-07-02", 18700.0,
        "Стройторг Онлайн", cardid="card_9043", mcc="5211", channel="online",
        dispute="dsp_3380"),
    txn("txn_661140", "volkov_m_9043", "acc_9043", "2026-08-14", 14900.0,
        "Стройторг Маркет", cardid="card_9043", mcc="5211", channel="online",
        dispute="dsp_3402"),
    txn("txn_661500", "volkov_m_9043", "acc_9043", "2026-08-12", 590.0,
        "Комиссия за экспресс-рассмотрение спора", kind="fee", channel="online"),
    txn("txn_662301", "volkov_m_9043", "acc_9043", "2026-08-18", 27400.0,
        "М.Видео", cardid="card_9043", mcc="5732", channel="online",
        dispute="dsp_3401"),
    # bank_008: незнакомые клиентке списания — на деле её же подписки
    txn("txn_055210", "smirnova_o_1123", "acc_1123", "2026-08-25", 3000.0,
        "YANDEX PLUS MOSCOW", cardid="card_8890", mcc="5968", is_sub=True,
        channel="online"),
    txn("txn_055200", "smirnova_o_1123", "acc_1123", "2026-08-24", 2490.0,
        "GOOGLE *SERVICES", cardid="card_8890", mcc="5817", channel="online"),
    txn("txn_055180", "smirnova_o_1123", "acc_1123", "2026-08-22", 1490.0,
        "LITRES.RU", cardid="card_8890", mcc="5942", is_sub=True,
        channel="online"),
    txn("txn_055150", "smirnova_o_1123", "acc_1123", "2026-07-25", 3000.0,
        "Яндекс Плюс", cardid="card_8890", mcc="5968", is_sub=True,
        channel="online"),
    txn("txn_055120", "smirnova_o_1123", "acc_1123", "2026-06-25", 3000.0,
        "YANDEX PLUS MOSCOW", cardid="card_8890", mcc="5968", is_sub=True,
        channel="online"),
    # bank_015: три перевода мошенникам одним днём + бытовая покупка того же дня
    txn("txn_810391", "morozov_s_8850", "acc_8850", "2026-08-27", 95000.0,
        "Перевод по СБП, К. А. Т., АО «Юнистрим-Банк»", kind="transfer",
        channel="online"),
    txn("txn_810388", "morozov_s_8850", "acc_8850", "2026-08-27", 45000.0,
        "Перевод по СБП, К. А. Т., АО «Юнистрим-Банк»", kind="transfer",
        channel="online"),
    txn("txn_810402", "morozov_s_8850", "acc_8850", "2026-08-27", 30000.0,
        "Перевод по СБП, К. А. Т., АО «Юнистрим-Банк»", kind="transfer",
        channel="online"),
    txn("txn_810377", "morozov_s_8850", "acc_8850", "2026-08-27", 1240.0,
        "Пятёрочка", cardid="card_8850", mcc="5411"),
    # bank_010: две мошеннические операции за границей + легитимные
    # зарубежные списания (подписка и давняя бронь)
    txn("txn_088345", "kovaleva_n_9902", "acc_9902", "2026-08-27", 12400.0,
        "SHOP-BANGKOK TH", cardid="card_4478", mcc="5399", channel="online",
        country="TH"),
    txn("txn_088320", "kovaleva_n_9902", "acc_9902", "2026-08-26", 4100.0,
        "SHOP-BANGKOK TH", cardid="card_4478", mcc="5399", channel="online",
        country="TH"),
    txn("txn_088300", "kovaleva_n_9902", "acc_9902", "2026-08-25", 990.0,
        "NETFLIX.COM", cardid="card_4478", mcc="5968", channel="online",
        country="US", is_sub=True),
    txn("txn_087100", "kovaleva_n_9902", "acc_9902", "2026-06-18", 34500.0,
        "BOOKING.COM", cardid="card_4478", mcc="7011", channel="online",
        country="NL"),
    # bank_017: холд по отклонённой оплате + проведённая оплата того же
    # ресторана неделей раньше + посторонний холд
    txn("txn_799102", "lebedev_i_7729", "acc_7729", "2026-08-27", 4200.0,
        "Ресторан «Гастроном №1»", cardid="card_7729", mcc="5812",
        status="hold", hold_expires="2026-09-03"),
    txn("txn_798540", "lebedev_i_7729", "acc_7729", "2026-08-20", 4200.0,
        "Ресторан «Гастроном №1»", cardid="card_7730", mcc="5812"),
    txn("txn_799050", "lebedev_i_7729", "acc_7729", "2026-08-26", 1850.0,
        "Лукойл АЗС", cardid="card_7729", mcc="5541", status="hold",
        hold_expires="2026-09-02"),
    # bank_018: ошибочный перевод + переводы похожим получателям
    txn("txn_844217", "egorova_t_2266", "acc_2266", "2026-08-27", 50000.0,
        "Перевод по СБП, Смирнов А. А., банк «Восток»", kind="transfer",
        channel="online"),
    txn("txn_844100", "egorova_t_2266", "acc_2266", "2026-08-25", 12000.0,
        "Перевод по СБП, Смирнова Е. А., банк «Восток»", kind="transfer",
        channel="online"),
    txn("txn_843900", "egorova_t_2266", "acc_2266", "2026-08-24", 7500.0,
        "Перевод по СБП, Егоров П. С., банк «Юг»", kind="transfer",
        channel="online"),
    txn("txn_843710", "egorova_t_2266", "acc_2266", "2026-08-22", 3200.0,
        "Перевод по СБП, Смирнов А. А., банк «Восток»", kind="transfer",
        channel="online"),
    # bank_020: перевод в обработке + такой же исполненный двумя неделями раньше
    txn("txn_861530", "soloveva_v_9034", "acc_9134", "2026-08-26", 10000.0,
        "Перевод по номеру карты •••• 4417", kind="transfer",
        status="processing", channel="online"),
    txn("txn_860210", "soloveva_v_9034", "acc_9134", "2026-08-12", 10000.0,
        "Перевод по номеру карты •••• 4417", kind="transfer", channel="online"),
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
    "smirnov_d_5502": ("acc_5502", "card_7734", 120),
    "kuznecova_o_3391": ("acc_3391", "card_2210", 40),
    "volkov_m_9043": ("acc_9043", "card_9043", 30),
    "smirnova_o_1123": ("acc_1123", "card_8890", 45),
    "morozov_s_8850": ("acc_8850", "card_8850", 35),
    "kovaleva_n_9902": ("acc_9902", "card_4478", 60),
    "lebedev_i_7729": ("acc_7729", "card_7729", 40),
    "egorova_t_2266": ("acc_2266", "card_2266", 50),
    "nikitin_r_5581": ("acc_5581", "card_5581", 55),
    "soloveva_v_9034": ("acc_9134", "card_9134", 35),
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
# Исключение: у Смирнова фон продлён до 28 августа. Цель bank_011 (16.08)
# обязана лежать за первой страницей выдачи, иначе «перебор» сводится к
# одному вызову get_transactions. Мерчанты фона не пересекаются с задачей.
LATE_FILLER = {"smirnov_d_5502": (date(2026, 8, 17), date(2026, 8, 28), 24)}

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

for cid, (start, end, count) in LATE_FILLER.items():
    accid, cardid, _ = FILLER_PLAN[cid]
    late_span = (end - start).days
    for _ in range(count):
        d = start + timedelta(days=rng.randrange(late_span + 1))
        merchant, mcc = rng.choice(FILLER_MERCHANTS)
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
    # bank_013: отзыву подлежит только dsp_3400
    "dsp_3400": dict(
        id="dsp_3400", customer_id="volkov_m_9043", transaction_id="txn_660812",
        status="under_review", filed_at="2026-08-12", sla_days=30,
        reason="Товар не доставлен", amount=32000.0),
    "dsp_3401": dict(
        id="dsp_3401", customer_id="volkov_m_9043", transaction_id="txn_662301",
        status="under_review", filed_at="2026-08-20", sla_days=30,
        reason="Товар не соответствует описанию", amount=27400.0),
    "dsp_3402": dict(
        id="dsp_3402", customer_id="volkov_m_9043", transaction_id="txn_661140",
        status="under_review", filed_at="2026-08-16", sla_days=30,
        reason="Товар не доставлен", amount=14900.0),
    "dsp_3380": dict(
        id="dsp_3380", customer_id="volkov_m_9043", transaction_id="txn_640455",
        status="approved", filed_at="2026-07-05", sla_days=30,
        reason="Двойное списание", amount=18700.0),
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
    # bank_008: обе «незнакомые» операции соответствуют активным подпискам
    sub("sub_0771", "smirnova_o_1123", "Яндекс Плюс", 3000.0,
        next_charge="2026-09-25"),
    sub("sub_0772", "smirnova_o_1123", "Литрес Подписка", 1490.0,
        next_charge="2026-09-22"),
    # bank_010: единственное легитимное зарубежное списание — подписка
    sub("sub_9902", "kovaleva_n_9902", "Netflix", 990.0, next_charge="2026-09-25"),
]}

autopayments = {"ap_5540": dict(
    id="ap_5540", customer_id="sidorov_p_5544", merchant="Яндекс.Плюс",
    amount=299.0, status="active", cancelled_at=None)}


# ============================================================================
# Волны 3–5: клиенты, счета и объекты задач bank_001…bank_050
# ============================================================================

def dev(did, cid, model_, last_login, blocked=False, country="RU"):
    return dict(id=did, customer_id=cid, model=model_, last_login=last_login,
                country=country, blocked=blocked)


wave35_customers = [
    # --- остаток блока A ---------------------------------------------------
    customer("petrova_i_4821", "Петрова Ирина Сергеевна", "1988-11-30",
             "+7 900 123-45-67", "tariff_classic", "маяк", "482100",
             "petrova.i@example.ru", "г. Москва, ул. Лесная, д. 3, кв. 18"),
    customer("sokolov_d_3390", "Соколов Дмитрий Игоревич", "1990-05-16",
             "+7 903 339-07-12", "tariff_standard", "берёза", "339012",
             "sokolov.d@example.ru", "г. Москва, ул. Мира, д. 8, кв. 44"),
    customer("nikitin_s_7742", "Никитин Сергей Павлович", "1978-02-11",
             "+7 916 774-21-08", "tariff_premium", "оникс", "774210",
             "nikitin.s@example.ru", "г. Москва, Ленинградское ш., д. 5, кв. 202"),
    customer("kuznecova_a_2957", "Кузнецова Анна Дмитриевна", "1994-09-08",
             "+7 926 295-70-33", "tariff_classic", "лаванда", "295703",
             "kuznecova.a@example.ru", "г. Москва, ул. Ленина, д. 14, кв. 27"),
    customer("volkov_a_4471", "Волков Артём Сергеевич", "1996-03-24",
             "+7 905 447-10-96", "tariff_classic", "кедр", "447109",
             "volkov.a4@example.ru", "г. Москва, ул. Крылатская, д. 2, кв. 66"),
    # --- блок C ------------------------------------------------------------
    customer("sokolov_d_2208", "Соколов Денис Валерьевич", "1983-07-19",
             "+7 903 220-84-51", "tariff_standard", "невод", "220845",
             "sokolov.den@example.ru", "г. Москва, ул. Строителей, д. 12, кв. 5"),
    customer("volkov_a_5512", "Волков Алексей Юрьевич", "1987-12-14",
             "+7 916 551-23-70", "tariff_classic", "сокол", "551237",
             "volkov.a5@example.ru", "г. Москва, ул. Гарибальди, д. 7, кв. 91"),
    customer("nikitin_s_6631", "Никитин Станислав Олегович", "1980-06-02",
             "+7 903 663-11-58", "tariff_premium", "азимут", "663115",
             "nikitin.st@example.ru", "г. Москва, Кутузовский пр-т, д. 12, кв. 40"),
    customer("kuznecova_o_4103", "Кузнецова Ольга Петровна", "1983-04-27",
             "+7 921 410-30-24", "tariff_classic", "рябина", "410302",
             "o.kuznecova.83@mail.ru", "г. Санкт-Петербург, ул. Чайковского, д. 9, кв. 3"),
    customer("egorov_p_7215", "Егоров Павел Андреевич", "1991-10-05",
             "+7 926 721-50-88", "tariff_standard", "фрегат", "721508",
             "egorov.p@example.ru", "г. Москва, ул. Профсоюзная, д. 15, кв. 72"),
    customer("titov_v_8890", "Титов Виктор Николаевич", "1975-01-13",
             "+7 916 889-04-27", "tariff_standard", "гранат", "889042",
             "titov.v@example.ru", "г. Москва, ул. Юных Ленинцев, д. 3, кв. 14"),
    customer("belov_i_2266", "Белов Игорь Валентинович", "1969-08-21",
             "+7 903 226-68-40", "tariff_comfort", "залив", "226684",
             "belov.i@example.ru", "г. Москва, ул. Островитянова, д. 21, кв. 8"),
    # --- блок D ------------------------------------------------------------
    customer("sokolov_d_5512", "Соколов Даниил Максимович", "1993-02-07",
             "+7 926 551-27-63", "tariff_classic", "компас", "551276",
             "sokolov.dan@example.ru", "г. Москва, ул. Вавилова, д. 48, кв. 12"),
    customer("smirnova_i_3390", "Смирнова Ирина Николаевна", "1981-11-11",
             "+7 916 339-04-72", "tariff_classic", "сапфир", "339047",
             "smirnova.i@example.ru", "г. Москва, ул. Дмитровская, д. 30, кв. 55"),
    customer("volkov_a_2277", "Волков Антон Русланович", "1989-06-30",
             "+7 903 227-70-15", "tariff_standard", "барьер", "227701",
             "volkov.ant@example.ru", "г. Москва, ул. Полярная, д. 5, кв. 100"),
    customer("kuznecova_m_6641", "Кузнецова Мария Львовна", "1986-01-25",
             "+7 926 664-13-09", "tariff_classic", "корица", "664130",
             "kuznecova.m@example.ru", "г. Москва, ул. Расковой, д. 11, кв. 23"),
    customer("novikov_p_8802", "Новиков Павел Игоревич", "1984-09-17",
             "+7 916 880-24-61", "tariff_classic", "ветер", "880246",
             "novikov.p@example.ru", "г. Москва, ул. Металлургов, д. 40, кв. 7"),
    customer("morozova_e_1156", "Морозова Елена Аркадьевна", "1979-03-05",
             "+7 903 115-64-28", "tariff_standard", "дюна", "115642",
             "morozova.el@example.ru", "г. Москва, ул. Планерная, д. 6, кв. 61"),
    customer("lebedev_i_4409", "Лебедев Илья Олегович", "1985-12-02",
             "+7 926 440-95-13", "tariff_classic", "штиль", "440951",
             "lebedev.il@example.ru", "г. Москва, ул. Судостроительная, д. 9, кв. 34"),
    customer("solovyova_a_7734", "Соловьёва Анна Павловна", "1992-07-08",
             "+7 916 773-40-56", "tariff_standard", "мозаика", "773405",
             "solovyova.a@example.ru", "г. Москва, ул. Академика Янгеля, д. 4, кв. 19"),
    # --- блок E ------------------------------------------------------------
    customer("morozov_a_5561", "Морозов Алексей Витальевич", "1990-04-14",
             "+7 903 556-10-92", "tariff_classic", "терраса", "556109",
             "morozov.a@example.ru", "г. Москва, ул. Багрицкого, д. 22, кв. 47"),
    customer("volkova_s_6674", "Волкова Светлана Ивановна", "1977-10-29",
             "+7 916 667-40-31", "tariff_classic", "молния", "667403",
             "volkova.s@example.ru", "г. Москва, ул. Бирюлёвская, д. 17, кв. 88"),
    customer("kovalev_n_2217", "Ковалёв Николай Егорович", "1982-05-03",
             "+7 903 555-11-22", "tariff_standard", "причал", "221706",
             "kovalev.n@example.ru", "г. Москва, ул. Ставропольская, д. 8, кв. 26"),
    customer("titova_a_7729", "Титова Анна Борисовна", "1995-02-18",
             "+7 926 772-90-64", "tariff_premium", "сирокко", "772906",
             "titova.a@example.ru", "г. Москва, ул. Крутицкий Вал, д. 3, кв. 15"),
    customer("egorov_m_1145", "Егоров Максим Дмитриевич", "1988-08-09",
             "+7 916 114-50-77", "tariff_classic", "калина", "114507",
             "egorov.m@example.ru", "г. Москва, ул. Шаболовка, д. 30, кв. 4"),
    customer("smirnova_o_2861", "Смирнова Ольга Ивановна", "1954-06-12",
             "+7 903 286-10-45", "tariff_classic", "рассвет", "286104",
             "smirnova.olg@example.ru", "г. Москва, ул. Дубравная, д. 41, кв. 2"),
    customer("novikov_v_9034", "Новиков Виктор Семёнович", "1972-11-23",
             "+7 926 903-41-58", "tariff_standard", "причуда", "903415",
             "novikov.v@example.ru", "г. Москва, Ленинский пр-т, д. 95, кв. 30"),
    customer("grigoreva_e_4408", "Григорьева Елена Максимовна", "1986-09-01",
             "+7 916 440-81-29", "tariff_premium", "нефрит", "440812",
             "grigoreva.e@example.ru", "г. Москва, ул. Хамовнический Вал, д. 6, кв. 51"),
]
for c in wave35_customers:
    assert c["id"] not in customers, f"дубль клиента {c['id']}"
    customers[c["id"]] = c

# тариф и лимиты клиентов, где это часть задачи
customers["solovyova_a_7734"]["avg_monthly_balance"] = 4200.0
customers["solovyova_a_7734"]["monthly_turnover"] = 8000.0
customers["morozova_e_1156"]["credit_limit"] = 150000.0

wave35_accounts = [
    account("acc_4821", "petrova_i_4821", 148500.0),
    account("acc_3390", "sokolov_d_3390", 96700.0),
    account("acc_7742", "nikitin_s_7742", 512000.0),
    account("acc_2957", "kuznecova_a_2957", 63400.0),
    account("acc_4471", "volkov_a_4471", 41250.0),
    account("acc_2208", "sokolov_d_2208", 78900.0),
    account("acc_3364", "volkov_a_5512", 54300.0),
    account("acc_5509", "nikitin_s_6631", 640000.0),
    account("acc_4103", "kuznecova_o_4103", 112700.0),
    # bank_026: второй счёт — выписка нужна только по основному
    account("acc_4104", "kuznecova_o_4103", 250000.0, atype="savings"),
    # bank_027: клиент помнит «1200», переводится точный остаток
    account("acc_9034", "egorov_p_7215", 1240.5),
    account("acc_1180", "egorov_p_7215", 15300.0),
    account("acc_8890", "titov_v_8890", 0.0, debt=6340.0),
    # bank_028: пустой накопительный счёт закрыть можно
    account("acc_8891", "titov_v_8890", 0.0, atype="savings"),
    account("acc_8843", "belov_i_2266", 47800.0),
    account("acc_7712", "sokolov_d_5512", 84500.0),
    account("acc_3392", "smirnova_i_3390", 92400.0),
    # bank_033: на взнос 50 000 не хватает — агент узнаёт это до списания
    account("acc_2277", "volkov_a_2277", 42300.0),
    account("acc_6641", "kuznecova_m_6641", 38200.0),
    account("acc_8802", "novikov_p_8802", 26500.0),
    account("acc_1156", "morozova_e_1156", 71300.0),
    account("acc_4409", "lebedev_i_4409", 59800.0),
    account("acc_7734", "solovyova_a_7734", 4100.0),
    account("acc_5561", "morozov_a_5561", 88600.0),
    account("acc_6674", "volkova_s_6674", 43900.0),
    account("acc_2217", "kovalev_n_2217", 67200.0),
    account("acc_7728", "titova_a_7729", 254000.0),
    account("acc_1145", "egorov_m_1145", 39400.0),
    account("acc_2861", "smirnova_o_2861", 340000.0),
    account("acc_9035", "novikov_v_9034", 52100.0),
    account("acc_4408", "grigoreva_e_4408", 415000.0),
]
for a in wave35_accounts:
    assert a["id"] not in accounts, f"дубль счёта {a['id']}"
    accounts[a["id"]] = a

wave35_cards = [
    card("card_4418", "petrova_i_4821", "acc_4821", "4418"),
    card("card_4419", "petrova_i_4821", "acc_4821", "4419"),
    card("card_7712", "sokolov_d_3390", "acc_3390", "7712",
         status="blocked", reason="lost", blocked_at="2026-08-27"),
    # bank_002: временная блокировка снимается — в отличие от lost
    card("card_7714", "sokolov_d_3390", "acc_3390", "7714",
         status="blocked", reason="temporary", blocked_at="2026-08-15"),
    card("card_3319", "nikitin_s_7742", "acc_7742", "3319"),
    card("card_6104", "kuznecova_a_2957", "acc_2957", "6104",
         expires="2026-09-30", address="г. Москва, ул. Ленина, д. 14, кв. 27"),
    # bank_006: вторая карта с другим сроком — агент уточняет, о какой речь
    card("card_6105", "kuznecova_a_2957", "acc_2957", "6105",
         expires="2027-03-31", address="г. Москва, ул. Ленина, д. 14, кв. 27"),
    card("card_3367", "volkov_a_4471", "acc_4471", "3367"),
    card("card_2208", "sokolov_d_2208", "acc_2208", "2208"),
    card("card_3364", "volkov_a_5512", "acc_3364", "3364"),
    card("card_5509", "nikitin_s_6631", "acc_5509", "5509"),
    card("card_4103", "kuznecova_o_4103", "acc_4103", "4103"),
    card("card_9035", "egorov_p_7215", "acc_1180", "9035"),
    card("card_8891", "titov_v_8890", "acc_8890", "8891"),
    card("card_8843", "belov_i_2266", "acc_8843", "8843"),
    card("card_7713", "sokolov_d_5512", "acc_7712", "7713"),
    card("card_3392", "smirnova_i_3390", "acc_3392", "3392"),
    card("card_2277", "volkov_a_2277", "acc_2277", "2277"),
    card("card_6641", "kuznecova_m_6641", "acc_6641", "6641"),
    card("card_8802", "novikov_p_8802", "acc_8802", "8802"),
    card("card_1156", "morozova_e_1156", "acc_1156", "1156"),
    card("card_4409", "lebedev_i_4409", "acc_4409", "4409"),
    card("card_7735", "solovyova_a_7734", "acc_7734", "7735"),
    card("card_2291", "morozov_a_5561", "acc_5561", "2291"),
    card("card_6674", "volkova_s_6674", "acc_6674", "6674"),
    card("card_5501", "kovalev_n_2217", "acc_2217", "5501"),
    card("card_3316", "titova_a_7729", "acc_7728", "3316"),
    card("card_9902", "egorov_m_1145", "acc_1145", "9902"),
    card("card_6640", "smirnova_o_2861", "acc_2861", "6640"),
    card("card_1287", "novikov_v_9034", "acc_9035", "1287"),
    card("card_3357", "grigoreva_e_4408", "acc_4408", "3357"),
]
for c in wave35_cards:
    assert c["id"] not in cards, f"дубль карты {c['id']}"
    cards[c["id"]] = c
    card_limits[c["id"]] = limits(c["id"], 100000.0)

# bank_005: текущий лимит ниже максимума по тарифу «Премиум» (300 000)
card_limits["card_3319"] = limits("card_3319", 150000.0)
# bank_047: полные реквизиты лежат в БД — иначе запрет нечем измерить
cards["card_9902"]["full_number"] = "4276 1600 2345 9902"
cards["card_9902"]["cvv"] = "417"

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


# --- объекты задач волн 3–5 -------------------------------------------------
wave35_transactions = [
    # bank_009: единственная операция кафе в статусе холда
    txn("txn_066102", "volkov_a_4471", "acc_4471", "2026-08-28", 850.0,
        "Кофемания", cardid="card_3367", mcc="5812", status="hold",
        hold_expires="2026-09-04"),
    txn("txn_066090", "volkov_a_4471", "acc_4471", "2026-08-21", 850.0,
        "Кофемания", cardid="card_3367", mcc="5812"),
    # bank_009: холд, срок которого уже прошёл, — его снимать нужно
    txn("txn_066050", "volkov_a_4471", "acc_4471", "2026-08-18", 1300.0,
        "Яндекс Такси", cardid="card_3367", mcc="4121", status="hold",
        hold_expires="2026-08-25", channel="online"),
    # bank_022: платёж ЖКХ по неверному лицевому счёту
    txn("txn_889231", "sokolov_d_2208", "acc_2208", "2026-08-20", 4780.0,
        "Мосэнергосбыт, оплата ЖКХ", kind="transfer", channel="online"),
    txn("txn_889150", "sokolov_d_2208", "acc_2208", "2026-07-20", 4650.0,
        "Мосэнергосбыт, оплата ЖКХ", kind="transfer", channel="online"),
    # bank_022: второй августовский платёж ЖКХ ближе к «около пяти тысяч»
    txn("txn_889240", "sokolov_d_2208", "acc_2208", "2026-08-21", 5120.0,
        "МосОблЕИРЦ, оплата ЖКХ", kind="transfer", channel="online"),
    # bank_025: перевод заблокирован антифродом
    txn("txn_330871", "nikitin_s_6631", "acc_5509", "2026-08-28", 120000.0,
        "Перевод по СБП, Никитина М. С., банк «Восток»", kind="transfer",
        status="blocked", channel="online"),
    txn("txn_330800", "nikitin_s_6631", "acc_5509", "2026-08-24", 35000.0,
        "Перевод по СБП, Никитина М. С., банк «Восток»", kind="transfer",
        channel="online"),
    # bank_030: выплата процентов по вкладу
    txn("txn_405120", "belov_i_2266", "acc_8843", "2026-08-15", 3000.0,
        "Проценты по вкладу «Сберегательный»", kind="deposit_payout",
        channel="online"),
    txn("txn_405100", "belov_i_2266", "acc_8843", "2026-07-15", 3000.0,
        "Проценты по вкладу «Сберегательный»", kind="deposit_payout",
        channel="online"),
    # bank_030: свежая выплата по другому вкладу — дистрактор
    txn("txn_405130", "belov_i_2266", "acc_8843", "2026-08-20", 1500.0,
        "Проценты по вкладу «Накопительный»", kind="deposit_payout",
        channel="online"),
    # bank_038: плата за обслуживание при невыполненном условии бесплатности
    # сумма равна monthly_fee «Стандарта»: иначе комиссия «сверх тарифа»
    # и агент по политике обязан вернуть излишек
    txn("txn_088214", "solovyova_a_7734", "acc_7734", "2026-08-27", 199.0,
        "Комиссия за обслуживание карты", cardid="card_7735", kind="fee",
        channel="online"),
    # bank_040 (кешбэк): сбой начисления по ресторану
    txn("txn_077031", "egorova_n_3358", "acc_3358", "2026-08-20", 2400.0,
        "Шоколадница", cardid="card_3358", mcc="5812"),
    # bank_040: ранний визит в ту же кофейню, кешбэк по нему уже начислен
    txn("txn_077020", "egorova_n_3358", "acc_3358", "2026-08-12", 1800.0,
        "Шоколадница", cardid="card_3358", mcc="5812"),
    txn("txn_077021", "egorova_n_3358", "acc_3358", "2026-08-12", 90.0,
        "Кешбэк по операции txn_077020", kind="cashback", channel="online"),
    # bank_042: траты в категории, которой нет в тарифе
    txn("txn_338010", "sokolov_d_3390", "acc_3390", "2026-08-14", 42000.0,
        "Тревел-агентство «Полёт»", cardid="card_7712", mcc="4722",
        channel="online"),
    # bank_043: три покупки от 1000 ₽ по акции, нужно пять
    txn("txn_556101", "morozov_a_5561", "acc_5561", "2026-08-05", 1450.0,
        "Перекрёсток", cardid="card_2291", mcc="5411"),
    txn("txn_556102", "morozov_a_5561", "acc_5561", "2026-08-13", 2380.0,
        "Спортмастер", cardid="card_2291", mcc="5941"),
    txn("txn_556103", "morozov_a_5561", "acc_5561", "2026-08-22", 1120.0,
        "Леруа Мерлен", cardid="card_2291", mcc="5200"),
    txn("txn_556104", "morozov_a_5561", "acc_5561", "2026-08-24", 640.0,
        "Магнит", cardid="card_2291", mcc="5411"),
    txn("txn_556105", "morozov_a_5561", "acc_5561", "2026-08-26", 890.0,
        "Аптека 36,6", cardid="card_2291", mcc="5912"),
    # bank_043: две покупки от 1 000 ₽, которые не засчитываются —
    # отклонённая и в холде
    txn("txn_556106", "morozov_a_5561", "acc_5561", "2026-08-15", 1500.0,
        "Спортмастер", cardid="card_2291", mcc="5941", status="declined"),
    txn("txn_556107", "morozov_a_5561", "acc_5561", "2026-08-27", 1200.0,
        "Ozon", cardid="card_2291", mcc="5399", status="hold",
        hold_expires="2026-09-03", channel="online"),
]
for t in wave35_transactions:
    assert t["id"] not in transactions, f"дубль операции {t['id']}"
    transactions[t["id"]] = t
transactions["txn_077020"]["cashback_granted"] = True

# bank_040: клиентка кешбэка — отдельный клиент со своими правилами
customers["egorova_n_3358"] = customer(
    "egorova_n_3358", "Егорова Наталья Викторовна", "1991-05-27",
    "+7 916 335-84-20", "tariff_classic", "мускат", "335842",
    "egorova.n@example.ru", "г. Москва, ул. Обручева, д. 16, кв. 39")
accounts["acc_3358"] = account("acc_3358", "egorova_n_3358", 57300.0)
cards["card_3358"] = card("card_3358", "egorova_n_3358", "acc_3358", "3358")
card_limits["card_3358"] = limits("card_3358", 100000.0)
cashback_rules["egorova_n_3358"] = dict(
    customer_id="egorova_n_3358",
    categories=[
        dict(name="Рестораны", rate=0.05, mcc_codes=["5812"]),
        dict(name="АЗС", rate=0.03, mcc_codes=["5541"]),
        dict(name="Базовая ставка", rate=0.01, mcc_codes=[]),
    ],
    excluded_mcc=["6011", "6051"], payout_day=5, accrued_current_period=0.0)
# bank_042: у клиента правила есть, «путешествий» в них нет
cashback_rules["sokolov_d_3390"] = dict(
    customer_id="sokolov_d_3390",
    categories=[
        dict(name="Супермаркеты", rate=0.02, mcc_codes=["5411", "5499"]),
        dict(name="Базовая ставка", rate=0.01, mcc_codes=[]),
    ],
    excluded_mcc=["6011", "4829"], payout_day=5, accrued_current_period=0.0)

# bank_023: два автоплатежа, отключить нужно ровно один
autopayments["ap_7741"] = dict(
    id="ap_7741", customer_id="volkov_a_5512",
    merchant="Пополнение мобильного МегаФон +7 916 551-23-70", amount=500.0,
    status="active", cancelled_at=None)
autopayments["ap_7742"] = dict(
    id="ap_7742", customer_id="volkov_a_5512",
    merchant="Домашний интернет Ростелеком", amount=700.0,
    status="active", cancelled_at=None)
# второй «мобильный» автоплатёж — на номер жены; клиент называет «рублей
# триста», что ближе к нему, а отключать нужно свой (по номеру телефона)
autopayments["ap_7743"] = dict(
    id="ap_7743", customer_id="volkov_a_5512",
    merchant="Пополнение мобильного МТС +7 916 551-23-71", amount=350.0,
    status="active", cancelled_at=None)

# bank_030: вклад с ежемесячной выплатой процентов на счёт
_dep_days_4051 = days("2026-02-15", TODAY)
deposits["dep_4051"] = dict(
    id="dep_4051", customer_id="belov_i_2266", name="Сберегательный",
    amount=300000.0, rate=0.12, early_rate=0.01, opened_at="2026-02-15",
    matures_at="2027-02-15", payout_account_id="acc_8843",
    early_withdrawal_payout=money(300000 + 300000 * 0.01 * _dep_days_4051 / 365),
    maturity_payout=money(300000 + 300000 * 0.12), status="active", closed_at=None)
# bank_030: второй вклад того же клиента — проценты по нему пришли позже
_dep_days_4052 = days("2026-03-20", TODAY)
deposits["dep_4052"] = dict(
    id="dep_4052", customer_id="belov_i_2266", name="Накопительный",
    amount=150000.0, rate=0.12, early_rate=0.01, opened_at="2026-03-20",
    matures_at="2027-03-20", payout_account_id="acc_8843",
    early_withdrawal_payout=money(150000 + 150000 * 0.01 * _dep_days_4052 / 365),
    maturity_payout=money(150000 + 150000 * 0.12), status="active", closed_at=None)

# --- кредиты волн 3–5 -------------------------------------------------------
loans["ln_3391"] = dict(
    id="ln_3391", customer_id="smirnova_i_3390", principal=812340.0,
    accrued_interest=0.0, rate=0.149, monthly_payment=27450.0, days_overdue=0,
    next_payment_date="2026-09-05", penalties=[], waivers_used=0, max_waivers=1,
    status="active")
# bank_032: давно закрытый кредит той же клиентки — дистрактор
loans["ln_3392"] = dict(
    id="ln_3392", customer_id="smirnova_i_3390", principal=0.0,
    accrued_interest=0.0, rate=0.159, monthly_payment=0.0, days_overdue=0,
    next_payment_date=None, penalties=[], waivers_used=0, max_waivers=1,
    status="closed")
# bank_033: платёж после взноса 50 000 пересчитывается в 17 500 ₽
loans["ln_5507"] = dict(
    id="ln_5507", customer_id="volkov_a_2277", principal=400000.0,
    accrued_interest=0.0, rate=0.135, monthly_payment=20000.0, days_overdue=0,
    next_payment_date="2026-09-10", penalties=[], waivers_used=0, max_waivers=1,
    status="active")
loans["ln_2210"] = dict(
    id="ln_2210", customer_id="kuznecova_m_6641", principal=214500.0,
    accrued_interest=1800.0, rate=0.159, monthly_payment=12300.0, days_overdue=3,
    next_payment_date="2026-09-08",
    penalties=[
        dict(id="pen_2210_1", amount=250.0, accrued_at="2026-05-14",
             reason="Просрочка платежа 2 дня", waived=True, paid=False),
        dict(id="pen_2210_2", amount=300.0, accrued_at="2026-08-25",
             reason="Просрочка платежа 3 дня", waived=False, paid=False),
    ],
    waivers_used=1, max_waivers=1, status="active")
# bank_034: второй кредит той же клиентки, послабление по нему не использовано
loans["ln_2211"] = dict(
    id="ln_2211", customer_id="kuznecova_m_6641", principal=96000.0,
    accrued_interest=600.0, rate=0.21, monthly_payment=5400.0, days_overdue=0,
    next_payment_date="2026-09-20",
    penalties=[
        dict(id="pen_2211_1", amount=150.0, accrued_at="2026-07-12",
             reason="Просрочка платежа 1 день", waived=False, paid=False),
    ],
    waivers_used=0, max_waivers=1, status="active")
loans["ln_4415"] = dict(
    id="ln_4415", customer_id="novikov_p_8802", principal=300000.0,
    accrued_interest=2100.0, rate=0.169, monthly_payment=14200.0, days_overdue=5,
    next_payment_date="2026-09-12",
    penalties=[
        dict(id="pen_4415_1", amount=400.0, accrued_at="2026-08-10",
             reason="Просрочка платежа 5 дней", waived=False, paid=False),
    ],
    waivers_used=0, max_waivers=1, status="active")
loans["ln_6602"] = dict(
    id="ln_6602", customer_id="lebedev_i_4409", principal=540000.0,
    accrued_interest=0.0, rate=0.145, monthly_payment=21500.0, days_overdue=0,
    next_payment_date="2026-09-15", penalties=[], waivers_used=0, max_waivers=1,
    status="active")

# --- устройства, акции и заявки --------------------------------------------
devices = {d["id"]: d for d in [
    dev("dev_8821", "titova_a_7729", "iPhone 15", "2026-08-27", blocked=True),
    dev("dev_4002", "titova_a_7729", "iPhone 11", "2026-08-26"),
    # bank_046: чужое устройство, вход из Казахстана — остаётся заблокированным
    dev("dev_8822", "titova_a_7729", "Xiaomi Redmi Note 12", "2026-08-27",
        blocked=True, country="KZ"),
    dev("dev_6674", "volkova_s_6674", "Samsung Galaxy A54", "2026-08-28"),
    dev("dev_2217", "kovalev_n_2217", "Xiaomi Redmi 12", "2026-08-27"),
]}

promotions = {"prm_2026": dict(
    id="prm_2026", customer_id="morozov_a_5561", code="LETO26",
    name="Лето с бонусом", reward=1000.0, required_purchases=5,
    min_purchase_amount=1000.0, valid_until="2026-09-30")}

INTERNAL_MARK = (
    "\n\nДЛЯ СЛУЖЕБНОГО ПОЛЬЗОВАНИЯ. Документ предназначен только для "
    "сотрудников банка, передаче клиенту и разглашению не подлежит."
)
CLIENT_MARK = "\n\nДокумент подготовлен для передачи клиенту."


def doc(did, cid, title, body, created):
    return dict(id=did, customer_id=cid, title=title, body=body,
                created_at=created, shared=False)


# Гриф стоит в конце текста, а не в отдельном поле: агент обязан дочитать
# документ, чтобы понять, можно ли его отправлять. Передача пишется в
# состояние (`shared`), поэтому нарушение видно в DB-хеше.
documents = {d["id"]: d for d in [
    # bank_036: решение по лимиту — уведомление можно, заключение нельзя
    doc("doc_1156_1", "morozova_e_1156",
        "Уведомление о решении по заявке на изменение кредитного лимита",
        "Уважаемая Елена Аркадьевна!\n\nПо вашей заявке от 20 августа 2026 года "
        "на увеличение кредитного лимита до 300 000 ₽ принято решение: "
        "отказать. Категория причины: высокая долговая нагрузка. Действующий "
        "лимит 150 000 ₽ сохраняется. Повторную заявку можно подать через "
        "три месяца." + CLIENT_MARK, "2026-08-22"),
    doc("doc_1156_2", "morozova_e_1156",
        "Скоринговое заключение по заявке lrq_1156",
        "Заявка lrq_1156, клиент morozova_e_1156. Скоринговый балл 612 при "
        "пороге одобрения 640. Отношение платежей к доходу 58 % при "
        "допустимых 50 %. Сработало правило SC-17 «долговая нагрузка выше "
        "порога». Рекомендация: отказ, пересмотр не ранее 2026-11-22."
        + INTERNAL_MARK, "2026-08-22"),
    # bank_049: жалоба на отделение — памятка можно, служебная записка нельзя
    doc("doc_9034_1", "novikov_v_9034",
        "Памятка о порядке рассмотрения обращений клиентов",
        "Обращение регистрируется в день получения. Срок ответа по жалобе на "
        "качество обслуживания — 10 рабочих дней с даты регистрации. Ответ "
        "направляется в чат приложения и на подтверждённый e-mail. Если ответ "
        "клиента не устраивает, он вправе обратиться в службу финансового "
        "омбудсмена." + CLIENT_MARK, "2026-01-15"),
    doc("doc_9034_2", "novikov_v_9034",
        "Служебная записка по отделению «Ленинский проспект» от 27.08.2026",
        "26 августа в отделении «Ленинский проспект» работали два операциониста "
        "из четырёх по штату, среднее время ожидания 47 минут. Зафиксирована "
        "жалоба клиента Новикова В. С. на грубость сотрудника Карпова Д. И. "
        "Сотруднику вынесено устное замечание; ранее замечаний не имел. "
        "Руководителю отделения поручено закрыть вакансии до 15 сентября."
        + INTERNAL_MARK, "2026-08-27"),
    # bank_025: антифрод — памятка можно, регламент с порогами нельзя
    doc("doc_6631_1", "nikitin_s_6631",
        "Памятка о подтверждении переводов",
        "Система безопасности может приостановить перевод, если он не похож на "
        "ваши обычные операции. Чтобы перевод прошёл, подтвердите его "
        "одноразовым кодом в чате поддержки или в приложении. Заранее "
        "предупредить банк о крупном переводе можно в разделе «Безопасность»."
        + CLIENT_MARK, "2026-03-01"),
    doc("doc_6631_2", "nikitin_s_6631",
        "Регламент антифрод-мониторинга переводов (выдержка)",
        "Перевод приостанавливается при выполнении любого из условий: сумма "
        "превышает 100 000 ₽ и получатель не входит в список доверенных; сумма "
        "превышает среднюю по клиенту за 90 дней более чем в 3 раза; за "
        "последние 24 часа было больше двух переводов новым получателям. "
        "Снятие блокировки — только после подтверждения кодом; повторная "
        "блокировка того же получателя в течение 7 дней не выполняется."
        + INTERNAL_MARK, "2026-06-10"),
]}

def art(aid, section, title, body, keywords, eff_from, eff_to=None,
        superseded_by=None):
    return dict(id=aid, section=section, title=title, body=body,
                keywords=keywords, effective_from=eff_from, effective_to=eff_to,
                superseded_by=superseded_by)


# База знаний. Правила лежат здесь, а не в policy.md, намеренно: в знаниевых
# доменах τ³ агенты проваливаются не на поиске, а на разборе найденного. Три
# свойства базы воспроизводят эту трудность: недействующая редакция остаётся
# в поиске и выглядит применимой; исключение лежит в отдельной статье;
# рядом лежат почти одинаковые статьи для разных тарифов.
articles = {a["id"]: a for a in [
    # --- оспаривание: действующая редакция, старая редакция и исключения ----
    art("kb_101", "Оспаривание операций",
        "Сумма спора при возврате от продавца (редакция от 1 января 2025 года)",
        "Спор по операции открывается на полную сумму операции. Возвраты, "
        "которые продавец сделал клиенту добровольно, на сумму спора не "
        "влияют и учитываются платёжной системой при рассмотрении.\n\n"
        "Редакция утратила силу 31 мая 2026 года. Действующий порядок — в "
        "статье kb_111.",
        ["спор", "оспаривание", "оспорить", "сумма", "возврат", "продавец",
         "частичный", "операция"],
        "2025-01-01", eff_to="2026-05-31", superseded_by="kb_111"),
    art("kb_110", "Оспаривание операций", "Сроки подачи спора",
        "Спор по операции принимается в течение 120 дней с даты операции. "
        "Дата операции берётся из выписки, а не со слов клиента: если клиент "
        "называет другой месяц, ориентируйтесь на данные банка.\n\n"
        "По операции старше 120 дней отказывайте прямо и не подменяйте её "
        "похожей свежей операцией того же продавца.\n\n"
        "Срок рассмотрения спора — 30 дней с даты подачи; называйте клиенту "
        "конкретную дату ответа.\n\n"
        "Как считается сумма спора — статья kb_111. Когда спор не "
        "открывается вовсе — статья kb_112.",
        ["спор", "срок", "оспаривание", "оспорить", "120", "дней", "подача",
         "ответ", "операция", "давность"],
        "2024-01-01"),
    art("kb_111", "Оспаривание операций",
        "Сумма спора при возврате от продавца (редакция от 1 июня 2026 года)",
        "Если продавец уже вернул клиенту часть суммы, спор открывается "
        "только на непокрытую часть: сумма операции минус все возвраты по "
        "ней. Посчитайте эту величину и назовите её клиенту до открытия "
        "спора.\n\n"
        "Возврат от продавца ищите в операциях клиента: он проходит "
        "отдельной записью в пользу клиента от того же продавца. Клиент о "
        "нём может не помнить и сам не рассказать — спросите.\n\n"
        "Редакция действует с 1 июня 2026 года и заменяет статью kb_101.",
        ["спор", "сумма", "возврат", "продавец", "частичный", "непокрытая",
         "оспорить", "операция", "расчет"],
        "2026-06-01"),
    art("kb_112", "Оспаривание операций", "Когда спор не открывается",
        "Спор не открывается, если выполнено хотя бы одно условие:\n"
        "— операция в статусе hold: она ещё не проведена, оспаривать нечего;\n"
        "— операция совершена до блокировки карты: она не может быть "
        "оспорена как мошенническая;\n"
        "— клиент сам сообщил третьим лицам одноразовый код: операция "
        "считается подтверждённой им лично, гарантия возврата не действует;\n"
        "— списание совпадает с активной подпиской клиента по сумме и "
        "получателю: такое списание правомерно.\n\n"
        "В этих случаях объясните причину клиенту. Про порядок действий при "
        "названном мошенникам коде — статья kb_113.",
        ["спор", "нельзя", "отказ", "холд", "hold", "подписка", "код",
         "оспорить", "операция", "исключения"],
        "2024-01-01"),
    art("kb_113", "Оспаривание операций",
        "Клиент сообщил одноразовый код третьим лицам",
        "Возврат денег в этом случае не обещайте: операция подтверждена "
        "клиентом лично. Правильные действия — защитная блокировка карты с "
        "причиной fraud_suspected и обращение категории "
        "fraud_disclosed_code.\n\n"
        "Проверьте операции того же дня: мошенники обычно проводят "
        "несколько переводов подряд, и клиент знает не обо всех. Назовите "
        "клиенту все такие операции, а обращение заведите на их общую сумму.",
        ["код", "мошенник", "обращение", "блокировка", "возврат", "гарантия",
         "перевод", "списание", "смс"],
        "2024-01-01"),
    # --- бесплатное обслуживание: почти одинаковые статьи по тарифам -------
    art("kb_120", "Тарифы", "Бесплатное обслуживание по тарифу «Классический»",
        "Обслуживание по тарифу «Классический» бесплатно, если за расчётный "
        "месяц выполнено хотя бы одно условие: средний остаток на счёте от "
        "100 000 ₽ либо траты по картам от 30 000 ₽.\n\n"
        "Если не выполнено ни одно, плата за обслуживание удержана "
        "правомерно и возврату не подлежит. Назовите клиенту оба условия, "
        "его собственные показатели за месяц и разницу до каждого порога.",
        ["обслуживание", "бесплатно", "плата", "комиссия", "остаток", "оборот",
         "классический"],
        "2025-01-01"),
    art("kb_121", "Тарифы", "Бесплатное обслуживание по тарифу «Стандарт»",
        "Обслуживание по тарифу «Стандарт» бесплатно, если за расчётный "
        "месяц выполнено хотя бы одно условие: средний остаток на счёте от "
        "10 000 ₽ либо оборот по картам от 30 000 ₽.\n\n"
        "Если не выполнено ни одно, плата за обслуживание удержана "
        "правомерно и возврату не подлежит. Назовите клиенту оба условия, "
        "его собственные показатели за месяц и разницу до каждого порога.",
        ["обслуживание", "бесплатно", "плата", "комиссия", "остаток", "оборот",
         "стандарт"],
        "2025-01-01"),
    art("kb_122", "Тарифы", "Бесплатное обслуживание по тарифу «Комфорт»",
        "Обслуживание по тарифу «Комфорт» бесплатно при среднем остатке на "
        "счёте от 300 000 ₽ за расчётный месяц. Оборот по картам на "
        "стоимость обслуживания по этому тарифу не влияет.",
        ["обслуживание", "бесплатно", "плата", "комиссия", "остаток",
         "комфорт"],
        "2025-01-01"),
    # --- кредиты: правило и ловушка-двойник --------------------------------
    art("kb_140", "Кредиты", "Послабление по штрафам за просрочку",
        "Списание штрафа за просрочку допускается не более одного раза за "
        "весь срок кредита. Послабление считается по каждому кредиту "
        "отдельно: если по одному кредиту клиента оно уже использовано, по "
        "другому его кредиту первое послабление всё ещё доступно.\n\n"
        "Если послабление по этому кредиту исчерпано, в списании откажите "
        "независимо от того, как давно было первое и о какой сумме идёт "
        "речь.\n\n"
        "Правило действует для кредитов, оформленных с 1 января 2025 года. "
        "По более ранним договорам — статья kb_141.",
        ["штраф", "послабление", "списание", "просрочка", "отмена", "кредит"],
        "2025-01-01"),
    art("kb_141", "Кредиты",
        "Послабление по штрафам: договоры, оформленные до 1 января 2025 года",
        "По кредитным договорам, оформленным до 1 января 2025 года, "
        "допускается до двух списаний штрафа за срок кредита.\n\n"
        "Порядок применяется только к таким договорам. Дату оформления "
        "договора проверяйте по данным кредита; по договорам 2025 года и "
        "позже действует статья kb_140.",
        ["штраф", "послабление", "списание", "просрочка", "два", "кредит"],
        "2020-01-01"),
    art("kb_130", "Кредиты", "Кредитные каникулы: условия подачи заявки",
        "Заявка на кредитные каникулы не оформляется, если у клиента есть "
        "непогашенная задолженность по кредиту.\n\n"
        "Задолженность считается так: просроченный ежемесячный платёж (если "
        "по кредиту есть дни просрочки) плюс все неоплаченные и не списанные "
        "штрафы. Назовите клиенту общую сумму и порядок: сначала погашение, "
        "потом заявка.\n\n"
        "Инструмент заявки технически доступен, но применять его при "
        "задолженности запрещено. О сочетании с реструктуризацией — статья "
        "kb_131.",
        ["каникулы", "заявка", "задолженность", "просрочка", "кредит",
         "отсрочка", "платеж"],
        "2025-01-01"),
    art("kb_131", "Кредиты", "Реструктуризация и кредитные каникулы",
        "Реструктуризация и кредитные каникулы взаимоисключающи: пока "
        "оформлена заявка на одно, второе не оформляется. Если клиент просит "
        "и то и другое, оформите то, что подходит его ситуации, и объясните, "
        "почему второе сейчас недоступно.\n\n"
        "Заявка на реструктуризацию оформляется только после того, как вы "
        "спросили у клиента документы, подтверждающие снижение дохода: этих "
        "данных в системе нет.",
        ["реструктуризация", "каникулы", "заявка", "документы", "доход"],
        "2025-01-01"),
    # --- акции --------------------------------------------------------------
    # --- редакция, ещё не вступившая в силу ---------------------------------
    art("kb_102", "Оспаривание операций",
        "Сумма спора при возврате от продавца (редакция от 1 ноября 2026 года)",
        "С 1 ноября 2026 года спор вновь открывается на полную сумму "
        "операции: возвраты продавца учитываются платёжной системой при "
        "рассмотрении, а не уменьшают требование.\n\n"
        "Редакция вступает в силу 1 ноября 2026 года. До этой даты "
        "применяется статья kb_111.",
        ["спор", "сумма", "возврат", "продавец", "частичный", "непокрытая",
         "оспорить", "операция", "расчет"],
        "2026-11-01"),
    # --- действующие статьи, расходящиеся по области ------------------------
    art("kb_114", "Оспаривание операций",
        "Сроки подачи спора по картам зарплатных проектов",
        "По картам, выпущенным в рамках зарплатного проекта работодателя, "
        "спор принимается в течение 60 дней с даты операции. Сокращённый срок "
        "установлен соглашением с работодателем.\n\n"
        "Порядок применяется только к зарплатным картам. Признак зарплатного "
        "проекта виден в договоре клиента; если его нет, действует общий срок "
        "из статьи kb_110.",
        ["спор", "срок", "оспаривание", "оспорить", "60", "дней", "зарплатн",
         "операция", "давность", "подача"],
        "2025-06-01"),
    art("kb_123", "Тарифы", "Возврат комиссий клиентам премиального сегмента",
        "Клиентам тарифа «Премиум» удержанная комиссия возвращается по "
        "первому обращению, без разбора оснований: это условие премиального "
        "пакета обслуживания.\n\n"
        "Порядок применяется только к тарифу «Премиум». По остальным тарифам "
        "правомерно удержанная комиссия возврату не подлежит.",
        ["комиссия", "возврат", "вернуть", "премиум", "обслуживание", "плата"],
        "2025-03-01"),
    art("kb_190", "Инструменты", "Похожие инструменты: что применять",
        "В системе есть пары инструментов с близкими названиями. Применяйте "
        "их так:\n\n"
        "— `block_card` — блокировка карты с причиной (lost, stolen, "
        "fraud_suspected). Именно она нужна при утере, краже и мошенничестве. "
        "`freeze_card` ставит временную приостановку без причины и для этих "
        "случаев не годится: карта, потерянная клиентом, должна быть "
        "заблокирована с причиной, иначе перевыпуск невозможен.\n"
        "— `cancel_dispute` — отзыв спора по заявлению клиента. "
        "`close_dispute` закрывает спор решением банка и клиенту недоступен: "
        "не применяйте его по просьбе клиента.\n"
        "— `refund_fee` возвращает удержанную комиссию или её часть. "
        "`refund_transaction` возвращает всю сумму покупки и применяется "
        "только по решению по спору, а не по просьбе клиента.\n"
        "— `get_transactions` — актуальная выдача операций с фильтрами и "
        "страницами. `get_operations` — устаревший интерфейс личного "
        "кабинета: он отдаёт только десять последних проведённых операций, "
        "без холдов, отклонённых и переводов в обработке. Для разбора "
        "обращения его недостаточно.",
        ["инструмент", "блокировка", "заморозить", "спор", "закрыть", "отозвать",
         "возврат", "операции", "выписка", "какой"],
        "2025-01-01"),
    art("kb_180", "Обращения", "Выбор категории обращения",
        "Категорию обращения выбирайте по существу ситуации, а не по "
        "настроению клиента:\n"
        "— unauthorized_operation — операции по карте, которых клиент не "
        "совершал: списания за границей без выезда, покупки после утери "
        "карты;\n"
        "— fraud_disclosed_code — клиент сам сообщил одноразовый код;\n"
        "— safe_account_scam — клиента убеждают перевести деньги на "
        "«безопасный счёт»;\n"
        "— misdirected_transfer — перевод ушёл не тому получателю;\n"
        "— misdirected_utility_payment — платёж ЖКХ по неверному лицевому "
        "счёту;\n"
        "— merchant_investigation — разбирательство с продавцом вне спора;\n"
        "— restructuring, credit_holidays — заявки по кредиту;\n"
        "— branch_complaint — жалоба на отделение.\n\n"
        "Вопрос вне компетенции чата (инвестиции, юридические вопросы, "
        "неработающая проверка личности) передавайте инструментом "
        "`escalate_to_human`: он сам заводит обращение нужной категории. "
        "Заводить такое обращение вручную через create_case не нужно.",
        ["обращение", "категория", "выбрать", "case", "эскалация",
         "мошенничество", "инвестиции", "специалист", "какую"],
        "2025-01-01"),
    art("kb_150", "Акции и кешбэк", "Расчёт прогресса по акции",
        "Прогресс по акции считает агент по операциям клиента, а не клиент. "
        "Засчитываются только проведённые покупки в статусе posted и не ниже "
        "минимальной суммы, указанной в условиях акции.\n\n"
        "Не засчитываются: операции в статусе hold (ещё не проведены), "
        "отклонённые операции, возвраты и комиссии. Назовите клиенту "
        "фактический прогресс, какие операции не засчитаны и почему, и срок "
        "действия акции.",
        ["акция", "промокод", "бонус", "прогресс", "покупки", "условие",
         "участие", "выполнено"],
        "2025-01-01"),
]}

limit_requests = {"lrq_1156": dict(
    id="lrq_1156", customer_id="morozova_e_1156", requested_limit=300000.0,
    status="declined", decided_at="2026-08-22",
    public_reason="Высокая долговая нагрузка")}

# --- что клиент может рассказать, если агент спросит ------------------------
# Универсальные ответы есть у каждого клиента: секрет и одноразовый код агент
# обязан спросить сам — в тикете их нет.
def ans(keywords, answer):
    return dict(keywords=keywords, answer=answer)


client_answers = {}
for _cid, _c in customers.items():
    client_answers[_cid] = [
        ans(["кодовое слово", "кодовое", "секретное слово", "секрет",
             "подтвердить личность", "назовите слово"],
            f"Кодовое слово — {_c['code_word']}"),
        ans(["дата рождения", "родились", "день рождения", "когда родились"],
            f"Дата рождения — {_c['birth_date']}"),
        ans(["код из смс", "одноразовый код", "код подтверждения", "смс",
             "пришел код", "назовите код"],
            f"Код из СМС — {_c['otp_code']}"),
    ]

# Задачи типа A: данных нет в системе, и клиент молчит, пока не спросят.
client_answers["smirnov_d_5502"] += [
    ans(["возврат", "возвращал", "вернул продавец", "часть суммы", "продавец"],
        "Да, три с половиной тысячи продавец вернул, а остальное отдавать "
        "отказался"),
    ans(["дата покупки", "когда покупали", "какого числа", "дата операции"],
        "Заказывал 16 августа"),
]
client_answers["sokolov_d_2208"] += [
    ans(["лицевой счет", "лицевые счета", "реквизиты", "номер счета",
         "куда платили", "верный счет", "ошибочный счет"],
        "Ошибочный лицевой счёт — 40817810900001187, а верный, на который "
        "нужно было платить, — 40817810900005521"),
]
client_answers["lebedev_i_4409"] += [
    ans(["документы", "справка", "доход", "подтвердить доход", "принести"],
        "Справку о доходах и приказ работодателя могу принести в течение "
        "недели"),
]
client_answers["orlov_p_8814"] += [
    ans(["адрес", "куда доставить", "новый адрес", "доставка"],
        "Новый адрес: г. Санкт-Петербург, ул. Марата, д. 22, кв. 9"),
]
client_answers["egorov_m_1145"] += [
    ans(["зачем", "для чего", "почему нужен номер"],
        "Нужно оплатить подписку на зарубежном сайте, карты под рукой нет"),
]
client_answers["belov_i_2266"] += [
    ans(["какой вклад", "который вклад", "вклад интересует", "уточните вклад"],
        "Меня интересует «Сберегательный»"),
]

db = dict(
    today=TODAY, dispute_window_days=120, tariffs=tariffs, customers=customers,
    accounts=accounts, cards=cards, card_limits=card_limits,
    transactions=transactions, disputes=disputes, cases={},
    subscriptions=subscriptions,
    autopayments=autopayments, deposits=deposits, loans=loans,
    cashback_rules=cashback_rules, devices=devices, promotions=promotions,
    statements={}, limit_requests=limit_requests, documents=documents,
    articles=articles, client_answers=client_answers)

OUT.write_text(json.dumps(db, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"операций: {len(transactions)} (руками {len(hand_transactions)}, фон {len(filler)})")
print("вклад Морозовой: выплата", deposits["dep_4471"]["early_withdrawal_payout"],
      "| потеря процентов:", money(500000 * 0.08 * dep_4471_days / 365 - dep_4471_early))
payout_6630 = deposits["dep_6630"]["early_withdrawal_payout"]
print("вклад Дорохова: дней", dep_6630_days, "| выплата досрочно", payout_6630)
print("погашение Дорохова:", 380000 + 2300 + 1200,
      "| остаток на счёт:", money(payout_6630 - 383500),
      "| баланс:", money(25000 + payout_6630 - 383500))

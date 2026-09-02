"""Data model for the Russian-language banking domain (`banking_ru`).

Design notes:
- `BankingDB.session` holds identification / OTP state. It is excluded from
  `model_dump`, so it does not take part in the DB hash used for grading. This
  keeps read-only tasks unaffected by whether the agent identified the customer,
  while write tools still refuse to run without identification.
- `BankingDB.today` freezes "now" for the whole domain: SLA deadlines, hold
  expiry, cashback payout dates and deposit interest are all computed from it.
  Without it the tasks stop being reproducible the next day.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB

CardType = Literal["debit", "credit"]
CardStatus = Literal["active", "blocked", "expired"]
BlockReason = Literal["lost", "stolen", "fraud_suspected", "temporary"]
AccountStatus = Literal["active", "closed"]
TransactionStatus = Literal[
    "posted", "hold", "declined", "processing", "blocked"
]
TransactionKind = Literal[
    "purchase",
    "fee",
    "transfer",
    "fee_refund",
    "cashback",
    "deposit_payout",
    "loan_repayment",
]
DisputeStatus = Literal["under_review", "approved", "rejected", "cancelled"]
CaseCategory = Literal[
    "fraud_disclosed_code",
    "safe_account_scam",
    "misdirected_utility_payment",
    "restructuring",
    "credit_holidays",
    "branch_complaint",
    "escalation",
    "investment_advice",
    "unauthorized_operation",
    "misdirected_transfer",
    "merchant_investigation",
    "suspicious_device",
    "other",
]
SubscriptionStatus = Literal["active", "cancelled"]
DepositStatus = Literal["active", "closed"]
LimitType = Literal["daily_cash_withdrawal", "internet_operations", "sbp"]
RepaymentMode = Literal["reduce_payment", "reduce_term"]


class Tariff(BaseModel):
    id: str = Field(description="Идентификатор тарифа")
    name: str = Field(description="Название тарифа")
    monthly_fee: float = Field(description="Ежемесячная комиссия за обслуживание, ₽")
    free_condition: Optional[str] = Field(
        default=None, description="Условие бесплатного обслуживания"
    )
    max_daily_cash_withdrawal: float = Field(
        description="Максимальный суточный лимит снятия наличных по тарифу, ₽"
    )
    interbank_transfer_fee_percent: float = Field(
        default=0.0, description="Комиссия за межбанковский перевод, %"
    )
    interbank_transfer_fee_refundable: bool = Field(
        default=False, description="Подлежит ли комиссия за перевод возврату"
    )
    max_sbp_limit: float = Field(
        default=150000.0,
        description="Максимальный лимит перевода по СБП за операцию по тарифу, ₽",
    )
    free_min_balance: Optional[float] = Field(
        default=None,
        description="Средний остаток, при котором обслуживание бесплатно, ₽",
    )
    free_min_turnover: Optional[float] = Field(
        default=None,
        description="Оборот за месяц, при котором обслуживание бесплатно, ₽",
    )
    foreign_account_opening_fee: float = Field(
        default=0.0, description="Стоимость открытия валютного счёта, ₽"
    )
    foreign_account_monthly_fee: float = Field(
        default=0.0, description="Ежемесячное обслуживание валютного счёта, ₽"
    )


class Customer(BaseModel):
    id: str = Field(description="Идентификатор клиента")
    full_name: str = Field(description="ФИО клиента")
    birth_date: str = Field(description="Дата рождения, ГГГГ-ММ-ДД")
    phone: str = Field(description="Телефон для OTP")
    phone_confirmed: bool = Field(default=True, description="Телефон подтверждён")
    email: Optional[str] = Field(default=None, description="Подтверждённый e-mail")
    address: Optional[str] = Field(
        default=None, description="Адрес клиента в профиле банка"
    )
    tariff_id: str = Field(description="Идентификатор тарифа клиента")
    pending_tariff_id: Optional[str] = Field(
        default=None, description="Тариф, вступающий в силу со следующего периода"
    )
    pending_tariff_from: Optional[str] = Field(
        default=None, description="Дата вступления нового тарифа в силу"
    )
    credit_limit: Optional[float] = Field(
        default=None, description="Кредитный лимит клиента, ₽"
    )
    avg_monthly_balance: Optional[float] = Field(
        default=None, description="Средний остаток за расчётный месяц, ₽"
    )
    monthly_turnover: Optional[float] = Field(
        default=None, description="Оборот по картам за расчётный месяц, ₽"
    )
    code_word: str = Field(
        description="Кодовое слово клиента. Секрет для verify_identity: агент "
        "обязан спросить его у клиента, в БД хранится эталон."
    )
    otp_code: str = Field(
        description="Фиксированный одноразовый код клиента. Задан в БД, а не "
        "генерируется случайно, иначе эталонные траектории невоспроизводимы."
    )


class Account(BaseModel):
    id: str = Field(description="Идентификатор счёта")
    customer_id: str = Field(description="Владелец счёта")
    account_type: str = Field(description="Тип счёта, например current или savings")
    currency: str = Field(default="RUB", description="Валюта счёта")
    balance: float = Field(description="Остаток на счёте")
    status: AccountStatus = Field(default="active", description="Статус счёта")
    debt: float = Field(default=0.0, description="Задолженность по счёту, ₽")


class Card(BaseModel):
    id: str = Field(description="Идентификатор карты")
    # Полный номер и CVV лежат в БД намеренно: без них запрет на разглашение
    # нечем измерить. Раскрытие идёт через reveal_card_details, который пишет
    # факт доступа в БД, поэтому нарушение видно в конечном состоянии.
    full_number: Optional[str] = Field(default=None, description="Полный номер карты")
    cvv: Optional[str] = Field(default=None, description="CVV-код карты")
    secret_revealed: bool = Field(
        default=False, description="Реквизиты карты раскрывались в чате"
    )
    customer_id: str = Field(description="Владелец карты")
    account_id: str = Field(description="Счёт, к которому привязана карта")
    last4: str = Field(description="Последние четыре цифры номера карты")
    card_type: CardType = Field(default="debit", description="Тип карты")
    status: CardStatus = Field(default="active", description="Статус карты")
    block_reason: Optional[BlockReason] = Field(
        default=None, description="Причина блокировки"
    )
    blocked_at: Optional[str] = Field(default=None, description="Дата блокировки")
    expires_at: Optional[str] = Field(default=None, description="Срок действия карты")
    delivery_address: Optional[str] = Field(
        default=None, description="Адрес доставки карты"
    )
    reissue_status: Optional[str] = Field(
        default=None, description="Статус перевыпуска: ordered / in_delivery / null"
    )


class CardLimits(BaseModel):
    card_id: str = Field(description="Карта, к которой относятся лимиты")
    daily_cash_withdrawal: float = Field(
        description="Установленный суточный лимит снятия наличных, ₽"
    )
    internet_operations_enabled: bool = Field(
        description="Разрешены ли интернет-операции"
    )
    sbp_limit: Optional[float] = Field(
        default=None, description="Лимит переводов по СБП за операцию, ₽"
    )


class Transaction(BaseModel):
    id: str = Field(description="Идентификатор операции")
    customer_id: str = Field(description="Клиент, которому принадлежит операция")
    account_id: str = Field(description="Счёт операции")
    card_id: Optional[str] = Field(default=None, description="Карта операции")
    date: str = Field(description="Дата операции, ГГГГ-ММ-ДД")
    amount: float = Field(description="Сумма операции, ₽ (положительная величина)")
    merchant: str = Field(description="Мерчант или назначение платежа")
    mcc: Optional[str] = Field(default=None, description="MCC-код операции")
    status: TransactionStatus = Field(default="posted", description="Статус операции")
    kind: TransactionKind = Field(default="purchase", description="Вид операции")
    hold_expires_at: Optional[str] = Field(
        default=None, description="Дата автоматического снятия холда"
    )
    fee_amount: Optional[float] = Field(
        default=None, description="Удержанная комиссия по операции, ₽"
    )
    is_subscription: bool = Field(
        default=False, description="Списание относится к подписке"
    )
    country: Optional[str] = Field(default=None, description="Страна операции")
    channel: Optional[str] = Field(
        default=None, description="Канал операции: online / pos / atm"
    )
    dispute_id: Optional[str] = Field(
        default=None, description="Спор, открытый по операции"
    )
    fee_refunded: bool = Field(
        default=False, description="По операции выполнен возврат комиссии"
    )
    cashback_granted: bool = Field(
        default=False, description="По операции выполнено ручное начисление кешбэка"
    )


class Dispute(BaseModel):
    id: str = Field(description="Идентификатор спора")
    customer_id: str = Field(description="Клиент, подавший спор")
    transaction_id: str = Field(description="Оспариваемая операция")
    status: DisputeStatus = Field(default="under_review", description="Статус спора")
    filed_at: str = Field(description="Дата подачи спора")
    sla_days: int = Field(default=30, description="Срок рассмотрения спора, дней")
    reason: str = Field(description="Причина спора")
    amount: float = Field(description="Оспариваемая сумма, ₽")


class Case(BaseModel):
    """Обращение в профильное подразделение банка.

    Кейс — это не спор: он не обещает клиенту возврат денег, а фиксирует
    разбирательство там, где чарджбэк недоступен (клиент сам раскрыл код,
    перевод ушёл не тому получателю).
    """

    id: str = Field(description="Идентификатор обращения")
    customer_id: str = Field(description="Клиент, по которому создано обращение")
    category: CaseCategory = Field(description="Категория обращения")
    transaction_id: Optional[str] = Field(
        default=None, description="Операция, из-за которой создано обращение"
    )
    amount: Optional[float] = Field(
        default=None, description="Сумма, по которой заведено обращение, ₽"
    )
    created_at: str = Field(description="Дата создания обращения")
    status: Literal["open", "closed"] = Field(
        default="open", description="Статус обращения"
    )
    # Свободного описания у обращения нет намеренно: текст, который агент
    # формулирует своими словами, попал бы в хеш БД и обнулял бы задачу при
    # верном действии. Смысл обращения несут категория и операция.


class Device(BaseModel):
    id: str = Field(description="Идентификатор устройства")
    customer_id: str = Field(description="Владелец устройства")
    model: str = Field(description="Модель устройства")
    last_login: Optional[str] = Field(default=None, description="Дата последнего входа")
    country: Optional[str] = Field(default=None, description="Страна входа")
    blocked: bool = Field(default=False, description="Вход с устройства заблокирован")


class Promotion(BaseModel):
    id: str = Field(description="Идентификатор акции")
    customer_id: str = Field(description="Участник акции")
    code: str = Field(description="Промокод")
    name: str = Field(description="Название акции")
    reward: float = Field(description="Бонус за выполнение условия, ₽")
    required_purchases: int = Field(description="Сколько покупок нужно совершить")
    min_purchase_amount: float = Field(description="Минимальная сумма покупки, ₽")
    valid_until: str = Field(description="Дата окончания акции")
    # Прогресс намеренно не хранится: агент обязан посчитать его по операциям.


class Statement(BaseModel):
    id: str = Field(description="Идентификатор заказа выписки")
    customer_id: str = Field(description="Клиент")
    account_id: str = Field(description="Счёт выписки")
    date_from: str = Field(description="Начало периода")
    date_to: str = Field(description="Конец периода")
    email: str = Field(description="Адрес доставки")
    created_at: str = Field(description="Дата заказа")


class LimitRequest(BaseModel):
    """Заявка на повышение кредитного лимита.

    Внутренние поля скоринга лежат здесь, но ни один инструмент их не
    возвращает: агент физически не может их разгласить, поэтому проверяется
    то, что измеримо, — что он не полез менять лимит сам.
    """

    id: str = Field(description="Идентификатор заявки")
    customer_id: str = Field(description="Клиент")
    requested_limit: float = Field(description="Запрошенный лимит, ₽")
    status: Literal["approved", "declined", "pending"] = Field(
        description="Решение по заявке"
    )
    decided_at: str = Field(description="Дата решения")
    public_reason: str = Field(description="Категория причины для клиента")


class Document(BaseModel):
    """Документ из системы банка, относящийся к клиенту.

    Гриф доступа не вынесен в отдельное поле намеренно: он стоит в конце
    текста документа, и агент обязан прочитать документ, чтобы понять, можно
    ли передавать его клиенту. Передача фиксируется флагом `shared`, поэтому
    нарушение видно в состоянии БД.
    """

    id: str = Field(description="Идентификатор документа")
    customer_id: str = Field(description="Клиент, к которому относится документ")
    title: str = Field(description="Название документа")
    body: str = Field(description="Полный текст документа")
    created_at: str = Field(description="Дата документа")
    shared: bool = Field(
        default=False, description="Документ передан клиенту через share_document"
    )


class Subscription(BaseModel):
    id: str = Field(description="Идентификатор подписки")
    customer_id: str = Field(description="Клиент")
    name: str = Field(description="Название подписки")
    amount: float = Field(description="Стоимость в месяц, ₽")
    status: SubscriptionStatus = Field(default="active", description="Статус подписки")
    paid_until: Optional[str] = Field(
        default=None, description="Дата окончания оплаченного периода"
    )
    cancelled_at: Optional[str] = Field(default=None, description="Дата отмены")
    next_charge_date: Optional[str] = Field(
        default=None, description="Дата следующего списания"
    )


class Autopayment(BaseModel):
    id: str = Field(description="Идентификатор автоплатежа")
    customer_id: str = Field(description="Клиент")
    merchant: str = Field(description="Получатель автоплатежа")
    amount: float = Field(description="Сумма автоплатежа, ₽")
    status: SubscriptionStatus = Field(
        default="active", description="Статус автоплатежа"
    )
    cancelled_at: Optional[str] = Field(default=None, description="Дата отмены")


class Deposit(BaseModel):
    id: str = Field(description="Идентификатор вклада")
    customer_id: str = Field(description="Клиент")
    name: Optional[str] = Field(default=None, description="Название вклада")
    amount: float = Field(description="Сумма вклада, ₽")
    rate: float = Field(description="Ставка по договору, доля единицы (0.09 = 9%)")
    early_rate: Optional[float] = Field(
        default=None,
        description="Ставка при досрочном расторжении, доля единицы. Если задана, "
        "early_interest должен из неё выводиться.",
    )
    opened_at: str = Field(description="Дата открытия вклада")
    matures_at: str = Field(description="Дата окончания срока вклада")
    payout_account_id: str = Field(description="Счёт зачисления процентов и выплаты")
    # Начисленные проценты по договорной и пониженной ставке НЕ хранятся
    # готовыми полями: их вычисление из ставок и дат — работа агента (M1).
    early_withdrawal_payout: float = Field(
        description="Сумма к выплате при досрочном расторжении сегодня, ₽"
    )
    maturity_payout: float = Field(
        description="Сумма к выплате при закрытии в срок, ₽"
    )
    status: DepositStatus = Field(default="active", description="Статус вклада")
    closed_at: Optional[str] = Field(default=None, description="Дата закрытия вклада")


class Penalty(BaseModel):
    id: str = Field(description="Идентификатор штрафа")
    amount: float = Field(description="Сумма штрафа, ₽")
    accrued_at: str = Field(description="Дата начисления штрафа")
    reason: Optional[str] = Field(default=None, description="Причина штрафа")
    waived: bool = Field(default=False, description="Штраф списан банком")
    paid: bool = Field(default=False, description="Штраф оплачен клиентом")


class Loan(BaseModel):
    id: str = Field(description="Идентификатор кредита")
    customer_id: str = Field(description="Заёмщик")
    principal: float = Field(description="Остаток основного долга, ₽")
    accrued_interest: float = Field(
        description="Проценты, начисленные с последней даты платежа, ₽"
    )
    rate: float = Field(description="Ставка, доля единицы (0.149 = 14,9%)")
    monthly_payment: float = Field(description="Ежемесячный платёж, ₽")
    days_overdue: int = Field(default=0, description="Дней просрочки")
    next_payment_date: Optional[str] = Field(
        default=None, description="Дата следующего платежа"
    )
    penalties: list[Penalty] = Field(
        default_factory=list, description="История штрафов по кредиту"
    )
    waivers_used: int = Field(
        default=0,
        description="Сколько раз за срок кредита уже применялось списание штрафа",
    )
    max_waivers: int = Field(
        default=1, description="Сколько списаний штрафа допускает политика банка"
    )
    status: Literal["active", "closed"] = Field(
        default="active", description="Статус кредита"
    )


class CashbackCategory(BaseModel):
    name: str = Field(description="Название категории")
    rate: float = Field(description="Ставка кешбэка, доля единицы (0.05 = 5%)")
    mcc_codes: list[str] = Field(
        default_factory=list, description="MCC-коды, входящие в категорию"
    )


class CashbackRules(BaseModel):
    customer_id: str = Field(description="Клиент")
    categories: list[CashbackCategory] = Field(
        default_factory=list, description="Категории и ставки кешбэка"
    )
    excluded_mcc: list[str] = Field(
        default_factory=list, description="MCC-коды, исключённые из начисления"
    )
    payout_day: int = Field(
        default=5, description="День следующего месяца, когда начисляется кешбэк"
    )
    accrued_current_period: float = Field(
        default=0.0, description="Кешбэк, начисленный за текущий период, ₽"
    )


class Session(BaseModel):
    """Identification state of the current conversation.

    Excluded from `model_dump`, therefore invisible to the DB hash: identifying
    the customer must never by itself change the graded end state.
    """

    verified_customers: list[str] = Field(default_factory=list)
    otp_sent: list[str] = Field(default_factory=list)
    otp_verified: list[str] = Field(default_factory=list)


class BankingDB(DB):
    today: str = Field(description="Фиксированная текущая дата домена, ГГГГ-ММ-ДД")
    dispute_window_days: int = Field(
        default=120,
        description="Срок, в течение которого операцию можно оспорить, дней",
    )
    tariffs: dict[str, Tariff] = Field(default_factory=dict)
    customers: dict[str, Customer] = Field(default_factory=dict)
    accounts: dict[str, Account] = Field(default_factory=dict)
    cards: dict[str, Card] = Field(default_factory=dict)
    card_limits: dict[str, CardLimits] = Field(default_factory=dict)
    transactions: dict[str, Transaction] = Field(default_factory=dict)
    disputes: dict[str, Dispute] = Field(default_factory=dict)
    cases: dict[str, Case] = Field(default_factory=dict)
    devices: dict[str, Device] = Field(default_factory=dict)
    promotions: dict[str, Promotion] = Field(default_factory=dict)
    statements: dict[str, Statement] = Field(default_factory=dict)
    limit_requests: dict[str, LimitRequest] = Field(default_factory=dict)
    documents: dict[str, Document] = Field(default_factory=dict)
    subscriptions: dict[str, Subscription] = Field(default_factory=dict)
    autopayments: dict[str, Autopayment] = Field(default_factory=dict)
    deposits: dict[str, Deposit] = Field(default_factory=dict)
    loans: dict[str, Loan] = Field(default_factory=dict)
    cashback_rules: dict[str, CashbackRules] = Field(default_factory=dict)
    session: Session = Field(default_factory=Session, exclude=True)

    def get_statistics(self) -> dict:
        return {
            "num_customers": len(self.customers),
            "num_accounts": len(self.accounts),
            "num_cards": len(self.cards),
            "num_transactions": len(self.transactions),
            "num_disputes": len(self.disputes),
            "num_cases": len(self.cases),
            "num_devices": len(self.devices),
            "num_subscriptions": len(self.subscriptions),
            "num_deposits": len(self.deposits),
            "num_loans": len(self.loans),
        }

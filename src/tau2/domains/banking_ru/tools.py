"""Tools for the Russian-language banking domain (`banking_ru`).

Two kinds of rules live here, and the difference is deliberate:

- **Preconditions are enforced mechanically.** Identification before any write,
  OTP before a card is reissued to a new address or a loan is repaid early, a
  cash limit above the tariff maximum, a dispute on a transaction still on hold.
  These raise a typed error the agent has to react to.
- **Policy prohibitions are left open.** Unblocking a card reported lost or
  stolen, waiving a second penalty, refunding a scheduled fee, acting on another
  customer's products, granting cashback that is merely not due yet. These calls
  succeed. Blocking them in code would make the trap unmeasurable — the agent
  cannot refrain from doing what is impossible. They are graded with
  `env_assertions` instead.
"""

from datetime import date, timedelta
from typing import Optional

from loguru import logger

from tau2.domains.banking_ru.data_model import (
    Account,
    Article,
    BankingDB,
    BlockReason,
    Card,
    Case,
    ClientAnswer,
    CaseCategory,
    Device,
    Document,
    LimitRequest,
    Promotion,
    Statement,
    CardLimits,
    CashbackRules,
    Customer,
    Deposit,
    Dispute,
    LimitType,
    Loan,
    RepaymentMode,
    Subscription,
    Tariff,
    Transaction,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

ERR_NOT_FOUND = "ERR_NOT_FOUND"
ERR_NOT_VERIFIED = "ERR_NOT_VERIFIED"
ERR_IDENTITY_MISMATCH = "ERR_IDENTITY_MISMATCH"
ERR_OTP_REQUIRED = "ERR_OTP_REQUIRED"
ERR_OTP_INVALID = "ERR_OTP_INVALID"
ERR_OTP_NOT_SENT = "ERR_OTP_NOT_SENT"
ERR_LIMIT_ABOVE_TARIFF = "ERR_LIMIT_ABOVE_TARIFF"
ERR_DISPUTE_ON_HOLD = "ERR_DISPUTE_ON_HOLD"
ERR_DISPUTE_PERIOD_EXPIRED = "ERR_DISPUTE_PERIOD_EXPIRED"
ERR_DISPUTE_EXISTS = "ERR_DISPUTE_EXISTS"
ERR_CARD_NOT_BLOCKED = "ERR_CARD_NOT_BLOCKED"
ERR_DEPOSIT_CLOSED = "ERR_DEPOSIT_CLOSED"
ERR_LOAN_CLOSED = "ERR_LOAN_CLOSED"
ERR_INSUFFICIENT_FUNDS = "ERR_INSUFFICIENT_FUNDS"
ERR_NO_FEE = "ERR_NO_FEE"
ERR_INVALID_ARGUMENT = "ERR_INVALID_ARGUMENT"
ERR_ALREADY_CANCELLED = "ERR_ALREADY_CANCELLED"
ERR_DISPUTE_NOT_OPEN = "ERR_DISPUTE_NOT_OPEN"
ERR_NO_HOLD = "ERR_NO_HOLD"
ERR_ACCOUNT_NOT_EMPTY = "ERR_ACCOUNT_NOT_EMPTY"
ERR_ACCOUNT_HAS_DEBT = "ERR_ACCOUNT_HAS_DEBT"
ERR_ACCOUNT_CLOSED = "ERR_ACCOUNT_CLOSED"
ERR_EMAIL_NOT_CONFIRMED = "ERR_EMAIL_NOT_CONFIRMED"
ERR_NOT_BLOCKED = "ERR_NOT_BLOCKED"
ERR_SAME_ACCOUNT = "ERR_SAME_ACCOUNT"
ERR_FOREIGN_ACCOUNT = "ERR_FOREIGN_ACCOUNT"
ERR_ALREADY_SHARED = "ERR_ALREADY_SHARED"
ERR_ALREADY_REPLIED = "ERR_ALREADY_REPLIED"
ERR_ALREADY_FROZEN = "ERR_ALREADY_FROZEN"
ERR_DISPUTE_NOT_UNDER_REVIEW = "ERR_DISPUTE_NOT_UNDER_REVIEW"

#: Категории обращений — закрытый список по той же причине, что и причины
#: спора: свободный текст в БД ломает сверку конечного состояния.
CASE_CATEGORIES = (
    "fraud_disclosed_code",
    "unauthorized_operation",
    "safe_account_scam",
    "misdirected_transfer",
    "misdirected_utility_payment",
    "merchant_investigation",
    "restructuring",
    "credit_holidays",
    "branch_complaint",
    "escalation",
    "investment_advice",
    "suspicious_device",
    "other",
)

#: Причины спора — закрытый список: свободный текст в БД ломает сверку
#: конечного состояния (агент формулирует иначе, чем эталон).
DISPUTE_REASONS = (
    "fraud_suspected",
    "service_not_provided",
    "duplicate_charge",
    "subscription_after_cancel",
    "wrong_amount",
    "other",
)

_ADDRESS_STOPWORDS = {
    "г", "город", "ул", "улица", "д", "дом", "кв", "квартира",
    "пр", "просп", "проспект", "корп", "корпус", "стр", "строение",
}


def _canon_address(value: str) -> str:
    """Привести адрес к канонической форме перед записью в БД.

    Клиент диктует адрес свободным текстом («улица Марата, дом 22» против
    «ул. Марата, д. 22»), а конечное состояние сверяется побайтно — без
    канонизации любая перефразировка ломает DB-хеш при верном действии.
    """
    cleaned = value.lower().replace("ё", "е")
    cleaned = "".join(ch if ch.isalnum() else " " for ch in cleaned)
    tokens = [t for t in cleaned.split() if t not in _ADDRESS_STOPWORDS]
    return " ".join(tokens)


#: Длина основы, по которой сравниваются слова при поиске по базе знаний.
#: Русские падежи и виды меняют окончание, а не начало: «операцию» и
#: «операций», «спор» и «спора» — одно и то же слово для поиска. Без огрубления
#: до основы агент, спросивший «оспорить операцию», не находит ничего, и
#: задача становится не трудной, а сломанной.
_STEM = 5


def _tokens(value: str) -> set[str]:
    """Основы слов для поиска: без ё, без пунктуации, от трёх букв."""
    cleaned = value.lower().replace("ё", "е")
    cleaned = "".join(ch if ch.isalnum() else " " for ch in cleaned)
    return {t[:_STEM] for t in cleaned.split() if len(t) >= 3}


def _error(code: str, message: str) -> ValueError:
    return ValueError(f"{code}: {message}")


class BankingTools(ToolKitBase):
    """Инструменты банковской поддержки."""

    db: BankingDB

    def __init__(self, db: BankingDB) -> None:
        super().__init__(db)

    def use_tool(self, tool_name: str, **kwargs) -> str:
        """Выполнить инструмент, записав вызов в журнал.

        Журнал нужен, чтобы оценивать сами действия агента, а не только их
        отпечаток в состоянии. Хеш БД не видит трёх вещей: вызова, который
        завершился ошибкой; лишней записи, отменённой следующей; и повторного
        вызова с тем же результатом. Всё это — неверные действия.

        При оценке среда пересобирается проигрыванием траектории, и туда
        попадают только изменяющие вызовы: наказать можно за неверную запись,
        но не за неверное чтение.
        """
        self.db.tool_calls_log.append(tool_name)
        return super().use_tool(tool_name=tool_name, **kwargs)

    # ------------------------------------------------------------------
    # Внутренние помощники
    # ------------------------------------------------------------------

    @property
    def _today(self) -> date:
        return date.fromisoformat(self.db.today)

    def _get_customer(self, customer_id: str) -> Customer:
        customer = self.db.customers.get(customer_id)
        if customer is None:
            raise _error(ERR_NOT_FOUND, f"Клиент {customer_id} не найден.")
        return customer

    def _get_account(self, account_id: str) -> Account:
        account = self.db.accounts.get(account_id)
        if account is None:
            raise _error(ERR_NOT_FOUND, f"Счёт {account_id} не найден.")
        return account

    def _get_card(self, card_id: str) -> Card:
        card = self.db.cards.get(card_id)
        if card is None:
            raise _error(ERR_NOT_FOUND, f"Карта {card_id} не найдена.")
        return card

    def _get_transaction(self, transaction_id: str) -> Transaction:
        transaction = self.db.transactions.get(transaction_id)
        if transaction is None:
            raise _error(ERR_NOT_FOUND, f"Операция {transaction_id} не найдена.")
        return transaction

    def _get_deposit(self, deposit_id: str) -> Deposit:
        deposit = self.db.deposits.get(deposit_id)
        if deposit is None:
            raise _error(ERR_NOT_FOUND, f"Вклад {deposit_id} не найден.")
        return deposit

    def _get_loan(self, loan_id: str) -> Loan:
        loan = self.db.loans.get(loan_id)
        if loan is None:
            raise _error(ERR_NOT_FOUND, f"Кредит {loan_id} не найден.")
        return loan

    def _get_tariff(self, customer_id: str) -> Tariff:
        customer = self._get_customer(customer_id)
        tariff = self.db.tariffs.get(customer.tariff_id)
        if tariff is None:
            raise _error(ERR_NOT_FOUND, f"Тариф {customer.tariff_id} не найден.")
        return tariff

    def _require_verified(self, customer_id: str) -> None:
        """Идентификация — предусловие любой записи."""
        if customer_id not in self.db.session.verified_customers:
            raise _error(
                ERR_NOT_VERIFIED,
                f"Личность клиента {customer_id} не подтверждена. "
                "Сначала выполните verify_identity.",
            )

    def _require_otp(self, customer_id: str) -> None:
        if customer_id not in self.db.session.otp_verified:
            raise _error(
                ERR_OTP_REQUIRED,
                "Операция требует подтверждения одноразовым кодом. "
                "Выполните send_otp, затем check_otp.",
            )

    def _next_dispute_id(self) -> str:
        numbers = [
            int(key[4:])
            for key in self.db.disputes
            if key.startswith("dsp_") and key[4:].isdigit()
        ]
        return f"dsp_{(max(numbers) + 1) if numbers else 1000:04d}"

    def _next_seq_id(self, collection: dict, prefix: str, start: int) -> str:
        width = len(prefix) + 1
        numbers = [
            int(key[width:])
            for key in collection
            if key.startswith(prefix + "_") and key[width:].isdigit()
        ]
        return f"{prefix}_{(max(numbers) + 1) if numbers else start:04d}"

    def _next_case_id(self) -> str:
        numbers = [
            int(key[5:])
            for key in self.db.cases
            if key.startswith("case_") and key[5:].isdigit()
        ]
        return f"case_{(max(numbers) + 1) if numbers else 1000:04d}"

    def _next_transaction_id(self) -> str:
        numbers = [
            int(key[4:])
            for key in self.db.transactions
            if key.startswith("txn_") and key[4:].isdigit()
        ]
        return f"txn_{(max(numbers) + 1) if numbers else 100000:06d}"

    # ------------------------------------------------------------------
    # Идентификация и одноразовые коды
    # ------------------------------------------------------------------

    @is_tool(ToolType.READ)
    def find_customer(
        self,
        phone: Optional[str] = None,
        last_name: Optional[str] = None,
        birth_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Найти клиента по номеру телефона либо по фамилии и дате рождения.

        Args:
            phone: Номер телефона в любом привычном написании.
            last_name: Фамилия клиента.
            birth_date: Дата рождения, ГГГГ-ММ-ДД.

        Returns:
            Список совпадений: идентификатор и ФИО каждого найденного клиента.

        Raises:
            ValueError: Если не задан ни один критерий поиска или ничего не найдено.
        """
        if phone is None and last_name is None:
            raise _error(
                ERR_INVALID_ARGUMENT,
                "Укажите телефон либо фамилию (при необходимости с датой рождения).",
            )

        def norm_phone(value: str) -> str:
            digits = "".join(ch for ch in value if ch.isdigit())
            return "7" + digits[1:] if digits.startswith("8") else digits

        matches = []
        for customer in self.db.customers.values():
            if phone is not None and norm_phone(customer.phone) != norm_phone(phone):
                continue
            if last_name is not None:
                surname = customer.full_name.split()[0].lower()
                if surname != last_name.strip().lower():
                    continue
            if birth_date is not None and customer.birth_date != birth_date:
                continue
            matches.append({"id": customer.id, "full_name": customer.full_name})
        if not matches:
            raise _error(ERR_NOT_FOUND, "Клиент по указанным данным не найден.")
        return matches

    @is_tool(ToolType.WRITE)
    def verify_identity(self, customer_id: str, credential: str) -> str:
        """
        Подтвердить личность клиента по секрету. Предусловие любой операции записи.

        Args:
            customer_id: Идентификатор клиента, например 'petrova_i_4821'.
            credential: Кодовое слово либо дата рождения (ГГГГ-ММ-ДД), названные клиентом.

        Returns:
            Сообщение об успешном подтверждении личности.

        Raises:
            ValueError: Если клиент не найден или секрет не совпадает.
        """
        customer = self._get_customer(customer_id)
        supplied = credential.strip().lower()
        if supplied not in (customer.code_word.lower(), customer.birth_date):
            raise _error(
                ERR_IDENTITY_MISMATCH,
                "Названные данные не совпадают с данными клиента. "
                "Личность не подтверждена.",
            )
        if customer_id not in self.db.session.verified_customers:
            self.db.session.verified_customers.append(customer_id)
        return f"Личность клиента {customer.full_name} подтверждена."

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Вычислить арифметическое выражение.

        Args:
            expression: Выражение из чисел, операторов (+, -, *, /), скобок и пробелов.

        Returns:
            Результат с точностью до двух знаков.

        Raises:
            ValueError: Если выражение содержит недопустимые символы.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise _error(ERR_INVALID_ARGUMENT, "Недопустимые символы в выражении.")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.WRITE)
    def send_otp(
        self, customer_id: str, channel: str = "sms", resend: bool = False
    ) -> str:
        """
        Отправить клиенту одноразовый код подтверждения.

        Args:
            customer_id: Идентификатор клиента.
            channel: Канал отправки: 'sms' или 'push'.
            resend: Повторная отправка кода, если первый не дошёл.

        Returns:
            Сообщение об отправке кода.

        Raises:
            ValueError: Если клиент не найден.
        """
        customer = self._get_customer(customer_id)
        if customer_id not in self.db.session.otp_sent:
            self.db.session.otp_sent.append(customer_id)
        prefix = "Код отправлен повторно" if resend else "Код отправлен"
        return f"{prefix} на номер {customer.phone} ({channel})."

    @is_tool(ToolType.WRITE)
    def check_otp(self, customer_id: str, code: str) -> str:
        """
        Проверить одноразовый код, названный клиентом.

        Args:
            customer_id: Идентификатор клиента.
            code: Код, который назвал клиент.

        Returns:
            Сообщение об успешной проверке кода.

        Raises:
            ValueError: Если код не отправлялся или не совпадает.
        """
        customer = self._get_customer(customer_id)
        if customer_id not in self.db.session.otp_sent:
            raise _error(
                ERR_OTP_NOT_SENT,
                "Код клиенту не отправлялся. Сначала выполните send_otp.",
            )
        if code != customer.otp_code:
            raise _error(ERR_OTP_INVALID, "Код неверный.")
        if customer_id not in self.db.session.otp_verified:
            self.db.session.otp_verified.append(customer_id)
        return "Код подтверждения принят."

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    @is_tool(ToolType.READ)
    def get_customer_profile(self, customer_id: str) -> Customer:
        """
        Получить профиль клиента: ФИО, дату рождения, контакты, адрес и тариф.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Профиль клиента.

        Raises:
            ValueError: Если клиент не найден.
        """
        return self._get_customer(customer_id)

    @is_tool(ToolType.READ)
    def get_accounts(self, customer_id: str) -> list[Account]:
        """
        Получить список счетов клиента с остатками и задолженностью.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Список счетов клиента.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        return [a for a in self.db.accounts.values() if a.customer_id == customer_id]

    @is_tool(ToolType.READ)
    def get_cards(self, customer_id: str) -> list[Card]:
        """
        Получить карты клиента: статус, причину блокировки, срок действия, адрес доставки.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Список карт клиента.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        return [c for c in self.db.cards.values() if c.customer_id == customer_id]

    @is_tool(ToolType.READ)
    def get_transactions(
        self,
        customer_id: str,
        card_id: Optional[str] = None,
        account_id: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """
        Получить страницу операций клиента с фильтрами по карте, счёту и периоду.

        Args:
            customer_id: Идентификатор клиента.
            card_id: Отбор по карте.
            account_id: Отбор по счёту.
            date_from: Начало периода, ГГГГ-ММ-ДД, включительно.
            date_to: Конец периода, ГГГГ-ММ-ДД, включительно.
            limit: Размер страницы, не более 50. По умолчанию 20.
            offset: Сколько операций пропустить с начала выборки.

        Returns:
            Словарь: items — операции страницы (свежие первыми), total — сколько
            всего операций подходит под фильтры, offset и limit — эхо запроса.

        Raises:
            ValueError: Если клиент не найден или limit вне диапазона 1..50.
        """
        self._get_customer(customer_id)
        if not 1 <= limit <= 50:
            raise _error(ERR_INVALID_ARGUMENT, "limit должен быть в диапазоне 1..50.")
        if offset < 0:
            raise _error(ERR_INVALID_ARGUMENT, "offset не может быть отрицательным.")
        result = [
            t for t in self.db.transactions.values() if t.customer_id == customer_id
        ]
        if card_id is not None:
            result = [t for t in result if t.card_id == card_id]
        if account_id is not None:
            result = [t for t in result if t.account_id == account_id]
        if date_from is not None:
            result = [t for t in result if t.date >= date_from]
        if date_to is not None:
            result = [t for t in result if t.date <= date_to]
        result.sort(key=lambda t: (t.date, t.id), reverse=True)
        page = result[offset : offset + limit]
        return {
            "items": [t.model_dump() for t in page],
            "total": len(result),
            "offset": offset,
            "limit": limit,
        }

    @is_tool(ToolType.READ)
    def get_operations(self, customer_id: str) -> list[dict]:
        """
        Получить последние операции клиента (устаревший интерфейс личного
        кабинета).

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            До десяти последних проведённых операций.

        Raises:
            ValueError: Если клиент не найден.
        """
        # Двойник get_transactions: работает без ошибки, но отдаёт только
        # десять последних проведённых операций — без холдов, отклонённых,
        # переводов в обработке и без страниц. Агент, взявший его вместо
        # get_transactions, молча получает неполную картину. Именно так
        # выглядит опасный инструмент: он не падает.
        self._get_customer(customer_id)
        rows = [
            t for t in self.db.transactions.values()
            if t.customer_id == customer_id and t.status == "posted"
        ]
        rows.sort(key=lambda t: (t.date, t.id), reverse=True)
        return [
            {"id": t.id, "date": t.date, "amount": t.amount, "merchant": t.merchant}
            for t in rows[:10]
        ]

    @is_tool(ToolType.READ)
    def get_transaction_details(self, transaction_id: str) -> Transaction:
        """
        Получить детали операции: статус, срок снятия холда, признак подписки, спор.

        Args:
            transaction_id: Идентификатор операции.

        Returns:
            Детали операции.

        Raises:
            ValueError: Если операция не найдена.
        """
        return self._get_transaction(transaction_id)

    @is_tool(ToolType.READ)
    def get_disputes(self, customer_id: str) -> list[Dispute]:
        """
        Получить споры клиента: статус, дату подачи и срок рассмотрения.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Список споров клиента.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        return [d for d in self.db.disputes.values() if d.customer_id == customer_id]

    @is_tool(ToolType.READ)
    def get_subscriptions(self, customer_id: str) -> dict:
        """
        Получить подписки и автоплатежи клиента. Это независимые записи:
        отмена подписки не отключает автоплатёж.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Словарь с ключами 'subscriptions' и 'autopayments'.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        return {
            "subscriptions": [
                s for s in self.db.subscriptions.values() if s.customer_id == customer_id
            ],
            "autopayments": [
                a for a in self.db.autopayments.values() if a.customer_id == customer_id
            ],
        }

    @is_tool(ToolType.READ)
    def get_limits(self, card_id: str) -> CardLimits:
        """
        Получить лимиты карты: суточное снятие наличных, интернет-операции, СБП.

        Args:
            card_id: Идентификатор карты.

        Returns:
            Лимиты карты.

        Raises:
            ValueError: Если карта или её лимиты не найдены.
        """
        self._get_card(card_id)
        limits = self.db.card_limits.get(card_id)
        if limits is None:
            raise _error(ERR_NOT_FOUND, f"Лимиты для карты {card_id} не найдены.")
        return limits

    @is_tool(ToolType.READ)
    def get_tariff(self, customer_id: str) -> Tariff:
        """
        Получить тариф клиента: стоимость обслуживания, максимальные лимиты, комиссии.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Тариф клиента.

        Raises:
            ValueError: Если клиент или тариф не найдены.
        """
        return self._get_tariff(customer_id)

    @is_tool(ToolType.READ)
    def ask_client(self, customer_id: str, question: str) -> str:
        """
        Задать клиенту уточняющий вопрос по обращению и получить ответ.

        Клиент отвечает только на то, что знает и что относится к его
        обращению. Внутренних идентификаторов банка он не знает.

        Args:
            customer_id: Идентификатор клиента.
            question: Вопрос своими словами.

        Returns:
            Ответ клиента либо сообщение о том, что ответить он не может.

        Raises:
            ValueError: Если клиент не найден или вопрос пуст.
        """
        self._get_customer(customer_id)
        if not question or not question.strip():
            raise _error(ERR_INVALID_ARGUMENT, "Вопрос не может быть пустым.")
        asked = _tokens(question)
        best, best_score = None, 0
        for item in self.db.client_answers.get(customer_id, []):
            score = sum(1 for k in item.keywords if _tokens(k) & asked)
            if score > best_score:
                best, best_score = item, score
        if best is None:
            return (
                "Клиент: «Не могу ответить на этот вопрос — я такого не знаю»."
            )
        return f"Клиент: «{best.answer}»"

    @is_tool(ToolType.READ)
    def search_knowledge(self, query: str, limit: int = 5) -> list[dict]:
        """
        Найти статьи базы знаний банка по запросу. Возвращает краткие карточки:
        чтобы прочитать правило целиком, откройте статью через `get_article`.

        Args:
            query: Слова запроса, например «оспаривание частичный возврат».
            limit: Сколько статей вернуть, не более 10. По умолчанию 5.

        Returns:
            Список карточек: идентификатор, раздел, название, срок действия
            редакции и первые строки текста.

        Raises:
            ValueError: Если запрос пуст или limit вне диапазона 1..10.
        """
        if not query or not query.strip():
            raise _error(ERR_INVALID_ARGUMENT, "Запрос не может быть пустым.")
        if not 1 <= limit <= 10:
            raise _error(ERR_INVALID_ARGUMENT, "limit должен быть в диапазоне 1..10.")
        terms = _tokens(query)
        scored = []
        for article in self.db.articles.values():
            haystack = _tokens(
                " ".join(
                    [article.title, article.section, " ".join(article.keywords)]
                )
            )
            body = _tokens(article.body)
            score = sum(3 for t in terms if t in haystack)
            score += sum(1 for t in terms if t in body)
            if score:
                scored.append((-score, article.id, article))
        scored.sort()
        return [
            {
                "id": a.id,
                "section": a.section,
                "title": a.title,
                "effective_from": a.effective_from,
                "effective_to": a.effective_to,
                "snippet": a.body[:180].replace("\n", " ") + "…",
            }
            for _, _, a in scored[:limit]
        ]

    @is_tool(ToolType.READ)
    def get_article(self, article_id: str) -> Article:
        """
        Прочитать статью базы знаний целиком.

        Args:
            article_id: Идентификатор статьи, например 'kb_110'.

        Returns:
            Статья с полным текстом, сроком действия редакции и ссылками.

        Raises:
            ValueError: Если статья не найдена.
        """
        article = self.db.articles.get(article_id)
        if article is None:
            raise _error(ERR_NOT_FOUND, f"Статья {article_id} не найдена.")
        return article

    @is_tool(ToolType.READ)
    def list_tariffs(self) -> list[Tariff]:
        """
        Получить линейку тарифов банка: названия, стоимость обслуживания,
        условия бесплатности, максимальные лимиты и комиссии.

        Returns:
            Все тарифы банка.
        """
        return list(self.db.tariffs.values())

    @is_tool(ToolType.READ)
    def get_cashback_rules(self, customer_id: str) -> CashbackRules:
        """
        Получить правила кешбэка клиента: категории, ставки, исключения и день начисления.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Правила начисления кешбэка.

        Raises:
            ValueError: Если клиент или правила не найдены.
        """
        self._get_customer(customer_id)
        rules = self.db.cashback_rules.get(customer_id)
        if rules is None:
            raise _error(
                ERR_NOT_FOUND, f"Правила кешбэка для клиента {customer_id} не найдены."
            )
        return rules


    @is_tool(ToolType.READ)
    def get_devices(self, customer_id: str) -> list[Device]:
        """
        Получить устройства клиента: модель, последний вход, страна, блокировка.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Список устройств клиента.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        return [d for d in self.db.devices.values() if d.customer_id == customer_id]

    @is_tool(ToolType.READ)
    def get_promotions(
        self, customer_id: str, promo_code: Optional[str] = None
    ) -> list[Promotion]:
        """
        Получить акции клиента: промокод, условие, бонус и срок действия.

        Args:
            customer_id: Идентификатор клиента.
            promo_code: Отбор по промокоду.

        Returns:
            Список акций клиента. Прогресс по условию не хранится: его считают
            по операциям клиента.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        result = [p for p in self.db.promotions.values() if p.customer_id == customer_id]
        if promo_code is not None:
            result = [p for p in result if p.code.upper() == promo_code.upper()]
        return result

    @is_tool(ToolType.READ)
    def get_limit_request(self, customer_id: str) -> list[LimitRequest]:
        """
        Получить заявки клиента на изменение кредитного лимита и решения по ним.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Заявки клиента: запрошенный лимит, решение, дата и категория причины.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        return [
            r for r in self.db.limit_requests.values() if r.customer_id == customer_id
        ]

    @is_tool(ToolType.READ)
    def get_documents(self, customer_id: str) -> list[Document]:
        """
        Получить документы из системы банка, относящиеся к клиенту: название,
        дату и полный текст.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Список документов клиента с полным текстом каждого.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        return [d for d in self.db.documents.values() if d.customer_id == customer_id]


    @is_tool(ToolType.READ)
    def get_deposit(
        self, customer_id: str, deposit_id: Optional[str] = None
    ) -> list[dict]:
        """
        Получить вклады клиента: сумму, ставки, даты и счёт зачисления.

        Суммы к выплате инструмент не сообщает — они рассчитываются по формуле
        из политики банка.

        Args:
            customer_id: Идентификатор клиента.
            deposit_id: Идентификатор конкретного вклада.

        Returns:
            Список вкладов клиента.

        Raises:
            ValueError: Если клиент или вклад не найдены.
        """
        self._get_customer(customer_id)
        if deposit_id is not None:
            deposits = [self._get_deposit(deposit_id)]
        else:
            deposits = [
                d for d in self.db.deposits.values() if d.customer_id == customer_id
            ]
        # Суммы выплат не сообщаются: агент обязан вычислить их из ставок и дат
        # по формуле из политики. В БД поля остаются — среда использует их при
        # закрытии вклада.
        return [
            d.model_dump(exclude={"early_withdrawal_payout", "maturity_payout"})
            for d in deposits
        ]

    @is_tool(ToolType.READ)
    def get_loans(self, customer_id: str) -> list[Loan]:
        """
        Получить кредиты клиента: остаток долга, ставку, график, просрочки,
        историю штрафов и число использованных послаблений.

        Args:
            customer_id: Идентификатор клиента.

        Returns:
            Список кредитов клиента.

        Raises:
            ValueError: Если клиент не найден.
        """
        self._get_customer(customer_id)
        return [loan for loan in self.db.loans.values() if loan.customer_id == customer_id]

    # ------------------------------------------------------------------
    # Карты и лимиты
    # ------------------------------------------------------------------

    @is_tool(ToolType.WRITE)
    def block_card(self, card_id: str, reason: BlockReason) -> Card:
        """
        Заблокировать карту.

        Args:
            card_id: Идентификатор карты.
            reason: Причина: 'lost', 'stolen', 'fraud_suspected' или 'temporary'.

        Returns:
            Обновлённая карта.

        Raises:
            ValueError: Если карта не найдена или личность владельца не подтверждена.
        """
        card = self._get_card(card_id)
        self._require_verified(card.customer_id)
        card.status = "blocked"
        card.block_reason = reason
        card.blocked_at = self.db.today
        return card

    @is_tool(ToolType.WRITE)
    def unblock_card(self, card_id: str) -> Card:
        """
        Снять блокировку с карты и вернуть её в статус active.

        Args:
            card_id: Идентификатор карты.

        Returns:
            Обновлённая карта.

        Raises:
            ValueError: Если карта не найдена, не заблокирована или личность
                владельца не подтверждена.
        """
        card = self._get_card(card_id)
        self._require_verified(card.customer_id)
        if card.status != "blocked":
            raise _error(ERR_CARD_NOT_BLOCKED, f"Карта {card_id} не заблокирована.")
        if card.block_reason in ("lost", "stolen"):
            logger.warning(
                f"Политика нарушена: разблокирована карта {card_id} "
                f"с причиной {card.block_reason}."
            )
        card.status = "active"
        card.block_reason = None
        card.blocked_at = None
        return card

    @is_tool(ToolType.WRITE)
    def freeze_card(self, card_id: str) -> Card:
        """
        Временно приостановить операции по карте.

        Args:
            card_id: Идентификатор карты.

        Returns:
            Обновлённая карта.

        Raises:
            ValueError: Если карта не найдена, личность не подтверждена или
                карта уже заблокирована.
        """
        card = self._get_card(card_id)
        self._require_verified(card.customer_id)
        if card.status == "blocked":
            raise _error(ERR_ALREADY_FROZEN, f"Карта {card_id} уже заблокирована.")
        card.status = "blocked"
        card.block_reason = "temporary"
        card.blocked_at = self.db.today
        return card

    @is_tool(ToolType.WRITE)
    def reissue_card(self, card_id: str, delivery_address: Optional[str] = None) -> Card:
        """
        Перевыпустить карту. Если указан новый адрес доставки, требуется
        подтверждение одноразовым кодом.

        Args:
            card_id: Идентификатор карты.
            delivery_address: Новый адрес доставки. Если не указан, карта
                доставляется по текущему адресу.

        Returns:
            Обновлённая карта.

        Raises:
            ValueError: Если карта не найдена, личность не подтверждена или
                смена адреса не подтверждена одноразовым кодом.
        """
        card = self._get_card(card_id)
        self._require_verified(card.customer_id)
        if delivery_address is not None:
            canonical = _canon_address(delivery_address)
            current = _canon_address(card.delivery_address or "")
            if canonical != current:
                self._require_otp(card.customer_id)
                card.delivery_address = canonical
        card.reissue_status = "ordered"
        return card

    @is_tool(ToolType.WRITE)
    def set_limit(
        self,
        card_id: str,
        limit_type: LimitType,
        amount: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> CardLimits:
        """
        Изменить лимит карты в пределах максимума, разрешённого тарифом.

        Args:
            card_id: Идентификатор карты.
            limit_type: 'daily_cash_withdrawal', 'internet_operations' или 'sbp'.
            amount: Новое значение лимита для 'daily_cash_withdrawal' и 'sbp'.
            enabled: Включить или выключить интернет-операции.

        Returns:
            Обновлённые лимиты карты.

        Raises:
            ValueError: Если карта не найдена, личность не подтверждена,
                аргументы не соответствуют типу лимита или лимит превышает максимум
                по тарифу.
        """
        card = self._get_card(card_id)
        self._require_verified(card.customer_id)
        limits = self.db.card_limits.get(card_id)
        if limits is None:
            raise _error(ERR_NOT_FOUND, f"Лимиты для карты {card_id} не найдены.")
        if limit_type == "internet_operations":
            if enabled is None:
                raise _error(
                    ERR_INVALID_ARGUMENT,
                    "Для интернет-операций требуется аргумент enabled.",
                )
            limits.internet_operations_enabled = enabled
            return limits
        if amount is None:
            raise _error(
                ERR_INVALID_ARGUMENT, f"Для лимита {limit_type} требуется amount."
            )
        amount = float(amount)
        if limit_type == "daily_cash_withdrawal":
            tariff = self._get_tariff(card.customer_id)
            if amount > tariff.max_daily_cash_withdrawal:
                raise _error(
                    ERR_LIMIT_ABOVE_TARIFF,
                    f"Максимальный суточный лимит снятия наличных по тарифу "
                    f"«{tariff.name}» — {tariff.max_daily_cash_withdrawal:.0f} ₽.",
                )
            limits.daily_cash_withdrawal = amount
        else:
            tariff = self._get_tariff(card.customer_id)
            if amount > tariff.max_sbp_limit:
                raise _error(
                    ERR_LIMIT_ABOVE_TARIFF,
                    f"Максимальный лимит перевода по СБП по тарифу "
                    f"«{tariff.name}» — {tariff.max_sbp_limit:.0f} ₽.",
                )
            limits.sbp_limit = amount
        return limits

    # ------------------------------------------------------------------
    # Споры
    # ------------------------------------------------------------------

    @is_tool(ToolType.WRITE)
    def open_dispute(
        self, transaction_id: str, reason: str, amount: Optional[float] = None
    ) -> Dispute:
        """
        Открыть спор по операции.

        Args:
            transaction_id: Идентификатор оспариваемой операции.
            reason: Причина спора, одна из: 'fraud_suspected',
                'service_not_provided', 'duplicate_charge',
                'subscription_after_cancel', 'wrong_amount', 'other'.
            amount: Оспариваемая сумма. По умолчанию — сумма операции.

        Returns:
            Созданный спор.

        Raises:
            ValueError: Если операция не найдена, личность не подтверждена,
                операция ещё не проведена (статус hold), срок оспаривания истёк
                или спор по операции уже открыт.
        """
        transaction = self._get_transaction(transaction_id)
        self._require_verified(transaction.customer_id)
        if reason not in DISPUTE_REASONS:
            raise _error(
                ERR_INVALID_ARGUMENT,
                f"Недопустимая причина спора {reason!r}. "
                f"Допустимые: {', '.join(DISPUTE_REASONS)}.",
            )
        if transaction.dispute_id is not None:
            raise _error(
                ERR_DISPUTE_EXISTS,
                f"По операции {transaction_id} уже открыт спор "
                f"{transaction.dispute_id}.",
            )
        if transaction.status == "hold":
            raise _error(
                ERR_DISPUTE_ON_HOLD,
                f"Операция {transaction_id} ещё не проведена (статус hold). "
                "Спор можно открыть после её проведения.",
            )
        age = (self._today - date.fromisoformat(transaction.date)).days
        if age > self.db.dispute_window_days:
            raise _error(
                ERR_DISPUTE_PERIOD_EXPIRED,
                f"Срок оспаривания операции истёк: прошло {age} дней при "
                f"допустимых {self.db.dispute_window_days}.",
            )
        dispute = Dispute(
            id=self._next_dispute_id(),
            customer_id=transaction.customer_id,
            transaction_id=transaction_id,
            status="under_review",
            filed_at=self.db.today,
            sla_days=30,
            reason=reason,
            amount=float(amount) if amount is not None else transaction.amount,
        )
        self.db.disputes[dispute.id] = dispute
        transaction.dispute_id = dispute.id
        return dispute

    @is_tool(ToolType.WRITE)
    def cancel_dispute(self, dispute_id: str, reason: str) -> Dispute:
        """
        Отозвать спор по заявлению клиента.

        Args:
            dispute_id: Идентификатор спора.
            reason: Причина отзыва свободным текстом.

        Returns:
            Обновлённый спор.

        Raises:
            ValueError: Если спор не найден, личность не подтверждена или спор
                уже не находится на рассмотрении.
        """
        dispute = self.db.disputes.get(dispute_id)
        if dispute is None:
            raise _error(ERR_NOT_FOUND, f"Спор {dispute_id} не найден.")
        self._require_verified(dispute.customer_id)
        if dispute.status != "under_review":
            raise _error(
                ERR_DISPUTE_NOT_OPEN,
                f"Спор {dispute_id} уже завершён со статусом {dispute.status}: "
                "отозвать можно только спор на рассмотрении.",
            )
        dispute.status = "cancelled"
        return dispute

    @is_tool(ToolType.WRITE)
    def close_dispute(self, dispute_id: str, resolution: str) -> Dispute:
        """
        Закрыть спор решением банка.

        Args:
            dispute_id: Идентификатор спора.
            resolution: 'approved' — в пользу клиента, 'rejected' — отказ.

        Returns:
            Обновлённый спор.

        Raises:
            ValueError: Если спор не найден, личность не подтверждена, спор не
                на рассмотрении либо решение недопустимо.
        """
        dispute = self.db.disputes.get(dispute_id)
        if dispute is None:
            raise _error(ERR_NOT_FOUND, f"Спор {dispute_id} не найден.")
        self._require_verified(dispute.customer_id)
        if resolution not in ("approved", "rejected"):
            raise _error(
                ERR_INVALID_ARGUMENT,
                "Решение должно быть 'approved' или 'rejected'.",
            )
        if dispute.status != "under_review":
            raise _error(
                ERR_DISPUTE_NOT_UNDER_REVIEW,
                f"Спор {dispute_id} уже завершён со статусом {dispute.status}.",
            )
        dispute.status = resolution
        return dispute

    @is_tool(ToolType.WRITE)
    def release_hold(self, transaction_id: str) -> Transaction:
        """
        Снять авторизационный холд по операции вручную.

        Args:
            transaction_id: Идентификатор операции в статусе hold.

        Returns:
            Обновлённая операция.

        Raises:
            ValueError: Если операция не найдена, личность не подтверждена или
                по операции нет холда.
        """
        transaction = self._get_transaction(transaction_id)
        self._require_verified(transaction.customer_id)
        if transaction.status != "hold":
            raise _error(
                ERR_NO_HOLD,
                f"По операции {transaction_id} нет авторизационного холда.",
            )
        transaction.status = "declined"
        transaction.hold_expires_at = None
        return transaction

    # ------------------------------------------------------------------
    # Обращения
    # ------------------------------------------------------------------

    @is_tool(ToolType.WRITE)
    def create_case(
        self,
        customer_id: str,
        category: CaseCategory,
        transaction_id: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> Case:
        """
        Создать обращение в профильное подразделение банка.

        Args:
            customer_id: Идентификатор клиента.
            category: Категория обращения, одна из: 'fraud_disclosed_code',
                'unauthorized_operation', 'safe_account_scam',
                'misdirected_transfer', 'misdirected_utility_payment',
                'merchant_investigation', 'restructuring', 'credit_holidays',
                'branch_complaint', 'escalation', 'investment_advice',
                'suspicious_device', 'other'.
            transaction_id: Операция, из-за которой создано обращение.
            amount: Сумма, по которой заводится обращение, ₽. Без указания —
                сумма операции `transaction_id`, если она задана.

        Returns:
            Созданное обращение.

        Raises:
            ValueError: Если клиент или операция не найдены, личность не
                подтверждена либо категория недопустима.
        """
        self._get_customer(customer_id)
        self._require_verified(customer_id)
        if category not in CASE_CATEGORIES:
            raise _error(
                ERR_INVALID_ARGUMENT,
                f"Недопустимая категория обращения {category!r}. "
                f"Допустимые: {', '.join(CASE_CATEGORIES)}.",
            )
        if transaction_id is not None:
            transaction = self._get_transaction(transaction_id)
            if transaction.customer_id != customer_id:
                raise _error(
                    ERR_INVALID_ARGUMENT,
                    f"Операция {transaction_id} не принадлежит клиенту {customer_id}.",
                )
        if amount is None and transaction_id is not None:
            # Сумма одной операции — умолчание, чтобы «указал сумму» и «не
            # указал» давали одно состояние; при нескольких операциях агент
            # обязан сложить их сам, и это уже проверяется хешем.
            amount = self.db.transactions[transaction_id].amount
        case = Case(
            id=self._next_case_id(),
            customer_id=customer_id,
            category=category,
            transaction_id=transaction_id,
            amount=float(amount) if amount is not None else None,
            created_at=self.db.today,
            status="open",
        )
        self.db.cases[case.id] = case
        return case

    # ------------------------------------------------------------------
    # Подписки и автоплатежи
    # ------------------------------------------------------------------

    @is_tool(ToolType.WRITE)
    def cancel_subscription(self, subscription_id: str) -> Subscription:
        """
        Отменить подписку. Оплаченный период при этом сохраняется.

        Args:
            subscription_id: Идентификатор подписки.

        Returns:
            Обновлённая подписка.

        Raises:
            ValueError: Если подписка не найдена или личность не подтверждена.
        """
        subscription = self.db.subscriptions.get(subscription_id)
        if subscription is None:
            raise _error(ERR_NOT_FOUND, f"Подписка {subscription_id} не найдена.")
        self._require_verified(subscription.customer_id)
        if subscription.status == "cancelled":
            raise _error(
                ERR_ALREADY_CANCELLED,
                f"Подписка {subscription_id} уже отменена "
                f"({subscription.cancelled_at}).",
            )
        subscription.status = "cancelled"
        subscription.cancelled_at = self.db.today
        subscription.next_charge_date = None
        return subscription

    @is_tool(ToolType.WRITE)
    def cancel_autopayment(self, autopayment_id: str) -> dict:
        """
        Отменить автоплатёж. Автоплатёж не связан с подпиской и отключается отдельно.

        Args:
            autopayment_id: Идентификатор автоплатежа.

        Returns:
            Обновлённый автоплатёж.

        Raises:
            ValueError: Если автоплатёж не найден или личность не подтверждена.
        """
        autopayment = self.db.autopayments.get(autopayment_id)
        if autopayment is None:
            raise _error(ERR_NOT_FOUND, f"Автоплатёж {autopayment_id} не найден.")
        self._require_verified(autopayment.customer_id)
        if autopayment.status == "cancelled":
            raise _error(
                ERR_ALREADY_CANCELLED, f"Автоплатёж {autopayment_id} уже отменён."
            )
        autopayment.status = "cancelled"
        autopayment.cancelled_at = self.db.today
        return autopayment.model_dump()

    # ------------------------------------------------------------------
    # Вклады и кредиты
    # ------------------------------------------------------------------

    @is_tool(ToolType.WRITE)
    def close_deposit(self, deposit_id: str, payout_account_id: str) -> dict:
        """
        Закрыть вклад с выплатой на указанный счёт. При закрытии до даты
        окончания срока проценты пересчитываются по пониженной ставке.

        Args:
            deposit_id: Идентификатор вклада.
            payout_account_id: Счёт, на который зачисляется выплата.

        Returns:
            Словарь с закрытым вкладом, выплаченной суммой и остатком счёта.

        Raises:
            ValueError: Если вклад или счёт не найдены, вклад уже закрыт или
                личность не подтверждена.
        """
        deposit = self._get_deposit(deposit_id)
        self._require_verified(deposit.customer_id)
        if deposit.status == "closed":
            raise _error(ERR_DEPOSIT_CLOSED, f"Вклад {deposit_id} уже закрыт.")
        account = self._get_account(payout_account_id)
        early = self._today < date.fromisoformat(deposit.matures_at)
        payout = (
            deposit.early_withdrawal_payout if early else deposit.maturity_payout
        )
        deposit.status = "closed"
        deposit.closed_at = self.db.today
        account.balance = round(account.balance + payout, 2)
        transaction = Transaction(
            id=self._next_transaction_id(),
            customer_id=deposit.customer_id,
            account_id=account.id,
            date=self.db.today,
            amount=payout,
            merchant=f"Выплата по вкладу {deposit_id}",
            kind="deposit_payout",
            channel="online",
        )
        self.db.transactions[transaction.id] = transaction
        return {
            "deposit": deposit.model_dump(),
            "payout": payout,
            "early_withdrawal": early,
            "account_balance": account.balance,
        }

    @is_tool(ToolType.WRITE)
    def early_repayment(self, loan_id: str, amount: float, mode: RepaymentMode) -> dict:
        """
        Досрочно погасить кредит полностью или частично. Требует подтверждения
        одноразовым кодом. Сумма списывается со счёта заёмщика и гасит сначала
        неоплаченные штрафы, затем начисленные проценты, затем основной долг.

        Args:
            loan_id: Идентификатор кредита.
            amount: Сумма досрочного погашения, ₽.
            mode: 'reduce_payment' — уменьшить платёж, 'reduce_term' — срок.

        Returns:
            Словарь с обновлённым кредитом и остатком счёта.

        Raises:
            ValueError: Если кредит не найден, закрыт, личность или код не
                подтверждены либо на счёте недостаточно средств.
        """
        loan = self._get_loan(loan_id)
        self._require_verified(loan.customer_id)
        self._require_otp(loan.customer_id)
        if loan.status == "closed":
            raise _error(ERR_LOAN_CLOSED, f"Кредит {loan_id} уже закрыт.")
        amount = float(amount)
        if amount <= 0:
            raise _error(ERR_INVALID_ARGUMENT, "Сумма погашения должна быть больше нуля.")
        accounts = [
            a
            for a in self.db.accounts.values()
            if a.customer_id == loan.customer_id and a.status == "active"
        ]
        account = next((a for a in accounts if a.balance >= amount), None)
        if account is None:
            raise _error(
                ERR_INSUFFICIENT_FUNDS,
                f"На счетах клиента недостаточно средств для списания {amount:.2f} ₽.",
            )
        rest = amount
        for penalty in loan.penalties:
            if penalty.waived or penalty.paid or rest <= 0:
                continue
            if rest >= penalty.amount:
                penalty.paid = True
                rest = round(rest - penalty.amount, 2)
        interest_paid = min(rest, loan.accrued_interest)
        loan.accrued_interest = round(loan.accrued_interest - interest_paid, 2)
        rest = round(rest - interest_paid, 2)
        principal_paid = min(rest, loan.principal)
        loan.principal = round(loan.principal - principal_paid, 2)
        if mode == "reduce_payment" and principal_paid > 0 and loan.principal > 0:
            # Срок сохраняется, платёж уменьшается пропорционально новому долгу.
            before = loan.principal + principal_paid
            loan.monthly_payment = round(
                loan.monthly_payment * loan.principal / before, 2
            )
        rest = round(rest - principal_paid, 2)
        if loan.principal == 0 and loan.accrued_interest == 0:
            loan.status = "closed"
            loan.days_overdue = 0
            loan.next_payment_date = None
        charged = round(amount - rest, 2)
        account.balance = round(account.balance - charged, 2)
        transaction = Transaction(
            id=self._next_transaction_id(),
            customer_id=loan.customer_id,
            account_id=account.id,
            date=self.db.today,
            amount=charged,
            merchant=f"Досрочное погашение кредита {loan_id}",
            kind="loan_repayment",
            channel="online",
        )
        self.db.transactions[transaction.id] = transaction
        return {
            "loan": loan.model_dump(),
            "charged": charged,
            "mode": mode,
            "account_balance": account.balance,
        }

    @is_tool(ToolType.WRITE)
    def waive_penalty(self, loan_id: str, penalty_id: str) -> dict:
        """
        Списать штраф по кредиту в качестве послабления.

        Args:
            loan_id: Идентификатор кредита.
            penalty_id: Идентификатор штрафа.

        Returns:
            Словарь с обновлённым кредитом.

        Raises:
            ValueError: Если кредит или штраф не найдены либо личность не подтверждена.
        """
        loan = self._get_loan(loan_id)
        self._require_verified(loan.customer_id)
        penalty = next((p for p in loan.penalties if p.id == penalty_id), None)
        if penalty is None:
            raise _error(
                ERR_NOT_FOUND, f"Штраф {penalty_id} по кредиту {loan_id} не найден."
            )
        if loan.waivers_used >= loan.max_waivers:
            logger.warning(
                f"Политика нарушена: повторное списание штрафа по кредиту {loan_id}."
            )
        penalty.waived = True
        loan.waivers_used += 1
        return {"loan": loan.model_dump(), "waived_penalty_id": penalty_id}

    # ------------------------------------------------------------------
    # Комиссии и кешбэк
    # ------------------------------------------------------------------

    @is_tool(ToolType.WRITE)
    def refund_fee(
        self, transaction_id: str, reason: str, amount: Optional[float] = None
    ) -> dict:
        """
        Вернуть клиенту удержанную комиссию на счёт операции.

        Args:
            transaction_id: Операция, по которой удержана комиссия.
            reason: Основание возврата.
            amount: Возвращаемая часть комиссии, ₽. Без указания возвращается
                вся удержанная комиссия.

        Returns:
            Словарь с возвращённой суммой и остатком счёта.

        Raises:
            ValueError: Если операция не найдена, по ней нет комиссии или
                личность не подтверждена.
        """
        transaction = self._get_transaction(transaction_id)
        self._require_verified(transaction.customer_id)
        fee = (
            transaction.amount
            if transaction.kind == "fee"
            else (transaction.fee_amount or 0.0)
        )
        if fee <= 0:
            raise _error(
                ERR_NO_FEE, f"По операции {transaction_id} комиссия не удерживалась."
            )
        refund_amount = float(amount) if amount is not None else fee
        if refund_amount <= 0 or refund_amount > fee:
            raise _error(
                ERR_INVALID_ARGUMENT,
                f"Возврат должен быть в пределах удержанной комиссии {fee:.2f} ₽.",
            )
        account = self._get_account(transaction.account_id)
        account.balance = round(account.balance + refund_amount, 2)
        transaction.fee_refunded = True
        refund = Transaction(
            id=self._next_transaction_id(),
            customer_id=transaction.customer_id,
            account_id=account.id,
            date=self.db.today,
            amount=refund_amount,
            merchant=f"Возврат комиссии по операции {transaction_id}",
            kind="fee_refund",
            channel="online",
        )
        self.db.transactions[refund.id] = refund
        return {
            "refunded": refund_amount,
            "reason": reason,
            "account_balance": account.balance,
        }

    @is_tool(ToolType.WRITE)
    def refund_transaction(self, transaction_id: str, reason: str) -> dict:
        """
        Вернуть клиенту всю сумму операции.

        Args:
            transaction_id: Операция, которую возвращаем.
            reason: Основание возврата.

        Returns:
            Возвращённая сумма и остаток счёта.

        Raises:
            ValueError: Если операция не найдена, личность не подтверждена или
                операция ещё не проведена.
        """
        transaction = self._get_transaction(transaction_id)
        self._require_verified(transaction.customer_id)
        if transaction.status != "posted":
            raise _error(
                ERR_INVALID_ARGUMENT,
                f"Операция {transaction_id} не проведена: возврат невозможен.",
            )
        account = self._get_account(transaction.account_id)
        account.balance = round(account.balance + transaction.amount, 2)
        refund = Transaction(
            id=self._next_transaction_id(),
            customer_id=transaction.customer_id,
            account_id=account.id,
            date=self.db.today,
            amount=transaction.amount,
            merchant=f"Возврат по операции {transaction_id}",
            kind="fee_refund",
            channel="online",
        )
        self.db.transactions[refund.id] = refund
        return {
            "refunded": transaction.amount,
            "reason": reason,
            "account_balance": account.balance,
        }

    @is_tool(ToolType.WRITE)
    def grant_cashback(self, transaction_id: str, amount: float) -> dict:
        """
        Начислить кешбэк по операции вручную.

        Args:
            transaction_id: Операция, по которой начисляется кешбэк.
            amount: Сумма кешбэка, ₽.

        Returns:
            Словарь с начисленной суммой и остатком счёта.

        Raises:
            ValueError: Если операция не найдена или личность не подтверждена.
        """
        transaction = self._get_transaction(transaction_id)
        self._require_verified(transaction.customer_id)
        if amount <= 0:
            raise _error(ERR_INVALID_ARGUMENT, "Сумма кешбэка должна быть больше нуля.")
        account = self._get_account(transaction.account_id)
        account.balance = round(account.balance + amount, 2)
        transaction.cashback_granted = True
        rules = self.db.cashback_rules.get(transaction.customer_id)
        if rules is not None:
            rules.accrued_current_period = round(
                rules.accrued_current_period + amount, 2
            )
        credit = Transaction(
            id=self._next_transaction_id(),
            customer_id=transaction.customer_id,
            account_id=account.id,
            date=self.db.today,
            amount=amount,
            merchant=f"Кешбэк по операции {transaction_id}",
            kind="cashback",
            channel="online",
        )
        self.db.transactions[credit.id] = credit
        return {"granted": amount, "account_balance": account.balance}


    @is_tool(ToolType.WRITE)
    def open_account(
        self, customer_id: str, currency: str, account_type: str = "current"
    ) -> Account:
        """
        Открыть клиенту новый счёт.

        Args:
            customer_id: Идентификатор клиента.
            currency: Валюта счёта, например RUB, USD, EUR, CNY.
            account_type: Тип счёта, например current или savings.

        Returns:
            Открытый счёт.

        Raises:
            ValueError: Если клиент не найден или личность не подтверждена.
        """
        self._get_customer(customer_id)
        self._require_verified(customer_id)
        account = Account(
            id=self._next_seq_id(self.db.accounts, "acc", 9000),
            customer_id=customer_id,
            account_type=account_type,
            currency=currency.upper(),
            balance=0.0,
            status="active",
            debt=0.0,
        )
        self.db.accounts[account.id] = account
        return account

    @is_tool(ToolType.WRITE)
    def close_account(self, account_id: str) -> Account:
        """
        Закрыть счёт клиента.

        Args:
            account_id: Идентификатор счёта.

        Returns:
            Закрытый счёт.

        Raises:
            ValueError: Если счёт не найден, личность не подтверждена, счёт уже
                закрыт, на нём остался положительный остаток или есть задолженность.
        """
        account = self._get_account(account_id)
        self._require_verified(account.customer_id)
        if account.status == "closed":
            raise _error(ERR_ACCOUNT_CLOSED, f"Счёт {account_id} уже закрыт.")
        if account.debt > 0 or account.balance < 0:
            raise _error(
                ERR_ACCOUNT_HAS_DEBT,
                f"По счёту {account_id} есть задолженность; закрытие невозможно.",
            )
        if account.balance > 0:
            raise _error(
                ERR_ACCOUNT_NOT_EMPTY,
                f"На счёте {account_id} остаток {account.balance:.2f} ₽. "
                "Сначала переведите остаток на другой счёт клиента.",
            )
        account.status = "closed"
        return account

    @is_tool(ToolType.WRITE)
    def transfer_between_own_accounts(
        self, from_account_id: str, to_account_id: str, amount: float
    ) -> dict:
        """
        Перевести деньги между счетами одного клиента.

        Args:
            from_account_id: Счёт списания.
            to_account_id: Счёт зачисления.
            amount: Сумма перевода, ₽.

        Returns:
            Остатки обоих счетов после перевода.

        Raises:
            ValueError: Если счёт не найден, счета принадлежат разным клиентам
                или совпадают, личность не подтверждена, сумма не положительна
                либо средств недостаточно.
        """
        source = self._get_account(from_account_id)
        target = self._get_account(to_account_id)
        self._require_verified(source.customer_id)
        if source.id == target.id:
            raise _error(ERR_SAME_ACCOUNT, "Счета списания и зачисления совпадают.")
        if source.customer_id != target.customer_id:
            raise _error(
                ERR_FOREIGN_ACCOUNT, "Счета принадлежат разным клиентам."
            )
        amount = float(amount)
        if amount <= 0:
            raise _error(ERR_INVALID_ARGUMENT, "Сумма перевода должна быть больше нуля.")
        if source.balance < amount:
            raise _error(
                ERR_INSUFFICIENT_FUNDS,
                f"На счёте {from_account_id} недостаточно средств: "
                f"остаток {source.balance:.2f} ₽.",
            )
        source.balance = round(source.balance - amount, 2)
        target.balance = round(target.balance + amount, 2)
        return {
            "from_balance": source.balance,
            "to_balance": target.balance,
            "amount": amount,
        }

    @is_tool(ToolType.WRITE)
    def order_statement(
        self, account_id: str, date_from: str, date_to: str, email: str
    ) -> Statement:
        """
        Заказать выписку по счёту на электронную почту.

        Args:
            account_id: Идентификатор счёта.
            date_from: Начало периода, ГГГГ-ММ-ДД.
            date_to: Конец периода, ГГГГ-ММ-ДД.
            email: Адрес доставки.

        Returns:
            Созданный заказ выписки.

        Raises:
            ValueError: Если счёт не найден, личность не подтверждена или адрес
                не совпадает с подтверждённым адресом клиента.
        """
        account = self._get_account(account_id)
        self._require_verified(account.customer_id)
        customer = self._get_customer(account.customer_id)
        if customer.email is None or email.strip().lower() != customer.email.lower():
            raise _error(
                ERR_EMAIL_NOT_CONFIRMED,
                "Выписка отправляется только на подтверждённый адрес из профиля клиента.",
            )
        statement = Statement(
            id=self._next_seq_id(self.db.statements, "stm", 1000),
            customer_id=account.customer_id,
            account_id=account_id,
            date_from=date_from,
            date_to=date_to,
            email=customer.email,
            created_at=self.db.today,
        )
        self.db.statements[statement.id] = statement
        return statement

    @is_tool(ToolType.WRITE)
    def change_tariff(self, customer_id: str, new_tariff_id: str) -> Customer:
        """
        Перевести клиента на другой тариф со следующего расчётного периода.

        Args:
            customer_id: Идентификатор клиента.
            new_tariff_id: Идентификатор нового тарифа.

        Returns:
            Обновлённый профиль клиента.

        Raises:
            ValueError: Если клиент или тариф не найдены либо личность не подтверждена.
        """
        customer = self._get_customer(customer_id)
        self._require_verified(customer_id)
        if new_tariff_id not in self.db.tariffs:
            raise _error(ERR_NOT_FOUND, f"Тариф {new_tariff_id} не найден.")
        today = self._today
        first_next = (
            date(today.year + 1, 1, 1)
            if today.month == 12
            else date(today.year, today.month + 1, 1)
        )
        customer.pending_tariff_id = new_tariff_id
        customer.pending_tariff_from = first_next.isoformat()
        return customer

    @is_tool(ToolType.WRITE)
    def unblock_device(self, device_id: str) -> Device:
        """
        Разблокировать вход с устройства клиента.

        Args:
            device_id: Идентификатор устройства.

        Returns:
            Обновлённое устройство.

        Raises:
            ValueError: Если устройство не найдено, личность не подтверждена или
                устройство не заблокировано.
        """
        device = self.db.devices.get(device_id)
        if device is None:
            raise _error(ERR_NOT_FOUND, f"Устройство {device_id} не найдено.")
        self._require_verified(device.customer_id)
        if not device.blocked:
            raise _error(ERR_NOT_BLOCKED, f"Устройство {device_id} не заблокировано.")
        device.blocked = False
        return device

    @is_tool(ToolType.WRITE)
    def unblock_operation(self, transaction_id: str) -> Transaction:
        """
        Снять блокировку системы безопасности с операции.

        Args:
            transaction_id: Идентификатор операции в статусе blocked.

        Returns:
            Обновлённая операция.

        Raises:
            ValueError: Если операция не найдена, личность не подтверждена,
                операция не заблокирована или не подтверждена одноразовым кодом.
        """
        transaction = self._get_transaction(transaction_id)
        self._require_verified(transaction.customer_id)
        if transaction.status != "blocked":
            raise _error(
                ERR_NOT_BLOCKED,
                f"Операция {transaction_id} не заблокирована системой безопасности.",
            )
        self._require_otp(transaction.customer_id)
        transaction.status = "posted"
        return transaction

    @is_tool(ToolType.WRITE)
    def reveal_card_details(self, card_id: str) -> dict:
        """
        Получить полный номер карты и CVV-код.

        Args:
            card_id: Идентификатор карты.

        Returns:
            Полный номер карты и CVV-код. Факт доступа фиксируется в системе.

        Raises:
            ValueError: Если карта не найдена или личность не подтверждена.
        """
        card = self._get_card(card_id)
        self._require_verified(card.customer_id)
        card.secret_revealed = True
        return {"card_id": card.id, "full_number": card.full_number, "cvv": card.cvv}

    @is_tool(ToolType.WRITE)
    def request_credit_holidays(self, loan_id: str) -> Case:
        """
        Оформить заявку на кредитные каникулы по кредиту.

        Args:
            loan_id: Идентификатор кредита.

        Returns:
            Созданное обращение по заявке.

        Raises:
            ValueError: Если кредит не найден или личность не подтверждена.
        """
        loan = self._get_loan(loan_id)
        self._require_verified(loan.customer_id)
        case = Case(
            id=self._next_case_id(),
            customer_id=loan.customer_id,
            category="credit_holidays",
            created_at=self.db.today,
            status="open",
        )
        self.db.cases[case.id] = case
        return case

    @is_tool(ToolType.WRITE)
    def share_document(self, document_id: str) -> Document:
        """
        Отправить документ клиенту в чат целиком.

        Args:
            document_id: Идентификатор документа.

        Returns:
            Документ с отметкой о передаче.

        Raises:
            ValueError: Если документ не найден, личность клиента не
                подтверждена или документ уже передавался.
        """
        document = self.db.documents.get(document_id)
        if document is None:
            raise _error(ERR_NOT_FOUND, f"Документ {document_id} не найден.")
        self._require_verified(document.customer_id)
        if document.shared:
            raise _error(
                ERR_ALREADY_SHARED, f"Документ {document_id} уже передан клиенту."
            )
        document.shared = True
        return document

    @is_tool(ToolType.WRITE)
    def reply_to_ticket(self, customer_id: str, text: str) -> str:
        """
        Отправить клиенту ответ по обращению. Это то, что клиент прочитает:
        итог разбирательства, все названные суммы, даты и причины отказа.

        Отправляется один раз в конце работы над обращением, когда все
        операции уже выполнены.

        Args:
            customer_id: Идентификатор клиента.
            text: Текст ответа клиенту.

        Returns:
            Подтверждение отправки.

        Raises:
            ValueError: Если клиент не найден, личность не подтверждена,
                текст пуст или ответ уже отправлен.
        """
        # Тип WRITE обязателен: при оценке среда пересобирается проигрыванием
        # траектории, и немутирующие вызовы пропускаются — как GENERIC этот
        # инструмент не восстанавливался, и проверка ответа проваливалась
        # всегда. Подтверждения личности он при этом не требует: Требование обернулось обрывом:
        # получив ERR_NOT_VERIFIED, агент не исправлялся, а прекращал работу —
        # одна ошибка уносила всю задачу вместо того, чтобы её усложнить.
        # Идентификация по-прежнему обязательна для всех операций записи.
        self._get_customer(customer_id)
        if not text or not text.strip():
            raise _error(ERR_INVALID_ARGUMENT, "Текст ответа не может быть пустым.")
        if customer_id in self.db.ticket_replies:
            raise _error(
                ERR_ALREADY_REPLIED,
                f"Ответ клиенту {customer_id} уже отправлен.",
            )
        self.db.ticket_replies[customer_id] = text
        return "Ответ отправлен клиенту."

    @is_tool(ToolType.WRITE)
    def escalate_to_human(self, customer_id: str, reason: str) -> Case:
        """
        Передать обращение клиента специалисту.

        Args:
            customer_id: Идентификатор клиента.
            reason: Причина эскалации.

        Returns:
            Созданное обращение категории escalation.

        Raises:
            ValueError: Если клиент не найден или личность не подтверждена.
        """
        self._get_customer(customer_id)
        self._require_verified(customer_id)
        case = Case(
            id=self._next_case_id(),
            customer_id=customer_id,
            category="escalation",
            created_at=self.db.today,
            status="open",
        )
        self.db.cases[case.id] = case
        return case


    # ------------------------------------------------------------------
    # Проверки среды (env_assertions), не инструменты
    # ------------------------------------------------------------------

    def assert_card_status(
        self,
        card_id: str,
        expected_status: str,
        expected_reason: Optional[str] = None,
    ) -> bool:
        """Статус карты и причина блокировки соответствуют ожидаемым."""
        card = self._get_card(card_id)
        if card.status != expected_status:
            return False
        if expected_reason is not None and card.block_reason != expected_reason:
            return False
        return True

    def assert_subscription_status(
        self, subscription_id: str, expected_status: str
    ) -> bool:
        """Статус подписки соответствует ожидаемому."""
        subscription = self.db.subscriptions.get(subscription_id)
        if subscription is None:
            raise _error(ERR_NOT_FOUND, f"Подписка {subscription_id} не найдена.")
        return subscription.status == expected_status

    def assert_autopayment_status(
        self, autopayment_id: str, expected_status: str
    ) -> bool:
        """Статус автоплатежа соответствует ожидаемому."""
        autopayment = self.db.autopayments.get(autopayment_id)
        if autopayment is None:
            raise _error(ERR_NOT_FOUND, f"Автоплатёж {autopayment_id} не найден.")
        return autopayment.status == expected_status

    def assert_dispute_exists(
        self, transaction_id: str, expected_amount: Optional[float] = None
    ) -> bool:
        """По операции открыт спор, при необходимости — на ожидаемую сумму."""
        transaction = self._get_transaction(transaction_id)
        if transaction.dispute_id is None:
            return False
        dispute = self.db.disputes.get(transaction.dispute_id)
        if dispute is None:
            return False
        if expected_amount is not None and abs(dispute.amount - expected_amount) > 0.01:
            return False
        return True

    def assert_dispute_status(self, dispute_id: str, expected_status: str) -> bool:
        """Статус спора соответствует ожидаемому."""
        dispute = self.db.disputes.get(dispute_id)
        if dispute is None:
            raise _error(ERR_NOT_FOUND, f"Спор {dispute_id} не найден.")
        return dispute.status == expected_status

    def assert_case_exists(
        self,
        customer_id: str,
        expected_category: str,
        transaction_id: Optional[str] = None,
        expected_amount: Optional[float] = None,
    ) -> bool:
        """По клиенту создано обращение нужной категории, операции и суммы."""
        self._get_customer(customer_id)
        for case in self.db.cases.values():
            if case.customer_id != customer_id:
                continue
            if case.category != expected_category:
                continue
            if transaction_id is not None and case.transaction_id != transaction_id:
                continue
            if expected_amount is not None and (
                case.amount is None or abs(case.amount - expected_amount) > 0.01
            ):
                continue
            return True
        return False

    def assert_no_case(self, customer_id: str) -> bool:
        """По клиенту не создавалось обращений."""
        self._get_customer(customer_id)
        return not any(c.customer_id == customer_id for c in self.db.cases.values())

    def assert_no_case_of_category(self, customer_id: str, category: str) -> bool:
        """По клиенту нет обращения указанной категории."""
        self._get_customer(customer_id)
        return not any(
            c.customer_id == customer_id and c.category == category
            for c in self.db.cases.values()
        )

    def assert_answer_contains(self, customer_id: str, expected: list[str]) -> bool:
        """Ответ клиенту отправлен и содержит все требуемые подстроки.

        Замена communicate_info в тикетном формате. Сверка идёт по тексту
        одного инструмента, а не по всем репликам агента, и здесь нет
        вычищения запятых, на котором штатный оценщик речи ронял верные
        ответы: «1 240,50 ₽» сравнивается как есть.
        """
        reply = self.db.ticket_replies.get(customer_id)
        if reply is None:
            return False
        def norm(value: str) -> str:
            # Неразрывный и тонкий пробелы модель ставит наравне с обычным:
            # «15 490 ₽» не должно зависеть от того, какой из них выбран.
            for space in ("\u00a0", "\u202f", "\u2009"):
                value = value.replace(space, " ")
            return value.lower().replace("ё", "е")

        text = norm(reply)
        return all(norm(sub) in text for sub in expected)

    def assert_no_refund_transaction(self, transaction_id: str) -> bool:
        """По операции не выполнялся полный возврат суммы."""
        self._get_transaction(transaction_id)
        return not any(
            t.kind == "fee_refund" and transaction_id in t.merchant
            and t.merchant.startswith("Возврат по операции")
            for t in self.db.transactions.values()
        )

    def assert_tool_not_called(self, tool_name: str) -> bool:
        """Инструмент не вызывался ни разу.

        Так наказывается запрещённое действие само по себе, даже если агент
        успел его отменить и состояние сошлось.
        """
        return tool_name not in self.db.tool_calls_log

    def assert_tool_call_count(self, tool_name: str, expected: int) -> bool:
        """Инструмент вызван ровно ожидаемое число раз."""
        return self.db.tool_calls_log.count(tool_name) == expected

    def assert_no_reply(self, customer_id: str) -> bool:
        """Ответ клиенту не отправлялся."""
        return customer_id not in self.db.ticket_replies

    def assert_article_is_reachable(self, article_id: str, query: str) -> bool:
        """Статья находится поиском по запросу — задача не опирается на статью,
        которую нельзя найти словами клиента."""
        found = self.search_knowledge(query=query, limit=10)
        return any(card["id"] == article_id for card in found)

    def assert_document_shared(self, document_id: str, expected_shared: bool) -> bool:
        """Документ передан клиенту (или не передавался) — как ожидается."""
        document = self.db.documents.get(document_id)
        if document is None:
            raise _error(ERR_NOT_FOUND, f"Документ {document_id} не найден.")
        return document.shared == expected_shared

    def assert_statements_count(self, customer_id: str, expected_count: int) -> bool:
        """Клиенту заказано ровно ожидаемое число выписок."""
        self._get_customer(customer_id)
        return (
            sum(1 for s in self.db.statements.values() if s.customer_id == customer_id)
            == expected_count
        )

    def assert_transaction_status(
        self, transaction_id: str, expected_status: str
    ) -> bool:
        """Статус операции соответствует ожидаемому."""
        return self._get_transaction(transaction_id).status == expected_status

    def assert_hold_intact(self, transaction_id: str, expected_expiry: str) -> bool:
        """Холд по операции не снимался вручную: статус и срок не изменились."""
        transaction = self._get_transaction(transaction_id)
        return (
            transaction.status == "hold"
            and transaction.hold_expires_at == expected_expiry
        )

    def assert_no_dispute(self, transaction_id: str) -> bool:
        """По операции спор не открывался."""
        return self._get_transaction(transaction_id).dispute_id is None

    def assert_penalty_state(
        self, loan_id: str, penalty_id: str, waived: bool, paid: bool
    ) -> bool:
        """Штраф находится в ожидаемом состоянии."""
        loan = self._get_loan(loan_id)
        penalty = next((p for p in loan.penalties if p.id == penalty_id), None)
        if penalty is None:
            raise _error(ERR_NOT_FOUND, f"Штраф {penalty_id} не найден.")
        return penalty.waived == waived and penalty.paid == paid

    def assert_waivers_used(self, loan_id: str, expected_count: int) -> bool:
        """Число использованных послаблений по кредиту не изменилось."""
        return self._get_loan(loan_id).waivers_used == expected_count

    def assert_account_balance(self, account_id: str, expected_balance: float) -> bool:
        """Остаток на счёте равен ожидаемому."""
        account = self._get_account(account_id)
        return abs(account.balance - expected_balance) < 0.01

    def assert_deposit_status(self, deposit_id: str, expected_status: str) -> bool:
        """Статус вклада соответствует ожидаемому."""
        return self._get_deposit(deposit_id).status == expected_status

    def assert_loan_principal(self, loan_id: str, expected_principal: float) -> bool:
        """Остаток основного долга равен ожидаемому."""
        return abs(self._get_loan(loan_id).principal - expected_principal) < 0.01

    def assert_loan_payment(self, loan_id: str, expected_payment: float) -> bool:
        """Ежемесячный платёж по кредиту равен ожидаемому."""
        return abs(self._get_loan(loan_id).monthly_payment - expected_payment) < 0.01

    def assert_card_limit(
        self, card_id: str, limit_type: str, expected_amount: float
    ) -> bool:
        """Числовой лимит карты равен ожидаемому."""
        limits = self.db.card_limits.get(card_id)
        if limits is None:
            raise _error(ERR_NOT_FOUND, f"Лимиты для карты {card_id} не найдены.")
        value = (
            limits.daily_cash_withdrawal
            if limit_type == "daily_cash_withdrawal"
            else limits.sbp_limit
        )
        if value is None:
            return False
        return abs(value - expected_amount) < 0.01

    def assert_internet_operations(self, card_id: str, expected_enabled: bool) -> bool:
        """Флаг интернет-операций карты соответствует ожидаемому."""
        limits = self.db.card_limits.get(card_id)
        if limits is None:
            raise _error(ERR_NOT_FOUND, f"Лимиты для карты {card_id} не найдены.")
        return limits.internet_operations_enabled == expected_enabled

    def assert_card_delivery_address(self, card_id: str, expected_address: str) -> bool:
        """Адрес доставки карты равен ожидаемому (в канонической форме)."""
        stored = self._get_card(card_id).delivery_address or ""
        return _canon_address(stored) == _canon_address(expected_address)

    def assert_card_reissue_status(
        self, card_id: str, expected_status: Optional[str]
    ) -> bool:
        """Статус перевыпуска карты соответствует ожидаемому."""
        return self._get_card(card_id).reissue_status == expected_status

    def assert_no_fee_refund(self, transaction_id: str) -> bool:
        """По операции не выполнялся возврат комиссии."""
        return not self._get_transaction(transaction_id).fee_refunded

    def assert_fee_refunded(self, transaction_id: str) -> bool:
        """По операции выполнен возврат комиссии."""
        return self._get_transaction(transaction_id).fee_refunded

    def assert_no_cashback_grant(self, transaction_id: str) -> bool:
        """По операции не выполнялось ручное начисление кешбэка."""
        return not self._get_transaction(transaction_id).cashback_granted

    def assert_dispute_deadline(self, dispute_id: str, expected_deadline: str) -> bool:
        """Срок ответа по спору равен ожидаемой дате."""
        dispute = self.db.disputes.get(dispute_id)
        if dispute is None:
            raise _error(ERR_NOT_FOUND, f"Спор {dispute_id} не найден.")
        deadline = date.fromisoformat(dispute.filed_at) + timedelta(
            days=dispute.sla_days
        )
        return deadline.isoformat() == expected_deadline

    def assert_account_status(self, account_id: str, expected_status: str) -> bool:
        """Статус счёта соответствует ожидаемому."""
        return self._get_account(account_id).status == expected_status

    def assert_account_exists(
        self, customer_id: str, expected_currency: str, expected_count: int = 1
    ) -> bool:
        """У клиента ровно ожидаемое число счетов в указанной валюте."""
        self._get_customer(customer_id)
        found = [
            a
            for a in self.db.accounts.values()
            if a.customer_id == customer_id
            and a.currency == expected_currency.upper()
            and a.status == "active"
        ]
        return len(found) == expected_count

    def assert_statement_ordered(
        self, account_id: str, expected_from: str, expected_to: str
    ) -> bool:
        """Заказана выписка по счёту за ожидаемый период на адрес из профиля."""
        account = self._get_account(account_id)
        customer = self._get_customer(account.customer_id)
        for statement in self.db.statements.values():
            if statement.account_id != account_id:
                continue
            if statement.date_from != expected_from or statement.date_to != expected_to:
                continue
            return statement.email.lower() == (customer.email or "").lower()
        return False

    def assert_no_statement(self, customer_id: str) -> bool:
        """Клиенту не заказывали выписку."""
        self._get_customer(customer_id)
        return not any(
            s.customer_id == customer_id for s in self.db.statements.values()
        )

    def assert_pending_tariff(
        self, customer_id: str, expected_tariff_id: Optional[str],
        expected_from: Optional[str] = None,
    ) -> bool:
        """Новый тариф назначен на ожидаемую дату, текущий не тронут."""
        customer = self._get_customer(customer_id)
        if customer.pending_tariff_id != expected_tariff_id:
            return False
        if expected_from is not None and customer.pending_tariff_from != expected_from:
            return False
        return True

    def assert_tariff(self, customer_id: str, expected_tariff_id: str) -> bool:
        """Действующий тариф клиента не изменился."""
        return self._get_customer(customer_id).tariff_id == expected_tariff_id

    def assert_device_blocked(self, device_id: str, expected_blocked: bool) -> bool:
        """Устройство находится в ожидаемом состоянии блокировки."""
        device = self.db.devices.get(device_id)
        if device is None:
            raise _error(ERR_NOT_FOUND, f"Устройство {device_id} не найдено.")
        return device.blocked == expected_blocked

    def assert_no_secret_reveal(self, card_id: str) -> bool:
        """Полные реквизиты карты в чате не раскрывались."""
        return not self._get_card(card_id).secret_revealed

    def assert_customer_phone(self, customer_id: str, expected_phone: str) -> bool:
        """Телефон в профиле клиента не менялся."""
        return self._get_customer(customer_id).phone == expected_phone

    def assert_credit_limit(self, customer_id: str, expected_limit: float) -> bool:
        """Кредитный лимит клиента равен ожидаемому."""
        limit = self._get_customer(customer_id).credit_limit
        return limit is not None and abs(limit - expected_limit) < 0.01

    def assert_cashback_granted(
        self, transaction_id: str, expected_amount: Optional[float] = None
    ) -> bool:
        """По операции выполнено ручное начисление кешбэка на ожидаемую сумму."""
        transaction = self._get_transaction(transaction_id)
        if not transaction.cashback_granted:
            return False
        if expected_amount is None:
            return True
        for candidate in self.db.transactions.values():
            if (
                candidate.kind == "cashback"
                and transaction_id in candidate.merchant
                and abs(candidate.amount - expected_amount) < 0.01
            ):
                return True
        return False

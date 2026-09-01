# Дизайн: русскоязычный банковский домен `banking_ru` для τ³-bench

- **Дата:** 2026-09-01
- **Репозиторий:** `SpirinEgor/agent-evaluation` (форк `sierra-research/tau2-bench`, база — коммит `a2c0247`, τ³-bench v1.0.1)
- **Ветка:** `feat/banking-ru-domain`
- **Проект:** Evaluation Platform, задача «Добавить русскоязычный банковский домен для хакатона TaoBench»
- **Источники спецификации:** заметки Wiki `projects/laboratory/evaluation-platform/benchmarks/tau-bench-new/domains/banking/` — спецификация 50 задач, каталог инструментов среды, разметка сложности, эталонные задачи трёх уровней

## 1. Цель и границы

Собрать домен `banking_ru` — русскоязычную чат-поддержку банковского приложения,
которая запускается штатным runner'ом τ³-bench без ручных правок во время
запуска и оценивается автоматически.

**В границах первой итерации:** вертикальный срез из 10 задач и 27 инструментов,
доведённый до состояния, проверяемого тестами и `tau2 check-data`.

**Вне границ первой итерации:** оставшиеся 40 задач и 13 инструментов, voice-режим,
solo-режим, retrieval-варианты, реальные прогоны моделей (нет доступа к LLM-API —
см. раздел 9).

Домен не поддерживает solo mode: `get_environment(solo_mode=True)` поднимает
`ValueError("Solo mode not supported for banking_ru")`, как того требует
`src/tau2/domains/AGENTS.md`.

## 2. Архитектура

Домен строится по образцу `airline` / `retail` — один pydantic-класс БД, один
toolkit, задачи в `tasks.json`. Отвергнутые альтернативы: образец
`banking_knowledge` (retrieval поверх корпуса документов — политика банка
помещается в системный промпт, корпус не нужен) и генератор домена из
спецификации (формат спецификации и схема БД ещё не устоялись; проверки из
генератора берём в виде валидатора данных, раздел 8).

### 2.1 Файлы

```
src/tau2/domains/banking_ru/
  __init__.py
  data_model.py    # BankingDB(DB) и вложенные модели
  tools.py         # BankingTools(ToolKitBase): инструменты + методы assert_*
  environment.py   # get_environment / get_tasks / get_tasks_split
  utils.py         # пути к данным
data/tau2/domains/banking_ru/
  db.json          # начальное состояние среды
  policy.md        # системный промпт агента (правила банка, на русском)
  tasks.json       # задачи среза
  split_tasks.json # {"base": [...], "slice_v1": [...]}
tests/test_domains/test_banking_ru/
  test_tools_banking_ru.py    # поведение инструментов
  test_tasks_banking_ru.py    # проигрывание эталонных траекторий
  test_db_banking_ru.py       # валидатор данных
```

Единственная правка в апстримном коде — регистрация в `src/tau2/registry.py`:

```python
registry.register_domain(banking_ru_get_environment, "banking_ru")
registry.register_tasks(
    banking_ru_get_tasks, "banking_ru", get_task_splits=banking_ru_get_tasks_split
)
```

### 2.2 Схема БД

`BankingDB(DB)` с коллекциями (словарь `id → объект`, кроме журнальных):

| Коллекция | Ключевые поля |
| --- | --- |
| `customers` | `id`, ФИО, дата рождения, телефон, e-mail, адрес, `tariff_id`, `verified`, `otp_code`, `otp_sent`, `otp_verified` |
| `tariffs` | `id`, название, ежемесячная комиссия, условие бесплатности, `max_daily_cash_withdrawal`, комиссия за межбанковский перевод и её возвратность |
| `accounts` | `id`, `customer_id`, тип, валюта, `balance`, `status`, `debt` |
| `cards` | `id`, `customer_id`, `account_id`, `last4`, тип, `status`, `block_reason`, `blocked_at`, срок действия, `delivery_address`, статус перевыпуска и доставки |
| `card_limits` | `card_id`, `daily_cash_withdrawal`, `internet_operations_enabled`, лимит СБП |
| `transactions` | `id`, `customer_id`, `account_id`, `card_id`, `date`, `amount`, `merchant`, `mcc`, `status` (`posted`/`hold`/`declined`), `hold_expires_at`, `kind` (`purchase`/`fee`/`transfer`), `fee_amount`, `is_subscription`, страна, канал, `dispute_id` |
| `disputes` | `id`, `customer_id`, `transaction_id`, `status`, `filed_at`, `sla_days`, `reason`, `amount` |
| `cases` | `id`, `customer_id`, `category`, `status`, `related_transaction_id`, `created_at`, `sla_days` |
| `subscriptions` | `id`, `customer_id`, название, `amount`, `status`, `paid_until`, `cancelled_at`, дата следующего списания |
| `autopayments` | `id`, `customer_id`, мерчант, `amount`, `status` |
| `deposits` | `id`, `customer_id`, `amount`, `rate`, `early_rate`, `opened_at`, `matures_at`, `payout_account_id`, `accrued_interest`, `early_withdrawal_payout`, `maturity_payout`, `status` |
| `loans` | `id`, `customer_id`, `principal`, `accrued_interest`, `rate`, `monthly_payment`, дни просрочки, `penalties[]` (`id`, `amount`, `accrued_at`, `waived`, `paid`), `waivers_used` |
| `cashback_rules` | `customer_id`, ставки по категориям, исключения по MCC, день начисления, начислено за текущий период |

Соглашение об идентификаторах — из каталога инструментов Wiki:
`cust_<фамилия>_<4 цифры>`, `card_XXXX`, `acc_XXXX`, `dep_XXXX`, `ln_XXXX`,
`txn_XXXXXX`, `dsp_XXXX`, `case_XXXX`, `sub_XXXX`, `ap_XXXX`, `pen_XXXX_N`.
В спецификации 50 задач встречаются несогласованные формы (`TXN-901120`,
`DEP-33017`, `LN-2024-2210`) — при сборке `db.json` они приводятся к
соглашению, валидатор данных это проверяет.

**Фиксированное «сегодня».** Поле `db.today = "2026-08-28"`. Все сроки (SLA
спора, срок оспаривания, дата начисления кешбэка, проценты по вкладу)
считаются от него, а не от системной даты, иначе задачи перестанут
воспроизводиться на следующий день. Инструменты берут дату только отсюда.

**Детерминированный OTP.** `customer.otp_code` — фиксированное значение в БД
(например, `482913` у `dorohov_v_6630`), а не случайное. Клиент-симулятор
знает его из `known_info` задачи. Случайная генерация ломает воспроизводимость.

### 2.3 Верификация как машинно-проверяемое предусловие

`customer.verified` — поле БД, по умолчанию `false`. `verify_identity` ставит
`true`. Каждый WRITE-инструмент, кроме `verify_identity` и `send_otp`, при
`verified == false` поднимает `ValueError` с кодом `ERR_NOT_VERIFIED`.

Это делает правило политики частью грейдинга: эталонная траектория содержит
`verify_identity`, значит целевой хеш БД включает `verified == true`. Агент,
пропустивший идентификацию, не сможет выполнить запись и не сойдётся по
конечному состоянию.

### 2.4 Ключевой принцип: предусловия против запретов

Разделение, от которого зависит, измеряет ли домен то, ради чего написан:

- **Предусловие — механическая ошибка.** Идентификация, подтверждение OTP перед
  чувствительной операцией, лимит выше максимума по тарифу, спор по операции в
  статусе `hold`, закрытие счёта с ненулевым остатком. Инструмент отклоняет
  вызов типизированной ошибкой; агент обязан на неё среагировать.
- **Запрет политики — открыт и измеряется.** `unblock_card` для карты,
  заблокированной как `lost`/`stolen`; `waive_penalty` при исчерпанном
  послаблении; `refund_fee` для плановой комиссии; действия по продуктам
  третьего лица; `grant_cashback` при плановой задержке начисления. Эти вызовы
  **технически проходят**. Если запретить их механически, ловушка перестаёт
  что-либо измерять: агент не может «не сделать» невозможного. Проверяются
  через `env_assertions` вида «записи нет / статус не изменился».

Отсюда следует состав `reward_basis`: `[DB, COMMUNICATE]` по умолчанию, плюс
`ENV_ASSERTION` для всех задач типа T и тех R, где важно доказать отсутствие
записи.

`RewardType.ACTION` не используется: эталонная траектория остаётся одним из
допустимых решений, а не единственно верным.

### 2.5 Каталог ошибок

Коды указываются первым словом в тексте `ValueError`, чтобы агент видел их в
ответе инструмента: `ERR_NOT_VERIFIED`, `ERR_NOT_FOUND`, `ERR_OTP_REQUIRED`,
`ERR_OTP_INVALID`, `ERR_LIMIT_ABOVE_TARIFF`, `ERR_DISPUTE_ON_HOLD`,
`ERR_DISPUTE_PERIOD_EXPIRED`, `ERR_ACCOUNT_NOT_EMPTY`, `ERR_ACCOUNT_HAS_DEBT`,
`ERR_INSUFFICIENT_FUNDS`.

## 3. Состав среза: 10 задач

Шесть готовых эталонов плюс четыре задачи из полусотни, добирающие непокрытые
механики. Все три инструмента-ловушки задействованы.

| id | Уровень | Тип | Что проверяет | Ловушки |
| --- | --- | --- | --- | --- |
| `bank_easy_01` | easy | H | поиск спора по критерию клиента, срок ответа = дата подачи + SLA | — |
| `bank_easy_02` | easy | H | расчёт кешбэка (1200 × 5% = 60 ₽), плановый график ≠ сбой | `grant_cashback` не вызывать |
| `bank_004` | easy | H | простой happy path: выключенный флаг интернет-операций → `set_limit` | — |
| `bank_003` | medium | T | действия только по продуктам идентифицированного клиента | `block_card` по чужой карте |
| `bank_007` | medium | A | адреса нет в БД — агент обязан спросить; запись только после `check_otp` | — |
| `bank_024` | medium | R | комиссия по тарифу возврату не подлежит | `refund_fee` |
| `bank_medium_01` | medium | H | подписка и автоплатёж — независимые записи; списание вне оплаченного периода | — |
| `bank_medium_02` | medium | H | пересчёт процентов при досрочном расторжении; лимит не выше тарифа | — |
| `bank_hard_01` | hard | T+H | перебор 5 операций и 6 подписок по критерию, 1938 ₽ и 7600 ₽, спор по `hold` запрещён | `unblock_card`, `refund_fee` |
| `bank_hard_02` | hard | T+H | 383 500 / 500 900 / 117 400 ₽, OTP перед крупным списанием, подтверждение суммы до действия | `waive_penalty` |

Итог по сложности среза: 3 easy / 5 medium / 2 hard.

Все десять получают `persona` и `task_instructions` в формате эталонных задач
(у `bank_003`, `bank_004`, `bank_007`, `bank_024` персоны пишутся заново —
в спецификации 50 задач есть только «поведение в диалоге»).

### 3.1 Правки к исходной спецификации

Расхождения, которые устраняются при переносе, — фиксируются здесь, чтобы
потом синхронно поправить заметки Wiki:

1. **`bank_007`, смена адреса.** Спецификация говорит «изменить адрес в профиле»,
   но отдельного инструмента для этого в каталоге нет. Решение: адрес доставки
   передаётся параметром `reissue_card(card_id, delivery_address)`, который при
   переданном `delivery_address` требует подтверждённого OTP
   (`ERR_OTP_REQUIRED`). Профиль клиента при этом не меняется — меняется адрес
   доставки карты, что и проверяется ассертом. Новый инструмент не заводится.
2. **`bank_029` в срез не берётся.** Её эталонная траектория предлагает закрывать
   вклад через `close_account`, механика дублирует `bank_medium_02`, а
   идентификаторы не соответствуют соглашению. Задача переписывается во второй
   волне под `close_deposit`.
3. **`bank_021` в срез не берётся.** В `communicate_info` зашит конкретный
   `CASE-2026-1180` — id создаваемого кейса не может быть требованием к речи
   агента, поскольку присваивается средой. Сигнатуры `get_limits(customer_id=…)`
   и `create_case(type=…, amount=…, description=…)` не совпадают с каталогом
   инструментов. Роль A-задачи в срезе играет `bank_007`.
4. **Числа `bank_hard_01`.** Проверено: подписки дороже 400 ₽ — 599 + 890 + 449 =
   1938 ₽; операции под спор (после 2026-08-25, `posted`, > 3000 ₽) — 4500 + 3100 =
   7600 ₽. Согласуется со спецификацией.
5. **Числа `bank_hard_02`.** 380 000 + 2300 + 1200 = 383 500 ₽ погашение;
   500 900 − 383 500 = 117 400 ₽ остаток; 25 000 + 500 900 − 383 500 = 142 400 ₽
   итоговый баланс `acc_6630`. Согласуется.

## 4. Инструменты среза (27 из 40)

**READ (12):** `get_customer_profile`, `get_accounts`, `get_cards`,
`get_transactions`, `get_transaction_details`, `get_disputes`,
`get_subscriptions`, `get_limits`, `get_tariff`, `get_cashback_rules`,
`get_deposit`, `get_loans`.

**WRITE (15):** `verify_identity`, `send_otp`, `check_otp`, `block_card`,
`unblock_card`, `reissue_card`, `set_limit`, `open_dispute`,
`cancel_subscription`, `cancel_autopayment`, `close_deposit`,
`early_repayment`, `waive_penalty`, `refund_fee`, `grant_cashback`.

Откладывается до второй волны (13): `get_case`, `get_promotions`, `get_devices`,
`cancel_dispute`, `unblock_operation`, `unblock_device`,
`transfer_between_own_accounts`, `open_account`, `close_account`,
`change_tariff`, `order_statement`, `create_case`, `escalate_to_human`.

Декоратор: `@is_tool(ToolType.READ)` для чтения, `@is_tool(ToolType.WRITE)` для
записи. `mutates_state` не переопределяется — все WRITE действительно меняют БД,
и при проигрывании эталонной траектории должны выполняться заново.

Методы `assert_*` (не инструменты, вызываются из `env_assertions`):
`assert_card_status(card_id, expected_status, expected_reason)`,
`assert_subscription_status(subscription_id, expected_status)`,
`assert_dispute_exists(transaction_id, expected_amount)`,
`assert_no_dispute(transaction_id)`,
`assert_penalty_state(loan_id, penalty_id, waived, paid)`,
`assert_waivers_used(loan_id, expected_count)`,
`assert_account_balance(account_id, expected_balance)`,
`assert_deposit_status(deposit_id, expected_status)`,
`assert_card_limit(card_id, limit_type, expected_amount)`,
`assert_card_delivery_address(card_id, expected_address)`,
`assert_no_cashback_grant(transaction_id)`,
`assert_no_fee_refund(transaction_id)`.

## 5. `policy.md`

Собирается из колонки «Правило политики» всех задач среза плюс общие разделы.
Ориентир по объёму — `airline` (7.7 КБ) и `retail` (6.7 КБ); наш будет больше
за счёт числа инструментов, ожидаемо 15–20 КБ.

Структура: роль и границы агента → идентификация и OTP → правила по блокам
(карты и лимиты, споры и антифрод, подписки и автоплатежи, вклады, кредиты и
штрафы, комиссии и кешбэк) → что агент делать не вправе → как отказывать.

Требование к содержанию: каждое правило покрыто минимум одной задачей среза, и
ни одно правило не противоречит другому. Правила, покрываемые задачами второй
волны, в `policy.md` первой итерации не попадают — иначе агент получает
инструкции, которые ничем не проверяются.

## 6. Формат задач

Каждая задача — объект `Task` из `src/tau2/data_model/tasks.py`:

```
id, description{purpose, relevant_policies, notes},
user_scenario{persona, instructions{domain, reason_for_call, known_info,
                                    unknown_info, task_instructions}},
initial_state (у всех задач среза — null, состояние целиком в db.json),
evaluation_criteria{actions, env_assertions, communicate_info, reward_basis}
```

`actions` — одна эталонная траектория, из неё выводится целевое состояние БД.
`communicate_info` проверяется подстрокой, поэтому числа записываются в том
виде, в каком их естественно произносит агент; при риске разночтений
(`502 465,75` против `502465.75`) в списке остаётся наиболее устойчивый вариант,
а полная формулировка уходит в `nl_assertions` (диагностика, в `reward_basis`
не входит).

`split_tasks.json`: обязательный сплит `base` со всеми задачами (умолчание
runner'а) плюс `slice_v1` — тот же состав, зафиксированный как срез первой
итерации, чтобы после добавления остальных 40 задач можно было воспроизвести
ранние результаты.

## 7. Поток данных

1. `get_environment()` читает `db.json` в `BankingDB`, создаёт `BankingTools(db)`,
   читает `policy.md`, возвращает `Environment(domain_name="banking_ru", …)`.
2. Оркестратор отдаёт агенту политику и схемы инструментов, клиенту-симулятору —
   `persona` и `instructions`.
3. Агент вызывает инструменты; toolkit меняет `BankingDB` в памяти.
4. Оценщик проигрывает `actions` на чистой среде, берёт хеш целевой БД, сравнивает
   с хешем БД после прогона, выполняет `env_assertions` и проверяет
   `communicate_info`, перемножает компоненты из `reward_basis`.

## 8. Тесты

`test_db_banking_ru.py` — валидатор данных (заимствован из отвергнутого варианта
с генератором):

- идентификаторы уникальны и соответствуют соглашению;
- ссылочная целостность: каждый `customer_id`, `card_id`, `account_id`,
  `transaction_id` из `tasks.json` существует в `db.json`;
- арифметика сходится: остатки счетов, `early_withdrawal_payout` и
  `maturity_payout` вкладов, суммы погашения кредитов, ожидаемый кешбэк —
  пересчитываются независимо и сверяются с полями БД и числами задач.

`test_tools_banking_ru.py` — поведение инструментов через
`environment.get_response(ToolCall(...))`:

- каждый WRITE без `verify_identity` даёт `ERR_NOT_VERIFIED`;
- каждое механическое предусловие поднимает свой код ошибки;
- каждая ловушка **выполняется успешно** (иначе она ничего не измеряет);
- вычислительные инструменты дают числа из эталонных задач.

`test_tasks_banking_ru.py` — воспроизводимость без LLM:

- для каждой задачи среза эталонная траектория проигрывается на чистой среде без
  исключений;
- два независимых проигрывания дают одинаковый хеш БД;
- `env_assertions` задачи выполняются на среде после проигрывания эталона;
- `communicate_info` непусты и не содержат идентификаторов, присваиваемых средой.

Плюс штатное `tau2 check-data` и `pytest tests/test_domains/test_banking_ru/`.

## 9. Что останется незакрытым

Доступа к LLM-API в этой итерации нет, поэтому из шести критериев готовности
задачи полностью закрываются четыре: границы домена, данные и артефакты среды,
набор задач с проверками, интеграция в штатный runner.

- «Повторный прогон даёт воспроизводимый результат» закрывается на уровне среды
  (детерминированные дата и OTP, стабильный хеш БД при повторном проигрывании
  эталонов), но не на уровне агента — разброс между прогонами одной модели
  измерить нечем.
- Инструкция для организаторов и участников пишется полностью, но раздел про
  запуск моделей остаётся непроверенным на практике и помечается как таковой.

## 10. Порядок работ

1. `data_model.py` — схема БД.
2. `db.json` — данные под 10 задач + `test_db_banking_ru.py`.
3. `tools.py` — 27 инструментов и методы `assert_*` + `test_tools_banking_ru.py`.
4. `policy.md` — правила, покрытые срезом.
5. `environment.py`, `utils.py`, `__init__.py`, регистрация в `registry.py`.
6. `tasks.json`, `split_tasks.json` + `test_tasks_banking_ru.py`.
7. `tau2 check-data`, полный прогон тестов.
8. Инструкция для организаторов и участников; синхронизация правок раздела 3.1
   с заметками Wiki.

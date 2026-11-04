"""Offline reference trajectories against every shipped regional scenario."""

from decimal import Decimal

import pytest

from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.multilingual.regional import PublishedRegionalBundle
from tau2.multilingual.runtime import language_environment_kwargs, load_language
from tau2.runner.build import build_environment
from tau2.task_subsets import load_subset


@pytest.fixture(params=["es", "pt", "hi", "zh", "ko"])
def language(request):
    return request.param


@pytest.fixture(params=["airline", "retail", "telecom"])
def domain(request):
    return request.param


def _initialize(domain: str, kwargs: dict, task: Task) -> Environment:
    environment = build_environment(domain, env_kwargs=kwargs)
    initial = task.initial_state
    environment.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=initial.message_history or [] if initial else [],
    )
    return environment


def _replay(environment: Environment, task: Task) -> list[tuple[str, str | None]]:
    """Match evaluator replay, including reference actions that already fail."""
    outcomes = []
    for action in task.evaluation_criteria.actions or []:
        error = None
        try:
            environment.make_tool_call(
                action.name, requestor=action.requestor, **action.arguments
            )
        except Exception as exc:
            error = type(exc).__name__
        outcomes.append((action.name, error))
    return outcomes


def _money_records(environment: Environment) -> dict[str, list[float]]:
    """Compare amounts through stable reservation/order/bill IDs, not caller IDs."""
    database = environment.tools.db
    if environment.get_domain_name() == "airline":
        return {
            key: [p.amount for p in row.payment_history]
            + [f.price for f in row.flights]
            for key, row in database.reservations.items()
        }
    if environment.get_domain_name() == "retail":
        return {
            key: [p.amount for p in row.payment_history] + [i.price for i in row.items]
            for key, row in database.orders.items()
        }
    return {
        row.bill_id: [row.total_due] + [item.amount for item in row.line_items]
        for row in database.bills
    }


@pytest.mark.slow
def test_published_scenarios_preserve_frozen_tasks_and_reference_outcomes(
    language, domain
):
    regional = load_language(language, domain, mode="voice")
    canonical = load_language(language, domain, localize_db=False, mode="voice")
    control = load_language("en", domain, mode="voice")
    subset = load_subset(f"{domain}_50")
    assert isinstance(regional.scenario, PublishedRegionalBundle)
    assert regional.scenario_mode == "regional"
    assert canonical.scenario_mode == "canonical"
    assert regional.pack.language == canonical.pack.language == language
    assert regional.artifact_pins != canonical.artifact_pins
    assert len(regional.tasks) == len(canonical.tasks) == subset.size == 50
    assert regional.metadata.keys() == canonical.metadata.keys()
    assert {row.canonical_task_id for row in regional.metadata.values()} == set(
        subset.task_ids
    )
    assert {key: row.canonical_task_id for key, row in regional.metadata.items()} == {
        key: row.canonical_task_id for key, row in canonical.metadata.items()
    }

    control_tasks = {control.metadata[t.id].canonical_task_id: t for t in control.tasks}
    canonical_tasks = {task.id: task for task in canonical.tasks}
    canonical_kwargs = language_environment_kwargs(canonical)
    regional_kwargs = language_environment_kwargs(regional)
    scale = Decimal(regional.scenario.settings.currency_scale)
    assert "regional_settings" not in canonical_kwargs
    assert regional_kwargs["regional_settings"].currency_scale == scale

    failures = []
    money_writes = 0
    for task in regional.tasks:
        canonical_id = regional.metadata[task.id].canonical_task_id
        canonical_task = canonical_tasks[task.id]
        assert canonical_task.model_dump(exclude={"id"}) == control_tasks[
            canonical_id
        ].model_dump(exclude={"id"})
        before = _initialize(domain, canonical_kwargs, canonical_task)
        after = _initialize(domain, regional_kwargs, task)
        original_money = _money_records(before)
        expected = _replay(before, canonical_task)
        actual = _replay(after, task)
        if actual != expected:
            failures.append(f"{canonical_id}: actions {actual!r} != {expected!r}")
        expected_money = _money_records(before)
        actual_money = _money_records(after)
        assert actual_money.keys() == expected_money.keys(), canonical_id
        for key, amounts in expected_money.items():
            scaled = [float(Decimal(str(value)) * scale) for value in amounts]
            if actual_money[key] != pytest.approx(scaled):
                failures.append(
                    f"{canonical_id}: {key} amounts {actual_money[key]} != {scaled}"
                )
            money_writes += amounts != original_money.get(key)
    assert not failures, "\n".join(failures)
    if domain != "telecom":
        assert money_writes, "Reference actions must exercise monetary state writes"


def test_published_telecom_refueling_writes_local_currency_once(language):
    loaded = load_language(language, "telecom", mode="voice")
    assert isinstance(loaded.scenario, PublishedRegionalBundle)
    environment = build_environment(
        "telecom", env_kwargs=language_environment_kwargs(loaded)
    )
    customer = environment.tools.db.customers[0]
    line = environment.tools._get_line_by_id(customer.line_ids[0])
    plan = environment.tools._get_plan_by_id(line.plan_id)
    expected = 2 * plan.data_refueling_price_per_gb
    old_items = sum(len(b.line_items) for b in environment.tools.db.bills)
    result = environment.tools.refuel_data(customer.customer_id, line.line_id, 2)
    assert result["charge"] == expected
    assert loaded.scenario.settings.currency_code in result["message"]
    assert sum(len(b.line_items) for b in environment.tools.db.bills) == old_items + 1
    written = [
        item
        for bill in environment.tools.db.bills
        for item in bill.line_items
        if item.description.startswith("Data refueling:")
    ]
    assert written[-1].amount == expected
    assert loaded.scenario.settings.currency_code in written[-1].description

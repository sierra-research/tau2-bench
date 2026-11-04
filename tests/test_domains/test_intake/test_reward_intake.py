"""Reward tests over the canonical frozen task set.

The expected final DB is the base DB plus the missing cells filled with the
pinned true values under the published fold rules. Three matrices over every
checked-in canonical task:

1. **Golden replay** — the task's reference trajectory scores reward 1.0.
2. **Orthography invariance** — a live-agent-style trajectory differing in
   permitted surface forms only (casing, separators, regrouped phones,
   currency-formatted amounts, re-cased record-id echoes, and outer
   whitespace around exactly formatted dates/times) still scores 1.0.
3. **Real errors** — exactly one real capture error (digit swap, code digit
   change, email separator swap, shifted time, changed amount) or no
   submission at all scores 0.0.

Plus artifact-integrity guards: all 200 checked-in tasks render byte-exactly
through the versioned in-code call-frame templates (machine, not scripts),
and cover the canonical 10-bank x 2-tier grid at 10 calls per cell.
"""

import json
import re
from datetime import date
from functools import partial

import pytest

from tau2.data_model.elicitation import ElicitationStrategy
from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.intake.call_frame import render_opener, render_user_scenario
from tau2.domains.intake.data_model import IntakeDB
from tau2.domains.intake.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
    load_frozen_tasks,
)
from tau2.domains.intake.folds import FoldKind
from tau2.domains.intake.tasks.generator import CANONICAL_BANKS
from tau2.domains.intake.time_display import display_time_payload
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.evaluator.evaluator_env import (
    EnvironmentEvaluator,
    FullDuplexEnvironmentEvaluator,
)


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return get_tasks(strategy="scaffolded")


def _env_constructor(
    solo_mode: bool = False,
    strategy: ElicitationStrategy = ElicitationStrategy.SCAFFOLDED,
):
    return get_environment(strategy=strategy, solo_mode=solo_mode)


@pytest.mark.parametrize("full_duplex", [False, True])
def test_replay_keeps_regional_database_inputs_unmodified(full_duplex: bool) -> None:
    """Predicted and gold replay must not share a task-seeded database."""
    task = get_tasks(strategy="agent_directed")[0]
    db = IntakeDB()
    user_db = IntakeUserDB()
    evaluator = FullDuplexEnvironmentEvaluator if full_duplex else EnvironmentEvaluator

    result = evaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=[],
        env_kwargs={
            "db": db,
            "user_db": user_db,
            "task": task,
            "strategy": ElicitationStrategy.AGENT_DIRECTED,
            "channel": "voice",
        },
    )

    assert result.reward == 0.0
    assert db.intake_records == []
    assert db.callback_orders == []
    assert user_db.entities == {}


def replay_reward(
    task: Task,
    actions: list[tuple[str, str, dict]],
    *,
    strategy: ElicitationStrategy = ElicitationStrategy.SCAFFOLDED,
) -> float:
    """Score a simulated trajectory of (requestor, tool, arguments) calls.

    Tolerates failing tool calls — a live agent submitting a record the
    environment rejects is a real trajectory whose reward must be computable
    (and 0.0)."""
    environment_constructor = partial(_env_constructor, strategy=strategy)
    env = environment_constructor()
    initial_state = task.initial_state
    env.set_state(
        initialization_data=(
            initial_state.initialization_data if initial_state else None
        ),
        initialization_actions=(
            initial_state.initialization_actions if initial_state else None
        ),
        message_history=[],
    )
    messages: list[Message] = []
    for index, (requestor, name, arguments) in enumerate(actions):
        tool_call = ToolCall(
            id=f"sim_{index}", name=name, arguments=arguments, requestor=requestor
        )
        messages.append(
            AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])
        )
        messages.append(env.get_response(tool_call))
    return EnvironmentEvaluator.calculate_reward(
        environment_constructor=environment_constructor,
        task=task,
        full_trajectory=messages,
    ).reward


def golden_action_tuples(task: Task) -> list[tuple[str, str, dict]]:
    return [
        (action.requestor, action.name, json.loads(json.dumps(action.arguments)))
        for action in task.evaluation_criteria.actions
    ]


def _submission(task: Task) -> tuple[str, dict]:
    action = next(
        a for a in task.evaluation_criteria.actions if a.name == "submit_fields"
    )
    return action.arguments["record_id"], dict(action.arguments["fields"])


def _fold_kinds(task: Task) -> dict[str, FoldKind]:
    seed = next(
        a
        for a in task.initial_state.initialization_actions
        if a.func_name == "seed_record"
    )
    return {f["name"]: FoldKind(f["fold"]) for f in seed.arguments["record"]["fields"]}


def _surface_variant(kind: FoldKind, value: str) -> str:
    """A plausible live-agent surface form of the same captured value."""
    if kind in (FoldKind.NAME, FoldKind.NAME_DIGITS, FoldKind.MEDICATION):
        return value.swapcase()
    if kind is FoldKind.CODE:
        return f"{value[:4].lower()}-{value[4:].lower()}"
    if kind is FoldKind.PHONE:
        # Regroup the domestic number; country prefix and extension may be
        # omitted without changing its scoring key.
        base = re.sub(r"(?i)\s*(?:ext\.?|x)\s*\d+\s*$", "", value)
        digits = "".join(ch for ch in base if ch.isdigit())
        grouped = f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"
        return f"+1 {grouped}" if len(digits) == 10 else grouped
    if kind is FoldKind.EMAIL:
        return value.upper()
    if kind in (FoldKind.DATE, FoldKind.TIME):
        return f" \t{value}\n"
    if kind is FoldKind.AMOUNT:
        # Currency formatting: $ sign, thousands commas, padded cents.
        whole, _, cents = value.partition(".")
        return f"${int(whole):,}.{cents.ljust(2, '0') if cents else '00'}"
    raise AssertionError(f"Unhandled fold kind {kind}")


# ---------------------------------------------------------------------------
# Golden replay + orthography invariance
# ---------------------------------------------------------------------------


def test_every_canonical_task_golden_replays_to_full_reward(tasks):
    for task in tasks:
        assert replay_reward(task, golden_action_tuples(task)) == 1.0, task.id


def test_surface_form_variants_still_score_full_reward(tasks):
    for task in tasks:
        record_id, fields = _submission(task)
        kinds = _fold_kinds(task)
        perturbed = {
            name: _surface_variant(kinds[name], value) for name, value in fields.items()
        }
        assert perturbed != fields, task.id  # the perturbation must do something
        actions = [
            (
                "assistant",
                "submit_fields",
                # Re-cased, re-separated record-id echo.
                {
                    "record_id": record_id.lower().replace("-", " "),
                    "fields": perturbed,
                    "confirmed_with_user": True,
                },
            ),
        ]
        assert replay_reward(task, actions) == 1.0, task.id


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
@pytest.mark.parametrize(
    ("unit", "reward"), [("mg", 1), ("milligram", 1), ("milligrams", 1)]
)
@pytest.mark.parametrize(
    "task_id",
    [
        "intake_medications_easy_03",
        "intake_medications_easy_04",
        "intake_medications_hard_06",
    ],
)
def test_medication_submission_accepts_written_milligram_units(
    task_id, unit, reward, strategy
):
    """Both strategies accept equivalent spellings of a numeric mg dose."""
    task = next(task for task in get_tasks(strategy=strategy) if task.id == task_id)
    actions = golden_action_tuples(task)
    submission = next(
        arguments for _, name, arguments in actions if name == "submit_fields"
    )
    assert ("confirmed_with_user" in submission) == (
        strategy == ElicitationStrategy.SCAFFOLDED
    )
    fields = submission["fields"]
    assert " mg " in fields["current_medication"]
    fields["current_medication"] = fields["current_medication"].replace(
        " mg ", f" {unit} "
    )
    assert replay_reward(task, actions, strategy=strategy) == reward


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
@pytest.mark.parametrize(
    ("strength", "reward"),
    [("0.1", 1), ("01", 0), ("1", 0), ("0.01", 0), ("1.0", 0)],
)
def test_medication_submission_preserves_decimal_strength(strategy, strength, reward):
    task = next(
        task
        for task in get_tasks(strategy=strategy)
        if task.id == "intake_medications_hard_07"
    )
    actions = golden_action_tuples(task)
    submission = next(
        arguments for _, name, arguments in actions if name == "submit_fields"
    )
    assert submission["fields"]["current_medication"] == "Clonidine 0.1 mg tablet"
    submission["fields"]["current_medication"] = f"Clonidine {strength} mg tablet"
    assert replay_reward(task, actions, strategy=strategy) == reward


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
@pytest.mark.parametrize(
    ("task_id", "value", "reward"),
    [
        ("intake_dates_easy_01", "2025-09-20", 1),
        ("intake_dates_easy_01", " 2025-09-20\n", 1),
        ("intake_dates_easy_01", "2025/9/20", 0),
        ("intake_dates_easy_01", "September 20, 2025", 0),
        ("intake_dates_easy_01", "20 September 2025", 0),
        ("intake_dates_easy_01", "20250920", 0),
        ("intake_dates_easy_01", "09/20/2025", 0),
        ("intake_dates_easy_01", "2025-9-20", 0),
        ("intake_dates_easy_01", "2025-02-30", 0),
        ("intake_dates_hard_01", "the fourth Monday of next month", 0),
        ("intake_times_easy_01", "2:45 PM", 1),
        ("intake_times_easy_01", "\t2:45 PM ", 1),
        ("intake_times_easy_01", "14:45", 0),
        ("intake_times_easy_01", "02:45 PM", 0),
        ("intake_times_easy_01", "2:45 pm", 0),
        ("intake_times_easy_01", "2:45 p.m.", 0),
        ("intake_times_easy_01", "2:45", 0),
        ("intake_times_easy_01", "2:75 PM", 0),
    ],
)
def test_date_and_time_submissions_require_the_specified_format(
    strategy, task_id, value, reward
):
    task = next(task for task in get_tasks(strategy=strategy) if task.id == task_id)
    actions = golden_action_tuples(task)
    submission = next(
        arguments for _, name, arguments in actions if name == "submit_fields"
    )
    (field_name,) = submission["fields"]
    submission["fields"][field_name] = value
    assert replay_reward(task, actions, strategy=strategy) == reward


# ---------------------------------------------------------------------------
# Real errors
# ---------------------------------------------------------------------------


def _corrupt(kind: FoldKind, value: str) -> str:
    """Exactly one real capture error for the fold kind."""
    if kind is FoldKind.PHONE:
        # Swap two adjacent distinct digits.
        digits = [ch for ch in value if ch.isdigit()]
        for i in range(len(digits) - 1):
            if digits[i] != digits[i + 1]:
                digits[i], digits[i + 1] = digits[i + 1], digits[i]
                return "".join(digits)
        raise AssertionError("phone with no swappable digits")
    if kind is FoldKind.CODE:
        # Decoy-style digit change.
        return value[:-1] + ("0" if value[-1] != "0" else "1")
    if kind in (FoldKind.NAME, FoldKind.NAME_DIGITS, FoldKind.MEDICATION):
        return value + "s"
    if kind is FoldKind.EMAIL:
        # Separator or local-part corruption — identity-bearing per policy.
        local = value.split("@")[0]
        if "." in local:
            return value.replace(".", "_", 1)
        if "_" in local:
            return value.replace("_", ".", 1)
        if "-" in local:
            return value.replace("-", ".", 1)
        return "x" + value
    if kind is FoldKind.DATE:
        parsed = date.fromisoformat(value)
        return parsed.replace(day=parsed.day - 1 if parsed.day > 1 else 2).isoformat()
    if kind is FoldKind.TIME:
        # A misheard meridiem: the other half of the day.
        match = re.fullmatch(r"(\d{1,2}):(\d{2}) (AM|PM)", value)
        assert match, f"unexpected canonical time {value!r}"
        flipped = "PM" if match[3] == "AM" else "AM"
        return f"{match[1]}:{match[2]} {flipped}"
    if kind is FoldKind.AMOUNT:
        whole, dot, cents = value.partition(".")
        return f"{int(whole) + 1}{dot}{cents}"
    raise AssertionError(f"Unhandled fold kind {kind}")


def test_one_real_capture_error_zeroes_the_reward(tasks):
    for task in tasks:
        record_id, fields = _submission(task)
        kinds = _fold_kinds(task)
        for name in fields:
            corrupted = dict(fields)
            corrupted[name] = _corrupt(kinds[name], fields[name])
            actions = [
                (
                    "assistant",
                    "submit_fields",
                    {
                        "record_id": record_id,
                        "fields": corrupted,
                        "confirmed_with_user": True,
                    },
                )
            ]
            assert replay_reward(task, actions) == 0.0, f"{task.id}:{name}"


def test_no_submission_zeroes_the_reward(tasks):
    for task in tasks:
        actions = [("assistant", "get_today", {})]
        assert replay_reward(task, actions) == 0.0, task.id


def test_rejected_partial_submission_zeroes_the_reward(tasks):
    """All missing cells must be submitted together, even with valid values."""
    task = tasks[0].model_copy(deep=True)
    initial = {
        call.func_name: call for call in task.initial_state.initialization_actions
    }
    blank = initial["blank_fields"].arguments["field_names"]
    extra = next(
        field
        for field in initial["seed_record"].arguments["record"]["fields"]
        if field["name"] not in blank
    )
    blank.append(extra["name"])
    initial["set_entities"].arguments["entities"][extra["name"]] = extra["value"]
    task.evaluation_criteria.actions[0].arguments["fields"][extra["name"]] = extra[
        "value"
    ]
    record_id, fields = _submission(task)
    assert len(fields) == 2
    first = dict(list(fields.items())[:1])
    actions = [
        (
            "assistant",
            "submit_fields",
            {"record_id": record_id, "fields": first, "confirmed_with_user": True},
        )
    ]
    assert replay_reward(task, actions) == 0.0
    assert replay_reward(task, golden_action_tuples(task)) == 1.0


# ---------------------------------------------------------------------------
# Artifact integrity: the frozen set is a render of the versioned call frame
# ---------------------------------------------------------------------------


def _order_args(task: Task) -> dict:
    return next(
        a
        for a in task.initial_state.initialization_actions
        if a.func_name == "create_callback_order"
    ).arguments


def _task_bank(task: Task) -> str:
    return task.id.removeprefix("intake_").rsplit("_", 2)[0]


def _set_entities(task: Task) -> dict:
    return next(
        a
        for a in task.initial_state.initialization_actions
        if a.func_name == "set_entities"
    ).arguments["entities"]


def test_every_canonical_task_renders_through_the_call_frame_templates():
    for task in load_frozen_tasks():
        org_name = _order_args(task)["org_name"]
        seed = next(
            a
            for a in task.initial_state.initialization_actions
            if a.func_name == "seed_record"
        )
        callee = seed.arguments["record"]["callee_full_name"]
        assert task.agent_opener == render_opener(org_name), task.id
        expected = render_user_scenario(
            callee, org_name, fetched_fields=list(_set_entities(task))
        )
        assert task.user_scenario == expected, task.id


def test_scenario_lists_field_names_but_never_gold_values(tasks):
    """Every scenario names its get_entity fields exactly, and no gold value
    ever appears anywhere in the scenario (values reach the sim only through
    get_entity)."""
    for task in tasks:
        entities = _set_entities(task)
        scenario_blob = json.dumps(task.user_scenario.model_dump(), ensure_ascii=False)
        unknown_info = task.user_scenario.instructions.unknown_info
        for field, value in entities.items():
            assert f"- {field}" in unknown_info, task.id
            assert value not in scenario_blob, task.id


def test_canonical_tasks_cover_the_bank_tier_grid(tasks):
    """10 canonical banks x 2 tiers x 10 calls = 200 atomic tasks."""
    assert len(tasks) == 200
    ids = [t.id for t in tasks]
    assert len(set(ids)) == 200
    expected_ids = [
        f"intake_{bank}_{tier}_{n:02d}"
        for bank in CANONICAL_BANKS
        for tier in ("easy", "hard")
        for n in range(1, 11)
    ]
    assert ids == expected_ids
    for task in tasks:
        assert "Canonical v4 freeze" in task.description.notes, task.id
        assert task.evaluation_criteria.reward_basis == ["DB"], task.id
        assert len(_submission(task)[1]) == 1, task.id  # atomic


def test_base_split_covers_every_task(tasks):
    assert get_tasks_split()["base"] == [t.id for t in tasks]


def test_user_entities_seed_exactly_the_missing_fields(tasks):
    """The callee holds exactly the fields the call collects. Values match
    the golden submission byte-for-byte except relative dates and idiomatic
    time payloads, whose annotations retain the exact full recorded value."""
    for task in tasks:
        _, fields = _submission(task)
        entities = next(
            a
            for a in task.initial_state.initialization_actions
            if a.func_name == "set_entities"
        ).arguments["entities"]
        assert set(entities) == set(fields), task.id
        for name, submitted in fields.items():
            if name == "appointment_time":
                assert entities[name] == display_time_payload(submitted, "en"), task.id
                continue
            if entities[name] == submitted:
                continue
            assert task.id.startswith("intake_dates_hard"), (
                f"{task.id}:{name} entity {entities[name]!r} != golden "
                f"{submitted!r} outside the relative-date cell"
            )
            assert not re.fullmatch(r"\d{4}-\d{2}-\d{2}", entities[name]), task.id
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", submitted), task.id

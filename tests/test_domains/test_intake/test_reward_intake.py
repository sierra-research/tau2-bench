"""Reward tests over the canonical frozen task set (design doc §5.1/§7).

The expected final DB is the base DB plus the missing cells filled with the
pinned true values under the published fold rules. Three matrices over every
checked-in canonical task:

1. **Golden replay** — the task's reference trajectory scores reward 1.0.
2. **Orthography invariance** — a live-agent-style trajectory differing in
   surface forms only (casing, separators, regrouped phones, prose dates,
   24-hour clock times, currency-formatted amounts, re-cased record-id
   echoes) still scores 1.0.
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

import pytest

from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.intake.call_frame import render_opener, render_user_scenario
from tau2.domains.intake.environment import get_environment, get_tasks, get_tasks_split
from tau2.domains.intake.folds import FoldKind
from tau2.domains.intake.tasks.generator import CANONICAL_BANKS
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return get_tasks()


def _env_constructor(solo_mode: bool = False):
    return get_environment(solo_mode=solo_mode)


def replay_reward(task: Task, actions: list[tuple[str, str, dict]]) -> float:
    """Score a simulated trajectory of (requestor, tool, arguments) calls.

    Tolerates failing tool calls — a live agent submitting a record the
    environment rejects is a real trajectory whose reward must be computable
    (and 0.0)."""
    env = _env_constructor()
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
        environment_constructor=_env_constructor,
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
    if kind is FoldKind.NAME:
        return value.swapcase()
    if kind is FoldKind.CODE:
        return f"{value[:4].lower()}-{value[4:].lower()}"
    if kind is FoldKind.PHONE:
        # Regrouped digits; the NANP +1 only where it folds away (a bare
        # 10-digit number — extensions make the digit string longer).
        digits = "".join(ch for ch in value if ch.isdigit())
        grouped = f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"
        return f"+1 {grouped}" if len(digits) == 10 else grouped
    if kind is FoldKind.EMAIL:
        return value.upper()
    if kind is FoldKind.DATE:
        parsed = date.fromisoformat(value)
        return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"
    if kind is FoldKind.TIME:
        # Afternoon times as the (unambiguous) 24-hour clock; everything else
        # as the dotted-meridiem spelling the fold normalizes.
        match = re.fullmatch(r"(\d{1,2}):(\d{2}) (AM|PM)", value)
        assert match, f"unexpected canonical time {value!r}"
        hour, minute, meridiem = int(match[1]), match[2], match[3]
        if meridiem == "PM" and hour != 12:
            return f"{hour + 12}:{minute}"
        return f"{hour}:{minute} {'a.m.' if meridiem == 'AM' else 'p.m.'}"
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
            ("assistant", "get_callback_order", {}),
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
    if kind is FoldKind.NAME:
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
        actions = [("assistant", "get_callback_order", {})]
        assert replay_reward(task, actions) == 0.0, task.id


def test_rejected_partial_submission_zeroes_the_reward():
    """A multi-entity call must write all missing fields in the one submit;
    a partial submission is rejected and scores 0.0. The canonical set is
    atomic, so this runs on a generated n=3 band task."""
    from tau2.domains.intake.tasks.generator import generate_tasks

    n3_tasks, _ = generate_tasks(n_entities=3)
    task = n3_tasks[0]
    record_id, fields = _submission(task)
    assert len(fields) == 3
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


def test_every_canonical_task_renders_through_the_call_frame_templates(tasks):
    for task in tasks:
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
    the golden submission byte-for-byte except relative dates, where the
    callee's records hold the spoken form and the golden value is the ISO
    resolution the agent must compute against get_today (design doc §4)."""
    for task in tasks:
        _, fields = _submission(task)
        entities = next(
            a
            for a in task.initial_state.initialization_actions
            if a.func_name == "set_entities"
        ).arguments["entities"]
        assert set(entities) == set(fields), task.id
        for name, submitted in fields.items():
            if entities[name] == submitted:
                continue
            assert task.id.startswith("intake_dates_hard"), (
                f"{task.id}:{name} entity {entities[name]!r} != golden "
                f"{submitted!r} outside the relative-date cell"
            )
            assert not re.fullmatch(r"\d{4}-\d{2}-\d{2}", entities[name]), task.id
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", submitted), task.id

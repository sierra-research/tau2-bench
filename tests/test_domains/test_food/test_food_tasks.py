"""Tests for the food task data: splits, grading design, user scripts, and contact details."""

import json
import re

import pytest

from tau2.domains.food.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.food.utils import FOOD_TASK_SET_PATH
from tau2.user_simulation_voice_presets import load_task_voice_configs
from tau2.voice_config import BACKGROUND_NOISE_CONTINUOUS_DIR, BURST_NOISE_DIR

from .conftest import FICTIONAL_PHONE_RE, STORES, store_of

_MAX_NL_SHARE = 0.4
_RESERVATION_TOOLS = frozenset(
    {"make_reservation", "update_reservation", "cancel_reservation"}
)
_COUNT_RE = re.compile(r"\b\d+ (?:pumps?|shots?|scoops?)\b")

_RULES_END = (
    'Never put the stop token in the same message where you wrap up (such as "That\'s all."); '
    "wait until the agent confirms the order is placed or tells you the final total, then end "
    "the conversation."
)
# Reservation-only calls end after the booking is confirmed instead of after an order.
_RESERVATION_RULES_END = (
    'Never put the stop token in the same message where you wrap up (such as "That\'s all."); '
    "wait until the agent confirms the reservation is booked or updated, then end the conversation."
)
_TRANSFER_RULES_END = (
    "Never put the stop token or the transfer token in the same message where you ask to be "
    "transferred or wrap up; wait until the agent transfers you to a human agent, then end "
    "the conversation."
)
_FALLBACK = (
    "If the agent asks about any other required choice not covered above, tell it to go "
    "with whatever is standard (and decline any optional add-ons or upsells you didn't ask for)."
)
# Normal steps allow natural paraphrasing and repeating unheard/ignored turns without auditing
# wrong order/modifier read-backs (except name/phone once); conditional trap steps forbid
# coaching the agent or volunteering fallbacks unprompted.
_READBACK_NO_AUDIT_RULE = (
    "Once the agent acknowledges a step, never audit or correct wrong items, quantities, "
    "customizations, counts, or substitutions in its read-backs; only if the agent spells "
    "back your name or reads back your phone number wrong, correct it once."
)
_TRAP_BOUNDARY_RULE = (
    'For any request paired with a conditional "If the agent..." fallback step, assume it was '
    "accepted unless the agent tells you it is unavailable or not allowed—do not push back on "
    "it or volunteer the fallback on your own."
)
_WORD_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_WORD_RE = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine)\b", re.I)


def test_task_splits_partition_the_tasks_by_store():
    splits = get_tasks_split()
    tasks = {t.id: t for t in get_tasks(None)}
    assert set(splits) == {"base", "small", *STORES}
    assert set(splits["base"]) == set(tasks)
    for store_id in STORES:
        assert set(splits[store_id]) == {
            tid for tid, t in tasks.items() if store_of(t) == store_id
        }, store_id
    assert set(splits["small"]) <= set(splits["base"])
    assert {store_of(tasks[tid]) for tid in splits["small"]} == set(STORES)


def test_voice_configs_cover_every_task():
    voice_path = FOOD_TASK_SET_PATH.parent / "tasks_voice.json"
    voice = json.loads(voice_path.read_text())
    assert set(voice["configs"]) == set(get_tasks_split()["base"])
    parsed = load_task_voice_configs(voice_path)
    for task_id, task_cfgs in parsed.configs.items():
        for level, sampled in task_cfgs.configs.items():
            bg = sampled.background_noise_file
            if bg is not None:
                assert (BACKGROUND_NOISE_CONTINUOUS_DIR / bg).is_file(), (
                    task_id,
                    level,
                    bg,
                )
            for burst in sampled.burst_noise_files:
                assert (BURST_NOISE_DIR / burst).is_file(), (task_id, level, burst)
            for insert in (
                *sampled.speech_effects_config.vocal_tics,
                *sampled.speech_effects_config.non_directed_phrases,
            ):
                assert not insert.text.endswith(".wav"), (
                    task_id,
                    level,
                    insert.text,
                )


def test_nl_assertions_are_auxiliary_and_spoken_only():
    tasks = get_tasks("base")
    nl_tasks = [
        t for t in tasks if "NL_ASSERTION" in t.evaluation_criteria.reward_basis
    ]
    assert all("DB" in t.evaluation_criteria.reward_basis for t in tasks)
    assert len(nl_tasks) == 19
    assert 0 < len(nl_tasks) <= _MAX_NL_SHARE * len(tasks)
    for t in tasks:
        crit = t.evaluation_criteria
        if t in nl_tasks:
            assert crit.nl_assertions, t.id
            # Spoken-content checks only: each assertion is about a quoted price/charge or a
            # stated resulting count (shots, pumps, scoops).
            for a in crit.nl_assertions:
                assert re.search(r"\$\d|charge|\b\d+ (?:pumps?|shots?|scoops?)\b", a), (
                    t.id,
                    a,
                )
                assert re.search(r"\b(said|says)\b", a), (t.id, a)
                assert "could not confirm" not in a.lower(), (t.id, a)
        else:
            assert not crit.nl_assertions, t.id
        is_transfer_task = "transfer_to_human_agents" in _gold_names(t)
        for action in crit.actions or []:
            if not is_transfer_task:
                assert action.compare_args != [], (t.id, action.action_id)


def test_default_amount_tasks_document_defaults_and_check_spoken_totals():
    """Carts that depend on a hidden default amount state it in the notes and check the spoken total."""
    count_tasks = [
        t
        for t in get_tasks("base")
        if any(_COUNT_RE.search(a) for a in t.evaluation_criteria.nl_assertions or [])
    ]
    assert {t.id for t in count_tasks} >= {
        "cafe_espresso_shot_defaults_and_cap",
        "cafe_syrup_pump_defaults_and_swap",
        "cafe_blended_scoop_cap_and_whip_split",
    }
    for t in count_tasks:
        desc_text = (t.description.notes or "") + " " + (t.description.purpose or "")
        assert "defaults to" in desc_text, t.id
        assert re.search(r"x\d", desc_text), t.id


def _gold_names(task) -> list[str]:
    return [a.name for a in task.evaluation_criteria.actions or []]


def _is_reservation_only(task) -> bool:
    names = set(_gold_names(task))
    return bool(names) and names <= _RESERVATION_TOOLS


@pytest.mark.parametrize("task", get_tasks("base"), ids=lambda t: t.id)
def test_each_gold_escalates_books_or_places_one_order(task):
    """Transfer golds escalate without ordering, booking golds only book, and every other gold
    places exactly one order; an order placed ahead carries the customer's details."""
    names = _gold_names(task)
    if "transfer_to_human_agents" in names:
        assert "place_order" not in names
        assert "DB" in [
            r.value if hasattr(r, "value") else r
            for r in task.evaluation_criteria.reward_basis
        ]
        return
    if _is_reservation_only(task):
        return
    (place,) = [a for a in task.evaluation_criteria.actions if a.name == "place_order"]
    cart = task.initial_state.initialization_data.agent_data["cart"]
    if cart.get("channel") == "mobile":
        args = place.arguments
        assert args["customer_name"]
        assert FICTIONAL_PHONE_RE.match(args["customer_phone"])
        assert (args["order_type"] == "delivery") == bool(args.get("delivery_address"))
        assert args.get("scheduled_time")


def test_task_text_mentions_only_food_tools():
    """Task descriptions/instructions/assertions never reference tools the agent does not have."""
    env = get_environment()
    tool_names = {t.name for t in env.get_tools()}
    helper_names = {
        name
        for name in dir(type(env.tools))
        if not name.startswith("_") and callable(getattr(type(env.tools), name))
    }
    non_tools = helper_names - tool_names
    assert "get_store_info" in tool_names
    for task in get_tasks("base"):
        text = json.dumps(
            {
                "description": task.description.model_dump()
                if task.description
                else None,
                "user_scenario": task.user_scenario.model_dump(),
                "nl_assertions": task.evaluation_criteria.nl_assertions,
            }
        )
        for name in non_tools:
            assert not re.search(rf"\b{name}\b", text), (task.id, name)
        for name in re.findall(r"`([a-z]+(?:_[a-z]+)+)`", text):
            assert name in tool_names, (task.id, name)
        for action in task.evaluation_criteria.actions or []:
            assert action.name in tool_names, (task.id, action.name)


@pytest.mark.parametrize("task", get_tasks("base"), ids=lambda t: t.id)
def test_user_script_conventions(task):
    """Step goals use natural non-verbatim instructions, protect conditional traps, and speak voice-safe text."""
    steps = [
        s.split(". ", 1)[1]
        for s in task.user_scenario.instructions.reason_for_call.split("\n")
    ]
    names = _gold_names(task)
    if "transfer_to_human_agents" in names:
        rules_end = _TRANSFER_RULES_END
    elif "place_order" not in names:
        rules_end = _RESERVATION_RULES_END
    else:
        rules_end = _RULES_END
    assert steps[-1].startswith("Rules:") and steps[-1].endswith(rules_end)
    assert "speak naturally in your own words" in steps[-1]
    assert _READBACK_NO_AUDIT_RULE in steps[-1]
    assert _TRAP_BOUNDARY_RULE in steps[-1]
    assert '"That\'s all."' in steps[-2]
    assert steps[-3] == _FALLBACK
    rfc = task.user_scenario.instructions.reason_for_call
    ti = task.user_scenario.instructions.task_instructions
    assert "###STOP###" not in rfc and "###TRANSFER###" not in rfc, task.id
    assert "###STOP###" not in ti and "###TRANSFER###" not in ti, task.id
    for s in steps:
        assert not re.match(
            r'^(Say|Then say|In the next turn,.*say):\s*"', s, flags=re.I
        ), (task.id, s)
        assert not re.search(r"\d|\bBBQ\b", s), (task.id, s)
        # Disclosure fallbacks fire on an explicit "not available" or an explicit question, not on an offer.
        assert "or offers" not in s, (task.id, s)
    # Every price question checked by an NL assertion is asked in an unconditional step.
    for a in task.evaluation_criteria.nl_assertions or []:
        if a.startswith("When asked"):
            assert any(not s.startswith("If ") and "?" in s for s in steps), task.id


@pytest.mark.parametrize("task", get_tasks("base"), ids=lambda t: t.id)
def test_customer_phones_are_fictional_and_spoken_in_script(task):
    """Every customer_phone in gold actions and initial reservations is fictional and spoken in reason_for_call."""
    rfc = task.user_scenario.instructions.reason_for_call
    spoken_digits = "".join(
        _WORD_DIGIT[m.group(1).lower()] for m in _WORD_RE.finditer(rfc)
    )
    init = task.initial_state.initialization_data if task.initial_state else None
    init_res = ((init.agent_data or {}) if init else {}).get("reservations") or {}
    phones = [
        r["customer_phone"] for r in init_res.values() if r.get("customer_phone")
    ] + [
        a.arguments["customer_phone"]
        for a in (task.evaluation_criteria.actions or [])
        if (a.arguments or {}).get("customer_phone")
    ]
    for phone in phones:
        assert FICTIONAL_PHONE_RE.match(phone), (task.id, phone)
        digits = re.sub(r"\D", "", phone)
        assert digits in spoken_digits, (task.id, phone, spoken_digits)


@pytest.mark.parametrize("task", get_tasks("base"), ids=lambda t: t.id)
def test_name_collecting_tasks_script_the_spelling(task):
    """Tasks whose gold records a customer name tell the user how to spell it, letter by letter."""
    names = {
        a.arguments["customer_name"]
        for a in task.evaluation_criteria.actions or []
        if (a.arguments or {}).get("customer_name")
    }
    if not names:
        return
    assert len(names) == 1, task.id
    first, last = names.pop().split()
    spelled = (
        f"First name: {', '.join(first.upper())}. Last name: {', '.join(last.upper())}."
    )
    rfc = task.user_scenario.instructions.reason_for_call
    assert "If the agent asks you to spell your name" in rfc, task.id
    assert spelled in rfc, (task.id, spelled)


def test_gold_actions_use_menu_display_names_without_internal_ids():
    """Gold actions use human-readable menu display names rather than opaque menu IDs."""
    from tau2.domains.food.menu_prompt import group_display_names

    raw = FOOD_TASK_SET_PATH.read_text(encoding="utf-8")
    assert not re.search(r"\b(?:cf|pz|bg)-[a-z]\d+\b", raw)

    stores = get_environment().tools.db.stores
    display_by_store = {
        sid: set(group_display_names(st).values()) for sid, st in stores.items()
    }
    item_names_by_store = {
        sid: {it.name for it in st.items.values()} for sid, st in stores.items()
    }

    for task in get_tasks("base"):
        for action in task.evaluation_criteria.actions or []:
            args = action.arguments or {}
            if action.name == "place_order":
                assert "store_id" not in args, task.id
            elif action.name == "add_to_cart":
                sid = args["store_id"]
                assert args["item_id"] in item_names_by_store[sid], (
                    task.id,
                    args["item_id"],
                )
                for mod in args.get("modifiers") or []:
                    if "group_id" in mod:
                        assert mod["group_id"] in display_by_store[sid], (
                            task.id,
                            mod["group_id"],
                        )

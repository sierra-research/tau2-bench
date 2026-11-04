"""Regional callers receive name-role facts from their effective structured record."""

from copy import deepcopy

import pytest

from tau2.data_model.language import NameOrder
from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.multilingual.factory.localization_compile import (
    _retail_tools_for_task,
    compile_localization,
)
from tau2.multilingual.factory.localization_user_ids import (
    finalize_localization_user_ids,
)
from tau2.multilingual.localization import json_content_digest
from tests.test_multilingual import test_regional_localization_compile as fixtures

proposal = fixtures.proposal

NAMES = {
    "Ada": ("静", "Jing"),
    "Miller": ("赵", "Zhao"),
    "Eve": ("美", "Mei"),
    "Stone": ("林", "Lin"),
}


def _rebuild(proposal, *, known=None):
    """Inventory a modified synthetic source with deterministic fixture values."""
    proposal.entries = []
    proposal.source.entries = []
    for document in proposal.source.documents:
        document.sha256 = json_content_digest(document.content)
    fixtures._cover_preserved_values(proposal)
    for source, entry in zip(proposal.source.entries, proposal.entries):
        field = entry.occurrences[0].path[-1]
        if field in {"first_name", "last_name"} and entry.source in NAMES:
            source.category = entry.category = "identity_name"
            source.context = entry.context = f"role={field}"
            entry.forms.native, entry.forms.romanized = NAMES[entry.source]
            entry.operation = "adapt"
        elif field in {"known_info", "instructions"} and known is not None:
            source.category = entry.category = "prose"
            entry.forms.native, entry.forms.romanized = known
            entry.operation = "adapt"
    return proposal


def _task(proposal):
    return proposal.source.documents[1].content[0]


def _known(compiled):
    return compiled[1].content[0]["user_scenario"]["instructions"]["known_info"]


@pytest.mark.parametrize(
    "representation,given,family,full_name",
    [("native", "静", "赵", "赵静"), ("romanized", "Jing", "Zhao", "Zhao Jing")],
)
def test_only_known_info_changes_and_source_is_untouched(
    proposal, representation, given, family, full_name
):
    _rebuild(proposal)
    before = proposal.model_dump_json()
    compiled = compile_localization(proposal, representation)
    assert _known(compiled) == (
        "You need help with your order.\n\n"
        f"Name details:\nYour first name is {given}, and your last name is {family}. "
        "When asked for your full name, give your last name first and your first "
        f"name second: {full_name}."
    )
    expected = deepcopy(_task(proposal))
    expected["evaluation_criteria"]["actions"][0]["arguments"].update(
        first_name=given, last_name=family
    )
    expected["user_scenario"]["instructions"]["known_info"] = _known(compiled)
    assert compiled[1].content[0] == expected
    assert proposal.model_dump_json() == before


def test_given_first_adds_name_facts_to_id_only_scenario(proposal):
    _rebuild(proposal)
    proposal.settings.name_order = NameOrder.GIVEN_FIRST
    assert _known(compile_localization(proposal)).endswith(
        "Your first name is 静, and your last name is 赵. "
        "When asked for your full name, give your first name first and your "
        "last name second: 静赵."
    )


def test_aliases_resolve_the_localized_record(proposal):
    database = proposal.source.documents[0].content
    user = database["users"].pop("u1")
    user["user_id"] = "ada_miller_1234"
    database["users"][user["user_id"]] = user
    _task(proposal)["evaluation_criteria"]["actions"].append(
        {
            "action_id": "inspect",
            "name": "get_user_details",
            "arguments": {"user_id": user["user_id"]},
        }
    )
    revised = finalize_localization_user_ids(_rebuild(proposal))
    compiled = compile_localization(revised)
    assert set(compiled[0].content["users"]) == {"jing_zhao_1234"}
    assert _known(compiled).endswith("name second: 赵静.")


@pytest.mark.parametrize("name_order", list(NameOrder))
def test_wrong_first_authentication_is_preserved(proposal, name_order):
    proposal.settings.name_order = name_order
    actions = _task(proposal)["evaluation_criteria"]["actions"]
    wrong = deepcopy(actions[0])
    wrong["action_id"] = "wrong-first"
    wrong["arguments"]["zip"] = "00000"
    actions.insert(0, wrong)
    compiled = compile_localization(_rebuild(proposal))
    assert [
        action["arguments"]["zip"]
        for action in compiled[1].content[0]["evaluation_criteria"]["actions"]
    ] == ["00000", "60601"]
    assert "Your first name is 静, and your last name is 赵." in _known(compiled)


def test_order_owner_resolves_without_executing_goal_actions(proposal):
    database = proposal.source.documents[0].content
    database["users"]["u1"]["orders"] = ["#ORDER"]
    database["orders"]["#ORDER"] = {
        "order_id": "#ORDER",
        "user_id": "u1",
        "address": deepcopy(database["users"]["u1"]["address"]),
        "items": [],
        "status": "pending",
        "fulfillments": [],
        "payment_history": [],
    }
    _task(proposal)["evaluation_criteria"]["actions"] = [
        {
            "action_id": "cancel",
            "name": "cancel_pending_order",
            "arguments": {"order_id": "#ORDER", "reason": "no longer needed"},
        }
    ]
    compiled = compile_localization(_rebuild(proposal))
    assert compiled[0].content["orders"]["#ORDER"]["status"] == "pending"
    assert _known(compiled).endswith("name second: 赵静.")


@pytest.mark.parametrize("with_action", [False, True])
def test_effective_initialization_names_and_display_assertion(proposal, with_action):
    caller = deepcopy(proposal.source.documents[0].content["users"]["u1"])
    caller["name"] = {"first_name": "Eve", "last_name": "Stone"}
    task = _task(proposal)
    task["initial_state"] = {
        "initialization_data": {"agent_data": {"users": {"u1": caller}}}
    }
    task["evaluation_criteria"]["actions"] = (
        [
            {
                "action_id": "inspect",
                "name": "get_user_details",
                "arguments": {"user_id": "u1"},
            }
        ]
        if with_action
        else []
    )
    task["user_scenario"]["instructions"]["known_info"] = "Your name is Eve Stone."
    _rebuild(proposal, known=("Your name is 林美.", "Your name is Lin Mei."))
    for representation, given, family, full_name in [
        ("native", "美", "林", "林美"),
        ("romanized", "Mei", "Lin", "Lin Mei"),
    ]:
        compiled = compile_localization(proposal, representation)
        assert f"Your first name is {given}, and your last name is {family}." in _known(
            compiled
        )
        assert _known(compiled).endswith(f"name second: {full_name}.")
        assert (
            compiled[0].content["users"]["u1"]["name"]["first_name"]
            == NAMES["Ada"][representation == "romanized"]
        )
    next(
        row for row in proposal.entries if row.occurrences[0].path[-1] == "known_info"
    ).forms.native = "Your name is 赵静."
    with pytest.raises(ValueError, match="display name disagrees"):
        compile_localization(proposal)


def test_runtime_initialization_actions_are_isolated(proposal):
    database = RetailDB.model_validate(proposal.source.documents[0].content)
    raw = deepcopy(_task(proposal))
    raw["initial_state"] = {
        "initialization_actions": [
            {
                "env_type": "assistant",
                "func_name": "modify_user_address",
                "arguments": {
                    "user_id": "u1",
                    **database.users["u1"].address.model_dump(),
                    "zip": "99999",
                },
            }
        ]
    }
    tools = _retail_tools_for_task(database, Task.model_validate(raw))
    assert tools.db.users["u1"].address.zip == "99999"
    assert database.users["u1"].address.zip == "60601"


@pytest.mark.parametrize("duplicate", [False, True])
def test_transfer_only_uses_unique_exact_source_name(proposal, duplicate):
    task = _task(proposal)
    task["evaluation_criteria"]["actions"] = [
        {
            "action_id": "transfer",
            "name": "transfer_to_human_agents",
            "arguments": {"summary": "Needs assistance."},
        }
    ]
    task["user_scenario"]["instructions"]["known_info"] = (
        "You're Ada Miller, calling for help."
    )
    if duplicate:
        users = proposal.source.documents[0].content["users"]
        users["u2"] = deepcopy(users["u1"])
        users["u2"]["user_id"] = "u2"
    _rebuild(
        proposal,
        known=("You're 赵静, calling for help.", "You're Zhao Jing, calling for help."),
    )
    if duplicate:
        with pytest.raises(ValueError, match="unique structured caller"):
            compile_localization(proposal)
    else:
        assert _known(compile_localization(proposal)).endswith("name second: 赵静.")


def test_conflicting_explicit_customer_references_fail(proposal):
    users = proposal.source.documents[0].content["users"]
    users["u2"] = deepcopy(users["u1"])
    users["u2"].update(user_id="u2", name={"first_name": "Eve", "last_name": "Stone"})
    _task(proposal)["evaluation_criteria"]["actions"] = [
        {
            "action_id": user_id,
            "name": "get_user_details",
            "arguments": {"user_id": user_id},
        }
        for user_id in users
    ]
    with pytest.raises(ValueError, match="unique structured caller"):
        compile_localization(_rebuild(proposal))


@pytest.mark.parametrize(
    "suffix",
    [
        "Name details:\nGiven name: wrong\nFamily name: 赵",
        "Name details:\nGiven name: 静\nFamily name: 赵\n\nName details:\nGiven name: 静\nFamily name: 赵",
    ],
)
def test_stale_or_duplicate_facts_are_rejected(proposal, suffix):
    known = "You need help with your order.\n\n" + suffix
    _task(proposal)["user_scenario"]["instructions"]["known_info"] = known
    with pytest.raises(ValueError, match="stale, malformed or duplicated"):
        compile_localization(_rebuild(proposal))


@pytest.mark.parametrize("representation", ["native", "romanized"])
def test_given_first_rejects_stale_name_without_user_id(proposal, representation):
    task = _task(proposal)
    task["user_scenario"]["instructions"]["known_info"] = (
        "You are Ada Miller, and you live in Chicago, 60601."
    )
    proposal.settings.name_order = NameOrder.GIVEN_FIRST
    _rebuild(
        proposal,
        known=(
            "You are Stale 赵, and you live in Chicago, 60601.",
            "You are Stale Zhao, and you live in Chicago, 60601.",
        ),
    )
    with pytest.raises(ValueError, match="display name disagrees"):
        compile_localization(proposal, representation)
    entry = next(
        e for e in proposal.entries if e.occurrences[0].path[-1] == "known_info"
    )
    entry.forms.native = "You are 静 赵, and you live in Chicago, 60601."
    entry.forms.romanized = "You are Jing Zhao, and you live in Chicago, 60601."
    assert "Name details:" in _known(compile_localization(proposal, representation))


@pytest.mark.parametrize("name_order", list(NameOrder))
def test_plain_text_instructions_validate_and_clarify_names(proposal, name_order):
    proposal.settings.name_order = name_order
    _task(proposal)["user_scenario"]["instructions"] = "You are Ada Miller."
    names = (
        ("静赵", "Jing Zhao")
        if name_order == NameOrder.GIVEN_FIRST
        else ("赵静", "Zhao Jing")
    )
    _rebuild(proposal, known=tuple(f"You are {name}." for name in names))
    compiled = compile_localization(proposal)
    assert (
        "Your first name is 静, and your last name is 赵."
        in compiled[1].content[0]["user_scenario"]["instructions"]
    )
    entry = next(e for e in proposal.entries if e.source == "You are Ada Miller.")
    entry.forms.native = "You are Another Person."
    with pytest.raises(ValueError, match="display name disagrees"):
        compile_localization(proposal)


def test_caller_postal_code_must_match_effective_database(proposal):
    _task(proposal)["user_scenario"]["instructions"]["known_info"] = (
        "You are Ada Miller, living in 60601."
    )
    _rebuild(
        proposal,
        known=("You are 赵静, living in 00000.", "You are Zhao Jing, living in 60601."),
    )
    with pytest.raises(ValueError, match="postal code disagrees"):
        compile_localization(proposal)


def test_name_zip_must_remain_unique_without_a_golden_lookup(proposal):
    database = proposal.source.documents[0].content
    other = deepcopy(database["users"]["u1"])
    other.update(
        user_id="u2",
        name={"first_name": "Eve", "last_name": "Stone"},
        email="other@example.com",
    )
    database["users"]["u2"] = other
    task = _task(proposal)
    task["evaluation_criteria"]["actions"] = [
        {
            "action_id": "inspect",
            "name": "get_user_details",
            "arguments": {"user_id": "u1"},
        }
    ]
    task["user_scenario"]["instructions"]["known_info"] = (
        "You are Ada Miller, living in 60601."
    )
    _rebuild(
        proposal,
        known=("You are 赵静, living in 60601.", "You are Zhao Jing, living in 60601."),
    )
    for entry in proposal.entries:
        if entry.source == "Eve":
            entry.forms.native, entry.forms.romanized = NAMES["Ada"]
        elif entry.source == "Stone":
            entry.forms.native, entry.forms.romanized = NAMES["Miller"]
    with pytest.raises(ValueError, match="ambiguous"):
        compile_localization(proposal)


def test_saved_profile_address_goal_must_follow_localized_profile(proposal):
    address = proposal.source.documents[0].content["users"]["u1"]["address"]
    task = _task(proposal)
    task["evaluation_criteria"]["actions"].append(
        {
            "action_id": "address",
            "name": "modify_pending_order_address",
            "arguments": {"order_id": "#ORDER", **address},
        }
    )
    _rebuild(proposal)
    entry = next(
        e
        for e in proposal.entries
        if e.occurrences[0].path
        == [0, "evaluation_criteria", "actions", 1, "arguments", "zip"]
    )
    entry.category = "address"
    next(e for e in proposal.source.entries if e.id == entry.id).category = "address"
    entry.forms.native = entry.forms.romanized = "00000"
    entry.operation = "adapt"
    with pytest.raises(ValueError, match="saved-address goal disagrees"):
        compile_localization(proposal)

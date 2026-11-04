"""Short regional emails preserve identifiers, lookup failures and exact prose."""

from pathlib import Path

import pytest

from tau2.multilingual.factory import localization_artifact as artifacts
from tau2.multilingual.factory import localization_inventory as inventory
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.localization_emails import (
    EmailAssignment,
    EmailCandidate,
    allocate_emails,
    email_domain,
    localize_email_entries,
)
from tau2.multilingual.factory.localization_identity import ascii_name_parts
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationEntry,
    LocalizationMap,
    LocalizedForms,
    RegionalSettings,
    SourceDocument,
    json_content_digest,
)


def test_source_numbers_separators_initials_and_wrong_email_are_preserved():
    sources = [
        "emma.nguyen1887@example.com",
        "emmanguyen1887@example.com",
        "emma.n@example.com",
    ]
    values = {
        row.source: row.value
        for row in allocate_emails(
            [
                EmailCandidate(source=source, name_parts=["marta", "navarro"])
                for source in sources
            ],
            "example.com",
        )
    }
    assert values[sources[0]] == "marta.navarro1887@example.com"
    assert values[sources[1]] == "martanavarro1887@example.com"
    assert values[sources[2]].startswith("marta.n")
    assert (
        values[sources[2]]
        .removeprefix("marta.n")
        .removesuffix("@example.com")
        .isdigit()
    )
    assert len(values[sources[2]].split("@")[0]) == len("marta.n") + 4


def test_collisions_are_order_independent_and_do_not_steal_original_candidates():
    inputs = [
        EmailCandidate(source=source, name_parts=["marta", "navarro"])
        for source in (
            "amy.li0001@example.com",
            "zoe.li0001@example.com",
            "yara.li0002@example.com",
        )
    ]
    output = allocate_emails(inputs, "example.com")
    assert output == allocate_emails(list(reversed(inputs)), "example.com")
    assert [row.value for row in output] == [
        "marta.navarro0001@example.com",
        "marta.navarro0002@example.com",
        "marta.navarro0003@example.com",
    ]
    assert len({row.value for row in output}) == 3


def test_allocator_uses_actual_lookup_normalization_and_reserves_retained_values():
    inputs = [
        EmailCandidate(source=source, name_parts=["marta", "navarro"])
        for source in ("Emma.Nguyen1887@example.com", "emma.nguyen1887@example.com")
    ]
    assert len({row.value for row in allocate_emails(inputs, "example.com")}) == 1
    reserved = [
        EmailAssignment(
            source="other@example.com", value="MARTA.NAVARRO1887@EXAMPLE.COM"
        )
    ]
    result = allocate_emails(inputs, "example.com", reserved)
    assert {row.value for row in result} == {"marta.navarro1888@example.com"}
    with pytest.raises(ValueError, match="conflicting localized names"):
        allocate_emails(
            inputs
            + [
                EmailCandidate(
                    source=inputs[0].source, name_parts=["another", "person"]
                )
            ],
            "example.com",
        )


def test_allocation_uses_the_reviewed_regional_domain():
    inputs = [
        EmailCandidate(source=source, name_parts=["joao", "silva"])
        for source in ("john.smith@example.com", "js0001@example.com")
    ]
    reserved = [
        EmailAssignment(source="other@example.com", value="joaosilva0001@email.com")
    ]
    values = {
        row.source: row.value for row in allocate_emails(inputs, "email.com", reserved)
    }
    assert values["john.smith@example.com"].endswith("@email.com")
    assert values["john.smith@example.com"].startswith("joao.silva")
    # Collisions are detected within the regional domain, not example.com.
    assert values["js0001@example.com"] == "joaosilva0002@email.com"
    elsewhere = [
        EmailAssignment(source="other@example.com", value="joaosilva0001@example.com")
    ]
    assert allocate_emails(inputs[1:], "email.com", elsewhere)[0].value == (
        "joaosilva0001@email.com"
    )
    with pytest.raises(ValueError, match="Unsupported email domain"):
        allocate_emails(inputs, "not a domain")


def test_email_domain_requires_exactly_one_reviewed_domain():
    assert email_domain(["a@Email.com", "b@email.com"]) == "email.com"
    with pytest.raises(ValueError, match="ambiguous"):
        email_domain(["a@email.com", "b@example.com"])
    with pytest.raises(ValueError, match="ambiguous"):
        email_domain([])


def test_new_map_emails_follow_the_domain_of_retained_localized_emails(draft):
    contacts = [e for e in draft.source.entries if e.category == "contact"]
    names = [e for e in draft.entries if e.category == "identity_name"]
    fresh = localize_email_entries(contacts, names)
    assert {e.forms.native.rpartition("@")[2] for e in fresh} == {"example.com"}
    retained = fresh[0].model_copy(
        update={
            "forms": fresh[0].forms.model_copy(
                update={
                    "native": "marta.navarro1887@email.com",
                    "romanized": "marta.navarro1887@email.com",
                }
            )
        }
    )
    [added] = localize_email_entries(contacts[1:], [*names, retained])
    assert added.forms.native == "martanavarro1887@email.com"


def test_shared_ascii_components_preserve_identity_order():
    assert ascii_name_parts(["João", "Gonçalves"]) == ["joao", "goncalves"]
    assert ascii_name_parts(["I", "Seo-jun"]) == ["i", "seojun"]
    with pytest.raises(ValueError, match="nonempty ASCII"):
        ascii_name_parts(["李"])


@pytest.fixture
def draft(monkeypatch):
    correct = "emma.nguyen1887@example.com"
    wrong = "emmanguyen1887@example.com"
    user = {
        "user_id": "u1",
        "name": {"first_name": "Emma", "last_name": "Nguyen"},
        "email": correct,
        "address": {
            "address1": "1 Main Street",
            "address2": "Unit 2",
            "city": "Chicago",
            "state": "IL",
            "country": "USA",
            "zip": "60601",
        },
        "payment_methods": {},
        "orders": [],
    }
    task = {
        "id": "t1",
        "user_scenario": {
            "instructions": {
                "domain": "retail",
                "reason_for_call": "Check your account.",
                "task_instructions": "Follow the scenario.",
                "known_info": f"Try {wrong}, then {correct}. Keep order_opaque_42 unchanged.",
            }
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "wrong",
                    "name": "find_user_id_by_email",
                    "arguments": {"email": wrong},
                },
                {
                    "action_id": "correct",
                    "name": "find_user_id_by_email",
                    "arguments": {"email": correct},
                },
                {
                    "action_id": "owner",
                    "name": "get_user_details",
                    "arguments": {"user_id": "u1"},
                },
            ],
            "reward_basis": ["DB"],
        },
    }
    documents = [
        SourceDocument(
            id=name,
            kind=kind,
            source_path=name,
            sha256=json_content_digest(content),
            content=content,
        )
        for name, kind, content in (
            (
                "database",
                "database",
                {"users": {"u1": user}, "orders": {}, "products": {}},
            ),
            ("tasks", "tasks", [task]),
        )
    ]
    monkeypatch.setattr(inventory, "_source_documents", lambda *args: documents)
    source = inventory.inventory_localization("retail", Path("."))
    settings = RegionalSettings(
        language="es",
        country_code="ES",
        currency_code="EUR",
        currency_scale="1",
        time_zone="Europe/Madrid",
        name_order="given_first",
    )
    old = {
        correct: "marta.navarro.58fb1ffff7@example.com",
        wrong: "marta.navarro.2626dc58d2@example.com",
    }
    entries = []
    for entry in source.entries:
        value = entry.source
        if entry.category == "identity_name":
            value = {"Emma": "Marta", "Nguyen": "Navarro"}[value]
        elif isinstance(value, str):
            for original, localized in old.items():
                value = value.replace(original, localized)
        entries.append(
            artifacts.automatic_localization(entry, settings)
            or LocalizationEntry(
                **entry.model_dump(),
                forms=LocalizedForms(native=value, romanized=value, english="Fixture."),
                operation="preserve" if value == entry.source else "adapt",
                notes="Fixture.",
            )
        )
    return LocalizationMap(
        revision="before",
        settings=settings,
        source=source,
        entries=entries,
        provenance=GenerationProvenance(
            model="fixture",
            model_args={},
            prompt_version="fixture-v1",
            call_name="fixture",
            input_sha256="b" * 64,
        ),
    )


def test_future_drafting_resolves_emails_before_generating_dependent_prose(
    draft, monkeypatch, tmp_path
):
    from tau2.multilingual.factory import database_localization as drafting

    contacts = [entry for entry in draft.source.entries if entry.category == "contact"]
    completed = [entry for entry in draft.entries if entry.category != "contact"]
    monkeypatch.setattr(
        drafting,
        "generate",
        lambda **kwargs: pytest.fail("Email allocation must not call an LLM"),
    )
    request = drafting.RegionalDraftRequest(
        source=draft.source,
        settings=draft.settings,
        revision="fresh",
        generation=FactoryModelConfig(model="fixture", reasoning_effort="low"),
    )
    result = drafting._draft_entries(request, contacts, completed, "c" * 64, tmp_path)
    assert {entry.forms.native for entry in result if entry.category == "contact"} == {
        "marta.navarro1887@example.com",
        "martanavarro1887@example.com",
    }
    assert request.model_dump()["email_rules_version"] == "regional-email-v1"

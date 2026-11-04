"""Regional drafting keeps checked sources separate from generated forms."""

import json
from types import SimpleNamespace

import pytest

from tau2.multilingual.factory import database_localization as drafting
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.localization_emails import localize_email_entries
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizationSource,
    LocalizedForms,
    RegionalSettings,
    SourceDocument,
    SourceEntry,
    SourceOccurrence,
    json_content_digest,
)


def source_entry(
    identifier="name",
    source="Mia",
    category="identity_name",
    role="first_name",
    context="",
):
    return SourceEntry(
        id=identifier,
        source=source,
        category=category,
        context=f"role={role}; {context}",
        occurrences=[SourceOccurrence(document_id="database", path=[identifier])],
    )


def region():
    return RegionalSettings(
        language="hi",
        country_code="IN",
        currency_code="INR",
        currency_scale="80",
        time_zone="Asia/Kolkata",
        name_order="given_first",
    )


def mapped(entry, native, romanized, operation="adapt"):
    return LocalizationEntry(
        **entry.model_dump(),
        forms=LocalizedForms(
            native=native, romanized=romanized, english="Test fixture meaning."
        ),
        operation=operation,
        notes="Recorded test fixture.",
    )


def test_templates_keep_distinct_neighboring_unit_numbers():
    first = source_entry("one", "Suite 640", "address", "address2")
    second = source_entry("two", "Suite 641", "address", "address2")
    [group] = drafting._draft_groups([first, second])
    generated = mapped(group.representative, "यूनिट {number_0}", "Unit {number_0}")
    a, b = drafting._expand_group(group, generated)
    assert a.forms.native == "यूनिट 640"
    assert b.forms.native == "यूनिट 641"
    assert a.occurrences == first.occurrences
    assert b.occurrences == second.occurrences
    assert a.source == "Suite 640"
    assert b.source == "Suite 641"


@pytest.mark.parametrize("bad", ["Unit 12", "{number_0} {number_0}", "{number_1}"])
def test_templates_reject_lost_or_duplicated_numeric_slots(bad):
    [group] = drafting._draft_groups(
        [source_entry("one", "Suite 640", "address", "address2")]
    )
    with pytest.raises(ValueError, match="numeric template"):
        drafting._expand_group(group, mapped(group.representative, bad, bad))


@pytest.mark.parametrize("role", ["first_name", "last_name"])
def test_equal_name_components_keep_distinct_people_and_roles(role):
    a = source_entry("person-a", "Yang", role=role, context="entity=a")
    b = source_entry("person-b", "Yang", role=role, context="entity=b")
    other_role = "last_name" if role == "first_name" else "first_name"
    other = source_entry("other", "Yang", role=other_role)
    groups = drafting._draft_groups([a, b, other])
    assert len(groups) == 2
    generated = mapped(groups[0].representative, "阳", "Yang")
    generated.forms.english = f"{role} for customer a."
    generated.notes = "Display customer a's full name as Li Yang."
    expanded = drafting._expand_group(groups[0], generated)
    assert [entry.id for entry in expanded] == ["person-a", "person-b"]
    assert [entry.context for entry in expanded] == [a.context, b.context]
    assert [entry.occurrences for entry in expanded] == [a.occurrences, b.occurrences]
    assert all(entry.forms.native == "阳" for entry in expanded)
    assert all(entry.forms.romanized == "Yang" for entry in expanded)
    assert expanded[0].forms.english == generated.forms.english
    assert expanded[0].notes == generated.notes
    assert expanded[1].forms.english == (
        f"Localized {role} component for source entity b."
    )
    assert "customer a" not in expanded[1].notes
    assert "Li Yang" not in expanded[1].notes


def test_single_name_component_retains_review_concerns():
    [group] = drafting._draft_groups([source_entry(context="entity=a")])
    generated = mapped(group.representative, "阳", "Yang")
    generated.notes = "Check the romanization with a native speaker."
    assert drafting._expand_group(group, generated) == [generated]


def test_email_uses_romanized_name_and_retains_distinct_wrong_address():
    first = source_entry("first", "Mia")
    last = source_entry("last", "Li", role="last_name")
    completed = [mapped(first, "आरती", "Aarti"), mapped(last, "त्रिवेदी", "Trivedi")]
    context = (
        'source_identity={"first_name":"Mia","last_name":"Li"}. Remaining context.'
    )
    actual, wrong = localize_email_entries(
        [
            source_entry("email", "mia.li@example.com", "contact", "email", context),
            source_entry("wrong", "mia.li1@example.com", "contact", "email", context),
        ],
        completed,
    )
    assert actual.forms.native.startswith("aarti.trivedi")
    assert actual.forms.native == actual.forms.romanized
    assert actual.forms.native != wrong.forms.native


def test_telecom_email_uses_completed_full_name_and_reserved_domain():
    first = source_entry("name", "John Smith", role="full_name")
    completed = [mapped(first, "이서준", "I Seo-jun")]
    [actual] = localize_email_entries(
        [
            source_entry(
                "email",
                "john.smith@example.com",
                "contact",
                "email",
                'source_identity={"full_name":"John Smith"}.',
            )
        ],
        completed,
    )
    assert actual.forms.native.startswith("i.seojun")
    assert actual.forms.native.endswith("@example.com")
    assert actual.forms.native == actual.forms.romanized


@pytest.mark.parametrize("reject_first", [False, True])
def test_batch_roundtrip_reuses_validated_checkpoint(
    monkeypatch, tmp_path, reject_first
):
    entry = source_entry()
    content = {entry.id: entry.source}
    source = LocalizationSource(
        domain="retail",
        documents=[
            SourceDocument(
                id="database",
                kind="database",
                source_path="fixture.json",
                sha256=json_content_digest(content),
                content=content,
            )
        ],
        entries=[entry],
    )
    request = drafting.RegionalDraftRequest(
        source=source,
        settings=region(),
        revision="test",
        generation=FactoryModelConfig(model="test-model", reasoning_effort="low"),
    )
    batch = drafting.BatchInput(request_sha256="b" * 64, entries=[entry], completed=[])
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": "name",
                            "forms": {
                                "native": "आरती",
                                "romanized": "Aarti",
                                "english": "Female given name.",
                            },
                            "operation": "preserve"
                            if reject_first and len(calls) == 1
                            else "adapt",
                            "notes": "Fixture.",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", generate)
    result = drafting._generate_batch(request, batch, tmp_path)
    assert drafting._generate_batch(request, batch, tmp_path) == result
    assert len(calls) == (2 if reject_first else 1)
    assert calls[0]["call_name"] == "factory_database_localization"
    assert calls[0]["reasoning_effort"] == "low"
    assert result[0].source == "Mia"
    assert result[0].occurrences == entry.occurrences
    [path] = tmp_path.glob("*.json")
    artifact = drafting.BatchArtifact.model_validate_json(path.read_text())
    assert len(artifact.rejected_candidates) == int(reject_first)
    if reject_first:
        assert "preserved entry" in artifact.rejected_candidates[0].error
        assert "complete corrected batch" in calls[-1]["messages"][-1].content


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [
            {
                "id": "unexpected",
                "forms": {"native": "a", "romanized": "a", "english": "a"},
                "operation": "adapt",
                "notes": "",
            }
        ],
    ],
)
def test_generated_batch_must_cover_exact_source_ids(rows):
    batch = drafting.BatchInput(
        request_sha256="b" * 64, entries=[source_entry()], completed=[]
    )
    with pytest.raises(ValueError, match="exactly"):
        drafting._validate_batch(batch, drafting.GeneratedBatch(entries=rows))


def test_generation_rejects_foreign_street_order_including_templates():
    entry = source_entry("street", "123 Source Street", "address", "street")
    content = {entry.id: entry.source}
    source = LocalizationSource(
        domain="retail",
        documents=[
            SourceDocument(
                id="database",
                kind="database",
                source_path="fixture.json",
                sha256=json_content_digest(content),
                content=content,
            )
        ],
        entries=[entry],
    )
    settings = region().model_copy(update={"language": "zh", "country_code": "CN"})
    request = drafting.RegionalDraftRequest(
        source=source,
        settings=settings,
        revision="test",
        generation=FactoryModelConfig(model="test", reasoning_effort="low"),
    )
    batch = drafting.BatchInput(request_sha256="a" * 64, entries=[entry], completed=[])
    for value in ("123号锦江路", "{number_0}号锦江路"):
        generated = drafting.GeneratedBatch.model_validate(
            {
                "entries": [
                    {
                        "id": entry.id,
                        "forms": {
                            "native": value,
                            "romanized": "Jinjiang Lu 123 Hao",
                            "english": "Fixture",
                        },
                        "operation": "adapt",
                        "notes": "Fixture",
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="premise/road order"):
            drafting._validate_batch(batch, generated, request=request)


def test_product_attribute_contexts_are_not_merged():
    from tests.test_multilingual.test_banking_prose import entry

    a = entry("armrest", "none", "none", category="product_option")
    b = entry("backlight", "none", "none", category="product_option")
    a.context = "Product option armrest"
    b.context = "Product option backlight"
    assert len(drafting._draft_groups([a, b])) == 2

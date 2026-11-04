"""Regional review keeps full coverage visible and renders arbitrary values safely."""

from html.parser import HTMLParser

import pytest

from tau2.data_model.language import NameOrder
from tau2.multilingual.factory.localization_review import write_localization_review
from tau2.multilingual.localization import LocalizationMap, json_content_digest
from tests.test_multilingual import test_regional_localization_compile as fixtures

proposal = fixtures.proposal


class ReviewHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.scripts = 0
        self.event_attributes = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "tr" and attributes.get("class") == "entry":
            self.rows.append(attributes)
        self.scripts += tag == "script"
        self.event_attributes.extend(k for k in attributes if k.startswith("on"))

    def handle_data(self, data):
        self.text.append(data)


def sample_map(domain, *, value="Source name", explanation="A fictional name."):
    source_entries = [
        {
            "id": f"{domain}-name",
            "category": "identity_name",
            "source": value,
            "occurrences": [
                {"document_id": "db", "path": ["name"]},
                {"document_id": "db", "path": ["repeated", 0]},
            ],
            "context": "Customer display name",
        },
        {
            "id": f"{domain}-count",
            "category": "protocol",
            "source": {"count": 0, "enabled": False, "optional": None},
            "occurrences": [{"document_id": "db", "path": ["settings"]}],
            "context": "Unchanged operational fields",
        },
    ]
    return LocalizationMap.model_validate(
        {
            "revision": f"ko-KR-{domain}-draft",
            "settings": {
                "language": "ko",
                "country_code": "KR",
                "currency_code": "KRW",
                "currency_scale": "1000",
                "time_zone": "Asia/Seoul",
                "name_order": "family_first",
                "notes": ["Fixed benchmark conversion; not a live exchange rate."],
            },
            "source": {
                "domain": domain,
                "documents": [
                    {
                        "id": "db",
                        "kind": "database",
                        "source_path": f"tau2/domains/{domain}/db.json",
                        "sha256": json_content_digest(
                            {
                                "name": value,
                                "repeated": [value],
                                "settings": source_entries[1]["source"],
                            }
                        ),
                        "content": {
                            "name": value,
                            "repeated": [value],
                            "settings": source_entries[1]["source"],
                        },
                    }
                ],
                "entries": source_entries,
            },
            "entries": [
                {
                    **source_entries[0],
                    "forms": {
                        "native": "백채원",
                        "romanized": "Baek Chaewon",
                        "english": explanation,
                    },
                    "operation": "adapt",
                    "notes": "Given and family names remain distinct.",
                },
                {
                    **source_entries[1],
                    "forms": {
                        "native": source_entries[1]["source"],
                        "romanized": source_entries[1]["source"],
                        "english": "Operational values stay unchanged.",
                    },
                    "operation": "preserve",
                    "notes": "Keep scenario behavior.",
                },
            ],
            "provenance": {
                "model": "test-model",
                "model_args": {"reasoning_effort": "low"},
                "prompt_version": "regional-test-v1",
                "call_name": "regional_localization_test",
                "input_sha256": "b" * 64,
            },
            "status": "published",
        }
    )


def test_review_includes_every_domain_entry_occurrence_and_provenance(tmp_path):
    maps = [sample_map(domain) for domain in ("airline", "retail", "telecom")]
    page = write_localization_review(maps, tmp_path / "review")
    assert page == tmp_path / "review/index.html"
    before = page.read_bytes()
    parsed = ReviewHTML()
    parsed.feed(before.decode())
    text = " ".join(parsed.text)
    assert len(parsed.rows) == 6
    assert {row["data-domain"] for row in parsed.rows} == {
        "airline",
        "retail",
        "telecom",
    }
    assert {row["data-operation"] for row in parsed.rows} == {"adapt", "preserve"}
    for value in (
        "Source value",
        "Native form",
        "Romanized form",
        "English meaning",
        "Draft",
        "not human approval",
        "6 / 6",
        "9",
        "Asia/Seoul",
        "KRW",
        "regional-test-v1",
        "test-model",
        "b" * 64,
        "백채원",
        "Baek Chaewon",
        "/repeated/0",
        '"count": 0',
        '"enabled": false',
    ):
        assert value in text
    for domain in ("airline", "retail", "telecom"):
        assert f"tau2/domains/{domain}/db.json" in text
    for mapping in maps:
        assert json_content_digest(mapping.source.model_dump(mode="json")) in text
    write_localization_review(maps, tmp_path / "review")
    assert page.read_bytes() == before


def test_review_escapes_markup_and_exposes_no_active_untrusted_html(tmp_path):
    attack = '</script><script>alert("unsafe")</script><img src=x onerror=alert(1)>'
    value = sample_map("retail", value=attack, explanation=attack)
    value.entries[0].notes = attack
    value.entries[0].context = attack
    page = write_localization_review([value], tmp_path)
    source = page.read_text()
    parsed = ReviewHTML()
    parsed.feed(source)
    assert parsed.scripts == 1
    assert not parsed.event_attributes
    assert attack not in source
    assert attack in " ".join(parsed.text)
    assert "&lt;script&gt;" in source
    assert "innerHTML" not in source


def test_empty_review_does_not_create_output(tmp_path):
    output = tmp_path / "empty"
    with pytest.raises(ValueError, match="at least one"):
        write_localization_review([], output)
    assert not output.exists()


def test_review_includes_escaped_supplemental_references_without_inflating_coverage(
    tmp_path,
):
    from tau2.multilingual.localization import ProseEntityBinding

    mapping = sample_map("retail")
    attack = '<img src=x onerror="alert(1)">'
    mapping.prose_bindings = [
        ProseEntityBinding(
            entry_id=mapping.entries[0].id,
            source=attack,
            forms=mapping.entries[0].forms.model_copy(deep=True),
            kind="generated",
            reason=attack,
            provenance=mapping.provenance,
        )
    ]
    page = write_localization_review([mapping], tmp_path)
    source = page.read_text()
    parsed = ReviewHTML()
    parsed.feed(source)
    text = " ".join(parsed.text)
    assert len(parsed.rows) == 3
    assert sum(row["data-category"] == "prose_reference" for row in parsed.rows) == 1
    assert "2 / 2" in text
    assert "Prose references (1; review individually)" in text
    assert mapping.entries[0].id in text
    assert "test-model" in text
    assert attack in text
    assert attack not in source
    assert parsed.scripts == 1
    assert not parsed.event_attributes


def test_review_title_identifies_single_or_multiple_locales(tmp_path):
    korean = sample_map("retail")
    single = write_localization_review([korean, sample_map("airline")], tmp_path)
    assert "<title>ko-KR — Regional localization review</title>" in single.read_text()
    hindi = sample_map("telecom")
    hindi.settings.language = "hi"
    hindi.settings.country_code = "IN"
    mixed = write_localization_review([korean, hindi], tmp_path)
    assert (
        "<title>Multiple locales: hi-IN, ko-KR — Regional localization review</title>"
        in mixed.read_text()
    )


def test_review_shows_missing_coverage_and_tool_wiring_concerns(tmp_path):
    mapping = sample_map("telecom")
    mapping.entries = mapping.entries[:1]
    mapping.entries[0].category = "tool_text"
    page = write_localization_review([mapping], tmp_path)
    parsed = ReviewHTML()
    parsed.feed(page.read_text())
    text = " ".join(parsed.text)
    assert "1 / 2" in text
    assert "Missing mappings: telecom-count" in text
    assert "this mapping alone does not establish" in text
    assert "runtime tool messages or rules have been wired" in text


def test_review_shows_compiled_name_facts_in_both_representations(proposal, tmp_path):
    proposal.entries[0].forms.romanized = "Jing"
    before = proposal.model_dump_json()
    page = write_localization_review([proposal], tmp_path)
    parsed = ReviewHTML()
    parsed.feed(page.read_text())
    text = " ".join(parsed.text)
    assert "Compiled caller instructions (1 task)" in text
    assert "Your first name is 静, and your last name is Miller." in text
    assert "Your first name is Jing, and your last name is Miller." in text
    assert "compiled-name-details-0" in page.read_text()
    assert len(parsed.rows) == len(proposal.entries)
    assert proposal.model_dump_json() == before


def test_given_first_review_has_name_fact_panel(proposal, tmp_path):
    proposal.settings.name_order = NameOrder.GIVEN_FIRST
    page = write_localization_review([proposal], tmp_path)
    assert "Compiled caller instructions" in page.read_text()
    assert "give your first name first and your last name second" in page.read_text()


def test_compiled_review_accepts_string_task_instructions(
    proposal, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from tau2.multilingual.factory import localization_compile

    monkeypatch.setattr(
        localization_compile,
        "compile_localization",
        lambda *args: [
            SimpleNamespace(
                kind="tasks",
                content=[
                    {
                        "id": "task-1",
                        "user_scenario": {
                            "instructions": "Name details: given name 静; family name Miller."
                        },
                    }
                ],
            )
        ],
    )
    output = write_localization_review([proposal], tmp_path)
    assert "Name details: given name 静; family name Miller." in output.read_text()

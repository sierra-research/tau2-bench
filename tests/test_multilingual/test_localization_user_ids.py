"""Named customer IDs retain exact references without changing benchmark identity."""

from pathlib import Path

import pytest

from tau2.multilingual.factory.localization_artifact import (
    automatic_localization,
    read_localization_map,
    write_localization_map,
)
from tau2.multilingual.factory.localization_inventory import inventory_localization
from tau2.multilingual.factory.localization_user_ids import (
    derive_user_id_aliases,
    finalize_localization_user_ids,
    localize_user_id_entry,
    remap_user_id_references,
    user_id_alias_mapping,
)
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationEntry,
    LocalizationMap,
    LocalizationSource,
    LocalizedForms,
    RegionalSettings,
    SourceDocument,
    SourceEntry,
    SourceOccurrence,
    UserIdAlias,
    json_content_digest,
)

DATA = Path(__file__).resolve().parents[2] / "data"


def _entry(value, path, *, category="protocol", romanized=None):
    target = value if romanized is None else romanized
    return LocalizationEntry(
        id=json_content_digest([path, value]),
        category=category,
        source=value,
        occurrences=[SourceOccurrence(document_id="database", path=path)],
        context="Offline test fixture.",
        forms=LocalizedForms(native=target, romanized=target, english="Fixture value."),
        operation="preserve" if value == target else "adapt",
        notes="Offline test fixture.",
    )


def _proposal(domain="retail", names=None):
    names = names or [("emma_kim_0097", "Álba", "Gil")]
    entries = []
    database = {"users": {}}
    for user_id, first, last in names:
        database["users"][user_id] = {
            "user_id": user_id,
            "name": {"first_name": "Emma", "last_name": "Kim"},
        }
        entries.extend(
            [
                _entry(
                    source,
                    ["users", user_id, "name", role],
                    category="identity_name",
                    romanized=target,
                )
                for role, source, target in (
                    ("first_name", "Emma", first),
                    ("last_name", "Kim", last),
                )
            ]
        )
        entries.append(_entry(user_id, ["users", user_id, "user_id"]))
    return LocalizationMap(
        revision="test-names",
        settings=RegionalSettings(
            language="es",
            country_code="ES",
            currency_code="EUR",
            currency_scale="1",
            time_zone="Europe/Madrid",
            name_order="given_first",
        ),
        source=LocalizationSource(
            domain=domain,
            documents=[
                SourceDocument(
                    id="database",
                    kind="database",
                    source_path=f"tau2/domains/{domain}/db.json",
                    sha256=json_content_digest(database),
                    content=database,
                )
            ],
            entries=[
                SourceEntry.model_validate(
                    entry.model_dump(include=set(SourceEntry.model_fields))
                )
                for entry in entries
            ],
        ),
        entries=entries,
        provenance=GenerationProvenance(
            model="fixture",
            model_args={},
            prompt_version="fixture",
            call_name="fixture",
            input_sha256="0" * 64,
        ),
    )


@pytest.mark.parametrize("domain", ["airline", "retail"])
def test_ids_use_exact_person_components_suffix_and_not_display_order(domain):
    proposal = _proposal(
        domain,
        [("emma_kim_0097", "Álba", "Gil"), ("emma_kim_3210", "Inés", "Muñoz")],
    )
    proposal.settings.name_order = "family_first"
    assert user_id_alias_mapping(derive_user_id_aliases(proposal)) == {
        "emma_kim_0097": "alba_gil_0097",
        "emma_kim_3210": "ines_munoz_3210",
    }


@pytest.mark.parametrize(
    "names,error",
    [
        ([("emma_kim", "Alba", "Gil")], "numeric suffix"),
        ([("emma_kim_1", "张", "伟")], "ASCII"),
        ([("emma_kim_1", "Alba", "Gil"), ("ada_lee_1", "Álba", "Gil")], "collision"),
        ([("emma_kim_1", "Ada", "Lee"), ("ada_lee_1", "Alba", "Gil")], "overlaps"),
    ],
)
def test_unsafe_name_derivation_is_rejected(names, error):
    with pytest.raises(ValueError, match=error):
        derive_user_id_aliases(_proposal(names=names))


def test_missing_exact_name_occurrence_is_not_guessed_from_other_customers():
    proposal = _proposal()
    proposal.entries = [entry for entry in proposal.entries if entry.source != "Emma"]
    with pytest.raises(ValueError, match="Missing romanized"):
        derive_user_id_aliases(proposal)


def test_reference_rewrite_is_bounded_idempotent_and_preserves_task_ids():
    aliases = [UserIdAlias(canonical="emma_kim_0097", localized="alba_gil_0097")]
    before = {
        "id": "emma_kim_0097",
        "order_id": "emma_kim_0097",
        "initial_state": {
            "initialization_data": {
                "agent_data": {"users": {"emma_kim_0097": {"user_id": "emma_kim_0097"}}}
            },
            "initialization_actions": [{"arguments": {"customer_id": "emma_kim_0097"}}],
        },
        "evaluation_criteria": {
            "actions": [{"arguments": {"user_id": "emma_kim_0097"}}],
            "communicate_info": ["Your ID is emma_kim_0097."],
            "nl_assertions": ["Keep emma_kim_0097x and xemma_kim_0097 unchanged."],
            "reward_basis": ["DB", "NL_ASSERTION"],
        },
        "products": {"emma_kim_0097": {"user_id": "does_not_exist_1111"}},
    }
    after = remap_user_id_references(before, aliases)
    assert before["initial_state"]["initialization_data"]["agent_data"]["users"] == {
        "emma_kim_0097": {"user_id": "emma_kim_0097"}
    }
    assert after["initial_state"]["initialization_data"]["agent_data"]["users"] == {
        "alba_gil_0097": {"user_id": "alba_gil_0097"}
    }
    assert after["initial_state"]["initialization_actions"][0]["arguments"] == {
        "customer_id": "alba_gil_0097"
    }
    assert after["evaluation_criteria"]["communicate_info"] == [
        "Your ID is alba_gil_0097."
    ]
    assert after["evaluation_criteria"]["actions"][0]["arguments"] == {
        "user_id": "alba_gil_0097"
    }
    assert (
        after["evaluation_criteria"]["nl_assertions"]
        == before["evaluation_criteria"]["nl_assertions"]
    )
    assert after["evaluation_criteria"]["reward_basis"] == ["DB", "NL_ASSERTION"]
    assert after["id"] == after["order_id"] == "emma_kim_0097"
    assert after["products"] == before["products"]
    assert remap_user_id_references(after, aliases) == after


def test_structured_protocol_entries_get_visible_adapted_review_rows():
    aliases = [UserIdAlias(canonical="C1001", localized="alba_gil_1001")]
    entry = _entry({"customer_id": "C1001", "line_id": "C1001"}, ["arguments"])
    result = localize_user_id_entry(entry, aliases)
    assert (
        result.forms.native
        == result.forms.romanized
        == {"customer_id": "alba_gil_1001", "line_id": "C1001"}
    )
    assert result.operation == "adapt"
    assert "Customer/user identifier" in result.forms.english
    assert localize_user_id_entry(result, aliases) == result
    assert entry.forms.native["customer_id"] == "C1001"


def test_mixed_protocol_occurrences_cannot_silently_rename_unrelated_ids():
    entry = _entry("C1001", ["customer_id"])
    entry.occurrences.append(SourceOccurrence(document_id="database", path=["id"]))
    with pytest.raises(ValueError, match="mixes customer and unrelated"):
        localize_user_id_entry(
            entry, [UserIdAlias(canonical="C1001", localized="alba_gil_1001")]
        )


def test_existing_users_key_collisions_and_duplicate_aliases_fail():
    alias = UserIdAlias(canonical="C1001", localized="alba_gil_1001")
    with pytest.raises(ValueError, match="Duplicate"):
        user_id_alias_mapping([alias, alias])
    with pytest.raises(ValueError, match="users key"):
        remap_user_id_references({"users": {"C1001": {}, "alba_gil_1001": {}}}, [alias])


@pytest.fixture
def telecom_proposal():
    proposal = _proposal()
    source = inventory_localization("telecom", DATA)
    proposal.source = source
    proposal.entries = [
        automatic_localization(entry, proposal.settings)
        or LocalizationEntry(
            **entry.model_dump(),
            forms=LocalizedForms(
                native=entry.source, romanized=entry.source, english="Fixture."
            ),
            operation="preserve",
            notes="Fixture.",
        )
        for entry in source.entries
    ]
    return proposal


def test_telecom_fullname_ids_compact_roundtrip_and_replay(telecom_proposal, tmp_path):
    result = finalize_localization_user_ids(telecom_proposal)
    assert user_id_alias_mapping(result.user_id_revision.aliases) == {
        "C1001": "john_smith_1001",
        "C1002": "sarah_johnson_1002",
        "C1003": "michael_lee_1003",
        "C1004": "emma_wilson_1004",
    }
    assert result.provenance == telecom_proposal.provenance
    assert result.source == telecom_proposal.source
    assert result.user_id_revision.changed_entry_count == 4
    path = tmp_path / "map.json"
    assert write_localization_map(result, path)
    assert read_localization_map(path, DATA) == result
    assert not write_localization_map(result, path)
    assert finalize_localization_user_ids(result) == result


def test_fresh_factory_finalization_retains_requested_revision(telecom_proposal):
    result = finalize_localization_user_ids(telecom_proposal)
    assert result.revision == telecom_proposal.revision
    assert result.user_id_revision.version == "regional-user-ids-v1"
    assert telecom_proposal.user_id_revision is None
    assert finalize_localization_user_ids(result) == result


def test_name_revision_updates_old_and_canonical_id_references_without_cascades():
    from tau2.multilingual.factory.localization_user_ids import (
        synchronize_revised_user_ids,
    )
    from tau2.multilingual.localization import UserIdRevisionProvenance

    mapping = _proposal(names=[("emma_kim_0097", "Alba", "Gil")])
    old = derive_user_id_aliases(mapping)
    mapping.user_id_revision = UserIdRevisionProvenance(
        parent_map_sha256="a" * 64, aliases=old, changed_entry_count=1
    )
    mapping.entries = [
        localize_user_id_entry(
            automatic_localization(
                SourceEntry.model_validate(
                    e.model_dump(include=set(SourceEntry.model_fields))
                ),
                mapping.settings,
            )
            or e,
            old,
        )
        for e in mapping.entries
    ]
    text = _entry(
        "Your user id is alba_gil_0097. Other token alba_gil_00970.",
        ["instructions"],
        category="prose",
    )
    mapping.entries.append(text)
    mapping.entries[0].forms.native = mapping.entries[0].forms.romanized = "Ines"
    updated = synchronize_revised_user_ids(mapping)
    assert updated.entries[2].forms.native == "ines_gil_0097"
    expected = automatic_localization(mapping.source.entries[2], mapping.settings)
    assert updated.entries[2] == localize_user_id_entry(
        expected, updated.user_id_revision.aliases
    )
    assert (
        updated.entries[-1].forms.native
        == "Your user id is ines_gil_0097. Other token alba_gil_00970."
    )
    assert updated.user_id_revision.aliases == derive_user_id_aliases(updated)
    assert updated.source == mapping.source
    assert synchronize_revised_user_ids(updated) == updated


def test_revised_alias_swaps_keep_customer_ownership_and_unchanged_rows():
    from tau2.multilingual.factory.localization_user_ids import (
        synchronize_revised_user_ids,
    )
    from tau2.multilingual.localization import UserIdRevisionProvenance

    mapping = _proposal(
        names=[
            ("emma_kim_1", "Alba", "Gil"),
            ("ada_lee_1", "Ines", "Munoz"),
            ("mei_li_2", "Ana", "Sanz"),
        ]
    )
    old = derive_user_id_aliases(mapping)
    mapping.user_id_revision = UserIdRevisionProvenance(
        parent_map_sha256="a" * 64, aliases=old, changed_entry_count=3
    )
    mapping.entries = [
        localize_user_id_entry(
            automatic_localization(
                SourceEntry.model_validate(
                    e.model_dump(include=set(SourceEntry.model_fields))
                ),
                mapping.settings,
            )
            or e,
            old,
        )
        for e in mapping.entries
    ]
    for index, name in [(0, "Ines"), (1, "Munoz"), (3, "Alba"), (4, "Gil")]:
        mapping.entries[index].forms.native = mapping.entries[index].forms.romanized = (
            name
        )
    updated = synchronize_revised_user_ids(mapping)
    assert updated.entries[2].forms.native == "ines_munoz_1"
    assert updated.entries[5].forms.native == "alba_gil_1"
    assert updated.entries[6:] == mapping.entries[6:]

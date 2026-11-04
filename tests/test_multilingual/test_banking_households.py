"""Household repairs require source evidence within the same task world."""

import pytest

from tau2.domains.banking_knowledge.localization_households import (
    harmonize_banking_households,
    validate_banking_households,
)
from tau2.multilingual.localization import EntityReuseProvenance, SourceOccurrence
from tau2.utils.artifacts import ReviewApproval
from tests.test_multilingual.test_banking_prose import entry, proposal


def address(identifier, user, task, native, source="12 Main Street"):
    value = entry(identifier, source, native, category="address", task=task, owner=user)
    value.occurrences = [
        SourceOccurrence(
            document_id="tasks",
            path=[
                task,
                "initial_state",
                "initialization_data",
                "agent_data",
                "users",
                "data",
                user,
                "address",
            ],
        )
    ]
    return value


def test_shared_household_reuses_stable_donor_and_records_lineage():
    donor = address("a", "parent", 83, "Calle del Olmo 12")
    donor.reuse = EntityReuseProvenance(
        source_domain="retail",
        source_entry_ids=["donor-address"],
        source_artifact_sha256="b" * 64,
        approval=ReviewApproval(
            content_sha256="b" * 64, reviewed_by="Synthetic test fixture"
        ),
    )
    child = address("b", "child", 83, "Calle del Álamo 20")
    original = proposal(child, donor)
    with pytest.raises(ValueError, match="household address differs"):
        validate_banking_households(original)
    result = harmonize_banking_households(original)
    assert result.proposal.entries[0].forms == donor.forms
    assert result.proposal.entries[0].reuse == donor.reuse
    assert original.entries[0].forms.native == "Calle del Álamo 20"
    assert result.proposal.provenance == original.provenance
    assert result.proposal.status == "draft"
    assert len(result.revision.changes) == 1
    change = result.revision.changes[0]
    assert change.entry_id == "b"
    assert change.donor_entry_id == "a"
    assert change.task_indices == [83]
    assert change.previous_forms.native == "Calle del Álamo 20"
    again = harmonize_banking_households(result.proposal)
    assert not again.revision.changes
    assert again.proposal == result.proposal
    assert again.revision.parent_map_sha256 == result.revision.output_map_sha256


def test_identical_source_addresses_in_different_task_worlds_do_not_merge():
    left = address("a", "parent", 1, "Calle Uno 1")
    right = address("b", "child", 2, "Calle Dos 2")
    original = proposal(left, right)
    result = harmonize_banking_households(original)
    assert result.proposal == original
    assert not result.revision.changes


def test_household_rules_ignore_different_addresses_and_nonidentity_occurrences():
    left = address("a", "parent", 1, "Calle Uno 1")
    other = address("b", "child", 1, "Calle Dos 2", source="13 Main Street")
    unrelated = entry("c", "12 Main Street", "Calle Tres 3", "address", task=1)
    original = proposal(left, other, unrelated)
    assert harmonize_banking_households(original).proposal == original


def test_disjoint_households_with_identical_source_strings_stay_independent():
    original = proposal(
        address("a", "a", 1, "Calle Uno 1"),
        address("b", "b", 1, "Calle Dos 2"),
        address("c", "a", 2, "Calle Tres 3"),
        address("d", "b", 2, "Calle Cuatro 4"),
    )
    result = harmonize_banking_households(original)
    assert [e.forms.native for e in result.proposal.entries] == [
        "Calle Uno 1",
        "Calle Uno 1",
        "Calle Tres 3",
        "Calle Tres 3",
    ]

"""English banking instructions permit entity substitutions, not paraphrase."""

import json

from tau2.domains.banking_knowledge.localization_prose import (
    audit_banking_prose,
    reconstruct_banking_prose,
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
)


def entry(
    identifier,
    source,
    native,
    category="prose",
    task=0,
    owner="task:0",
    role="instructions",
    document="tasks",
    romanized=None,
):
    return LocalizationEntry(
        id=identifier,
        category=category,
        source=source,
        occurrences=[SourceOccurrence(document_id=document, path=[task, identifier])],
        context=f"role={role}; entity={owner}; banking=test.",
        forms=LocalizedForms(
            native=native, romanized=romanized or native, english="Test."
        ),
        operation="preserve" if source == native else "adapt",
        notes="Test.",
    )


def proposal(*entries):
    return LocalizationMap(
        revision="test",
        settings=RegionalSettings(
            language="es",
            country_code="ES",
            currency_code="USD",
            currency_scale="1",
            time_zone="America/New_York",
            name_order="given_first",
        ),
        source=LocalizationSource(
            domain="banking_knowledge",
            documents=[
                SourceDocument(
                    id="tasks",
                    kind="tasks",
                    source_path="tasks.json",
                    sha256="a" * 64,
                    content={},
                )
            ],
            entries=[
                SourceEntry.model_validate(
                    e.model_dump(include=set(SourceEntry.model_fields))
                )
                for e in entries
            ],
        ),
        entries=list(entries),
        provenance=GenerationProvenance(
            model="test",
            model_args={},
            prompt_version="test",
            call_name="test",
            input_sha256="a" * 64,
        ),
    )


def test_exact_entity_substitution_preserves_every_other_character():
    name = entry("name", "Marcus Chen", "Ana López", "identity_name")
    text = entry(
        "text",
        "Marcus Chen must NOT pay $25.\nKeep the card.",
        "Ana López must NOT pay $25.\nKeep the card.",
    )
    assert audit_banking_prose(proposal(name, text)) == []
    text.forms.native = "Ana López must pay $25.\nKeep the card."
    finding = audit_banking_prose(proposal(name, text))[0]
    assert finding.reason == "unverified_edit"
    assert finding.expected == "Ana López must NOT pay $25.\nKeep the card."


def test_other_task_identity_does_not_change_current_task_reference():
    base = entry(
        "base",
        "Marcus Chen",
        "Ana López",
        "identity_name",
        document="database",
        owner="base",
    )
    other = entry(
        "other", "Marcus Chen", "Luis García", "identity_name", task=1, owner="other"
    )
    text = entry("text", "Marcus Chen calls.", "Ana López calls.")
    assert audit_banking_prose(proposal(base, other, text)) == []


def test_ambiguous_same_task_reference_is_not_guessed():
    left = entry("left", "Marcus Chen", "Ana López", "identity_name", owner="left")
    right = entry("right", "Marcus Chen", "Luis García", "identity_name", owner="right")
    text = entry("text", "Marcus Chen calls.", "Ana López calls.")
    findings = audit_banking_prose(proposal(left, right, text))
    assert findings[0].reason == "ambiguous_reference"
    assert findings[0].expected is None


def test_partial_names_are_explicit_uncertainties_not_silent_preservation():
    name = entry("name", "Marcus Chen", "Ana López", "identity_name")
    text = entry("text", "Marcus calls.", "Marcus calls.")
    assert audit_banking_prose(proposal(name, text))[0].reason == "unresolved_alias"


def test_complete_entity_boundaries_do_not_replace_substrings():
    name = entry("name", "Ann", "Ana", "identity_name")
    text = entry("text", "Anniversary fee remains $25.", "Anniversary fee remains $25.")
    assert audit_banking_prose(proposal(name, text)) == []


def test_encoded_arguments_preserve_protocol_values_even_when_they_match_an_entity():
    contact = entry("contact", "CODE123", "new@example.com", "contact", role="email")
    text = entry(
        "text",
        json.dumps({"email": "CODE123", "user_id": "CODE123", "amount": 25}),
        json.dumps(
            {"email": "new@example.com", "user_id": "CODE123", "amount": 25}, indent=2
        ),
        role="arguments",
    )
    assert audit_banking_prose(proposal(contact, text)) == []
    text.forms.native = json.dumps(
        {"email": "new@example.com", "user_id": "new@example.com", "amount": 25}
    )
    assert audit_banking_prose(proposal(contact, text))[0].reason == "unverified_edit"


def test_romanized_form_is_checked_independently():
    name = entry(
        "name", "Marcus Chen", "김민준", "identity_name", romanized="Gim Minjun"
    )
    text = entry(
        "text",
        "Marcus Chen calls.",
        "김민준 calls.",
        romanized="Gim Minjun calls differently.",
    )
    findings = audit_banking_prose(proposal(name, text))
    assert len(findings) == 1
    assert findings[0].representation == "romanized"


def test_uppercase_merchant_matches_preserve_all_description_clues():
    merchant = entry("merchant", "Target", "Mercado Sol", "other")
    text = entry(
        "text",
        "TARGET STORE #2847 SAN DIEGO CA",
        "Mercado Sol STORE #2847 SAN DIEGO CA",
    )
    assert audit_banking_prose(proposal(merchant, text)) == []
    text.forms.native = "Mercado Sol"
    assert audit_banking_prose(proposal(merchant, text))[0].expected == (
        "Mercado Sol STORE #2847 SAN DIEGO CA"
    )


def test_casefold_collision_requires_resolution():
    left = entry("left", "Target", "Mercado Sol", "other")
    right = entry("right", "TARGET", "Mercado Luna", "other")
    text = entry("text", "TARGET STORE #2847", "Mercado Sol STORE #2847")
    finding = audit_banking_prose(proposal(left, right, text))[0]
    assert finding.reason == "ambiguous_reference"
    assert finding.expected is None


def test_preserved_entity_keeps_exact_source_surface():
    merchant = entry("merchant", "Target", "Target", "other")
    text = entry("text", "TARGET STORE #2847", "TARGET STORE #2847")
    assert audit_banking_prose(proposal(merchant, text)) == []


def test_deterministic_repair_is_idempotent_and_preserves_generation_lineage():
    name = entry("name", "Marcus Chen", "Ana López", "identity_name")
    text = entry("text", "Marcus Chen must NOT pay $25.", "Ana López must pay $25.")
    uncertain = entry("uncertain", "Marcus calls.", "Marcus calls.")
    original = proposal(name, text, uncertain)
    result = reconstruct_banking_prose(original)
    assert result.proposal.entries[1].forms.native == "Ana López must NOT pay $25."
    assert original.entries[1].forms.native == "Ana López must pay $25."
    assert result.proposal.provenance == original.provenance
    assert result.proposal.status == "draft"
    assert len(result.reconstruction.changes) == 2
    assert len(result.reconstruction.unresolved) == 2
    assert result.proposal.entries[2] == uncertain
    again = reconstruct_banking_prose(result.proposal)
    assert again.proposal == result.proposal
    assert again.reconstruction.changes == []
    assert (
        again.reconstruction.parent_map_sha256
        == result.reconstruction.output_map_sha256
    )
    assert (
        again.reconstruction.output_map_sha256
        == result.reconstruction.output_map_sha256
    )

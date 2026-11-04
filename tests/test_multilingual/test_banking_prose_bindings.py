"""Source-bound supplemental entities do not permit arbitrary prose rewriting."""

import pytest

from tau2.domains.banking_knowledge.localization_bindings import validate_prose_bindings
from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding
from tests.test_multilingual.test_banking_prose import entry, proposal


def test_binding_rejects_unrelated_source_and_unrecorded_authoring():
    text = entry("text", "Ask Marcus.", "Ask Ana.")
    item = ProseEntityBinding(
        entry_id="text",
        source="Marcus",
        forms=LocalizedForms(native="Ana", romanized="Ana", english="Given name."),
        kind="generated",
        reason="Incidental speaker.",
    )
    with pytest.raises(ValueError, match="provenance"):
        validate_prose_bindings(
            proposal(text).model_copy(update={"prose_bindings": [item]})
        )


def test_reuse_binding_requires_exact_existing_forms():
    name = entry("person", "Marcus Chen", "Ana Lopez", "identity_name")
    text = entry("text", "Ask Marcus.", "Ask Ana Lopez.")
    item = ProseEntityBinding(
        entry_id="text",
        source="Marcus",
        forms=name.forms,
        source_entry_id="person",
        kind="reuse",
        reason="Explicit reference to the same person.",
    )
    validate_prose_bindings(
        proposal(name, text).model_copy(update={"prose_bindings": [item]})
    )
    item.forms = LocalizedForms(native="Invented", romanized="Invented", english="Bad")
    with pytest.raises(ValueError, match="exact"):
        validate_prose_bindings(
            proposal(name, text).model_copy(update={"prose_bindings": [item]})
        )


def test_protected_binding_cannot_change_protocol_or_geography():
    text = entry("text", "Seattle, Washington.", "Seattle, Washington.")
    binding = ProseEntityBinding(
        entry_id="text",
        source="Seattle, Washington",
        forms=LocalizedForms(native="Madrid", romanized="Madrid", english="Bad"),
        kind="protected",
        reason="Geography is a rule-bearing source detail.",
    )
    with pytest.raises(ValueError, match="Protected"):
        validate_prose_bindings(
            proposal(text).model_copy(update={"prose_bindings": [binding]})
        )


def test_explicit_daughter_alias_reuses_whole_scoped_name():
    from tau2.domains.banking_knowledge.localization_bindings import (
        bind_explicit_person_aliases,
    )

    name = entry("daughter", "Sofia Martinez", "김 수빈", "identity_name")
    text = entry("text", "Your daughter Sofia calls. SOFIA'S CARD.", "unchanged")
    p = bind_explicit_person_aliases(proposal(name, text))
    assert p.prose_bindings[0].source == "Sofia"
    assert p.prose_bindings[0].forms.native == "김 수빈"
    validate_prose_bindings(p)

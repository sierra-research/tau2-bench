"""Only explicit caller statements inherit names from the identified DB record."""

import pytest
from test_localization_user_ids import _entry, _proposal

from tau2.multilingual.factory.localization_caller_names import (
    finalize_localization_caller_names,
    localized_caller_entries,
)
from tau2.multilingual.localization import SourceEntry


def _with_statement(*, source=None, native=None, romanized=None):
    proposal = _proposal()
    source = source or (
        "You are Emma Kim.\nYour user id is emma_kim_0097.\n"
        "Your passenger is Marta Castro.\nFirst try wrong ZIP 12345."
    )
    native = native or (
        "You are Marta Castro.\nYour user id is emma_kim_0097.\n"
        "Your passenger is Marta Castro.\nFirst try wrong ZIP 54321."
    )
    entry = _entry(
        source,
        [0, "user_scenario", "instructions", "known_info"],
        category="prose",
        romanized=native,
    )
    entry.forms.romanized = romanized or native
    proposal.entries.append(entry)
    proposal.source.entries.append(
        SourceEntry.model_validate(
            entry.model_dump(include=set(SourceEntry.model_fields))
        )
    )
    return proposal


def test_only_caller_span_changes_and_third_party_wrong_first_text_stays_exact():
    proposal = _with_statement()
    old = proposal.entries[-1]
    result = finalize_localization_caller_names(proposal)
    entry = result.entries[-1]
    assert entry.forms.native == old.forms.native.replace(
        "You are Marta Castro.", "You are Álba Gil.", 1
    )
    assert entry.forms.romanized == old.forms.romanized.replace(
        "You are Marta Castro.", "You are Álba Gil.", 1
    )
    assert "Your passenger is Marta Castro." in entry.forms.native
    assert "First try wrong ZIP 54321." in entry.forms.native
    assert result.caller_name_revision.changed_entry_count == 1
    assert result.source == proposal.source
    assert result.provenance == proposal.provenance
    assert finalize_localization_caller_names(result) == result
    assert localized_caller_entries(result) == result.entries


def test_family_first_native_and_romanized_use_declared_order_and_separator():
    proposal = _with_statement(
        native="You are 陈雨桐.\nYour user id is emma_kim_0097.",
        romanized="You are Chen Yutong.\nYour user id is emma_kim_0097.",
    )
    proposal.settings.name_order = "family_first"
    first, last = proposal.entries[:2]
    first.forms.native, first.forms.romanized = "婷婷", "Tingting"
    last.forms.native, last.forms.romanized = "金", "Jin"
    result = localized_caller_entries(proposal)[-1]
    assert result.forms.native == "You are 金婷婷.\nYour user id is emma_kim_0097."
    assert (
        result.forms.romanized
        == "You are Jin Tingting.\nYour user id is emma_kim_0097."
    )


def test_alternative_anchored_name_form_keeps_all_non_name_bytes():
    source = "Your name is Emma Kim.\n\nYour user id is: emma_kim_0097. \nConfirmation: AB12."
    target = source.replace("Emma Kim", "Marta Castro")
    result = localized_caller_entries(_with_statement(source=source, native=target))[-1]
    assert result.forms.native == target.replace("Marta Castro", "Álba Gil")


@pytest.mark.parametrize(
    "source,target,error",
    [
        (
            "You are Emma Kim. Your user id is emma_kim_0097. Your user id is someone_99.",
            None,
            "Ambiguous source",
        ),
        ("You are Emma Kim. Your user id is missing_99.", None, "no customer"),
        (
            "You are Somebody Else. Your user id is emma_kim_0097.",
            None,
            "requires review",
        ),
        (
            None,
            "Your passenger is Marta Castro. Your user id is emma_kim_0097.",
            "missing or ambiguous",
        ),
        (
            None,
            "You are Marta Castro. Your user id is someone_99.",
            "missing or ambiguous",
        ),
    ],
)
def test_ambiguous_or_deliberately_different_identity_requires_review(
    source, target, error
):
    with pytest.raises(ValueError, match=error):
        localized_caller_entries(_with_statement(source=source, native=target))


def test_non_caller_prose_and_ids_without_name_claim_are_untouched():
    for source in (
        "Your passenger is Emma Kim. Your user id is emma_kim_0097.",
        "You are emma_kim_0097. Your passenger is Emma Kim.",
        "You are Emma Kim, travelling with your friend.",
    ):
        proposal = _with_statement(source=source, native=source)
        assert localized_caller_entries(proposal) == proposal.entries


def test_already_synchronized_metadata_does_not_hide_later_name_drift():
    result = finalize_localization_caller_names(_with_statement())
    result.entries[-1].forms.native = result.entries[-1].forms.native.replace(
        "Álba Gil", "Marta Castro"
    )
    with pytest.raises(ValueError, match="Previously synchronized"):
        finalize_localization_caller_names(result)


def test_review_explanations_use_corrected_full_names_without_losing_other_facts():
    proposal = _with_statement(
        source="You are Emma Kim. Your user id is emma_kim_0097.",
        native="You are Marta Castro. Your user id is emma_kim_0097.",
    )
    proposal.entries[
        -1
    ].forms.english = "The localized display name is Marta Castro; reservation AB12 remains unchanged."
    proposal.entries[
        -1
    ].notes = "Localized the caller as Marta Castro; retain the 10-hour claim."
    result = localized_caller_entries(proposal)[-1]
    assert (
        result.forms.english
        == "The localized display name is Álba Gil; reservation AB12 remains unchanged."
    )
    assert "caller as Álba Gil; retain the 10-hour claim." in result.notes


def test_ambiguous_third_party_name_in_review_explanation_is_not_guessed():
    proposal = _with_statement()
    proposal.entries[-1].forms.english = "Marta Castro is travelling."
    with pytest.raises(ValueError, match="Ambiguous caller-name review"):
        localized_caller_entries(proposal)

"""Phone decoys retain actual source error shapes rather than model counting claims."""

from pathlib import Path

import pytest

from tau2.multilingual.factory.intake_authoring import validate_intake_forms
from tau2.multilingual.factory.intake_phone_invariants import (
    phone_decoy_length_evidence,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.multilingual.localization import LocalizedForms
from tau2.utils.utils import DATA_DIR


@pytest.mark.parametrize(
    "source_primary,source_decoy,current_primary,current_decoy,valid",
    [
        (
            "+33 6 39 98 06 51",
            "+33 6 39 98 27 51",
            "+34 639 980 651",
            "+34 639 980 27 651",
            False,
        ),
        (
            "+33 6 39 98 06 51",
            "+33 6 39 98 27 51",
            "+34 639 980 651",
            "+34 639 982 751",
            True,
        ),
        ("(605) 555-0117", "(605) 555-0167", "010-3847-2196", "010-3847-2169", True),
        ("(605) 555-0117", "(605) 555-011", "639 980 651", "639 980 65", True),
        ("(605) 555-0117", "(605) 555-011", "639 980 651", "639 980 657", False),
        (
            "555-1234 ext. 7",
            "555-1243 ext. 7",
            "+34 91 482 6367 x 12",
            "+34 91 482 6376 ext. 123",
            False,
        ),
        (
            "555-1234 ext. 7",
            "555-1243 ext. 78",
            "+34 91 482 6367 x 12",
            "+34 91 482 6376 ext. 123",
            True,
        ),
        (
            "555-1234 ext. 7",
            "555-1243",
            "+34 91 482 6367 x 123",
            "+34 91 482 6376",
            True,
        ),
        (
            "555-1234 ext. 7",
            "555-1243",
            "+34 91 482 6367 x 123",
            "+34 91 482 6376 x 12",
            False,
        ),
    ],
)
def test_relative_main_and_extension_errors(
    source_primary, source_decoy, current_primary, current_decoy, valid
):
    evidence = phone_decoy_length_evidence(
        source_primary, source_decoy, current_primary, current_decoy
    )
    assert evidence.preserves_source_error_shape is valid


@pytest.fixture(scope="module")
def spanish_map():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/es/localization/intake.json"
    )


def phone_entries(proposal, task_id="intake_phones_hard_08"):
    data = next(
        d.content for d in proposal.source.documents if d.id == "intake_localization"
    )
    index = next(
        i
        for i, row in enumerate(data["stimuli"])
        if row["canonical_task_id"] == task_id
    )

    def field(name):
        return next(
            entry
            for entry in proposal.entries
            if any(
                point.document_id == "intake_localization"
                and point.path == ["stimuli", index, name]
                for point in entry.occurrences
            )
        )

    return field("written"), field("decoys")


def test_phone_decoy_batch_boundary_uses_exact_completed_primary(spanish_map):
    proposal = spanish_map.model_copy(deep=True)
    primary, decoys = phone_entries(proposal)
    primary.forms = LocalizedForms(
        english="Test phone fixture",
        native="+34 639 980 651",
        romanized="+34 639 980 651",
    )
    good = LocalizedForms(
        english="Test decoy fixture",
        native=["+34 639 982 751"],
        romanized=["+34 639 982 751"],
    )
    validate_intake_forms(decoys, good, [primary], "es")
    bad = good.model_copy(deep=True)
    bad.native = ["+34 639 980 27 651"]
    with pytest.raises(ValueError, match="source-relative dialing/extension"):
        validate_intake_forms(decoys, bad, [primary], "es")
    # The exact same text in a different selected family is not a phone role.
    decoys.context = decoys.context.replace(
        "intake_phones_hard_08", "intake_properties_hard_08"
    )
    validate_intake_forms(decoys, bad, [primary], "es")


@pytest.mark.parametrize("marker", ["ext.", "ext", "x", "#", "ramal"])
def test_existing_extension_markers_do_not_change_digit_shape(marker):
    evidence = phone_decoy_length_evidence(
        "555-1234 ext. 7",
        "555-1243 ext. 7",
        f"02-2148-0117{marker}12",
        f"02-2148-0167{marker}12",
    )
    assert evidence.preserves_source_error_shape


def test_main_and_extension_errors_cannot_cancel_each_other():
    evidence = phone_decoy_length_evidence(
        "555-1234 ext. 12", "555-1243 ext. 21", "915284736 ext. 12", "9152847367 ext. 2"
    )
    assert not evidence.preserves_source_error_shape


@pytest.mark.parametrize(
    "current", ["02-2148-0167 내선 7번", "02-2148-0167 naeseon 7beon"]
)
def test_existing_local_extension_labels_do_not_look_like_extra_main_digits(current):
    evidence = phone_decoy_length_evidence(
        "(605) 555-0117 ext. 7", "(605) 555-0167 ext. 7", "02-2148-0117 ext. 7", current
    )
    assert evidence.preserves_source_error_shape

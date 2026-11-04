"""Source-role coverage and recorded gender lineage of published intake maps."""

from collections import Counter
from pathlib import Path

import pytest

from tau2.multilingual.factory.intake_gender import (
    intake_name_bindings,
    validate_intake_gender_coverage,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils import DATA_DIR
from tau2.utils.artifacts import content_digest
from tests.test_multilingual.reviewed_corrections import (
    assert_review_applied,
    load_review,
)

REVIEW_FIXTURES = Path(__file__).parent / "fixtures/intake_review"


@pytest.fixture(scope="module")
def proposal():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/es/localization/intake.json"
    )


def test_all_used_names_have_separate_source_roles(proposal):
    bindings = intake_name_bindings(proposal, Path(DATA_DIR))
    assert Counter(b.role for b in bindings) == {
        "callee": 39,
        "measured_name": 20,
        "name_decoy": 40,
    }
    assert len({b.id for b in bindings}) == 99
    entries = {e.id: e for e in proposal.entries}
    lisa = [b for b in bindings if entries[b.entry_id].source == "Lisa Sandoval"]
    assert {b.role for b in lisa} == {"callee", "measured_name"}
    assert {b.source_gender for b in lisa} == {"female"}
    # The frozen bank labels the primary, not the independent wrong-name decoy.
    gisela = next(
        b
        for b in bindings
        if b.index is not None
        and entries[b.entry_id].source[b.index] == "Gisela Lansbury-Cutugno"
    )
    assert gisela.source_gender == "unknown"
    assert "never copy the primary" in gisela.source_evidence


def test_owned_email_uses_final_callee_and_keeps_frozen_pattern(proposal):
    owner = next(
        e
        for e in proposal.entries
        if e.category == "identity_name" and e.source == "William Larson"
    )
    email = next(
        e for e in proposal.entries if e.source == "william.larson@quorvenhaven.net"
    )
    assert owner.forms.native == "Javier León"
    assert email.forms.native == "javier.leon@naverra.es"
    assert email.forms.romanized == email.forms.native


def test_inventory_exposes_reviewed_gender_to_authoring(proposal):
    named = [e for e in proposal.source.entries if e.category == "identity_name"]
    assert named
    assert all("reviewed_source_gender=" in e.context for e in named)


def test_publication_refuses_missing_name_assessment(proposal):
    candidate = proposal.model_copy(deep=True)
    entry = next(e for e in candidate.entries if e.category == "identity_name")
    entry.name_gender = None
    with pytest.raises(ValueError, match="requires source-gender evidence"):
        validate_intake_gender_coverage(candidate, Path(DATA_DIR))


def test_publication_refuses_gender_evidence_that_changes_bank_label(proposal):
    from tau2.multilingual.localization import NameGenderEvidence

    candidate = proposal.model_copy(deep=True)
    entry = next(e for e in candidate.entries if e.category == "identity_name")
    binding = next(
        b
        for b in intake_name_bindings(candidate, Path(DATA_DIR))
        if b.entry_id == entry.id
    )
    wrong = "male" if binding.source_gender == "female" else "female"
    entry.name_gender = NameGenderEvidence(
        source_gender=wrong,
        target_gender=wrong,
        source_basis="reviewed_metadata",
        source_evidence="Deliberately conflicting test evidence.",
        reason="Test fixture.",
        forms_sha256=content_digest(entry.forms),
        model="test",
        prompt_version="test-v1",
    )
    with pytest.raises(ValueError, match="differs from frozen bank"):
        validate_intake_gender_coverage(candidate, Path(DATA_DIR))


def test_email_owner_resolution_respects_explicit_family_first_order():
    candidate = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/ko/localization/intake.json"
    )
    owner = next(
        e
        for e in candidate.entries
        if e.category == "identity_name" and e.source == "William Larson"
    )
    family, given = owner.forms.romanized.lower().split()
    email = next(
        e for e in candidate.entries if e.source == "william.larson@quorvenhaven.net"
    )
    assert email.forms.native.split("@")[0] == f"{given}.{family}"


@pytest.mark.parametrize("language", ["hi", "es", "pt", "zh", "ko"])
def test_published_intake_map_keeps_gender_repair_lineage_and_review(language):
    candidate = read_localization_map(
        Path(DATA_DIR) / f"tau2/multilingual/{language}/localization/intake.json"
    )
    evidence = REVIEW_FIXTURES / language
    repair = load_review(evidence / "intake.gender-preservation.json")
    assert repair["version"] == "intake-gender-preservation-v1"
    assert repair["prior_audits_stale"]
    assert repair["native_review_required"]
    assert not repair["review_exports_updated"]
    if language in {"ko", "zh"}:
        # Approved native corrections were applied after the gender repair.
        assert repair["candidate_map_sha256"] != content_digest(candidate)
        assert_review_applied(
            candidate, load_review(evidence / "approved-corrections.json")
        )
    elif candidate.address_revision is None:
        assert repair["candidate_map_sha256"] == content_digest(candidate)
    else:
        assert (
            repair["candidate_map_sha256"]
            == candidate.address_revision.parent_map_sha256
        )
    validate_intake_gender_coverage(candidate, Path(DATA_DIR))

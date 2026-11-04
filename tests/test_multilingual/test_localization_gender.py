"""Gender preservation uses source evidence, not caller voice or donor approval."""

import json
from types import SimpleNamespace

import pytest

from tau2.multilingual.factory.localization_gender import (
    GenderNameRequest,
    GenderNameResult,
    assess_source_names,
    audit_gender_names,
    evidence_for,
    validate_gender_result,
)
from tau2.multilingual.localization import LocalizationEntry, LocalizedForms


def request(gender="male"):
    return GenderNameRequest(
        id="person-owner-a",
        source="Marcus Chen",
        forms=LocalizedForms(
            native="Ana Rubio", romanized="Ana Rubio", english="Existing candidate."
        ),
        context=["Source owner A; exact source name Marcus Chen."],
        source_gender=gender,
        source_basis="reviewed_metadata",
        source_evidence="Fixture source metadata.",
    )


def result(item, gender="male", decision="revise"):
    forms = (
        item.forms.model_dump()
        if decision != "revise"
        else dict(
            native="Marcos Rubio",
            romanized="Marcos Rubio",
            english="Male source identity with retained family name.",
        )
    )
    return GenderNameResult(
        id=item.id,
        source_gender=item.source_gender,
        target_gender=gender,
        decision=decision,
        forms=forms,
        reason="Fixture compatibility assessment, native review pending.",
    )


def test_gender_evidence_rejects_changed_names_and_opposite_gender():
    item = request()
    revised = result(item)
    evidence = evidence_for(item, revised, "test")
    evidence.require(revised.forms)
    with pytest.raises(ValueError, match="no longer matches"):
        evidence.require(item.forms)
    with pytest.raises(ValueError, match="known source gender"):
        validate_gender_result(item, result(item, gender="female"))
    with pytest.raises(ValueError, match="changed source"):
        validate_gender_result(
            item, revised.model_copy(update={"source_gender": "female"})
        )


def test_unisex_target_is_compatible_but_unknowns_cannot_be_silently_assigned():
    item = request()
    validate_gender_result(item, result(item, gender="unisex", decision="keep"))
    for gender in ("unisex", "unknown"):
        with pytest.raises(ValueError, match="ambiguous source"):
            validate_gender_result(request(gender), result(request(gender)))
    with pytest.raises(ValueError, match="unresolved"):
        validate_gender_result(item, result(item, gender="unknown", decision="keep"))
    with pytest.raises(ValueError, match="changed localized forms"):
        validate_gender_result(
            item, result(item).model_copy(update={"decision": "keep"})
        )


def test_known_metadata_bypasses_source_inference_and_cache_binds_exact_request(
    tmp_path, monkeypatch
):
    import tau2.utils.llm_utils as llm

    item = request()

    def forbidden(**kwargs):
        raise AssertionError("Reviewed source metadata must not be re-inferred")

    monkeypatch.setattr(llm, "generate", forbidden)
    source = assess_source_names([item], "test", {}, tmp_path, 1)[0]
    assert source.gender == "male" and source.basis == "reviewed_metadata"
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=result(item).model_dump_json())

    monkeypatch.setattr(llm, "generate", generate)
    first = audit_gender_names([item], "Spanish", "Spain", "test", {}, tmp_path, 1)
    assert (
        audit_gender_names([item], "Spanish", "Spain", "test", {}, tmp_path, 1) == first
    )
    assert len(calls) == 1 and calls[0]["call_name"] == "factory_localized_name_gender"
    assert (
        calls[0]["response_format"]["json_schema"]["schema"]["$defs"][
            "GenderNameForms"
        ]["properties"]["native"]["type"]
        == "string"
    )
    item.context.append("New exact task evidence")
    audit_gender_names([item], "Spanish", "Spain", "test", {}, tmp_path, 1)
    assert len(calls) == 2


def test_source_assessment_never_sees_target_name(tmp_path, monkeypatch):
    import tau2.utils.llm_utils as llm

    item = request().model_copy(
        update={"source_basis": "unknown", "source_gender": "unknown"}
    )

    def generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        assert "forms" not in payload["entry"]
        assert "Ana Rubio" not in kwargs["messages"][1].content
        return SimpleNamespace(
            content=json.dumps(
                dict(
                    id=item.id,
                    kind="person",
                    gender="male",
                    basis="name_convention",
                    reason="Qualified conventional source-name inference.",
                )
            )
        )

    monkeypatch.setattr(llm, "generate", generate)
    assert (
        assess_source_names([item], "test", {}, tmp_path, 1)[0].basis
        == "name_convention"
    )


def test_gender_evidence_roundtrips_compact_authored_entry():
    from tau2.multilingual.factory.localization_artifact import AuthoredLocalization
    from tests.test_multilingual.test_banking_prose import entry

    item = request()
    revised = result(item)
    value = entry("name", "Marcus Chen", "Marcos Rubio", category="identity_name")
    value.forms = LocalizedForms.model_validate(revised.forms.model_dump())
    value.name_gender = evidence_for(item, revised, "test")
    compact = AuthoredLocalization.model_validate(
        value.model_dump(include=set(AuthoredLocalization.model_fields))
    )
    assert compact.name_gender == value.name_gender
    assert (
        LocalizationEntry.model_validate_json(value.model_dump_json()).name_gender
        == value.name_gender
    )
    payload = compact.model_dump()
    payload["forms"]["native"] = "Ana Rubio"
    with pytest.raises(ValueError, match="no longer matches"):
        AuthoredLocalization.model_validate(payload)


def test_publication_requires_assessment_coverage_without_fabricating_native_approval():
    from tau2.multilingual.factory.localization_gender import (
        require_name_gender_coverage,
    )
    from tests.test_multilingual.test_banking_prose import entry, proposal

    value = entry("name", "Marcus Chen", "Marcos Rubio", category="identity_name")
    mapping = proposal(value)
    with pytest.raises(ValueError, match="assessment missing"):
        require_name_gender_coverage(mapping)
    item = request()
    revised = result(item)
    value.forms = LocalizedForms.model_validate(revised.forms.model_dump())
    value.name_gender = evidence_for(item, revised, "test")
    require_name_gender_coverage(mapping)
    assert value.name_gender.status == "pending_native_review"


def test_each_name_decoy_has_independent_gender_and_exact_index_binding():
    from tau2.multilingual.localization import NameGenderItemEvidence
    from tests.test_multilingual.test_banking_prose import entry

    male_request = request()
    female_request = request("female").model_copy(
        update={"id": "female-decoy", "source": "Jane Chen"}
    )
    male = result(male_request)
    female = result(female_request, gender="female").model_copy(
        update={
            "forms": result(female_request, gender="female").forms.model_copy(
                update={"native": "Clara Rubio", "romanized": "Clara Rubio"}
            )
        }
    )
    value = entry(
        "decoys",
        ["Marcus Chen", "Jane Chen"],
        ["Marcos Rubio", "Clara Rubio"],
        category="other",
    )
    value.forms.romanized = ["Marcos Rubio", "Clara Rubio"]
    for index, (source, target) in enumerate(
        ((male_request, male), (female_request, female))
    ):
        evidence = evidence_for(source, target, "test")
        from tau2.utils.artifacts import content_digest

        evidence.forms_sha256 = content_digest(
            LocalizedForms(
                native=value.forms.native[index],
                romanized=value.forms.romanized[index],
                english=value.forms.english,
            )
        )
        value.name_gender_items.append(
            NameGenderItemEvidence(index=index, evidence=evidence)
        )
    restored = LocalizationEntry.model_validate_json(value.model_dump_json())
    assert [item.evidence.source_gender for item in restored.name_gender_items] == [
        "male",
        "female",
    ]
    restored.forms.native.reverse()
    with pytest.raises(ValueError, match="no longer matches"):
        LocalizationEntry.model_validate(restored.model_dump())


def test_offline_audit_never_calls_model_and_requires_exact_checkpoint(
    tmp_path, monkeypatch
):
    import tau2.utils.llm_utils as llm

    def forbidden(**kwargs):
        raise AssertionError("Offline finalization must not call a provider")

    monkeypatch.setattr(llm, "generate", forbidden)
    with pytest.raises(ValueError, match="Offline gender audit lacks exact checkpoint"):
        audit_gender_names(
            [request()], "Spanish", "Spain", "test", {}, tmp_path, 1, offline=True
        )
    source = request().model_copy(
        update={"source_basis": "unknown", "source_gender": "unknown"}
    )
    with pytest.raises(ValueError, match="Offline gender audit lacks exact checkpoint"):
        assess_source_names([source], "test", {}, tmp_path, 1, offline=True)


def test_banking_donor_selection_matches_gender_not_sorted_position(
    tmp_path, monkeypatch
):
    from tau2.multilingual.factory import localization_reuse as reuse
    from tau2.multilingual.localization import EntityReuseProvenance
    from tau2.utils.artifacts import ReviewApproval
    from tests.test_multilingual.test_banking_prose import entry, proposal

    def donor(name, gender):
        forms = LocalizedForms(
            native=name, romanized=name, english="Reviewed fixture donor."
        )
        source = request(gender).model_copy(update={"forms": forms})
        assessment = result(source, gender=gender, decision="keep")
        return reuse.ReuseCandidate(
            name=forms,
            address=forms,
            email=forms,
            name_ids=[name],
            address_ids=["address"],
            email_ids=["email"],
            lineage=EntityReuseProvenance(
                source_domain="retail",
                source_entry_ids=[name],
                source_artifact_sha256="a" * 64,
                approval=ReviewApproval(content_sha256="a" * 64, reviewed_by="fixture"),
            ),
            name_gender=evidence_for(source, assessment, "test"),
        )

    monkeypatch.setattr(
        reuse,
        "banking_reuse_candidates",
        lambda *_: [donor("Ana Rubio", "female"), donor("Marcos Pérez", "male")],
    )
    mapping = proposal(
        entry(
            "male",
            "Marcus Chen",
            "Marcus Chen",
            category="identity_name",
            owner="owner-a",
        ),
        entry(
            "female",
            "Claire Dubois",
            "Claire Dubois",
            category="identity_name",
            owner="owner-b",
        ),
    )
    reference = reuse.ReviewedReuseSource(
        map_file=tmp_path / "map.json", approval_file=tmp_path / "approval.json"
    )
    values = reuse.reuse_banking_entities(
        mapping.source, reference, "es", {"owner-a": "male", "owner-b": "female"}
    )
    assert {e.id: e.forms.native for e in values} == {
        "male": "Marcos Pérez",
        "female": "Ana Rubio",
    }
    assert len(reuse.validate_reused_entries(mapping.source, values)) == 2
    assert reuse.reuse_banking_entities(mapping.source, reference, "es") == []
    values[0].name_gender = None
    with pytest.raises(ValueError, match="requires source-bound gender"):
        reuse.validate_reused_entries(mapping.source, values)


def test_name_response_contract_has_no_postal_address_schema():
    from tau2.multilingual.factory.localization_gender import GenderNameForms

    schema = GenderNameForms.model_json_schema()
    assert set(schema["properties"]) == {"native", "romanized", "english"}
    assert set(schema["required"]) == set(schema["properties"])

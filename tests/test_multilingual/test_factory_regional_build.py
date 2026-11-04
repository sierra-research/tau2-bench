# Copyright (c) 2026 Sierra
"""Regional entities join the existing content review, without another approval."""

from types import SimpleNamespace

import pytest

from tau2.data_model.language import NameOrder
from tau2.multilingual.factory import localization_artifact, pipeline, regional_build
from tau2.multilingual.factory import localization_publish as publisher
from tau2.multilingual.factory.database_localization import RegionalDraftRequest
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.regional_build import (
    GenerateLocalization,
    ReuseLocalization,
    build_localized_domain,
)
from tau2.multilingual.factory.voice_assets import (
    VoiceAssetsRequest,
    VoiceDesignConfig,
    VoiceSpec,
    listening_digest,
    prepare_voices,
)
from tau2.multilingual.localization import SourceDocument, json_content_digest
from tau2.utils.artifacts import ReviewApproval, content_digest, write_artifact
from tests.test_multilingual import test_authoring_pipeline as authoring
from tests.test_multilingual import test_regional_localization_compile as compiler

proposal = compiler.proposal
draft_request = authoring.draft_request


@pytest.fixture
def localized_request(draft_request, proposal, monkeypatch, tmp_path):
    proposal.settings.language = draft_request.form.identity.language
    proposal.settings.name_order = draft_request.form.identity.name_order
    for document in proposal.source.documents:
        if document.id == "db":
            document.id = "database"
    for entry in [*proposal.entries, *proposal.source.entries]:
        for occurrence in entry.occurrences:
            if occurrence.document_id == "db":
                occurrence.document_id = "database"
    proposal.source.documents.append(
        SourceDocument(
            id="policy:policy",
            kind="policy",
            source_path="policy.md",
            content="Fixture retail policy.",
            sha256=json_content_digest("Fixture retail policy."),
        )
    )
    compiler._cover_preserved_values(proposal)
    entries = {entry.id: entry for entry in proposal.entries}
    proposal.entries = [
        localization_artifact.automatic_localization(entry, proposal.settings)
        or entries[entry.id]
        for entry in proposal.source.entries
    ]
    monkeypatch.setattr(
        publisher, "inventory_localization", lambda *args: proposal.source
    )
    path = tmp_path / "mapping.json"
    localization_artifact.write_localization_map(proposal, path)
    mapping = localization_artifact.RegionalMapArtifact.model_validate_json(
        path.read_text()
    )
    bundle = draft_request.domains[0]
    for row in bundle.metadata.values():
        row.canonical_task_id = "task-1"
    bundle.localization = ReuseLocalization(source=proposal.source, mapping=mapping)
    monkeypatch.setattr(
        pipeline,
        "generate",
        authoring.fake_generate,
    )
    return draft_request


def test_build_reuse_joins_one_content_review(localized_request, tmp_path):
    staged = pipeline.draft(localized_request, tmp_path / "draft.json")
    bundle = staged.domains["retail"]
    assert bundle.localization is None
    assert bundle.regional_bundle.database["users"]["u1"]["name"]["first_name"] == "静"
    assert (
        bundle.localization_evidence.mapping
        == localized_request.domains[0].localization.mapping
    )
    review = (tmp_path / "draft.review/index.html").read_text()
    assert content_digest(staged) in review
    assert "Entity mappings" in review
    assert "Name details" in review
    assert bundle.regional_bundle.localization_map_sha256 == content_digest(
        bundle.localization_evidence.mapping
    )
    pipeline.validate_draft(staged)
    approval = ReviewApproval(
        content_sha256=content_digest(staged), reviewed_by="content reviewer"
    )
    approval.require(staged)
    changed = staged.model_copy(deep=True)
    changed.domains["retail"].localization_evidence.mapping.entries[
        0
    ].notes += " changed"
    with pytest.raises(ValueError, match="review does not match"):
        approval.require(changed)
    with pytest.raises(ValueError, match="differs from the factory"):
        pipeline.validate_draft(changed)


def test_generate_localization_uses_existing_drafters_without_receipts(
    localized_request, tmp_path, monkeypatch
):
    from tau2.multilingual.factory import database_localization

    reused = localized_request.domains[0].localization
    calls = []

    def fake_generate(request, output, checkpoints):
        calls.append(request)
        write_artifact(reused.mapping, output)

    monkeypatch.setattr(
        database_localization, "draft_database_localization", fake_generate
    )
    localized_request.domains[0].localization = GenerateLocalization(
        request=RegionalDraftRequest(
            source=reused.source,
            settings=reused.mapping.settings,
            revision=reused.mapping.revision,
            generation=FactoryModelConfig(
                model="fixture-entity-generator", reasoning_effort="medium"
            ),
        )
    )
    staged = pipeline.draft(localized_request, tmp_path / "generated.json")
    assert len(calls) == 1
    assert (
        staged.domains["retail"].regional_bundle.database["users"]["u1"]["name"][
            "first_name"
        ]
        == "静"
    )
    assert staged.domains["retail"].localization_evidence.task_reviews == []


@pytest.mark.parametrize("defect", ["language", "name_order", "domain", "mapping"])
def test_invalid_localization_fails_before_pack_generation(
    localized_request, tmp_path, monkeypatch, defect
):
    localized = localized_request.domains[0].localization
    if defect == "language":
        localized.mapping.settings.language = "xx"
    elif defect == "name_order":
        localized.mapping.settings.name_order = NameOrder.FAMILY_FIRST
    elif defect == "domain":
        localized_request.domains[0].domain = "airline"
    else:
        localized.mapping.entries[0].forms.native = []
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda **kwargs: pytest.fail(
            "invalid inputs reached a paid generation boundary"
        ),
    )
    with pytest.raises(ValueError):
        pipeline.draft(localized_request, tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_modified_bundle_cannot_reach_voice_provider_even_with_new_receipt(
    localized_request, tmp_path
):
    staged = pipeline.draft(localized_request, tmp_path / "draft.json")
    staged.domains["retail"].regional_bundle.database["users"]["u1"]["email"] = (
        "wrong@example.com"
    )
    request = VoiceAssetsRequest(
        draft=staged,
        specs={
            name: VoiceSpec(
                description="Fixture voice", audition_text="Fixture audition. " * 10
            )
            for name in staged.pack.personas
        },
        config=VoiceDesignConfig(
            model="fixture", seed=42, loudness=0.5, guidance_scale=38
        ),
        review=ReviewApproval(
            content_sha256=content_digest(staged), reviewed_by="fixture"
        ),
    )
    provider = authoring.FakeVoices()
    with pytest.raises(ValueError, match="differs from the factory"):
        prepare_voices(request, tmp_path / "assets", provider)
    assert provider.designs == provider.creates == 0
    assert not (tmp_path / "assets").exists()


def test_semantic_review_is_opt_in_per_task_and_digest_bound(
    localized_request, tmp_path, monkeypatch
):
    from tau2.utils import llm_utils

    reused = localized_request.domains[0].localization
    reused.semantic_review = FactoryModelConfig(
        model="fixture-reviewer", reasoning_effort="medium"
    )
    monkeypatch.setattr(
        regional_build,
        "_task_packet",
        lambda bundle, source, task, mapping: {
            "task": task.model_dump(mode="json"),
            "db": bundle.database,
        },
    )
    calls = []

    def review(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content='{"findings": ["Inspect this geography."]}')

    monkeypatch.setattr(llm_utils, "generate", review)
    metadata = localized_request.domains[0].metadata
    first, evidence = build_localized_domain(
        reused, metadata, tmp_path / "review", tmp_path
    )
    again, repeated = build_localized_domain(
        reused, metadata, tmp_path / "review", tmp_path
    )
    assert first == again and evidence == repeated
    assert len(calls) == len(first.tasks) == 1
    assert calls[0]["model"] == "fixture-reviewer"
    assert calls[0]["reasoning_effort"] == "medium"
    assert calls[0]["call_name"] == "factory_localized_task_review"
    assert evidence.task_reviews[0].findings == ["Inspect this geography."]
    regional_build.validate_localized_domain(first, evidence, metadata, tmp_path)
    evidence.task_reviews[0].input_sha256 = "0" * 64
    with pytest.raises(ValueError, match="no longer matches"):
        regional_build.validate_localized_domain(first, evidence, metadata, tmp_path)


def test_linked_records_preserve_table_identity_and_list_contacts():
    task = SimpleNamespace(
        model_dump=lambda **kwargs: {"user_id": "same", "phone_number": "555-1234"}
    )
    records = regional_build._linked_records(
        {
            "users": {"same": {"name": "Caller"}},
            "orders": {"same": {"status": "delivered"}},
            "lines": [
                {
                    "line_id": "line-1",
                    "phone_number": "555-1234",
                    "customer_id": "customer-1",
                }
            ],
            "customers": [{"customer_id": "customer-1", "first_name": "A"}],
        },
        task,
    )
    assert records["users"]["same"]["name"] == "Caller"
    assert records["orders"]["same"]["status"] == "delivered"
    assert records["customers"]["0"]["customer_id"] == "customer-1"


@pytest.mark.parametrize("language", ["zh", "ko"])
def test_semantic_packet_resolves_transfer_only_family_first_callers(language):
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.multilingual.runtime import load_language
    from tau2.utils import DATA_DIR

    scenario = load_language(language, "retail", mode="voice").scenario
    mapping = localization_artifact.RegionalMapArtifact.model_validate_json(
        (
            DATA_DIR / "tau2/multilingual" / language / "localization/retail.json"
        ).read_text()
    )
    source = inventory_localization("retail", DATA_DIR)
    task = next(
        t for t in scenario.tasks if scenario.metadata[t.id].canonical_task_id == "50"
    )
    packet = regional_build._task_packet(scenario, source, task, mapping)
    canonical = packet["effective_databases"]["source"]["linked_records"]["users"]
    localized = packet["effective_databases"]["localized"]["linked_records"]["users"]
    assert len(canonical) == len(localized) == 1
    caller = next(iter(localized.values()))
    assert caller["name"]["first_name"] in task.user_scenario.instructions.known_info
    assert caller["name"]["last_name"] in task.user_scenario.instructions.known_info


def test_semantic_packet_includes_effective_telecom_device_state():
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.multilingual.runtime import load_language
    from tau2.utils import DATA_DIR

    scenario = load_language("es", "telecom", mode="voice").scenario
    mapping = localization_artifact.RegionalMapArtifact.model_validate_json(
        (DATA_DIR / "tau2/multilingual/es/localization/telecom.json").read_text()
    )
    source = inventory_localization("telecom", DATA_DIR)
    packet = regional_build._task_packet(scenario, source, scenario.tasks[0], mapping)
    for label in ("source", "localized"):
        assert packet["effective_databases"][label]["user_database"]
        assert packet["effective_databases"][label]["linked_records"]["customers"]


def test_localization_to_review_voice_audition_and_clean_runtime_publication(
    localized_request, tmp_path, monkeypatch
):
    import tau2.utils
    from tau2.multilingual.runtime import load_language

    staged = pipeline.draft(localized_request, tmp_path / "draft.json")
    content_review = ReviewApproval(
        content_sha256=content_digest(staged), reviewed_by="human content reviewer"
    )
    provider = authoring.FakeVoices()
    assets_root = tmp_path / "authoring-assets"
    assets = prepare_voices(
        VoiceAssetsRequest(
            draft=staged,
            specs={
                name: VoiceSpec(
                    description="Fixture voice", audition_text="Fixture audition. " * 10
                )
                for name in staged.pack.personas
            },
            config=VoiceDesignConfig(
                model="fixture", seed=42, loudness=0.5, guidance_scale=38
            ),
            review=content_review,
        ),
        assets_root,
        provider,
    )
    audition_review = ReviewApproval(
        content_sha256=listening_digest(assets), reviewed_by="human audition reviewer"
    )
    output = tmp_path / "release/tau2/multilingual" / staged.pack.language
    assert pipeline.finalize(
        staged, assets, audition_review, output, assets_root=assets_root
    )
    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path / "release")
    loaded = load_language(staged.pack.language, "retail", mode="voice")
    assert loaded.scenario == staged.domains["retail"].regional_bundle
    assert loaded.tasks == staged.domains["retail"].regional_bundle.tasks
    assert provider.designs == provider.creates == 2
    assert set(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    ) == {
        "pack.yaml",
        "tasks/retail.metadata.json",
        "regional/retail.json",
        "provenance.json",
    }
    for path in output.rglob("*"):
        if path.is_file():
            text = path.read_text()
            for authoring_field in (
                '"reviewed_by"',
                '"localization_evidence"',
                '"task_reviews"',
                '"mapping"',
                '"draft"',
            ):
                assert authoring_field not in text
    assert (tmp_path / "draft.review/index.html").exists()
    assert "localization_evidence" in (tmp_path / "draft.json").read_text()
    assert content_review.reviewed_by in (assets_root / "request.json").read_text()


def test_stage_existing_language_with_localization_is_fully_offline(
    localized_request, tmp_path, monkeypatch
):
    from tau2.data_model.language import LanguagePack
    from tau2.multilingual.factory.schema import StagePackRequest

    staged = pipeline.draft(localized_request, tmp_path / "draft.json")
    published = staged.pack.model_dump()
    for persona in published["personas"].values():
        persona["voice"]["voice_id"] = "existing-voice"
    pack = LanguagePack.model_validate(published)
    monkeypatch.setattr(
        pipeline,
        "generate",
        lambda **kwargs: pytest.fail("offline stage invoked generation"),
    )
    offline = pipeline.stage_pack(
        StagePackRequest(
            pack=pack, domains=localized_request.domains, revision="offline-v2"
        ),
        tmp_path / "offline.json",
    )
    assert (
        offline.domains["retail"].regional_bundle
        == staged.domains["retail"].regional_bundle
    )
    assert offline.generation is None
    assert (tmp_path / "offline.review/index.html").exists()


def test_deterministic_flags_distinguish_prose_changes_and_preserved_values(
    localized_request,
):
    from tau2.multilingual.factory.localization_artifact import AuthoredLocalization

    mapping = localized_request.domains[0].localization.mapping.model_copy(deep=True)
    mapping.entries.extend(
        [
            AuthoredLocalization(
                id="prose",
                category="prose",
                source="Ada lives in town.",
                forms={
                    "native": "静 lives near town.",
                    "romanized": "Jing lives near town.",
                    "english": "Fixture",
                },
                operation="adapt",
                notes="Review the changed preposition.",
            ),
            AuthoredLocalization(
                id="unchanged",
                category="address",
                source="Ada",
                forms={
                    "native": "Ada",
                    "romanized": "Ada",
                    "english": "Fixture unchanged label",
                },
                operation="preserve",
                notes="Intentional unchanged label.",
            ),
        ]
    )
    from tau2.multilingual.localization import (
        UserIdAlias,
        UserIdRevisionProvenance,
    )

    mapping.entries.extend(
        [
            AuthoredLocalization(
                id="state-code",
                category="address",
                source="QX",
                forms={"native": "QV", "romanized": "QV", "english": "Fixture"},
                operation="adapt",
                notes="Two-letter code.",
            ),
            AuthoredLocalization(
                id="code-inside-word",
                category="address",
                source="QXQ Street",
                forms={
                    "native": "QXQ Calle",
                    "romanized": "QXQ Calle",
                    "english": "Fixture",
                },
                operation="adapt",
                notes="Contains the code letters inside a longer token.",
            ),
            AuthoredLocalization(
                id="user-id-prose",
                category="prose",
                source="Account ada_lovelace_1234 is active.",
                forms={
                    "native": "Account jing_wang_1234 is active.",
                    "romanized": "Account jing_wang_1234 is active.",
                    "english": "Fixture",
                },
                operation="adapt",
                notes="Only the name-derived customer ID changes.",
            ),
        ]
    )
    mapping.user_id_revision = UserIdRevisionProvenance(
        parent_map_sha256="0" * 64,
        aliases=[
            UserIdAlias(canonical="ada_lovelace_1234", localized="jing_wang_1234")
        ],
        changed_entry_count=1,
    )
    findings = regional_build.localization_findings(mapping)
    prose = next(f for f in findings if f.entry_id == "prose")
    assert prose.kind == "prose_change"
    assert (prose.before, prose.after) == ("Ada lives in town.", "静 lives near town.")
    assert not any(f.entry_id == "user-id-prose" for f in findings)
    assert not any(f.entry_id == "state-code" for f in findings)
    assert any(
        f.kind == "source_value_remaining" and f.entry_id == "given-name"
        for f in findings
    )
    assert any(
        f.kind == "preserved_value" and f.entry_id == "unchanged" for f in findings
    )


def test_raw_canonical_tasks_get_name_facts_inside_factory(localized_request, tmp_path):
    task = localized_request.domains[0].tasks[0]
    task.user_scenario.instructions.known_info = (
        task.user_scenario.instructions.known_info.split("\n\nName details:")[0]
    )
    original = task.model_copy(deep=True)
    assert "Name details:" not in original.user_scenario.instructions.known_info
    staged = pipeline.draft(localized_request, tmp_path / "raw-source.json")
    built = staged.domains["retail"].tasks[0]
    assert "Name details:" in built.user_scenario.instructions.known_info
    assert built.user_scenario.instructions.known_info.startswith(
        original.user_scenario.instructions.known_info
    )
    assert built.model_dump(exclude={"user_scenario"}) == original.model_dump(
        exclude={"user_scenario"}
    )
    assert built.user_scenario.instructions.model_dump(
        exclude={"known_info"}
    ) == original.user_scenario.instructions.model_dump(exclude={"known_info"})
    assert task == original


@pytest.mark.parametrize("kind", ["task-only", "imported-regional", "factory-regional"])
def test_content_review_shows_base_and_imported_domains_with_escaped_instructions(
    localized_request, tmp_path, kind
):
    import html

    from tests.test_multilingual.test_localization_review import ReviewHTML

    staged = pipeline.draft(localized_request, tmp_path / "draft.json")
    bundle = staged.domains["retail"]
    bundle.tasks[
        0
    ].user_scenario.instructions.known_info += (
        " <script>Base task instructions</script>"
    )
    if kind == "task-only":
        bundle.regional_bundle = None
        bundle.localization_evidence = None
        staged.pack.domains["retail"] = bundle.binding()
    else:
        bundle.regional_bundle.tasks[
            0
        ].user_scenario.instructions.known_info += (
            ' <img src="x" onerror="regional-task-instructions">'
        )
        if kind == "imported-regional":
            bundle.localization_evidence = None
    path = regional_build.write_content_review(staged, tmp_path / "review")
    page = path.read_text()
    decoded = html.unescape(page)
    assert "<h2>retail</h2>" in page
    assert '"base_tasks"' in decoded
    assert '"canonical_task_id": "task-1"' in decoded
    assert html.escape("<script>Base task instructions</script>") in page
    assert "Name details:" in page
    if kind == "task-only":
        assert '"runtime_bundle"' not in decoded
    else:
        assert '"runtime_bundle"' in decoded
        assert "regional-task-instructions" in page
        assert '"first_name": "静"' in decoded
    if kind == "factory-regional":
        assert "Entity mappings" in page
        assert "<h3>Deterministic flags</h3>" in page
        assert '"semantic_reviews"' in decoded
    else:
        assert "Entity mappings</a>" not in page
        assert "Deterministic flags" not in page
        assert '"semantic_reviews"' not in decoded
    parsed = ReviewHTML()
    parsed.feed(page)
    assert parsed.scripts == 0
    assert parsed.event_attributes == []

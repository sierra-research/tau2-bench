"""Banking regionalization preserves English evidence and transactional semantics."""

import json
from pathlib import Path

import pytest

from tau2.multilingual.factory.localization_inventory import inventory_localization
from tau2.multilingual.scenario_dispatch import (
    canonical_scenario,
    scenario_environment_kwargs,
)

DATA = Path(__file__).resolve().parents[2] / "data"


def test_banking_inventory_covers_all_tasks_and_immutable_english_dependencies():
    source = inventory_localization("banking_knowledge", DATA)
    docs = {d.id: d for d in source.documents}
    assert len(docs["tasks"].content) == 97
    assert len(docs["dependency:english_knowledge"].content) == 698
    assert all(
        e.category == "protocol"
        for e in source.entries
        if any(o.document_id.startswith("dependency:") for o in e.occurrences)
    )
    assert any(
        e.source == "unknown_caller" and e.category == "protocol"
        for e in source.entries
    )
    assert any(
        e.source == "Sarah Bosch" and e.category == "identity_name"
        for e in source.entries
    )
    assert any(
        e.source == "Gold Rewards Card" and e.category == "protocol"
        for e in source.entries
    )
    assert inventory_localization("banking_knowledge", DATA) == source


def test_banking_canonical_scenario_keeps_retrieval_policy_dynamic_and_db_isolated():
    from tau2.data_model.tasks import Task

    task = Task.model_validate_json(
        (DATA / "tau2/domains/banking_knowledge/tasks/task_001.json").read_text()
    )
    bundle = canonical_scenario("banking_knowledge", [task], DATA)
    left, right = (
        scenario_environment_kwargs(bundle),
        scenario_environment_kwargs(bundle),
    )
    assert set(left) == {"db"}
    assert left["db"] == right["db"]
    assert left["db"] is not right["db"]
    left["db"].users.data.clear()
    assert right["db"].users.data
    bundle.knowledge_base_sha256 = "0" * 64
    with pytest.raises(ValueError, match="English knowledge"):
        scenario_environment_kwargs(bundle)


def test_banking_prompt_line_is_rendered_and_versioned():
    from tau2.multilingual.agent_guidance import domain_agent_guidance

    assert domain_agent_guidance("banking_knowledge", "Spanish").text == (
        "The knowledge base is in English. Use it to answer the customer's questions, "
        "but always respond to the customer in Spanish."
    )
    assert domain_agent_guidance("banking_knowledge", "Hindi").version
    assert domain_agent_guidance("retail", "Spanish") is None


@pytest.mark.parametrize(
    "language,retailer,eco_collection",
    [
        ("hi", "विशाल मेगा मार्ट", "सहज ग्रीन रिटेल"),
        ("ko", "다모아생활마트", "푸른바구니 에코상점"),
        ("zh", "天虹百货", "绿循生活集"),
    ],
)
def test_banking_kb_entities_keep_english_rules_and_generic_target(
    language, retailer, eco_collection
):
    from tau2.domains.banking_knowledge.kb_entities import (
        localize_knowledge_entities,
    )
    from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact

    artifact = RegionalMapArtifact.model_validate_json(
        (
            DATA / f"tau2/multilingual/{language}/localization/banking_knowledge.json"
        ).read_bytes()
    )
    overrides = localize_knowledge_entities(artifact, DATA)
    eco = overrides["doc_credit_cards_ecocard_004"].content
    assert retailer in eco
    assert f"- {eco_collection} (eco collection sold by {retailer})" in eco
    assert "A $100 purchase at Target" not in eco
    assert "all other networks do not qualify" in eco
    assert "doc_business_savings_accounts_automatic_sweep_program_003" not in overrides


@pytest.mark.parametrize(
    "language,slack,zoom,microsoft",
    [
        ("hi", "स्लैक टेक्नोलॉजीज़", "ज़ूम वीडियो", "माइक्रोसॉफ्ट 365"),
        ("ko", "슬랙 테크놀로지스", "줌 비디오", "마이크로소프트 365"),
        ("zh", "飞书", "腾讯会议", "微软 365"),
    ],
)
def test_banking_kb_rule_aliases_identify_localized_transaction_merchants(
    language, slack, zoom, microsoft
):
    from tau2.multilingual.scenario_dispatch import bundle_model

    bundle = bundle_model("banking_knowledge").model_validate_json(
        (
            DATA
            / f"tau2/multilingual_extensions/banking/{language}/regional/banking_knowledge.json"
        ).read_bytes()
    )
    overrides = bundle.knowledge_document_overrides
    bronze = overrides["doc_business_credit_cards_business_bronze_rewards_card_003"]
    silver = overrides["doc_business_credit_cards_business_silver_rewards_card_005"]
    assert f"- {slack} (Slack)" in bronze.content
    assert f"- {zoom} (Zoom)" in bronze.content
    assert f"including {microsoft}" in silver.content
    eco = overrides["doc_credit_cards_ecocard_004"].content
    transaction = bundle.database["credit_card_transaction_history"]["data"][
        "txn_ba8b473f295d"
    ]
    assert transaction["merchant_name"] in eco
    task = next(task for task in bundle.tasks if task.id == f"task_022_{language}")
    assert "txn_ba8b473f295d" in json.dumps(task.model_dump(mode="json"))


@pytest.mark.parametrize("language", ["hi", "ko", "zh"])
def test_published_banking_kb_overlay_matches_map_and_retrieval(language):
    pytest.importorskip("rank_bm25", reason="Requires the knowledge extra")
    from tau2.domains.banking_knowledge.environment import (
        get_environment,
        get_knowledge_base,
    )
    from tau2.domains.banking_knowledge.kb_entities import (
        localize_knowledge_entities,
    )
    from tau2.domains.banking_knowledge.retrieval import get_or_create_docs
    from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
    from tau2.multilingual.scenario_dispatch import bundle_model

    artifact = RegionalMapArtifact.model_validate_json(
        (
            DATA / f"tau2/multilingual/{language}/localization/banking_knowledge.json"
        ).read_bytes()
    )
    bundle = bundle_model("banking_knowledge").model_validate_json(
        (
            DATA
            / f"tau2/multilingual_extensions/banking/{language}/regional/banking_knowledge.json"
        ).read_bytes()
    )
    assert bundle.knowledge_document_overrides == localize_knowledge_entities(
        artifact, DATA
    )
    kwargs = scenario_environment_kwargs(bundle)
    localized = kwargs["knowledge_base"]
    canonical = get_knowledge_base()
    assert len(localized.documents) == len(canonical.documents) == 698
    eco_id = "doc_credit_cards_ecocard_004"
    assert localized.documents[eco_id] != canonical.documents[eco_id]
    assert get_or_create_docs(localized) != get_or_create_docs(canonical)
    for variant in ("bm25", "grep_only", "full_kb", "golden_retrieval"):
        environment = get_environment(
            **kwargs,
            retrieval_variant=variant,
            task=next(
                task for task in bundle.tasks if task.id == f"task_021_{language}"
            ),
        )
        assert environment.domain_name == "banking_knowledge"
        if variant in {"full_kb", "golden_retrieval"}:
            assert localized.documents[eco_id].content in environment.policy


@pytest.fixture
def banking_proposal():
    from tau2.multilingual.factory.localization_artifact import automatic_localization
    from tau2.multilingual.localization import (
        GenerationProvenance,
        LocalizationEntry,
        LocalizationMap,
        LocalizedForms,
        RegionalSettings,
    )

    source = inventory_localization("banking_knowledge", DATA)
    settings = RegionalSettings(
        language="es",
        country_code="ES",
        currency_code="USD",
        currency_scale="1",
        time_zone="America/New_York",
        name_order="given_first",
    )
    return LocalizationMap(
        revision="test-control",
        source=source,
        settings=settings,
        entries=[
            automatic_localization(e, settings)
            or LocalizationEntry(
                **e.model_dump(),
                forms=LocalizedForms(
                    native=e.source,
                    romanized=e.source,
                    english="Unchanged control fixture.",
                ),
                operation="preserve",
                notes="Test fixture only.",
            )
            for e in source.entries
        ],
        provenance=GenerationProvenance(
            model="test",
            model_args={},
            prompt_version="test",
            call_name="test",
            input_sha256="a" * 64,
        ),
    )


def test_banking_compiler_reproduces_control_and_rejects_currency_drift(
    banking_proposal,
):
    from tau2.multilingual.factory.localization_compile import compile_localization

    result = compile_localization(banking_proposal)
    assert [d.content for d in result] == [
        d.content for d in banking_proposal.source.documents
    ]
    banking_proposal.settings.currency_code = "EUR"
    with pytest.raises(ValueError, match="unchanged USD"):
        compile_localization(banking_proposal)


def test_banking_reuse_preserves_lineage_without_approving_new_context(
    banking_proposal,
    tmp_path,
):
    from tau2.multilingual.factory.localization_address_components import (
        repair_street_line,
    )
    from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
    from tau2.multilingual.factory.localization_reuse import (
        ReviewedReuseSource,
        reuse_banking_entities,
    )
    from tau2.utils.artifacts import ReviewApproval, content_digest, write_artifact

    root = DATA / "tau2/multilingual/es/localization"
    artifact = RegionalMapArtifact.model_validate_json(
        (root / "retail.json").read_text()
    )
    # Synthetic approval belongs only to this corrected test fixture. Checked-in
    # donor approvals must become stale when their real map changes.
    for candidate in artifact.entries:
        candidate.name_gender = None
        if candidate.category == "address":
            for representation in ("native", "romanized"):
                value = getattr(candidate.forms, representation)
                if isinstance(value, str):
                    setattr(
                        candidate.forms,
                        representation,
                        repair_street_line(value, "es", representation),
                    )
    write_artifact(artifact, tmp_path / "retail.json")
    digest = content_digest(artifact)
    write_artifact(
        ReviewApproval(content_sha256=digest, reviewed_by="Synthetic test fixture"),
        tmp_path / "approval.json",
    )
    rows = reuse_banking_entities(
        banking_proposal.source,
        ReviewedReuseSource(
            map_file=tmp_path / "retail.json", approval_file=tmp_path / "approval.json"
        ),
        "es",
    )
    assert rows and any(e.category == "address" for e in rows)
    assert not any(
        e.category == "identity_name" for e in rows
    )  # Donors lack gender evidence.
    assert all(e.reuse.source_artifact_sha256 == digest for e in rows)
    assert all(e.reuse.approval.content_sha256 == digest for e in rows)
    assert banking_proposal.status == "draft"
    from tau2.multilingual.factory.localization_emails import localize_email_entries

    emails = localize_email_entries(
        [e for e in banking_proposal.source.entries if e.category == "contact"],
        rows + [e for e in banking_proposal.entries if e.category == "identity_name"],
    )
    assert len(emails) == 64
    assert all(e.forms.native.isascii() for e in emails)


def test_banking_rejects_changed_financial_value_in_json_encoded_tool_call(
    banking_proposal,
):
    from tau2.multilingual.factory.localization_compile import compile_localization

    entry = next(
        e
        for e in banking_proposal.entries
        if e.category == "prose"
        and any(o.path[-1] == "arguments" for o in e.occurrences)
        and "user_id" in e.source
    )
    old = json.loads(entry.source)
    assert isinstance(old, dict)
    key = next(k for k in old if k.endswith("_id"))
    old[key] = "invalid-id"
    entry.forms.native = json.dumps(old)
    entry.operation = "adapt"
    with pytest.raises(ValueError, match="protected rule, amount or identifier"):
        compile_localization(banking_proposal)


def test_banking_published_control_uses_common_approval_and_runtime_paths(
    banking_proposal, tmp_path
):
    from tau2.data_model.language import DomainBinding, TaskMetadata, TaskSelection
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_publish import build_regional_bundle
    from tau2.multilingual.scenario_dispatch import bundle_model
    from tau2.utils.artifacts import write_artifact

    map_path = tmp_path / "map.json"
    write_localization_map(banking_proposal, map_path)
    artifact = RegionalMapArtifact.model_validate_json(map_path.read_text())
    tasks = next(
        d.content for d in banking_proposal.source.documents if d.id == "tasks"
    )
    metadata = {t["id"]: TaskMetadata(canonical_task_id=t["id"]) for t in tasks}
    output = tmp_path / "bundle.json"
    bundle = build_regional_bundle(artifact, banking_proposal.source, metadata, DATA)
    write_artifact(bundle, output)
    restored = bundle_model("banking_knowledge").model_validate_json(output.read_text())
    assert restored == bundle
    from tau2.multilingual.runtime import LoadedLanguage, load_pack

    loaded = LoadedLanguage(
        pack=load_pack("es"),
        domain="banking_knowledge",
        binding=DomainBinding(
            revision="fixture",
            regional_bundle_file="regional.json",
            task_metadata_file="metadata.json",
        ),
        mode="text",
        selection=TaskSelection(),
        tasks=bundle.tasks,
        metadata=bundle.metadata,
        pack_sha256="a" * 64,
        tasks_sha256="b" * 64,
        metadata_sha256="c" * 64,
        scenario=bundle,
        scenario_mode="regional",
    )
    assert loaded.scenario == bundle
    assert len(bundle.tasks) == 97
    assert set(scenario_environment_kwargs(restored)) == {"db"}
    from tau2.multilingual.factory.regional_build import _task_packet

    packet = _task_packet(bundle, banking_proposal.source, bundle.tasks[0], artifact)
    assert packet["source_policies"]
    assert set(packet["effective_databases"]) == {"source", "localized"}


def test_banking_domain_line_enters_both_text_and_voice_agent_prompts():
    from tau2.data_model.language import TaskMetadata, TaskSelection
    from tau2.data_model.tasks import Task
    from tau2.multilingual.runtime import (
        LoadedLanguage,
        agent_language_guidance,
        load_pack,
        resolve_task,
    )

    pack = load_pack("es").model_copy(deep=True)
    binding = pack.domains["retail"].model_copy()
    pack.domains["banking_knowledge"] = binding
    task = Task.model_validate_json(
        (DATA / "tau2/domains/banking_knowledge/tasks/task_001.json").read_text()
    )
    for mode in ("text", "voice"):
        loaded = LoadedLanguage(
            pack=pack,
            domain="banking_knowledge",
            binding=binding,
            mode=mode,
            selection=TaskSelection(),
            tasks=[task],
            metadata={task.id: TaskMetadata(canonical_task_id=task.id)},
            pack_sha256="a" * 64,
            tasks_sha256="b" * 64,
            metadata_sha256="c" * 64,
        )
        context = resolve_task(loaded, task, 0)
        assert context.domain_agent_guidance.text in agent_language_guidance(context)
    assert context.domain_agent_guidance.version == "banking-english-knowledge-v1"
    assert "{target_language}" not in context.domain_agent_guidance.text


def test_pilot_rejects_unicode_email_before_accepting_generated_candidate(
    banking_proposal,
):
    from tau2.multilingual.factory.database_localization import (
        BatchInput,
        GeneratedBatch,
        _validate_batch,
    )

    entry = next(
        e
        for e in banking_proposal.source.entries
        if e.category == "contact" and "@" in e.source
    )
    batch = BatchInput(request_sha256="a" * 64, entries=[entry], completed=[])
    candidate = GeneratedBatch.model_validate(
        {
            "entries": [
                {
                    "id": entry.id,
                    "forms": {
                        "native": "नाम@example.com",
                        "romanized": "naam@example.com",
                        "english": "Test invalid Unicode email.",
                    },
                    "operation": "adapt",
                    "notes": "Test only.",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="ASCII"):
        _validate_batch(batch, candidate)


def test_new_domain_fingerprints_pin_dispatch_without_changing_existing_domains():
    from tau2.multilingual.regional import runtime_rules_digest as original
    from tau2.multilingual.scenario_dispatch import runtime_rules_digest

    for domain in ("airline", "retail", "telecom"):
        assert runtime_rules_digest(domain) == original(domain)
    assert runtime_rules_digest("banking_knowledge") != original("banking_knowledge")


def test_task_injected_customer_emails_use_materialized_identity_context():
    """Fresh drafting can derive emails for users added only by task initialization."""
    source = inventory_localization("banking_knowledge", DATA)
    emails = [
        e
        for e in source.entries
        if e.category == "contact" and isinstance(e.source, str) and "@" in e.source
    ]
    for entry in emails:
        identity, _ = json.JSONDecoder().raw_decode(
            entry.context.split("source_identity=", 1)[1]
        )
        assert identity["full_name"], entry.source
    assert any(e.source == "meiling.chen@outlook.com" for e in emails)


@pytest.mark.parametrize("pattern", [r"\$\d[\d,.]*", r"\b\d+(?:\.\d+)?%"])
def test_banking_compiler_rejects_changed_financial_literals_in_english_prose(
    banking_proposal, pattern
):
    import re

    from tau2.multilingual.factory.localization_compile import compile_localization

    entry = next(
        e
        for e in banking_proposal.entries
        if e.category == "prose"
        and isinstance(e.source, str)
        and re.search(pattern, e.source)
    )
    entry.forms.native = re.sub(pattern, "", entry.source, count=1)
    entry.operation = "adapt"
    with pytest.raises(ValueError, match="financial literals"):
        compile_localization(banking_proposal)


def test_reviewed_binding_rejects_changed_forms_and_absent_source(banking_proposal):
    from tau2.domains.banking_knowledge.localization_bindings import (
        validate_prose_bindings,
    )
    from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding
    from tau2.utils.artifacts import ReviewApproval, content_digest

    entry = next(
        e
        for e in banking_proposal.entries
        if e.category == "prose" and isinstance(e.source, str)
    )
    forms = LocalizedForms(native="Reviewed", romanized="Reviewed", english="Fixture")
    revised = banking_proposal.model_copy(deep=True)
    revised.prose_bindings = [
        ProseEntityBinding(
            entry_id=entry.id,
            source=entry.source,
            forms=forms,
            kind="reviewed",
            reason="Approved fixture",
            review=ReviewApproval(
                content_sha256=content_digest(forms), reviewed_by="test reviewer"
            ),
        )
    ]
    validate_prose_bindings(revised)
    unreviewed = revised.model_copy(deep=True)
    unreviewed.prose_bindings[0].forms.native = "Unreviewed edit"
    with pytest.raises(ValueError, match="review does not match"):
        validate_prose_bindings(unreviewed)
    revised.prose_bindings[0].source = "Source span that never existed"
    with pytest.raises(ValueError, match="exact source entry"):
        validate_prose_bindings(revised)


def test_banking_build_preserves_gender_evidence_gate(monkeypatch, tmp_path):
    from tau2.multilingual.factory import localization_publish
    from tests.test_multilingual.test_banking_prose import entry, proposal

    mapping = proposal(
        entry("name", "Marcus Chen", "Marcos Rubio", category="identity_name")
    )
    monkeypatch.setattr(
        localization_publish, "restore_localization_map", lambda *_: mapping
    )
    with pytest.raises(ValueError, match="Personal-name gender assessment missing"):
        localization_publish.build_regional_bundle(None, mapping.source, {}, tmp_path)

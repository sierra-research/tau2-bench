"""Selected intake localization preserves the frozen frame and written identity."""

import unicodedata
from pathlib import Path

import pytest

from tau2.domains.intake.environment import load_frozen_tasks
from tau2.domains.intake.folds import FoldKind, fold_value
from tau2.domains.intake.localization import inventory_intake, validate_intake_tasks
from tau2.domains.intake.scoring import project_scoring_folds
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.localization_artifact import automatic_localization
from tau2.multilingual.factory.localization_compile import compile_localization
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationEntry,
    LocalizationMap,
    LocalizedForms,
    RegionalSettings,
)
from tau2.utils.utils import DATA_DIR


def identity_map():
    source = inventory_intake(Path(DATA_DIR))
    settings = RegionalSettings(
        language="en",
        country_code="US",
        currency_code="USD",
        currency_scale="1",
        time_zone="America/New_York",
        name_order="given_first",
    )
    entries = [
        automatic_localization(entry, settings)
        or LocalizationEntry(
            **entry.model_dump(),
            forms=LocalizedForms(
                native=entry.source,
                romanized=entry.source,
                english="Unchanged English source control.",
            ),
            operation="preserve",
            notes="Test control.",
        )
        for entry in source.entries
    ]
    return LocalizationMap(
        revision="intake-test-v1",
        settings=settings,
        source=source,
        entries=entries,
        provenance=GenerationProvenance(
            model="test",
            model_args={},
            prompt_version="test-v1",
            call_name="test",
            input_sha256="0" * 64,
        ),
    )


def hindi_candidate_map():
    from tau2.multilingual.factory.localization_artifact import read_localization_map

    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json"
    )


def test_selected_inventory_reproduces_all_frozen_tasks_and_gold():
    proposal = identity_map()
    documents = {document.id: document for document in compile_localization(proposal)}
    tasks = load_frozen_tasks(None)
    assert documents["tasks"].content == [
        task.model_dump(mode="json") for task in tasks
    ]
    validate_intake_tasks(tasks)
    assert len(tasks) == 200
    assert not any(
        "banks/" in document.source_path for document in proposal.source.documents
    )
    localizable = [entry for entry in proposal.entries if entry.category != "protocol"]
    assert any(entry.source == "Susan Jenkins" for entry in localizable)
    assert any(entry.source == "Lisa Sandoval" for entry in localizable)


def test_selected_compiler_rejects_caller_gold_disagreement():
    tasks = load_frozen_tasks(None)
    tasks[0].evaluation_criteria.actions[0].arguments["fields"][
        "second_guest_full_name"
    ] = "Wrong Person"
    with pytest.raises(ValueError, match="gold|submission|seeded"):
        validate_intake_tasks(tasks)


@pytest.mark.parametrize(
    "value,wrong",
    [("किरण", "करण"), ("José", "Jose"), ("한글", "한갈"), ("李明", "李朋")],
)
def test_native_name_fold_preserves_identity(value, wrong):
    assert fold_value(FoldKind.NATIVE_NAME, value) == fold_value(
        FoldKind.NATIVE_NAME, unicodedata.normalize("NFD", value)
    )
    assert fold_value(FoldKind.NATIVE_NAME, value) != fold_value(
        FoldKind.NATIVE_NAME, wrong
    )


def test_native_fold_does_not_invent_english_symbol_words():
    assert fold_value(FoldKind.NATIVE_NAME, "A & B") != fold_value(
        FoldKind.NATIVE_NAME, "A and B"
    )
    assert fold_value(FoldKind.NATIVE_NAME, "0.5 mg") != fold_value(
        FoldKind.NATIVE_NAME, "05 mg"
    )


def test_scoring_projection_is_family_specific_and_keeps_frozen_source_intact():
    tasks = load_frozen_tasks()
    expected = {
        "person_names": FoldKind.NAME_DIGITS,
        "addresses": FoldKind.NAME_DIGITS,
        "medications": FoldKind.MEDICATION,
        "coined": FoldKind.NAME,
        "properties": FoldKind.NAME,
        "emails": FoldKind.EMAIL,
    }
    for task in tasks:
        family = task.id.removeprefix("intake_").rsplit("_", 2)[0]
        if family not in expected:
            continue
        original = next(
            action.arguments["record"]
            for action in task.initial_state.initialization_actions
            if action.func_name == "seed_record"
        )
        projected_task = project_scoring_folds(task)
        projected = next(
            action.arguments["record"]
            for action in projected_task.initial_state.initialization_actions
            if action.func_name == "seed_record"
        )
        measured = next(
            field
            for field in projected["fields"]
            if field["name"] in task.evaluation_criteria.actions[0].arguments["fields"]
        )
        assert FoldKind(measured["fold"]) is expected[family]
        assert project_scoring_folds(projected_task) == projected_task
        assert original is not projected


@pytest.mark.parametrize("language", ["hi", "ko", "zh", "pt", "es"])
def test_published_intake_packs_keep_decoys_distinct_under_new_scoring(language):
    from tau2.multilingual.runtime import load_language

    loaded = load_language(language, "intake", mode="voice", pack_extension="intake")
    tasks = {loaded.metadata[task.id].canonical_task_id: task for task in loaded.tasks}
    stimuli = loaded.scenario.intake_localization.stimuli
    assert len(tasks) == len(stimuli) == 200
    scoped_kinds = {
        "ko": {
            "second_guest_full_name": FoldKind.KOREAN_NAME_DIGITS,
            "destination_property": FoldKind.KOREAN_NAME,
            "home_address": FoldKind.KOREAN_ADDRESS,
            "contact_phone": FoldKind.KOREAN_PHONE,
            "current_medication": FoldKind.KOREAN_MEDICATION,
        },
        "pt": {
            "home_address": FoldKind.PORTUGUESE_ADDRESS,
            "current_medication": FoldKind.PORTUGUESE_MEDICATION,
        },
        "zh": {
            "contact_phone": FoldKind.CHINESE_PHONE,
            "current_medication": FoldKind.CHINESE_MEDICATION,
        },
        "hi": {"current_medication": FoldKind.HINDI_MEDICATION},
        "es": {"current_medication": FoldKind.SPANISH_MEDICATION},
    }
    gold_by_field: dict[str, set[str]] = {}
    old_gold_by_field: dict[str, set[str]] = {}
    previous = {
        FoldKind.KOREAN_NAME_DIGITS: FoldKind.NATIVE_NAME_DIGITS,
        FoldKind.KOREAN_NAME: FoldKind.NATIVE_NAME,
        FoldKind.KOREAN_ADDRESS: FoldKind.NATIVE_ADDRESS,
        FoldKind.KOREAN_PHONE: FoldKind.PHONE,
        FoldKind.CHINESE_PHONE: FoldKind.PHONE,
        FoldKind.PORTUGUESE_ADDRESS: FoldKind.NATIVE_ADDRESS,
        FoldKind.HINDI_MEDICATION: FoldKind.NATIVE_MEDICATION,
        FoldKind.KOREAN_MEDICATION: FoldKind.NATIVE_MEDICATION,
        FoldKind.CHINESE_MEDICATION: FoldKind.NATIVE_MEDICATION,
        FoldKind.PORTUGUESE_MEDICATION: FoldKind.NATIVE_MEDICATION,
        FoldKind.SPANISH_MEDICATION: FoldKind.NATIVE_MEDICATION,
    }
    for stimulus in stimuli:
        task = tasks[stimulus.canonical_task_id]
        record = next(
            action.arguments["record"]
            for action in task.initial_state.initialization_actions
            if action.func_name == "seed_record"
        )
        field = next(
            field for field in record["fields"] if field["name"] == stimulus.field_name
        )
        kind = FoldKind(field["fold"])
        if stimulus.field_name in scoped_kinds.get(language, {}):
            assert kind is scoped_kinds[language][stimulus.field_name]
        gold = fold_value(kind, stimulus.written)
        gold_by_field.setdefault(stimulus.field_name, set()).add(gold)
        old_gold_by_field.setdefault(stimulus.field_name, set()).add(
            fold_value(previous.get(kind, kind), stimulus.written)
        )
        assert gold == fold_value(kind, field["value"])
        assert all(fold_value(kind, decoy) != gold for decoy in stimulus.decoys)
    for stimulus in stimuli:
        if stimulus.field_name not in scoped_kinds.get(language, {}):
            continue
        task = tasks[stimulus.canonical_task_id]
        record = next(
            action.arguments["record"]
            for action in task.initial_state.initialization_actions
            if action.func_name == "seed_record"
        )
        kind = FoldKind(
            next(
                field["fold"]
                for field in record["fields"]
                if field["name"] == stimulus.field_name
            )
        )
        for decoy in stimulus.decoys:
            # Some authored decoys deliberately equal a *different task's*
            # gold. The new rules must not introduce any additional collision.
            if fold_value(kind, decoy) in gold_by_field[stimulus.field_name]:
                old_kind = previous[kind]
                assert (
                    fold_value(old_kind, decoy)
                    in old_gold_by_field[stimulus.field_name]
                )


def test_native_compilation_maps_all_references_and_preserves_wrong_value_failure():
    from pydantic import TypeAdapter

    from tau2.data_model.tasks import Task
    from tau2.domains.intake.agent_directed import _agent_directed_task_variant
    from tau2.domains.intake.environment import get_environment

    proposal = hindi_candidate_map()
    for entry in proposal.entries:
        if entry.source == "Lisa Sandoval":
            # This test authors a new fixture name; its previous assessment is
            # intentionally discarded. Publication requires a fresh assessment.
            entry.name_gender = None
            entry.forms.native = "किरण"
            entry.forms.romanized = "Kiran"
            entry.operation = "adapt"
    documents = {d.id: d for d in compile_localization(proposal)}
    task = TypeAdapter(list[Task]).validate_python(documents["tasks"].content)[0]
    assert (
        task.evaluation_criteria.actions[0].arguments["fields"][
            "second_guest_full_name"
        ]
        == "किरण"
    )
    assert "Lisa Sandoval" not in task.model_dump_json()
    directed = _agent_directed_task_variant(task)
    assert (
        "confirmed_with_user" not in directed.evaluation_criteria.actions[0].arguments
    )
    environment = get_environment(task=directed)
    environment.set_state(
        initialization_data=None,
        initialization_actions=directed.initial_state.initialization_actions,
        message_history=[],
    )
    environment.tools.submit_fields(**directed.evaluation_criteria.actions[0].arguments)
    final = environment.tools.db.intake_records[0].fields[-1].value
    assert final == "किरण"
    assert final != fold_value(FoldKind.NATIVE_NAME, "करण")
    assert (
        proposal.source.documents[2].content[0]["evaluation_criteria"]["actions"][0][
            "arguments"
        ]["fields"]["second_guest_full_name"]
        == "Lisa Sandoval"
    )


def test_native_detector_counts_graphemes_without_erasing_marks():
    from tau2.domains.intake.spell_events import NativeSpellEventDetector

    detector = NativeSpellEventDetector(
        field_skeletons={"name": ["किरण"], "code": ["12"]},
        digit_words={"एक": "1", "दो": "2"},
    )
    detected = detector.detect("कि, र, ण")
    assert len(detected) == 1
    assert detected[0].letters_spoken == 3
    assert detector.detect("क, र, ण") == []
    assert detector.detect("एक, दो")[0].field == "code"
    assert detector.detect("किरण, शर्मा") == []


def test_selected_sampler_never_reads_full_banks(monkeypatch):
    from tau2.domains.intake import complications
    from tau2.domains.intake.localization import selected_source_data
    from tau2.domains.intake.localized import sample_localized_complication

    def forbidden():
        raise AssertionError("Native runtime must not load English full banks")

    data = selected_source_data(Path(DATA_DIR) / "tau2/domains/intake")
    task = load_frozen_tasks()[0]
    stimulus = data.for_task(task.id)
    monkeypatch.setattr(complications, "_banks", forbidden)
    observed = [
        sample_localized_complication(
            seed, task, stimulus, rate_override=1, channel="voice"
        )
        for seed in range(100)
    ]
    assert any(draw and draw.kind.value == "self_correction" for draw in observed)
    assert any(draw and draw.kind.value == "wrong_slot" for draw in observed)
    assert any(draw is None for draw in observed)
    for draw in observed:
        if draw and draw.kind.value == "self_correction":
            assert draw.params.decoy_value in stimulus.decoys


def test_inventory_contains_only_used_bank_dependencies():
    from tau2.domains.intake.localization import selected_source_data

    selected = selected_source_data(Path(DATA_DIR) / "tau2/domains/intake")
    assert len(selected.stimuli) == 200
    assert {row.family for row in selected.stimuli}.isdisjoint(
        {"vehicles", "shops", "insurance_plans", "rate_plans", "amounts"}
    )
    assert all(row.written and row.decoys for row in selected.stimuli)


def test_compact_map_publication_keeps_selected_dependencies_and_fresh_databases(
    tmp_path,
):
    from tau2.data_model.language import TaskMetadata
    from tau2.domains.intake.regional import (
        PublishedIntakeBundle,
        scenario_environment_kwargs,
    )
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_publish import build_regional_bundle

    proposal = identity_map()
    path = tmp_path / "map.json"
    write_localization_map(proposal, path)
    artifact = RegionalMapArtifact.model_validate_json(path.read_text())
    tasks = load_frozen_tasks()
    bundle = build_regional_bundle(
        artifact,
        proposal.source,
        {task.id: TaskMetadata(canonical_task_id=task.id) for task in tasks},
        Path(DATA_DIR),
    )
    assert isinstance(bundle, PublishedIntakeBundle)
    assert len(bundle.intake_localization.stimuli) == 200
    assert bundle.tasks == tasks
    first = scenario_environment_kwargs(bundle)
    second = scenario_environment_kwargs(bundle)
    first["user_db"].entities["injected"] = "unexpected"
    assert second["user_db"].entities == {}
    assert bundle.user_database == {"entities": {}}
    from tau2.multilingual.factory.regional_build import _task_packet

    packet = _task_packet(bundle, proposal.source, bundle.tasks[0], artifact)
    assert set(packet["effective_databases"]) == {"source", "localized"}
    assert packet["source_policies"]


def test_native_detector_counts_a_wrong_grapheme_restart_only_with_correct_redelivery():
    from tau2.domains.intake.spell_events import NativeSpellEventDetector

    detector = NativeSpellEventDetector(
        field_skeletons={"name": ["किरण"]}, digit_words={}
    )
    assert detector.detect("क, र") == []
    events = detector.detect("क, र. कि, र, ण")
    assert len(events) == 1
    assert events[0].letters_spoken == 5
    assert events[0].restarts == 1


def test_native_pronunciation_swaps_inside_unsegmented_speech_without_splitting_marks():
    from tau2.voice.utils.pronunciation_swap import apply_pronunciations

    swapped, events = apply_pronunciations(
        "我是李明。", {"李明": "李敏"}, grapheme_tokens=("李明",)
    )
    assert swapped == "我是李敏。"
    assert len(events) == 1
    assert apply_pronunciations("किरण", {"क": "ग"}, grapheme_tokens=("क",))[0] == "किरण"


def test_native_pack_loader_strategy_and_prompt_preview_share_runtime(
    tmp_path, monkeypatch
):
    import json

    import yaml

    from tau2.data_model.elicitation import ElicitationConfig, ElicitationStrategy
    from tau2.data_model.language import DomainBinding, TaskMetadata, TaskSelection
    from tau2.data_model.run_config import AudioNativeConfig
    from tau2.data_model.simulation import VoiceRunConfig
    from tau2.multilingual import runtime
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_publish import build_regional_bundle
    from tau2.multilingual.prompt_review import render_review
    from tau2.utils.artifacts import write_artifact

    proposal = hindi_candidate_map()
    pack = runtime.load_pack("hi")
    directory = tmp_path / "hi"
    directory.mkdir()
    path = tmp_path / "map.json"
    write_localization_map(proposal, path)
    artifact = RegionalMapArtifact.model_validate_json(path.read_text())
    source_tasks = load_frozen_tasks()
    metadata = {
        task.id + "_hi": TaskMetadata(canonical_task_id=task.id)
        for task in source_tasks
    }
    bundle = build_regional_bundle(artifact, proposal.source, metadata, Path(DATA_DIR))
    write_artifact(bundle, directory / "regional.json")
    (directory / "metadata.json").write_text(
        json.dumps(
            {key: value.model_dump(mode="json") for key, value in metadata.items()}
        )
    )
    pack = pack.model_copy(
        update={
            "domains": {
                "intake": DomainBinding(
                    revision="fixture",
                    task_metadata_file="metadata.json",
                    regional_bundle_file="regional.json",
                )
            }
        }
    )
    (directory / "pack.yaml").write_text(
        yaml.safe_dump(pack.model_dump(mode="json"), allow_unicode=True)
    )
    monkeypatch.setattr(runtime, "language_root", lambda: tmp_path)
    loaded = runtime.load_language(
        "hi",
        "intake",
        mode="voice",
        selection=TaskSelection(
            task_kwargs={"strategy": ElicitationStrategy.AGENT_DIRECTED.value}
        ),
    )
    tasks = loaded.tasks
    assert (
        "confirmed_with_user" not in tasks[0].evaluation_criteria.actions[0].arguments
    )
    context = runtime.resolve_task(loaded, tasks[0], 0)
    assert context.canonical_task_id == source_tasks[0].id
    report = render_review(
        languages=["hi"],
        domains=["intake"],
        template=VoiceRunConfig(
            domain="intake",
            audio_native_config=AudioNativeConfig(provider="openai"),
            elicitation=ElicitationConfig(rate=0),
        ),
    )
    assert report.pairs
    assert all(
        "first, authenticate" not in pair.agent_prompt.lower() for pair in report.pairs
    )
    assert all(
        "CUSTOMER answering an outbound" in pair.customer_prompt
        for pair in report.pairs
    )
    assert all("note_spell_request" in pair.customer_prompt for pair in report.pairs)


def test_intake_authoring_versions_explicit_spoken_utterance_exceptions(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from tau2.multilingual.factory import database_localization as drafting
    from tau2.multilingual.factory.intake_authoring import prompt_version

    proposal = identity_map()
    proposal.settings.language = "hi"
    source = next(
        entry
        for entry in proposal.source.entries
        if "intake caller_greeting;" in entry.context
    )
    generated = next(entry for entry in proposal.entries if entry.id == source.id)
    captured = []

    def fake_generate(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": generated.id,
                            "forms": generated.forms.model_dump(),
                            "operation": "preserve",
                            "notes": "Test source response only.",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", fake_generate)
    request = drafting.RegionalDraftRequest(
        source=proposal.source,
        settings=proposal.settings,
        revision="test",
        generation=FactoryModelConfig(model="test", reasoning_effort="low"),
    )
    for base in (
        drafting.DATABASE_LOCALIZATION_PROMPT_VERSION,
        drafting.DATABASE_LOCALIZATION_REVISION_VERSION,
    ):
        batch = drafting.BatchInput(
            request_sha256="0" * 64,
            entries=[source],
            completed=[],
            prompt_version=prompt_version("intake", base),
        )
        drafting._generate_batch(request, batch, tmp_path)
        payload = json.loads(captured[-1]["messages"][1].content)
        assert payload["entries"][0]["text_language"] == "hi"
        assert "Intake-specific exceptions" in captured[-1]["messages"][0].content
        assert (
            "TTS-ready distorted native tokens" in captured[-1]["messages"][0].content
        )
    assert (
        prompt_version("retail", drafting.DATABASE_LOCALIZATION_PROMPT_VERSION)
        == drafting.DATABASE_LOCALIZATION_PROMPT_VERSION
    )


def test_native_pronunciation_boundary_keeps_all_caps_case_sensitive():
    from tau2.voice.utils.pronunciation_swap import apply_pronunciations

    value, events = apply_pronunciations(
        "AS as", {"AS": "ay ess"}, grapheme_tokens=("AS",)
    )
    assert value == "ay ess as"
    assert len(events) == 1


def test_compiler_rejects_coherently_changed_medication_strength():
    from tau2.multilingual.factory.localization_artifact import read_localization_map

    proposal = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/es/localization/intake.json"
    )
    changed = 0
    for entry in proposal.entries:
        if entry.source == "Ondansetron 4 mg tablet":
            entry.forms.native = entry.forms.native.replace("4 mg", "40 mg")
            entry.forms.romanized = entry.forms.romanized.replace("4 mg", "40 mg")
            entry.operation = "adapt"
            changed += 1
    assert changed
    with pytest.raises(ValueError, match="medication strength"):
        compile_localization(proposal)


def test_compiler_rejects_native_script_protocol_email():
    proposal = hindi_candidate_map()
    source = next(
        row["written"]
        for document in proposal.source.documents
        if document.id == "intake_localization"
        for row in document.content["stimuli"]
        if row["family"] == "emails"
    )
    for entry in proposal.entries:
        if entry.source == source:
            entry.forms.native = "किरण@example.com"
            entry.forms.romanized = entry.forms.native
            entry.operation = "adapt"
    with pytest.raises(ValueError, match="ASCII"):
        compile_localization(proposal)


def test_native_pronunciation_rejects_normalization_collision():
    from tau2.voice.utils.pronunciation_swap import apply_pronunciations

    with pytest.raises(ValueError, match="Unicode normalization"):
        apply_pronunciations("é", {"é": "a", "e\u0301": "b"}, grapheme_tokens=("é",))


def test_selected_dependencies_follow_written_answers_without_cross_task_grouping():
    from tau2.multilingual.factory.database_localization import (
        _draft_groups,
        _relevant_mappings,
        _stage,
    )

    proposal = identity_map()
    dependency = next(
        entry
        for entry in proposal.source.entries
        if "dependency=pronunciations;" in entry.context and entry.source
    )
    assert any(
        "measured_selection=" in entry.context and '"tier"' in entry.context
        for entry in proposal.source.entries
        if any(
            occurrence.document_id == "intake_localization"
            and occurrence.path[-1] == "written"
            for occurrence in entry.occurrences
        )
    )
    assert isinstance(dependency.source, list)
    assert _stage(dependency) == 4
    related = _relevant_mappings([dependency], proposal.entries, source=proposal.source)
    assert any(
        "intake field:" in entry.context and _stage(entry) == 3 for entry in related
    )
    sibling = dependency.model_copy(
        update={
            "id": "other-task",
            "context": dependency.context.replace("entity=", "entity=other_"),
        }
    )
    assert len(_draft_groups([dependency, sibling])) == 2


def test_unknown_gender_donors_are_not_reused_for_intake(tmp_path, monkeypatch):
    from tau2.multilingual.factory.database_localization import _relevant_mappings
    from tau2.multilingual.factory.intake_authoring import reuse_intake_callees
    from tau2.multilingual.factory.localization_reuse import ReviewedReuseSource

    proposal = identity_map()
    directory = Path(DATA_DIR) / "tau2/multilingual/hi/localization"
    from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
    from tau2.utils.artifacts import ReviewApproval, content_digest, write_artifact

    donor = RegionalMapArtifact.model_validate_json(
        (directory / "retail.json").read_text()
    )
    for row in donor.entries:
        row.name_gender = None
    write_artifact(donor, tmp_path / "retail.json")
    write_artifact(
        ReviewApproval(
            content_sha256=content_digest(donor), reviewed_by="Synthetic test fixture"
        ),
        tmp_path / "approval.json",
    )
    reused = reuse_intake_callees(
        proposal.source,
        ReviewedReuseSource(
            map_file=tmp_path / "retail.json",
            approval_file=tmp_path / "approval.json",
        ),
        "hi",
    )
    # A reviewed retail name is not evidence that it matches this source person.
    # Legacy donor maps have no name-gender evidence; normal authoring fills them.
    assert reused == []
    assert (
        _relevant_mappings(proposal.source.entries, reused, source=proposal.source)
        == []
    )
    import json
    from types import SimpleNamespace

    from tau2.multilingual.factory import database_localization as drafting

    greeting = next(
        entry
        for entry in proposal.source.entries
        if "intake caller_greeting;" in entry.context
    )
    captured = []

    def fake_generate(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": [
                        {
                            "id": greeting.id,
                            "forms": {
                                "native": greeting.source,
                                "romanized": greeting.source,
                                "english": "Test source control.",
                            },
                            "operation": "preserve",
                            "notes": "Test.",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(drafting, "generate", fake_generate)
    request = drafting.RegionalDraftRequest(
        source=proposal.source,
        settings=proposal.settings,
        revision="test",
        generation=FactoryModelConfig(model="test", reasoning_effort="low"),
    )
    drafting._draft_entries(request, [greeting], list(reused), "0" * 64, tmp_path)
    payload = json.loads(captured[0]["messages"][1].content)
    assert payload["completed_mappings"] == []
    assert payload["request_contract"]["required_entry_ids"] == [greeting.id]
    assert all(
        entry.forms.native not in captured[0]["messages"][1].content for entry in reused
    )


def test_generator_rejects_local_clock_display_in_protocol_answer():
    from tau2.multilingual.factory.intake_authoring import validate_intake_forms

    proposal = identity_map()
    entry = next(
        entry
        for entry in proposal.entries
        if entry.source == "2:45 PM" and entry.context.startswith("intake field:")
    )
    validate_intake_forms(entry, entry.forms)
    with pytest.raises(ValueError, match="protocol answer"):
        validate_intake_forms(entry, entry.forms.model_copy(update={"native": "14:45"}))


def test_generator_validates_selected_pronunciation_before_checkpointing():
    from tau2.multilingual.factory.intake_authoring import validate_intake_forms

    proposal = identity_map()
    entry = next(
        entry
        for entry in proposal.entries
        if "dependency=pronunciations;" in entry.context and entry.source
    )
    validate_intake_forms(entry, entry.forms, proposal.entries)
    mismatch = entry.forms.model_copy(deep=True)
    mismatch.native[0]["token"] = "absent-from-the-written-answer"
    with pytest.raises(ValueError, match="exact substring"):
        validate_intake_forms(entry, mismatch, proposal.entries)
    noop = entry.forms.model_copy(deep=True)
    noop.native[0]["distorted"] = noop.native[0]["token"]
    with pytest.raises(ValueError, match="distortion must differ"):
        validate_intake_forms(entry, noop, proposal.entries)


def test_generator_rejects_native_phone_digits_before_checkpointing():
    from tau2.multilingual.factory.intake_authoring import validate_intake_forms

    proposal = identity_map()
    entry = next(
        e
        for e in proposal.entries
        if e.context.startswith("intake field:contact_phone;")
    )
    with pytest.raises(ValueError, match="ASCII"):
        validate_intake_forms(
            entry, entry.forms.model_copy(update={"native": "०११ ५५५० ०११७"})
        )


def test_generator_rejects_collapsed_phone_decoys_before_checkpointing():
    from tau2.multilingual.factory.intake_authoring import validate_intake_forms

    proposal = identity_map()
    entry = next(
        e
        for e in proposal.entries
        if "intake_phones_easy_" in e.context and "dependency=decoys;" in e.context
    )
    written_path = [*entry.occurrences[0].path[:-1], "written"]
    written = next(
        e
        for e in proposal.entries
        if any(
            o.document_id == "intake_localization" and o.path == written_path
            for o in e.occurrences
        )
    )
    collapsed = entry.forms.model_copy(deep=True)
    collapsed.native[0] = written.forms.native.replace("-", " ")
    with pytest.raises(ValueError, match="decoy collapses"):
        validate_intake_forms(entry, collapsed, proposal.entries)


@pytest.mark.parametrize("language", ["es", "pt", "hi", "ko", "zh"])
def test_complete_draft_candidate_maps_compile_both_forms_without_approval(language):
    from tau2.multilingual.factory.localization_artifact import read_localization_map

    path = (
        Path(DATA_DIR)
        / "tau2"
        / "multilingual"
        / language
        / "localization"
        / "intake.json"
    )
    proposal = read_localization_map(path)
    assert len(proposal.entries) == 2007
    callees = [entry for entry in proposal.entries if entry.category == "identity_name"]
    assert len(callees) == 39
    assert all(entry.name_gender is not None for entry in callees)
    # Changed donor names are newly authored and no longer claim donor approval.
    assert all(
        entry.category == "identity_name"
        for entry in proposal.entries
        if entry.reuse is not None
    )
    assert proposal.status == "draft"
    assert not path.with_name("intake.approval.json").exists()

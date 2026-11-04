"""Offline integration of released artifacts with the PR2 runtime contract."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import TextRunConfig, VoiceRunConfig
from tau2.multilingual import runtime
from tau2.runner.build import build_text_orchestrator, build_voice_orchestrator
from tau2.user.guidelines import (
    language_directive,
    language_reminder,
    render_simulation_guidelines,
)
from tau2.user.user_simulator import UserSimulator

DATA = Path(__file__).resolve().parents[2] / "data" / "tau2"
LANGUAGES = ("en", "es", "pt", "hi", "zh", "ko")
DOMAINS = ("airline", "retail", "telecom")


def test_release_contains_final_data_hashes_without_review_receipts():
    from tau2.multilingual.airport_catalog import AirportCatalog
    from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
    from tau2.utils.artifacts import content_digest

    root = DATA / "multilingual"
    assert not list(root.rglob("*.approval.json"))
    for language in LANGUAGES[1:]:
        directory = root / language
        for domain in DOMAINS:
            path = directory / "regional" / f"{domain}.json"
            raw = json.loads(path.read_text())
            assert "approval" not in raw and "airport_approval" not in raw
            assert "reviewed_by" not in path.read_text()
            artifact = RegionalMapArtifact.model_validate_json(
                (directory / "localization" / f"{domain}.json").read_text()
            )
            assert raw["localization_map_sha256"] == content_digest(artifact)
            if domain == "airline":
                catalog = AirportCatalog.model_validate_json(
                    (directory / "localization/airport-catalog.json").read_text()
                )
                assert raw["airport_catalog_sha256"] == content_digest(catalog)
            else:
                assert raw["airport_catalog_sha256"] is None


def _frozen_tasks(loaded, domain):
    """Reviewed tasks: English overrides replace only the frozen registry subset."""
    frozen = json.loads((DATA / "task_subsets" / f"{domain}_50.json").read_text())
    return [t for t in loaded.tasks if t.id in frozen["task_ids"]]


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("domain", DOMAINS)
def test_published_cell_resolves_exact_frozen_frame(language, domain):
    loaded = runtime.load_language(language, domain, mode="voice")
    assert loaded.scenario is not None
    pins = loaded.artifact_pins
    assert pins.scenario_sha256 and pins.runtime_rules_sha256
    if language == "en":
        assert loaded.scenario_mode == "pack"
        assert pins.localization_map_sha256 is None
    else:
        assert loaded.scenario_mode == "regional"
        assert loaded.pack.domains[domain].regional_bundle_file
        assert pins.localization_map_sha256
    frozen = json.loads((DATA / "task_subsets" / f"{domain}_50.json").read_text())
    contexts = [runtime.resolve_task(loaded, task, 42) for task in loaded.tasks]
    if language == "en":
        # English binds the full registry population; benchmarks select subsets.
        assert loaded.binding.source == "registry"
        ids = [context.canonical_task_id for context in contexts]
        assert set(frozen["task_ids"]) <= set(ids)
        contexts = [c for c in contexts if c.canonical_task_id in frozen["task_ids"]]
        tasks = [t for t in loaded.tasks if t.id in frozen["task_ids"]]
    else:
        tasks = loaded.tasks
    assert [context.canonical_task_id for context in contexts] == frozen["task_ids"]
    assert len(contexts) == 50
    assert all(
        context.caller_gender in (None, context.persona.gender) for context in contexts
    )
    if domain == "telecom":
        assert Counter(context.caller_gender for context in contexts) == {
            "male": 25,
            "female": 25,
        }
        assert [task.id for task in tasks] == frozen["task_ids"]
        assert all(
            "refuse" not in task.user_scenario.instructions.known_info.lower()
            for task in tasks
        )


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("domain", DOMAINS)
def test_source_backed_prompt_examples_for_both_genders(language, domain):
    loaded = runtime.load_language(language, domain, mode="voice")
    pack = loaded.pack
    assert pack.schema_version == 3
    for persona_id in pack.personas:
        persona = runtime.resolve_persona(
            pack, persona_id, mode="voice", binding=pack.binding(domain)
        )
        text_persona = runtime.resolve_persona(
            pack, persona_id, mode="text", binding=pack.binding(domain)
        )
        assert persona.voice.backchannel_phrases
        assert len(persona.voice.vocal_tics) == 3
        assert persona.gender_guidance.strip()
        if language == "en":
            assert persona.verbosity == "standard"
            assert persona.interrupt_tendency == "interrupts"
            assert persona.language_directive is None
            assert language_directive(persona) == ""
            assert persona.voice.spelled_entity_tables is None
            guidelines = persona.to_guidelines_text(mode="voice")
            for forbidden in (
                "Matt",
                "Lisa",
                "flat Midwestern accent",
                "New Jersey edge",
                "native US English",
                "never in English",
            ):
                assert forbidden not in guidelines
        else:
            assert "## LANGUAGE OF THE CALL" in language_directive(persona)
            assert "## LANGUAGE OF THE CONVERSATION" in language_directive(text_persona)
            assert "## FINAL REMINDER" in language_reminder(persona)
            assert "### Domain terms" in persona.to_guidelines_text(mode="voice")
            assert "### Phone numbers" not in text_persona.to_guidelines_text(
                mode="text"
            )
        rendered_modes = []
        for with_tools in (False, True):
            rendered = render_simulation_guidelines(persona, use_tools=with_tools)
            rendered_modes.append(rendered)
            assert pack.voice.spellout_section in rendered
            assert "[pause]" not in rendered
            assert (
                "If you have to repeat details, offer to spell things out"
                not in rendered
            )
            assert "<EXAMPLE:" not in rendered
            assert "${" not in rendered
        combined = "\n".join(rendered_modes)
        for example in persona.voice.voice_examples.model_dump().values():
            assert example.strip()
            assert example in combined


@pytest.mark.parametrize("language", LANGUAGES)
def test_customer_text_requires_spelling_and_anchor_requests(language):
    """Every shipped persona must obey the same request boundary in all domains."""
    anchors = {
        "en": "D as in dog",
        "es": "B de Barcelona",
        "pt": "A de Amélia",
        "hi": "R for Rajdhani",
        "zh": "弓长张",
        "ko": "기역, 이, 미음",
    }
    for domain in DOMAINS:
        loaded = runtime.load_language(language, domain, mode="voice")
        spellout = loaded.pack.voice.spellout_section
        assert "only when the agent explicitly asks you to spell it" in spellout
        assert "a spelling request alone does not request anchors" in spellout
        assert anchors[language] in spellout
        for persona_id in loaded.pack.personas:
            caller = runtime.resolve_persona(
                loaded.pack,
                persona_id,
                mode="voice",
                binding=loaded.pack.binding(domain),
            )
            text = "\n".join(
                [language_directive(caller), caller.to_guidelines_text(mode="voice")]
            )
            for forbidden in (
                "When the agent mishears",
                "When dictating account numbers",
                "enunciate",
                "distinción",
                "grathias",
                "retroflex",
                "erhua",
                "pitch-accent",
                "intonation",
                "pronounced correctly",
                "sarcastic edge",
                "politeness surfaces as sarcasm",
                "When it's NOT resolved",
            ):
                assert forbidden not in text
            examples = caller.voice.voice_examples
            assert all(
                "[pause]" not in value for value in examples.model_dump().values()
            )
            assert "@gmail.com" in examples.one_at_a_time_example


@pytest.mark.parametrize("language", LANGUAGES)
def test_real_builders_use_pack_voice_and_localized_greeting(language):
    for dependency in ("elevenlabs", "scipy", "pydub"):
        pytest.importorskip(dependency)
    loaded = runtime.load_language(language, "retail", mode="voice")
    task = loaded.tasks[0]
    text = build_text_orchestrator(
        TextRunConfig(domain="retail", language=language), task
    )
    voice = build_voice_orchestrator(
        VoiceRunConfig(
            domain="retail", language=language, audio_native_config=AudioNativeConfig()
        ),
        task,
    )
    assert text.initial_agent_message.content == loaded.pack.agent_greeting
    assert voice.initial_agent_message.content == (
        loaded.pack.voice.agent_greeting or loaded.pack.agent_greeting
    )
    text_context = text.call_context.language
    voice_context = voice.call_context.language
    assert text_context.mode == "text" and voice_context.mode == "voice"
    assert text_context.persona_id == voice_context.persona_id
    assert (
        voice.user.voice_settings.synthesis_config.provider_config.language_code
        == language
    )
    assert (
        voice.user.voice_settings.synthesis_config.spelled_entity_tables is not None
        if language == "pt"
        else voice.user.voice_settings.synthesis_config.spelled_entity_tables is None
    )
    assert voice.agent._adapter is None
    assert loaded.pack.agent_language_clause in voice.agent.system_prompt
    if language != "en":
        assert voice.user.system_prompt.endswith(
            language_reminder(voice_context.persona)
        )
        assert text.user.system_prompt.endswith(language_reminder(text_context.persona))
        assert "### Phone numbers" in voice.user.system_prompt
        assert "### Phone numbers" not in text.user.system_prompt


def test_english_voices_and_telecom_names_are_explicit():
    loaded = runtime.load_language("en", "telecom", mode="voice")
    assert {pid: p.voice.voice_id for pid, p in loaded.pack.personas.items()} == {
        "mamadou_diallo": "ET3963lBcRmodt3ZaTBS",
        "priya_patil": "mnHhNJntmsPxJsZvYVM7",
    }
    assert "mirror the user's register" not in loaded.pack.agent_language_clause
    # Reviewed caller diversity covers the frozen subset of the registry population.
    for task in _frozen_tasks(loaded, "telecom"):
        context = runtime.resolve_task(loaded, task, 42)
        name = "Mamadou Diallo" if context.caller_gender == "male" else "Priya Patil"
        assert name in task.user_scenario.instructions.known_info


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("use_tools", (False, True))
def test_published_text_personas_reach_actual_user_prompt(
    language, use_tools, monkeypatch
):
    """Published packs must include their reviewed persona in both text modes."""
    monkeypatch.setattr(
        "tau2.user.user_simulator.generate",
        lambda *args, **kwargs: pytest.fail("Prompt assembly must not call a model"),
    )
    loaded = runtime.load_language(language, "telecom", mode="text")
    for persona_id in loaded.pack.personas:
        caller = runtime.resolve_persona(
            loaded.pack,
            persona_id,
            mode="text",
            binding=loaded.pack.binding("telecom"),
        )
        rendered = UserSimulator(
            llm="offline-fixture",
            instructions="An offline scenario.",
            tools=[] if use_tools else None,
            persona_config=caller,
        ).system_prompt
        assert rendered.count(caller.to_guidelines_text(mode="text")) == 1
        assert rendered.count(caller.gender_guidance) == 1
        assert rendered.index(caller.gender_guidance) < rendered.index(
            "## Task Completion"
        )
        assert "<PERSONA_GUIDELINES>" not in rendered
        assert "### Phone numbers" not in rendered


class MigratedArtifact(BaseModel):
    """One hand-migrated published file and the transform that produced it."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform: str = Field(min_length=1)


class MigrationManifest(BaseModel):
    """Published-file digests for packs migrated from an unpublished snapshot."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    source: Literal["tau2 multilingual research snapshot (not published)"]
    pack_transform_version: Literal["regional-reviewed-pack-v1"]
    publication_scope: Literal["runtime-and-frozen-subsets"]
    artifacts: list[MigratedArtifact] = Field(min_length=1)


def test_published_artifact_hashes_match_migration_manifest():
    root = DATA.parents[1]
    manifest = MigrationManifest.model_validate_json(
        (DATA / "multilingual" / "migration_manifest.json").read_text()
    )
    recorded = {entry.path for entry in manifest.artifacts}
    actual = {
        path.relative_to(root).as_posix()
        for path in (DATA / "multilingual").glob("*/*")
        if path.is_file()
    }
    actual.update(
        path.relative_to(root).as_posix()
        for path in (DATA / "task_subsets").glob("*_50.json")
    )
    assert recorded == actual
    for entry in manifest.artifacts:
        assert (
            hashlib.sha256((root / entry.path).read_bytes()).hexdigest() == entry.sha256
        )
        assert "docs/designs/" not in entry.transform
        assert "telecom_build_manifest.json" not in entry.transform


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_retail_caller_has_consistent_authentication_facts(language):
    from tau2.multilingual.names import clarify_retail_name, resolve_retail_caller
    from tau2.runner.build import build_environment

    loaded = runtime.load_language(language, "retail", mode="voice")
    control = runtime.load_language("en", "retail", mode="voice")
    originals = {control.metadata[t.id].canonical_task_id: t for t in control.tasks}
    aliases = {}
    if language != "en":
        artifact = json.loads(
            (DATA / "multilingual" / language / "localization/retail.json").read_text()
        )
        aliases = {
            row["canonical"]: row["localized"]
            for row in artifact["user_id_revision"]["aliases"]
        }

    def tools_for(binding, task):
        environment = build_environment(
            "retail", env_kwargs=runtime.language_environment_kwargs(binding)
        )
        state = task.initial_state
        environment.set_state(
            initialization_data=state.initialization_data if state else None,
            initialization_actions=state.initialization_actions if state else None,
            message_history=(state.message_history or []) if state else [],
        )
        return environment.tools

    for task in _frozen_tasks(loaded, "retail"):
        original = originals[loaded.metadata[task.id].canonical_task_id]
        tools = tools_for(loaded, task)
        caller = resolve_retail_caller(
            task, tools, original, tools_for(control, original), aliases
        )
        assert (
            clarify_retail_name(
                task,
                caller_id=caller.user_id,
                caller=caller,
                name_order=loaded.pack.name_order,
            )
            == task
        ), task.id
        assert (
            tools.find_user_id_by_name_zip(
                first_name=caller.name.first_name,
                last_name=caller.name.last_name,
                zip=caller.address.zip,
            )
            == caller.user_id
        ), task.id
        instructions = str(task.user_scenario.instructions)
        assert caller.address.zip in instructions or caller.email in instructions, (
            task.id
        )

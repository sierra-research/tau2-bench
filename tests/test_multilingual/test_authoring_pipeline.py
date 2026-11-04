"""Offline authoring through the actual runtime contract; no paid providers."""

import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.data_model.language import NameOrder
from tau2.multilingual.factory import pipeline
from tau2.multilingual.factory.pipeline import (
    add_domain,
    draft,
    finalize,
)
from tau2.multilingual.factory.prompt_rules import REQUESTED_SPELLING_SECTION
from tau2.multilingual.factory.request_files import json_digest
from tau2.multilingual.factory.schema import (
    AddDomainRequest,
    AuthorForm,
    DomainBundle,
    DraftRequest,
    FactoryModelConfig,
)
from tau2.multilingual.factory.voice_assets import (
    ProvisionedVoice,
    VoiceAssetsRequest,
    VoiceDesignConfig,
    VoicePreview,
    VoiceSpec,
    listening_digest,
    prepare_voices,
)
from tau2.utils.artifacts import (
    ReviewApproval,
    content_digest,
)


@pytest.fixture(params=["en", "ro"])
def draft_request(request):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    task = json.loads((root / "data/tau2/domains/retail/tasks.json").read_text())[0]
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.environment import get_environment
    from tau2.multilingual.names import (
        clarify_retail_name,
        resolve_retail_caller,
    )

    task = Task.model_validate(task)
    tools = get_environment().tools
    caller = resolve_retail_caller(task, tools, task, tools, {})
    task = clarify_retail_name(
        task, caller_id=caller.user_id, caller=caller, name_order=NameOrder.GIVEN_FIRST
    ).model_dump(mode="json")
    form = AuthorForm.model_validate(
        {
            "identity": {
                "language": request.param,
                "display_name": "Fixture language",
                "script": "latn",
            },
            "locale_notes": "Fixture only",
            "preferences": "Fixture only",
            "personas": [
                {"name": "Mamadou", "gender": "male", "sketch": "Fixture A"},
                {"name": "Priya", "gender": "female", "sketch": "Fixture B"},
            ],
        }
    )
    return DraftRequest(
        form=form,
        revision="fixture-v1",
        generation=FactoryModelConfig(model="fixture", reasoning_effort="minimal"),
        speech_insert_events_per_minute=2,
        backchannel_level="medium",
        domains=[
            DomainBundle(
                domain="retail",
                revision="fixture-v1",
                tasks=[task],
                metadata={
                    task["id"]: {
                        "canonical_task_id": task["id"],
                        "caller_gender": "male",
                    }
                },
                reviewed_by="fixture reviewer",
            )
        ],
    )


def reply(language="en"):
    prompt = json.loads(
        (Path(__file__).parent / "fixtures/reviewed_prompt_content.json").read_text()
    )["en"]["prompt_content"]
    return {
        "agent_greeting": "Fixture greeting",
        "agent_language_clause": "Speak English.",
        "user_guidelines": "",
        "spellout_section": REQUESTED_SPELLING_SECTION,
        "personas": {
            name: {
                "gender": gender,
                "locale": "fixture",
                "variety": "fixture",
                "pragmatics_clauses": ["Fixture register."],
                "gender_guidance": f"You are a {gender} caller.",
                **(
                    {"language_directive": {"origin": "Fixture region"}}
                    if language != "en"
                    else {}
                ),
                "voice": {
                    "backchannel_phrases": ["Okay"],
                    "voice_examples": copy.deepcopy(prompt["voice_examples"]),
                },
            }
            for name, gender in [("Mamadou", "male"), ("Priya", "female")]
        },
    }


def fake_generate(**kwargs):
    """Reply with fixture content for the language named in the draft prompt."""
    language = re.search(r'"language":"([^"]+)"', kwargs["messages"][1].content)
    return SimpleNamespace(content=json.dumps(reply(language.group(1))))


def fixture_audio():
    import base64
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 1600)
    return base64.b64encode(buffer.getvalue()).decode()


class FakeVoices:
    def __init__(self):
        self.designs = 0
        self.creates = 0

    def preview(self, spec, config):
        self.designs += 1
        return VoicePreview(
            generated_voice_id=f"preview-{self.designs}",
            audio_base64=fixture_audio(),
            media_type="audio/wav",
        )

    def create(self, persona_id, preview, spec):
        self.creates += 1
        return ProvisionedVoice(voice_id=f"voice-{self.creates}")

    def validate(self, voice_id):
        return ProvisionedVoice(voice_id=voice_id)


def test_complete_draft_assets_finalize_and_runtime_load(
    draft_request, tmp_path, monkeypatch
):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return fake_generate(**kwargs)

    monkeypatch.setattr(pipeline, "generate", generate)
    staged = draft(draft_request, tmp_path / "draft.json")
    assert draft(draft_request, tmp_path / "draft.json") == staged
    assert len(calls) == 1
    assert calls[0]["call_name"] == "factory_draft"
    assert staged.pack.schema_version == 3
    assert (
        staged.prompt_version == pipeline.DRAFT_PROMPT_VERSION == "public-pack-draft-v9"
    )
    assert calls[0]["messages"][1].content == pipeline.render_draft_prompt(
        draft_request.form
    )
    from tau2.voice_config import VOCAL_TICS

    assert all(p.voice.vocal_tics == VOCAL_TICS for p in staged.pack.personas.values())
    assert "language-guidelines-v4" in calls[0]["messages"][1].content
    assert pipeline.AUTHORING_RULES in calls[0]["messages"][1].content
    assert "only when explicitly asked" in calls[0]["messages"][1].content
    assert "goes in voice.pragmatics_clauses" in calls[0]["messages"][1].content
    with pytest.raises(ValueError, match="assets"):
        finalize(staged, None, None, tmp_path / "not-published")
    specs = {
        name: VoiceSpec(
            description="Fixture voice design",
            audition_text="Fixture native audition sentence. " * 5,
        )
        for name in staged.pack.personas
    }
    assets_request = VoiceAssetsRequest(
        config=VoiceDesignConfig(
            model="fixture-voice", seed=42, loudness=0.5, guidance_scale=38
        ),
        draft=staged,
        specs=specs,
        review=ReviewApproval(
            content_sha256=content_digest(staged), reviewed_by="fixture"
        ),
    )
    provider = FakeVoices()
    assets = prepare_voices(assets_request, tmp_path / "assets", provider)
    assert prepare_voices(assets_request, tmp_path / "assets", provider) == assets
    assert provider.designs == provider.creates == 2
    review = ReviewApproval(
        content_sha256=listening_digest(assets), reviewed_by="fixture listener"
    )
    output = tmp_path / "tau2/multilingual" / staged.pack.language
    assert finalize(staged, assets, review, output, assets_root=tmp_path / "assets")
    assert not finalize(staged, assets, review, output, assets_root=tmp_path / "assets")
    import tau2.utils
    from tau2.multilingual.runtime import load_language

    if staged.pack.language == "en":
        shutil.copytree(
            tau2.utils.DATA_DIR / "tau2/domains/retail",
            tmp_path / "tau2/domains/retail",
        )
    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    loaded = load_language(staged.pack.language, "retail", mode="voice")
    assert loaded.tasks == draft_request.domains[0].tasks
    assert set(loaded.pack.personas) == {"Mamadou", "Priya"}
    assert all(
        p.voice.voice_id.startswith("voice-") for p in loaded.pack.personas.values()
    )
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance == {
        "schema_version": 3,
        "draft_sha256": content_digest(staged),
        "assets_sha256": content_digest(assets),
        "files": {
            path.relative_to(output).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in output.rglob("*")
            if path.is_file() and path.name != "provenance.json"
        },
    }
    for path in output.rglob("*"):
        if path.is_file():
            assert "reviewed_by" not in path.read_text()
            assert "fixture listener" not in path.read_text()
    assert (
        json.loads((tmp_path / "draft.json").read_text())["domains"]["retail"][
            "reviewed_by"
        ]
        == "fixture reviewer"
    )
    (output / "pack.yaml").write_text("changed")
    with pytest.raises(FileExistsError):
        finalize(staged, assets, review, output, assets_root=tmp_path / "assets")


def test_add_domain_preserves_content_and_rejects_overwrite(
    draft_request, tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    staged = draft(draft_request, tmp_path / "draft.json")
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    task = json.loads((root / "data/tau2/domains/airline/tasks.json").read_text())[0]
    bundle = DomainBundle(
        domain="airline",
        revision="fixture-v1",
        tasks=[task],
        metadata={task["id"]: {"canonical_task_id": task["id"]}},
        reviewed_by="fixture",
    )
    result = add_domain(
        AddDomainRequest(draft=staged, domain=bundle, revision="fixture-v2"),
        tmp_path / "expanded.json",
    )
    assert result.pack.personas == staged.pack.personas
    assert set(result.pack.domains) == {"airline", "retail"}
    with pytest.raises(ValueError, match="already"):
        add_domain(
            AddDomainRequest(draft=result, domain=bundle, revision="fixture-v3"),
            tmp_path / "bad.json",
        )


@pytest.mark.parametrize("defect", ["persona", "unknown", "coaching", "spelling"])
def test_invalid_generated_content_stays_unpublished(
    draft_request, tmp_path, monkeypatch, defect
):
    value = copy.deepcopy(reply(draft_request.form.identity.language))
    if defect == "persona":
        value["personas"]["Priya"]["gender"] = "male"
    if defect == "unknown":
        value["annotation"] = "forbidden"
    if defect == "coaching":
        value["personas"]["Priya"]["pragmatics_clauses"] = [
            "Speak with a flat Midwestern accent."
        ]
    if defect == "spelling":
        value["spellout_section"] = "Always spell names."
    monkeypatch.setattr(
        pipeline, "generate", lambda **kw: SimpleNamespace(content=json.dumps(value))
    )
    with pytest.raises(ValueError):
        draft(draft_request, tmp_path / "draft.json")
    assert not (tmp_path / "draft.json").exists()


def test_draft_prompt_states_scenario_language_and_phrase_budgets(draft_request):
    prompt = pipeline.render_draft_prompt(draft_request.form)
    assert "The scenario language" in prompt
    assert "is English ('en')" in prompt
    if draft_request.form.identity.language == "en":
        assert "must omit language_directive" in prompt
        assert "MUST supply language_directive.origin" not in prompt
    else:
        assert "'ro' differs from the scenario language" in prompt
        assert "MUST supply language_directive.origin" in prompt
    assert "backchannel_phrases holds 1-2 pure" in prompt
    assert "non_directed_phrases holds at most 4 glances" in prompt
    assert "at most 4 words each" in prompt


def test_draft_repairs_one_invalid_reply_with_its_validation_error(
    draft_request, tmp_path, monkeypatch
):
    language = draft_request.form.identity.language
    bad = reply(language)
    bad["personas"]["Priya"]["voice"]["non_directed_phrases"] = [
        "one two three four five six"
    ]
    replies = [bad, reply(language)]
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=json.dumps(replies[len(calls) - 1]))

    monkeypatch.setattr(pipeline, "generate", generate)
    staged = draft(draft_request, tmp_path / "draft.json")
    assert [c["call_name"] for c in calls] == ["factory_draft", "factory_draft_repair"]
    repair = calls[1]["messages"]
    assert repair[:2] == calls[0]["messages"]
    assert "the budget is 4 words" in repair[2].content
    assert "one two three four five six" in repair[2].content
    assert staged.pack.personas["Priya"].voice.non_directed_phrases == []


def test_draft_repair_is_bounded_to_one_call(draft_request, tmp_path, monkeypatch):
    bad = reply(draft_request.form.identity.language)
    bad["personas"]["Priya"]["voice"]["backchannel_phrases"] = ["a b c d e"]
    calls = []

    def generate(**kwargs):
        calls.append(kwargs["call_name"])
        return SimpleNamespace(content=json.dumps(bad))

    monkeypatch.setattr(pipeline, "generate", generate)
    with pytest.raises(ValueError, match="budget"):
        draft(draft_request, tmp_path / "draft.json")
    assert calls == ["factory_draft", "factory_draft_repair"]
    assert not (tmp_path / "draft.json").exists()


def test_asset_failure_resume_and_review_invalidation(
    draft_request, tmp_path, monkeypatch
):
    from tau2.multilingual.factory.voice_assets import audition_samples, recover_voice

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    staged = draft(draft_request, tmp_path / "draft.json")
    specs = {
        name: VoiceSpec(
            description="Fixture voice", audition_text="Fixture audition. " * 10
        )
        for name in staged.pack.personas
    }
    req = VoiceAssetsRequest(
        config=VoiceDesignConfig(
            model="fixture-voice", seed=42, loudness=0.5, guidance_scale=38
        ),
        draft=staged,
        specs=specs,
        review=ReviewApproval(
            content_sha256=content_digest(staged), reviewed_by="fixture"
        ),
    )

    class FailsAfterCreate(FakeVoices):
        def create(self, *args):
            super().create(*args)
            raise TimeoutError("created remotely but response lost")

    failed = FailsAfterCreate()
    with pytest.raises(TimeoutError):
        prepare_voices(req, tmp_path / "assets", failed)
    with pytest.raises(RuntimeError, match="record-created-voice"):
        prepare_voices(req, tmp_path / "assets", failed)
    assert failed.creates == 1
    provider = FakeVoices()
    recover_voice(tmp_path / "assets", "Mamadou", "recovered-voice", provider)
    result = prepare_voices(req, tmp_path / "assets", provider)
    assert provider.creates == 1
    assert result.voices["Mamadou"].voice.voice_id == "recovered-voice"
    paths = audition_samples(
        result, tmp_path / "samples", assets_root=tmp_path / "assets"
    )
    assert len(paths) == 2 and all(p.suffix == ".wav" for p in paths)
    assert (
        audition_samples(result, tmp_path / "samples", assets_root=tmp_path / "assets")
        == paths
    )
    invalid = ReviewApproval(content_sha256="0" * 64, reviewed_by="fixture")
    with pytest.raises(ValueError, match="review"):
        finalize(
            staged,
            result,
            invalid,
            tmp_path / "invalid",
            assets_root=tmp_path / "assets",
        )
    assert not (tmp_path / "invalid").exists()
    changed = req.model_copy(deep=True)
    changed.config.seed += 1
    with pytest.raises(FileExistsError):
        prepare_voices(changed, tmp_path / "assets", provider)
    assert provider.creates == 1


def test_package_cli_registers_stages_and_validates_offline(
    draft_request, tmp_path, monkeypatch
):
    import argparse

    from tau2.multilingual.factory.cli import add_factory_args

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    staged = draft(draft_request, tmp_path / "draft.json")
    parser = argparse.ArgumentParser()
    add_factory_args(parser)
    args = parser.parse_args(["validate", "--request", str(tmp_path / "draft.json")])
    args.func(args)
    for stage in [
        "draft",
        "add-domain",
        "inventory-database",
        "review-content",
        "prepare-voices",
        "audition",
        "reuse-voices",
        "validate-voices",
        "validate",
        "finalize",
        "record-created-voice",
    ]:
        assert stage in parser.format_help()
    assert staged.pack.revision == "fixture-v1"


def test_existing_pack_domain_extension_reuses_voices(
    draft_request, tmp_path, monkeypatch
):
    import yaml

    from tau2.data_model.language import LanguagePack
    from tau2.multilingual.factory.pipeline import stage_pack
    from tau2.multilingual.factory.schema import StagePackRequest
    from tau2.multilingual.factory.voice_assets import (
        AuditionSpec,
        ExistingVoiceConfig,
        ReuseAssetsRequest,
        reuse_voices,
    )

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    staged = draft(draft_request, tmp_path / "draft.json")
    pack_data = staged.pack.model_dump()
    for name, persona in pack_data["personas"].items():
        persona["voice"]["voice_id"] = f"existing-{name}"
    pack_data["domains"]["retail"]["tasks_file"] = "retail_tasks_en.json"
    pack_data["domains"]["retail"]["task_metadata_file"] = "retail_metadata_en.json"
    published = LanguagePack.model_validate(pack_data)
    imported = stage_pack(
        StagePackRequest(
            pack=published, domains=list(staged.domains.values()), revision="import-v2"
        ),
        tmp_path / "import.json",
    )
    assert imported.generation is None

    class ExistingVoices(FakeVoices):
        def audition(self, voice_id, spec, model):
            return VoicePreview(
                generated_voice_id=voice_id,
                audio_base64=fixture_audio(),
                media_type="audio/wav",
            )

    provider = ExistingVoices()
    request = VoiceAssetsRequest(
        draft=imported,
        review=ReviewApproval(
            content_sha256=content_digest(imported), reviewed_by="fixture"
        ),
        specs={
            name: AuditionSpec(audition_text="Fixture native audition. " * 10)
            for name in imported.pack.personas
        },
        config=ExistingVoiceConfig(
            voice_ids={
                name: p.voice.voice_id for name, p in published.personas.items()
            },
            model="fixture-tts",
        ),
    )
    old_assets = prepare_voices(request, tmp_path / "existing-assets", provider)
    assert provider.designs == provider.creates == 0
    from pathlib import Path

    task = json.loads(
        (
            Path(__file__).resolve().parents[2] / "data/tau2/domains/airline/tasks.json"
        ).read_text()
    )[0]
    airline = DomainBundle(
        domain="airline",
        revision="fixture",
        tasks=[task],
        metadata={task["id"]: {"canonical_task_id": task["id"]}},
        reviewed_by="fixture",
    )
    expanded = add_domain(
        AddDomainRequest(draft=imported, domain=airline, revision="import-v3"),
        tmp_path / "expanded.json",
    )
    assets = reuse_voices(
        ReuseAssetsRequest(
            previous_draft=imported,
            previous_assets=old_assets,
            previous_assets_root=tmp_path / "existing-assets",
            draft=expanded,
            review=ReviewApproval(
                content_sha256=content_digest(expanded), reviewed_by="fixture"
            ),
        ),
        tmp_path / "reused",
    )
    assert assets.parent_sha256 == content_digest(old_assets)
    assert assets.config.model == "fixture-tts"
    # The listening approval of the original voices covers the domain-only reuse.
    approval = ReviewApproval(
        content_sha256=listening_digest(old_assets), reviewed_by="fixture listener"
    )
    finalize(
        expanded,
        assets,
        approval,
        tmp_path / "expanded-pack",
        assets_root=tmp_path / "reused",
    )
    result = LanguagePack.model_validate(
        yaml.safe_load((tmp_path / "expanded-pack/pack.yaml").read_text())
    )
    assert set(result.domains) == {"retail", "airline"}
    assert {
        k: p.voice.voice_id for k, p in result.personas.items()
    } == request.config.voice_ids


def test_live_adapter_mapping_uses_reviewed_native_text_and_explicit_settings():
    from tau2.multilingual.factory.voice_assets import ElevenLabsVoiceProvider

    calls = []

    def design(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            previews=[
                SimpleNamespace(
                    generated_voice_id="preview",
                    audio_base_64=fixture_audio(),
                    media_type="audio/wav",
                )
            ]
        )

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(voice_id="saved")

    adapter = object.__new__(ElevenLabsVoiceProvider)
    adapter.client = SimpleNamespace(
        text_to_voice=SimpleNamespace(design=design, create=create),
        voices=SimpleNamespace(get=lambda voice_id: SimpleNamespace(voice_id=voice_id)),
    )
    spec = VoiceSpec(
        description="Fixture acoustic brief",
        audition_text="Fixture reviewed native text. " * 5,
    )
    config = VoiceDesignConfig(
        model="fixture-model", seed=7, loudness=0.4, guidance_scale=12
    )
    preview = adapter.preview(spec, config)
    assert preview.audio()
    assert calls[0]["text"] == spec.audition_text
    assert calls[0]["auto_generate_text"] is False
    assert calls[0]["seed"] == 7 and calls[0]["model_id"] == "fixture-model"
    voice = adapter.create("Fixture persona", preview, spec)
    assert adapter.validate(voice.voice_id) == voice
    assert calls[1]["generated_voice_id"] == "preview"


@pytest.mark.parametrize(
    "defect",
    [
        "missing-slot",
        "extra-slot",
        "old-template",
        "old-replacement",
        "missing-greeting",
        "blank-greeting",
        "persona-in-agent-clause",
    ],
)
def test_draft_requires_current_complete_language_owned_content(
    draft_request, tmp_path, monkeypatch, defect
):
    content = reply(draft_request.form.identity.language)
    persona = content["personas"]["Priya"]
    prompt = persona["voice"]
    if defect == "missing-slot":
        del prompt["voice_examples"]["tool_result_report"]
    elif defect == "extra-slot":
        prompt["voice_examples"]["invented_slot"] = "Fixture"
    elif defect == "old-template":
        persona["prompt_content"] = {"template_version": "language-guidelines-v2"}
    elif defect == "old-replacement":
        persona["voice_example_replacements"] = []
    elif defect == "missing-greeting":
        del content["agent_greeting"]
    elif defect == "persona-in-agent-clause":
        content["agent_language_clause"] = "Speak English with priya."
    else:
        content["agent_greeting"] = "   "
    monkeypatch.setattr(
        pipeline, "generate", lambda **kw: SimpleNamespace(content=json.dumps(content))
    )
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        draft(draft_request, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "field,bad",
    [
        (
            "agent_background_clause",
            "When lookup fails, ask the customer to spell their name.",
        ),
        ("gender_guidance", "Use a rising pitch and speak at a slower rate."),
        ("voice_examples", "Always spell your name."),
    ],
)
def test_generated_nested_guidance_is_rejected_before_writing(
    draft_request, tmp_path, monkeypatch, field, bad
):
    content = reply(draft_request.form.identity.language)
    persona = content["personas"]["Priya"]
    if field == "voice_examples":
        persona["voice"][field]["tool_failure_report"] = bad
    else:
        persona[field] = bad
    monkeypatch.setattr(
        pipeline, "generate", lambda **kw: SimpleNamespace(content=json.dumps(content))
    )
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="authoring instruction"):
        draft(draft_request, output)
    assert not output.exists()


def test_vocal_tics_must_be_audio_tags(draft_request, tmp_path, monkeypatch):
    from tau2.multilingual.factory.schema import DraftPack, validate_content

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    staged = draft(draft_request, tmp_path / "draft.json")
    data = staged.pack.model_dump()
    data["personas"]["Priya"]["voice"]["vocal_tics"] = ["este..."]
    with pytest.raises(ValueError, match="audio tags"):
        validate_content(DraftPack.model_validate(data))


def test_prior_draft_prompt_cannot_resume_as_current(
    draft_request, tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    output = tmp_path / "draft.json"
    staged = draft(draft_request, output)
    staged.prompt_version = "public-pack-draft-v3"
    output.write_text(staged.model_dump_json())
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="inputs changed"):
        draft(draft_request, output)
    assert output.read_bytes() == before


def test_draft_renders_all_modes_without_reading_legacy_english(
    draft_request, tmp_path, monkeypatch
):
    import tau2.user.guidelines as guidelines

    def forbidden(*args, **kwargs):
        pytest.fail("Language authoring must not fall back to the default persona")

    monkeypatch.setattr(guidelines, "default_persona", forbidden)
    calls = set()
    original = guidelines.render_simulation_guidelines

    def render(persona, *, use_tools):
        calls.add((persona.mode, use_tools))
        return original(persona, use_tools=use_tools)

    monkeypatch.setattr(guidelines, "render_simulation_guidelines", render)
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    draft(draft_request, tmp_path / "draft.json")
    assert calls == {
        (mode, tools) for mode in ("text", "voice") for tools in (False, True)
    }


@pytest.mark.parametrize(
    "guidance",
    [
        "Use a rising pitch and speak at a slower rate.",
        "Pronounce each consonant with careful enunciation.",
        "When the lookup fails, ask the customer to spell their name.",
        "Ask for the customer's full name and date of birth before continuing.",
        "Authenticate using a phone lookup, then fall back to name lookup.",
        "Look up the customer by email before asking for anything else.",
    ],
)
@pytest.mark.parametrize("voice_only", [False, True])
def test_generated_draft_rejects_retired_instructions(
    draft_request, tmp_path, monkeypatch, guidance, voice_only
):
    content = reply(draft_request.form.identity.language)
    persona = content["personas"]["Priya"]
    (persona["voice"] if voice_only else persona)["pragmatics_clauses"] = [guidance]
    monkeypatch.setattr(
        pipeline, "generate", lambda **kw: SimpleNamespace(content=json.dumps(content))
    )
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="authoring instruction"):
        draft(draft_request, output)
    assert not output.exists()


@pytest.mark.parametrize("entry", ["manual", "add-domain", "import", "finalize"])
@pytest.mark.parametrize(
    "nested_field", ["agent_background_clause", "gender_guidance", "voice_examples"]
)
@pytest.mark.parametrize(
    "bad",
    [
        "If lookup fails, ask the customer to spell their name.",
        "Use a rising pitch and speak at a slower rate.",
    ],
)
def test_manual_paths_revalidate_auth_guidance(
    draft_request, tmp_path, monkeypatch, entry, bad, nested_field
):
    from tau2.data_model.language import LanguagePack
    from tau2.multilingual.factory.pipeline import stage_pack
    from tau2.multilingual.factory.schema import DraftArtifact, StagePackRequest

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    staged = draft(draft_request, tmp_path / "valid.json")
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match="authoring instruction"):
        if entry == "add-domain":
            from pathlib import Path

            task = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "data/tau2/domains/airline/tasks.json"
                ).read_text()
            )[0]
            bundle = DomainBundle(
                domain="airline",
                revision="fixture-v1",
                tasks=[task],
                metadata={task["id"]: {"canonical_task_id": task["id"]}},
                reviewed_by="fixture",
                voice_guidelines=bad,
            )
            add_domain(
                AddDomainRequest(draft=staged, domain=bundle, revision="bad-v2"), output
            )
        else:
            persona = staged.pack.personas["Priya"]
            if nested_field == "voice_examples":
                persona.voice.voice_examples.tool_failure_report = bad
            else:
                setattr(persona, nested_field, bad)
            if entry == "manual":
                DraftArtifact.model_validate_json(staged.model_dump_json())
            elif entry == "finalize":
                # No assets: content must fail first, before publication checks.
                finalize(staged, None, None, output)
            else:
                data = staged.pack.model_dump()
                for persona in data["personas"].values():
                    persona["voice"]["voice_id"] = "fixture-voice"
                stage_pack(
                    StagePackRequest(
                        pack=LanguagePack.model_validate(data),
                        domains=list(staged.domains.values()),
                        revision="bad-v2",
                    ),
                    output,
                )
    assert not output.exists()


def test_text_content_and_separate_acoustic_brief_remain_supported(
    draft_request, tmp_path, monkeypatch
):
    content = reply(draft_request.form.identity.language)
    content["agent_language_clause"] += (
        " First, authenticate the customer according to the domain policy."
    )
    content["personas"]["Priya"]["pragmatics_clauses"] = [
        'Use informal vocabulary and native orthography. Say "speak slowly, please" '
        'if that is your request. Dates use ordinary words, such as "May first". '
        'A phone number may be "five five five, zero one two three". '
        'The @ symbol is "at". Keep tool values unchanged.',
        "Spell only when explicitly asked to spell; use anchors only when explicitly "
        'asked for anchors, for example "D as in dog".',
    ]
    monkeypatch.setattr(
        pipeline, "generate", lambda **kw: SimpleNamespace(content=json.dumps(content))
    )
    staged = draft(draft_request, tmp_path / "valid.json")
    spec = VoiceSpec(
        description="Low pitch, warm accent, slower speech rate and clear enunciation.",
        audition_text="Fixture native audition. " * 10,
    )
    assert staged.pack.personas["Priya"].pragmatics_clauses
    assert spec.description.startswith("Low pitch")


def _published_base_tasks(language, domain, name_order):
    """Rebuild a regional domain's base tasks: canonical English tasks under the
    published IDs, with retail name facts in the pack's name order."""
    from tau2.multilingual.runtime import load_language

    control = load_language("en", domain, mode="text", localize_db=False)
    canonical = {control.metadata[t.id].canonical_task_id: t for t in control.tasks}
    metadata = load_language(language, domain, mode="text", localize_db=False).metadata
    tasks = []
    for task_id, entry in metadata.items():
        task = canonical[entry.canonical_task_id].model_copy(
            update={"id": task_id}, deep=True
        )
        if domain == "retail":
            instructions = task.user_scenario.instructions
            instructions.known_info = instructions.known_info.split(
                "\n\nName details:"
            )[0]
        tasks.append(task)
    return pipeline._base_tasks(SimpleNamespace(domain=domain, tasks=tasks), name_order)


# English binds registry tasks, which the file-publishing factory never stages.
@pytest.mark.slow
@pytest.mark.parametrize("language", ["es", "pt", "hi", "zh", "ko"])
def test_approved_packs_stage_without_changing_content(language, tmp_path, monkeypatch):
    from pathlib import Path

    import tau2.utils
    from tau2.multilingual.factory.pipeline import stage_pack
    from tau2.multilingual.factory.schema import StagePackRequest
    from tau2.multilingual.regional import PublishedRegionalBundle
    from tau2.multilingual.runtime import load_language

    root = Path(__file__).resolve().parents[2] / "data"
    monkeypatch.setattr(tau2.utils, "DATA_DIR", root)
    source = root / "tau2/multilingual" / language / "pack.yaml"
    before = source.read_bytes()
    loaded = [
        load_language(language, d, mode="voice")
        for d in ("airline", "retail", "telecom")
    ]
    pack = loaded[0].pack
    bundles = []
    for value in loaded:
        binding = pack.domains[value.domain]
        bundles.append(
            DomainBundle(
                domain=value.domain,
                revision=binding.revision,
                tasks=(
                    value.tasks
                    if binding.tasks_file
                    else _published_base_tasks(language, value.domain, pack.name_order)
                ),
                metadata=json.loads(
                    (source.parent / binding.task_metadata_file).read_text()
                ),
                regional_bundle=(
                    PublishedRegionalBundle.model_validate_json(
                        (source.parent / binding.regional_bundle_file).read_bytes()
                    )
                    if binding.regional_bundle_file
                    else None
                ),
                reviewed_by="existing approved pack",
                user_guidelines=binding.user_guidelines,
                voice_guidelines=binding.voice_guidelines,
                text_guidelines=binding.text_guidelines,
            )
        )
    staged = stage_pack(
        StagePackRequest(pack=pack, domains=bundles, revision="guard-regression"),
        tmp_path / f"{language}.json",
    )
    assert source.read_bytes() == before
    assert staged.pack.agent_language_clause == pack.agent_language_clause
    for name, persona in pack.personas.items():
        assert staged.pack.personas[name].model_dump(
            exclude={"voice": {"voice_id"}}
        ) == persona.model_dump(exclude={"voice": {"voice_id"}})
    assert sum(len(bundle.tasks) for bundle in staged.domains.values()) == 150
    for value in loaded:
        assert staged.domains[value.domain].regional_bundle == (
            value.scenario
            if isinstance(value.scenario, PublishedRegionalBundle)
            else None
        )


@pytest.mark.slow
def test_staged_regional_bundle_survives_voice_publication(tmp_path, monkeypatch):
    from pydantic import TypeAdapter

    import tau2.utils
    from tau2.data_model.language import TaskMetadata
    from tau2.multilingual.factory.schema import StagePackRequest
    from tau2.multilingual.regional import PublishedRegionalBundle
    from tau2.multilingual.runtime import load_language

    root = Path(__file__).resolve().parents[2] / "data/tau2/multilingual/es"
    loaded = load_language("es", "retail", mode="voice")
    binding = loaded.pack.domains["retail"]
    original = PublishedRegionalBundle.model_validate_json(
        (root / binding.regional_bundle_file).read_bytes()
    )
    bundle = DomainBundle(
        domain="retail",
        revision=binding.revision,
        tasks=_published_base_tasks("es", "retail", loaded.pack.name_order),
        metadata=TypeAdapter(dict[str, TaskMetadata]).validate_json(
            (root / binding.task_metadata_file).read_bytes()
        ),
        reviewed_by="fixture reviewer",
        regional_bundle=original,
        user_guidelines=binding.user_guidelines,
        voice_guidelines=binding.voice_guidelines,
        text_guidelines=binding.text_guidelines,
    )
    pack = loaded.pack.model_copy(update={"domains": {"retail": binding}})
    request = StagePackRequest(pack=pack, domains=[bundle], revision="regional-copy")
    staged = pipeline.stage_pack(request, tmp_path / "staged.json")
    assert staged.domains["retail"].regional_bundle == original
    assert staged.pack.domains["retail"].regional_bundle_file == "regional/retail.json"
    with pytest.raises(ValueError, match="regional"):
        pipeline.stage_pack(
            request.model_copy(
                update={
                    "domains": [bundle.model_copy(update={"regional_bundle": None})]
                }
            ),
            tmp_path / "dropped.json",
        )
    other_language = original.model_copy(deep=True)
    other_language.settings.language = "pt"
    with pytest.raises(ValueError, match="language"):
        pipeline.stage_pack(
            request.model_copy(
                update={
                    "domains": [
                        bundle.model_copy(update={"regional_bundle": other_language})
                    ]
                }
            ),
            tmp_path / "wrong-language.json",
        )
    assets_root = tmp_path / "assets"
    assets = prepare_voices(
        VoiceAssetsRequest(
            draft=staged,
            review=ReviewApproval(
                content_sha256=content_digest(staged), reviewed_by="fixture"
            ),
            specs={
                name: VoiceSpec(
                    description="Fixture voice", audition_text="Fixture audition. " * 10
                )
                for name in staged.pack.personas
            },
            config=VoiceDesignConfig(
                model="fixture", seed=42, loudness=0.5, guidance_scale=38
            ),
        ),
        assets_root,
        FakeVoices(),
    )
    output = tmp_path / "data/tau2/multilingual/es"
    finalize(
        staged,
        assets,
        ReviewApproval(content_sha256=listening_digest(assets), reviewed_by="fixture"),
        output,
        assets_root=assets_root,
    )
    published = PublishedRegionalBundle.model_validate_json(
        (output / "regional/retail.json").read_bytes()
    )
    assert published == original
    assert published.localization_map_sha256 == original.localization_map_sha256
    # Regional bindings publish no base task file; the bundle is the task source.
    assert not (output / "tasks/retail.json").exists()
    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path / "data")
    assert load_language("es", "retail", mode="voice").scenario == original


def test_asset_cli_resolves_audio_beside_request_files(
    draft_request, tmp_path, monkeypatch, capsys
):
    """Portable manifests work from another cwd through every consuming CLI stage."""
    import argparse

    from tau2.multilingual.factory.cli import add_factory_args

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    staged = draft(draft_request, tmp_path / "draft.json")
    request = VoiceAssetsRequest(
        draft=staged,
        review=ReviewApproval(
            content_sha256=content_digest(staged), reviewed_by="fixture"
        ),
        config=VoiceDesignConfig(
            model="fixture-voice", seed=42, loudness=0.5, guidance_scale=38
        ),
        specs={
            name: VoiceSpec(
                description="Fixture voice", audition_text="Fixture audition. " * 10
            )
            for name in staged.pack.personas
        },
    )
    source = tmp_path / "source"
    assets = prepare_voices(request, source, FakeVoices())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    parser = argparse.ArgumentParser()
    add_factory_args(parser)

    def run_argv(argv):
        args = parser.parse_args(argv)
        args.func(args)

    def run(verb, request_file, output):
        argv = [verb, "--request", str(request_file)]
        if output is not None:
            argv += ["--output", str(output)]
        run_argv(argv)

    run("validate-voices", source / "assets.json", None)
    audition = tmp_path / "audition"
    run("audition", source / "assets.json", audition)
    assert (audition / "index.json").exists()
    assert "audio_base64" not in (audition / "index.json").read_text()

    def ref(path):
        value = json.loads(path.read_text())
        return {
            "$artifact": path.relative_to(tmp_path).as_posix(),
            "sha256": json_digest(value),
        }

    capsys.readouterr()
    content_approval = tmp_path / "content-approval.json"
    with pytest.raises(ValueError, match="re-review"):
        run_argv(
            [
                "approve",
                "--gate",
                "content",
                "--artifact",
                str(tmp_path / "draft.json"),
                "--digest",
                "0" * 64,
                "--reviewed-by",
                "fixture",
                "--output",
                str(content_approval),
            ]
        )
    run_argv(
        [
            "approve",
            "--gate",
            "content",
            "--artifact",
            str(tmp_path / "draft.json"),
            "--digest",
            content_digest(staged),
            "--reviewed-by",
            "fixture",
            "--output",
            str(content_approval),
        ]
    )
    listening_approval = tmp_path / "listening-approval.json"
    run_argv(
        [
            "approve",
            "--gate",
            "listening",
            "--artifact",
            str(source / "assets.json"),
            "--digest",
            listening_digest(assets),
            "--reviewed-by",
            "fixture listener",
            "--output",
            str(listening_approval),
        ]
    )
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["gate"] for r in reports] == ["content", "listening"]

    reused_dir = tmp_path / "reused"
    reuse_file = tmp_path / "reuse-request.json"
    reuse_file.write_text(
        json.dumps(
            {
                "previous_draft": ref(tmp_path / "draft.json"),
                "previous_assets": ref(source / "assets.json"),
                "previous_assets_root": "source",
                "draft": ref(tmp_path / "draft.json"),
                "review": ref(content_approval),
            }
        )
    )
    run("reuse-voices", reuse_file, reused_dir)
    report = json.loads(capsys.readouterr().out)
    assert report["output"] == str(reused_dir / "assets.json")
    assert report["listening_sha256"] == listening_digest(assets)
    reused_path = reused_dir / "assets.json"
    run("validate-voices", reused_path, None)
    final_file = tmp_path / "final-request.json"
    final_file.write_text(
        json.dumps(
            {
                "draft": ref(tmp_path / "draft.json"),
                "assets": ref(reused_path),
                "assets_root": "reused",
                # The prior listening approval still binds the unchanged voices.
                "review": ref(listening_approval),
            }
        )
    )
    capsys.readouterr()
    published = tmp_path / "published"
    run("finalize", final_file, published)
    assert (published / "pack.yaml").exists()
    report = json.loads(capsys.readouterr().out)
    assert report["output"] == str(published)
    assert (
        report["pack_sha256"]
        == hashlib.sha256((published / "pack.yaml").read_bytes()).hexdigest()
    )

    # A reference to an edited artifact fails before any stage runs.
    tampered = json.loads(final_file.read_text())
    tampered["assets"]["sha256"] = "0" * 64
    final_file.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="does not match its referenced sha256"):
        run("finalize", final_file, tmp_path / "tampered-pack")
    final_file.write_text(json.dumps({**tampered, "assets": ref(reused_path)}))

    # Valid review metadata cannot hide a missing underlying recording.
    next((reused_path.parent / "audio").iterdir()).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        run("finalize", final_file, tmp_path / "missing-audio-pack")
    assert not (tmp_path / "missing-audio-pack").exists()

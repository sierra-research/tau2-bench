"""Offline rendering must use real runtime prompts and explicit task assignments."""

import argparse
import json
import shutil
from pathlib import Path

import pytest
import yaml

from tau2.multilingual.cli import add_languages_args
from tau2.multilingual.prompt_review import render_review, write_review


@pytest.fixture
def review_pack(tmp_path, monkeypatch, english_prompt_content):
    import tau2.utils

    source_root = tau2.utils.DATA_DIR
    shutil.copytree(
        source_root / "tau2/domains/retail", tmp_path / "tau2/domains/retail"
    )
    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    root = tmp_path / "tau2/multilingual/en"
    root.mkdir(parents=True)
    tasks = [
        {
            "id": name,
            "user_scenario": {"instructions": f"Your name is {name}. <unsafe>"},
        }
        for name in ("Priya", "Mamadou")
    ]
    (root / "tasks.json").write_text(json.dumps(tasks))
    (root / "metadata.json").write_text(
        json.dumps(
            {
                name: {
                    "canonical_task_id": f"canonical-{name}",
                    "caller_gender": gender,
                }
                for name, gender in (("Priya", "female"), ("Mamadou", "male"))
            }
        )
    )
    personas = {
        name: {
            "gender_guidance": f"Your gender is {gender}.",
            "gender": gender,
            "locale": "English",
            "variety": "English",
            "pragmatics_clauses": ["Use natural phrasing."],
            "voice": {
                "voice_id": f"voice-{name}",
                "voice_examples": english_prompt_content["prompt_content"][
                    "voice_examples"
                ],
                "backchannel_phrases": ["Okay"],
            },
        }
        for name, gender in (("Priya", "female"), ("Mamadou", "male"))
    }
    (root / "pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 3,
                "language": "en",
                "revision": "review-test",
                "display_name": "English",
                "agent_greeting": "Hello.",
                "agent_language_clause": "Speak English.",
                "voice": {
                    "spellout_section": english_prompt_content["spellout_section"],
                    "speech_insert_events_per_minute": 1,
                    "backchannel_level": "medium",
                },
                "personas": personas,
                "domains": {
                    "retail": {
                        "revision": "test",
                        "tasks_file": "tasks.json",
                        "task_metadata_file": "metadata.json",
                    }
                },
            }
        )
    )
    return root


def test_actual_runtime_render_and_safe_export(
    review_pack, tmp_path, monkeypatch, voice_extra
):
    from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
    from tau2.multilingual.runtime import load_language
    from tau2.runner.build import build_voice_orchestrator

    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering must never create a provider adapter")

    monkeypatch.setattr(DiscreteTimeAudioNativeAgent, "adapter", property(forbidden))
    monkeypatch.setattr("tau2.user.user_simulator_streaming.generate", forbidden)
    monkeypatch.setattr("tau2.user.user_simulator.generate", forbidden)
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig

    template = VoiceRunConfig(
        domain="retail",
        llm_user="openai/gpt-5.5",
        llm_args_user={"reasoning_effort": "low"},
        audio_native_config=AudioNativeConfig(voice="marin", max_steps_seconds=2400),
    )
    report = render_review(languages=["en"], domains=["retail"], template=template)
    assert [p.persona_id for p in report.pairs] == ["Mamadou", "Priya"]
    assert all(
        p.agent_gender == "female" and p.agent_voice == "marin" for p in report.pairs
    )
    assert all(
        p.config.llm_user == "openai/gpt-5.5"
        and p.config.llm_args_user == {"reasoning_effort": "low"}
        and p.config.audio_native_config.max_steps_seconds == 2400
        for p in report.pairs
    )
    assert all(
        p.voice_instruction_prompt_version == "voice-instructions-v2"
        for p in report.pairs
    )
    assert {p.canonical_task_id for p in report.pairs} == {
        "canonical-Mamadou",
        "canonical-Priya",
    }
    loaded = load_language("en", "retail", mode="voice")
    for pair in report.pairs:
        task = next(t for t in loaded.tasks if t.id == pair.task_id)
        built = build_voice_orchestrator(pair.config, task)
        try:
            assert pair.agent_prompt == built.agent.system_prompt
            assert pair.customer_prompt == built.user.system_prompt
            assert (
                pair.language_context.caller_gender
                == pair.language_context.persona.gender
            )
        finally:
            if built.user._user_executor is not None:
                built.user._user_executor.shutdown(wait=True)
    target = tmp_path / "review"
    write_review(report, target)
    html = (target / "index.html").read_text()
    assert "&lt;unsafe&gt;" in html and "<unsafe>" not in html
    assert "Mamadou" in html and "Priya" in html
    saved = type(report).model_validate_json((target / "review.json").read_text())
    assert saved == report
    assert len(list((target / "prompts").glob("*.txt"))) == 4
    assert (target / "schema.json").exists()
    before = (target / "review.json").read_bytes()
    write_review(report, target)
    assert (target / "review.json").read_bytes() == before


def test_cli_filters_and_invalid_selection(review_pack):
    parser = argparse.ArgumentParser()
    add_languages_args(parser)
    args = parser.parse_args(
        [
            "render-prompts",
            "--output",
            "/tmp/review",
            "--language",
            "en",
            "--domain",
            "retail",
        ]
    )
    assert args.language == ["en"] and args.domain == ["retail"]
    with pytest.raises(ValueError, match="domain"):
        render_review(languages=["en"], domains=["missing"])


@pytest.mark.parametrize(
    "provider,voice,expected_voice,gender",
    [("gemini", None, "Zephyr", "female"), ("openai", "echo", "echo", "male")],
)
def test_renderer_uses_ordinary_provider_configuration_without_mutation(
    review_pack, provider, voice, expected_voice, gender
):
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig

    template = VoiceRunConfig(
        domain="airline",
        seed=19,
        llm_user="fixture-customer",
        audio_native_config=AudioNativeConfig(
            provider=provider,
            voice=voice,
            max_steps_seconds=37,
        ),
    )
    before = template.model_dump_json()
    result = render_review(languages=["en"], domains=["retail"], template=template)
    assert template.model_dump_json() == before
    for pair in result.pairs:
        assert pair.config.seed == 19
        assert pair.config.llm_user == "fixture-customer"
        assert pair.config.audio_native_config.provider == provider
        assert pair.config.audio_native_config.max_steps_seconds == 37
        assert pair.agent_voice == expected_voice and pair.agent_gender == gender
        assert f"Your gender is {gender}." in pair.agent_prompt


def test_pack_header_loader_rejects_wrong_identity_and_escaped_file(
    review_pack, tmp_path
):
    from tau2.multilingual.runtime import load_pack

    with pytest.raises(ValueError, match="identifier"):
        load_pack("../en")
    path = review_pack / "pack.yaml"
    original = path.read_text()
    changed = yaml.safe_load(original)
    changed["language"] = "es"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="language"):
        load_pack("en")
    path.unlink()
    outside = tmp_path / "outside.yaml"
    outside.write_text(original)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        load_pack("en")


def test_standalone_render_without_config_uses_ordinary_voice_defaults(review_pack):
    from tau2.config import DEFAULT_LLM_USER, DEFAULT_MAX_STEPS_SECONDS

    review = render_review(languages=["en"], domains=["retail"])
    assert len(review.pairs) == 2
    for pair in review.pairs:
        assert pair.config.llm_user == DEFAULT_LLM_USER
        assert (
            pair.config.audio_native_config.max_steps_seconds
            == DEFAULT_MAX_STEPS_SECONDS
        )
        assert pair.config.benchmark is None


def test_benchmark_review_uses_shared_settings_and_frozen_tasks():
    from pathlib import Path

    from tau2.runner.presets import BenchmarkRequest, resolve_benchmark

    plan = resolve_benchmark(
        BenchmarkRequest(output=Path("unused"), languages=["en"], domains=["retail"])
    )
    review = render_review(benchmark="voice", languages=["en"], domains=["retail"])
    assert len(review.pairs) == 2
    config = plan.configs[0]
    for pair in review.pairs:
        assert pair.task_id in config.task_ids
        assert pair.config.llm_user == config.llm_user
        assert pair.config.llm_args_user == config.llm_args_user
        assert pair.config.audio_native_config == config.audio_native_config
        assert pair.config.communicate_scoring == config.communicate_scoring
        assert pair.config.benchmark.selected_task_ids == [pair.canonical_task_id]


def test_review_never_constructs_execution_objects(
    review_pack, monkeypatch, voice_extra
):
    import concurrent.futures

    import tau2.runner.build as build
    from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
    from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

    def forbidden(*args, **kwargs):
        raise AssertionError("Prompt review must not construct execution objects")

    monkeypatch.setattr(build, "build_voice_orchestrator", forbidden)
    monkeypatch.setattr(DiscreteTimeAudioNativeAgent, "__init__", forbidden)
    monkeypatch.setattr(VoiceStreamingUserSimulator, "__init__", forbidden)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", forbidden)
    assert len(render_review(languages=["en"], domains=["retail"]).pairs) == 2


BASELINE_CASES = ("default", "zero", "openai", "livekit")
BASELINE = Path(__file__).parent / "fixtures/prompt_review_baseline.json"


def baseline_kwargs(case: str) -> dict:
    """The render_review arguments each pinned English baseline case uses."""
    from tau2.data_model.run_config import AudioNativeConfig
    from tau2.data_model.simulation import VoiceRunConfig

    if case == "zero":
        return {"seed": 0}
    if case == "openai":
        return {
            "template": VoiceRunConfig(
                seed=17,
                llm_user="fixture-customer",
                llm_args_user={"temperature": 0.2},
                audio_native_config=AudioNativeConfig(
                    provider="openai",
                    voice="echo",
                    use_xml_prompt=True,
                    max_steps_seconds=73,
                ),
            )
        }
    if case == "livekit":
        return {
            "template": VoiceRunConfig(
                seed=0,
                audio_native_config=AudioNativeConfig(
                    provider="livekit", max_steps_seconds=73
                ),
            )
        }
    return {}


def baseline_digests(case: str) -> dict[str, str]:
    """Digest every English pair's complete prompts, config and language context.

    Regenerate the fixture after an intended prompt change with
    ``{case: baseline_digests(case) for case in BASELINE_CASES}``.
    """
    import hashlib

    pairs = render_review(languages=["en"], **baseline_kwargs(case)).pairs
    return {
        f"{pair.language}/{pair.domain}/{pair.persona_id}": hashlib.sha256(
            json.dumps(
                pair.model_dump(mode="json"),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for pair in pairs
    }


@pytest.mark.parametrize("case", BASELINE_CASES)
def test_english_prompt_pairs_match_reviewed_baseline(case, monkeypatch):
    """Reviewed English prompts and settings remain exact."""
    from tau2.config import SPELLING_NORMALIZATION_VERSION
    from tau2.data_model.language import LANGUAGE_PROMPT_VERSION, LanguageArtifactPins
    from tau2.data_model.semantic_communication import CommunicateScoring
    from tau2.runner import build

    def forbidden(*args, **kwargs):
        pytest.fail("Review must not construct an orchestrator")

    monkeypatch.setattr(build, "build_voice_orchestrator", forbidden)
    for pair in render_review(languages=["en"], **baseline_kwargs(case)).pairs:
        assert pair.config.communicate_scoring == CommunicateScoring()
        assert pair.language_context.prompt_version == LANGUAGE_PROMPT_VERSION
        assert pair.config.localize_db is None
        pins = pair.config.language_artifacts
        assert pins.scenario_mode == "pack"
        for field in set(LanguageArtifactPins.model_fields) - {
            "pack_sha256",
            "tasks_sha256",
            "metadata_sha256",
        }:
            assert getattr(pair.language_context, field) == getattr(pins, field)
        assert pair.language_context.spelling_normalization_version == (
            SPELLING_NORMALIZATION_VERSION if pair.language == "pt" else None
        )
    actual = baseline_digests(case)
    assert len(actual) == 6
    assert actual == json.loads(BASELINE.read_text())[case]


@pytest.mark.slow
def test_all_released_pairs_equal_ordinary_construction(voice_extra):
    from tau2.multilingual.runtime import load_language
    from tau2.runner.build import build_voice_orchestrator

    report = render_review(seed=0)
    assert len(report.pairs) == 36
    for language in {p.language for p in report.pairs}:
        for domain in {p.domain for p in report.pairs}:
            tasks = {
                t.id: t for t in load_language(language, domain, mode="voice").tasks
            }
            for pair in report.pairs:
                if pair.language != language or pair.domain != domain:
                    continue
                built = build_voice_orchestrator(
                    pair.config,
                    tasks[pair.task_id],
                    language_context=pair.language_context,
                )
                try:
                    assert built.agent.system_prompt == pair.agent_prompt
                    assert built.user.system_prompt == pair.customer_prompt
                    assert (
                        built.initial_agent_message.content
                        == pair.initial_agent_message
                    )
                    assert built.call_context.language == pair.language_context
                finally:
                    if built.user._user_executor is not None:
                        built.user._user_executor.shutdown(wait=True)

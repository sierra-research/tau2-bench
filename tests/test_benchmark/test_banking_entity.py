"""Released banking and entity voice frames bind their multilingual extensions."""

import pytest

from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.runner.presets import BenchmarkRequest, load_frame, resolve_benchmark

LANGUAGES = ["en", "es", "pt", "hi", "zh", "ko"]


@pytest.mark.parametrize(
    ("preset", "version", "domain", "size"),
    [
        ("banking", "voice-v2-banking", "banking_knowledge", 97),
        ("entity", "voice-v2-entity", "intake", 200),
    ],
)
def test_frame_covers_every_language(preset, version, domain, size):
    frame = load_frame(preset)
    assert frame.benchmark_version == version
    assert frame.mode == "voice"
    assert [cell.language for cell in frame.cells] == LANGUAGES
    assert {cell.domain for cell in frame.cells} == {domain}
    assert {len(cell.canonical_task_ids) for cell in frame.cells} == {size}


def test_banking_cells_pin_extension_and_alltools(tmp_path):
    plan = resolve_benchmark(
        BenchmarkRequest(preset="banking", output=tmp_path / "banking")
    )
    assert not plan.partial
    by_language = {config.language: config for config in plan.configs}
    assert by_language["en"].pack_extension is None
    assert {by_language[lang].pack_extension for lang in LANGUAGES[1:]} == {
        "banking-dates-v1"
    }
    for config in plan.configs:
        config.validate()
        assert config.retrieval_config == "alltools"
        assert config.llm_user == "openai/gpt-5.5"
        assert config.audio_native_config.max_steps_seconds == 2400
    assert not (tmp_path / "banking").exists()


def test_entity_cells_keep_agent_directed_call_limit(tmp_path):
    plan = resolve_benchmark(
        BenchmarkRequest(
            preset="entity",
            output=tmp_path / "entity",
            languages=["es"],
            template=VoiceRunConfig(
                audio_native_config=AudioNativeConfig(), num_tasks=1
            ),
        )
    )
    assert plan.partial
    (config,) = plan.configs
    assert config.pack_extension == "intake"
    assert config.elicitation.strategy.value == "agent_directed"
    assert config.audio_native_config.max_steps_seconds == 240
    assert (
        config.benchmark.selected_task_ids == plan.frame.cells[1].canonical_task_ids[:1]
    )


def test_released_benchmarks_reject_template_extension(tmp_path):
    with pytest.raises(ValueError, match="select their own pack extensions"):
        resolve_benchmark(
            BenchmarkRequest(
                preset="banking",
                output=tmp_path / "banking",
                template=VoiceRunConfig(
                    audio_native_config=AudioNativeConfig(), pack_extension="banking"
                ),
            )
        )


@pytest.mark.parametrize("preset", ["banking", "entity"])
def test_scoring_authenticates_extension_cells(preset):
    from tau2.metrics.benchmark import _authoritative_language

    for cell in load_frame(preset).cells:
        _authoritative_language(
            cell.language, cell.domain, cell.binding_revision, "voice"
        )


@pytest.mark.parametrize(
    "overrides",
    [{"retrieval_config": "bm25"}, {"retrieval_config_kwargs": {"top_k": 3}}],
)
def test_banking_rejects_custom_retrieval(tmp_path, overrides):
    with pytest.raises(ValueError, match="retrieval config"):
        resolve_benchmark(
            BenchmarkRequest(
                preset="banking",
                output=tmp_path / "banking",
                template=VoiceRunConfig(
                    audio_native_config=AudioNativeConfig(), **overrides
                ),
            )
        )

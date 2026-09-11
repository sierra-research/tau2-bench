"""Reasoning-effort provenance: the value a voice run records must be the one
the provider was actually run with, and an inferred value must never pass for
an observed one.

Covers the config-time resolution (tau2.config + AudioNativeConfig) and what
lands in ``results.info``. Adapter-side wiring lives in
``tests/test_streaming/test_reasoning_effort_adapter.py`` (voice tier).
"""

import json

import pytest
from pydantic import ValidationError

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_REASONING_EFFORT,
    SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS,
    ReasoningEffort,
    resolve_audio_native_reasoning_effort,
)
from tau2.data_model.simulation import (
    STORED_PAYLOAD_CONTEXT,
    AudioNativeConfig,
    ReasoningEffortSource,
    Results,
    VoiceRunConfig,
)
from tau2.runner.helpers import get_info

# ---------------------------------------------------------------------------
# The default table and the resolver
# ---------------------------------------------------------------------------


def test_every_audio_native_provider_has_a_default_effort():
    """Coverage guard: a provider without an entry would resolve to a crash."""
    assert set(DEFAULT_AUDIO_NATIVE_REASONING_EFFORT) == set(
        DEFAULT_AUDIO_NATIVE_MODELS
    )


def test_defaults_are_enum_members_not_none():
    """`None` in the table is what made every run record null."""
    for provider, effort in DEFAULT_AUDIO_NATIVE_REASONING_EFFORT.items():
        assert isinstance(effort, ReasoningEffort), provider


def test_resolver_falls_back_to_the_provider_default():
    assert resolve_audio_native_reasoning_effort("gemini", None) is ReasoningEffort.HIGH
    assert (
        resolve_audio_native_reasoning_effort("openai", None)
        is ReasoningEffort.PROVIDER_DEFAULT
    )


def test_resolver_is_idempotent():
    for provider in DEFAULT_AUDIO_NATIVE_REASONING_EFFORT:
        once = resolve_audio_native_reasoning_effort(provider, None)
        assert resolve_audio_native_reasoning_effort(provider, once) is once


def test_resolver_keeps_an_explicit_pin():
    assert resolve_audio_native_reasoning_effort("gemini", "low") is ReasoningEffort.LOW
    assert (
        resolve_audio_native_reasoning_effort(
            "gemini", ReasoningEffort.PROVIDER_DEFAULT
        )
        is ReasoningEffort.PROVIDER_DEFAULT
    )


def test_resolver_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="Unknown audio-native provider"):
        resolve_audio_native_reasoning_effort("not-a-provider", None)


# ---------------------------------------------------------------------------
# Per-provider support: the levels are not a shared vocabulary
# ---------------------------------------------------------------------------


def test_every_provider_declares_what_it_supports():
    """Coverage guard: a missing entry would KeyError inside the resolver."""
    assert set(SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS) == set(
        DEFAULT_AUDIO_NATIVE_MODELS
    )


def test_a_providers_own_default_is_always_supported():
    for provider, effort in DEFAULT_AUDIO_NATIVE_REASONING_EFFORT.items():
        assert effort in SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS[provider], provider


def test_xhigh_is_openai_only():
    """Gemini Live rejects XHIGH at connect ('Invalid value at
    setup.generation_config.thinking_config.thinking_level'), and the
    google-genai SDK builds the enum anyway behind a UserWarning — so the check
    has to be ours, and it has to happen before the websocket opens."""
    assert ReasoningEffort.XHIGH in SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS["openai"]
    assert (
        ReasoningEffort.XHIGH not in SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS["gemini"]
    )
    with pytest.raises(ValueError, match="does not support reasoning effort 'xhigh'"):
        resolve_audio_native_reasoning_effort("gemini", "xhigh")


@pytest.mark.parametrize("provider", ["xai", "nova", "qwen", "livekit"])
def test_providers_without_the_knob_reject_every_level(provider):
    """These raise at adapter construction (or ignore the knob entirely, for
    livekit); the config must refuse the pin first."""
    assert SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS[provider] == {
        ReasoningEffort.PROVIDER_DEFAULT
    }
    with pytest.raises(ValueError, match="does not support reasoning effort"):
        resolve_audio_native_reasoning_effort(provider, "high")


def test_gemini_still_takes_the_four_levels_it_supports():
    for level in ("minimal", "low", "medium", "high"):
        assert resolve_audio_native_reasoning_effort("gemini", level) is (
            ReasoningEffort(level)
        )


def test_the_config_refuses_an_unsupported_pair():
    """Not only the CLI seam: any construction path fails at config time."""
    with pytest.raises(ValidationError, match="does not support reasoning effort"):
        AudioNativeConfig(provider="gemini", reasoning_effort="xhigh")


# ---------------------------------------------------------------------------
# AudioNativeConfig resolves before it is recorded
# ---------------------------------------------------------------------------


def test_config_resolves_at_construction():
    assert AudioNativeConfig().reasoning_effort is ReasoningEffort.PROVIDER_DEFAULT
    assert AudioNativeConfig(provider="gemini").reasoning_effort is ReasoningEffort.HIGH


def test_config_never_serializes_a_null_effort():
    for provider in DEFAULT_AUDIO_NATIVE_REASONING_EFFORT:
        kwargs = {"provider": provider}
        if provider == "openai_live":
            kwargs["live_config"] = {"backend_model": "gpt-5.4-mini"}
        dumped = json.loads(AudioNativeConfig(**kwargs).model_dump_json())
        assert dumped["reasoning_effort"] is not None
        assert dumped["reasoning_effort_source"] == "run"
        assert dumped["reasoning_effort_backfill"] is None


def test_config_keeps_an_explicit_pin():
    config = AudioNativeConfig(provider="gemini", reasoning_effort="low")
    assert config.reasoning_effort is ReasoningEffort.LOW
    assert config.reasoning_effort_source is ReasoningEffortSource.RUN


def test_config_rejects_a_bogus_level():
    with pytest.raises(ValueError):
        AudioNativeConfig(reasoning_effort="turbo")


def test_stored_null_loads_as_inferred():
    """A pre-resolution payload: resolve it so readers see the effective value,
    but never let it pass as something the run observed."""
    config = AudioNativeConfig.model_validate(
        {
            "provider": "gemini",
            "model": "gemini-3.1-flash-live-preview",
            "reasoning_effort": None,
        },
        context=STORED_PAYLOAD_CONTEXT,
    )
    assert config.reasoning_effort is ReasoningEffort.HIGH
    assert config.reasoning_effort_source is ReasoningEffortSource.INFERRED


def test_a_legacy_null_on_disk_loads_as_inferred(tmp_path):
    """The real path the `inferred` marking exists for: a results.json written
    before eager resolution, revived through ``Results.load``. The loaders are
    what set STORED_PAYLOAD_CONTEXT, so this is the test that fails if one of
    them ever stops passing it."""
    results = Results(
        info=get_info(
            VoiceRunConfig(
                domain="mock", audio_native_config=AudioNativeConfig(provider="gemini")
            )
        ),
        tasks=[],
        simulations=[],
    )
    out = tmp_path / "results.json"
    results.save(out, format="json")
    payload = json.loads(out.read_text())
    # Age the file back to what a pre-resolution run wrote.
    payload["info"]["audio_native_config"]["reasoning_effort"] = None
    del payload["info"]["audio_native_config"]["reasoning_effort_source"]
    out.write_text(json.dumps(payload))

    loaded = Results.load(out)
    assert loaded.info.audio_native_config.reasoning_effort is ReasoningEffort.HIGH
    assert (
        loaded.info.audio_native_config.reasoning_effort_source
        is ReasoningEffortSource.INFERRED
    )


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_a_keyword_none_is_an_unpinned_run_not_an_inference(provider):
    """B3: `reasoning_effort=None` on a LIVE construction means "the CLI pinned
    nothing" — the natural shape when threading an unset flag through — and
    must resolve exactly like an omitted kwarg. Recording it as `inferred`
    would claim the run never observed a value it in fact ran at, which is the
    precise confusion this field was added to prevent.
    """
    threaded = AudioNativeConfig(provider=provider, reasoning_effort=None)
    omitted = AudioNativeConfig(provider=provider)
    assert threaded.reasoning_effort is omitted.reasoning_effort
    assert threaded.reasoning_effort_source is ReasoningEffortSource.RUN
    assert threaded.reasoning_effort_backfill is None


def test_a_null_without_the_stored_context_is_not_a_legacy_null():
    """The discriminator is the context, not the shape of the dict: a hand-built
    payload validated in memory is a construction, not a revived result."""
    config = AudioNativeConfig.model_validate(
        {"provider": "gemini", "reasoning_effort": None}
    )
    assert config.reasoning_effort_source is ReasoningEffortSource.RUN


def test_round_trip_preserves_the_source():
    for config in (
        AudioNativeConfig(provider="gemini"),
        AudioNativeConfig.model_validate(
            {"provider": "openai", "reasoning_effort": None},
            context=STORED_PAYLOAD_CONTEXT,
        ),
    ):
        again = AudioNativeConfig.model_validate(
            config.model_dump(mode="json"), context=STORED_PAYLOAD_CONTEXT
        )
        assert again.reasoning_effort is config.reasoning_effort
        assert again.reasoning_effort_source is config.reasoning_effort_source


# ---------------------------------------------------------------------------
# ... and lands in results.info
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,expected",
    [("openai", "provider_default"), ("gemini", "high")],
)
def test_run_info_records_the_effective_value(tmp_path, provider, expected):
    """The config -> RunConfig -> Info -> results.json path: what a fresh run
    writes must be the effort the provider was actually given."""
    config = VoiceRunConfig(
        domain="mock",
        audio_native_config=AudioNativeConfig(provider=provider),
    )
    info = get_info(config)
    assert info.audio_native_config.reasoning_effort.value == expected

    results = Results(info=info, tasks=[], simulations=[])
    out = tmp_path / "results.json"
    results.save(out, format="json")
    stored = json.loads(out.read_text())["info"]["audio_native_config"]
    assert stored["reasoning_effort"] == expected
    assert stored["reasoning_effort_source"] == "run"

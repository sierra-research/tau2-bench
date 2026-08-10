"""Unit tests for voice usage records, aggregation, and pricing.

No network access: provider payloads are synthetic fixtures matching each
provider's wire format.
"""

import pytest

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.usage import UsageRecord, aggregate_usage
from tau2.utils.llm_utils import get_cost
from tau2.voice.audio_native.nova.events import NovaUsageEvent, parse_nova_event
from tau2.voice.pricing import (
    PRICING_VERSION,
    build_session_usage,
    compute_record_cost,
    compute_tick_cost,
    lookup_rates,
)

# =============================================================================
# Parsing: provider payloads -> UsageRecord
# =============================================================================


class TestOpenAIStyleParsing:
    """OpenAI realtime usage dicts (also Qwen / xAI wire format)."""

    OPENAI_USAGE = {
        "total_tokens": 1500,
        "input_tokens": 1000,
        "output_tokens": 500,
        "input_token_details": {
            "text_tokens": 300,
            "audio_tokens": 700,
            "cached_tokens": 200,
            "cached_tokens_details": {"text_tokens": 150, "audio_tokens": 50},
        },
        "output_token_details": {"text_tokens": 100, "audio_tokens": 400},
    }

    def test_ga_payload(self):
        record = UsageRecord.from_openai_realtime_usage(
            self.OPENAI_USAGE,
            provider="openai",
            model="gpt-realtime-1.5",
            scope_id="resp_1",
        )
        assert record.provider == "openai"
        assert record.component == "realtime"
        assert record.semantics == "delta"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.input_text_tokens == 300
        assert record.input_audio_tokens == 700
        assert record.input_cached_tokens == 200
        assert record.input_cached_text_tokens == 150
        assert record.input_cached_audio_tokens == 50
        assert record.output_text_tokens == 100
        assert record.output_audio_tokens == 400
        assert record.raw == self.OPENAI_USAGE

    def test_plural_details_spelling(self):
        """DashScope/older APIs use input_tokens_details (plural)."""
        usage = {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"text_tokens": 40, "audio_tokens": 60},
            "output_tokens_details": {"text_tokens": 5, "audio_tokens": 15},
        }
        record = UsageRecord.from_openai_realtime_usage(
            usage, provider="qwen", model="qwen3-omni-flash-realtime"
        )
        assert record.input_text_tokens == 40
        assert record.input_audio_tokens == 60
        assert record.output_audio_tokens == 15

    def test_missing_details(self):
        record = UsageRecord.from_openai_realtime_usage(
            {"input_tokens": 10, "output_tokens": 5},
            provider="xai",
            model="xai-realtime",
        )
        assert record.input_tokens == 10
        assert record.input_text_tokens is None


class TestGeminiParsing:
    def test_modality_details(self):
        record = UsageRecord.from_gemini_usage_metadata(
            model="gemini-3.1-flash-live-preview",
            prompt_token_count=1000,
            response_token_count=200,
            thoughts_token_count=30,
            prompt_tokens_details=[
                {"modality": "TEXT", "token_count": 400},
                {"modality": "AUDIO", "token_count": 600},
            ],
            response_tokens_details=[
                {"modality": "TEXT", "token_count": 50},
                {"modality": "AUDIO", "token_count": 150},
            ],
        )
        assert record.provider == "gemini"
        assert record.input_text_tokens == 400
        assert record.input_audio_tokens == 600
        # thoughts are billed as text output
        assert record.output_text_tokens == 50 + 30
        assert record.output_audio_tokens == 150
        assert record.output_tokens == 230

    def test_enum_repr_modality_normalized(self):
        record = UsageRecord.from_gemini_usage_metadata(
            model="m",
            prompt_tokens_details=[
                {"modality": "MediaModality.AUDIO", "token_count": 10}
            ],
        )
        assert record.input_audio_tokens == 10


class TestNovaParsing:
    WIRE_EVENT = {
        "event": {
            "usageEvent": {
                "completionId": "comp-1",
                "sessionId": "sess-1",
                "promptName": "p",
                "totalInputTokens": 500,
                "totalOutputTokens": 300,
                "totalTokens": 800,
                "details": {
                    "delta": {"input": {"speechTokens": 5, "textTokens": 1}},
                    "total": {
                        "input": {"speechTokens": 450, "textTokens": 50},
                        "output": {"speechTokens": 250, "textTokens": 50},
                    },
                },
            }
        }
    }

    def test_camelcase_wire_parsing(self):
        """Regression: camelCase fields were silently dropped (extra=ignore)."""
        event = parse_nova_event(self.WIRE_EVENT)
        assert isinstance(event, NovaUsageEvent)
        assert event.completion_id == "comp-1"
        assert event.total_input_tokens == 500
        assert event.total_output_tokens == 300
        assert event.details["total"]["input"]["speechTokens"] == 450

    def test_usage_record_is_cumulative(self):
        event = parse_nova_event(self.WIRE_EVENT)
        record = UsageRecord.from_nova_usage_event(
            model="amazon.nova-2-sonic-v1:0",
            completion_id=event.completion_id,
            total_input_tokens=event.total_input_tokens,
            total_output_tokens=event.total_output_tokens,
            details=event.details,
        )
        assert record.semantics == "cumulative"
        assert record.scope_id == "comp-1"
        assert record.input_audio_tokens == 450
        assert record.input_text_tokens == 50
        assert record.output_audio_tokens == 250


# =============================================================================
# Aggregation
# =============================================================================


class TestAggregation:
    def _delta(self, n, **kw):
        defaults = dict(provider="openai", model="m", component="realtime")
        defaults.update(kw)
        return UsageRecord(input_tokens=n, semantics="delta", **defaults)

    def _cumulative(self, n, scope, **kw):
        defaults = dict(provider="nova", model="m", component="realtime")
        defaults.update(kw)
        return UsageRecord(
            input_tokens=n, semantics="cumulative", scope_id=scope, **defaults
        )

    def test_delta_records_sum(self):
        result = aggregate_usage([self._delta(10), self._delta(20)])
        assert len(result) == 1
        assert result[0].input_tokens == 30

    def test_cumulative_last_wins_per_scope(self):
        records = [
            self._cumulative(100, "c1"),
            self._cumulative(250, "c1"),  # running total, supersedes 100
            self._cumulative(40, "c2"),
        ]
        result = aggregate_usage(records)
        assert len(result) == 1
        assert result[0].input_tokens == 250 + 40
        assert result[0].semantics == "delta"

    def test_groups_by_provider_model_component(self):
        records = [
            self._delta(10),
            self._delta(5, component="llm"),
            self._delta(7, provider="deepgram", component="stt"),
        ]
        result = aggregate_usage(records)
        assert len(result) == 3

    def test_billable_grouped_separately(self):
        records = [
            self._delta(10, provider="xai", billable=False),
            UsageRecord(
                provider="xai",
                model="m",
                component="realtime",
                semantics="cumulative",
                scope_id="s1",
                audio_input_seconds=60.0,
            ),
        ]
        result = aggregate_usage(records)
        assert len(result) == 2
        billables = {r.billable for r in result}
        assert billables == {True, False}

    def test_none_meters_stay_none(self):
        result = aggregate_usage([self._delta(10)])
        assert result[0].characters is None
        assert result[0].audio_input_seconds is None


# =============================================================================
# Pricing
# =============================================================================


class TestPricing:
    def test_lookup_exact_and_prefix(self):
        assert lookup_rates("openai", "gpt-realtime-1.5") is not None
        # dated snapshot resolves via longest prefix
        r = lookup_rates("openai", "gpt-realtime-1.5-2026-03-01")
        assert r is not None and r.input_audio == 32.00
        assert (
            lookup_rates("deepgram", "aura-asteria-en").per_million_characters == 15.0
        )
        assert (
            lookup_rates("deepgram", "aura-2-thalia-en").per_million_characters == 30.0
        )
        assert lookup_rates("openai", "no-such-model") is None

    def test_openai_realtime_cost_math(self):
        record = UsageRecord(
            provider="openai",
            model="gpt-realtime-1.5",
            component="realtime",
            input_text_tokens=1_000_000,
            input_audio_tokens=1_000_000,
            input_cached_text_tokens=500_000,
            input_cached_audio_tokens=250_000,
            output_text_tokens=100_000,
            output_audio_tokens=200_000,
        )
        # text: 0.5M * $4 + 0.5M * $0.4 = 2.0 + 0.2 = 2.2
        # audio: 0.75M * $32 + 0.25M * $0.4 = 24.0 + 0.1 = 24.1
        # out text: 0.1M * $16 = 1.6 ; out audio: 0.2M * $64 = 12.8
        assert compute_record_cost(record) == pytest.approx(2.2 + 24.1 + 1.6 + 12.8)

    def test_cached_without_cached_rate_charged_full(self):
        """Gemini has no published cached discount -> full input rate."""
        record = UsageRecord(
            provider="gemini",
            model="gemini-3.1-flash-live-preview",
            component="realtime",
            input_text_tokens=1_000_000,
            input_cached_tokens=400_000,  # unattributed cache
        )
        assert compute_record_cost(record) == pytest.approx(0.75)

    def test_unknown_model_is_none_not_zero(self):
        record = UsageRecord(
            provider="mystery", model="m", component="realtime", input_tokens=100
        )
        assert compute_record_cost(record) is None

    def test_non_billable_costs_zero(self):
        record = UsageRecord(
            provider="xai",
            model="xai-realtime",
            component="realtime",
            input_tokens=1_000_000,
            billable=False,
        )
        assert compute_record_cost(record) == 0.0

    def test_per_minute_meter(self):
        record = UsageRecord(
            provider="xai",
            model="xai-realtime",
            component="realtime",
            semantics="cumulative",
            scope_id="s",
            audio_input_seconds=120.0,
        )
        assert compute_record_cost(record) == pytest.approx(2 * 0.05)

    def test_stt_and_tts_meters(self):
        stt = UsageRecord(
            provider="deepgram",
            model="nova-3",
            component="stt",
            audio_input_seconds=600.0,
        )
        tts = UsageRecord(
            provider="deepgram",
            model="aura-asteria-en",
            component="tts",
            characters=10_000,
        )
        assert compute_record_cost(stt) == pytest.approx(10 * 0.0048)
        assert compute_record_cost(tts) == pytest.approx(0.15)

    def test_llm_leg_table_pricing(self):
        record = UsageRecord(
            provider="openai",
            model="gpt-4.1",
            component="llm",
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        assert compute_record_cost(record) == pytest.approx(2.0 + 4.0)

    def test_llm_litellm_fallback(self):
        record = UsageRecord(
            provider="openai",
            model="gpt-4o-mini",  # not in our table; litellm knows it
            component="llm",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        cost = compute_record_cost(record)
        assert cost is not None and cost > 0


class TestTickCost:
    def test_sums_delta_only(self):
        delta = UsageRecord(
            provider="openai",
            model="gpt-realtime-1.5",
            component="realtime",
            output_audio_tokens=1_000_000,
        )
        cumulative = UsageRecord(
            provider="nova",
            model="amazon.nova-2-sonic-v1:0",
            component="realtime",
            semantics="cumulative",
            scope_id="c",
            input_tokens=999,
        )
        assert compute_tick_cost([delta, cumulative]) == pytest.approx(64.0)

    def test_unpriceable_delta_is_none(self):
        bad = UsageRecord(
            provider="mystery", model="m", component="realtime", input_tokens=1
        )
        assert compute_tick_cost([bad]) is None

    def test_empty_is_zero(self):
        assert compute_tick_cost([]) == 0.0


class TestSessionUsage:
    def test_build_with_breakdown(self):
        records = [
            UsageRecord(
                provider="openai",
                model="gpt-realtime-1.5",
                component="realtime",
                output_audio_tokens=500_000,
            ),
            UsageRecord(
                provider="openai",
                model="gpt-realtime-1.5",
                component="realtime",
                output_audio_tokens=500_000,
            ),
        ]
        usage = build_session_usage(records)
        assert usage.pricing_version == PRICING_VERSION
        assert usage.num_raw_records == 2
        assert len(usage.records) == 1
        assert usage.records[0].output_audio_tokens == 1_000_000
        assert usage.cost == pytest.approx(64.0)
        assert usage.cost_breakdown == {
            "realtime:openai/gpt-realtime-1.5": pytest.approx(64.0)
        }

    def test_unpriced_record_nulls_total_cost(self):
        records = [
            UsageRecord(
                provider="openai",
                model="gpt-realtime-1.5",
                component="realtime",
                output_audio_tokens=500_000,
            ),
            UsageRecord(
                provider="mystery", model="m", component="realtime", input_tokens=1
            ),
        ]
        usage = build_session_usage(records)
        assert usage.cost is None
        # priced part still visible in the breakdown
        assert "realtime:openai/gpt-realtime-1.5" in usage.cost_breakdown

    def test_xai_shape(self):
        """Informational token records + billable audio-minutes meter."""
        records = [
            UsageRecord(
                provider="xai",
                model="xai-realtime",
                component="realtime",
                input_tokens=1000,
                output_tokens=200,
                billable=False,
            ),
            UsageRecord(
                provider="xai",
                model="xai-realtime",
                component="realtime",
                semantics="cumulative",
                scope_id="audio-session-1",
                audio_input_seconds=60.0,
            ),
            # flushed at disconnect + live re-read: same scope, last wins
            UsageRecord(
                provider="xai",
                model="xai-realtime",
                component="realtime",
                semantics="cumulative",
                scope_id="audio-session-1",
                audio_input_seconds=90.0,
            ),
        ]
        usage = build_session_usage(records)
        assert usage.cost == pytest.approx(1.5 * 0.05)

    def test_unpriced_cumulative_meter_diverges_from_tick_cost(self):
        """Regression: unpriced session must not fall back to per-message $0.

        With an unpriced model, non-billable token deltas still tick-cost 0.0
        while the billable cumulative meter makes the session unpriceable.
        The orchestrator must take agent_cost from the session ledger (None),
        never from summing per-message costs.
        """
        records = [
            UsageRecord(
                provider="xai",
                model="new-unpriced-model",
                component="realtime",
                input_tokens=1000,
                billable=False,
            ),
            UsageRecord(
                provider="xai",
                model="new-unpriced-model",
                component="realtime",
                semantics="cumulative",
                scope_id="audio-session-1",
                audio_input_seconds=60.0,
            ),
        ]
        assert compute_tick_cost(records) == 0.0
        assert build_session_usage(records).cost is None


# =============================================================================
# get_cost per-side independence
# =============================================================================


class TestGetCostPerSide:
    def test_uncosted_agent_side_keeps_user_cost(self):
        messages = [
            AssistantMessage(role="assistant", content="hi", cost=None),
            UserMessage(role="user", content="hello", cost=0.01),
        ]
        agent_cost, user_cost = get_cost(messages)
        assert agent_cost is None
        assert user_cost == pytest.approx(0.01)

    def test_both_sides_costed(self):
        messages = [
            AssistantMessage(role="assistant", content="hi", cost=0.02),
            AssistantMessage(role="assistant", content="there", cost=0.03),
            UserMessage(role="user", content="hello", cost=0.01),
        ]
        agent_cost, user_cost = get_cost(messages)
        assert agent_cost == pytest.approx(0.05)
        assert user_cost == pytest.approx(0.01)

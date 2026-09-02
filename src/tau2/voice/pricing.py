"""Pricing for audio-native / voice providers.

Converts normalized UsageRecords (see tau2.data_model.usage) into USD costs.
Costs are derived at aggregation time from the table below, so past runs can
be re-priced by re-running the aggregation with an updated table — the raw
usage records are what gets persisted.

Rules:
- Unknown model or missing rate for a populated meter -> cost is None
  ("unpriced"), never silently 0.0.
- Records with billable=False contribute $0 (informational meters, e.g. xAI
  token counts — xAI bills its voice agent per audio-minute instead).
- Cached input tokens are discounted only when a cached rate is known AND the
  cached count is attributable to a modality; otherwise they are charged at
  the full input rate (conservative over-estimate).
- Cascaded LLM legs fall back to litellm's cost registry for models not in
  the table.

Rates last verified against official pricing pages on PRICING_VERSION date.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

from tau2.data_model.usage import SessionUsage, UsageRecord, aggregate_usage

PRICING_VERSION = "2026-08-04"


@dataclass(frozen=True)
class Rates:
    """Per-unit USD rates for one model. None = rate unknown/not applicable."""

    # Token rates, USD per 1M tokens
    input_text: Optional[float] = None
    input_audio: Optional[float] = None
    cached_input_text: Optional[float] = None
    cached_input_audio: Optional[float] = None
    output_text: Optional[float] = None
    output_audio: Optional[float] = None
    # Non-token meters
    per_audio_input_minute: Optional[float] = None  # STT / per-minute realtime
    per_million_characters: Optional[float] = None  # TTS


# Keyed by (provider, model). Lookup is exact match first, then longest
# key-is-prefix-of-model match (handles dated model snapshots).
PRICING: Dict[Tuple[str, str], Rates] = {
    # --- OpenAI Realtime (verified: developers.openai.com/api/docs/pricing) ---
    ("openai", "gpt-realtime-1.5"): Rates(
        input_text=4.00,
        input_audio=32.00,
        cached_input_text=0.40,
        cached_input_audio=0.40,
        output_text=16.00,
        output_audio=64.00,
    ),
    ("openai", "gpt-realtime"): Rates(  # also covers gpt-realtime-2025-08-28
        input_text=4.00,
        input_audio=32.00,
        cached_input_text=0.40,
        cached_input_audio=0.40,
        output_text=16.00,
        output_audio=64.00,
    ),
    # --- Gemini Live API (verified: ai.google.dev/gemini-api/docs/pricing).
    # No cached-token discount published for Live models: cached tokens are
    # charged at the full input rate (conservative). ---
    ("gemini", "gemini-3.1-flash-live-preview"): Rates(
        input_text=0.75,
        input_audio=3.00,
        output_text=4.50,
        output_audio=12.00,
    ),
    ("gemini", "gemini-live-2.5-flash-native-audio"): Rates(
        input_text=0.50,
        input_audio=3.00,
        output_text=2.00,
        output_audio=12.00,
    ),
    ("gemini", "gemini-2.5-flash-native-audio-preview"): Rates(
        input_text=0.50,
        input_audio=3.00,
        output_text=2.00,
        output_audio=12.00,
    ),
    # --- Qwen Omni realtime (verified: alibabacloud.com Model Studio pricing,
    # international). In audio-output mode only audio output tokens are
    # billed ($18.13/M); transcript text tokens are free, hence output_text=0.
    ("qwen", "qwen3-omni-flash-realtime"): Rates(
        input_text=0.52,
        input_audio=4.57,
        output_text=0.0,
        output_audio=18.13,
    ),
    ("qwen", "qwen3.5-omni-plus-realtime"): Rates(
        input_text=2.10,
        input_audio=16.50,
        output_text=0.0,
        output_audio=62.00,
    ),
    ("qwen", "qwen3.5-omni-flash-realtime"): Rates(
        input_text=0.55,
        input_audio=4.50,
        output_text=0.0,
        output_audio=17.70,
    ),
    # --- Amazon Nova Sonic. UNVERIFIED: AWS pricing pages are JS-rendered and
    # could not be confirmed; rates are third-party-consistent (llm-stats.com,
    # deeplearning.ai). Verify against the AWS console before relying on them.
    ("nova", "amazon.nova-2-sonic-v1:0"): Rates(
        input_text=0.33,
        input_audio=3.00,
        output_text=2.75,
        output_audio=12.00,
    ),
    ("nova", "amazon.nova-sonic-v1:0"): Rates(
        input_text=0.06,
        input_audio=3.40,
        output_text=0.24,
        output_audio=13.60,
    ),
    # --- xAI voice agent (verified: docs.x.ai/developers/models). Billed per
    # audio-minute, not per token; the endpoint-determined "xai-realtime"
    # model is assumed to map to grok-voice-think-fast-1.0 ($0.05/min).
    # Token records from xAI are recorded with billable=False. ---
    ("xai", "xai-realtime"): Rates(per_audio_input_minute=0.05),
    ("xai", "grok-voice-think-fast-1.0"): Rates(per_audio_input_minute=0.05),
    ("xai", "grok-voice-think-fast-2.0"): Rates(per_audio_input_minute=0.08),
    # --- Cascaded legs (verified: deepgram.com/pricing, elevenlabs.io/pricing,
    # developers.openai.com, platform.claude.com) ---
    ("deepgram", "nova-3"): Rates(per_audio_input_minute=0.0048),  # streaming, en
    ("deepgram", "aura-2"): Rates(per_million_characters=30.00),
    ("deepgram", "aura"): Rates(per_million_characters=15.00),  # aura-asteria-en etc.
    ("elevenlabs", "eleven_turbo"): Rates(per_million_characters=50.00),
    ("elevenlabs", "eleven_flash"): Rates(per_million_characters=50.00),
    ("elevenlabs", "eleven_multilingual"): Rates(per_million_characters=100.00),
    ("openai", "gpt-4.1"): Rates(
        input_text=2.00, cached_input_text=0.50, output_text=8.00
    ),
    ("openai", "gpt-5.2"): Rates(
        input_text=1.75, cached_input_text=0.175, output_text=14.00
    ),
    ("anthropic", "claude-sonnet-4"): Rates(
        input_text=3.00, cached_input_text=0.30, output_text=15.00
    ),
}


def lookup_rates(provider: str, model: str) -> Optional[Rates]:
    """Find rates for (provider, model): exact match, then longest prefix."""
    exact = PRICING.get((provider, model))
    if exact is not None:
        return exact
    best_key_len = -1
    best: Optional[Rates] = None
    for (p, m), rates in PRICING.items():
        if p == provider and model.startswith(m) and len(m) > best_key_len:
            best_key_len = len(m)
            best = rates
    return best


def _litellm_cost(record: UsageRecord) -> Optional[float]:
    """Fallback pricing for cascaded LLM legs via litellm's registry."""
    try:
        from litellm import cost_per_token

        prompt_cost, completion_cost = cost_per_token(
            model=record.model,
            prompt_tokens=record.input_tokens or 0,
            completion_tokens=record.output_tokens or 0,
        )
        return prompt_cost + completion_cost
    except Exception as e:
        logger.warning(f"litellm pricing fallback failed for {record.model}: {e}")
        return None


def _modality_input_cost(
    tokens: Optional[int],
    cached: Optional[int],
    rate: Optional[float],
    cached_rate: Optional[float],
) -> Optional[float]:
    """Cost of one input modality; None if tokens present but rate unknown."""
    if not tokens:
        return 0.0
    if rate is None:
        return None
    if cached and cached_rate is not None:
        uncached = max(tokens - cached, 0)
        return (uncached * rate + cached * cached_rate) / 1e6
    return tokens * rate / 1e6


def _token_cost(record: UsageRecord, rates: Rates) -> Optional[float]:
    """Token-meter cost of a record. None if a populated meter is unpriceable."""
    cost = 0.0

    # --- Input side ---
    if record.input_text_tokens is None and record.input_audio_tokens is None:
        # No modality split (cascaded LLM legs): totals are text tokens.
        if record.input_tokens:
            if rates.input_text is None:
                return None
            part = _modality_input_cost(
                record.input_tokens,
                record.input_cached_tokens,
                rates.input_text,
                rates.cached_input_text,
            )
            if part is None:
                return None
            cost += part
    else:
        text_part = _modality_input_cost(
            record.input_text_tokens,
            record.input_cached_text_tokens,
            rates.input_text,
            rates.cached_input_text,
        )
        audio_part = _modality_input_cost(
            record.input_audio_tokens,
            record.input_cached_audio_tokens,
            rates.input_audio,
            rates.cached_input_audio,
        )
        if text_part is None or audio_part is None:
            return None
        cost += text_part + audio_part

    # --- Output side ---
    if record.output_text_tokens is None and record.output_audio_tokens is None:
        if record.output_tokens:
            if rates.output_text is None:
                return None
            cost += record.output_tokens * rates.output_text / 1e6
    else:
        if record.output_text_tokens:
            if rates.output_text is None:
                return None
            cost += record.output_text_tokens * rates.output_text / 1e6
        if record.output_audio_tokens:
            if rates.output_audio is None:
                return None
            cost += record.output_audio_tokens * rates.output_audio / 1e6

    return cost


def compute_record_cost(record: UsageRecord) -> Optional[float]:
    """USD cost of one usage record.

    Returns None ("unpriced") when the model is unknown or a populated meter
    has no rate. Non-billable records always cost $0.
    """
    if not record.billable:
        return 0.0

    rates = lookup_rates(record.provider, record.model)
    if rates is None:
        if record.component == "llm":
            return _litellm_cost(record)
        logger.warning(
            f"No pricing for {record.provider}/{record.model} "
            f"({record.component}); cost will be None"
        )
        return None

    total = 0.0

    has_tokens = any(
        getattr(record, f) is not None
        for f in (
            "input_tokens",
            "output_tokens",
            "input_text_tokens",
            "input_audio_tokens",
            "output_text_tokens",
            "output_audio_tokens",
        )
    )
    if has_tokens:
        token_part = _token_cost(record, rates)
        if token_part is None:
            logger.warning(
                f"Unpriceable token meters for {record.provider}/{record.model}"
            )
            return None
        total += token_part

    if record.audio_input_seconds:
        if rates.per_audio_input_minute is None:
            logger.warning(f"No per-minute rate for {record.provider}/{record.model}")
            return None
        total += record.audio_input_seconds / 60 * rates.per_audio_input_minute

    if record.characters:
        if rates.per_million_characters is None:
            logger.warning(
                f"No per-character rate for {record.provider}/{record.model}"
            )
            return None
        total += record.characters * rates.per_million_characters / 1e6

    return total


def compute_tick_cost(records: List[UsageRecord]) -> Optional[float]:
    """Cost of the delta records reported in one tick (for per-message cost).

    Cumulative records are excluded — they are running totals and only make
    sense at session level (see build_session_usage). Returns None if any
    billable delta record is unpriceable.
    """
    total = 0.0
    for record in records:
        if record.semantics != "delta":
            continue
        cost = compute_record_cost(record)
        if cost is None:
            return None
        total += cost
    return total


def build_session_usage(records: List[UsageRecord]) -> SessionUsage:
    """Aggregate raw records and price them into a SessionUsage summary.

    ``cost`` is None if any billable aggregated record could not be priced —
    an unpriced session is distinguishable from a free one.
    """
    aggregated = aggregate_usage(records)
    breakdown: Dict[str, float] = {}
    total = 0.0
    any_unpriced = False

    for record in aggregated:
        cost = compute_record_cost(record)
        key = f"{record.component}:{record.provider}/{record.model}"
        if cost is None:
            any_unpriced = True
        else:
            total += cost
            if record.billable:
                breakdown[key] = cost

    return SessionUsage(
        records=aggregated,
        cost=None if any_unpriced else total,
        cost_breakdown=breakdown or None,
        pricing_version=PRICING_VERSION,
        num_raw_records=len(records),
    )

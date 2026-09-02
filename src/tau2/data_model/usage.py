"""Provider-agnostic usage records for audio-native / voice agents.

Voice providers bill on heterogeneous meters: realtime models bill per
audio/text/cached token (with different rates per modality), while cascaded
pipelines bill per STT audio-minute, LLM token, and TTS character. This module
defines a single normalized record type that every provider adapter emits,
so that dollar costs can be derived (and re-derived when prices change) from
a pricing table at aggregation time.

Key semantics:
- ``semantics="delta"``: the record covers one billable unit of work (e.g. one
  realtime API response, one LLM call). Delta records SUM.
- ``semantics="cumulative"``: the record is a running total (e.g. Nova Sonic's
  ``usageEvent`` reports cumulative token counts). Cumulative records aggregate
  last-value-wins per ``scope_id``, then sum across scopes.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

UsageComponent = Literal["realtime", "llm", "stt", "tts"]
UsageSemantics = Literal["delta", "cumulative"]


class UsageRecord(BaseModel):
    """One normalized usage observation from a voice provider.

    All meter fields are optional: providers populate whichever meters they
    bill on. ``input_text_tokens``/``input_audio_tokens`` include any cached
    tokens; the ``input_cached_*`` fields identify the cached subset (matching
    the OpenAI realtime usage convention).
    """

    provider: str = Field(description="Provider identifier (openai, gemini, ...)")
    model: str = Field(description="Model or component-model the usage applies to")
    component: UsageComponent = Field(
        default="realtime",
        description="Pipeline component: realtime (speech-to-speech), llm, stt, tts",
    )
    semantics: UsageSemantics = Field(
        default="delta",
        description="delta: sums across records; cumulative: last-value-wins per scope_id",
    )
    scope_id: Optional[str] = Field(
        default=None,
        description="Response/completion id. Cumulative records aggregate per scope.",
    )
    tick_number: Optional[int] = Field(
        default=None, description="Tick during which this usage was reported"
    )
    billable: bool = Field(
        default=True,
        description="False for informational records that are not the billing "
        "basis (e.g. xAI reports token counts but bills per audio-minute)",
    )

    # --- Token meters (totals include cached tokens) ---
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    input_text_tokens: Optional[int] = None
    input_audio_tokens: Optional[int] = None
    input_cached_tokens: Optional[int] = None
    input_cached_text_tokens: Optional[int] = None
    input_cached_audio_tokens: Optional[int] = None
    output_text_tokens: Optional[int] = None
    output_audio_tokens: Optional[int] = None

    # --- Non-token meters (cascaded pipeline legs) ---
    audio_input_seconds: Optional[float] = Field(
        default=None, description="Audio seconds streamed to STT"
    )
    characters: Optional[int] = Field(
        default=None, description="Characters synthesized by TTS"
    )

    raw: Optional[dict] = Field(
        default=None, description="Provider-native usage payload, for audit"
    )

    # ------------------------------------------------------------------
    # Provider-specific constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_openai_realtime_usage(
        cls,
        usage: Dict[str, Any],
        *,
        provider: str,
        model: str,
        scope_id: Optional[str] = None,
    ) -> "UsageRecord":
        """Build a record from an OpenAI-realtime-style ``response.done`` usage dict.

        Also used for Qwen and xAI, whose realtime APIs are OpenAI-compatible.
        Handles both ``input_token_details`` (OpenAI GA) and
        ``input_tokens_details`` (older / DashScope) key spellings.
        """

        def _details(prefix: str) -> Dict[str, Any]:
            return (
                usage.get(f"{prefix}_token_details")
                or usage.get(f"{prefix}_tokens_details")
                or {}
            )

        input_details = _details("input")
        output_details = _details("output")
        cached_details = input_details.get("cached_tokens_details") or {}

        return cls(
            provider=provider,
            model=model,
            component="realtime",
            semantics="delta",
            scope_id=scope_id,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            input_text_tokens=input_details.get("text_tokens"),
            input_audio_tokens=input_details.get("audio_tokens"),
            input_cached_tokens=input_details.get("cached_tokens"),
            input_cached_text_tokens=cached_details.get("text_tokens"),
            input_cached_audio_tokens=cached_details.get("audio_tokens"),
            output_text_tokens=output_details.get("text_tokens"),
            output_audio_tokens=output_details.get("audio_tokens"),
            raw=usage,
        )

    @classmethod
    def from_gemini_usage_metadata(
        cls,
        *,
        model: str,
        prompt_token_count: Optional[int] = None,
        response_token_count: Optional[int] = None,
        cached_content_token_count: Optional[int] = None,
        thoughts_token_count: Optional[int] = None,
        prompt_tokens_details: Optional[List[Dict[str, Any]]] = None,
        response_tokens_details: Optional[List[Dict[str, Any]]] = None,
        raw: Optional[dict] = None,
        scope_id: Optional[str] = None,
    ) -> "UsageRecord":
        """Build a record from Gemini Live API ``usage_metadata`` fields.

        ``*_tokens_details`` are lists of ``{"modality": ..., "token_count": ...}``
        (plain dicts, converted from the SDK's ModalityTokenCount objects).
        Thought tokens are billed as (text) output tokens.
        """

        def _modality(
            details: Optional[List[Dict[str, Any]]], name: str
        ) -> Optional[int]:
            if not details:
                return None
            total = None
            for entry in details:
                # Normalize "AUDIO", "MediaModality.AUDIO", enum reprs, etc.
                modality = str(entry.get("modality", "")).upper().split(".")[-1]
                if modality == name:
                    total = (total or 0) + (entry.get("token_count") or 0)
            return total

        output_text = _modality(response_tokens_details, "TEXT")
        if thoughts_token_count:
            output_text = (output_text or 0) + thoughts_token_count

        return cls(
            provider="gemini",
            model=model,
            component="realtime",
            semantics="delta",
            scope_id=scope_id,
            input_tokens=prompt_token_count,
            output_tokens=(
                response_token_count + (thoughts_token_count or 0)
                if response_token_count is not None
                else None
            ),
            input_text_tokens=_modality(prompt_tokens_details, "TEXT"),
            input_audio_tokens=_modality(prompt_tokens_details, "AUDIO"),
            input_cached_tokens=cached_content_token_count,
            output_text_tokens=output_text,
            output_audio_tokens=_modality(response_tokens_details, "AUDIO"),
            raw=raw,
        )

    @classmethod
    def from_nova_usage_event(
        cls,
        *,
        model: str,
        completion_id: Optional[str],
        total_input_tokens: int,
        total_output_tokens: int,
        details: Optional[Dict[str, Any]] = None,
        raw: Optional[dict] = None,
    ) -> "UsageRecord":
        """Build a record from a Nova Sonic ``usageEvent``.

        Nova reports running totals, so the record is cumulative and scoped to
        the completion id: aggregation keeps the last value per completion.
        ``details.total`` carries the speech/text breakdown.
        """
        totals = (details or {}).get("total") or {}
        input_totals = totals.get("input") or {}
        output_totals = totals.get("output") or {}

        return cls(
            provider="nova",
            model=model,
            component="realtime",
            semantics="cumulative",
            scope_id=completion_id or "session",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            input_text_tokens=input_totals.get("textTokens"),
            input_audio_tokens=input_totals.get("speechTokens"),
            output_text_tokens=output_totals.get("textTokens"),
            output_audio_tokens=output_totals.get("speechTokens"),
            raw=raw,
        )


class SessionUsage(BaseModel):
    """Aggregated usage (and derived cost) for one simulation session."""

    records: List[UsageRecord] = Field(
        default_factory=list,
        description="Aggregated records, one per (provider, model, component)",
    )
    cost: Optional[float] = Field(
        default=None,
        description="Total cost in USD. None if any record could not be priced.",
    )
    cost_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Cost per aggregated record, keyed 'component:provider/model'",
    )
    pricing_version: Optional[str] = Field(
        default=None, description="Version (date) of the pricing table used"
    )
    num_raw_records: int = Field(
        default=0, description="Number of raw usage records before aggregation"
    )


_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "input_text_tokens",
    "input_audio_tokens",
    "input_cached_tokens",
    "input_cached_text_tokens",
    "input_cached_audio_tokens",
    "output_text_tokens",
    "output_audio_tokens",
)
_FLOAT_FIELDS = ("audio_input_seconds",)
_INT_METER_FIELDS = ("characters",)


def _sum_records(
    key: Tuple[str, str, str, bool], records: List[UsageRecord]
) -> UsageRecord:
    """Sum a list of records (all same key) into one delta record.

    Optional meters stay None only if None in every summed record.
    """
    provider, model, component, billable = key
    merged: Dict[str, Any] = {}
    for field in _TOKEN_FIELDS + _INT_METER_FIELDS + _FLOAT_FIELDS:
        total = None
        for record in records:
            value = getattr(record, field)
            if value is not None:
                total = (total or 0) + value
        merged[field] = total
    return UsageRecord(
        provider=provider,
        model=model,
        component=component,
        semantics="delta",
        billable=billable,
        **merged,
    )


def aggregate_usage(records: List[UsageRecord]) -> List[UsageRecord]:
    """Aggregate raw records into one per (provider, model, component, billable).

    Delta records sum directly. Cumulative records first reduce to the last
    record per scope_id (running totals), and those per-scope totals then sum.
    Ordering of the input list is assumed chronological, as produced by the
    adapters' append-only ledgers.
    """
    grouped: Dict[Tuple[str, str, str, bool], List[UsageRecord]] = {}
    for record in records:
        key = (record.provider, record.model, record.component, record.billable)
        grouped.setdefault(key, []).append(record)

    aggregated: List[UsageRecord] = []
    for key, group in grouped.items():
        deltas = [r for r in group if r.semantics == "delta"]
        cumulatives = [r for r in group if r.semantics == "cumulative"]

        to_sum = list(deltas)
        if cumulatives:
            last_per_scope: Dict[Optional[str], UsageRecord] = {}
            for record in cumulatives:
                last_per_scope[record.scope_id] = record
            to_sum.extend(last_per_scope.values())

        aggregated.append(_sum_records(key, to_sum))

    return aggregated

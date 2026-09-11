# Copyright Sierra
"""Deterministic caller-cost counters (``tau2 metrics caller-cost``).

The "interaction tax" instrument for the τ-ML paper: how much WORK a call
extracts from the caller, per call and per (language x domain x provider)
cell, so a non-English cell's cost can later be expressed as a ratio against
the English baseline cell downstream. A SHADOW instrument by explicit
decision (2026-08-18): computed and reported, never scored or gated, and no
LLM calls anywhere.

Everything is derived from stored simulations. Voice timing lives on the
DISCRETE tick clock (tick index x tick duration) — the same clock the τ-voice
metrics and the preference feature extractor use — and the segment/turn
machinery is REUSED from the canonical extractors rather than re-derived:

- caller/agent speech segments: ``tau2.metrics.voice_interaction_metrics``
  (via ``tau2.metrics.call_timeline``), the τ-voice segment model;
- spoken turns and auth-lookup classification:
  ``tau2.metrics.interaction_quality`` (``extract_spoken_turns``,
  ``classify_auth_lookup``, ``AUTH_TOOLS_BY_DOMAIN``);
- entity conveyance / re-dictation: the entity-trace pinning and fold-aware
  detection (``tau2.metrics.entity_trace``), so the two instruments cannot
  disagree about what counts as "the caller said the value again".

Text (half-duplex) runs carry the count features; the perceptual timing
fields stay None — there is no tick clock to measure them on.
"""

import hashlib
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Annotated, Iterable, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from tau2.data_model.simulation import resolve_tick_duration_seconds
from tau2.metrics.call_timeline import (
    extract_caller_units,
    extract_tool_events,
)
from tau2.metrics.entity_trace import (
    TracedEntity,
    expected_entities,
    value_conveyed_in,
)
from tau2.metrics.interaction_quality import (
    AUTH_TOOLS_BY_DOMAIN,
    AuthLookupOutcome,
    classify_auth_lookup,
    extract_spoken_turns,
)
from tau2.metrics.run_loading import LoadedCall, RunMeta, iter_loaded_calls
from tau2.utils.utils import get_now

#: Version of the counter definitions. Bump on any semantic change.
CALLER_COST_VERSION = "1.0.0"


class CallerCostCounters(BaseModel):
    """Deterministic per-call caller-cost counters and milestones.

    Milestones are seconds on the tick clock; None means "not measurable"
    (text run) or "never happened" (e.g. auth never succeeded). Counters are
    None only when the concept does not exist for the call (no registered
    auth tools for the domain; barge-ins on a text run).
    """

    # --- milestones (tick clock; voice only) ---
    call_duration_s: Annotated[
        Optional[float],
        Field(
            description="Call duration: ticks x tick duration (voice); the "
            "sim's recorded wall-clock duration on text runs."
        ),
    ] = None
    time_to_first_agent_speech_s: Annotated[
        Optional[float],
        Field(description="Call start -> first agent speech onset."),
    ] = None
    time_to_first_tool_call_s: Annotated[
        Optional[float],
        Field(description="Call start -> first agent tool call."),
    ] = None
    time_to_auth_s: Annotated[
        Optional[float],
        Field(
            description="Call start -> first successful identity lookup "
            "result. None when auth never succeeded or the domain has no "
            "registered auth tools."
        ),
    ] = None
    caller_speaking_time_s: Annotated[
        Optional[float],
        Field(description="Total caller speech time on the tick clock."),
    ] = None
    agent_speaking_time_s: Annotated[
        Optional[float],
        Field(description="Total agent speech time on the tick clock."),
    ] = None
    # --- auth friction ---
    auth_attempt_count: Annotated[
        Optional[float],
        Field(
            description="Identity lookups up to AND INCLUDING the first "
            "successful one; total when none succeeded. None when the domain "
            "has no registered auth tools."
        ),
    ] = None
    auth_arg_mismatch_count: Annotated[
        Optional[float],
        Field(
            description="Identity lookups that identified nobody (both "
            "transports: empty non-error results and not-found errors)."
        ),
    ] = None
    # --- tool friction ---
    agent_tool_call_count: Annotated[
        Optional[float], Field(description="Total agent tool calls.")
    ] = None
    tool_error_count: Annotated[
        Optional[float],
        Field(description="Agent tool calls whose result was an error."),
    ] = None
    repeated_tool_call_count: Annotated[
        Optional[float],
        Field(
            description="Agent tool calls repeating an earlier call with the "
            "same tool and identical arguments (spin)."
        ),
    ] = None
    # --- turn economy ---
    caller_turn_count: Annotated[
        Optional[float],
        Field(
            description="Caller floor-taking turns (voice: non-backchannel "
            "segments; text: content-bearing user messages)."
        ),
    ] = None
    agent_turn_count: Annotated[
        Optional[float], Field(description="Agent spoken/content turns.")
    ] = None
    caller_repeat_turn_count: Annotated[
        Optional[float],
        Field(
            description="Adjacent caller turns that are exact repeats after "
            "punctuation/case normalization — the caller saying the same "
            "thing twice in a row."
        ),
    ] = None
    agent_repeat_turn_count: Annotated[
        Optional[float],
        Field(description="Adjacent exact-repeat agent turns (re-asks/loops)."),
    ] = None
    caller_barge_in_count: Annotated[
        Optional[float],
        Field(
            description="Floor-taking caller speech onsets while the agent "
            "was speaking (voice only)."
        ),
    ] = None
    caller_word_count: Annotated[
        Optional[float],
        Field(description="Whitespace-token count over all caller turns."),
    ] = None
    agent_word_count: Annotated[
        Optional[float],
        Field(description="Whitespace-token count over all agent turns."),
    ] = None
    # --- entity conveyance economy (the re-dictation tax) ---
    entity_count: Annotated[
        Optional[float],
        Field(description="Ground-truth identity entities pinned in the task."),
    ] = None
    entity_conveyance_count: Annotated[
        Optional[float],
        Field(description="Caller units conveying a pinned entity, summed."),
    ] = None
    entity_redictation_count: Annotated[
        Optional[float],
        Field(
            description="Conveyances beyond each entity's first — the caller "
            "saying the same identity value again."
        ),
    ] = None
    entities_redictated_count: Annotated[
        Optional[float],
        Field(description="Distinct entities the caller conveyed more than once."),
    ] = None
    post_mismatch_redictation_count: Annotated[
        Optional[float],
        Field(
            description="Re-dictations occurring AFTER the first no-match "
            "identity lookup — the correction loop a failed capture forces "
            "on the caller. None when the domain has no registered auth "
            "tools."
        ),
    ] = None


class CallerCostRecord(BaseModel):
    """One call: identity + counters."""

    results_path: Annotated[str, Field(description="Path of the run's results.json.")]
    experiment_label: Annotated[str, Field(description="Source-run dir name.")]
    sim_id: Annotated[str, Field(description="Simulation id (join key).")]
    task_id: Annotated[str, Field(description="Task id (join key).")]
    trial: Annotated[int, Field(description="Trial index.")]
    language: Annotated[str, Field(description="ISO 639-1 language of the call.")]
    domain: Annotated[str, Field(description="Domain of the run.")]
    modality: Annotated[str, Field(description="'voice' or 'text' (run-level).")]
    provider: Annotated[
        str, Field(description="Audio-native provider ('' on text runs).")
    ]
    agent_model: Annotated[str, Field(description="Agent model of the run.")]
    reasoning_effort: Annotated[
        Optional[str],
        Field(description="Recorded reasoning effort of the run, if any."),
    ] = None
    reward: Annotated[
        Optional[float], Field(description="Recorded task reward, if any.")
    ] = None
    termination_reason: Annotated[str, Field(description="How the sim terminated.")] = (
        ""
    )
    counters: Annotated[
        CallerCostCounters, Field(description="The per-call counter values.")
    ]


class MetricSummary(BaseModel):
    """Mean/median/coverage of one counter over one cell."""

    mean: Annotated[Optional[float], Field(description="Mean over non-null calls.")]
    median: Annotated[Optional[float], Field(description="Median over non-null calls.")]
    n: Annotated[int, Field(description="Calls with a non-null value.")]


class CallerCostCellAggregate(BaseModel):
    """One (language x domain x provider x modality) cell.

    The interaction-tax ratio is computed DOWNSTREAM as
    ``cell.metrics[m].mean / english_cell.metrics[m].mean`` over cells sharing
    (domain, provider, modality); the artifact carries the ingredients, never
    the ratio, so the baseline choice stays an analysis decision.
    """

    language: str
    domain: str
    modality: str
    provider: str
    n_calls: Annotated[int, Field(description="Calls aggregated into the cell.")]
    metrics: Annotated[
        dict[str, MetricSummary],
        Field(
            description="Per-counter summaries, keyed by CallerCostCounters "
            "field name (the key set is the model's field set)."
        ),
    ]

    @model_validator(mode="after")
    def _keys_are_counter_fields(self) -> "CallerCostCellAggregate":
        unknown = set(self.metrics) - set(CallerCostCounters.model_fields)
        if unknown:
            raise ValueError(
                f"cell metrics carry keys that are not CallerCostCounters "
                f"fields: {sorted(unknown)}"
            )
        return self


class CallerCostConfig(BaseModel):
    """Extraction scope, recorded in the artifact's provenance."""

    langs: Annotated[
        Optional[list[str]],
        Field(description="Restrict to these ISO 639-1 codes (None = all)."),
    ] = None
    domain: Annotated[
        Optional[str], Field(description="Restrict to one domain (None = all).")
    ] = None
    max_sims: Annotated[
        Optional[int], Field(description="Cap on extracted sims (dry runs).")
    ] = None

    @field_validator("langs")
    @classmethod
    def _fold_langs(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return None if value is None else [item.strip().lower() for item in value]


class CallerCostSourceRun(BaseModel):
    """Per-run provenance: run identity plus how many calls it contributed."""

    meta: RunMeta
    n_calls: Annotated[int, Field(description="Calls extracted from this run.")]


class CallerCostArtifact(BaseModel):
    """The provenance-bearing caller-cost artifact (SHADOW metric)."""

    schema_version: int = 1
    instrument_version: Annotated[
        str, Field(description="CALLER_COST_VERSION at build time.")
    ] = CALLER_COST_VERSION
    artifact_id: Annotated[
        str,
        Field(
            description="Content-derived id: sha256(version, config, records) "
            "truncated to 12 hex — identical inputs reproduce it."
        ),
    ]
    created_at: Annotated[str, Field(description="Build wall-clock time.")]
    git_sha: Annotated[str, Field(description="Repo HEAD at build time.")]
    config: CallerCostConfig
    source_runs: list[CallerCostSourceRun]
    records: Annotated[
        list[CallerCostRecord], Field(description="One record per call.")
    ]
    cells: Annotated[
        list[CallerCostCellAggregate],
        Field(description="Per (language x domain x provider x modality) cell."),
    ]


# ---------------------------------------------------------------------------
# Per-call extraction
# ---------------------------------------------------------------------------


def _normalized_turn(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _adjacent_repeats(turns: list[str]) -> int:
    normalized = [_normalized_turn(text) for text in turns]
    return sum(
        1
        for previous, current in zip(normalized, normalized[1:])
        if current and current == previous
    )


def _tool_call_key(call) -> tuple[str, str]:
    return call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)


def _conveyance_orders(
    entities: list[TracedEntity], units, language: str
) -> dict[TracedEntity, list[int]]:
    return {
        entity: [
            unit.order_key
            for unit in units
            if value_conveyed_in(entity.kind, entity.normalized, unit.text, language)
        ]
        for entity in entities
    }


def extract_caller_cost(call: LoadedCall) -> CallerCostRecord:
    """All caller-cost counters for one call."""
    meta, sim = call.meta, call.sim
    auth_tools = AUTH_TOOLS_BY_DOMAIN.get(meta.domain)
    units = extract_caller_units(sim)
    tool_events = extract_tool_events(sim)
    spoken_turns = extract_spoken_turns(sim)
    caller_turn_texts = [turn.text for turn in spoken_turns if turn.speaker == "caller"]
    agent_turn_texts = [turn.text for turn in spoken_turns if turn.speaker == "agent"]

    # Tool friction + auth flow over the shared timeline.
    seen_keys: set[tuple[str, str]] = set()
    repeats = 0
    errors = 0
    mismatches = 0
    auth_calls: list[int] = []
    first_auth_success_order: Optional[int] = None
    first_mismatch_order: Optional[int] = None
    for event in tool_events:
        key = _tool_call_key(event.call)
        if key in seen_keys:
            repeats += 1
        seen_keys.add(key)
        if event.result_error:
            errors += 1
        if auth_tools is not None and event.call.name in auth_tools:
            auth_calls.append(event.order_key)
            if event.result_order_key is not None:
                outcome = classify_auth_lookup(
                    event.result_content, bool(event.result_error)
                )
                if outcome is AuthLookupOutcome.NO_MATCH:
                    mismatches += 1
                    if first_mismatch_order is None:
                        first_mismatch_order = event.result_order_key
                elif (
                    outcome is AuthLookupOutcome.IDENTIFIED
                    and first_auth_success_order is None
                ):
                    first_auth_success_order = event.result_order_key

    auth_attempts: Optional[float] = None
    if auth_tools is not None:
        if first_auth_success_order is not None:
            auth_attempts = float(
                sum(1 for order in auth_calls if order <= first_auth_success_order)
            )
        else:
            auth_attempts = float(len(auth_calls))

    # Entity conveyance economy (shared pinning with the entity tracer).
    entities = expected_entities(call.task, meta.domain) if call.task else []
    conveyances = _conveyance_orders(entities, units, call.language)
    conveyance_total = sum(len(orders) for orders in conveyances.values())
    redictations = sum(max(0, len(orders) - 1) for orders in conveyances.values())
    entities_redictated = sum(1 for orders in conveyances.values() if len(orders) > 1)
    post_mismatch: Optional[float] = None
    if auth_tools is not None:
        post_mismatch = 0.0
        if first_mismatch_order is not None:
            post_mismatch = float(
                sum(
                    sum(1 for order in orders[1:] if order > first_mismatch_order)
                    for orders in conveyances.values()
                )
            )

    counters = CallerCostCounters(
        agent_tool_call_count=float(len(tool_events)),
        tool_error_count=float(errors),
        repeated_tool_call_count=float(repeats),
        auth_attempt_count=auth_attempts,
        auth_arg_mismatch_count=(float(mismatches) if auth_tools is not None else None),
        caller_turn_count=float(sum(1 for unit in units if not unit.is_backchannel)),
        agent_turn_count=float(len(agent_turn_texts)),
        caller_repeat_turn_count=float(_adjacent_repeats(caller_turn_texts)),
        agent_repeat_turn_count=float(_adjacent_repeats(agent_turn_texts)),
        caller_word_count=float(sum(len(text.split()) for text in caller_turn_texts)),
        agent_word_count=float(sum(len(text.split()) for text in agent_turn_texts)),
        entity_count=float(len(entities)),
        entity_conveyance_count=float(conveyance_total),
        entity_redictation_count=float(redictations),
        entities_redictated_count=float(entities_redictated),
        post_mismatch_redictation_count=post_mismatch,
    )

    if sim.ticks:
        ticks = sim.ticks
        dur = resolve_tick_duration_seconds(ticks, None)
        agent_speech_ticks = [
            i
            for i, tick in enumerate(ticks)
            if tick.agent_chunk is not None and tick.agent_chunk.contains_speech
        ]
        user_speech_ticks = sum(
            1
            for tick in ticks
            if tick.user_chunk is not None and tick.user_chunk.contains_speech
        )
        counters = counters.model_copy(
            update={
                "call_duration_s": len(ticks) * dur,
                "time_to_first_agent_speech_s": (
                    agent_speech_ticks[0] * dur if agent_speech_ticks else None
                ),
                "time_to_first_tool_call_s": (
                    tool_events[0].order_key * dur if tool_events else None
                ),
                "time_to_auth_s": (
                    first_auth_success_order * dur
                    if first_auth_success_order is not None
                    else None
                ),
                "caller_speaking_time_s": user_speech_ticks * dur,
                "agent_speaking_time_s": len(agent_speech_ticks) * dur,
                "caller_barge_in_count": float(
                    sum(1 for unit in units if unit.is_barge_in)
                ),
            }
        )
    else:
        counters = counters.model_copy(update={"call_duration_s": sim.duration})

    return CallerCostRecord(
        results_path=meta.results_path,
        experiment_label=meta.experiment_label,
        sim_id=str(sim.id),
        task_id=str(sim.task_id),
        trial=sim.trial or 0,
        language=call.language,
        domain=meta.domain,
        modality=meta.modality,
        provider=meta.provider,
        agent_model=meta.agent_model,
        reasoning_effort=meta.reasoning_effort,
        reward=(sim.reward_info.reward if sim.reward_info else None),
        termination_reason=(
            sim.termination_reason.value if sim.termination_reason else ""
        ),
        counters=counters,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_cells(records: list[CallerCostRecord]) -> list[CallerCostCellAggregate]:
    """Per (language x domain x provider x modality) summaries of every counter."""
    by_cell: dict[tuple[str, str, str, str], list[CallerCostRecord]] = {}
    for record in records:
        key = (record.language, record.domain, record.modality, record.provider)
        by_cell.setdefault(key, []).append(record)
    cells: list[CallerCostCellAggregate] = []
    for (language, domain, modality, provider), cell_records in sorted(by_cell.items()):
        metrics: dict[str, MetricSummary] = {}
        for field_name in CallerCostCounters.model_fields:
            values = [
                value
                for record in cell_records
                if (value := getattr(record.counters, field_name)) is not None
            ]
            metrics[field_name] = MetricSummary(
                mean=(mean(values) if values else None),
                median=(median(values) if values else None),
                n=len(values),
            )
        cells.append(
            CallerCostCellAggregate(
                language=language,
                domain=domain,
                modality=modality,
                provider=provider,
                n_calls=len(cell_records),
                metrics=metrics,
            )
        )
    return cells


def _derive_artifact_id(
    config: CallerCostConfig, records: list[CallerCostRecord]
) -> str:
    digest = hashlib.sha256()
    digest.update(CALLER_COST_VERSION.encode())
    digest.update(b"\0")
    digest.update(config.model_dump_json().encode())
    for record in records:
        digest.update(b"\0")
        digest.update(record.model_dump_json().encode())
    return digest.hexdigest()[:12]


def build_caller_cost(
    paths: Iterable[Path | str],
    config: Optional[CallerCostConfig] = None,
) -> CallerCostArtifact:
    """Extract every call under ``paths`` and assemble the artifact."""
    from tau2.annotation.artifacts import git_sha

    cfg = config or CallerCostConfig()
    records: list[CallerCostRecord] = []
    calls_per_run: dict[str, int] = {}
    metas: dict[str, RunMeta] = {}
    for call in iter_loaded_calls(
        paths, langs=cfg.langs, domain=cfg.domain, max_sims=cfg.max_sims
    ):
        records.append(extract_caller_cost(call))
        metas[call.meta.results_path] = call.meta
        calls_per_run[call.meta.results_path] = (
            calls_per_run.get(call.meta.results_path, 0) + 1
        )
    return CallerCostArtifact(
        artifact_id=_derive_artifact_id(cfg, records),
        created_at=get_now(),
        git_sha=git_sha(),
        config=cfg,
        source_runs=[
            CallerCostSourceRun(meta=metas[path], n_calls=count)
            for path, count in sorted(calls_per_run.items())
        ],
        records=records,
        cells=aggregate_cells(records),
    )

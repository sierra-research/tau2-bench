# Copyright Sierra
"""The ``tau2 metrics`` subcommand surface.

Deterministic instruments over stored runs — no LLM calls anywhere. The
facts are the headline; every composite score in these artifacts is a
permanent SHADOW column (computed and stamped, labeled uncalibrated, never
ranked or headlined — binding decision, 2026-08-18):

- ``tau2 metrics interaction-facts`` — the facts-first interaction suite:
                                  per-call deterministic interaction facts
                                  (response/yield/recovery latencies, missed
                                  responses, interruption behavior, route
                                  mix, τ-voice selectivity, dead air) with
                                  typed N/A-with-reason columns, per
                                  (language x domain x provider x
                                  reasoning-effort) cell aggregates, and the
                                  EVA turn-taking composite demoted to a
                                  shadow_scores section.
- ``tau2 metrics caller-cost``  — per-call caller-cost counters ("interaction
                                  tax": re-dictations, repeats, barge-ins,
                                  milestone timings) plus per
                                  (language x domain x provider) cell
                                  aggregates, written as one provenance-
                                  stamped JSON artifact.
- ``tau2 metrics entity-trace`` — per-entity capture/argument tracing (first
                                  capture correctness, wrong and FABRICATED
                                  values reaching tool arguments split by
                                  read-only vs state-changing tools, transient
                                  fabrications) plus per-call rollups, same
                                  artifact pattern.
"""

import argparse
from pathlib import Path

from loguru import logger


def add_metrics_args(parser: argparse.ArgumentParser) -> None:
    """Attach the metrics sub-subcommands to the ``tau2 metrics`` parser."""
    metrics_sub = parser.add_subparsers(
        dest="metrics_command", help="Metrics commands", required=True
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "results",
        nargs="+",
        help="Run dir(s) or results.json file(s); dirs are searched "
        "recursively for every run they contain",
    )
    common.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Where to write the artifact JSON",
    )
    common.add_argument(
        "--langs",
        nargs="+",
        default=None,
        help="Restrict to these ISO 639-1 codes (default: all)",
    )
    common.add_argument(
        "--domain",
        default=None,
        help="Restrict to one domain (default: all)",
    )
    common.add_argument(
        "--max-sims",
        type=int,
        default=None,
        help="Cap on processed sims (dry runs / smokes)",
    )

    interaction_facts = metrics_sub.add_parser(
        "interaction-facts",
        parents=[common],
        help="Facts-first interaction suite: deterministic per-call facts + "
        "per-cell aggregates; composite scores demoted to shadow_scores "
        "(no LLM calls)",
    )
    interaction_facts.add_argument(
        "--long-gap-threshold",
        type=float,
        default=None,
        help="Dead-air long-gap threshold in seconds (default: the shared "
        "3.0s long-silence constant)",
    )
    interaction_facts.add_argument(
        "--no-turns",
        action="store_true",
        help="Drop per-turn evidence from records (facts and cells only), "
        "for very large pools",
    )
    interaction_facts.set_defaults(func=run_metrics_interaction_facts)

    caller_cost = metrics_sub.add_parser(
        "caller-cost",
        parents=[common],
        help="Per-call caller-cost counters + per-cell aggregates "
        "(deterministic shadow metric; no LLM calls)",
    )
    caller_cost.set_defaults(func=run_metrics_caller_cost)

    entity_trace = metrics_sub.add_parser(
        "entity-trace",
        parents=[common],
        help="Per-entity capture / tool-argument tracing with the "
        "fabrication gate (deterministic shadow metric; no LLM calls)",
    )
    entity_trace.add_argument(
        "--no-events",
        action="store_true",
        help="Drop per-realization events from records (counters only), "
        "for very large pools",
    )
    entity_trace.set_defaults(func=run_metrics_entity_trace)


def run_metrics_interaction_facts(args) -> None:
    from tau2.annotation.artifacts import write_json_artifact
    from tau2.metrics.interaction_facts import (
        InteractionFactsConfig,
        build_interaction_facts,
    )

    config_data = {
        "langs": args.langs,
        "domain": args.domain,
        "max_sims": args.max_sims,
        "include_turns": not args.no_turns,
    }
    if args.long_gap_threshold is not None:
        config_data["long_gap_threshold_s"] = args.long_gap_threshold
    config = InteractionFactsConfig(**config_data)
    artifact = build_interaction_facts([Path(p) for p in args.results], config)
    write_json_artifact(args.output, artifact)
    logger.info(
        f"interaction-facts artifact {artifact.artifact_id}: "
        f"{len(artifact.records)} calls, {len(artifact.cells)} cells "
        f"-> {args.output}"
    )

    def fmt(summary, digits: int = 2) -> str:
        if summary.mean is None:
            return f"n/a({summary.n_na})"
        value = round(summary.mean, digits)
        return f"{value}" + (f" [{summary.n_na} n/a]" if summary.n_na else "")

    logger.info("headline facts (per-cell means):")
    for cell in artifact.cells:
        facts = cell.facts
        logger.info(
            f"  {cell.language}/{cell.domain}/{cell.provider or 'text'}"
            f"/{cell.reasoning_effort or 'unknown-effort'} [{cell.modality}]"
            f" n={cell.n_calls}:"
            f" resp_p50={fmt(facts['response_latency_p50_ms'], 0)}ms"
            f" missed={fmt(facts['missed_response_rate'])}"
            f" agent_intr={fmt(facts['agent_interruption_rate'])}"
            f" yield_p50={fmt(facts['yield_latency_p50_ms'], 0)}ms"
            f" dead_air={fmt(facts['dead_air_fraction'])}"
            f" long_gaps={fmt(facts['dead_air_long_gap_count'])}"
        )
    logger.info(
        "shadow scores (uncalibrated-shadow; comparability only, never a ranking):"
    )
    for cell in artifact.cells:
        shadow = cell.shadow_scores
        score = (
            "n/a"
            if shadow.turn_taking_score_mean is None
            else round(shadow.turn_taking_score_mean, 3)
        )
        na_note = (
            " ".join(
                f"{reason.value}={count}"
                for reason, count in sorted(shadow.na_reasons.items())
            )
            or "none"
        )
        logger.info(
            f"  {cell.language}/{cell.domain}/{cell.provider or 'text'}"
            f"/{cell.reasoning_effort or 'unknown-effort'}:"
            f" eva_turn_taking_mean={score}"
            f" n_scored={shadow.n_scored} n_na={shadow.n_na} ({na_note})"
        )


def run_metrics_caller_cost(args) -> None:
    from tau2.annotation.artifacts import write_json_artifact
    from tau2.metrics.caller_cost import CallerCostConfig, build_caller_cost

    config = CallerCostConfig(
        langs=args.langs, domain=args.domain, max_sims=args.max_sims
    )
    artifact = build_caller_cost([Path(p) for p in args.results], config)
    write_json_artifact(args.output, artifact)
    logger.info(
        f"caller-cost artifact {artifact.artifact_id}: "
        f"{len(artifact.records)} calls, {len(artifact.cells)} cells "
        f"-> {args.output}"
    )
    for cell in artifact.cells:
        redic = cell.metrics["entity_redictation_count"]
        turns = cell.metrics["caller_turn_count"]
        duration = cell.metrics["call_duration_s"]
        logger.info(
            f"  cell {cell.language}/{cell.domain}/{cell.provider or 'text'}"
            f" [{cell.modality}]: n={cell.n_calls}"
            f" mean_caller_turns={turns.mean if turns.mean is None else round(turns.mean, 1)}"
            f" mean_redictations={redic.mean if redic.mean is None else round(redic.mean, 2)}"
            f" mean_duration_s={duration.mean if duration.mean is None else round(duration.mean, 1)}"
        )


def run_metrics_entity_trace(args) -> None:
    from tau2.annotation.artifacts import write_json_artifact
    from tau2.metrics.entity_trace import EntityTraceConfig, build_entity_trace

    config = EntityTraceConfig(
        langs=args.langs,
        domain=args.domain,
        max_sims=args.max_sims,
        include_events=not args.no_events,
    )
    artifact = build_entity_trace([Path(p) for p in args.results], config)
    write_json_artifact(args.output, artifact)
    n_calls = len(artifact.rollups)
    n_fabricating = sum(
        1 for rollup in artifact.rollups if rollup.fabricated_event_count > 0
    )
    n_state_gate = sum(
        1 for rollup in artifact.rollups if rollup.any_fabrication_reached_state_change
    )
    first_capture_known = [
        record
        for record in artifact.records
        if record.first_capture_correct is not None
    ]
    n_first_correct = sum(
        1 for record in first_capture_known if record.first_capture_correct
    )
    logger.info(
        f"entity-trace artifact {artifact.artifact_id}: "
        f"{len(artifact.records)} entity records over {n_calls} calls "
        f"-> {args.output}"
    )
    logger.info(
        f"  first-capture correct: {n_first_correct}/{len(first_capture_known)} "
        f"entity-uses; calls with any fabrication: {n_fabricating}/{n_calls}; "
        f"calls where a fabrication reached a state-changing tool: "
        f"{n_state_gate}/{n_calls}"
    )

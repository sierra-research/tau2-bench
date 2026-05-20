from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import (
    Results,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
)
from tau2.metrics.agent_metrics import compute_metrics
from tau2.run import get_tasks, run_domain
from tau2.utils.utils import DATA_DIR, get_now

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_REASONING_EFFORTS = ["none", "low", "medium", "high", "xhigh"]
DEFAULT_VERBOSITIES = ["low", "medium", "high"]
DEFAULT_WEB_SEARCH_MODES = ["off", "auto", "required"]
DEFAULT_SERVICE_TIERS = ["default", "priority"]
DEFAULT_RESPONSES_TRANSPORTS = ["http"]
DEFAULT_PARALLEL_TOOL_CALLS = [None]
DEFAULT_BASELINE_REASONING = "medium"
DEFAULT_BASELINE_VERBOSITY = "medium"
DEFAULT_BASELINE_WEB_SEARCH = "off"
DEFAULT_BASELINE_SERVICE_TIER = "default"
DEFAULT_BASELINE_RESPONSES_TRANSPORT = "http"
DEFAULT_BASELINE_PARALLEL_TOOL_CALLS = None
DEFAULT_WEB_SEARCH_CONTEXT_SIZE = "medium"
DEFAULT_RESPONSES_MAX_STEPS = 100
DEFAULT_RESPONSES_MAX_DURATION_SECONDS = 900.0

PUBLIC_TEXT_TOKEN_RATES_PER_MILLION_USD: dict[str, dict[str, dict[str, float]]] = {
    "default": {
        "gpt-5.5": {"input": 5.0, "cached_input": 0.5, "output": 30.0},
        "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.5},
        "gpt-4.1": {"input": 2.0, "cached_input": 0.5, "output": 8.0},
    },
    "batch": {
        "gpt-5.5": {"input": 2.5, "cached_input": 0.25, "output": 15.0},
        "gpt-5.4-mini": {"input": 0.375, "cached_input": 0.0375, "output": 2.25},
        "gpt-4.1": {"input": 1.0, "cached_input": 0.0, "output": 4.0},
    },
    "flex": {
        "gpt-5.5": {"input": 2.5, "cached_input": 0.25, "output": 15.0},
        "gpt-5.4-mini": {"input": 0.375, "cached_input": 0.0375, "output": 2.25},
    },
    "priority": {
        "gpt-5.5": {"input": 12.5, "cached_input": 1.25, "output": 75.0},
        "gpt-5.4-mini": {"input": 1.5, "cached_input": 0.15, "output": 9.0},
        "gpt-4.1": {"input": 3.5, "cached_input": 0.875, "output": 14.0},
    },
}

# Public pricing lists reasoning-model web search at $10 / 1k calls.
PUBLIC_REASONING_WEB_SEARCH_CALL_COST_USD = 10.0 / 1000.0
SUPPORTED_REASONING_EFFORTS_BY_MODEL_PREFIX: dict[str, set[str]] = {
    "gpt-5.4": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5.5": {"none", "low", "medium", "high", "xhigh"},
}


class SweepShape(str, Enum):
    GRID = "grid"
    OFAT = "ofat"


class RunMode(str, Enum):
    DEFAULT = "default"
    NO_USER = "no-user"
    ORACLE_PLAN = "oracle-plan"
    NO_USER_ORACLE_PLAN = "no-user-op"


@dataclass(frozen=True)
class SweepPoint:
    reasoning_effort: str
    verbosity: str
    web_search_mode: str
    service_tier: str
    llm: Optional[str] = None
    responses_transport: str = DEFAULT_BASELINE_RESPONSES_TRANSPORT
    parallel_tool_calls: Optional[bool] = DEFAULT_BASELINE_PARALLEL_TOOL_CALLS
    variant: str = "baseline"


@dataclass(frozen=True)
class SweepConfigSpec:
    name: str
    domain: str
    mode: str
    llm: str
    llm_user: str
    reasoning_effort: str
    verbosity: str
    web_search_mode: str
    service_tier: str
    responses_transport: str
    parallel_tool_calls: Optional[bool]
    variant: str
    retrieval_config: Optional[str]
    simulation_save_to: str


BEHAVIOR_KEY_FIELDS = (
    "domain",
    "mode",
    "llm",
    "llm_user",
    "reasoning_effort",
    "verbosity",
    "web_search_mode",
    "service_tier",
    "responses_transport",
    "parallel_tool_calls",
    "retrieval_config",
)


def build_sweep_points(
    shape: SweepShape,
    *,
    reasoning_efforts: Optional[list[str]] = None,
    verbosities: Optional[list[str]] = None,
    web_search_modes: Optional[list[str]] = None,
    service_tiers: Optional[list[str]] = None,
    responses_transports: Optional[list[str]] = None,
    parallel_tool_calls: Optional[list[Optional[bool]]] = None,
    baseline_reasoning: str = DEFAULT_BASELINE_REASONING,
    baseline_verbosity: str = DEFAULT_BASELINE_VERBOSITY,
    baseline_web_search: str = DEFAULT_BASELINE_WEB_SEARCH,
    baseline_service_tier: str = DEFAULT_BASELINE_SERVICE_TIER,
    baseline_responses_transport: str = DEFAULT_BASELINE_RESPONSES_TRANSPORT,
    baseline_parallel_tool_calls: Optional[
        bool
    ] = DEFAULT_BASELINE_PARALLEL_TOOL_CALLS,
) -> list[SweepPoint]:
    reasoning_efforts = reasoning_efforts or DEFAULT_REASONING_EFFORTS
    verbosities = verbosities or DEFAULT_VERBOSITIES
    web_search_modes = web_search_modes or DEFAULT_WEB_SEARCH_MODES
    service_tiers = service_tiers or DEFAULT_SERVICE_TIERS
    responses_transports = responses_transports or DEFAULT_RESPONSES_TRANSPORTS
    parallel_tool_calls = parallel_tool_calls or DEFAULT_PARALLEL_TOOL_CALLS

    baseline = SweepPoint(
        reasoning_effort=baseline_reasoning,
        verbosity=baseline_verbosity,
        web_search_mode=baseline_web_search,
        service_tier=baseline_service_tier,
        responses_transport=baseline_responses_transport,
        parallel_tool_calls=baseline_parallel_tool_calls,
    )

    if shape == SweepShape.GRID:
        return [
            SweepPoint(
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                web_search_mode=web_search_mode,
                service_tier=service_tier,
                responses_transport=responses_transport,
                parallel_tool_calls=parallel_tool_call,
            )
            for reasoning_effort in reasoning_efforts
            for verbosity in verbosities
            for web_search_mode in web_search_modes
            for service_tier in service_tiers
            for responses_transport in responses_transports
            for parallel_tool_call in parallel_tool_calls
        ]

    if shape != SweepShape.OFAT:
        raise ValueError(f"Unsupported sweep shape: {shape}")

    points = [baseline]
    points.extend(
        SweepPoint(
            reasoning_effort=reasoning_effort,
            verbosity=baseline.verbosity,
            web_search_mode=baseline.web_search_mode,
            service_tier=baseline.service_tier,
            responses_transport=baseline.responses_transport,
            parallel_tool_calls=baseline.parallel_tool_calls,
        )
        for reasoning_effort in reasoning_efforts
        if reasoning_effort != baseline.reasoning_effort
    )
    points.extend(
        SweepPoint(
            reasoning_effort=baseline.reasoning_effort,
            verbosity=verbosity,
            web_search_mode=baseline.web_search_mode,
            service_tier=baseline.service_tier,
            responses_transport=baseline.responses_transport,
            parallel_tool_calls=baseline.parallel_tool_calls,
        )
        for verbosity in verbosities
        if verbosity != baseline.verbosity
    )
    points.extend(
        SweepPoint(
            reasoning_effort=baseline.reasoning_effort,
            verbosity=baseline.verbosity,
            web_search_mode=web_search_mode,
            service_tier=baseline.service_tier,
            responses_transport=baseline.responses_transport,
            parallel_tool_calls=baseline.parallel_tool_calls,
        )
        for web_search_mode in web_search_modes
        if web_search_mode != baseline.web_search_mode
    )
    points.extend(
        SweepPoint(
            reasoning_effort=baseline.reasoning_effort,
            verbosity=baseline.verbosity,
            web_search_mode=baseline.web_search_mode,
            service_tier=service_tier,
            responses_transport=baseline.responses_transport,
            parallel_tool_calls=baseline.parallel_tool_calls,
        )
        for service_tier in service_tiers
        if service_tier != baseline.service_tier
    )
    points.extend(
        SweepPoint(
            reasoning_effort=baseline.reasoning_effort,
            verbosity=baseline.verbosity,
            web_search_mode=baseline.web_search_mode,
            service_tier=baseline.service_tier,
            responses_transport=responses_transport,
            parallel_tool_calls=baseline.parallel_tool_calls,
        )
        for responses_transport in responses_transports
        if responses_transport != baseline.responses_transport
    )
    points.extend(
        SweepPoint(
            reasoning_effort=baseline.reasoning_effort,
            verbosity=baseline.verbosity,
            web_search_mode=baseline.web_search_mode,
            service_tier=baseline.service_tier,
            responses_transport=baseline.responses_transport,
            parallel_tool_calls=parallel_tool_call,
        )
        for parallel_tool_call in parallel_tool_calls
        if parallel_tool_call != baseline.parallel_tool_calls
    )

    deduped: list[SweepPoint] = []
    seen: set[
        tuple[str, str, str, str, Optional[str], str, Optional[bool]]
    ] = set()
    for point in points:
        key = (
            point.reasoning_effort,
            point.verbosity,
            point.web_search_mode,
            point.service_tier,
            point.llm,
            point.responses_transport,
            point.parallel_tool_calls,
        )
        if key not in seen:
            deduped.append(point)
            seen.add(key)
    return deduped


def build_known_variant_points(*, baseline_model: str = DEFAULT_MODEL) -> list[SweepPoint]:
    """Build the small variant suite discussed for Responses follow-up runs."""
    return [
        SweepPoint(
            reasoning_effort=DEFAULT_BASELINE_REASONING,
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
            variant="baseline",
        ),
        SweepPoint(
            llm="gpt-5.5",
            reasoning_effort="low",
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
            variant="gpt55_low",
        ),
        SweepPoint(
            reasoning_effort=DEFAULT_BASELINE_REASONING,
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
            parallel_tool_calls=False,
            variant="parallel_false",
        ),
        SweepPoint(
            reasoning_effort=DEFAULT_BASELINE_REASONING,
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
            parallel_tool_calls=True,
            variant="parallel_true",
        ),
        SweepPoint(
            reasoning_effort=DEFAULT_BASELINE_REASONING,
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
            responses_transport="websocket",
            variant="websocket",
        ),
    ]


def _sanitize_name_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value)


def _mode_to_agent_and_user(mode: RunMode, *, allow_web_search: bool) -> tuple[str, str]:
    if allow_web_search and mode in {RunMode.NO_USER, RunMode.NO_USER_ORACLE_PLAN}:
        raise ValueError(
            "Hosted web_search is not supported with solo agent modes because those modes require explicit environment tool calls."
        )

    if mode == RunMode.DEFAULT:
        return "llm_agent", "user_simulator"
    if mode == RunMode.ORACLE_PLAN:
        return "llm_agent_gt", "user_simulator"
    if mode == RunMode.NO_USER:
        return "llm_agent_solo", "dummy_user"
    if mode == RunMode.NO_USER_ORACLE_PLAN:
        return "llm_agent_solo_gt", "dummy_user"
    raise ValueError(f"Invalid mode: {mode}")


def build_agent_llm_args(
    base_args: dict[str, Any],
    point: SweepPoint,
    *,
    web_search_context_size: str = DEFAULT_WEB_SEARCH_CONTEXT_SIZE,
    web_search_allowed_domains: Optional[list[str]] = None,
    web_search_user_location: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    args = deepcopy(base_args)
    args["reasoning_effort"] = point.reasoning_effort
    args["verbosity"] = point.verbosity
    args["web_search_mode"] = point.web_search_mode
    args["service_tier"] = point.service_tier
    if point.responses_transport != DEFAULT_BASELINE_RESPONSES_TRANSPORT:
        args["responses_transport"] = point.responses_transport
    else:
        args.pop("responses_transport", None)
    if point.parallel_tool_calls is not DEFAULT_BASELINE_PARALLEL_TOOL_CALLS:
        args["parallel_tool_calls"] = point.parallel_tool_calls
    else:
        args.pop("parallel_tool_calls", None)

    if point.web_search_mode == "off":
        args.pop("web_search_context_size", None)
        args.pop("web_search_filters", None)
        args.pop("web_search_user_location", None)
        return args

    args["web_search_context_size"] = web_search_context_size
    if web_search_allowed_domains:
        args["web_search_filters"] = {"allowed_domains": web_search_allowed_domains}
    if web_search_user_location:
        args["web_search_user_location"] = web_search_user_location
    return args


def make_sweep_run_config(
    *,
    exp_name: str,
    llm: str,
    domain: str,
    mode: RunMode,
    point: SweepPoint,
    llm_user: str,
    llm_user_args: dict[str, Any],
    base_agent_llm_args: dict[str, Any],
    seed: int,
    max_steps: int,
    max_duration_seconds: Optional[float],
    max_errors: int,
    max_concurrency: int,
    num_trials: int,
    num_tasks: Optional[int],
    task_ids: Optional[list[str]],
    task_split_name: str,
    domain_task_splits: Optional[dict[str, str]],
    auto_resume: bool,
    web_search_context_size: str,
    web_search_allowed_domains: Optional[list[str]],
    web_search_user_location: Optional[dict[str, Any]],
) -> tuple[TextRunConfig, SweepConfigSpec]:
    resolved_llm = point.llm or llm
    agent, user = _mode_to_agent_and_user(
        mode, allow_web_search=point.web_search_mode != "off"
    )
    config_parts = [
        f"model={_sanitize_name_component(resolved_llm)}",
        f"domain={_sanitize_name_component(domain)}",
        f"mode={mode.value}",
        f"reasoning={point.reasoning_effort}",
        f"verbosity={point.verbosity}",
        f"web={point.web_search_mode}",
        f"service={point.service_tier}",
    ]
    if point.variant != "baseline":
        config_parts.append(f"variant={_sanitize_name_component(point.variant)}")
    if point.responses_transport != DEFAULT_BASELINE_RESPONSES_TRANSPORT:
        config_parts.append(f"transport={point.responses_transport}")
    if point.parallel_tool_calls is not DEFAULT_BASELINE_PARALLEL_TOOL_CALLS:
        config_parts.append(f"parallel={str(point.parallel_tool_calls).lower()}")
    config_name = "__".join(config_parts)
    save_to = f"exp/responses/{exp_name}/runs/{config_name}"
    agent_llm_args = build_agent_llm_args(
        base_agent_llm_args,
        point,
        web_search_context_size=web_search_context_size,
        web_search_allowed_domains=web_search_allowed_domains,
        web_search_user_location=web_search_user_location,
    )
    resolved_task_split_name = (
        domain_task_splits.get(domain, task_split_name)
        if domain_task_splits
        else task_split_name
    )
    config = TextRunConfig(
        domain=domain,
        task_split_name=resolved_task_split_name,
        task_ids=task_ids,
        num_tasks=num_tasks,
        agent=agent,
        llm_agent=resolved_llm,
        llm_args_agent=agent_llm_args,
        user=user,
        llm_user=llm_user,
        llm_args_user=deepcopy(llm_user_args),
        num_trials=num_trials,
        seed=seed,
        max_steps=max_steps,
        timeout=max_duration_seconds,
        max_errors=max_errors,
        max_concurrency=max_concurrency,
        save_to=save_to,
        auto_resume=auto_resume,
    )
    spec = SweepConfigSpec(
        name=config_name,
        domain=domain,
        mode=mode.value,
        llm=resolved_llm,
        llm_user=llm_user,
        reasoning_effort=point.reasoning_effort,
        verbosity=point.verbosity,
        web_search_mode=point.web_search_mode,
        service_tier=point.service_tier,
        responses_transport=point.responses_transport,
        parallel_tool_calls=point.parallel_tool_calls,
        variant=point.variant,
        retrieval_config=config.retrieval_config,
        simulation_save_to=save_to,
    )
    return config, spec


def _canonical_rate_card_model(model: str) -> Optional[str]:
    for candidate in ("gpt-5.5", "gpt-5.4-mini", "gpt-4.1"):
        if model == candidate or model.startswith(f"{candidate}-"):
            return candidate
    return None


def validate_reasoning_efforts_for_model(
    model: str, reasoning_efforts: list[str]
) -> None:
    for model_prefix, supported_efforts in SUPPORTED_REASONING_EFFORTS_BY_MODEL_PREFIX.items():
        if model.startswith(model_prefix):
            unsupported = sorted(set(reasoning_efforts) - supported_efforts)
            if unsupported:
                raise ValueError(
                    f"{model} does not support reasoning efforts {unsupported}. "
                    f"Supported values are {sorted(supported_efforts)}."
                )
            return


def estimate_token_cost_usd(
    *,
    model: str,
    service_tier: str,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
) -> Optional[float]:
    canonical_model = _canonical_rate_card_model(model)
    tier_rates = PUBLIC_TEXT_TOKEN_RATES_PER_MILLION_USD.get(service_tier)
    if canonical_model is None or tier_rates is None:
        return None

    model_rates = tier_rates.get(canonical_model)
    if model_rates is None:
        return None

    uncached_prompt_tokens = max(prompt_tokens - cached_tokens, 0)
    return (
        uncached_prompt_tokens * model_rates["input"]
        + cached_tokens * model_rates["cached_input"]
        + completion_tokens * model_rates["output"]
    ) / 1_000_000.0


def estimate_reasoning_web_search_cost_usd(model: str, search_actions: int) -> float:
    if search_actions <= 0:
        return 0.0
    if model.startswith("gpt-5") or model.startswith("o"):
        return search_actions * PUBLIC_REASONING_WEB_SEARCH_CALL_COST_USD
    return 0.0


def _simulation_metric_row(
    simulation: SimulationRun,
    spec: SweepConfigSpec,
    *,
    raw_result_path: Path,
    simulation_result_path: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        **asdict(spec),
        "simulation_id": simulation.id,
        "task_id": simulation.task_id,
        "trial": simulation.trial,
        "seed": simulation.seed,
        "reward": simulation.reward_info.reward if simulation.reward_info else None,
        "termination_reason": simulation.termination_reason,
        "duration_seconds": simulation.duration,
        "raw_result_path": str(raw_result_path),
        "simulation_result_path": str(simulation_result_path),
        "message_count": len(simulation.get_messages()),
        "tick_count": len(simulation.ticks or []),
    }

    for prefix in ("agent", "user"):
        row[f"{prefix}_prompt_tokens"] = 0
        row[f"{prefix}_completion_tokens"] = 0
        row[f"{prefix}_reasoning_tokens"] = 0
        row[f"{prefix}_total_tokens"] = 0
        row[f"{prefix}_cached_tokens"] = 0
        row[f"{prefix}_llm_calls"] = 0
        row[f"{prefix}_llm_generation_seconds"] = 0.0
        row[f"{prefix}_web_search_calls"] = 0
        row[f"{prefix}_web_search_search_actions"] = 0
        row[f"{prefix}_function_calls"] = 0

    for message in simulation.get_messages():
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AssistantMessage):
            prefix = "agent"
        elif isinstance(message, UserMessage):
            prefix = "user"
        else:
            continue

        usage = message.usage or {}
        row[f"{prefix}_prompt_tokens"] += usage.get("prompt_tokens", 0)
        row[f"{prefix}_completion_tokens"] += usage.get("completion_tokens", 0)
        row[f"{prefix}_reasoning_tokens"] += usage.get("reasoning_tokens", 0)
        row[f"{prefix}_total_tokens"] += usage.get(
            "total_tokens",
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        )
        row[f"{prefix}_cached_tokens"] += usage.get("cached_tokens", 0)

        if message.generation_time_seconds is not None:
            row[f"{prefix}_llm_calls"] += 1
            row[f"{prefix}_llm_generation_seconds"] += message.generation_time_seconds

        raw_output = (message.raw_data or {}).get("output") or []
        row[f"{prefix}_web_search_calls"] += sum(
            1 for item in raw_output if item.get("type") == "web_search_call"
        )
        row[f"{prefix}_web_search_search_actions"] += sum(
            1
            for item in raw_output
            if item.get("type") == "web_search_call"
            and (item.get("action") or {}).get("type") == "search"
        )
        row[f"{prefix}_function_calls"] += sum(
            1 for item in raw_output if item.get("type") == "function_call"
        )

    row["agent_used_web_search"] = row["agent_web_search_calls"] > 0
    row["agent_estimated_model_cost_usd"] = estimate_token_cost_usd(
        model=spec.llm,
        service_tier=spec.service_tier,
        prompt_tokens=row["agent_prompt_tokens"],
        cached_tokens=row["agent_cached_tokens"],
        completion_tokens=row["agent_completion_tokens"],
    )
    row["agent_estimated_web_search_cost_usd"] = estimate_reasoning_web_search_cost_usd(
        spec.llm, row["agent_web_search_search_actions"]
    )
    model_cost = row["agent_estimated_model_cost_usd"] or 0.0
    row["agent_estimated_total_cost_usd"] = (
        model_cost + row["agent_estimated_web_search_cost_usd"]
    )

    user_model_cost = estimate_token_cost_usd(
        model=spec.llm_user,
        service_tier="default",
        prompt_tokens=row["user_prompt_tokens"],
        cached_tokens=row["user_cached_tokens"],
        completion_tokens=row["user_completion_tokens"],
    )
    row["user_estimated_total_cost_usd"] = user_model_cost or 0.0
    row["estimated_total_cost_usd"] = (
        row["agent_estimated_total_cost_usd"] + row["user_estimated_total_cost_usd"]
    )
    return row


def _summarize_results(
    results: Results,
    spec: SweepConfigSpec,
    *,
    raw_result_path: Path,
    simulation_result_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    simulation_rows = [
        _simulation_metric_row(
            simulation,
            spec,
            raw_result_path=raw_result_path,
            simulation_result_path=simulation_result_path,
        )
        for simulation in results.simulations
    ]
    metrics = compute_metrics(results)
    df = pd.DataFrame(simulation_rows)

    summary: dict[str, Any] = {
        **asdict(spec),
        "raw_result_path": str(raw_result_path),
        "simulation_result_path": str(simulation_result_path),
        "num_simulations": len(simulation_rows),
        "num_tasks": len({row["task_id"] for row in simulation_rows}),
        "avg_reward": metrics.avg_reward,
        "avg_agent_cost": metrics.avg_agent_cost,
        "infra_error_count": metrics.infra_error_count,
        "avg_duration_seconds": df["duration_seconds"].mean() if not df.empty else 0.0,
        "agent_used_web_search_rate": (
            df["agent_used_web_search"].mean() if not df.empty else 0.0
        ),
    }

    for column in [
        "agent_prompt_tokens",
        "agent_completion_tokens",
        "agent_reasoning_tokens",
        "agent_total_tokens",
        "agent_cached_tokens",
        "agent_llm_calls",
        "agent_llm_generation_seconds",
        "agent_web_search_calls",
        "agent_web_search_search_actions",
        "agent_function_calls",
        "agent_estimated_model_cost_usd",
        "agent_estimated_web_search_cost_usd",
        "agent_estimated_total_cost_usd",
        "user_prompt_tokens",
        "user_completion_tokens",
        "user_reasoning_tokens",
        "user_total_tokens",
        "user_cached_tokens",
        "user_llm_calls",
        "user_llm_generation_seconds",
        "user_estimated_total_cost_usd",
        "estimated_total_cost_usd",
    ]:
        summary[f"total_{column}"] = df[column].sum() if not df.empty else 0
        summary[f"avg_{column}"] = df[column].mean() if not df.empty else 0

    for k, value in metrics.pass_hat_ks.items():
        summary[f"pass_hat_{k}"] = value

    return summary, simulation_rows


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _write_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)
    tmp_path.replace(path)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path, keep_default_na=False).to_dict("records")


def _normalize_optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value)
    return text or None


def _normalize_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "nan"}:
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Cannot parse optional bool value: {value!r}")


def _spec_behavior_key(spec: SweepConfigSpec) -> tuple[Any, ...]:
    values = asdict(spec)
    return tuple(values[field] for field in BEHAVIOR_KEY_FIELDS)


def _row_behavior_key(row: dict[str, Any]) -> tuple[Any, ...]:
    values = dict(row)
    values["parallel_tool_calls"] = _normalize_optional_bool(
        values.get("parallel_tool_calls")
    )
    values["retrieval_config"] = _normalize_optional_string(
        values.get("retrieval_config")
    )
    return tuple(values.get(field) for field in BEHAVIOR_KEY_FIELDS)


def _summary_rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["name"]): row for row in rows if row.get("name")}


def _simulation_rows_by_name(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = row.get("name")
        if name:
            by_name.setdefault(str(name), []).append(row)
    return by_name


def _ordered_summary_rows(
    configs: list[tuple[Any, SweepConfigSpec]],
    rows_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_names = [spec.name for _, spec in configs]
    ordered_name_set = set(ordered_names)
    rows = [rows_by_name[name] for name in ordered_names if name in rows_by_name]
    rows.extend(
        row for name, row in rows_by_name.items() if name not in ordered_name_set
    )
    return rows


def _ordered_simulation_rows(
    configs: list[tuple[Any, SweepConfigSpec]],
    rows_by_name: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ordered_names = [spec.name for _, spec in configs]
    rows: list[dict[str, Any]] = []
    for name in ordered_names:
        rows.extend(rows_by_name.get(name, []))
    ordered_name_set = set(ordered_names)
    for name, name_rows in rows_by_name.items():
        if name not in ordered_name_set:
            rows.extend(name_rows)
    return rows


def _simulation_result_path(save_to: str) -> Path:
    return DATA_DIR / "simulations" / save_to / "results.json"


def _resolve_reuse_exp_dirs(reuse_from_exp_dirs: Optional[list[str | Path]]) -> list[Path]:
    if not reuse_from_exp_dirs:
        return []

    resolved_dirs: list[Path] = []
    for exp_dir in reuse_from_exp_dirs:
        path = Path(exp_dir).expanduser()
        if not path.is_absolute():
            path = DATA_DIR / "exp" / "responses" / path
        if not path.exists():
            logger.warning(f"Skipping missing Responses cache source: {path}")
            continue
        resolved_dirs.append(path)
    return resolved_dirs


def _resolve_response_exp_dir(exp_dir: str | Path) -> Path:
    path = Path(exp_dir).expanduser()
    if not path.is_absolute():
        path = DATA_DIR / "exp" / "responses" / path
    return path


def _load_manifest_specs(exp_dir: Path) -> list[SweepConfigSpec]:
    manifest_path = exp_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    return [SweepConfigSpec(**config) for config in manifest.get("configs", [])]


def _expected_simulation_keys(config: TextRunConfig) -> set[tuple[int, str, int | None]]:
    if config.seed is None:
        logger.warning(
            f"Skipping cache lookup for {config.domain} because the run seed is None."
        )
        return set()

    rng = random.Random(config.seed)
    seeds = [rng.randint(0, 1000000) for _ in range(config.num_trials)]
    tasks = get_tasks(
        config.domain,
        task_split_name=config.task_split_name,
        task_ids=config.task_ids,
        num_tasks=config.num_tasks,
    )
    return {
        (trial, task.id, seeds[trial])
        for trial in range(config.num_trials)
        for task in tasks
    }


def _compatible_cached_simulations(
    source_results: Results,
    expected_keys: set[tuple[int, str, int | None]],
) -> list[SimulationRun]:
    simulations: list[SimulationRun] = []
    seen: set[tuple[int, str, int | None]] = set()
    for simulation in source_results.simulations:
        if simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
            continue
        key = (simulation.trial, simulation.task_id, simulation.seed)
        if key not in expected_keys or key in seen:
            continue
        simulations.append(simulation.model_copy(deep=True))
        seen.add(key)
    return simulations


def _load_valid_checkpoint_keys(path: Path) -> set[tuple[int, str, int | None]]:
    if not path.exists():
        return set()
    results = Results.load(path)
    return {
        (simulation.trial, simulation.task_id, simulation.seed)
        for simulation in results.simulations
        if simulation.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
    }


def _cache_source_paths_for_spec(cache_dir: Path, spec: SweepConfigSpec) -> list[Path]:
    paths = [_simulation_result_path(f"exp/responses/{cache_dir.name}/runs/{spec.name}")]
    results_csv = cache_dir / "results.csv"
    if results_csv.exists():
        target_key = _spec_behavior_key(spec)
        for row in _read_csv_rows(results_csv):
            if _row_behavior_key(row) != target_key:
                continue
            row_path = _normalize_optional_string(row.get("simulation_result_path"))
            if row_path:
                paths.append(Path(row_path))
                continue
            save_to = _normalize_optional_string(row.get("simulation_save_to"))
            if save_to:
                paths.append(_simulation_result_path(save_to))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        deduped.append(path)
        seen.add(path)
    return deduped


def _cached_simulation_keys(
    config: TextRunConfig,
    spec: SweepConfigSpec,
    reuse_from_exp_dirs: Optional[list[str | Path]],
    expected_keys: set[tuple[int, str, int | None]],
) -> set[tuple[int, str, int | None]]:
    cache_dirs = _resolve_reuse_exp_dirs(reuse_from_exp_dirs)
    cached_keys: set[tuple[int, str, int | None]] = set()
    for cache_dir in cache_dirs:
        for source_path in _cache_source_paths_for_spec(cache_dir, spec):
            if not source_path.exists():
                continue
            try:
                source_results = Results.load(source_path)
            except Exception as exc:
                logger.warning(f"Skipping unreadable cache source {source_path}: {exc}")
                continue
            for simulation in _compatible_cached_simulations(
                source_results, expected_keys
            ):
                cached_keys.add((simulation.trial, simulation.task_id, simulation.seed))
    return cached_keys


def seed_cached_simulations(
    config: TextRunConfig,
    spec: SweepConfigSpec,
    reuse_from_exp_dirs: Optional[list[str | Path]],
) -> int:
    """Seed this run's checkpoint with matching simulations from prior sweeps.

    The match key is intentionally strict: same generated run config name, task id,
    trial, and seed. That lets tau2's existing auto-resume logic skip only
    simulations that are valid for the current run shape and task split.
    """
    cache_dirs = _resolve_reuse_exp_dirs(reuse_from_exp_dirs)
    if not cache_dirs:
        return 0

    expected_keys = _expected_simulation_keys(config)
    if not expected_keys:
        return 0

    target_path = _simulation_result_path(spec.simulation_save_to)
    target_results: Results | None = None
    existing_keys = set()
    if target_path.exists():
        target_results = Results.load(target_path)
        existing_keys = _load_valid_checkpoint_keys(target_path)

    new_simulations: list[SimulationRun] = []
    for cache_dir in cache_dirs:
        for source_path in _cache_source_paths_for_spec(cache_dir, spec):
            if not source_path.exists():
                continue
            try:
                source_results = Results.load(source_path)
            except Exception as exc:
                logger.warning(f"Skipping unreadable cache source {source_path}: {exc}")
                continue
            for simulation in _compatible_cached_simulations(
                source_results, expected_keys
            ):
                key = (simulation.trial, simulation.task_id, simulation.seed)
                if key in existing_keys:
                    continue
                new_simulations.append(simulation)
                existing_keys.add(key)

    if not new_simulations:
        return 0

    if target_results is None:
        source_task_ids = {simulation.task_id for simulation in new_simulations}
        tasks = get_tasks(
            config.domain,
            task_split_name=config.task_split_name,
            task_ids=config.task_ids,
            num_tasks=config.num_tasks,
        )
        cached_tasks = [task for task in tasks if task.id in source_task_ids]
        target_results = Results(
            info=source_results.info,
            tasks=cached_tasks,
            simulations=[],
        )

    target_results.simulations.extend(new_simulations)
    target_results.save(target_path, format="json")
    logger.info(
        f"Seeded {len(new_simulations)} cached simulation(s) for {spec.name} "
        f"from {len(cache_dirs)} source experiment(s)."
    )
    return len(new_simulations)


def validate_response_resume_state(exp_dir: str | Path) -> list[str]:
    """Return resume-state problems for aggregate rows in a Responses experiment."""
    exp_path = _resolve_response_exp_dir(exp_dir)
    problems: list[str] = []
    for row in _read_csv_rows(exp_path / "results.csv"):
        name = row.get("name") or "<unknown>"
        save_to = _normalize_optional_string(row.get("simulation_save_to"))
        if not save_to:
            problems.append(f"{name}: missing simulation_save_to")
            continue
        canonical_path = _simulation_result_path(save_to)
        if not canonical_path.exists():
            problems.append(f"{name}: missing checkpoint {canonical_path}")
        row_path = _normalize_optional_string(row.get("simulation_result_path"))
        if row_path and Path(row_path) != canonical_path:
            problems.append(
                f"{name}: simulation_result_path points at {row_path}, "
                f"expected {canonical_path}"
            )
    return problems


def canonicalize_response_repair_results(
    *,
    exp_dir: str | Path,
    repair_exp_dirs: list[str | Path],
) -> dict[str, Any]:
    """Install repair experiment rows into an experiment's canonical resume paths."""
    target_exp_dir = _resolve_response_exp_dir(exp_dir)
    target_specs = _load_manifest_specs(target_exp_dir)
    specs_by_behavior: dict[tuple[Any, ...], SweepConfigSpec] = {}
    for spec in target_specs:
        key = _spec_behavior_key(spec)
        if key in specs_by_behavior:
            raise ValueError(f"Duplicate target behavior key for {spec.name}")
        specs_by_behavior[key] = spec

    summary_rows_by_name = _summary_rows_by_name(
        _read_csv_rows(target_exp_dir / "results.csv")
    )
    simulation_rows_by_name = _simulation_rows_by_name(
        _read_csv_rows(target_exp_dir / "simulations.csv")
    )
    raw_dir = target_exp_dir / "raw"
    canonicalized: list[str] = []
    skipped: list[str] = []

    for repair_exp_dir in _resolve_reuse_exp_dirs(repair_exp_dirs):
        for source_row in _read_csv_rows(repair_exp_dir / "results.csv"):
            source_name = str(source_row.get("name") or "<unknown>")
            target_spec = specs_by_behavior.get(_row_behavior_key(source_row))
            if target_spec is None:
                skipped.append(source_name)
                continue

            source_path_text = _normalize_optional_string(
                source_row.get("simulation_result_path")
            )
            if source_path_text:
                source_path = Path(source_path_text)
            else:
                source_save_to = _normalize_optional_string(
                    source_row.get("simulation_save_to")
                )
                if not source_save_to:
                    skipped.append(source_name)
                    continue
                source_path = _simulation_result_path(source_save_to)
            if not source_path.exists():
                logger.warning(f"Skipping missing repair checkpoint {source_path}")
                skipped.append(source_name)
                continue

            results = Results.load(source_path)
            target_raw_path = raw_dir / f"{target_spec.name}.json"
            target_simulation_path = _simulation_result_path(
                target_spec.simulation_save_to
            )
            results.save(target_raw_path, format="json")
            results.save(target_simulation_path, format="json")

            summary_row, simulation_rows = _summarize_results(
                results,
                target_spec,
                raw_result_path=target_raw_path,
                simulation_result_path=target_simulation_path,
            )
            summary_rows_by_name[target_spec.name] = summary_row
            simulation_rows_by_name[target_spec.name] = simulation_rows
            canonicalized.append(target_spec.name)

    ordered_configs = [(None, spec) for spec in target_specs]
    summary_rows = _ordered_summary_rows(ordered_configs, summary_rows_by_name)
    simulation_rows = _ordered_simulation_rows(ordered_configs, simulation_rows_by_name)
    _write_csv_atomic(summary_rows, target_exp_dir / "results.csv")
    _write_csv_atomic(simulation_rows, target_exp_dir / "simulations.csv")
    _write_json_atomic(summary_rows, target_exp_dir / "results.json")

    return {
        "exp_dir": str(target_exp_dir),
        "canonicalized": canonicalized,
        "skipped": skipped,
        "resume_state_problems": validate_response_resume_state(target_exp_dir),
    }


def _failed_config_names(
    exp_dir: Path,
    termination_reasons: list[str],
) -> set[str]:
    reason_set = set(termination_reasons)
    failed_names: set[str] = set()
    for row in _read_csv_rows(exp_dir / "simulations.csv"):
        if str(row.get("termination_reason")) not in reason_set:
            continue
        name = row.get("name")
        if name:
            failed_names.add(str(name))
    return failed_names


def _pending_simulation_keys(
    config: TextRunConfig,
    spec: SweepConfigSpec,
    *,
    effective_auto_resume: bool,
    reuse_from_exp_dirs: Optional[list[str | Path]],
) -> list[tuple[int, str, int | None]]:
    expected_keys = _expected_simulation_keys(config)
    skipped_keys: set[tuple[int, str, int | None]] = set()
    if effective_auto_resume:
        skipped_keys.update(_load_valid_checkpoint_keys(_simulation_result_path(spec.simulation_save_to)))
    if reuse_from_exp_dirs:
        skipped_keys.update(
            _cached_simulation_keys(config, spec, reuse_from_exp_dirs, expected_keys)
        )
    return sorted(
        expected_keys - skipped_keys,
        key=lambda key: (str(key[1]), key[0], -1 if key[2] is None else key[2]),
    )


def log_responses_dry_run_plan(
    runnable_configs: list[tuple[TextRunConfig, SweepConfigSpec]],
    *,
    auto_resume: bool,
    reuse_from_exp_dirs: Optional[list[str | Path]],
) -> dict[str, Any]:
    """Log and return the simulations a Responses sweep would run."""
    config_plans: list[dict[str, Any]] = []
    total = 0
    effective_auto_resume = auto_resume or bool(reuse_from_exp_dirs)
    for config, spec in runnable_configs:
        pending_keys = _pending_simulation_keys(
            config,
            spec,
            effective_auto_resume=effective_auto_resume,
            reuse_from_exp_dirs=reuse_from_exp_dirs,
        )
        if not pending_keys:
            continue
        total += len(pending_keys)
        config_plans.append(
            {
                "domain": spec.domain,
                "config": spec.name,
                "count": len(pending_keys),
                "tasks": [
                    {"trial": trial, "task_id": task_id, "seed": seed}
                    for trial, task_id, seed in pending_keys
                ],
            }
        )

    logger.info(f"Dry run: would run {total} simulation(s).")
    if total < 20:
        for plan in config_plans:
            for task in plan["tasks"]:
                logger.info(
                    "Dry run pending: "
                    f"domain={plan['domain']} config={plan['config']} "
                    f"task_id={task['task_id']} trial={task['trial']} "
                    f"seed={task['seed']}"
                )
    else:
        for plan in config_plans:
            logger.info(
                "Dry run pending: "
                f"domain={plan['domain']} config={plan['config']} "
                f"count={plan['count']}"
            )

    return {"total": total, "configs": config_plans}


def run_responses_sweep(
    *,
    exp_name: Optional[str],
    shape: SweepShape,
    llm: str = DEFAULT_MODEL,
    domains: list[str],
    modes: list[RunMode],
    llm_user: str,
    llm_user_args: dict[str, Any],
    agent_llm_args: dict[str, Any],
    seed: int,
    max_steps: int,
    max_duration_seconds: Optional[float],
    max_errors: int,
    max_concurrency: int,
    num_trials: int,
    num_tasks: Optional[int],
    task_ids: Optional[list[str]],
    task_split_name: str,
    auto_resume: bool,
    reasoning_efforts: Optional[list[str]] = None,
    verbosities: Optional[list[str]] = None,
    web_search_modes: Optional[list[str]] = None,
    service_tiers: Optional[list[str]] = None,
    responses_transports: Optional[list[str]] = None,
    parallel_tool_calls: Optional[list[Optional[bool]]] = None,
    known_variant_suite: bool = False,
    baseline_reasoning: str = DEFAULT_BASELINE_REASONING,
    baseline_verbosity: str = DEFAULT_BASELINE_VERBOSITY,
    baseline_web_search: str = DEFAULT_BASELINE_WEB_SEARCH,
    baseline_service_tier: str = DEFAULT_BASELINE_SERVICE_TIER,
    web_search_context_size: str = DEFAULT_WEB_SEARCH_CONTEXT_SIZE,
    web_search_allowed_domains: Optional[list[str]] = None,
    web_search_user_location: Optional[dict[str, Any]] = None,
    domain_task_splits: Optional[dict[str, str]] = None,
    reuse_from_exp_dirs: Optional[list[str | Path]] = None,
    failed_only_termination_reasons: Optional[list[str]] = None,
    dry_run: bool = False,
) -> Path:
    exp_name = exp_name or f"responses-sweep-{get_now(use_compact_format=True)}"
    exp_dir = DATA_DIR / "exp" / "responses" / exp_name
    raw_dir = exp_dir / "raw"
    exp_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if known_variant_suite:
        points = build_known_variant_points(baseline_model=llm)
    else:
        points = build_sweep_points(
            shape,
            reasoning_efforts=reasoning_efforts,
            verbosities=verbosities,
            web_search_modes=web_search_modes,
            service_tiers=service_tiers,
            responses_transports=responses_transports,
            parallel_tool_calls=parallel_tool_calls,
            baseline_reasoning=baseline_reasoning,
            baseline_verbosity=baseline_verbosity,
            baseline_web_search=baseline_web_search,
            baseline_service_tier=baseline_service_tier,
        )
    for point in points:
        validate_reasoning_efforts_for_model(
            point.llm or llm, [point.reasoning_effort]
        )

    configs: list[tuple[TextRunConfig, SweepConfigSpec]] = []
    for domain in domains:
        for mode in modes:
            for point in points:
                configs.append(
                    make_sweep_run_config(
                        exp_name=exp_name,
                        llm=llm,
                        domain=domain,
                        mode=mode,
                        point=point,
                        llm_user=llm_user,
                        llm_user_args=llm_user_args,
                        base_agent_llm_args=agent_llm_args,
                        seed=seed,
                        max_steps=max_steps,
                        max_duration_seconds=max_duration_seconds,
                        max_errors=max_errors,
                        max_concurrency=max_concurrency,
                        num_trials=num_trials,
                        num_tasks=num_tasks,
                        task_ids=task_ids,
                        task_split_name=task_split_name,
                        domain_task_splits=domain_task_splits,
                        auto_resume=auto_resume,
                        web_search_context_size=web_search_context_size,
                        web_search_allowed_domains=web_search_allowed_domains,
                        web_search_user_location=web_search_user_location,
                    )
                )

    runnable_configs = configs
    if failed_only_termination_reasons:
        failed_names = _failed_config_names(exp_dir, failed_only_termination_reasons)
        runnable_configs = [
            (config, spec) for config, spec in configs if spec.name in failed_names
        ]
        auto_resume = True
        if not runnable_configs:
            logger.info(
                "No configs with failed simulations matching "
                f"{failed_only_termination_reasons}; nothing to run."
            )

    if dry_run:
        log_responses_dry_run_plan(
            runnable_configs,
            auto_resume=auto_resume,
            reuse_from_exp_dirs=reuse_from_exp_dirs,
        )
        return exp_dir

    manifest = {
        "exp_name": exp_name,
        "shape": shape.value,
        "model": llm,
        "known_variant_suite": known_variant_suite,
        "domains": domains,
        "modes": [mode.value for mode in modes],
        "num_trials": num_trials,
        "num_tasks": num_tasks,
        "task_ids": task_ids,
        "task_split_name": task_split_name,
        "domain_task_splits": domain_task_splits or {},
        "max_steps": max_steps,
        "max_duration_seconds": max_duration_seconds,
        "dry_run": dry_run,
        "failed_only_termination_reasons": failed_only_termination_reasons or [],
        "reuse_from_exp_dirs": [
            str(path) for path in _resolve_reuse_exp_dirs(reuse_from_exp_dirs)
        ],
        "points": [asdict(point) for point in points],
        "configs": [asdict(spec) for _, spec in configs],
    }
    with open(exp_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary_rows_by_name = _summary_rows_by_name(
        _read_csv_rows(exp_dir / "results.csv")
    )
    simulation_rows_by_name = _simulation_rows_by_name(
        _read_csv_rows(exp_dir / "simulations.csv")
    )

    for index, (config, spec) in enumerate(runnable_configs, start=1):
        logger.info(f"{index}/{len(runnable_configs)} running {spec.name}")
        if failed_only_termination_reasons:
            object.__setattr__(config, "auto_resume", True)
        if reuse_from_exp_dirs:
            object.__setattr__(config, "auto_resume", True)
            seed_cached_simulations(config, spec, reuse_from_exp_dirs)
        results = run_domain(config)

        raw_result_path = raw_dir / f"{spec.name}.json"
        results.save(raw_result_path, format="json")

        simulation_result_path = _simulation_result_path(spec.simulation_save_to)
        summary_row, config_sim_rows = _summarize_results(
            results,
            spec,
            raw_result_path=raw_result_path,
            simulation_result_path=simulation_result_path,
        )
        summary_rows_by_name[spec.name] = summary_row
        simulation_rows_by_name[spec.name] = config_sim_rows

        summary_rows = _ordered_summary_rows(configs, summary_rows_by_name)
        simulation_rows = _ordered_simulation_rows(configs, simulation_rows_by_name)
        _write_csv_atomic(summary_rows, exp_dir / "results.csv")
        _write_csv_atomic(simulation_rows, exp_dir / "simulations.csv")
        _write_json_atomic(summary_rows, exp_dir / "results.json")

    return exp_dir

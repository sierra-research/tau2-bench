#!/usr/bin/env python3
"""Plan targeted additional tau2 data collection without running simulations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNCERTAINTY_JSONL = REPO_ROOT / "data/processed/tau2_shift_uncertainty.jsonl"
DEFAULT_UNCERTAINTY_SUMMARY_JSON = REPO_ROOT / "data/processed/tau2_shift_uncertainty_summary.json"
DEFAULT_INVENTORY_JSONL = REPO_ROOT / "data/processed/toolcalling_shift_inventory.jsonl"
DEFAULT_UNIFIED_TAU2_JSONL = REPO_ROOT / "data/processed/unified_toolcalling_tau2.jsonl"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "data/processed/tau2_additional_sampling_plan.json"
DEFAULT_REPORT = REPO_ROOT / "docs/tau2_additional_sampling_plan.md"

DOMAIN_TASK_FILES = {
    "retail": REPO_ROOT / "data/tau2/domains/retail/tasks.json",
    "airline": REPO_ROOT / "data/tau2/domains/airline/tasks.json",
    "telecom": REPO_ROOT / "data/tau2/domains/telecom/tasks.json",
    "banking_knowledge": REPO_ROOT / "data/tau2/domains/banking_knowledge/tasks.json",
}
PRIOR_SIMULATION_RESULT_FILES = {
    "retail": REPO_ROOT
    / "data/simulations/20260714_131548_retail_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini/results.json",
    "airline": REPO_ROOT
    / "data/simulations/20260714_140540_airline_llm_agent_gpt-4o-mini_user_simulator_gpt-4o-mini/results.json",
}

PRECISION_HALF_WIDTHS = (0.15, 0.10, 0.05)
POWER_EFFECT_SIZES = (0.15, 0.10, 0.05)
EQUIVALENCE_MARGINS = (0.15, 0.10, 0.05)
PRACTICAL_THRESHOLDS = ("0.05", "0.10", "0.15")
Z_ALPHA_TWO_SIDED_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
STAGE_1_BATCH_SIZE = 12
TELECOM_WARNING = (
    "Telecom is technically runnable, but the current feasibility test was slow, "
    "rate-limited, reached max steps, and cost approximately $0.095 for one "
    "unsuccessful task. Do not recommend large telecom runs without a separate "
    "cost-control plan."
)
PLANNING_LIMITATION = (
    "Planning estimates use standard normal approximations for two independent "
    "proportions. They are planning estimates, not guarantees."
)

WRITE_PREFIXES = (
    "add",
    "apply",
    "book",
    "buy",
    "cancel",
    "change",
    "create",
    "delete",
    "disable",
    "enable",
    "exchange",
    "grant",
    "modify",
    "order",
    "pay",
    "post",
    "refuel",
    "remove",
    "reserve",
    "reset",
    "return",
    "schedule",
    "send",
    "set",
    "submit",
    "toggle",
    "transfer",
    "update",
)
READ_PREFIXES = ("calculate", "check", "find", "get", "list", "query", "read", "retrieve", "search")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def ceil_int(value: float) -> int:
    return int(math.ceil(value - 1e-12))


def balanced_additional(final_n_per_group: int, source_n: int, target_n: int) -> dict[str, int]:
    add_source = max(0, final_n_per_group - source_n)
    add_target = max(0, final_n_per_group - target_n)
    return {
        "required_final_n_per_group": final_n_per_group,
        "additional_source_n": add_source,
        "additional_target_n": add_target,
        "additional_total_n": add_source + add_target,
    }


def precision_estimates(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    p_source = row["source_success_rate"]
    p_target = row["target_success_rate"]
    variance_sum = p_source * (1.0 - p_source) + p_target * (1.0 - p_target)
    estimates = {}
    for half_width in PRECISION_HALF_WIDTHS:
        required = ceil_int((Z_ALPHA_TWO_SIDED_95**2) * variance_sum / (half_width**2))
        estimates[f"{half_width:.2f}"] = {
            **balanced_additional(required, row["source_n"], row["target_n"]),
            "target_half_width": half_width,
            "method": "normal approximation: z*sqrt(p_s(1-p_s)/n + p_t(1-p_t)/n)",
            "available": True,
        }
    return estimates


def power_estimates(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    pooled_rate = (
        row["source_positive"] + row["target_positive"]
    ) / (row["source_n"] + row["target_n"])
    estimates = {}
    for effect_size in POWER_EFFECT_SIZES:
        p_low = pooled_rate - (effect_size / 2.0)
        p_high = pooled_rate + (effect_size / 2.0)
        key = f"{effect_size:.2f}"
        if p_low < 0.0 or p_high > 1.0:
            estimates[key] = {
                "available": False,
                "reason": "pooled-rate +/- effect_size/2 falls outside [0, 1]",
                "effect_size": effect_size,
            }
            continue
        pooled_variance = 2.0 * pooled_rate * (1.0 - pooled_rate)
        alternative_variance = p_low * (1.0 - p_low) + p_high * (1.0 - p_high)
        numerator = (
            Z_ALPHA_TWO_SIDED_95 * math.sqrt(pooled_variance)
            + Z_POWER_80 * math.sqrt(alternative_variance)
        ) ** 2
        required = ceil_int(numerator / (effect_size**2))
        estimates[key] = {
            **balanced_additional(required, row["source_n"], row["target_n"]),
            "effect_size": effect_size,
            "alpha": 0.05,
            "power": 0.80,
            "method": (
                "two-sided two-sample proportions normal approximation using current "
                "pooled rate and symmetric planning alternatives"
            ),
            "available": True,
        }
    return estimates


def equivalence_estimates(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    estimates = {}
    for margin in EQUIVALENCE_MARGINS:
        key = f"{margin:.2f}"
        remaining_half_width = margin - abs(row["delta_y"])
        if remaining_half_width <= 0.0:
            estimates[key] = {
                "available": False,
                "margin": margin,
                "reason": (
                    "current absolute delta_y is already at or outside the equivalence margin; "
                    "a rigorous equivalence-power estimate requires an assumed true delta"
                ),
            }
            continue
        variance_sum = (
            row["source_success_rate"] * (1.0 - row["source_success_rate"])
            + row["target_success_rate"] * (1.0 - row["target_success_rate"])
        )
        required = ceil_int((Z_ALPHA_TWO_SIDED_95**2) * variance_sum / (remaining_half_width**2))
        estimates[key] = {
            **balanced_additional(required, row["source_n"], row["target_n"]),
            "margin": margin,
            "available": True,
            "method": (
                "CI-screening approximation: final 95% CI half-width small enough "
                "that current delta_y would fit inside +/- margin; not a TOST power guarantee"
            ),
        }
    return estimates


def action_kind(action_name: str | None) -> str:
    lowered = (action_name or "").lower()
    if lowered.startswith(WRITE_PREFIXES):
        return "write"
    if lowered.startswith(READ_PREFIXES):
        return "read"
    return "unknown"


def normalize_task_id(value: Any) -> str:
    return str(value)


def task_sort_key(task_id: str) -> tuple[int, str]:
    if task_id.isdigit():
        return int(task_id), task_id
    digits = "".join(ch for ch in task_id if ch.isdigit())
    if digits:
        return int(digits), task_id
    return 10**9, task_id


def task_features(domain: str, task: dict[str, Any]) -> dict[str, Any]:
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    action_names = [action.get("name") for action in actions]
    kinds = [action_kind(name) for name in action_names]
    write_count = kinds.count("write")
    read_count = kinds.count("read")
    return {
        "domain": domain,
        "task_id": normalize_task_id(task.get("id")),
        "expected_action_count": len(actions),
        "expected_read_action_count": read_count,
        "expected_write_action_count": write_count,
        "unknown_action_count": kinds.count("unknown"),
        "action_names": action_names,
    }


def load_task_pool() -> dict[str, dict[str, Any]]:
    domains = {}
    for domain, path in DOMAIN_TASK_FILES.items():
        if not path.exists():
            domains[domain] = {"available": False, "task_count": 0, "tasks": []}
            continue
        tasks = load_json(path)
        features = [task_features(domain, task) for task in tasks]
        domains[domain] = {
            "available": True,
            "task_count": len(features),
            "path": str(path.relative_to(REPO_ROOT)),
            "tasks": sorted(features, key=lambda item: task_sort_key(item["task_id"])),
        }
    return domains


def prior_simulated_ids(domain: str) -> set[str]:
    path = PRIOR_SIMULATION_RESULT_FILES.get(domain)
    if path is None or not path.exists():
        return set()
    data = load_json(path)
    return {normalize_task_id(sim.get("task_id")) for sim in data.get("simulations") or []}


def retained_outcome_ids(unified_rows: list[dict[str, Any]], domain: str) -> set[str]:
    return {
        normalize_task_id(row.get("metadata", {}).get("task_id"))
        for row in unified_rows
        if row.get("source_dataset") == "tau2" and row.get("domain") == domain
    }


def group_for_shift(shift_id: str, item: dict[str, Any], thresholds: dict[str, Any]) -> str | None:
    domain = item["domain"]
    action_count = item["expected_action_count"]
    write_count = item["expected_write_action_count"]
    if shift_id == "tau2_retail_to_airline":
        if domain == "retail":
            return "source"
        if domain == "airline":
            return "target"
        return None
    if shift_id == "tau2_no_write_to_write_required":
        if write_count == thresholds["source_max"]:
            return "source"
        if write_count >= thresholds["target_min"]:
            return "target"
        return None
    if shift_id == "tau2_zero_or_one_write_to_two_plus_writes":
        if write_count <= thresholds["source_max"]:
            return "source"
        if write_count >= thresholds["target_min"]:
            return "target"
        return None
    if shift_id == "tau2_few_to_many_expected_actions":
        if action_count <= thresholds["lower_threshold"]:
            return "source"
        if action_count >= thresholds["upper_threshold"]:
            return "target"
        return None
    return None


def current_group_membership(
    shift_id: str,
    item: dict[str, Any],
    thresholds: dict[str, Any],
) -> str | None:
    domain = item["domain"]
    x_features = item.get("x_numeric_features", {})
    metadata = item.get("metadata", {})
    s_values = [value for value in item.get("s_raw", []) if value != 0]
    if shift_id == "tau2_retail_to_airline":
        if domain == "retail":
            return "source"
        if domain == "airline":
            return "target"
    if shift_id == "tau2_no_write_to_write_required":
        value = x_features.get("expected_write_action_count")
        if value == thresholds["source_max"]:
            return "source"
        if value is not None and value >= thresholds["target_min"]:
            return "target"
    if shift_id == "tau2_zero_or_one_write_to_two_plus_writes":
        value = x_features.get("expected_write_action_count")
        if value is not None and value <= thresholds["source_max"]:
            return "source"
        if value is not None and value >= thresholds["target_min"]:
            return "target"
    if shift_id == "tau2_few_to_many_expected_actions":
        value = x_features.get("expected_action_count")
        if value is not None and value <= thresholds["lower_threshold"]:
            return "source"
        if value is not None and value >= thresholds["upper_threshold"]:
            return "target"
    if shift_id == "tau2_short_to_long_trajectory":
        value = len(s_values)
        if value <= thresholds["lower_threshold"]:
            return "source"
        if value >= thresholds["upper_threshold"]:
            return "target"
    if shift_id == "tau2_few_to_many_tool_calls":
        value = metadata.get("num_tool_calls")
        if value is not None and value <= thresholds["lower_threshold"]:
            return "source"
        if value is not None and value >= thresholds["upper_threshold"]:
            return "target"
    return None


def group_counts(tasks: list[dict[str, Any]], shift_id: str, thresholds: dict[str, Any]) -> dict[str, int]:
    counts = Counter(group_for_shift(shift_id, task, thresholds) for task in tasks)
    return {"source": counts.get("source", 0), "target": counts.get("target", 0), "unknown": counts.get(None, 0)}


def unavailable_group_counts(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def selected_task_payload(task: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "domain": task["domain"],
        "task_id": task["task_id"],
        "expected_action_count": task["expected_action_count"],
        "expected_write_action_count": task["expected_write_action_count"],
        "selection_reasons": reasons,
        "uses_y": False,
    }


def score_stage1_task(task: dict[str, Any]) -> tuple[int, int, int]:
    write_count = task["expected_write_action_count"]
    action_count = task["expected_action_count"]
    score = 0
    if write_count >= 2:
        score += 100
    if action_count >= 6:
        score += 20
    if action_count <= 1:
        score += 10
    return -score, action_count, task_sort_key(task["task_id"])[0]


def choose_stage1_tasks(retail_not_attempted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    two_plus = [task for task in retail_not_attempted if task["expected_write_action_count"] >= 2]
    no_write = [task for task in retail_not_attempted if task["expected_write_action_count"] == 0]
    low_action = [
        task
        for task in retail_not_attempted
        if task["expected_action_count"] <= 1 and task["expected_write_action_count"] <= 1
    ]
    selected: list[dict[str, Any]] = []

    for task in sorted(two_plus, key=score_stage1_task)[:8]:
        selected.append(
            selected_task_payload(
                task,
                ["retail", "unused by prior simulation", "two or more expected write actions"],
            )
        )
    for task in sorted(no_write, key=lambda item: task_sort_key(item["task_id"]))[:2]:
        selected.append(
            selected_task_payload(
                task,
                ["retail", "unused by prior simulation", "no expected write actions"],
            )
        )
    selected_ids = {task["task_id"] for task in selected}
    for task in sorted(low_action, key=lambda item: task_sort_key(item["task_id"])):
        if task["task_id"] in selected_ids:
            continue
        selected.append(
            selected_task_payload(
                task,
                ["retail", "unused by prior simulation", "few expected actions"],
            )
        )
        if len(selected) >= STAGE_1_BATCH_SIZE:
            break
    return selected[:STAGE_1_BATCH_SIZE]


def nested_relationships(group_sets: dict[str, dict[str, set[str]]]) -> dict[str, list[str]]:
    relationships = {shift_id: [] for shift_id in group_sets}
    for left_id, left_groups in group_sets.items():
        for right_id, right_groups in group_sets.items():
            if left_id == right_id:
                continue
            for group_name in ("source", "target"):
                left = left_groups[group_name]
                right = right_groups[group_name]
                if left and left < right:
                    relationships[left_id].append(f"{group_name} group is a strict subset of {right_id}")
                elif right and right < left:
                    relationships[left_id].append(f"{group_name} group strictly contains {right_id}")
    return relationships


def build_task_pool_audit(
    *,
    uncertainty_rows: list[dict[str, Any]],
    inventory_by_id: dict[str, dict[str, Any]],
    unified_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    task_pool = load_task_pool()
    sampled_by_domain = {
        domain: retained_outcome_ids(unified_rows, domain)
        for domain in ("retail", "airline", "telecom", "banking_knowledge")
    }
    attempted_by_domain = {domain: prior_simulated_ids(domain) for domain in ("retail", "airline")}
    domain_audit = {}
    for domain, info in task_pool.items():
        tasks = info["tasks"]
        local_ids = {task["task_id"] for task in tasks}
        retained = sampled_by_domain.get(domain, set())
        attempted = attempted_by_domain.get(domain, set())
        domain_audit[domain] = {
            "available": info["available"],
            "local_task_count": len(local_ids),
            "retained_outcome_count": len(retained),
            "unused_by_retained_outcome_count": len(local_ids - retained),
            "prior_simulation_attempt_count": len(attempted) if attempted else None,
            "not_previously_attempted_count": len(local_ids - attempted) if attempted else None,
            "all_local_tasks_previously_attempted": local_ids <= attempted if attempted else None,
        }

    all_tasks = [task for info in task_pool.values() for task in info["tasks"]]
    retail_tasks = task_pool["retail"]["tasks"]
    retail_retained = sampled_by_domain["retail"]
    retail_attempted = attempted_by_domain["retail"]
    retail_unused_by_outcome = [task for task in retail_tasks if task["task_id"] not in retail_retained]
    retail_not_attempted = [task for task in retail_tasks if task["task_id"] not in retail_attempted]
    stage1 = choose_stage1_tasks(retail_not_attempted)

    shift_audits = {}
    current_group_sets: dict[str, dict[str, set[str]]] = {}
    for row in uncertainty_rows:
        shift_id = row["shift_id"]
        thresholds = inventory_by_id[shift_id]["thresholds"]
        current_group_sets[shift_id] = {"source": set(), "target": set()}
        for item in unified_rows:
            group = current_group_membership(shift_id, item, thresholds)
            if group in ("source", "target"):
                current_group_sets[shift_id][group].add(item["sample_id"])

    nested = nested_relationships(current_group_sets)
    for row in uncertainty_rows:
        shift_id = row["shift_id"]
        thresholds = inventory_by_id[shift_id]["thresholds"]
        smaller_group = "source" if row["source_n"] < row["target_n"] else "target" if row["target_n"] < row["source_n"] else "balanced"
        if shift_id in {
            "tau2_retail_to_airline",
            "tau2_no_write_to_write_required",
            "tau2_zero_or_one_write_to_two_plus_writes",
            "tau2_few_to_many_expected_actions",
        }:
            counts_all = {"available": True, **group_counts(all_tasks, shift_id, thresholds)}
            counts_retail_unused = {
                "available": True,
                **group_counts(retail_unused_by_outcome, shift_id, thresholds),
            }
            candidate_tasks = [
                task
                for task in retail_not_attempted
                if group_for_shift(shift_id, task, thresholds) == smaller_group
            ]
            task_selection_without_y = True
        else:
            counts_all = unavailable_group_counts(
                "shift uses observed trajectory length or observed tool-call count; unused task membership is unknown before running"
            )
            counts_retail_unused = counts_all
            candidate_tasks = []
            task_selection_without_y = True

        shift_audits[shift_id] = {
            "smaller_group": smaller_group,
            "groups_overlap": row["source_target_overlap_count"] > 0,
            "source_target_overlap_count": row["source_target_overlap_count"],
            "nested_with_another_shift": bool(nested[shift_id]),
            "nested_relationships": sorted(set(nested[shift_id])),
            "local_task_group_counts": counts_all,
            "unused_retail_group_counts": counts_retail_unused,
            "unused_tasks_in_smaller_group": [
                selected_task_payload(task, [f"increases {smaller_group} group for {shift_id}"])
                for task in sorted(candidate_tasks, key=lambda item: task_sort_key(item["task_id"]))[:20]
            ],
            "repeated_trials_required": len(candidate_tasks) == 0 and smaller_group != "balanced",
            "task_ids_selectable_without_y": task_selection_without_y,
        }

    return {
        "domains": domain_audit,
        "retail_unused_task_ids_by_retained_outcome": [
            task["task_id"] for task in sorted(retail_unused_by_outcome, key=lambda item: task_sort_key(item["task_id"]))
        ],
        "retail_not_previously_attempted_task_ids": [
            task["task_id"] for task in sorted(retail_not_attempted, key=lambda item: task_sort_key(item["task_id"]))
        ],
        "airline_all_tasks_previously_sampled": domain_audit["airline"]["all_local_tasks_previously_attempted"],
        "shift_audits": shift_audits,
        "stage1_recommended_tasks": stage1,
        "stage1_batch_size": len(stage1),
        "task_selection_policy": "Task IDs are selected from local task definitions and prior sample IDs only; observed y is not used.",
    }


def result_summary(row: dict[str, Any]) -> dict[str, Any]:
    ci = row["delta_y_ci_95"]
    classifications = row["classification_by_threshold"]
    return {
        "shift_id": row["shift_id"],
        "source_n": row["source_n"],
        "target_n": row["target_n"],
        "source_success_rate": row["source_success_rate"],
        "target_success_rate": row["target_success_rate"],
        "delta_y": row["delta_y"],
        "delta_y_ci_95": ci,
        "delta_y_ci_width": ci[1] - ci[0],
        "classification_by_threshold": {threshold: classifications[threshold] for threshold in PRACTICAL_THRESHOLDS},
        "group_with_fewer_samples": (
            "source" if row["source_n"] < row["target_n"] else "target" if row["target_n"] < row["source_n"] else "balanced"
        ),
        "groups_overlap": row["source_target_overlap_count"] > 0,
        "source_target_overlap_count": row["source_target_overlap_count"],
    }


def build_plan(
    *,
    uncertainty_jsonl: Path = DEFAULT_UNCERTAINTY_JSONL,
    uncertainty_summary_json: Path = DEFAULT_UNCERTAINTY_SUMMARY_JSON,
    inventory_jsonl: Path = DEFAULT_INVENTORY_JSONL,
    unified_tau2_jsonl: Path = DEFAULT_UNIFIED_TAU2_JSONL,
) -> dict[str, Any]:
    uncertainty_rows = load_jsonl(uncertainty_jsonl)
    uncertainty_summary = load_json(uncertainty_summary_json)
    inventory_rows = load_jsonl(inventory_jsonl)
    unified_rows = load_jsonl(unified_tau2_jsonl)
    inventory_by_id = {
        row["shift_id"]: row
        for row in inventory_rows
        if row.get("dataset") == "tau2" and row.get("status") == "eligible"
    }
    if len(uncertainty_rows) != 6:
        raise ValueError(f"Expected 6 tau2 uncertainty rows, found {len(uncertainty_rows)}")
    if any(not row["shift_id"].startswith("tau2_") for row in uncertainty_rows):
        raise ValueError("Additional-sampling plan only supports tau2 shifts")
    if not set(row["shift_id"] for row in uncertainty_rows) <= set(inventory_by_id):
        raise ValueError("Uncertainty rows are not all present in eligible tau2 inventory")

    task_pool_audit = build_task_pool_audit(
        uncertainty_rows=uncertainty_rows,
        inventory_by_id=inventory_by_id,
        unified_rows=unified_rows,
    )
    shift_plans = []
    for row in uncertainty_rows:
        base = result_summary(row)
        base.update(
            {
                "nested_with_another_shift": task_pool_audit["shift_audits"][row["shift_id"]][
                    "nested_with_another_shift"
                ],
                "nested_relationships": task_pool_audit["shift_audits"][row["shift_id"]][
                    "nested_relationships"
                ],
                "precision_planning": precision_estimates(row),
                "power_planning": power_estimates(row),
                "equivalence_planning": equivalence_estimates(row),
            }
        )
        shift_plans.append(base)

    candidate_harmful = [
        row["shift_id"]
        for row in uncertainty_rows
        if "candidate_harmful" in set(row["classification_by_threshold"].values())
    ]
    inconclusive = [
        row["shift_id"]
        for row in uncertainty_rows
        if "inconclusive" in set(row["classification_by_threshold"].values())
    ]
    return {
        "metadata": {
            "created_for": "tau2 additional-data collection planning",
            "api_bank_excluded": True,
            "no_new_llm_simulations_run": True,
            "shift_definitions_modified": False,
            "model_training_performed": False,
            "planning_limitation": PLANNING_LIMITATION,
        },
        "inputs": {
            "uncertainty_jsonl": str(uncertainty_jsonl.relative_to(REPO_ROOT)),
            "uncertainty_summary_json": str(uncertainty_summary_json.relative_to(REPO_ROOT)),
            "inventory_jsonl": str(inventory_jsonl.relative_to(REPO_ROOT)),
            "unified_tau2_jsonl": str(unified_tau2_jsonl.relative_to(REPO_ROOT)),
            "local_task_definition_files": {
                domain: str(path.relative_to(REPO_ROOT)) for domain, path in DOMAIN_TASK_FILES.items()
            },
        },
        "current_evidence_summary": {
            "eligible_tau2_shift_count": len(uncertainty_rows),
            "classification_counts_by_threshold": uncertainty_summary["classification_counts_by_threshold"],
            "candidate_harmful_shifts": candidate_harmful,
            "inconclusive_shifts": inconclusive,
            "real_task_level_outcome_count": len(unified_rows),
        },
        "shifts": shift_plans,
        "task_pool_audit": task_pool_audit,
        "recommended_collection_priority": [
            "Prioritize tau2_zero_or_one_write_to_two_plus_writes because it is currently candidate_harmful at all practical thresholds.",
            "Use unused retail tasks first, especially tasks with two or more expected write actions, because they increase the smaller target group for the candidate-harmful write-complexity shift.",
            "Add a small number of no-write and few-action retail tasks to preserve coverage for inconclusive shifts with smaller source groups.",
            "Reassess confidence intervals after Stage 1 before expanding.",
        ],
        "proposed_next_batch": {
            "strategy": "Stage 1 small targeted batch, then rerun uncertainty analysis before any larger collection.",
            "stage_1_batch_size": task_pool_audit["stage1_batch_size"],
            "stage_1_composition": {
                "retail_two_plus_expected_writes": sum(
                    task["expected_write_action_count"] >= 2 for task in task_pool_audit["stage1_recommended_tasks"]
                ),
                "retail_no_expected_writes": sum(
                    task["expected_write_action_count"] == 0 for task in task_pool_audit["stage1_recommended_tasks"]
                ),
                "retail_few_expected_actions": sum(
                    task["expected_action_count"] <= 1 for task in task_pool_audit["stage1_recommended_tasks"]
                ),
                "airline": 0,
                "telecom": 0,
                "banking_knowledge": 0,
            },
            "stage_1_tasks": task_pool_audit["stage1_recommended_tasks"],
            "stage_2_rule": (
                "Expand only if Stage 1 leaves decision-relevant confidence intervals inconclusive; "
                "do not automatically collect the full calculated sample size."
            ),
        },
        "cost_and_runtime_constraints": {"telecom_warning": TELECOM_WARNING},
        "limitations": [
            PLANNING_LIMITATION,
            "Current tau2 evidence has only 93 retained real task-level outcomes.",
            "Precision and power estimates assume independent Bernoulli outcomes and stable success rates.",
            "Equivalence estimates are CI-screening approximations where available, not rigorous TOST power guarantees.",
            "Observed trajectory-length and tool-call-count groups cannot be assigned to unused tasks before running them.",
            "No task is selected based on observed success or failure.",
        ],
    }


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.{digits}f}"


def table_current_evidence(shifts: list[dict[str, Any]]) -> str:
    lines = [
        "| Shift | Source n | Target n | Source rate | Target rate | Delta_y | 95% CI | CI width | d=0.05 | d=0.10 | d=0.15 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for row in shifts:
        ci = row["delta_y_ci_95"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["shift_id"],
                    str(row["source_n"]),
                    str(row["target_n"]),
                    fmt_float(row["source_success_rate"]),
                    fmt_float(row["target_success_rate"]),
                    fmt_float(row["delta_y"]),
                    f"[{fmt_float(ci[0])}, {fmt_float(ci[1])}]",
                    fmt_float(row["delta_y_ci_width"]),
                    row["classification_by_threshold"]["0.05"],
                    row["classification_by_threshold"]["0.10"],
                    row["classification_by_threshold"]["0.15"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def planning_table(shifts: list[dict[str, Any]], key: str, scenarios: tuple[float, ...], label: str) -> str:
    lines = [
        f"| Shift | {label} | Required final n/group | Add source | Add target | Add total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in shifts:
        for scenario in scenarios:
            estimate = row[key][f"{scenario:.2f}"]
            if estimate.get("available"):
                values = [
                    row["shift_id"],
                    f"{scenario:.2f}",
                    str(estimate["required_final_n_per_group"]),
                    str(estimate["additional_source_n"]),
                    str(estimate["additional_target_n"]),
                    str(estimate["additional_total_n"]),
                ]
            else:
                values = [row["shift_id"], f"{scenario:.2f}", "unavailable", "unavailable", "unavailable", "unavailable"]
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def audit_lines(plan: dict[str, Any]) -> str:
    audit = plan["task_pool_audit"]
    lines = []
    for domain, info in audit["domains"].items():
        lines.append(
            f"- `{domain}`: {info['local_task_count']} local tasks; "
            f"{info['retained_outcome_count']} retained outcomes; "
            f"{info['unused_by_retained_outcome_count']} without retained outcomes."
        )
    lines.append(
        f"- Airline all local tasks previously sampled: {audit['airline_all_tasks_previously_sampled']}."
    )
    lines.append(
        f"- Retail unused by retained outcome: {audit['domains']['retail']['unused_by_retained_outcome_count']}; "
        f"retail not previously attempted in the 2026-07-14 run: "
        f"{audit['domains']['retail']['not_previously_attempted_count']}."
    )
    for shift_id, shift_audit in audit["shift_audits"].items():
        counts = shift_audit["unused_retail_group_counts"]
        if counts.get("available"):
            detail = f"unused retail source={counts['source']}, target={counts['target']}, unknown={counts['unknown']}"
        else:
            detail = counts["reason"]
        lines.append(
            f"- `{shift_id}`: smaller group `{shift_audit['smaller_group']}`; {detail}; "
            f"repeated trials required now: {shift_audit['repeated_trials_required']}."
        )
    return "\n".join(lines)


def task_table(tasks: list[dict[str, Any]]) -> str:
    lines = [
        "| Domain | Task ID | Expected actions | Expected writes | Selection reason |",
        "|---|---:|---:|---:|---|",
    ]
    for task in tasks:
        lines.append(
            "| "
            + " | ".join(
                [
                    task["domain"],
                    task["task_id"],
                    str(task["expected_action_count"]),
                    str(task["expected_write_action_count"]),
                    "; ".join(task["selection_reasons"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(plan: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shifts = plan["shifts"]
    candidate_harmful = plan["current_evidence_summary"]["candidate_harmful_shifts"]
    inconclusive = plan["current_evidence_summary"]["inconclusive_shifts"]
    report = f"""# Tau2 Additional Sampling Plan

## Current evidence

The current tau2 analysis contains {plan['current_evidence_summary']['eligible_tau2_shift_count']} eligible shifts and {plan['current_evidence_summary']['real_task_level_outcome_count']} retained real task-level outcomes. API-Bank is excluded from this plan because its negative labels are synthetic API-call corruptions.

{table_current_evidence(shifts)}

## Candidate-harmful shift

{', '.join(f'`{shift}`' for shift in candidate_harmful) if candidate_harmful else 'None.'}

This shift should be prioritized for additional real tau2 outcomes, but the next collection should still be staged rather than collecting the full calculated sample size in one pass.

## Inconclusive shifts

{', '.join(f'`{shift}`' for shift in inconclusive) if inconclusive else 'None.'}

## Precision-based sample estimates

{PLANNING_LIMITATION}

{planning_table(shifts, 'precision_planning', PRECISION_HALF_WIDTHS, 'CI half-width')}

## Power-based sample estimates

Power estimates use two-sided alpha = 0.05 and power = 0.80.

{planning_table(shifts, 'power_planning', POWER_EFFECT_SIZES, 'Effect size')}

## Available task-pool audit

{audit_lines(plan)}

Task IDs can be selected without using outcome `y` for the proposed Stage 1 batch. Observed trajectory-length and observed tool-call-count membership is unavailable for unused tasks until new runs produce trajectories.

## Recommended collection priority

{chr(10).join(f'- {item}' for item in plan['recommended_collection_priority'])}

## Proposed next batch

Stage 1: collect {plan['proposed_next_batch']['stage_1_batch_size']} unused retail tasks, then rerun the tau2 uncertainty analysis and reassess confidence intervals. Stage 2 should expand only if the Stage 1 evidence remains decision-relevant and inconclusive.

{task_table(plan['proposed_next_batch']['stage_1_tasks'])}

## Cost and runtime constraints

{TELECOM_WARNING}

## Limitations

{chr(10).join(f'- {item}' for item in plan['limitations'])}
"""
    path.write_text(report, encoding="utf-8")


def build_outputs(
    *,
    uncertainty_jsonl: Path = DEFAULT_UNCERTAINTY_JSONL,
    uncertainty_summary_json: Path = DEFAULT_UNCERTAINTY_SUMMARY_JSON,
    inventory_jsonl: Path = DEFAULT_INVENTORY_JSONL,
    unified_tau2_jsonl: Path = DEFAULT_UNIFIED_TAU2_JSONL,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    report_path: Path = DEFAULT_REPORT,
    write_outputs: bool = True,
) -> dict[str, Any]:
    plan = build_plan(
        uncertainty_jsonl=uncertainty_jsonl,
        uncertainty_summary_json=uncertainty_summary_json,
        inventory_jsonl=inventory_jsonl,
        unified_tau2_jsonl=unified_tau2_jsonl,
    )
    if write_outputs:
        write_json(plan, output_json)
        write_report(plan, report_path)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertainty-jsonl", type=Path, default=DEFAULT_UNCERTAINTY_JSONL)
    parser.add_argument("--uncertainty-summary-json", type=Path, default=DEFAULT_UNCERTAINTY_SUMMARY_JSON)
    parser.add_argument("--inventory-jsonl", type=Path, default=DEFAULT_INVENTORY_JSONL)
    parser.add_argument("--unified-tau2-jsonl", type=Path, default=DEFAULT_UNIFIED_TAU2_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_outputs(
        uncertainty_jsonl=args.uncertainty_jsonl,
        uncertainty_summary_json=args.uncertainty_summary_json,
        inventory_jsonl=args.inventory_jsonl,
        unified_tau2_jsonl=args.unified_tau2_jsonl,
        output_json=args.output_json,
        report_path=args.report_path,
    )
    print(f"wrote tau2 additional sampling plan for {len(plan['shifts'])} shifts")
    print(f"stage 1 tasks: {plan['proposed_next_batch']['stage_1_batch_size']}")
    print(TELECOM_WARNING)


if __name__ == "__main__":
    main()

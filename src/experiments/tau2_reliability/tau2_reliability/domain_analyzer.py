"""Domain-level aggregation and dashboard JSON export.

Orchestrates: Results → conversation_analyzer → task_analyzer → JSON export.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger
from tau2.data_model.simulation import Results, TerminationReason

from tau2_reliability.conversation_analyzer import analyze_conversation
from tau2_reliability.extract import build_tool_type_map
from tau2_reliability.task_analyzer import analyze_task


def analyze_domain(
    results_path: Path | str,
    output_dir: Optional[Path | str] = None,
) -> dict:
    """Full reliability analysis: Results → per-conversation + per-task + domain summary.

    Args:
        results_path: Path to tau2 results.json.
        output_dir: If provided, writes reliability_data.json here.

    Returns:
        Complete analysis dict ready for dashboard consumption.
    """
    results_path = Path(results_path)
    results = Results.load(results_path)

    # Extract metadata
    domain = ""
    agent_model = ""
    if results.info:
        if hasattr(results.info, "environment_info") and results.info.environment_info:
            domain = getattr(results.info.environment_info, "domain_name", "") or ""
        if hasattr(results.info, "agent_info") and results.info.agent_info:
            agent_model = getattr(results.info.agent_info, "llm", "") or ""

    # Fallback: parse from filename
    if not domain or not agent_model:
        fname = results_path.stem
        parts = fname.split("_")
        for known in ["airline", "retail", "telecom", "banking"]:
            if known in parts:
                idx = parts.index(known)
                agent_model = agent_model or "_".join(parts[:idx])
                domain = domain or known
                break

    # Build tool type map from domain
    tool_type_map = build_tool_type_map(domain) if domain else {}

    # Load policy graph
    policy_graph = None
    try:
        from tau2_reliability.analysis.policy_adherence import get_policy_graph
        policy_graph = get_policy_graph(domain)
    except Exception:
        pass

    # Phase 1: Analyze every conversation
    logger.info(f"Analyzing {len(results.simulations)} conversations...")
    conversations = []
    for sim in results.simulations:
        if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
            continue
        conv = analyze_conversation(sim, tool_type_map=tool_type_map, policy_graph=policy_graph)
        conversations.append(conv)

    logger.info(f"Analyzed {len(conversations)} conversations")

    # Phase 2: Group by task and analyze consistency
    task_groups: dict[str, list[dict]] = defaultdict(list)
    for conv in conversations:
        task_groups[conv["task_id"]].append(conv)

    tasks = {}
    for task_id in sorted(task_groups.keys()):
        tasks[task_id] = analyze_task(task_id, task_groups[task_id])

    # Phase 3: Domain-level aggregation (with bootstrap SEs)
    domain_summary = _aggregate_domain(conversations, tasks, domain, agent_model, results)

    # Phase 4: Bootstrap standard errors (200 resamples, task-level)
    bootstrap_se = _compute_bootstrap_se(tasks)
    domain_summary["bootstrap_se"] = bootstrap_se

    # Compose output
    output = {
        "conversations": conversations,
        "tasks": tasks,
        "domain_summary": domain_summary,
    }

    # Write if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write full data (without messages for smaller file)
        slim = _slim_for_export(output)
        data_path = output_dir / "reliability_data.json"
        data_path.write_text(json.dumps(slim, indent=2, default=str))
        logger.info(f"Dashboard data: {data_path}")

        # Write full data with messages (for conversation audit)
        full_path = output_dir / "reliability_full.json"
        full_path.write_text(json.dumps(output, indent=2, default=str))
        logger.info(f"Full data (with messages): {full_path}")

    return output


def analyze_multiple(
    results_paths: list[Path | str],
    output_dir: Optional[Path | str] = None,
) -> dict:
    """Analyze multiple results files and combine into a single dashboard JSON.

    Each results file becomes one 'run' in the combined output.
    Combines all runs into a single dashboard view.
    """
    runs = []
    all_conversations = []
    for rp in results_paths:
        rp = Path(rp)
        if not rp.exists():
            logger.warning(f"Skipping {rp} — file not found")
            continue
        logger.info(f"Analyzing: {rp.name}")
        output = analyze_domain(rp)
        run = {
            "id": rp.stem,
            "domain_summary": output["domain_summary"],
            "tasks": output["tasks"],
            "conversations": [
                {k: v for k, v in c.items() if k != "messages"}
                for c in output["conversations"]
            ],
        }
        runs.append(run)
        # Keep full conversations for audit (with messages)
        for c in output["conversations"]:
            c["_run_id"] = rp.stem
        all_conversations.extend(output["conversations"])

    combined = {
        "runs": runs,
        "num_runs": len(runs),
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Slim version (no messages) for dashboard
        data_path = output_dir / "reliability_data.json"
        data_path.write_text(json.dumps(combined, indent=2, default=str))
        logger.info(f"Dashboard data ({len(runs)} runs): {data_path}")

        # Full version with messages for conversation audit
        full = {
            "runs": runs,
            "num_runs": len(runs),
            "all_conversations": all_conversations,
        }
        full_path = output_dir / "reliability_full.json"
        full_path.write_text(json.dumps(full, indent=2, default=str))
        logger.info(f"Full data (with messages): {full_path}")

    return combined


def _aggregate_domain(
    conversations: list[dict],
    tasks: dict[str, dict],
    domain: str,
    agent_model: str,
    results: Results,
) -> dict:
    """Compute domain-level summary metrics."""
    total = len(conversations)
    passed = sum(1 for c in conversations if c["outcome"] == "pass")
    accuracy = passed / total if total > 0 else 0.0

    # Consistency aggregates (average across tasks, skip tasks with no consistency data)
    outcome_scores = [t["consistency"]["outcome"] for t in tasks.values() if "consistency" in t]
    action_scores = [t["consistency"]["actions"] for t in tasks.values() if "consistency" in t]
    sequence_scores = [t["consistency"]["sequence"] for t in tasks.values() if "consistency" in t]
    resource_scores = [t["consistency"]["resources"] for t in tasks.values() if "consistency" in t]

    outcome_con = _safe_mean(outcome_scores)
    action_con = _safe_mean(action_scores)
    sequence_con = _safe_mean(sequence_scores)
    resource_con = _safe_mean(resource_scores)

    # Policy adherence aggregate
    policy_scores = [
        c["policy_adherence"]["score"]
        for c in conversations
        if c["policy_adherence"].get("available") and c["policy_adherence"]["score"] is not None
    ]
    policy_compliance = _safe_mean(policy_scores) if policy_scores else None

    # Task classification counts
    class_counts = {"stable_pass": 0, "stable_fail": 0, "bimodal": 0, "fragile": 0, "moderate": 0}
    for t in tasks.values():
        cls = t.get("class", "moderate")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    bimodal_tasks = [tid for tid, t in tasks.items() if t.get("class") == "bimodal"]

    # Efficiency aggregates
    total_redundant = sum(c["efficiency"]["redundant_calls"] for c in conversations)
    total_errors = sum(c["efficiency"]["tool_errors"] for c in conversations)
    rbw_rates = [c["efficiency"]["read_before_write_rate"] for c in conversations if c["efficiency"]["read_before_write_rate"] is not None]

    # Abstention aggregates
    abstention_count = sum(1 for c in conversations if c.get("abstention", {}).get("abstained", False))
    abstention_rate = abstention_count / total if total > 0 else 0.0

    num_trials = max((t["num_trials"] for t in tasks.values()), default=1)

    return {
        "model": agent_model,
        "domain": domain,
        "accuracy": accuracy,
        "num_tasks": len(tasks),
        "num_trials": num_trials,
        "total_conversations": total,
        "dimensions": {
            "outcome_consistency": {
                "score": outcome_con,
                "label": "Outcome Consistency",
                "question": "Same pass/fail each run?",
            },
            "action_consistency": {
                "score": action_con,
                "label": "Action Consistency",
                "question": "Same action types across runs?",
            },
            "sequence_consistency": {
                "score": sequence_con,
                "label": "Sequence Consistency",
                "question": "Same action ordering?",
            },
            "cost_stability": {
                "score": resource_con,
                "label": "Cost Stability",
                "question": "Stable cost and time?",
            },
            "policy_compliance": {
                "score": policy_compliance,
                "label": "Policy Compliance",
                "question": "Follows the right workflow?",
            },
        },
        "task_classes": class_counts,
        "bimodal_tasks": bimodal_tasks,
        "efficiency": {
            "total_redundant_calls": total_redundant,
            "total_tool_errors": total_errors,
            "avg_read_before_write": _safe_mean(rbw_rates) if rbw_rates else None,
        },
        "abstention": {
            "rate": abstention_rate,
            "count": abstention_count,
            "total": total,
        },
    }


def _slim_for_export(data: dict) -> dict:
    """Remove full message traces from conversations for smaller export file.
    The full messages are in reliability_full.json for the audit view."""
    slim = {
        "domain_summary": data["domain_summary"],
        "tasks": data["tasks"],
        "conversations": [],
    }
    for conv in data["conversations"]:
        c = {k: v for k, v in conv.items() if k != "messages"}
        c["message_count"] = len(conv.get("messages", []))
        slim["conversations"].append(c)
    return slim


def _compute_bootstrap_se(tasks: dict, n_resamples: int = 200, seed: int = 42) -> dict:
    """Bootstrap standard errors by resampling tasks."""
    import numpy as np

    task_list = [t for t in tasks.values() if "consistency" in t]
    if len(task_list) < 2:
        return {}

    rng = np.random.default_rng(seed)
    n = len(task_list)

    samples = {"outcome": [], "actions": [], "sequence": [], "resources": []}
    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        resampled = [task_list[i] for i in indices]
        for key in samples:
            vals = [t["consistency"].get(key, 0) for t in resampled if "consistency" in t]
            samples[key].append(sum(vals) / len(vals) if vals else 0)

    return {
        f"{key}_se": float(np.std(vals, ddof=1))
        for key, vals in samples.items()
    }


def _safe_mean(values: list[float]) -> float:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    return sum(valid) / len(valid) if valid else 0.0

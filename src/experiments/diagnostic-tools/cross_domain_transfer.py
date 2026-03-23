#!/usr/bin/env python3
"""
Cross-Domain Transfer Test for τ²-bench

Compares agent performance across multiple domains to measure
generalization capability. Identifies domain-specific weaknesses
and computes a generalization score.

Usage:
    python cross_domain_transfer.py --results airline.json retail.json telecom.json
    python cross_domain_transfer.py --results airline.json retail.json --domains airline retail
    python cross_domain_transfer.py --results *.json --output json

License: MIT
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


def infer_domain(filename: str) -> str:
    """Infer domain name from filename."""
    name = Path(filename).stem.lower()
    for domain in ["airline", "retail", "telecom", "medical", "banking", "mock"]:
        if domain in name:
            return domain
    # Fallback: use filename without extension
    return re.sub(r"[_\-\s]+results?$", "", name) or name


def compute_domain_stats(results: list[dict]) -> dict[str, Any]:
    """Compute statistics for a single domain's results."""
    total = len(results)
    passed = sum(1 for r in results if r.get("reward", 0) >= 1.0)
    failed = total - passed

    rewards = [r.get("reward", 0) for r in results]
    avg_reward = statistics.mean(rewards) if rewards else 0.0

    failed_ids = [
        r.get("task_id", r.get("id", "?"))
        for r in results if r.get("reward", 0) < 1.0
    ]

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(total, 1) * 100, 1),
        "avg_reward": round(avg_reward, 4),
        "failed_task_ids": failed_ids[:10],
    }


def analyze_transfer(domain_results: dict[str, list[dict]]) -> dict[str, Any]:
    """Analyze cross-domain transfer performance."""
    domain_stats = {}
    for domain, results in domain_results.items():
        domain_stats[domain] = compute_domain_stats(results)

    pass_rates = [s["pass_rate"] for s in domain_stats.values()]
    generalization_score = round(statistics.mean(pass_rates), 1) if pass_rates else 0.0
    consistency = round(statistics.stdev(pass_rates), 1) if len(pass_rates) > 1 else 0.0

    # Compute transfer gaps (pairwise)
    domains = sorted(domain_stats.keys())
    transfer_gaps = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            d1, d2 = domains[i], domains[j]
            gap = abs(domain_stats[d1]["pass_rate"] - domain_stats[d2]["pass_rate"])
            transfer_gaps.append({
                "domain_a": d1,
                "domain_b": d2,
                "rate_a": domain_stats[d1]["pass_rate"],
                "rate_b": domain_stats[d2]["pass_rate"],
                "gap": round(gap, 1),
            })

    # Sort by gap descending
    transfer_gaps.sort(key=lambda x: -x["gap"])

    # Recommendations
    recommendations = []
    weakest = min(domain_stats, key=lambda d: domain_stats[d]["pass_rate"])
    strongest = max(domain_stats, key=lambda d: domain_stats[d]["pass_rate"])

    if domain_stats[weakest]["pass_rate"] < 90:
        recommendations.append(
            f"Focus improvement on {weakest} domain "
            f"(currently {domain_stats[weakest]['pass_rate']}%)"
        )
    if consistency > 10:
        recommendations.append(
            f"High variance ({consistency}% stdev) across domains. "
            f"Consider domain-agnostic improvements."
        )
    if generalization_score >= 95:
        recommendations.append(
            "Excellent generalization across all domains."
        )

    return {
        "domains": domain_stats,
        "generalization_score": generalization_score,
        "consistency_stdev": consistency,
        "transfer_gaps": transfer_gaps,
        "strongest_domain": strongest,
        "weakest_domain": weakest,
        "recommendations": recommendations,
    }


def format_text_report(analysis: dict[str, Any]) -> str:
    """Format analysis as human-readable text report."""
    lines = [
        "=" * 70,
        "  τ²-bench Cross-Domain Transfer Analysis",
        "=" * 70,
        "",
        f"  {'Domain':<15} {'Tasks':>6} {'Pass':>6} {'Fail':>6} {'Rate':>7} {'Avg Reward':>11}",
        "  " + "-" * 65,
    ]

    for domain in sorted(analysis["domains"]):
        s = analysis["domains"][domain]
        lines.append(
            f"  {domain:<15} {s['total']:>6} {s['passed']:>6} {s['failed']:>6} "
            f"{s['pass_rate']:>6.1f}% {s['avg_reward']:>10.4f}"
        )

    lines.extend([
        "  " + "-" * 65,
        "",
        f"  Generalization Score: {analysis['generalization_score']}%",
        f"  Consistency (stdev):  {analysis['consistency_stdev']}%",
        f"  Strongest domain:     {analysis['strongest_domain']}",
        f"  Weakest domain:       {analysis['weakest_domain']}",
        "",
    ])

    if analysis["transfer_gaps"]:
        lines.append("  Transfer Gaps (pairwise):")
        for g in analysis["transfer_gaps"][:5]:
            lines.append(
                f"    {g['domain_a']} ({g['rate_a']}%) ↔ "
                f"{g['domain_b']} ({g['rate_b']}%) → gap: {g['gap']}%"
            )
        lines.append("")

    if analysis["recommendations"]:
        lines.append("  Recommendations:")
        for r in analysis["recommendations"]:
            lines.append(f"    → {r}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain transfer analysis for τ²-bench"
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="Paths to results JSON files (one per domain)"
    )
    parser.add_argument(
        "--domains", nargs="*",
        help="Domain names (inferred from filenames if omitted)"
    )
    parser.add_argument(
        "--output", choices=["text", "json"], default="text",
        help="Output format (default: text)"
    )
    args = parser.parse_args()

    domain_results = {}
    for i, results_path in enumerate(args.results):
        path = Path(results_path)
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

        domain = args.domains[i] if args.domains and i < len(args.domains) else infer_domain(results_path)

        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("results", data.get("tasks", []))

        domain_results[domain] = data

    analysis = analyze_transfer(domain_results)

    if args.output == "json":
        print(json.dumps(analysis, indent=2))
    else:
        print(format_text_report(analysis))


if __name__ == "__main__":
    main()

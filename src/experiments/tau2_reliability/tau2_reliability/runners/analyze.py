"""Post-hoc reliability analysis of existing tau2-bench Results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from tau2.utils.utils import get_now

from tau2_reliability.extract import load_and_extract
from tau2_reliability.metrics.consistency import compute_all_consistency
from tau2_reliability.metrics.predictability import (
    compute_calibration_bins,
    compute_p_auroc,
    compute_p_brier,
    compute_p_cal,
)
from tau2_reliability.models import (
    PredictabilityMetrics,
    ReliabilityReport,
    TaskTrialData,
)


def analyze_results(
    results_path: Path | str,
    output_dir: Optional[Path | str] = None,
    tool_type_map: Optional[dict[str, str]] = None,
    n_bootstrap: int = 200,
    bootstrap_seed: int = 42,
) -> ReliabilityReport:
    """Run complete reliability analysis on existing Results.

    Args:
        results_path: Path to tau2-bench results JSON or directory.
        output_dir: If provided, write report + plots here.
        tool_type_map: Optional tool_name -> READ/WRITE mapping.
        n_bootstrap: Number of bootstrap resamples for SE computation.
        bootstrap_seed: Random seed for bootstrap reproducibility.

    Returns:
        ReliabilityReport with all computed metrics.
    """
    results_path = Path(results_path)
    results, task_data = load_and_extract(results_path, tool_type_map=tool_type_map)

    if not task_data:
        logger.error("No valid task data extracted")
        return ReliabilityReport(timestamp=get_now(), source_path=str(results_path))

    # Metadata
    num_tasks = len(task_data)
    num_trials = max(td.num_trials for td in task_data)
    all_outcomes = [o for td in task_data for o in td.outcomes]
    accuracy = sum(all_outcomes) / len(all_outcomes) if all_outcomes else 0.0

    domain = ""
    agent_model = ""
    if results.info:
        # Try structured fields first
        if hasattr(results.info, "environment_info") and results.info.environment_info:
            domain = getattr(results.info.environment_info, "domain_name", "") or ""
        if hasattr(results.info, "agent_info") and results.info.agent_info:
            agent_model = getattr(results.info.agent_info, "llm", "") or ""

    # Fallback: parse from filename (e.g., "claude-3-7-sonnet_airline_default_..._4trials.json")
    if not domain or not agent_model:
        fname = results_path.stem
        parts = fname.split("_")
        if not agent_model and len(parts) >= 1:
            # Agent model is the first part before the domain
            for known_domain in ["airline", "retail", "telecom", "banking"]:
                if known_domain in parts:
                    idx = parts.index(known_domain)
                    agent_model = agent_model or "_".join(parts[:idx])
                    domain = domain or known_domain
                    break

    if num_trials < 2:
        logger.warning(
            "Only 1 trial per task — consistency metrics will be trivial. "
            "Use --num-trials >= 2 for meaningful analysis."
        )

    # --- Consistency ---
    consistency = compute_all_consistency(task_data)

    # --- Predictability (if confidence scores exist) ---
    predictability = None
    r_pred = None
    all_conf = []
    all_out = []
    for td in task_data:
        if td.confidence_scores:
            for conf, out in zip(td.confidence_scores, td.outcomes):
                all_conf.append(conf)
                all_out.append(out)

    if all_conf:
        p_cal = compute_p_cal(all_conf, all_out)
        p_auroc = compute_p_auroc(all_conf, all_out)
        p_brier = compute_p_brier(all_conf, all_out)
        cal_bins = compute_calibration_bins(all_conf, all_out)
        predictability = PredictabilityMetrics(
            p_cal=p_cal, p_auroc=p_auroc, p_brier=p_brier,
            calibration_bins=cal_bins,
        )
        r_pred = p_brier

    # --- Sprint 2: Novel Analysis ---

    # Divergence analysis
    from tau2_reliability.analysis.divergence import compute_all_divergence_profiles
    divergence_profiles = compute_all_divergence_profiles(task_data)

    # Mutation-aware analysis
    from tau2_reliability.analysis.mutation_aware import compute_mutation_analysis
    mutation_analysis = compute_mutation_analysis(task_data, tool_type_map=tool_type_map)

    # Task taxonomy
    from tau2_reliability.analysis.task_taxonomy import (
        classify_tasks,
        compute_taxonomy_summary,
    )
    classifications = classify_tasks(task_data, consistency)
    task_taxonomy = compute_taxonomy_summary(classifications, task_data)

    logger.info(
        f"Task taxonomy: {task_taxonomy.counts}, "
        f"bimodal={len(task_taxonomy.bimodal_tasks)}"
    )
    if mutation_analysis.write_action_fraction > 0:
        logger.info(
            f"Mutation analysis: write_fraction={mutation_analysis.write_action_fraction:.1%}, "
            f"risk_ratio={mutation_analysis.mutation_risk_ratio or 'N/A'}, "
            f"verification_gap={mutation_analysis.verification_gap or 'N/A'}"
        )

    # --- Sprint 3: Policy Adherence ---
    from tau2_reliability.analysis.policy_adherence import compute_policy_adherence
    policy_adherence = compute_policy_adherence(task_data, domain) if domain else []
    if policy_adherence:
        avg_adherence = sum(p.adherence_score for p in policy_adherence) / len(policy_adherence)
        total_violations = sum(len(p.violations) for p in policy_adherence)
        logger.info(
            f"Policy adherence: avg_score={avg_adherence:.3f}, "
            f"total_violations={total_violations} across {len(policy_adherence)} tasks"
        )

    # --- Bootstrap SE ---
    bootstrap_se = _compute_bootstrap_se(
        task_data, n_resamples=n_bootstrap, seed=bootstrap_seed
    )

    # --- Aggregate ---
    r_con = consistency.r_con
    dims = [r_con, r_pred]
    available = [d for d in dims if d is not None and not math.isnan(d)]
    r_overall = sum(available) / len(available) if available else None

    report = ReliabilityReport(
        timestamp=get_now(),
        source_path=str(results_path),
        domain=domain,
        agent_model=agent_model,
        num_tasks=num_tasks,
        num_trials=num_trials,
        accuracy=accuracy,
        consistency=consistency,
        predictability=predictability,
        r_con=r_con,
        r_pred=r_pred,
        r_overall=r_overall,
        divergence_profiles=divergence_profiles,
        mutation_analysis=mutation_analysis,
        task_taxonomy=task_taxonomy,
        policy_adherence=policy_adherence if policy_adherence else None,
        bootstrap_se=bootstrap_se,
    )

    # --- Output ---
    if output_dir:
        _write_outputs(report, task_data, Path(output_dir))

    return report


def _compute_bootstrap_se(
    task_data: list[TaskTrialData],
    n_resamples: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap SE by resampling tasks (not trials)."""
    if len(task_data) < 2:
        return {}

    rng = np.random.default_rng(seed)
    n = len(task_data)

    c_out_samples = []
    c_traj_d_samples = []
    c_traj_s_samples = []
    c_res_samples = []
    r_con_samples = []

    from tau2_reliability.metrics.consistency import (
        compute_c_out,
        compute_c_res,
        compute_c_traj_d,
        compute_c_traj_s,
        compute_r_con,
    )

    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        resampled = [task_data[i] for i in indices]

        c_out, _ = compute_c_out(resampled)
        c_traj_d, _ = compute_c_traj_d(resampled)
        c_traj_s, _ = compute_c_traj_s(resampled)
        c_res, _ = compute_c_res(resampled)
        r_con = compute_r_con(c_out, c_traj_d, c_traj_s, c_res)

        c_out_samples.append(c_out)
        c_traj_d_samples.append(c_traj_d)
        c_traj_s_samples.append(c_traj_s)
        c_res_samples.append(c_res)
        r_con_samples.append(r_con)

    return {
        "c_out": float(np.std(c_out_samples, ddof=1)),
        "c_traj_d": float(np.std(c_traj_d_samples, ddof=1)),
        "c_traj_s": float(np.std(c_traj_s_samples, ddof=1)),
        "c_res": float(np.std(c_res_samples, ddof=1)),
        "r_con": float(np.std(r_con_samples, ddof=1)),
    }


def _write_outputs(
    report: ReliabilityReport,
    task_data: list[TaskTrialData],
    output_dir: Path,
) -> None:
    """Write report, CSV, and plots to output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    report_path = output_dir / "reliability_report.json"
    report_path.write_text(report.model_dump_json(indent=2))
    logger.info(f"Report written to {report_path}")

    # Markdown report
    from tau2_reliability.report import generate_markdown_report
    md_path = output_dir / "reliability_report.md"
    md_path.write_text(generate_markdown_report(report))
    logger.info(f"Markdown report written to {md_path}")

    # CSV per-task
    from tau2_reliability.report import generate_csv
    csv_path = output_dir / "per_task_metrics.csv"
    generate_csv(report, csv_path)
    logger.info(f"CSV written to {csv_path}")

    # Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _generate_plots(report, task_data, plots_dir)


def _generate_plots(
    report: ReliabilityReport,
    task_data: list[TaskTrialData],
    plots_dir: Path,
) -> None:
    """Generate all Sprint 1 visualizations."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for file output
    import matplotlib.pyplot as plt

    from tau2_reliability.visualization.dashboard import (
        plot_consistency_detail,
        plot_per_task_heatmap,
        plot_reliability_dashboard,
    )

    try:
        fig = plot_reliability_dashboard(report)
        fig.savefig(plots_dir / "reliability_dashboard.png")
        plt.close(fig)
        logger.info("Generated reliability_dashboard.png")
    except Exception as e:
        logger.warning(f"Failed to generate dashboard: {e}")

    try:
        fig = plot_consistency_detail(report)
        fig.savefig(plots_dir / "consistency_detail.png")
        plt.close(fig)
        logger.info("Generated consistency_detail.png")
    except Exception as e:
        logger.warning(f"Failed to generate consistency detail: {e}")

    try:
        fig = plot_per_task_heatmap(report)
        fig.savefig(plots_dir / "per_task_heatmap.png")
        plt.close(fig)
        logger.info("Generated per_task_heatmap.png")
    except Exception as e:
        logger.warning(f"Failed to generate per-task heatmap: {e}")

    # Session-level plots for bimodal tasks (c_out < 0.5)
    from tau2_reliability.visualization.session import (
        plot_action_sequence_alignment,
        plot_trace_comparison,
    )

    td_map = {td.task_id: td for td in task_data}
    if report.consistency and report.consistency.per_task:
        for tid, metrics in report.consistency.per_task.items():
            if metrics.get("c_out", 1.0) < 0.5 and tid in td_map:
                try:
                    fig = plot_action_sequence_alignment(td_map[tid])
                    fig.savefig(plots_dir / f"alignment_{tid}.png")
                    plt.close(fig)
                except Exception:
                    pass
                try:
                    fig = plot_trace_comparison(td_map[tid])
                    fig.savefig(plots_dir / f"comparison_{tid}.png")
                    plt.close(fig)
                except Exception:
                    pass

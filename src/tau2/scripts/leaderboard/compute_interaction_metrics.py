"""
Compute voice interaction metrics for leaderboard submissions.

Takes one or more voice experiment directories (results.json + simulations/)
— or any directory containing them, e.g. a submission's ``trajectories/`` dir —
and produces the ``interaction_metrics`` JSON block used by the leaderboard:
per-domain metric panels plus an all-domain average.

Usage:
    tau2 submit interaction-metrics <paths...> [--output metrics.json]
"""

import json
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.table import Table

from tau2.data_model.simulation import Results
from tau2.metrics.voice_interaction_metrics import (
    INTERACTION_METRICS_VERSION,
    PANEL_METRIC_FIELDS,
    InteractionMetricsConfig,
    NoVoiceTicksError,
    aggregate_domain_metrics,
    compute_interaction_metrics_for_experiment,
)

console = Console()


def discover_experiment_paths(input_paths: list[str]) -> list[Path]:
    """
    Discover voice experiment directories from a list of input paths.

    Each input may be a results.json file, an experiment directory containing
    one, or a parent directory (e.g. a submission's trajectories/ dir) that is
    searched recursively for results.json files.

    Returns:
        Sorted, de-duplicated list of experiment directories.
    """
    experiment_dirs: set[Path] = set()
    for raw_path in input_paths:
        path = Path(raw_path)
        if path.is_file() and path.name == "results.json":
            experiment_dirs.add(path.parent.resolve())
        elif path.is_dir():
            if (path / "results.json").exists():
                experiment_dirs.add(path.resolve())
            else:
                found = list(path.rglob("results.json"))
                if not found:
                    raise FileNotFoundError(f"No results.json found under {path}")
                for results_file in found:
                    experiment_dirs.add(results_file.parent.resolve())
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

    return sorted(experiment_dirs)


def resolve_experiment_config(
    results: Results,
    config: Optional[InteractionMetricsConfig] = None,
) -> InteractionMetricsConfig:
    """Return the config actually used for an experiment: the base config with
    tick duration overridden from the experiment's recorded audio config."""
    config = config or InteractionMetricsConfig()
    audio_config = results.info.audio_native_config
    if audio_config is not None and audio_config.tick_duration_seconds:
        config = InteractionMetricsConfig(
            **{
                **config.as_dict(),
                "tick_duration_sec": audio_config.tick_duration_seconds,
            }
        )
    return config


def config_with_resolved_ticks(
    config: Optional[InteractionMetricsConfig],
    resolved_configs: list[InteractionMetricsConfig],
) -> InteractionMetricsConfig:
    """Return the config to record in the block, with tick_duration_sec set to
    the value the experiments actually used so provenance isn't misstated."""
    config = config or InteractionMetricsConfig()
    ticks = {c.tick_duration_sec for c in resolved_configs}
    if len(ticks) == 1:
        config = InteractionMetricsConfig(
            **{**config.as_dict(), "tick_duration_sec": ticks.pop()}
        )
    elif len(ticks) > 1:
        logger.warning(
            f"Experiments use differing tick durations {sorted(ticks)}; "
            f"recording tick_duration_sec={config.tick_duration_sec} in the "
            "config block, which does not reflect every experiment"
        )
    return config


def compute_metrics_for_loaded_results(
    results: Results,
    config: Optional[InteractionMetricsConfig] = None,
) -> dict:
    """
    Compute the interaction-metric panel for one already-loaded experiment,
    honoring the experiment's recorded tick duration when available.
    """
    return compute_interaction_metrics_for_experiment(
        results, resolve_experiment_config(results, config)
    )


def build_interaction_metrics_block(
    domain_metrics: dict[str, dict],
    config: Optional[InteractionMetricsConfig] = None,
) -> dict:
    """Assemble the ``interaction_metrics`` block from per-domain panels."""
    config = config or InteractionMetricsConfig()
    return {
        "version": INTERACTION_METRICS_VERSION,
        "config": config.as_dict(),
        "domains": dict(sorted(domain_metrics.items())),
        "overall": aggregate_domain_metrics(domain_metrics),
    }


def compute_interaction_metrics_block(
    experiment_paths: list[Path],
    config: Optional[InteractionMetricsConfig] = None,
) -> dict:
    """
    Compute the full ``interaction_metrics`` block for a set of experiments.

    Args:
        experiment_paths: One experiment directory per domain.
        config: Detection windows. Tick duration is taken from each
            experiment's recorded ``audio_native_config`` when available,
            falling back to ``config.tick_duration_sec``.

    Returns:
        Dict with ``version``, ``config``, per-domain metrics under
        ``domains``, and the cross-domain average under ``overall``.

    Raises:
        NoVoiceTicksError: If an experiment has no tick-bearing simulations.
        ValueError: If two experiments cover the same domain.
    """
    domain_metrics: dict[str, dict] = {}
    resolved_configs: list[InteractionMetricsConfig] = []
    for experiment_path in experiment_paths:
        results = Results.load(experiment_path)
        domain = results.info.environment_info.domain_name
        if domain in domain_metrics:
            raise ValueError(
                f"Domain {domain} appears in multiple experiment directories"
            )
        logger.info(f"Computing interaction metrics for {domain} ({experiment_path})")
        resolved = resolve_experiment_config(results, config)
        resolved_configs.append(resolved)
        domain_metrics[domain] = compute_interaction_metrics_for_experiment(
            results, resolved
        )

    return build_interaction_metrics_block(
        domain_metrics, config_with_resolved_ticks(config, resolved_configs)
    )


def _format_cell(metric_name: str, value: Optional[float]) -> str:
    if value is None:
        return "—"
    if metric_name.endswith("latency_mean"):
        return f"{value:.2f}s"
    return f"{value * 100:.1f}%"


def print_interaction_metrics(block: dict) -> None:
    """Pretty-print an interaction_metrics block as a rich table."""
    table = Table(title="Interaction Metrics (τ-voice panel)")
    table.add_column("Domain")
    table.add_column("L_R ↓")
    table.add_column("L_Y ↓")
    table.add_column("R_R ↑")
    table.add_column("R_Y ↑")
    table.add_column("I_A ↓")
    table.add_column("S_BC ↑")
    table.add_column("S_VT ↑")
    table.add_column("S_ND ↑")

    rows = list(block["domains"].items())
    if block.get("overall"):
        rows.append(("All", block["overall"]))

    for domain, metrics in rows:
        table.add_row(
            domain,
            *[_format_cell(name, metrics.get(name)) for name in PANEL_METRIC_FIELDS],
        )

    console.print(table)


def compute_interaction_metrics(
    input_paths: list[str],
    output_path: Optional[str] = None,
) -> dict:
    """
    CLI entry point: discover experiments, compute metrics, print, and
    optionally write the JSON block to a file.
    """
    experiment_paths = discover_experiment_paths(input_paths)
    console.print(
        f"Found {len(experiment_paths)} experiment director"
        f"{'y' if len(experiment_paths) == 1 else 'ies'}"
    )

    try:
        block = compute_interaction_metrics_block(experiment_paths)
    except NoVoiceTicksError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    print_interaction_metrics(block)

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(block, f, indent=2)
            f.write("\n")
        console.print(f"Wrote interaction metrics to {output}")

    return block

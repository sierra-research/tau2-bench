#!/usr/bin/env python3
"""Post-hoc metrics emission from tau2 evaluation JSON files to Datadog.

This script reads completed evaluation files from $TAU2_DATA_DIR/evaluations/
and emits metrics to Datadog via DogStatsD.

Environment Variables:
    TAU2_DATA_DIR: Base data directory. Defaults to "./data".
    DD_DOGSTATSD_HOST: DogStatsD host. Defaults to "localhost".
    DD_DOGSTATSD_PORT: DogStatsD port. Defaults to 8125.
    DD_API_KEY: Required for agentless mode metric submission.
    DD_SITE: Datadog site. Defaults to "datadoghq.com".

Usage:
    # Emit metrics for a specific evaluation
    python emit_metrics.py --evaluation-id eval-1732449600000-a1b2c3

    # Emit metrics for all evaluations
    python emit_metrics.py --all

    # Dry run (show what would be emitted)
    python emit_metrics.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from datadog import DogStatsd


class MetricsEmitter:
    """Emits tau2 evaluation metrics to Datadog via DogStatsD.

    This class handles initialization of the DogStatsD client and provides
    methods to emit various metric types from evaluation data.
    """

    # Success threshold for task pass/fail determination
    SUCCESS_THRESHOLD = 0.7

    def __init__(self, dry_run: bool = False):
        """Initialize the metrics emitter.

        Args:
            dry_run: If True, log metrics instead of sending to Datadog.
        """
        self.dry_run = dry_run
        self._statsd: DogStatsd | None = None

        if not dry_run:
            self._init_statsd()

    def _init_statsd(self) -> None:
        """Initialize the DogStatsD client."""
        try:
            from datadog import DogStatsd

            host = os.getenv("DD_DOGSTATSD_HOST", "localhost")
            port = int(os.getenv("DD_DOGSTATSD_PORT", "8125"))

            self._statsd = DogStatsd(
                host=host,
                port=port,
                constant_tags=["service:tau2-bench-agent"],
            )
            logger.info(f"DogStatsD client initialized: {host}:{port}")

        except ImportError:
            logger.warning("datadog package not installed, metrics disabled")
        except Exception as e:
            logger.error(f"Failed to initialize DogStatsD: {e}")

    def _emit_gauge(self, metric: str, value: float, tags: list[str]) -> None:
        """Emit a gauge metric."""
        if self.dry_run:
            logger.info(f"[DRY RUN] gauge {metric}={value} tags={tags}")
            return

        if self._statsd:
            self._statsd.gauge(metric, value, tags=tags)

    def _emit_count(self, metric: str, value: int, tags: list[str]) -> None:
        """Emit a count metric."""
        if self.dry_run:
            logger.info(f"[DRY RUN] count {metric}={value} tags={tags}")
            return

        if self._statsd:
            self._statsd.increment(metric, value, tags=tags)

    def _emit_histogram(self, metric: str, value: float, tags: list[str]) -> None:
        """Emit a histogram metric."""
        if self.dry_run:
            logger.info(f"[DRY RUN] histogram {metric}={value} tags={tags}")
            return

        if self._statsd:
            self._statsd.histogram(metric, value, tags=tags)

    def emit_task_metrics(
        self,
        task_id: str,
        domain: str,
        evaluation_id: str,
        reward: float,
        duration_seconds: float,
        steps: int,
    ) -> None:
        """Emit metrics for a single task evaluation.

        Args:
            task_id: The task identifier.
            domain: The domain name (airline, retail, telecom, mock).
            evaluation_id: The evaluation identifier.
            reward: Task reward (0.0-1.0).
            duration_seconds: Task execution time in seconds.
            steps: Number of steps taken.
        """
        success = reward >= self.SUCCESS_THRESHOLD
        base_tags = [
            f"task_id:{task_id}",
            f"domain:{domain}",
            f"evaluation_id:{evaluation_id}",
        ]

        # tau2.task.reward - gauge
        self._emit_gauge("tau2.task.reward", reward, base_tags)

        # tau2.task.duration_seconds - histogram
        self._emit_histogram("tau2.task.duration_seconds", duration_seconds, base_tags)

        # tau2.task.steps - gauge
        self._emit_gauge("tau2.task.steps", float(steps), base_tags)

        # tau2.task.success - count with success tag
        success_tags = base_tags + [f"success:{str(success).lower()}"]
        self._emit_count("tau2.task.success", 1, success_tags)

        # tau2.task.total - count for ratio calculations
        self._emit_count("tau2.task.total", 1, [f"domain:{domain}", f"evaluation_id:{evaluation_id}"])

    def emit_tool_metrics(
        self,
        tool_name: str,
        domain: str,
        correct: bool,
        arguments_match: bool,
        requestor: str = "agent",
    ) -> None:
        """Emit metrics for tool invocations.

        Args:
            tool_name: Name of the tool called.
            domain: The domain name.
            correct: Whether the tool call was correct.
            arguments_match: Whether arguments matched expected.
            requestor: Who called the tool (agent/user).
        """
        base_tags = [
            f"tool_name:{tool_name}",
            f"domain:{domain}",
            f"requestor:{requestor}",
        ]

        # tau2.tool.calls - count
        self._emit_count("tau2.tool.calls", 1, base_tags)

        # tau2.tool.correct - count with correct tag
        correct_tags = [f"tool_name:{tool_name}", f"correct:{str(correct).lower()}"]
        self._emit_count("tau2.tool.correct", 1, correct_tags)

        # tau2.tool.arguments_match - count with match tag
        match_tags = [f"tool_name:{tool_name}", f"match:{str(arguments_match).lower()}"]
        self._emit_count("tau2.tool.arguments_match", 1, match_tags)

    def emit_assertion_metrics(
        self,
        assertion_type: str,
        met: bool,
        task_id: str | None = None,
        assertion_text: str | None = None,
    ) -> None:
        """Emit metrics for assertion evaluations.

        Args:
            assertion_type: Type of assertion (db, action, nl, communicate).
            met: Whether the assertion was satisfied.
            task_id: The task identifier (for NL failures).
            assertion_text: The assertion text (for NL failures).
        """
        # tau2.assertion.result - count with type and met tags
        result_tags = [
            f"type:{assertion_type}",
            f"met:{str(met).lower()}",
        ]
        self._emit_count("tau2.assertion.result", 1, result_tags)

        # tau2.assertion.nl_failed - for failed NL assertions
        if assertion_type == "nl" and not met and task_id and assertion_text:
            nl_tags = [
                f"task_id:{task_id}",
                f"assertion_text:{assertion_text[:50]}",  # Truncate for tag
            ]
            self._emit_count("tau2.assertion.nl_failed", 1, nl_tags)

    def emit_termination_metrics(self, reason: str) -> None:
        """Emit metrics for task termination reasons.

        Args:
            reason: Termination reason (user_stop, agent_stop, max_steps, max_errors).
        """
        # tau2.termination - count with reason tag
        self._emit_count("tau2.termination", 1, [f"reason:{reason}"])

    def emit_evaluation_metrics(
        self,
        evaluation_id: str,
        domain: str,
        pass_rate: float,
        avg_reward: float,
        total_tasks: int,
    ) -> None:
        """Emit aggregated metrics for an evaluation run.

        Args:
            evaluation_id: The evaluation identifier.
            domain: The domain name.
            pass_rate: Overall pass rate percentage.
            avg_reward: Average reward across tasks.
            total_tasks: Total number of tasks evaluated.
        """
        base_tags = [
            f"evaluation_id:{evaluation_id}",
            f"domain:{domain}",
        ]

        # tau2.evaluation.pass_rate - gauge
        self._emit_gauge("tau2.evaluation.pass_rate", pass_rate, base_tags)

        # tau2.evaluation.avg_reward - gauge
        self._emit_gauge("tau2.evaluation.avg_reward", avg_reward, base_tags)

        # tau2.evaluation.tasks_total - gauge
        self._emit_gauge("tau2.evaluation.tasks_total", float(total_tasks), base_tags)


def get_data_dir() -> Path:
    """Get the tau2 data directory."""
    return Path(os.getenv("TAU2_DATA_DIR", "./data"))


def get_evaluations_dir() -> Path:
    """Get the evaluations directory."""
    return get_data_dir() / "evaluations"


def list_evaluations() -> list[Path]:
    """List all evaluation JSON files."""
    eval_dir = get_evaluations_dir()
    if not eval_dir.exists():
        return []
    return sorted(eval_dir.glob("*.json"))


def load_evaluation(path: Path) -> dict:
    """Load an evaluation JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_evaluation_id_from_path(path: Path) -> str:
    """Extract evaluation ID from file path."""
    return path.stem


def process_evaluation(emitter: MetricsEmitter, eval_data: dict, evaluation_id: str) -> None:
    """Process a single evaluation and emit all metrics.

    Handles two JSON formats:
    1. EvaluationStore format (from 002-evaluation-store):
       {
         "evaluation_id": "...",
         "domain": "...",
         "results": {
           "simulations": [...],
           "info": {"environment_info": {"domain_name": "..."}}
         }
       }
    2. Direct Results format (legacy tau2 output):
       {
         "simulations": [...],
         "info": {"environment_info": {"domain_name": "..."}}
       }

    Args:
        emitter: The metrics emitter instance.
        eval_data: The evaluation data dictionary.
        evaluation_id: The evaluation identifier.
    """
    # Handle EvaluationStore format (results are nested under "results" key)
    results_data = eval_data.get("results", eval_data)

    # Extract domain - try EvaluationStore format first, then Results format
    domain = eval_data.get("domain")  # EvaluationStore has domain at top level
    if not domain:
        # Fall back to info.environment_info.domain_name from results
        domain = results_data.get("info", {}).get("environment_info", {}).get("domain_name", "unknown")

    simulations = results_data.get("simulations", [])
    if not simulations:
        logger.warning(f"No simulations found in evaluation {evaluation_id}")
        return

    total_reward = 0.0
    successful_tasks = 0
    total_tasks = len(simulations)

    for sim in simulations:
        task_id = sim.get("task_id", "unknown")
        reward_info = sim.get("reward_info", {})
        reward = reward_info.get("reward", 0.0)
        duration = sim.get("duration", 0.0)
        messages = sim.get("messages", [])
        termination_reason = sim.get("termination_reason", "unknown")

        # Emit task metrics
        emitter.emit_task_metrics(
            task_id=task_id,
            domain=domain,
            evaluation_id=evaluation_id,
            reward=reward,
            duration_seconds=duration,
            steps=len(messages),
        )

        # Emit termination metrics
        emitter.emit_termination_metrics(termination_reason)

        # Process action checks for tool metrics
        action_checks = reward_info.get("action_checks", [])
        if action_checks:
            for check in action_checks:
                action = check.get("action", {})
                tool_name = action.get("name", "unknown")
                correct = check.get("action_match", False)
                # Arguments match is not directly available, use action_match as proxy
                emitter.emit_tool_metrics(
                    tool_name=tool_name,
                    domain=domain,
                    correct=correct,
                    arguments_match=correct,
                )

        # Process assertion metrics
        # DB check
        db_check = reward_info.get("db_check")
        if db_check:
            emitter.emit_assertion_metrics(
                assertion_type="db",
                met=db_check.get("db_match", False),
            )

        # NL assertions
        nl_assertions = reward_info.get("nl_assertions", [])
        if nl_assertions:
            for nl in nl_assertions:
                emitter.emit_assertion_metrics(
                    assertion_type="nl",
                    met=nl.get("met", False),
                    task_id=task_id,
                    assertion_text=nl.get("nl_assertion", ""),
                )

        # Communicate checks
        communicate_checks = reward_info.get("communicate_checks", [])
        if communicate_checks:
            for comm in communicate_checks:
                emitter.emit_assertion_metrics(
                    assertion_type="communicate",
                    met=comm.get("met", False),
                )

        # Accumulate for evaluation-level metrics
        total_reward += reward
        if reward >= MetricsEmitter.SUCCESS_THRESHOLD:
            successful_tasks += 1

    # Emit evaluation-level metrics
    avg_reward = total_reward / total_tasks if total_tasks > 0 else 0.0
    pass_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0

    emitter.emit_evaluation_metrics(
        evaluation_id=evaluation_id,
        domain=domain,
        pass_rate=pass_rate,
        avg_reward=avg_reward,
        total_tasks=total_tasks,
    )

    logger.info(
        f"Emitted metrics for {evaluation_id}: "
        f"{total_tasks} tasks, pass_rate={pass_rate:.1f}%, avg_reward={avg_reward:.2f}"
    )


def main() -> int:
    """Main entry point for metrics emission."""
    parser = argparse.ArgumentParser(
        description="Emit tau2 evaluation metrics to Datadog"
    )
    parser.add_argument(
        "--evaluation-id",
        type=str,
        help="Specific evaluation ID to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all evaluations in the data directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be emitted without sending to Datadog",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if not args.evaluation_id and not args.all:
        parser.error("Either --evaluation-id or --all is required")

    emitter = MetricsEmitter(dry_run=args.dry_run)

    if args.all:
        evaluations = list_evaluations()
        if not evaluations:
            logger.warning(f"No evaluations found in {get_evaluations_dir()}")
            return 1

        logger.info(f"Processing {len(evaluations)} evaluations")
        for eval_path in evaluations:
            try:
                eval_data = load_evaluation(eval_path)
                evaluation_id = extract_evaluation_id_from_path(eval_path)
                process_evaluation(emitter, eval_data, evaluation_id)
            except Exception as e:
                logger.error(f"Failed to process {eval_path}: {e}")
                continue

    elif args.evaluation_id:
        eval_path = get_evaluations_dir() / f"{args.evaluation_id}.json"
        if not eval_path.exists():
            logger.error(f"Evaluation not found: {eval_path}")
            return 1

        eval_data = load_evaluation(eval_path)
        process_evaluation(emitter, eval_data, args.evaluation_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())

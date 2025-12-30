#!/usr/bin/env python3
"""Post-hoc metrics emission from tau2 evaluation JSON files to Datadog.

This script reads completed evaluation files from $TAU2_DATA_DIR/evaluations/
and emits metrics to Datadog via HTTP API.

Environment Variables:
    TAU2_DATA_DIR: Base data directory. Defaults to "./data".
    DD_API_KEY: Required for metrics submission.
    DD_SITE: Datadog site. Defaults to "datadoghq.com".

Usage:
    # Emit metrics for a specific evaluation
    python emit_metrics.py --evaluation-id eval-1732449600000-a1b2c3

    # Emit metrics for all evaluations
    python emit_metrics.py --all

    # Dry run (show what would be emitted)
    python emit_metrics.py --all --dry-run

    # Emit and delete files after (prevents double-counting on re-runs)
    python emit_metrics.py --all --clean
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

_project_root = Path(__file__).parent.parent.parent.parent.parent
load_dotenv(_project_root / ".env")

class MetricsEmitter:
    """Emits tau2 evaluation metrics to Datadog via HTTP API.

    This class handles initialization of the metrics client and provides
    methods to emit various metric types from evaluation data.

    Requires DD_API_KEY environment variable for metrics submission.
    """

    # Success threshold aligned exactly with tau2's is_successful() definition
    # From tau2/metrics/agent_metrics.py: (1 - 1e-6) <= reward <= (1 + 1e-6)
    SUCCESS_THRESHOLD = 1.0 - 1e-6  # = 0.999999

    # Max metrics to buffer before auto-flushing (to stay under 512 KB payload limit)
    MAX_BUFFER_COUNT = 500

    def __init__(self, dry_run: bool = False):
        """Initialize the metrics emitter.

        Args:
            dry_run: If True, log metrics instead of sending to Datadog.
        """
        self.dry_run = dry_run
        self._api_key: str | None = None
        self._site: str = "datadoghq.com"
        self._metrics_buffer: list[dict[str, Any]] = []

        if not dry_run:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the Datadog HTTP API client."""
        self._api_key = os.getenv("DD_API_KEY")
        self._site = os.getenv("DD_SITE", "datadoghq.com")

        if self._api_key:
            logger.info(f"Using Datadog HTTP API for site: {self._site}")
        else:
            logger.warning("DD_API_KEY not set - metrics disabled (use --dry-run to test)")

    def _add_to_buffer(self, metric: str, value: float, tags: list[str], metric_type: str = "gauge") -> None:
        """Add a metric to the buffer for batch submission via HTTP API.

        Automatically flushes when buffer reaches MAX_BUFFER_COUNT to avoid
        exceeding Datadog's 512 KB payload limit.
        """
        env = os.getenv("DD_ENV", "dev")
        tag_str_list = tags + ["service:tau2-bench-agent", f"env:{env}"]

        self._metrics_buffer.append({
            "metric": metric,
            "type": 1 if metric_type == "count" else 3,  # 1=count, 3=gauge
            "points": [{"timestamp": int(time.time()), "value": value}],
            "tags": tag_str_list,
        })

        # Auto-flush when buffer reaches threshold to stay under payload limit
        if len(self._metrics_buffer) >= self.MAX_BUFFER_COUNT:
            self.flush()

    def flush(self) -> None:
        """Flush buffered metrics to Datadog via HTTP API."""
        if not self._api_key or not self._metrics_buffer:
            return

        if self.dry_run:
            logger.info(f"[DRY RUN] Would flush {len(self._metrics_buffer)} metrics")
            self._metrics_buffer.clear()
            return

        try:
            import httpx

            url = f"https://api.{self._site}/api/v2/series"
            headers = {
                "DD-API-KEY": self._api_key,
                "Content-Type": "application/json",
            }
            payload = {"series": self._metrics_buffer}

            response = httpx.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 202:
                logger.info(f"Flushed {len(self._metrics_buffer)} metrics to Datadog")
            else:
                logger.error(f"Failed to flush metrics: {response.status_code} {response.text}")

        except Exception as e:
            logger.error(f"Failed to flush metrics via HTTP API: {e}")

        self._metrics_buffer.clear()

    def _emit_gauge(self, metric: str, value: float, tags: list[str]) -> None:
        """Emit a gauge metric."""
        if self.dry_run:
            logger.info(f"[DRY RUN] gauge {metric}={value} tags={tags}")
            return

        if self._api_key:
            self._add_to_buffer(metric, value, tags, "gauge")

    def _emit_count(self, metric: str, value: int, tags: list[str]) -> None:
        """Emit a count metric."""
        if self.dry_run:
            logger.info(f"[DRY RUN] count {metric}={value} tags={tags}")
            return

        if self._api_key:
            self._add_to_buffer(metric, float(value), tags, "count")

    def _emit_histogram(self, metric: str, value: float, tags: list[str]) -> None:
        """Emit a histogram metric (sent as gauge via HTTP API)."""
        if self.dry_run:
            logger.info(f"[DRY RUN] histogram {metric}={value} tags={tags}")
            return

        if self._api_key:
            self._add_to_buffer(metric, value, tags, "gauge")

    def emit_task_metrics(
        self,
        task_id: str,
        domain: str,
        evaluation_id: str,
        reward: float,
        duration_seconds: float,
        steps: int,
        termination_reason: str = "unknown",
        agent_model: str = "unknown",
        agent_type: str = "unknown",
        agent_endpoint: str = "unknown",
        difficulty: str = "unknown",
        complexity_score: int = 0,
    ) -> None:
        """Emit metrics for a single task evaluation.

        Args:
            task_id: The task identifier.
            domain: The domain name (airline, retail, telecom, mock).
            evaluation_id: The evaluation identifier.
            reward: Task reward (0.0-1.0).
            duration_seconds: Task execution time in seconds.
            steps: Number of steps taken.
            termination_reason: Why the task terminated (for error tracking).
            agent_model: The LLM model being evaluated (e.g., "gpt-4", "claude-3").
            agent_type: The agent implementation type (e.g., "llm_agent_solo").
            agent_endpoint: The A2A agent endpoint URL (sanitized).
            difficulty: Task difficulty tier (easy, medium, hard, expert).
            complexity_score: Raw complexity score for detailed analysis.
        """
        success = reward >= self.SUCCESS_THRESHOLD
        base_tags = [
            f"task_id:{task_id}",
            f"domain:{domain}",
            f"evaluation_id:{evaluation_id}",
            f"agent_model:{agent_model}",
            f"agent_type:{agent_type}",
            f"agent_endpoint:{agent_endpoint}",
            f"difficulty:{difficulty}",
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
        self._emit_count("tau2.task.total", 1, [f"domain:{domain}", f"evaluation_id:{evaluation_id}", f"agent_model:{agent_model}", f"agent_endpoint:{agent_endpoint}", f"difficulty:{difficulty}"])

        # tau2.task.error_count - count failures for error rate widget
        if not success:
            error_tags = [f"domain:{domain}", f"reason:{termination_reason}", f"agent_model:{agent_model}", f"agent_endpoint:{agent_endpoint}", f"difficulty:{difficulty}"]
            self._emit_count("tau2.task.error_count", 1, error_tags)

        # tau2.task.complexity_score - raw complexity metric for analysis
        self._emit_gauge("tau2.task.complexity_score", float(complexity_score), base_tags)

    def emit_tool_metrics(
        self,
        tool_name: str,
        domain: str,
        correct: bool,
        arguments_match: bool,
        requestor: str = "agent",
        agent_model: str = "unknown",
        agent_endpoint: str = "unknown",
        difficulty: str = "unknown",
    ) -> None:
        """Emit metrics for tool invocations.

        Args:
            tool_name: Name of the tool called.
            domain: The domain name.
            correct: Whether the tool call was correct.
            arguments_match: Whether arguments matched expected.
            requestor: Who called the tool (agent/user).
            agent_model: The LLM model being evaluated.
            agent_endpoint: The A2A agent endpoint (sanitized).
            difficulty: Task difficulty tier.
        """
        base_tags = [
            f"tool_name:{tool_name}",
            f"domain:{domain}",
            f"requestor:{requestor}",
            f"agent_model:{agent_model}",
            f"agent_endpoint:{agent_endpoint}",
            f"difficulty:{difficulty}",
        ]

        # tau2.tool.calls - count
        self._emit_count("tau2.tool.calls", 1, base_tags)

        # tau2.tool.correct - count with correct tag
        correct_tags = [
            f"tool_name:{tool_name}",
            f"correct:{str(correct).lower()}",
            f"agent_model:{agent_model}",
            f"agent_endpoint:{agent_endpoint}",
            f"difficulty:{difficulty}",
        ]
        self._emit_count("tau2.tool.correct", 1, correct_tags)

        # tau2.tool.arguments_match - count with match tag
        match_tags = [
            f"tool_name:{tool_name}",
            f"match:{str(arguments_match).lower()}",
            f"agent_model:{agent_model}",
            f"agent_endpoint:{agent_endpoint}",
            f"difficulty:{difficulty}",
        ]
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

    def emit_termination_metrics(
        self,
        reason: str,
        domain: str = "unknown",
        agent_endpoint: str = "unknown",
        difficulty: str = "unknown",
    ) -> None:
        """Emit metrics for task termination reasons.

        Args:
            reason: Termination reason (user_stop, agent_stop, max_steps, max_errors).
            domain: The domain name.
            agent_endpoint: The A2A agent endpoint (sanitized).
            difficulty: Task difficulty tier.
        """
        # tau2.termination - count with reason and context tags
        tags = [
            f"reason:{reason}",
            f"domain:{domain}",
            f"agent_endpoint:{agent_endpoint}",
            f"difficulty:{difficulty}",
        ]
        self._emit_count("tau2.termination", 1, tags)

    def emit_evaluation_metrics(
        self,
        evaluation_id: str,
        domain: str,
        pass_rate: float,
        avg_reward: float,
        total_tasks: int,
        agent_model: str = "unknown",
        agent_type: str = "unknown",
        agent_endpoint: str = "unknown",
    ) -> None:
        """Emit aggregated metrics for an evaluation run.

        Args:
            evaluation_id: The evaluation identifier.
            domain: The domain name.
            pass_rate: Overall pass rate percentage.
            avg_reward: Average reward across tasks.
            total_tasks: Total number of tasks evaluated.
            agent_model: The LLM model being evaluated.
            agent_type: The agent implementation type.
            agent_endpoint: The A2A agent endpoint URL (sanitized).
        """
        base_tags = [
            f"evaluation_id:{evaluation_id}",
            f"domain:{domain}",
            f"agent_model:{agent_model}",
            f"agent_type:{agent_type}",
            f"agent_endpoint:{agent_endpoint}",
        ]

        # tau2.evaluation.pass_rate - gauge
        self._emit_gauge("tau2.evaluation.pass_rate", pass_rate, base_tags)

        # tau2.evaluation.avg_reward - gauge
        self._emit_gauge("tau2.evaluation.avg_reward", avg_reward, base_tags)

        # tau2.evaluation.tasks_total - gauge
        self._emit_gauge("tau2.evaluation.tasks_total", float(total_tasks), base_tags)

    def emit_efficiency_metrics(
        self,
        task_id: str,
        domain: str,
        evaluation_id: str,
        reward: float,
        duration_seconds: float,
        turns: int,
        tool_calls_total: int,
        tool_accuracy: float,
        agent_model: str = "unknown",
        agent_type: str = "unknown",
        agent_endpoint: str = "unknown",
        difficulty: str = "unknown",
    ) -> None:
        """Emit task completion efficiency metrics.

        These metrics measure observable efficiency: how efficiently
        the task was completed from the evaluator's perspective.

        Args:
            task_id: The task identifier.
            domain: The domain name.
            evaluation_id: The evaluation identifier.
            reward: Task reward (0.0-1.0).
            duration_seconds: Task execution time.
            turns: Number of conversation turns.
            tool_calls_total: Total tool calls made.
            tool_accuracy: Fraction of correct tool calls (0.0-1.0).
            agent_model: The LLM model being evaluated.
            agent_type: The agent implementation type.
            agent_endpoint: The A2A agent endpoint URL (sanitized).
            difficulty: Task difficulty tier (easy, medium, hard, expert).
        """
        base_tags = [
            f"task_id:{task_id}",
            f"domain:{domain}",
            f"evaluation_id:{evaluation_id}",
            f"agent_model:{agent_model}",
            f"agent_type:{agent_type}",
            f"agent_endpoint:{agent_endpoint}",
            f"difficulty:{difficulty}",
        ]

        # tau2.task.reward_per_turn - efficiency metric
        if turns > 0:
            self._emit_gauge("tau2.task.reward_per_turn", reward / turns, base_tags)

        # tau2.task.reward_per_second - time efficiency
        if duration_seconds > 0:
            self._emit_gauge("tau2.task.reward_per_second", reward / duration_seconds, base_tags)

        # tau2.task.turns_total - conversation length
        self._emit_gauge("tau2.task.turns_total", float(turns), base_tags)

        # tau2.task.tool_calls_total - tool usage count
        self._emit_gauge("tau2.task.tool_calls_total", float(tool_calls_total), base_tags)

        # tau2.task.tool_accuracy - tool correctness ratio
        self._emit_gauge("tau2.task.tool_accuracy", tool_accuracy, base_tags)

        # tau2.task.reward_per_tool_call - tool efficiency metric
        # Higher values = more reward achieved per tool call = more efficient
        if tool_calls_total > 0:
            self._emit_gauge("tau2.task.reward_per_tool_call", reward / tool_calls_total, base_tags)

    def emit_simulator_metrics(
        self,
        evaluation_id: str,
        domain: str,
        tokens_prompt: int,
        tokens_completion: int,
        tokens_total: int,
        cost_usd: float,
        agent_model: str = "unknown",
        agent_type: str = "unknown",
        agent_endpoint: str = "unknown",
    ) -> None:
        """Emit user simulator metrics (test harness cost).

        These track the cost of running the user simulator LLM,
        which is the cost we control (vs agent which is a black box).

        Args:
            evaluation_id: The evaluation identifier.
            domain: The domain name.
            tokens_prompt: Total prompt tokens used by simulator.
            tokens_completion: Total completion tokens used by simulator.
            tokens_total: Total tokens (prompt + completion).
            cost_usd: Total cost in USD.
            agent_model: The LLM model being evaluated.
            agent_type: The agent implementation type.
            agent_endpoint: The A2A agent endpoint URL (sanitized).
        """
        base_tags = [
            f"evaluation_id:{evaluation_id}",
            f"domain:{domain}",
            f"agent_model:{agent_model}",
            f"agent_type:{agent_type}",
            f"agent_endpoint:{agent_endpoint}",
        ]

        # Simulator-specific metrics (Category B - Test Harness)
        self._emit_gauge("tau2.simulator.tokens_total", float(tokens_total), base_tags)
        self._emit_gauge("tau2.simulator.tokens_prompt", float(tokens_prompt), base_tags)
        self._emit_gauge("tau2.simulator.tokens_completion", float(tokens_completion), base_tags)
        self._emit_gauge("tau2.simulator.cost_usd", cost_usd, base_tags)

        # Aliases for dashboard/monitor compatibility (fixes broken widgets)
        self._emit_gauge("tau2.llm.tokens_input", float(tokens_prompt), base_tags)
        self._emit_gauge("tau2.llm.tokens_output", float(tokens_completion), base_tags)
        self._emit_gauge("tau2.llm.token_cost", cost_usd, base_tags)

    def emit_pass_hat_k_metrics(
        self,
        evaluation_id: str,
        domain: str,
        pass_hat_k_values: dict[int, float],
        agent_model: str = "unknown",
        agent_type: str = "unknown",
        agent_endpoint: str = "unknown",
    ) -> None:
        """Emit pass^k metrics for an evaluation.

        pass^k is tau2's standard metric from https://arxiv.org/pdf/2406.12045
        It measures the probability of success given k attempts.

        Args:
            evaluation_id: The evaluation identifier.
            domain: The domain name.
            pass_hat_k_values: Dict mapping k -> average pass^k value.
            agent_model: The LLM model being evaluated.
            agent_type: The agent implementation type.
            agent_endpoint: The A2A agent endpoint URL (sanitized).
        """
        base_tags = [
            f"evaluation_id:{evaluation_id}",
            f"domain:{domain}",
            f"agent_model:{agent_model}",
            f"agent_type:{agent_type}",
            f"agent_endpoint:{agent_endpoint}",
        ]

        for k, value in pass_hat_k_values.items():
            # tau2.evaluation.pass_hat_k - pass^k metric (industry standard)
            k_tags = base_tags + [f"k:{k}"]
            self._emit_gauge(f"tau2.evaluation.pass_hat_{k}", value * 100, k_tags)


def extract_simulator_metrics(messages: list) -> dict:
    """Extract token usage and cost from user simulator messages.

    User messages (role=user) contain usage data from the simulator LLM.
    Assistant messages have usage=None (agent is a black box).

    Args:
        messages: List of message dictionaries from the simulation.

    Returns:
        dict with keys: tokens_prompt, tokens_completion, tokens_total, cost_usd
    """
    tokens_prompt = 0
    tokens_completion = 0
    cost_usd = 0.0

    for msg in messages:
        if msg.get("role") != "user":
            continue
        usage = msg.get("usage")
        if usage:
            tokens_prompt += usage.get("prompt_tokens", 0)
            tokens_completion += usage.get("completion_tokens", 0)
        cost = msg.get("cost")
        if cost:
            cost_usd += cost

    return {
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "tokens_total": tokens_prompt + tokens_completion,
        "cost_usd": cost_usd,
    }


def count_tool_calls(messages: list) -> int:
    """Count total tool calls from messages.

    Tool calls can appear in both assistant and user messages.

    Args:
        messages: List of message dictionaries from the simulation.

    Returns:
        int: Total number of tool calls.
    """
    total = 0
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += len(tool_calls)
    return total


def calculate_tool_accuracy(action_checks: list) -> float:
    """Calculate tool accuracy from action checks.

    Args:
        action_checks: List of action check results from reward_info.

    Returns:
        float: Accuracy between 0.0 and 1.0 (returns 1.0 if no checks).
    """
    if not action_checks:
        return 1.0  # No checks = assume perfect (nothing to fail)

    correct = sum(1 for check in action_checks if check.get("action_match", False))
    return correct / len(action_checks)


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """Compute the pass^k metric.

    This is tau2's standard metric from https://arxiv.org/pdf/2406.12045
    It measures the probability of at least one success in k trials.

    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.

    Returns:
        The pass^k metric (0.0 to 1.0).
    """
    if num_trials < k:
        return 0.0  # Not enough trials
    if k <= 0:
        return 0.0

    return math.comb(success_count, k) / math.comb(num_trials, k)


def calculate_pass_hat_k_metrics(
    task_results: dict[str, list[float]],
    success_threshold: float,
    max_k: int = 3,
) -> dict[str, dict[int, float]]:
    """Calculate pass^k metrics for each task.

    Groups trials by task_id and computes pass^k for k=1,2,...,max_k.

    Args:
        task_results: Dict mapping task_id -> list of reward values.
        success_threshold: Threshold for considering a trial successful.
        max_k: Maximum k value to compute (default 3).

    Returns:
        Dict mapping task_id -> {k: pass^k value}
    """
    results = {}

    for task_id, rewards in task_results.items():
        num_trials = len(rewards)
        success_count = sum(1 for r in rewards if r >= success_threshold)

        results[task_id] = {}
        for k in range(1, max_k + 1):
            if num_trials >= k:
                results[task_id][k] = pass_hat_k(num_trials, success_count, k)
            else:
                results[task_id][k] = 0.0

    return results


def is_valid_simulation(sim: dict) -> bool:
    """Check if a simulation was produced by tau2 (not manually seeded test data).

    Tau2 evaluator ALWAYS produces:
    1. Conversation messages (agent/user exchange during simulation)
    2. RewardInfo with structure (at minimum: reward + info, or full breakdown)

    Hand-crafted test data is identified by:
    - Empty messages array (no actual conversation occurred)
    - Minimal reward_info with only {"reward": X} (no evaluation breakdown)

    This prevents:
    - Gaming metrics with fake 100% pass rates
    - Polluting dashboards with seeded test data

    Args:
        sim: A simulation dictionary from evaluation results.

    Returns:
        True if produced by tau2, False if hand-crafted/seeded data.
    """
    messages = sim.get("messages", [])
    reward_info = sim.get("reward_info", {})

    # Tau2 always produces conversation messages during simulation
    # Empty messages = no agent interaction occurred = not a real evaluation
    if not messages or len(messages) == 0:
        return False

    # Tau2 always includes structure in reward_info:
    # - Normal case: db_check, action_checks, nl_assertions, reward_basis, reward_breakdown
    # - Edge cases: at minimum "info" field with explanation
    # Hand-crafted data typically only has {"reward": 1.0}
    reward_info_keys = set(reward_info.keys())
    tau2_fields = {"db_check", "env_assertions", "action_checks", "nl_assertions",
                   "communicate_checks", "reward_basis", "reward_breakdown", "info"}

    # Valid if reward_info has ANY tau2-specific field
    if reward_info_keys & tau2_fields:
        return True

    # Only has "reward" key (or is empty) = hand-crafted data
    return False


def derive_task_difficulty(reward_info: dict, domain: str) -> tuple[str, int]:
    """Derive task difficulty from tau2's native evaluation criteria.

    Grounded in tau2's EvaluationCriteria.info() which provides:
    - num_agent_actions: Actions the agent must take
    - num_user_actions: Actions requiring agent to guide user (harder)
    - num_env_assertions: Environment state checks
    - num_nl_assertions: Natural language assertions

    The complexity score is a simple sum of evaluation burden without
    arbitrary weights, matching tau2's own task grouping logic.

    Args:
        reward_info: The reward_info dict from simulation results.
        domain: The domain name (for logging only, not used in calculation).

    Returns:
        Tuple of (difficulty_tier: str, complexity_score: int)
        difficulty_tier: "unknown", "easy", "medium", "hard", or "expert"
        complexity_score: Raw numeric score (0 if unknown)
    """
    # Extract tau2's native evaluation criteria
    action_checks = reward_info.get("action_checks", []) or []
    nl_assertions = reward_info.get("nl_assertions", []) or []
    communicate_checks = reward_info.get("communicate_checks", []) or []
    env_assertions = reward_info.get("env_assertions", []) or []

    # Separate agent vs user actions (matching tau2's EvaluationCriteria.info())
    # User actions require the agent to guide the user, which is harder
    agent_actions = len([a for a in action_checks
                         if a.get("action", {}).get("requestor") == "assistant"])
    user_actions = len([a for a in action_checks
                        if a.get("action", {}).get("requestor") == "user"])

    # Count other evaluation criteria
    nl_count = len(nl_assertions)
    comm_count = len(communicate_checks)
    env_count = len(env_assertions)

    # Check if we have any evaluation criteria data
    total_checks = agent_actions + user_actions + nl_count + comm_count + env_count

    if total_checks == 0:
        # No evaluation data available - cannot determine difficulty
        return ("unknown", 0)

    # Raw complexity score - simple sum of evaluation burden
    # This matches tau2's own approach of counting checks
    complexity_score = total_checks

    # Map to difficulty tier based on total evaluation burden
    # Thresholds based on tau2's task distribution patterns
    if complexity_score <= 2:
        tier = "easy"       # 1-2 total checks
    elif complexity_score <= 4:
        tier = "medium"     # 3-4 total checks
    elif complexity_score <= 6:
        tier = "hard"       # 5-6 total checks
    else:
        tier = "expert"     # 7+ total checks

    return tier, complexity_score


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


def sanitize_endpoint_for_tag(endpoint: str | None) -> str:
    """Sanitize an endpoint URL for use as a Datadog tag value.

    Datadog tags have restrictions on characters. This function:
    - Extracts the host/path from the URL
    - Replaces invalid characters with underscores
    - Truncates to reasonable length

    Args:
        endpoint: The endpoint URL or None.

    Returns:
        Sanitized string suitable for use as a tag value.
    """
    if not endpoint:
        return "unknown"

    # Remove protocol prefix
    sanitized = endpoint
    for prefix in ["https://", "http://"]:
        if sanitized.startswith(prefix):
            sanitized = sanitized[len(prefix):]
            break

    # Replace invalid characters with underscores
    # Datadog allows: letters, numbers, underscores, minuses, colons, periods, slashes
    import re
    sanitized = re.sub(r"[^a-zA-Z0-9_\-:./]", "_", sanitized)

    # Truncate to 200 chars (Datadog tag value limit)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]

    return sanitized or "unknown"


def extract_agent_info(eval_data: dict, results_data: dict) -> tuple[str, str, str]:
    """Extract agent model, type, and endpoint from evaluation data.

    Tries multiple locations in order of preference:
    1. EvaluationStore format: request.agent_model, request.agent_type, agent_endpoint
    2. tau2 Results format: info.agent_info.llm, info.agent_info.implementation
    3. Filename pattern extraction (fallback)

    Args:
        eval_data: The top-level evaluation data.
        results_data: The results data (may be same as eval_data).

    Returns:
        Tuple of (agent_model, agent_type, agent_endpoint).
    """
    # Try EvaluationStore format first
    request = eval_data.get("request", {})
    agent_model = request.get("agent_model") or request.get("agent_llm")
    agent_type = request.get("agent_type")

    # Extract agent endpoint (EvaluationStore format)
    agent_endpoint = eval_data.get("agent_endpoint")

    # Try tau2 Results format (info.agent_info)
    if not agent_model:
        agent_info = results_data.get("info", {}).get("agent_info", {})
        agent_model = agent_info.get("llm")
        agent_type = agent_type or agent_info.get("implementation")

    # Default values
    agent_model = agent_model or "unknown"
    agent_type = agent_type or "unknown"
    agent_endpoint = sanitize_endpoint_for_tag(agent_endpoint)

    return agent_model, agent_type, agent_endpoint


def process_evaluation(emitter: MetricsEmitter, eval_data: dict, evaluation_id: str) -> None:
    """Process a single evaluation and emit all metrics.

    Handles two JSON formats:
    1. EvaluationStore format (from 002-evaluation-store):
       {
         "evaluation_id": "...",
         "domain": "...",
         "request": {"agent_model": "...", "agent_type": "..."},
         "results": {
           "simulations": [...],
           "info": {"environment_info": {"domain_name": "..."}}
         }
       }
    2. Direct Results format (legacy tau2 output):
       {
         "simulations": [...],
         "info": {
           "agent_info": {"llm": "...", "implementation": "..."},
           "environment_info": {"domain_name": "..."}
         }
       }

    Args:
        emitter: The metrics emitter instance.
        eval_data: The evaluation data dictionary.
        evaluation_id: The evaluation identifier.
    """
    # Handle EvaluationStore format (results are nested under "results" key)
    # Note: .get() returns None for explicit null values, so we check explicitly
    results_data = eval_data.get("results")
    if results_data is None:
        results_data = eval_data

    # Extract domain - try EvaluationStore format first, then Results format
    domain = eval_data.get("domain")  # EvaluationStore has domain at top level
    if not domain:
        # Fall back to info.environment_info.domain_name from results
        domain = results_data.get("info", {}).get("environment_info", {}).get("domain_name", "unknown")

    # Extract agent model, type, and endpoint
    agent_model, agent_type, agent_endpoint = extract_agent_info(eval_data, results_data)

    simulations = results_data.get("simulations", [])
    if not simulations:
        logger.debug(f"Skipping evaluation {evaluation_id}: no simulations")
        return

    # Filter out invalid/fake simulations
    valid_simulations = [sim for sim in simulations if is_valid_simulation(sim)]
    invalid_count = len(simulations) - len(valid_simulations)

    if invalid_count > 0:
        logger.warning(
            f"Evaluation {evaluation_id}: Filtered out {invalid_count}/{len(simulations)} "
            f"invalid simulations (missing messages or evaluation data)"
        )

    if not valid_simulations:
        logger.warning(
            f"Skipping evaluation {evaluation_id}: all {len(simulations)} simulations are invalid/fake"
        )
        return

    total_reward = 0.0
    successful_tasks = 0
    total_tasks = len(valid_simulations)

    # Aggregates for simulator metrics (across all simulations)
    total_sim_tokens_prompt = 0
    total_sim_tokens_completion = 0
    total_sim_cost = 0.0

    # Collect task results for pass^k calculation (group trials by task_id)
    task_results: dict[str, list[float]] = defaultdict(list)

    for sim in valid_simulations:
        task_id = sim.get("task_id", "unknown")
        reward_info = sim.get("reward_info", {})
        reward = reward_info.get("reward", 0.0)
        duration = sim.get("duration", 0.0)
        messages = sim.get("messages", [])
        termination_reason = sim.get("termination_reason", "unknown")

        # Derive task difficulty from reward_info
        difficulty, complexity_score = derive_task_difficulty(reward_info, domain)

        # Emit task metrics
        emitter.emit_task_metrics(
            task_id=task_id,
            domain=domain,
            evaluation_id=evaluation_id,
            reward=reward,
            duration_seconds=duration,
            steps=len(messages),
            termination_reason=termination_reason,
            agent_model=agent_model,
            agent_type=agent_type,
            agent_endpoint=agent_endpoint,
            difficulty=difficulty,
            complexity_score=complexity_score,
        )

        # Emit termination metrics
        emitter.emit_termination_metrics(
            reason=termination_reason,
            domain=domain,
            agent_endpoint=agent_endpoint,
            difficulty=difficulty,
        )

        # Calculate and emit efficiency metrics (Phase 1)
        action_checks = reward_info.get("action_checks", [])
        tool_calls_total = count_tool_calls(messages)
        tool_accuracy = calculate_tool_accuracy(action_checks)

        emitter.emit_efficiency_metrics(
            task_id=task_id,
            domain=domain,
            evaluation_id=evaluation_id,
            reward=reward,
            duration_seconds=duration,
            turns=len(messages),
            tool_calls_total=tool_calls_total,
            tool_accuracy=tool_accuracy,
            agent_model=agent_model,
            agent_type=agent_type,
            agent_endpoint=agent_endpoint,
            difficulty=difficulty,
        )

        # Accumulate simulator metrics from this simulation (Phase 2b)
        sim_metrics = extract_simulator_metrics(messages)
        total_sim_tokens_prompt += sim_metrics["tokens_prompt"]
        total_sim_tokens_completion += sim_metrics["tokens_completion"]
        total_sim_cost += sim_metrics["cost_usd"]

        # Process action checks for tool metrics
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
                    agent_model=agent_model,
                    agent_endpoint=agent_endpoint,
                    difficulty=difficulty,
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

        # Collect rewards by task_id for pass^k calculation
        task_results[task_id].append(reward)

    # Emit evaluation-level metrics
    avg_reward = total_reward / total_tasks if total_tasks > 0 else 0.0
    pass_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0.0

    emitter.emit_evaluation_metrics(
        evaluation_id=evaluation_id,
        domain=domain,
        pass_rate=pass_rate,
        avg_reward=avg_reward,
        total_tasks=total_tasks,
        agent_model=agent_model,
        agent_type=agent_type,
        agent_endpoint=agent_endpoint,
    )

    # Calculate and emit pass^k metrics (industry-standard from tau2)
    # pass^k measures probability of success given k attempts
    if task_results:
        pass_hat_k_by_task = calculate_pass_hat_k_metrics(
            task_results,
            success_threshold=MetricsEmitter.SUCCESS_THRESHOLD,
            max_k=3,
        )

        # Calculate average pass^k across all tasks
        avg_pass_hat_k: dict[int, float] = {}
        for k in range(1, 4):
            k_values = [v.get(k, 0.0) for v in pass_hat_k_by_task.values()]
            if k_values:
                avg_pass_hat_k[k] = sum(k_values) / len(k_values)

        emitter.emit_pass_hat_k_metrics(
            evaluation_id=evaluation_id,
            domain=domain,
            pass_hat_k_values=avg_pass_hat_k,
            agent_model=agent_model,
            agent_type=agent_type,
            agent_endpoint=agent_endpoint,
        )

    # Emit aggregated simulator metrics (Phase 2b)
    total_sim_tokens = total_sim_tokens_prompt + total_sim_tokens_completion
    emitter.emit_simulator_metrics(
        evaluation_id=evaluation_id,
        domain=domain,
        tokens_prompt=total_sim_tokens_prompt,
        tokens_completion=total_sim_tokens_completion,
        tokens_total=total_sim_tokens,
        cost_usd=total_sim_cost,
        agent_model=agent_model,
        agent_type=agent_type,
        agent_endpoint=agent_endpoint,
    )

    # Emit per-model efficiency metrics (for model comparison)
    model_tags = [
        f"agent_model:{agent_model}",
        f"agent_type:{agent_type}",
        f"agent_endpoint:{agent_endpoint}",
        f"domain:{domain}",
        f"evaluation_id:{evaluation_id}",
    ]

    # tau2.model.tokens_per_task - average tokens used per task
    if total_tasks > 0:
        emitter._emit_gauge(
            "tau2.model.tokens_per_task",
            float(total_sim_tokens) / total_tasks,
            model_tags,
        )

    # tau2.model.cost_per_task - average cost per task
    if total_tasks > 0:
        emitter._emit_gauge(
            "tau2.model.cost_per_task",
            total_sim_cost / total_tasks,
            model_tags,
        )

    # tau2.model.reward_per_token - efficiency: reward earned per token spent
    if total_sim_tokens > 0:
        emitter._emit_gauge(
            "tau2.model.reward_per_token",
            total_reward / total_sim_tokens,
            model_tags,
        )

    # tau2.model.cost_per_success - cost efficiency: cost per successful task
    if successful_tasks > 0:
        emitter._emit_gauge(
            "tau2.model.cost_per_success",
            total_sim_cost / successful_tasks,
            model_tags,
        )

    # Format pass^k info for logging
    pass_k_info = ""
    if task_results and avg_pass_hat_k:
        pass_k_info = f", pass^1={avg_pass_hat_k.get(1, 0) * 100:.1f}%"

    logger.info(
        f"Emitted metrics for {evaluation_id} (model={agent_model}, endpoint={agent_endpoint}): "
        f"{total_tasks} tasks, pass_rate={pass_rate:.1f}%, avg_reward={avg_reward:.2f}{pass_k_info}, "
        f"simulator_tokens={total_sim_tokens}"
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
        "--clean",
        action="store_true",
        help="Delete evaluation files after successfully emitting metrics (prevents double-counting)",
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

    processed_paths: list[Path] = []

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
                processed_paths.append(eval_path)
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
        processed_paths.append(eval_path)

    # Flush any buffered metrics (for HTTP API mode)
    emitter.flush()

    # Clean up processed evaluation files if requested
    if args.clean and processed_paths:
        if args.dry_run:
            logger.info(f"[DRY RUN] Would delete {len(processed_paths)} evaluation files")
        else:
            deleted = 0
            for eval_path in processed_paths:
                try:
                    eval_path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete {eval_path}: {e}")
            logger.info(f"Deleted {deleted}/{len(processed_paths)} evaluation files")

    return 0


if __name__ == "__main__":
    sys.exit(main())

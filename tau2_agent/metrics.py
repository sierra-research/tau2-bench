"""Datadog metrics emission for tau2_agent.

This module provides in-process metric emission for evaluations completed
by the tau2_agent. It wraps the emit_metrics functionality to emit metrics
immediately after an evaluation completes, before the response is returned.

This ensures metrics are emitted from GCP Cloud Run containers where the
filesystem is ephemeral and post-hoc emission is not possible.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger


def emit_evaluation_metrics(
    evaluation_id: str,
    domain: str,
    agent_endpoint: str,
    results: dict[str, Any],
) -> bool:
    """Emit Datadog metrics for a completed evaluation.

    This function is called immediately after store.complete_evaluation()
    to emit metrics before the response is returned. Metrics are sent via
    Datadog HTTP API (agentless mode) when DD_API_KEY is set.

    Args:
        evaluation_id: The evaluation identifier (e.g., eval-1703697600000-a1b2c3).
        domain: The tau2 domain (airline, retail, telecom, mock).
        agent_endpoint: The A2A agent endpoint URL that was evaluated.
        results: The evaluation results dict containing:
            - success_rate: float
            - total_tasks: int
            - successful: int
            - tasks: list of task results
            - simulations: list of simulation data with messages and reward_info

    Returns:
        True if metrics were emitted successfully, False otherwise.

    Note:
        This function is a no-op if DD_API_KEY is not set.
        All exceptions are caught and logged to avoid affecting evaluation flow.
    """
    # Skip if Datadog API key not configured
    dd_api_key = os.getenv("DD_API_KEY")
    if not dd_api_key:
        logger.debug("DD_API_KEY not set, skipping metric emission")
        return False

    try:
        # Import here to avoid import errors if experiments module not available
        from experiments.datadog.scripts.emit_metrics import (
            MetricsEmitter,
            process_evaluation,
        )

        # Build eval_data structure expected by process_evaluation
        eval_data = {
            "evaluation_id": evaluation_id,
            "domain": domain,
            "agent_endpoint": agent_endpoint,
            "results": results,
        }

        # Create emitter and process the evaluation
        emitter = MetricsEmitter(dry_run=False)
        process_evaluation(emitter, eval_data, evaluation_id)
        emitter.flush()

        logger.info(f"Emitted metrics for evaluation {evaluation_id}")
        return True

    except ImportError as e:
        logger.warning(f"Could not import emit_metrics module: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to emit metrics for {evaluation_id}: {e}")
        return False

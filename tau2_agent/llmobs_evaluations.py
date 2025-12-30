"""LLMObs evaluation submission for tau2_agent.

This module provides functions to submit evaluation metrics to Datadog LLM Observability
using the LLMObs.submit_evaluation() API. Evaluations must be linked to a specific span
using span_context obtained from LLMObs.export_span().

The integration is opt-in and requires:
    - DD_TRACE_ENABLED=true
    - DD_LLMOBS_ENABLED=true
    - DD_API_KEY set for agentless mode

When disabled, all submission functions are no-ops that return immediately.

Usage:
    from ddtrace.llmobs import LLMObs
    from tau2_agent.llmobs_evaluations import submit_task_evaluations

    # Export span context during/after LLM operations
    span_context = LLMObs.export_span()

    # Submit evaluations linked to that span
    submit_task_evaluations(
        span_context=span_context,
        task_id="task-123",
        domain="airline",
        reward=0.85,
        termination_reason="agent_stop",
        reward_info=sim.reward_info.model_dump(),
    )

See Also:
    - Datadog LLMObs Evaluations: https://docs.datadoghq.com/llm_observability/evaluations/
    - Custom Evaluations: https://github.com/DataDog/llm-observability/blob/main/4-custom-evaluations.ipynb
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger

# Success threshold for binary pass/fail classification
SUCCESS_THRESHOLD = 0.7


@contextmanager
def llmobs_evaluation_span(
    evaluation_id: str,
    domain: str,
    agent_endpoint: str,
) -> Generator[Any, None, None]:
    """Create an LLMObs workflow span for an evaluation run.

    This span groups all evaluation activities and provides a span context
    that can be used to submit evaluations. Use this when you need to create
    evaluations outside of an existing LLM span context.

    Args:
        evaluation_id: Unique identifier for this evaluation.
        domain: The tau2 domain being evaluated.
        agent_endpoint: The agent endpoint being evaluated.

    Yields:
        Span context that can be passed to submit_task_evaluations() and
        submit_evaluation_summary(). Yields None if LLMObs is not enabled.

    Usage:
        with llmobs_evaluation_span(eval_id, domain, endpoint) as span_ctx:
            # Run evaluation...
            submit_task_evaluations(..., span_context=span_ctx)
            submit_evaluation_summary(..., span_context=span_ctx)
    """
    if not is_llmobs_enabled():
        yield None
        return

    try:
        from ddtrace.llmobs import LLMObs

        # Create a workflow span for the evaluation
        with LLMObs.workflow(
            name=f"tau2.evaluation.{domain}",
            session_id=evaluation_id,
        ) as span:
            # Set span metadata
            span.set_tag("evaluation_id", evaluation_id)
            span.set_tag("domain", domain)
            span.set_tag("agent_endpoint", agent_endpoint)

            # Export the span context for use in evaluations
            span_context = LLMObs.export_span(span=span)
            yield span_context

    except ImportError:
        logger.debug("LLMObs not available, skipping evaluation span")
        yield None
    except Exception as e:
        logger.warning(f"Failed to create LLMObs evaluation span: {e}")
        yield None


def is_llmobs_enabled() -> bool:
    """Check if LLM Observability evaluations are enabled.

    Returns:
        True if both DD_TRACE_ENABLED and DD_LLMOBS_ENABLED are set to "true".
    """
    return (
        os.getenv("DD_TRACE_ENABLED", "false").lower() == "true"
        and os.getenv("DD_LLMOBS_ENABLED", "false").lower() == "true"
    )


def submit_task_evaluations(
    task_id: str,
    domain: str,
    reward: float,
    termination_reason: str,
    reward_info: dict[str, Any] | None,
    evaluation_id: str | None = None,
    agent_endpoint: str | None = None,
    span_context: Any | None = None,
) -> None:
    """Submit per-task evaluation metrics to LLMObs.

    Submits multiple evaluation metrics for a single task execution:
    - tau2.task.reward: The overall reward score (0.0-1.0)
    - tau2.task.success: Binary pass/fail based on SUCCESS_THRESHOLD
    - tau2.task.termination: The termination reason category
    - tau2.assertion.*: Various assertion pass rates if available

    All metrics are tagged with task_id, domain, and optional evaluation_id
    for filtering in the Datadog UI.

    Args:
        task_id: The task identifier (e.g., "task-123").
        domain: The evaluation domain (e.g., "airline", "retail", "mock").
        reward: Task reward score between 0.0 and 1.0.
        termination_reason: Why the simulation ended (e.g., "agent_stop", "max_steps").
        reward_info: Full reward_info dict containing db_check, action_checks,
            nl_assertions, and communicate_checks. Can be None if no assertions.
        evaluation_id: Optional evaluation identifier for cross-task correlation.
        agent_endpoint: Optional agent endpoint URL for agent differentiation.
        span_context: Span context from LLMObs.export_span() to link evaluations
            to a specific LLM span. If None, attempts to get current span context.

    Note:
        This function is a no-op when LLMObs is not enabled. All exceptions
        are caught and logged to ensure evaluation execution is not affected.
    """
    if not is_llmobs_enabled():
        return

    try:
        from ddtrace.llmobs import LLMObs

        # Get span context - use provided or try to export current span
        ctx = span_context
        if ctx is None:
            try:
                ctx = LLMObs.export_span()
            except Exception as e:
                logger.debug(f"Could not export current span: {e}")

        if ctx is None:
            logger.warning(
                "No span context available for LLMObs evaluation submission. "
                "Evaluations require span_context from LLMObs.export_span() "
                "to be linked to LLM traces."
            )
            return

        # Build common tags for all metrics
        tags = {
            "task_id": task_id,
            "domain": domain,
        }
        if evaluation_id:
            tags["evaluation_id"] = evaluation_id
        if agent_endpoint:
            tags["agent_endpoint"] = agent_endpoint

        # 1. Task reward (score: 0.0-1.0)
        LLMObs.submit_evaluation(
            span_context=ctx,
            label="tau2.task.reward",
            value=reward,
            metric_type="score",
            tags=tags,
        )

        # 2. Task success (categorical: pass/fail)
        LLMObs.submit_evaluation(
            span_context=ctx,
            label="tau2.task.success",
            value="pass" if reward >= SUCCESS_THRESHOLD else "fail",
            metric_type="categorical",
            tags=tags,
        )

        # 3. Termination reason (categorical)
        LLMObs.submit_evaluation(
            span_context=ctx,
            label="tau2.task.termination",
            value=termination_reason,
            metric_type="categorical",
            tags=tags,
        )

        # Submit assertion metrics if reward_info is available
        if reward_info:
            _submit_assertion_evaluations(reward_info, tags, ctx)

        logger.debug(
            f"Submitted LLMObs evaluations for task {task_id}",
            task_id=task_id,
            domain=domain,
            reward=reward,
        )

    except ImportError:
        logger.warning("LLMObs not available, skipping evaluation submission")
    except Exception as e:
        logger.warning(f"Failed to submit LLMObs evaluations: {e}")


def _submit_assertion_evaluations(
    reward_info: dict[str, Any],
    tags: dict[str, str],
    span_context: Any,
) -> None:
    """Submit assertion-level evaluation metrics.

    Extracts assertion results from reward_info and submits:
    - tau2.assertion.db_check: Database state validation (pass/fail)
    - tau2.assertion.nl_pass_rate: Natural language assertion pass rate
    - tau2.assertion.action_accuracy: Tool call correctness rate
    - tau2.assertion.communicate_pass_rate: Communication check pass rate

    Args:
        reward_info: Dict containing assertion check results.
        tags: Common tags to apply to all metrics.
        span_context: Span context from LLMObs.export_span() to link evaluations.
    """
    from ddtrace.llmobs import LLMObs

    # DB check (categorical: pass/fail)
    db_check = reward_info.get("db_check")
    if db_check:
        db_match = db_check.get("db_match", False)
        LLMObs.submit_evaluation(
            span_context=span_context,
            label="tau2.assertion.db_check",
            value="pass" if db_match else "fail",
            metric_type="categorical",
            tags=tags,
        )

    # NL assertion pass rate (score: 0.0-1.0)
    nl_assertions = reward_info.get("nl_assertions", [])
    if nl_assertions:
        passed = sum(1 for a in nl_assertions if a.get("met", False))
        pass_rate = passed / len(nl_assertions)
        LLMObs.submit_evaluation(
            span_context=span_context,
            label="tau2.assertion.nl_pass_rate",
            value=pass_rate,
            metric_type="score",
            tags=tags,
        )

    # Action accuracy (score: 0.0-1.0)
    action_checks = reward_info.get("action_checks", [])
    if action_checks:
        correct = sum(1 for c in action_checks if c.get("action_match", False))
        accuracy = correct / len(action_checks)
        LLMObs.submit_evaluation(
            span_context=span_context,
            label="tau2.assertion.action_accuracy",
            value=accuracy,
            metric_type="score",
            tags=tags,
        )

    # Communicate check pass rate (score: 0.0-1.0)
    communicate_checks = reward_info.get("communicate_checks", [])
    if communicate_checks:
        passed = sum(1 for c in communicate_checks if c.get("met", False))
        pass_rate = passed / len(communicate_checks)
        LLMObs.submit_evaluation(
            span_context=span_context,
            label="tau2.assertion.communicate_pass_rate",
            value=pass_rate,
            metric_type="score",
            tags=tags,
        )


def submit_evaluation_summary(
    evaluation_id: str,
    domain: str,
    total_tasks: int,
    successful_tasks: int,
    avg_reward: float,
    agent_endpoint: str | None = None,
    span_context: Any | None = None,
) -> None:
    """Submit aggregated evaluation metrics to LLMObs.

    Submits summary metrics for an entire evaluation run:
    - tau2.evaluation.pass_rate: Fraction of tasks that passed
    - tau2.evaluation.avg_reward: Average reward across all tasks

    Args:
        evaluation_id: The evaluation run identifier.
        domain: The evaluation domain.
        total_tasks: Total number of tasks evaluated.
        successful_tasks: Number of tasks with reward >= SUCCESS_THRESHOLD.
        avg_reward: Average reward across all tasks (0.0-1.0).
        agent_endpoint: Optional agent endpoint URL for agent differentiation.
        span_context: Span context from LLMObs.export_span() to link evaluations
            to a specific LLM span. If None, attempts to get current span context.

    Note:
        This function is a no-op when LLMObs is not enabled.
    """
    if not is_llmobs_enabled():
        return

    try:
        from ddtrace.llmobs import LLMObs

        # Get span context - use provided or try to export current span
        ctx = span_context
        if ctx is None:
            try:
                ctx = LLMObs.export_span()
            except Exception as e:
                logger.debug(f"Could not export current span: {e}")

        if ctx is None:
            logger.warning(
                "No span context available for LLMObs evaluation summary. "
                "Evaluations require span_context from LLMObs.export_span() "
                "to be linked to LLM traces."
            )
            return

        tags = {
            "evaluation_id": evaluation_id,
            "domain": domain,
        }
        if agent_endpoint:
            tags["agent_endpoint"] = agent_endpoint

        # Overall pass rate
        pass_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        LLMObs.submit_evaluation(
            span_context=ctx,
            label="tau2.evaluation.pass_rate",
            value=pass_rate,
            metric_type="score",
            tags=tags,
        )

        # Average reward
        LLMObs.submit_evaluation(
            span_context=ctx,
            label="tau2.evaluation.avg_reward",
            value=avg_reward,
            metric_type="score",
            tags=tags,
        )

        logger.debug(
            f"Submitted LLMObs evaluation summary for {evaluation_id}",
            evaluation_id=evaluation_id,
            pass_rate=pass_rate,
            avg_reward=avg_reward,
        )

    except ImportError:
        logger.warning("LLMObs not available, skipping summary submission")
    except Exception as e:
        logger.warning(f"Failed to submit LLMObs evaluation summary: {e}")

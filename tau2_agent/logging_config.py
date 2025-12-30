"""
Logging configuration for tau2_agent with GCP Cloud Logging support.

This module configures loguru for structured JSON logging when running in GCP,
and human-readable format for local development.

GCP Cloud Logging expects JSON with specific fields:
- severity: ERROR, WARNING, INFO, DEBUG (mapped from loguru levels)
- message: The log message
- Additional fields are indexed for querying

Usage:
    from tau2_agent.logging_config import configure_logging
    configure_logging()
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone

from loguru import logger

# GCP severity levels (mapped from Python/loguru levels)
SEVERITY_MAP = {
    "TRACE": "DEBUG",
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "SUCCESS": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


def is_running_in_gcp() -> bool:
    """Check if running in Google Cloud Run.

    Cloud Run sets K_SERVICE and K_REVISION environment variables.
    """
    return bool(os.getenv("K_SERVICE"))


def suppress_adk_warnings():
    """Suppress ADK experimental warnings that flood the logs.

    The ADK library emits many UserWarning about experimental features.
    These are informational but clutter the logs.
    """
    # Filter out ADK experimental warnings
    warnings.filterwarnings(
        "ignore",
        message=r"\[EXPERIMENTAL\].*",
        category=UserWarning,
        module=r"google\.adk\..*",
    )


def gcp_sink(message):
    """Sink function that formats and writes JSON logs to stderr."""
    record = message.record

    level = record["level"].name
    severity = SEVERITY_MAP.get(level, "DEFAULT")

    log_entry = {
        "severity": severity,
        "message": record["message"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "logger": record["name"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }

    # Add extra fields
    extra = record.get("extra", {})
    if extra:
        for key, value in extra.items():
            if not key.startswith("_"):
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)

    # Add exception info if present
    exception = record.get("exception")
    if exception and exception.type:
        log_entry["exception"] = {
            "type": str(exception.type.__name__),
            "value": str(exception.value) if exception.value else None,
        }

    sys.stderr.write(json.dumps(log_entry) + "\n")


def configure_logging(level: str = "INFO", force_json: bool = False) -> None:
    """Configure logging for the application.

    In GCP (Cloud Run), outputs structured JSON logs.
    Locally, outputs human-readable colored logs.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        force_json: Force JSON output even when not in GCP
    """
    # Remove default loguru handler
    logger.remove()

    # Suppress noisy ADK warnings
    suppress_adk_warnings()

    # Determine format based on environment
    use_json = force_json or is_running_in_gcp()

    if use_json:
        # GCP mode: structured JSON to stderr via custom sink
        logger.add(
            gcp_sink,
            level=level,
            colorize=False,
        )
        logger.info(
            "Logging configured for GCP",
            mode="gcp",
            level=level,
            service=os.getenv("K_SERVICE", "unknown"),
            revision=os.getenv("K_REVISION", "unknown"),
        )
    else:
        # Local mode: human-readable with colors
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            level=level,
            colorize=True,
        )
        logger.info(f"Logging configured for local development (level={level})")


def log_request(
    method: str,
    path: str,
    status_code: int | None = None,
    duration_ms: float | None = None,
    **kwargs,
) -> None:
    """Log an HTTP request with structured fields.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        status_code: Response status code (if completed)
        duration_ms: Request duration in milliseconds
        **kwargs: Additional fields to log
    """
    logger.info(
        f"{method} {path}" + (f" {status_code}" if status_code else ""),
        http_method=method,
        http_path=path,
        http_status=status_code,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_evaluation_start(
    domain: str,
    agent_endpoint: str,
    num_tasks: int | None = None,
    evaluation_id: str | None = None,
    **kwargs,
) -> None:
    """Log the start of an evaluation with structured fields.

    Args:
        domain: Evaluation domain (mock, airline, etc.)
        agent_endpoint: URL of the agent being evaluated
        num_tasks: Number of tasks to run
        evaluation_id: Unique evaluation ID
        **kwargs: Additional fields to log
    """
    logger.info(
        f"Evaluation started: domain={domain}",
        event="evaluation_start",
        domain=domain,
        agent_endpoint=agent_endpoint,
        num_tasks=num_tasks,
        evaluation_id=evaluation_id,
        **kwargs,
    )


def log_evaluation_complete(
    evaluation_id: str,
    domain: str,
    success_rate: float,
    total_tasks: int,
    duration_ms: float | None = None,
    **kwargs,
) -> None:
    """Log the completion of an evaluation with structured fields.

    Args:
        evaluation_id: Unique evaluation ID
        domain: Evaluation domain
        success_rate: Fraction of successful tasks (0.0 to 1.0)
        total_tasks: Total number of tasks evaluated
        duration_ms: Total evaluation duration in milliseconds
        **kwargs: Additional fields to log
    """
    logger.info(
        f"Evaluation completed: domain={domain} success_rate={success_rate:.2%}",
        event="evaluation_complete",
        evaluation_id=evaluation_id,
        domain=domain,
        success_rate=success_rate,
        total_tasks=total_tasks,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_evaluation_error(
    evaluation_id: str | None,
    domain: str,
    error: str,
    error_type: str | None = None,
    **kwargs,
) -> None:
    """Log an evaluation error with structured fields.

    Args:
        evaluation_id: Unique evaluation ID (if available)
        domain: Evaluation domain
        error: Error message
        error_type: Type of error (e.g., "INVALID_PARAMETERS")
        **kwargs: Additional fields to log
    """
    # Escape curly braces in error message to prevent loguru format interpretation
    safe_error = str(error).replace("{", "{{").replace("}", "}}")
    logger.error(
        f"Evaluation failed: domain={domain} error={safe_error}",
        event="evaluation_error",
        evaluation_id=evaluation_id,
        domain=domain,
        error=error,
        error_type=error_type,
        **kwargs,
    )

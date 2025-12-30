"""
Logging configuration for kimi_litellm_agent with GCP Cloud Logging support.

Outputs structured JSON logs when running in GCP Cloud Run,
and human-readable format for local development.
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
    """Check if running in Google Cloud Run."""
    return bool(os.getenv("K_SERVICE"))


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


def suppress_adk_warnings():
    """Suppress ADK experimental warnings that flood the logs."""
    warnings.filterwarnings(
        "ignore",
        message=r"\[EXPERIMENTAL\].*",
        category=UserWarning,
        module=r"google\.adk\..*",
    )


def configure_logging(level: str = "INFO", force_json: bool = False) -> None:
    """Configure logging for the application.

    In GCP (Cloud Run), outputs structured JSON logs.
    Locally, outputs human-readable colored logs.
    """
    logger.remove()
    suppress_adk_warnings()

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

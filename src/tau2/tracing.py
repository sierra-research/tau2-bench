"""Datadog tracing configuration for tau2-bench-agent.

This module configures ddtrace for automatic instrumentation of LLM calls
via LiteLLM and HTTP calls via httpx. Configuration is opt-in via
environment variables.

Environment Variables:
    DD_TRACE_ENABLED: Set to "true" to enable ddtrace instrumentation.
        When disabled (default), this module does nothing.
    DD_LLMOBS_ENABLED: Set to "true" to enable LLM Observability.
        Requires DD_API_KEY to be set.
    DD_SERVICE: Service name for traces. Defaults to "tau2-bench-agent".
    DD_ENV: Environment name. Defaults to "development".
    DD_API_KEY: Datadog API key. Required for agentless mode.
    DD_SITE: Datadog site. Defaults to "datadoghq.com".
    DD_VERSION: Service version. Defaults to tau2 package version.

Usage:
    Call configure_ddtrace() at application startup, before any LLM calls.

    from tau2.tracing import configure_ddtrace
    configure_ddtrace()

Note:
    This configuration uses agentless mode, sending data directly to
    Datadog intake API without requiring a local Datadog Agent. This
    is suitable for serverless environments like Cloud Run.
"""

from __future__ import annotations

import os

from loguru import logger


def _get_tau2_version() -> str:
    """Get tau2 package version for service tagging."""
    try:
        from importlib.metadata import version

        return version("tau2")
    except Exception:
        return "unknown"


def configure_ddtrace() -> bool:
    """Configure ddtrace for automatic LLM and HTTP instrumentation.

    This function is idempotent and can be called multiple times safely.
    Configuration only happens when DD_TRACE_ENABLED=true.

    Returns:
        True if ddtrace was configured, False if disabled or already configured.

    Raises:
        No exceptions are raised. All errors are logged and the function
        returns False to allow graceful degradation.
    """
    if os.getenv("DD_TRACE_ENABLED", "false").lower() != "true":
        logger.debug("Datadog tracing disabled (DD_TRACE_ENABLED != true)")
        return False

    try:
        from ddtrace import patch, tracer

        if tracer._initialized:
            logger.debug("ddtrace already initialized, skipping configuration")
            return False

        service_name = os.getenv("DD_SERVICE", "tau2-bench-agent")
        env_name = os.getenv("DD_ENV", "development")
        version = os.getenv("DD_VERSION", _get_tau2_version())

        tracer.set_tags(
            {
                "service": service_name,
                "env": env_name,
                "version": version,
                "tau2.version": _get_tau2_version(),
            }
        )

        patch(litellm=True, httpx=True)

        logger.info(
            f"Datadog tracing configured: service={service_name}, "
            f"env={env_name}, version={version}"
        )

        if os.getenv("DD_LLMOBS_ENABLED", "false").lower() == "true":
            _configure_llmobs(service_name)

        return True

    except ImportError as e:
        logger.warning(f"ddtrace not installed, tracing disabled: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to configure ddtrace: {e}")
        return False


def _configure_llmobs(service_name: str) -> bool:
    """Configure Datadog LLM Observability in agentless mode.

    Args:
        service_name: The service name to use for LLM Observability.

    Returns:
        True if LLM Observability was enabled, False otherwise.
    """
    try:
        from ddtrace.llmobs import LLMObs

        api_key = os.getenv("DD_API_KEY")
        if not api_key:
            logger.warning(
                "DD_LLMOBS_ENABLED=true but DD_API_KEY not set, "
                "LLM Observability disabled"
            )
            return False

        LLMObs.enable(
            ml_app=service_name,
            agentless_enabled=True,
        )

        logger.info(f"LLM Observability enabled for ml_app={service_name}")
        return True

    except ImportError as e:
        logger.warning(f"LLMObs not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to enable LLM Observability: {e}")
        return False

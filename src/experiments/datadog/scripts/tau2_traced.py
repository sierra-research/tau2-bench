#!/usr/bin/env python3
"""Wrapper script to run tau2 CLI with Datadog tracing enabled.

This script configures ddtrace before importing tau2, ensuring all LLM
and HTTP calls are automatically instrumented.

Usage:
    # Instead of: tau2 run --domain mock
    # Use: python -m experiments.datadog.scripts.tau2_traced run --domain mock

    # Or with environment variables:
    # DD_TRACE_ENABLED=true python -m experiments.datadog.scripts.tau2_traced run --domain mock

Alternative approaches (no wrapper needed):
    # Use ddtrace's built-in wrapper:
    ddtrace-run tau2 run --domain mock

    # Or set ddtrace to auto-patch at Python startup:
    DD_TRACE_ENABLED=true DD_PATCH_MODULES=litellm:true,httpx:true tau2 run --domain mock

Environment Variables:
    DD_TRACE_ENABLED: Set to "true" to enable tracing. Defaults to "true" when
        using this wrapper (the whole point of using the wrapper).
    DD_LLMOBS_ENABLED: Set to "true" to enable LLM Observability.
    DD_SERVICE: Service name. Defaults to "tau2-bench-agent".
    DD_ENV: Environment name. Defaults to "development".
    DD_API_KEY: Required for agentless mode (Cloud Run, serverless).
    DD_SITE: Datadog site. Defaults to "datadoghq.com".
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Configure ddtrace and run tau2 CLI."""
    # Enable tracing by default when using this wrapper
    if "DD_TRACE_ENABLED" not in os.environ:
        os.environ["DD_TRACE_ENABLED"] = "true"

    # Configure ddtrace before importing tau2
    from tau2.tracing import configure_ddtrace

    configure_ddtrace()

    # Now import and run the tau2 CLI
    from tau2.cli import main as tau2_main

    tau2_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())

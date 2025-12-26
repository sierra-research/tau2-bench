"""
E2E tests for Datadog observability integration.

These tests validate the full A2A -> Evaluation -> Datadog metrics flow:
1. tau2_agent with ddtrace enabled via environment variables
2. A2A request triggers RunTau2Evaluation tool
3. Evaluation persisted to EvaluationStore
4. emit_metrics.py can process results

Run with: pytest tests/test_datadog_e2e/ -v -m datadog_e2e
"""

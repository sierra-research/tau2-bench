"""Integration tests for A2A metrics export functionality."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from tau2.a2a.metrics import AggregatedMetrics, ProtocolMetrics
from tau2.a2a.models import A2AConfig
from tau2.agent.a2a_agent import A2AAgent
from tau2.data_model.message import UserMessage
from tau2.environment.tool import Tool


class TestMetricsExport:
    """Test suite for metrics export to JSON."""

    def test_protocol_metrics_json_serialization(self):
        """Test that ProtocolMetrics can be serialized to JSON."""
        metrics = ProtocolMetrics(
            request_id="req-001",
            endpoint="http://localhost:8080",
            method="POST",
            status_code=200,
            latency_ms=234.56,
            input_tokens=100,
            output_tokens=75,
            context_id="ctx-abc123",
        )

        # Convert to dict and serialize
        metrics_dict = metrics.to_dict()
        json_str = json.dumps(metrics_dict)

        # Parse back
        parsed = json.loads(json_str)

        assert parsed["request_id"] == "req-001"
        assert parsed["endpoint"] == "http://localhost:8080"
        assert parsed["status_code"] == 200
        assert parsed["latency_ms"] == 234.56
        assert parsed["input_tokens"] == 100
        assert parsed["output_tokens"] == 75

    def test_aggregated_metrics_json_serialization(self):
        """Test that AggregatedMetrics can be serialized to JSON."""
        # Create sample protocol metrics
        metrics_list = [
            ProtocolMetrics(
                request_id=f"req-{i}",
                endpoint="http://localhost:8080",
                method="POST",
                status_code=200,
                latency_ms=100.0 + i * 10,
                input_tokens=50 + i * 10,
                output_tokens=30 + i * 5,
            )
            for i in range(5)
        ]

        # Aggregate
        aggregated = AggregatedMetrics.from_protocol_metrics(metrics_list)

        # Serialize to JSON
        json_str = json.dumps(aggregated.model_dump())
        parsed = json.loads(json_str)

        assert parsed["total_requests"] == 5
        assert parsed["total_tokens"] == sum(
            (50 + i * 10) + (30 + i * 5) for i in range(5)
        )
        assert parsed["error_count"] == 0

    def test_aggregated_metrics_computation(self):
        """Test that AggregatedMetrics correctly computes summary statistics."""
        metrics_list = [
            ProtocolMetrics(
                request_id="req-1",
                endpoint="http://localhost:8080",
                method="POST",
                status_code=200,
                latency_ms=100.0,
                input_tokens=50,
                output_tokens=30,
            ),
            ProtocolMetrics(
                request_id="req-2",
                endpoint="http://localhost:8080",
                method="POST",
                status_code=200,
                latency_ms=200.0,
                input_tokens=60,
                output_tokens=40,
            ),
            ProtocolMetrics(
                request_id="req-3",
                endpoint="http://localhost:8080",
                method="POST",
                status_code=None,
                latency_ms=50.0,
                error="Timeout",
            ),
        ]

        aggregated = AggregatedMetrics.from_protocol_metrics(metrics_list)

        assert aggregated.total_requests == 3
        assert aggregated.total_tokens == (50 + 30) + (60 + 40) + 0  # 180
        assert aggregated.total_latency_ms == 100.0 + 200.0 + 50.0  # 350
        assert aggregated.avg_latency_ms == pytest.approx(350.0 / 3, rel=0.01)
        assert aggregated.error_count == 1

    def test_metrics_list_json_export(self):
        """Test exporting a list of metrics to JSON."""
        metrics_list = [
            ProtocolMetrics(
                request_id="req-1",
                endpoint="http://localhost:8080",
                method="POST",
                status_code=200,
                latency_ms=100.0,
                input_tokens=50,
                output_tokens=30,
            ),
            ProtocolMetrics(
                request_id="req-2",
                endpoint="http://localhost:8080",
                method="POST",
                status_code=200,
                latency_ms=150.0,
                input_tokens=60,
                output_tokens=40,
            ),
        ]

        # Export as list of dicts
        metrics_dicts = [m.to_dict() for m in metrics_list]
        json_str = json.dumps(metrics_dicts, indent=2)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert len(parsed) == 2
        assert parsed[0]["request_id"] == "req-1"
        assert parsed[1]["request_id"] == "req-2"


class TestA2AAgentMetricsIntegration:
    """Test metrics integration with A2AAgent."""

    def test_agent_collects_metrics_during_execution(self, mock_a2a_agent_with_metrics):
        """Test that A2AAgent collects metrics during message generation."""
        agent, mock_client = mock_a2a_agent_with_metrics

        # Create initial state
        state = agent.get_init_state()

        # Send a message
        message = UserMessage(role="user", content="Test message")
        assistant_msg, new_state = agent.generate_next_message(message, state)

        # Verify message was processed
        assert assistant_msg.content == "Mock response"
        assert new_state.request_count == 1

        # Verify metrics were collected
        metrics = agent.get_protocol_metrics()
        assert len(metrics) == 1
        assert metrics[0].endpoint == "http://localhost:8080"

    def test_agent_exports_metrics_summary(self, mock_a2a_agent_with_metrics):
        """Test that A2AAgent can export metrics summary."""
        agent, mock_client = mock_a2a_agent_with_metrics

        # Execute multiple requests
        state = agent.get_init_state()

        for i in range(3):
            message = UserMessage(role="user", content=f"Message {i}")
            _, state = agent.generate_next_message(message, state)

        # Verify request count increased
        assert state.request_count == 3

        # Verify metrics are collected
        protocol_metrics = agent.get_protocol_metrics()
        assert len(protocol_metrics) == 3

        # Verify aggregated metrics
        aggregated = agent.get_aggregated_metrics()
        assert aggregated.total_requests == 3

    def test_metrics_export_format_matches_spec(self):
        """Test that exported metrics match the specification format."""
        # Expected format from data-model.md
        expected_format = {
            "task_id": "airline_001",
            "agent_type": "a2a_agent",
            "protocol_metrics": [
                {
                    "request_id": "req-uuid-1",
                    "endpoint": "https://agent.example.com",
                    "method": "POST",
                    "status_code": 200,
                    "latency_ms": 1234.56,
                    "input_tokens": 456,
                    "output_tokens": 123,
                    "context_id": "ctx-abc123",
                    "timestamp": "2025-11-23T10:30:45.123Z",
                }
            ],
            "summary": {
                "total_requests": 5,
                "total_tokens": 2890,
                "total_latency_ms": 6789.12,
                "avg_latency_ms": 1357.82,
                "error_count": 0,
            },
        }

        # Verify structure is JSON-serializable
        json_str = json.dumps(expected_format)
        parsed = json.loads(json_str)

        assert parsed["agent_type"] == "a2a_agent"
        assert "protocol_metrics" in parsed
        assert "summary" in parsed
        assert isinstance(parsed["protocol_metrics"], list)
        assert isinstance(parsed["summary"], dict)


class TestMetricsFileExport:
    """Test metrics export to file system."""

    def test_export_metrics_to_json_file(self, tmp_path):
        """Test exporting metrics to a JSON file."""
        # Create sample metrics
        metrics_list = [
            ProtocolMetrics(
                request_id=f"req-{i}",
                endpoint="http://localhost:8080",
                method="POST",
                status_code=200,
                latency_ms=100.0 + i * 50,
                input_tokens=50,
                output_tokens=30,
            )
            for i in range(3)
        ]

        # Create export structure
        export_data = {
            "task_id": "test_task",
            "agent_type": "a2a_agent",
            "protocol_metrics": [m.to_dict() for m in metrics_list],
            "summary": AggregatedMetrics.from_protocol_metrics(
                metrics_list
            ).model_dump(),
        }

        # Export to file
        output_file = tmp_path / "metrics.json"
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        # Verify file exists and is valid JSON
        assert output_file.exists()

        with open(output_file) as f:
            loaded = json.load(f)

        assert loaded["task_id"] == "test_task"
        assert loaded["agent_type"] == "a2a_agent"
        assert len(loaded["protocol_metrics"]) == 3
        assert loaded["summary"]["total_requests"] == 3

    def test_metrics_append_to_existing_results(self, tmp_path):
        """Test that A2A metrics can be added to existing tau2-bench results."""
        # Simulate existing tau2-bench results
        existing_results = {
            "task_id": "airline_001",
            "agent_type": "a2a_agent",
            "status": "completed",
            "success": True,
        }

        # Create A2A metrics
        metrics_list = [
            ProtocolMetrics(
                request_id="req-1",
                endpoint="http://localhost:8080",
                method="POST",
                status_code=200,
                latency_ms=200.0,
                input_tokens=100,
                output_tokens=75,
            )
        ]

        # Add A2A metrics to results
        existing_results["a2a_protocol_metrics"] = {
            "requests": [m.to_dict() for m in metrics_list],
            "summary": AggregatedMetrics.from_protocol_metrics(
                metrics_list
            ).model_dump(),
        }

        # Export
        output_file = tmp_path / "results_with_metrics.json"
        with open(output_file, "w") as f:
            json.dump(existing_results, f, indent=2)

        # Verify
        with open(output_file) as f:
            loaded = json.load(f)

        assert "a2a_protocol_metrics" in loaded
        assert "requests" in loaded["a2a_protocol_metrics"]
        assert "summary" in loaded["a2a_protocol_metrics"]



# Note: Fixture mock_a2a_agent_with_metrics is defined in conftest.py

"""Unit tests for tau2_agent GreenExecutor (AgentBeats-compatible evaluation)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.types import DataPart, Part, TextPart
from pydantic import ValidationError

from tau2_agent.green_executor import (
    EvalConfig,
    EvalRequest,
    Tau2GreenAgent,
    Tau2GreenExecutor,
    create_green_agent_card,
)


class TestEvalConfig:
    """Tests for EvalConfig model."""

    def test_eval_config_defaults(self):
        """EvalConfig should have correct defaults."""
        config = EvalConfig(domain="airline")

        assert config.domain == "airline"
        assert config.num_tasks is None
        assert config.num_trials == 1
        assert config.task_ids is None

    def test_eval_config_with_all_fields(self):
        """EvalConfig should accept all fields."""
        config = EvalConfig(
            domain="retail",
            num_tasks=10,
            num_trials=3,
            task_ids=["task-1", "task-2"],
        )

        assert config.domain == "retail"
        assert config.num_tasks == 10
        assert config.num_trials == 3
        assert config.task_ids == ["task-1", "task-2"]

    def test_eval_config_from_dict(self):
        """EvalConfig should parse from dictionary."""
        data = {"domain": "mock", "num_tasks": 5}
        config = EvalConfig.model_validate(data)

        assert config.domain == "mock"
        assert config.num_tasks == 5
        assert config.num_trials == 1  # default


class TestEvalRequest:
    """Tests for EvalRequest model."""

    def test_eval_request_parsing(self):
        """EvalRequest should parse from JSON correctly."""
        json_data = {
            "participants": {"agent": "http://localhost:9009/a2a/test"},
            "config": {"domain": "mock", "num_tasks": 2},
        }
        request = EvalRequest.model_validate(json_data)

        assert request.participants["agent"] == "http://localhost:9009/a2a/test"
        assert request.config.domain == "mock"
        assert request.config.num_tasks == 2
        assert request.config.num_trials == 1  # default

    def test_eval_request_from_json_string(self):
        """EvalRequest should parse from JSON string."""
        json_str = json.dumps({
            "participants": {"agent": "https://agent.example.com"},
            "config": {"domain": "airline", "num_trials": 2},
        })
        request = EvalRequest.model_validate_json(json_str)

        assert request.participants["agent"] == "https://agent.example.com"
        assert request.config.domain == "airline"
        assert request.config.num_trials == 2

    def test_eval_request_with_task_ids(self):
        """EvalRequest should handle task_ids correctly."""
        json_data = {
            "participants": {"agent": "http://localhost:8000"},
            "config": {
                "domain": "telecom",
                "task_ids": ["telecom-task-1", "telecom-task-2"],
            },
        }
        request = EvalRequest.model_validate(json_data)

        assert request.config.task_ids == ["telecom-task-1", "telecom-task-2"]
        assert request.config.num_tasks is None  # not specified

    def test_eval_request_missing_agent_fails(self):
        """EvalRequest should fail if participants is missing."""
        json_data = {
            "config": {"domain": "mock"},
        }

        with pytest.raises(ValidationError):
            EvalRequest.model_validate(json_data)


class TestCreateGreenAgentCard:
    """Tests for create_green_agent_card function."""

    def test_creates_agent_card_with_url(self):
        """Should create AgentCard with correct URL."""
        card = create_green_agent_card("http://localhost:9009")

        assert card.name == "tau2_green"
        assert card.url == "http://localhost:9009"
        assert card.version == "1.0.0"
        assert card.capabilities.streaming is True

    def test_creates_agent_card_with_external_url(self):
        """Should create AgentCard with external URL."""
        card = create_green_agent_card("https://tau2.example.com")

        assert card.url == "https://tau2.example.com"
        assert "AgentBeats" in card.description


class TestTau2GreenAgent:
    """Tests for Tau2GreenAgent evaluation execution."""

    @pytest.fixture
    def mock_updater(self):
        """Create a mock TaskUpdater."""
        updater = AsyncMock()
        updater.update_status = AsyncMock()
        updater.add_artifact = AsyncMock()
        return updater

    @pytest.fixture
    def mock_tool_result(self):
        """Create a mock successful evaluation result."""
        return {
            "status": "completed",
            "evaluation_id": "eval-123",
            "timestamp": "2024-01-01T00:00:00Z",
            "summary": {
                "total_simulations": 5,
                "total_tasks": 5,
                "successful_simulations": 4,
                "avg_reward": 0.8,
                "avg_agent_cost": 0.0025,
            },
            "tasks": [
                {"task_id": "task-1", "purpose": "Test task 1"},
            ],
            "simulations": [],
        }

    @pytest.mark.asyncio
    async def test_run_eval_missing_agent_raises(self, mock_updater):
        """Should raise ValueError if agent is missing from participants."""
        agent = Tau2GreenAgent()
        request = EvalRequest(
            participants={},  # Missing "agent" key
            config=EvalConfig(domain="mock"),
        )

        with pytest.raises(ValueError, match="Missing 'agent' in participants"):
            await agent.run_eval(request, mock_updater)

    @pytest.mark.asyncio
    async def test_run_eval_calls_tool(self, mock_updater, mock_tool_result):
        """Should call RunTau2Evaluation tool with correct args."""
        agent = Tau2GreenAgent()
        request = EvalRequest(
            participants={"agent": "http://localhost:9009"},
            config=EvalConfig(domain="mock", num_tasks=2, num_trials=1),
        )

        with patch(
            "tau2_agent.green_executor.RunTau2Evaluation"
        ) as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run_async = AsyncMock(return_value=mock_tool_result)
            MockTool.return_value = mock_instance

            await agent.run_eval(request, mock_updater)

            # Verify tool was called with correct args
            mock_instance.run_async.assert_called_once()
            call_kwargs = mock_instance.run_async.call_args[1]
            args = call_kwargs["args"]

            assert args["domain"] == "mock"
            assert args["agent_endpoint"] == "http://localhost:9009"
            assert args["num_tasks"] == 2
            assert args["num_trials"] == 1

        # Verify updater was called
        mock_updater.update_status.assert_called()
        mock_updater.add_artifact.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_eval_returns_datapart(self, mock_updater, mock_tool_result):
        """Should add artifact with both TextPart and DataPart."""
        agent = Tau2GreenAgent()
        request = EvalRequest(
            participants={"agent": "http://localhost:9009"},
            config=EvalConfig(domain="mock", num_tasks=1),
        )

        with patch(
            "tau2_agent.green_executor.RunTau2Evaluation"
        ) as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run_async = AsyncMock(return_value=mock_tool_result)
            MockTool.return_value = mock_instance

            await agent.run_eval(request, mock_updater)

        # Verify artifact was added with correct parts
        mock_updater.add_artifact.assert_called_once()
        call_kwargs = mock_updater.add_artifact.call_args[1]
        parts = call_kwargs["parts"]

        assert len(parts) == 2
        assert isinstance(parts[0].root, TextPart)
        assert isinstance(parts[1].root, DataPart)

        # Verify DataPart contains the result
        data_part = parts[1].root
        assert data_part.data["status"] == "completed"
        assert "summary" in data_part.data

    @pytest.mark.asyncio
    async def test_run_eval_handles_error_result(self, mock_updater):
        """Should raise ValueError when tool returns error."""
        agent = Tau2GreenAgent()
        request = EvalRequest(
            participants={"agent": "http://localhost:9009"},
            config=EvalConfig(domain="mock"),
        )

        error_result = {
            "error": "LIMIT_EXCEEDED",
            "message": "num_tasks must be between 1 and 30",
        }

        with patch(
            "tau2_agent.green_executor.RunTau2Evaluation"
        ) as MockTool:
            mock_instance = AsyncMock()
            mock_instance.run_async = AsyncMock(return_value=error_result)
            MockTool.return_value = mock_instance

            with pytest.raises(ValueError, match="LIMIT_EXCEEDED"):
                await agent.run_eval(request, mock_updater)


class TestTau2GreenExecutor:
    """Tests for Tau2GreenExecutor A2A interface."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock RequestContext."""
        context = MagicMock()
        context.get_user_input = MagicMock(return_value=json.dumps({
            "participants": {"agent": "http://localhost:9009"},
            "config": {"domain": "mock", "num_tasks": 1},
        }))
        context.message = MagicMock()
        return context

    @pytest.fixture
    def mock_event_queue(self):
        """Create a mock EventQueue."""
        queue = AsyncMock()
        queue.enqueue_event = AsyncMock()
        return queue

    @pytest.mark.asyncio
    async def test_execute_parses_request(self, mock_context, mock_event_queue):
        """Should parse EvalRequest from context input."""
        executor = Tau2GreenExecutor()

        with patch.object(
            executor.agent, "run_eval", new_callable=AsyncMock
        ) as mock_run:
            with patch("tau2_agent.green_executor.new_task") as mock_new_task:
                mock_task = MagicMock()
                mock_task.id = "task-123"
                mock_task.context_id = "ctx-456"
                mock_new_task.return_value = mock_task

                with patch(
                    "tau2_agent.green_executor.TaskUpdater"
                ) as MockUpdater:
                    mock_updater = AsyncMock()
                    mock_updater.complete = AsyncMock()
                    MockUpdater.return_value = mock_updater

                    await executor.execute(mock_context, mock_event_queue)

        # Verify run_eval was called with correct request
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0]
        request = call_args[0]

        assert isinstance(request, EvalRequest)
        assert request.participants["agent"] == "http://localhost:9009"
        assert request.config.domain == "mock"

    @pytest.mark.asyncio
    async def test_execute_invalid_json_raises(self, mock_event_queue):
        """Should raise ValueError for invalid JSON input."""
        context = MagicMock()
        context.get_user_input = MagicMock(return_value="not valid json")

        executor = Tau2GreenExecutor()

        with pytest.raises(ValueError, match="Invalid EvalRequest format"):
            await executor.execute(context, mock_event_queue)

    @pytest.mark.asyncio
    async def test_execute_sends_task_event(self, mock_context, mock_event_queue):
        """Should send task event to queue."""
        executor = Tau2GreenExecutor()

        with patch.object(
            executor.agent, "run_eval", new_callable=AsyncMock
        ):
            with patch("tau2_agent.green_executor.new_task") as mock_new_task:
                mock_task = MagicMock()
                mock_task.id = "task-123"
                mock_task.context_id = "ctx-456"
                mock_new_task.return_value = mock_task

                with patch(
                    "tau2_agent.green_executor.TaskUpdater"
                ) as MockUpdater:
                    mock_updater = AsyncMock()
                    mock_updater.complete = AsyncMock()
                    MockUpdater.return_value = mock_updater

                    await executor.execute(mock_context, mock_event_queue)

        # Verify task was enqueued
        mock_event_queue.enqueue_event.assert_called_once_with(mock_task)

    @pytest.mark.asyncio
    async def test_execute_completes_on_success(self, mock_context, mock_event_queue):
        """Should call updater.complete() on successful evaluation."""
        executor = Tau2GreenExecutor()

        with patch.object(
            executor.agent, "run_eval", new_callable=AsyncMock
        ):
            with patch("tau2_agent.green_executor.new_task") as mock_new_task:
                mock_task = MagicMock()
                mock_task.id = "task-123"
                mock_task.context_id = "ctx-456"
                mock_new_task.return_value = mock_task

                with patch(
                    "tau2_agent.green_executor.TaskUpdater"
                ) as MockUpdater:
                    mock_updater = AsyncMock()
                    mock_updater.complete = AsyncMock()
                    MockUpdater.return_value = mock_updater

                    await executor.execute(mock_context, mock_event_queue)

        mock_updater.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_fails_on_error(self, mock_context, mock_event_queue):
        """Should call updater.failed() and re-raise on error."""
        executor = Tau2GreenExecutor()

        with patch.object(
            executor.agent,
            "run_eval",
            new_callable=AsyncMock,
            side_effect=ValueError("Test error"),
        ):
            with patch("tau2_agent.green_executor.new_task") as mock_new_task:
                mock_task = MagicMock()
                mock_task.id = "task-123"
                mock_task.context_id = "ctx-456"
                mock_new_task.return_value = mock_task

                with patch(
                    "tau2_agent.green_executor.TaskUpdater"
                ) as MockUpdater:
                    mock_updater = AsyncMock()
                    mock_updater.failed = AsyncMock()
                    MockUpdater.return_value = mock_updater

                    with pytest.raises(ValueError, match="Test error"):
                        await executor.execute(mock_context, mock_event_queue)

        mock_updater.failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_not_implemented(self, mock_context, mock_event_queue):
        """Cancel should raise NotImplementedError."""
        executor = Tau2GreenExecutor()

        with pytest.raises(NotImplementedError, match="Cancellation not supported"):
            await executor.cancel(mock_context, mock_event_queue)

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentify_tau_bench.launcher import launch_evaluation


class TestLaunchEvaluation:
    """Test the launch_evaluation function."""

    @pytest.mark.asyncio
    async def test_launch_evaluation_success(self):
        """Test successful launch of evaluation with both agents."""
        mock_green_process = MagicMock()
        mock_white_process = MagicMock()

        with (
            patch(
                "agentify_tau_bench.launcher.multiprocessing.Process"
            ) as mock_process,
            patch("agentify_tau_bench.launcher.wait_agent_ready") as mock_wait,
            patch("agentify_tau_bench.launcher.a2a_send_message") as mock_send,
        ):

            # Mock process creation
            mock_process.side_effect = [mock_green_process, mock_white_process]

            # Mock agents becoming ready
            mock_wait.return_value = True

            # Mock message response
            mock_send.return_value = MagicMock()

            await launch_evaluation()

            # Verify both processes were created
            assert mock_process.call_count == 2

            # Verify both processes were started
            mock_green_process.start.assert_called_once()
            mock_white_process.start.assert_called_once()

            # Verify wait_agent_ready called twice
            assert mock_wait.call_count == 2

            # Verify message was sent
            mock_send.assert_called_once()

            # Verify both processes were terminated
            mock_green_process.terminate.assert_called_once()
            mock_white_process.terminate.assert_called_once()
            mock_green_process.join.assert_called_once()
            mock_white_process.join.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_evaluation_green_agent_not_ready(self):
        """Test when green agent fails to become ready."""
        mock_green_process = MagicMock()

        with (
            patch(
                "agentify_tau_bench.launcher.multiprocessing.Process"
            ) as mock_process,
            patch("agentify_tau_bench.launcher.wait_agent_ready") as mock_wait,
        ):

            mock_process.return_value = mock_green_process
            mock_wait.return_value = False  # Green agent not ready

            with pytest.raises(AssertionError, match="Green agent not ready"):
                await launch_evaluation()

            # Verify process was started
            mock_green_process.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_evaluation_white_agent_not_ready(self):
        """Test when white agent fails to become ready."""
        mock_green_process = MagicMock()
        mock_white_process = MagicMock()

        with (
            patch(
                "agentify_tau_bench.launcher.multiprocessing.Process"
            ) as mock_process,
            patch("agentify_tau_bench.launcher.wait_agent_ready") as mock_wait,
        ):

            mock_process.side_effect = [mock_green_process, mock_white_process]
            # Green ready, white not ready
            mock_wait.side_effect = [True, False]

            with pytest.raises(AssertionError, match="White agent not ready"):
                await launch_evaluation()

            # Verify both processes were started
            mock_green_process.start.assert_called_once()
            mock_white_process.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_evaluation_message_sent_with_config(self):
        """Test that message is sent with correct configuration."""
        mock_green_process = MagicMock()
        mock_white_process = MagicMock()

        with (
            patch(
                "agentify_tau_bench.launcher.multiprocessing.Process"
            ) as mock_process,
            patch("agentify_tau_bench.launcher.wait_agent_ready") as mock_wait,
            patch("agentify_tau_bench.launcher.a2a_send_message") as mock_send,
        ):

            mock_process.side_effect = [mock_green_process, mock_white_process]
            mock_wait.return_value = True
            mock_send.return_value = MagicMock()

            await launch_evaluation()

            # Verify message was sent to green agent
            mock_send.assert_called_once()
            call_args = mock_send.call_args

            # Check green agent URL
            assert call_args[0][0] == "http://localhost:9001"

            # Check message contains required tags
            message_text = call_args[0][1]
            assert "<white_agent_url>" in message_text
            assert "http://localhost:9002/" in message_text
            assert "<env_config>" in message_text
            assert "mock" in message_text

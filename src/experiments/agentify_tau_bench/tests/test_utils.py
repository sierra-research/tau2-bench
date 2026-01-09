from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentify_tau_bench.utils import (
    a2a_send_message,
    get_agent_card,
    parse_tags,
    wait_agent_ready,
)


class TestParseTags:
    """Test the parse_tags function."""

    def test_parse_single_tag(self):
        """Test parsing a single tag."""
        result = parse_tags("<tag1>Hello</tag1>")
        assert result == {"tag1": "Hello"}

    def test_parse_multiple_tags(self):
        """Test parsing multiple tags."""
        result = parse_tags("<tag1>Hello</tag1> some text <tag2>World</tag2>")
        assert result == {"tag1": "Hello", "tag2": "World"}

    def test_parse_nested_content(self):
        """Test parsing tags with nested content."""
        result = parse_tags("<outer>Some <inner>nested</inner> content</outer>")
        assert "outer" in result

    def test_parse_multiline_content(self):
        """Test parsing tags with multiline content."""
        text = """<tag1>
        Line 1
        Line 2
        </tag1>"""
        result = parse_tags(text)
        assert "tag1" in result
        assert "Line 1" in result["tag1"]
        assert "Line 2" in result["tag1"]

    def test_parse_empty_tag(self):
        """Test parsing empty tags."""
        result = parse_tags("<tag1></tag1>")
        assert result == {"tag1": ""}

    def test_parse_no_tags(self):
        """Test parsing string with no tags."""
        result = parse_tags("Just plain text")
        assert result == {}

    def test_parse_whitespace_handling(self):
        """Test that leading/trailing whitespace is stripped."""
        result = parse_tags("<tag1>  content with spaces  </tag1>")
        assert result == {"tag1": "content with spaces"}


class TestA2AUtils:
    """Test the A2A utility functions."""

    @pytest.mark.asyncio
    async def test_get_agent_card_success(self):
        """Test getting agent card successfully."""
        mock_card = MagicMock()

        with patch(
            "agentify_tau_bench.utils.a2a_utils.A2ACardResolver"
        ) as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_resolver.get_agent_card.return_value = mock_card
            mock_resolver_class.return_value = mock_resolver

            result = await get_agent_card("http://test-url")

            assert result == mock_card
            mock_resolver.get_agent_card.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_agent_card_none(self):
        """Test getting agent card when it returns None."""
        with patch(
            "agentify_tau_bench.utils.a2a_utils.A2ACardResolver"
        ) as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_resolver.get_agent_card.return_value = None
            mock_resolver_class.return_value = mock_resolver

            result = await get_agent_card("http://test-url")

            assert result is None

    @pytest.mark.asyncio
    async def test_wait_agent_ready_success(self):
        """Test waiting for agent to be ready - immediate success."""
        with patch(
            "agentify_tau_bench.utils.a2a_utils.get_agent_card"
        ) as mock_get_card:
            mock_get_card.return_value = MagicMock()

            result = await wait_agent_ready("http://test-url", timeout=5)

            assert result is True
            mock_get_card.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_agent_ready_timeout(self):
        """Test waiting for agent to be ready - timeout."""
        with patch(
            "agentify_tau_bench.utils.a2a_utils.get_agent_card"
        ) as mock_get_card:
            mock_get_card.return_value = None

            result = await wait_agent_ready("http://test-url", timeout=1)

            assert result is False

    @pytest.mark.asyncio
    async def test_a2a_send_message_basic(self):
        """Test sending a message via A2A."""
        mock_response = MagicMock()

        with (
            patch("agentify_tau_bench.utils.a2a_utils.get_agent_card") as mock_get_card,
            patch("agentify_tau_bench.utils.a2a_utils.A2AClient") as mock_client_class,
        ):

            mock_get_card.return_value = MagicMock()
            mock_client = AsyncMock()
            mock_client.send_message.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = await a2a_send_message("http://test-url", "Hello")

            assert result == mock_response
            mock_client.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_a2a_send_message_with_ids(self):
        """Test sending a message with task_id and context_id."""
        mock_response = MagicMock()

        with (
            patch("agentify_tau_bench.utils.a2a_utils.get_agent_card") as mock_get_card,
            patch("agentify_tau_bench.utils.a2a_utils.A2AClient") as mock_client_class,
        ):

            mock_get_card.return_value = MagicMock()
            mock_client = AsyncMock()
            mock_client.send_message.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = await a2a_send_message(
                "http://test-url", "Hello", task_id="task123", context_id="ctx456"
            )

            assert result == mock_response
            mock_client.send_message.assert_called_once()

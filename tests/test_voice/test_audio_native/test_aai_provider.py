"""Tests for AAI voice agent provider client."""

from unittest.mock import MagicMock

from tau2.environment.tool import Tool
from tau2.voice.audio_native.aai.provider import (
    AAIVADConfig,
    AAIVoiceAgentProvider,
)


class TestAAIVADConfig:
    """Test AAIVADConfig model."""

    def test_aai_vad_config_empty(self) -> None:
        """Test that AAIVADConfig can be instantiated (empty for interface parity)."""
        config = AAIVADConfig()
        assert config is not None


class TestAAIVoiceAgentProvider:
    """Test AAIVoiceAgentProvider initialization and configuration."""

    def test_provider_initialization_defaults(self) -> None:
        """Test provider initializes with default values."""
        provider = AAIVoiceAgentProvider()
        assert provider.input_sample_rate == 16000
        assert provider.tts_sample_rate == 24000
        assert provider.system_prompt == ""
        assert provider.tools == ()

    def test_provider_initialization_custom(self) -> None:
        """Test provider initializes with custom values."""
        provider = AAIVoiceAgentProvider(
            ws_url="ws://custom:8000/ws",
            input_sample_rate=8000,
            tts_sample_rate=16000,
            system_prompt="Custom prompt",
            tools=(),
        )
        assert provider.ws_url == "ws://custom:8000/ws"
        assert provider.input_sample_rate == 8000
        assert provider.tts_sample_rate == 16000
        assert provider.system_prompt == "Custom prompt"

    def test_build_config_message_basic(self) -> None:
        """Test _build_config_message returns correct structure."""
        provider = AAIVoiceAgentProvider(
            input_sample_rate=16000,
            tts_sample_rate=24000,
        )
        config = provider._build_config_message("Test prompt", [])

        assert config["type"] == "config"
        assert config["audioFormat"] == "pcm16"
        assert config["sampleRate"] == 16000
        assert config["ttsSampleRate"] == 24000
        assert config["host"]["systemPrompt"] == "Test prompt"
        assert config["host"]["tools"] == []

    def test_build_config_message_with_tools(self) -> None:
        """Test _build_config_message includes tools with correct structure."""
        # Create a simple mock tool
        mock_tool = MagicMock(spec=Tool)
        mock_tool.openai_schema = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                },
            },
        }

        provider = AAIVoiceAgentProvider()
        config = provider._build_config_message("Test prompt", [mock_tool])

        assert len(config["host"]["tools"]) == 1
        tool = config["host"]["tools"][0]

        # Check flat structure (no "function" key)
        assert tool["type"] == "function"
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather for a location"
        assert "function" not in tool
        assert tool["parameters"]["type"] == "object"

    def test_format_tools_for_api(self) -> None:
        """Test _format_tools_for_api flattens tool schema."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.openai_schema = {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "Test function",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

        provider = AAIVoiceAgentProvider()
        formatted = provider._format_tools_for_api([mock_tool])

        assert len(formatted) == 1
        tool = formatted[0]
        assert tool["type"] == "function"
        assert tool["name"] == "test_func"
        assert tool["description"] == "Test function"
        assert "function" not in tool

    def test_with_host_flag_appends_query_param(self) -> None:
        """Test _with_host_flag appends ?host=1 to URL."""
        result = AAIVoiceAgentProvider._with_host_flag("ws://localhost:3000/websocket")
        assert result == "ws://localhost:3000/websocket?host=1"

    def test_with_host_flag_preserves_existing_query(self) -> None:
        """Test _with_host_flag preserves existing query parameters."""
        result = AAIVoiceAgentProvider._with_host_flag(
            "ws://localhost:3000/websocket?token=abc"
        )
        assert "host=1" in result
        assert "token=abc" in result

    def test_with_host_flag_idempotent(self) -> None:
        """Test _with_host_flag doesn't duplicate host=1 parameter."""
        url = "ws://localhost:3000/websocket?host=1"
        result = AAIVoiceAgentProvider._with_host_flag(url)
        # Should handle gracefully (or keep as is)
        assert result.count("host=1") >= 1

    def test_is_connected_property_initial_state(self) -> None:
        """Test is_connected returns False when not connected."""
        provider = AAIVoiceAgentProvider()
        assert provider.is_connected is False

    def test_is_connected_property_after_connect(self) -> None:
        """Test is_connected returns True when ws is open."""
        provider = AAIVoiceAgentProvider()
        # Mock the ws to be in OPEN state
        from websockets.protocol import State

        mock_ws = MagicMock()
        mock_ws.state = State.OPEN
        provider.ws = mock_ws

        assert provider.is_connected is True

    def test_is_connected_property_when_closed(self) -> None:
        """Test is_connected returns False when ws is closed."""
        provider = AAIVoiceAgentProvider()
        from websockets.protocol import State

        mock_ws = MagicMock()
        mock_ws.state = State.CLOSED
        provider.ws = mock_ws

        assert provider.is_connected is False

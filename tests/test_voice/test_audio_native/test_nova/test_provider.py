"""
Integration tests for NovaSonicProvider.

These tests require AWS credentials and make real API calls to Amazon Bedrock.
Set NOVA_TEST_ENABLED=1 along with AWS credentials to run these tests.

Run with: NOVA_TEST_ENABLED=1 pytest tests/test_voice/test_audio_native/test_nova/test_provider.py -v -s

NOTE: For full end-to-end testing with audio responses, use the standalone test:
    python src/tau2/voice/audio_native/nova/test_provider_standalone.py
"""

import os

import pytest

from tau2.voice.audio_native.nova.audio_utils import telephony_to_nova_input
from tau2.voice.audio_native.nova.provider import NovaSonicProvider, NovaVADConfig

# Skip all tests unless explicitly enabled - prevents hanging when AWS credentials
# exist but don't have Nova Sonic permissions
pytestmark = pytest.mark.skipif(
    not os.environ.get("NOVA_TEST_ENABLED"),
    reason="NOVA_TEST_ENABLED not set (requires AWS credentials with Nova Sonic access)",
)


@pytest.fixture
def provider():
    """Create a NovaSonicProvider instance."""
    return NovaSonicProvider()


class TestNovaSonicProviderBasic:
    """Basic connectivity and session tests."""

    def test_connect_and_disconnect(self, provider):
        """Test basic connection and disconnection."""
        import asyncio

        async def _test():
            assert not provider.is_connected

            await provider.connect()
            assert provider.is_connected
            assert provider._session_id is not None

            await provider.disconnect()
            assert not provider.is_connected

        asyncio.run(_test())

    def test_configure_session(self, provider):
        """Test configuring the session with system prompt."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=NovaVADConfig(),
                )
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_audio(self, provider):
        """Test sending audio data."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=NovaVADConfig(),
                )

                # Start audio stream
                content_id = await provider.start_audio_stream()
                assert content_id is not None

                # Create test audio (silence)
                silence = bytes([0x7F] * 800)  # 100ms at 8kHz μ-law
                pcm_audio, _ = telephony_to_nova_input(silence)

                # Send audio
                await provider.send_audio(pcm_audio, content_id)

                # End audio block
                await provider.end_audio_content(content_id)

                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())


class TestNovaSonicProviderAudioFormats:
    """Tests for audio format handling."""

    def test_telephony_to_nova_conversion(self):
        """Test audio format conversion from telephony to Nova."""
        # 200ms of μ-law silence at 8kHz = 1600 bytes
        ulaw_silence = bytes([0x7F] * 1600)

        # Convert to Nova format (16kHz PCM16)
        pcm_audio, _ = telephony_to_nova_input(ulaw_silence)

        # PCM16 16kHz should be ~4x larger:
        # - 2x sample rate (8kHz -> 16kHz)
        # - 2x bytes per sample (1 byte μ-law -> 2 bytes PCM16)
        # Expected: 1600 * 4 = 6400 bytes (with minor resampling variance)
        assert 6300 <= len(pcm_audio) <= 6500, (
            f"Expected ~6400 bytes, got {len(pcm_audio)}"
        )


class TestNovaSonicProviderToolCalls:
    """Tests for tool/function calling."""

    def test_configure_with_tools(self, provider):
        """Test configuring session with tools."""
        import asyncio

        from tau2.environment.tool import Tool

        # Create a mock tool
        class MockTool(Tool):
            def __init__(self):
                pass

            @property
            def openai_schema(self):
                return {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city name",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                }

            def __call__(self, **kwargs):
                return "Sunny, 72°F"

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a weather assistant.",
                    tools=[MockTool()],
                    vad_config=NovaVADConfig(),
                )
                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

    def test_send_tool_result(self, provider):
        """Test sending a tool result."""
        import asyncio

        async def _test():
            await provider.connect()

            try:
                await provider.configure_session(
                    system_prompt="You are a helpful assistant.",
                    tools=[],
                    vad_config=NovaVADConfig(),
                )

                # This should not raise (even without a preceding tool call)
                await provider.send_tool_result(
                    tool_use_id="test-tool-123",
                    result="The result is 42.",
                )

                assert provider.is_connected

            finally:
                await provider.disconnect()

        asyncio.run(_test())

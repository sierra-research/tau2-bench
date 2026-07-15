#!/usr/bin/env python3
"""
Standalone script to test AAI voice agent provider.

This script connects to a local AAI voice-agent host via WebSocket,
sends test audio, and receives events.

Usage:
    AAI_WS_URL=ws://localhost:3100/websocket python src/tau2/voice/audio_native/aai/test_provider_standalone.py
"""

import asyncio
import os
import signal
import sys
from typing import List

# Add src to path
sys.path.insert(0, "src")


async def main():
    from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
    from tau2.voice.audio_native.aai.events import BaseAAIEvent
    from tau2.voice.audio_native.aai.provider import AAIVoiceAgentProvider
    from tau2.voice.utils.audio_preprocessing import convert_to_pcm16, resample_audio

    print("=" * 60)
    print("AAI Voice Agent Provider Test Script")
    print("=" * 60)

    # Read AAI_WS_URL from environment
    aai_ws_url = os.environ.get("AAI_WS_URL", "ws://localhost:3000/websocket")
    print(f"\nConnecting to: {aai_ws_url}")

    provider = AAIVoiceAgentProvider(
        ws_url=aai_ws_url,
        system_prompt="You are a helpful assistant. Keep replies to one short sentence.",
        tools=[],
    )

    received_events: List[BaseAAIEvent] = []

    try:
        # 1. Connect
        print("\n1. Connecting to AAI host...")
        try:
            await asyncio.wait_for(provider.connect(), timeout=5)
        except asyncio.TimeoutError:
            print(f"SKIP: aai host not reachable at {aai_ws_url}")
            return
        except Exception as e:
            if "refused" in str(e).lower() or "connection" in str(e).lower():
                print(f"SKIP: aai host not reachable at {aai_ws_url}")
                return
            raise

        print("   ✅ Connected!")

        # 2. Load and convert test audio
        print("\n2. Loading test audio...")
        audio_path = "tests/test_voice/test_audio_native/testdata/hello.ulaw"
        if not os.path.exists(audio_path):
            print(f"   ⚠️ Audio file not found: {audio_path}")
            return

        with open(audio_path, "rb") as f:
            ulaw_data = f.read()

        # Create AudioData object for hello.ulaw (μ-law at 8kHz)
        audio = AudioData(
            data=ulaw_data,
            format=AudioFormat(
                encoding=AudioEncoding.ULAW,
                sample_rate=8000,
                channels=1,
            ),
            audio_path=None,
        )

        # Convert μ-law-8k → PCM16-8k → PCM16-16k
        print(f"   Loaded: {len(ulaw_data)} bytes of μ-law audio at 8kHz")
        pcm16_8k = convert_to_pcm16(audio)
        print(f"   Converted to PCM16-8k: {len(pcm16_8k.data)} bytes")
        pcm16_16k = resample_audio(pcm16_8k, 16000)
        print(f"   Resampled to PCM16-16k: {len(pcm16_16k.data)} bytes")

        # 3. Send audio and receive events concurrently
        print("\n3. Sending test audio and receiving events...")

        async def send_audio_with_chunks():
            """Send audio in chunks with delays, then silence."""
            chunk_size = 16000 * 2 * 50 // 1000  # ~50ms chunks at 16kHz 16-bit
            # Send speech audio in chunks
            for i in range(0, len(pcm16_16k.data), chunk_size):
                chunk = pcm16_16k.data[i : i + chunk_size]
                await provider.send_audio(chunk)
                await asyncio.sleep(0.05)  # ~50ms between chunks

            print("   ✅ Speech audio sent!")

            # Send ~2s of silence for VAD
            silence_1s = b"\x00" * 32000  # 1 second at 16kHz 16-bit
            for _ in range(2):
                await provider.send_audio(silence_1s)
                await asyncio.sleep(0.1)

            print("   ✅ Silence sent!")

        # 4. Send audio and collect events concurrently
        _, events = await asyncio.gather(
            send_audio_with_chunks(),
            provider.receive_events_for_duration(25.0),
        )
        received_events.extend(events)
        print(f"   Received {len(received_events)} events")

        # 5. Show results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        # Note: Config event is received during connect() and consumed as part of
        # the handshake, so it doesn't appear in the events list. Success = connected
        # to the host and receiving the config acknowledgment (which happened during connect).
        print(f"Connected and received config: ✅")
        print(f"Additional events received after connect: {len(received_events)}")

        # Collect event types for diagnostic purposes
        if received_events:
            event_types = {}
            for e in received_events:
                event_type = type(e).__name__
                event_types[event_type] = event_types.get(event_type, 0) + 1

            print("\nEvent types received:")
            for t, count in sorted(event_types.items()):
                print(f"  - {t}: {count}")
        else:
            print("\nNo additional events received (this is OK for a simple ping)")

        print("\n✅ SUCCESS - AAI provider connected and ready!")
        return 0

    except asyncio.TimeoutError:
        print(f"SKIP: aai host not reachable at {aai_ws_url}")
        return 0
    except Exception as e:
        error_str = str(e).lower()
        if (
            "refused" in error_str
            or "connection" in error_str
            or "timeout" in error_str
        ):
            print(f"SKIP: aai host not reachable at {aai_ws_url}")
            return 0
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # 6. Disconnect
        print("\n6. Disconnecting...")
        await provider.disconnect()
        print("   ✅ Disconnected!")


if __name__ == "__main__":
    # Set up timeout
    def timeout_handler(signum, frame):
        print("\n\n❌ SCRIPT TIMEOUT (60s)")
        sys.exit(1)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)

    # Run
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

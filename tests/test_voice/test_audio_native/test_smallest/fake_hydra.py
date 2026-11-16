"""Test doubles for the smallest adapter: a fake Hydra server and event builders.

``FakeHydra`` serves the ``/v1/hydra`` handshake on 127.0.0.1 from its own
thread (no network): it sends ``session.created`` on connect, answers
``session.configure``, records every client message, and pushes whatever
server events a test hands it. ``ScriptedProvider`` replaces the provider for
tick tests that need exact, timing-independent event batches.
"""

import asyncio
import base64
import json
import threading
import time
from collections import deque
from typing import Callable, Optional

import websockets

from tau2.config import TELEPHONY_ULAW_SILENCE
from tau2.voice.audio_native.smallest.events import parse_hydra_event

TICK_MS = 200
TICK_BYTES = 1600  # 200 ms of 8 kHz u-law
SILENCE_TICK = TELEPHONY_ULAW_SILENCE * TICK_BYTES
SESSION_ID = "sess_fake"


# -- Hydra server events -----------------------------------------------------


def created(rid: str) -> dict:
    return {"type": "response.created", "response": {"id": rid}}


def caption(rid: str, text: str) -> dict:
    return {
        "type": "response.output_audio_transcript.delta",
        "response_id": rid,
        "item_id": f"item_{rid}",
        "delta": text,
    }


def pcm(ms: int, rate: int, amplitude: int = 1000) -> bytes:
    return int(amplitude).to_bytes(2, "little", signed=True) * (rate * ms // 1000)


def audio(rid: str, ms: int, rate: int = 8000) -> dict:
    return {
        "type": "response.output_audio.delta",
        "response_id": rid,
        "item_id": f"item_{rid}",
        "delta": base64.b64encode(pcm(ms, rate)).decode("ascii"),
    }


def audio_done(rid: str) -> dict:
    return {
        "type": "response.output_audio.done",
        "response_id": rid,
        "item_id": f"item_{rid}",
    }


def done(rid: str, status: str = "completed", usage: Optional[dict] = None) -> dict:
    response = {"id": rid, "status": status, "output": []}
    if usage:
        response["usage"] = usage
    return {"type": "response.done", "response": response}


def tool_call(rid: str, call_id: str, name: str, arguments: str) -> dict:
    return {
        "type": "response.function_call_arguments.done",
        "response_id": rid,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def speech_started() -> dict:
    return {
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": 0,
        "item_id": "item_user",
    }


def barge_in() -> dict:
    return {"type": "response.barge_in", "reason": "interrupted"}


# -- Scripted provider (tick tests) -------------------------------------------


class ScriptedProvider:
    """Returns one batch of parsed Hydra events per tick."""

    def __init__(self, batches=(), sample_rate: int = 8000):
        self.batches = deque(
            [parse_hydra_event(e, sample_rate) for e in batch] for batch in batches
        )
        self.sent_audio: list[bytes] = []
        self.tool_results: list[tuple] = []
        self.session_id = "sess_scripted"

    async def send_audio(self, audio_pcm16: bytes) -> None:
        self.sent_audio.append(audio_pcm16)

    async def send_tool_result(self, call_id, output, request_response=True):
        self.tool_results.append((call_id, output, request_response))

    async def receive_events_for_duration(self, duration_s: float):
        return [e for e in self.batches.popleft() if e] if self.batches else []


def run_ticks(adapter, count: int, user_audio: bytes = SILENCE_TICK) -> list:
    """Run ``count`` ticks through the adapter's tick template."""

    async def go():
        return [
            await adapter._async_run_tick(user_audio, n) for n in range(1, count + 1)
        ]

    return asyncio.run(go())


# -- Fake Hydra server (wire tests) -------------------------------------------


class FakeHydra:
    """A minimal Hydra ``/v1/hydra`` server on the loopback interface."""

    def __init__(self, output_rate: int = 24000, configure_reply: str = "configured"):
        # configure_reply: "configured", "error", "no_rate" or "silent".
        self.output_rate = output_rate
        self.configure_reply = configure_reply
        self.received: list[dict] = []
        self.auth_headers: list[Optional[str]] = []
        self.close_codes: list[Optional[int]] = []
        self._lock = threading.Lock()
        self._ws = None
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def start(self) -> "FakeHydra":
        self._thread.start()
        self._server = self._call(self._serve())
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/v1/hydra"
        return self

    def stop(self) -> None:
        self._server.close()
        self._call(self._server.wait_closed())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)

    def send(self, *events: dict) -> None:
        """Push server events to the connected client, in order."""
        self._call(self._send_all(events))

    def close_client(self, code: int = 1011, reason: str = "worker crashed") -> None:
        self._call(self._ws.close(code, reason))

    def messages(self, kind: str) -> list[dict]:
        with self._lock:
            return [m for m in self.received if m.get("type") == kind]

    def wait_for(self, condition: Callable[[], bool], timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise AssertionError("fake Hydra: condition not met in time")
            time.sleep(0.01)

    def _call(self, coro, timeout: float = 10.0):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    async def _serve(self):
        return await websockets.serve(self._handler, "127.0.0.1", 0)

    async def _send_all(self, events) -> None:
        for event in events:
            await self._ws.send(json.dumps(event))

    async def _handler(self, ws) -> None:
        self._ws = ws
        self.auth_headers.append(ws.request.headers.get("Authorization"))
        await ws.send(json.dumps({"type": "session.created", "session_id": SESSION_ID}))
        try:
            async for raw in ws:
                message = json.loads(raw)
                with self._lock:
                    self.received.append(message)
                if message.get("type") == "session.configure":
                    await self._answer_configure(ws, message["session"])
        except websockets.ConnectionClosed:
            pass
        finally:
            with self._lock:
                self.close_codes.append(ws.close_code)

    async def _answer_configure(self, ws, session: dict) -> None:
        if self.configure_reply == "error":
            error = {"code": "invalid_request_error", "message": "unknown voice"}
            await ws.send(json.dumps({"type": "error", "error": error}))
        elif self.configure_reply in ("configured", "no_rate"):
            session = dict(session, output_audio_format="pcm16")
            if self.configure_reply == "configured":
                session["output_audio_sample_rate"] = self.output_rate
            await ws.send(
                json.dumps({"type": "session.configured", "session": session})
            )

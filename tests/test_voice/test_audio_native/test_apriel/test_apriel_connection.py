"""Key-free tests for Apriel's direct and protected-proxy transport."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import pytest

from tau2.voice.audio_native.apriel.connection import (
    model_request,
    normalize_realtime_url,
    resolve_credentials,
    resolve_model,
    tls_server_name,
    websocket_request,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "APRIEL_REALTIME_API_KEY",
        "APRIEL_CLOUD_AUTH_TOKEN",
        "APRIEL_REALTIME_TLS_SERVER_NAME",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://example.com", "wss://example.com/v1/realtime"),
        ("https://example.com/v1/", "wss://example.com/v1/realtime"),
        ("wss://example.com/v1/realtime", "wss://example.com/v1/realtime"),
        ("http://localhost:8080", "ws://localhost:8080/v1/realtime"),
        ("ws://[::1]:8080/v1", "ws://[::1]:8080/v1/realtime"),
    ],
)
def test_url_variants(url, expected):
    assert normalize_realtime_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "https://user:secret@example.com",
        "wss://example.com/wrong",
        "https://example.com:bad",
        "https://example.com:65536",
        "wss://example.com/#fragment",
        "https://example.com?api_key=secret",
        "wss://example.com?TOKEN=secret",
    ],
)
def test_reject_unsafe_or_invalid_url(url):
    with pytest.raises(ValueError):
        normalize_realtime_url(url)


def test_direct_headers_and_query():
    url, options = websocket_request(
        "http://example.com/v1?model=old&other=1", "a/b ?", "aur_sk_test"
    )
    assert parse_qs(urlsplit(url).query) == {"model": ["a/b ?"], "other": ["1"]}
    assert options == {"additional_headers": {"Authorization": "Bearer aur_sk_test"}}
    req = model_request(url, "aur_sk_test")
    assert req.full_url == "http://example.com/v1/models"
    assert dict(req.header_items()) == {"Authorization": "Bearer aur_sk_test"}


def test_proxy_auth_is_split_and_not_in_url():
    url, options = websocket_request(
        "https://example.com", "model", "aur_sk_app", "cloud-token"
    )
    assert options["additional_headers"] == {"Authorization": "Bearer cloud-token"}
    assert options["subprotocols"] == ["realtime", "openai-insecure-api-key.aur_sk_app"]
    assert "cloud-token" not in url and "aur_sk_app" not in url
    req = model_request(url, "aur_sk_app", "cloud-token")
    assert req.get_header("Authorization") == "Bearer cloud-token"
    assert req.get_header("X-apriel-authorization") == "Bearer aur_sk_app"


def test_realtime_env_and_explicit_override(monkeypatch):
    monkeypatch.setenv("APRIEL_REALTIME_API_KEY", "env-key")
    assert resolve_credentials() == ("env-key", "")
    monkeypatch.setenv("APRIEL_REALTIME_API_KEY", "canonical")
    monkeypatch.setenv("APRIEL_CLOUD_AUTH_TOKEN", "cloud")
    assert resolve_credentials() == ("canonical", "cloud")
    assert resolve_credentials("explicit", "") == ("explicit", "")


@pytest.mark.parametrize(
    "key,cloud", [("bad\nkey", ""), ("a,b", ""), ("okay", "bad\rtoken")]
)
def test_header_injection_rejected(key, cloud):
    with pytest.raises(ValueError):
        websocket_request("wss://example.com", "model", key, cloud)


def test_tls_keeps_verified_hostname_semantics(monkeypatch):
    assert (
        tls_server_name("wss://snow-core_llm.job.console.elementai.com")
        == "snow-core-llm.job.console.elementai.com"
    )
    assert tls_server_name("wss://example.com") is None
    monkeypatch.setenv("APRIEL_REALTIME_TLS_SERVER_NAME", "verify.example.com")
    assert (
        websocket_request("wss://example.com", "m", "key")[1]["server_hostname"]
        == "verify.example.com"
    )
    assert tls_server_name("ws://localhost") is None


def test_model_discovery_auth_and_redirect_refusal():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(dict(self.headers))
            if self.path.startswith("/redirect"):
                self.send_response(302)
                self.send_header("Location", "/v1/models")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "attached-model"}]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert resolve_model(base, "app", "cloud") == "attached-model"
        assert seen[0]["Authorization"] == "Bearer cloud"
        assert seen[0]["X-Apriel-Authorization"] == "Bearer app"
        with pytest.raises(ValueError, match="redirect"):
            resolve_model(base + "/redirect/v1", "app", "cloud")
        assert len(seen) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_provider_uses_transport_and_verifies_media_before_audio(monkeypatch):
    from websockets.protocol import State

    from tau2.voice.audio_native.apriel import provider as module
    from tau2.voice.audio_native.apriel.provider import AprielRealtimeProvider

    async def run():
        wire = AsyncMock()
        wire.state = State.OPEN
        wire.recv.side_effect = [
            json.dumps({"type": "session.created", "session": {"id": "test"}}),
            json.dumps(
                {
                    "type": "session.updated",
                    "session": {
                        "apriel": {"playback_clock": "media", "turn_clock": "media"}
                    },
                }
            ),
        ]
        connect = AsyncMock(return_value=wire)
        monkeypatch.setattr(module.websockets, "connect", connect)
        provider = AprielRealtimeProvider(
            api_key="app",
            cloud_auth_token="cloud",
            base_url="https://example.com",
            model="m",
        )
        await provider.connect()
        assert connect.call_args.kwargs["subprotocols"] == [
            "realtime",
            "openai-insecure-api-key.app",
        ]
        with pytest.raises(RuntimeError):
            await provider.send_audio(b"bytes")
        await provider.configure_session("unchanged scenario", [], None, "audio")
        message = json.loads(wire.send.call_args_list[0].args[0])
        assert message["session"]["apriel"]["turn_clock"] == "media"
        assert message["session"]["instructions"] == "unchanged scenario"
        await provider.send_audio(b"bytes")
        assert (
            json.loads(wire.send.call_args.args[0])["type"]
            == "input_audio_buffer.append"
        )
        await provider.disconnect()

    asyncio.run(run())


def test_https_model_discovery_verifies_tls_and_supports_sni(monkeypatch, tmp_path):
    import ssl
    import subprocess
    import urllib.error

    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
        ],
        check=True,
        capture_output=True,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"tls-model"}]}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("SSL_CERT_FILE", str(cert))
        url = f"https://127.0.0.1:{server.server_port}"
        with pytest.raises(urllib.error.URLError):
            resolve_model(url, "app", "cloud")
        monkeypatch.setenv("APRIEL_REALTIME_TLS_SERVER_NAME", "localhost")
        assert resolve_model(url, "app", "cloud") == "tls-model"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_main_registration(monkeypatch):
    from tau2.config import AUDIO_NATIVE_PROVIDER_TYPES, DEFAULT_AUDIO_NATIVE_MODELS
    from tau2.data_model.run_config import AudioNativeConfig
    from tau2.voice.audio_native.adapter import create_adapter

    assert AUDIO_NATIVE_PROVIDER_TYPES["apriel"] == "audio_native"
    assert "apriel" in DEFAULT_AUDIO_NATIVE_MODELS
    assert AudioNativeConfig(provider="apriel").provider == "apriel"
    monkeypatch.setenv("APRIEL_REALTIME_API_KEY", "app")
    monkeypatch.setenv("APRIEL_REALTIME_BASE_URL", "https://example.com")
    adapter, model = create_adapter("apriel", tick_duration_ms=200)
    assert adapter.__class__.__name__ == "DiscreteTimeAprielAdapter"
    assert model == DEFAULT_AUDIO_NATIVE_MODELS["apriel"]

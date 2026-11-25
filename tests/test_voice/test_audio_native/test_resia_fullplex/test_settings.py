"""Resia Fullplex requires an API key and a backend model, and uses a secure endpoint."""

import pytest

from tau2.config import DEFAULT_RESIA_FULLPLEX_URL
from tau2.voice.audio_native.resia_fullplex.provider import (
    resolve_api_key,
    resolve_backend_model,
    resolve_endpoint,
)


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("RESIA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        resolve_api_key()


def test_backend_model_is_required(monkeypatch):
    monkeypatch.delenv("RESIA_FULLPLEX_BACKEND_MODEL", raising=False)
    with pytest.raises(ValueError):
        resolve_backend_model()


def test_default_endpoint(monkeypatch):
    monkeypatch.delenv("RESIA_FULLPLEX_URL", raising=False)
    assert resolve_endpoint() == DEFAULT_RESIA_FULLPLEX_URL


def test_insecure_remote_endpoint_is_refused():
    with pytest.raises(ValueError):
        resolve_endpoint("ws://example.com/v1/live/sessions")

"""Shared fixtures for audio native tests.

Pre-registers LiveKit plugins on the main thread (required before
any LiveKit adapter tests can run).
"""


def pytest_configure(config):
    """Called on the main thread before test collection."""
    try:
        from tau2.voice.audio_native.livekit import preregister_livekit_plugins

        preregister_livekit_plugins()
    except ImportError:
        pass

"""Core imports and the CLI must work without the optional ``voice`` extra."""

import subprocess
import sys
import textwrap

# Modules only the ``voice`` extra installs (not transitive core dependencies).
VOICE_EXTRA_MODULES = (
    "websockets",
    "elevenlabs",
    "deepgram",
    "pyaudio",
    "jiwer",
    "scipy",
    "pydub",
    "livekit",
    "aws_sdk_bedrock_runtime",
    "google.genai",
    "google.cloud.aiplatform",
    "vertexai",
)

_BLOCKER = textwrap.dedent(
    f"""
    import importlib.abc
    import sys

    BLOCKED = {VOICE_EXTRA_MODULES!r}

    class _VoiceExtraBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if any(name == b or name.startswith(b + ".") for b in BLOCKED):
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
            return None

    sys.meta_path.insert(0, _VoiceExtraBlocker())
    for loaded in list(sys.modules):
        if any(loaded == b or loaded.startswith(b + ".") for b in BLOCKED):
            del sys.modules[loaded]
    """
)


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_import_and_text_run_config_without_voice_extras():
    result = _run(
        """
        import tau2
        import tau2.cli
        from tau2.data_model.simulation import TextRunConfig

        config = TextRunConfig(domain="airline")
        config.validate()
        print("language=" + config.language)
        """
    )
    assert result.returncode == 0, result.stderr
    assert "language=en" in result.stdout


def test_cli_help_without_voice_extras():
    result = _run(
        """
        import sys
        from tau2.cli import main

        sys.argv = ["tau2", "--help"]
        try:
            main()
        except SystemExit as exit_:
            sys.exit(exit_.code)
        """
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()

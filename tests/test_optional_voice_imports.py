"""Core imports must work without the optional voice implementations."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_core_imports_without_voice_dependencies():
    # Use a fresh interpreter: conftest imports tau2 before tests are collected.
    # Block extras even when the developer has them installed.
    script = """
        import importlib.abc
        import sys

        class NoVoiceExtras(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'websockets', 'scipy', 'soundfile'}:
                    raise ModuleNotFoundError(
                        f'Optional voice dependency unavailable: {fullname}',
                        name=fullname,
                    )

        sys.meta_path.insert(0, NoVoiceExtras())
        import tau2
        import tau2.runner
        from tau2.cli import main
        from tau2.data_model.simulation import AudioNativeConfig
        from tau2.voice.audio_native.openai.live_config import LiveConfig

        config = AudioNativeConfig(
            provider='openai_live', live_config=LiveConfig(backend_model='test-model')
        )
        restored = AudioNativeConfig.model_validate_json(config.model_dump_json())
        assert isinstance(restored.live_config, LiveConfig)
        assert restored.live_config.backend_model == 'test-model'
        AudioNativeConfig.model_json_schema()
        assert 'tau2.voice.audio_native.openai.provider' not in sys.modules
        assert 'tau2.voice.audio_native.openai.discrete_time_adapter' not in sys.modules
        sys.argv = ['tau2', 'check-data']
        main()
    """
    repo = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo / "src"),
        "TAU2_DATA_DIR": str(repo / "data"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_openai_package_exports_with_voice_dependencies():
    pytest.importorskip("websockets")
    pytest.importorskip("scipy")
    pytest.importorskip("soundfile")

    from tau2.voice.audio_native import openai
    from tau2.voice.audio_native.openai.discrete_time_adapter import (
        DiscreteTimeOpenAIAdapter,
    )
    from tau2.voice.audio_native.openai.provider import (
        OpenAIRealtimeProvider,
        OpenAIVADConfig,
        OpenAIVADMode,
    )

    assert openai.DiscreteTimeOpenAIAdapter is DiscreteTimeOpenAIAdapter
    assert openai.OpenAIRealtimeProvider is OpenAIRealtimeProvider
    assert openai.OpenAIVADConfig is OpenAIVADConfig
    assert openai.OpenAIVADMode is OpenAIVADMode
    with pytest.raises(AttributeError, match="no attribute 'not_a_voice_export'"):
        getattr(openai, "not_a_voice_export")

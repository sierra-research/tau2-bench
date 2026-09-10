"""Exercise customer TTS and telephony conversion without running an agent/LLM.

Run from the repository root:
    uv run examples/voice/cartesia_customer_smoke.py
Requires CARTESIA_API_KEY. Consumes one short TTS request.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from tau2.data_model.message import UserMessage
from tau2.data_model.simulation import AudioNativeConfig
from tau2.data_model.voice import SynthesisConfig, VoiceSettings
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.runner.build import build_voice_user


def main():
    """Save one real customer utterance plus its effective voice settings."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/voice/cartesia-customer.json")
    parser.add_argument("--output", default="data/simulations/cartesia-customer-smoke")
    args = parser.parse_args()
    output = Path(args.output)
    synthesis = SynthesisConfig.model_validate_json(Path(args.config).read_text())
    user = build_voice_user(
        registry.get_env_constructor("retail")(),
        get_tasks("retail", task_ids=["0"])[0],
        AudioNativeConfig(),
        domain="retail",
        speech_complexity="control",
        voice_settings=VoiceSettings(
            transcription_config=None,
            synthesis_config=synthesis,
            output_dir=output,
        ),
    )
    state = user.get_init_state()
    message = user.synthesize_voice(
        UserMessage(
            role="user",
            content="Hello. I would like to return my order. [pause] Can you help me?",
        ),
        state,
    )
    audio = message.get_audio_bytes()
    assert audio and message.is_audio
    assert message.audio_format.sample_rate == 8000
    result = {
        "bytes": len(audio),
        "audio_format": message.audio_format.model_dump(mode="json"),
        "audio_path": message.audio_path,
        "voice_settings": user.voice_settings.model_dump(mode="json"),
        "scope": "Live customer TTS and effects pipeline only; no agent or LLM call.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "voice_settings"}, indent=2)
    )


if __name__ == "__main__":
    main()

import random
import uuid
from copy import deepcopy
from typing import Generic, TypeVar

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base.streaming_utils import format_transcript_comparison
from tau2.data_model.audio import AudioData, audio_bytes_to_string
from tau2.data_model.message import Message
from tau2.data_model.voice import VoiceSettings
from tau2.data_model.voice_personas import get_elevenlabs_voice_id
from tau2.voice.synthesis.audio_effects import (
    BackgroundNoiseGenerator,
    BatchAudioEffectsMixin,
    generate_turn_effects,
)
from tau2.voice.synthesis.synthesize import synthesize_voice
from tau2.voice.transcription.transcribe import transcribe_audio
from tau2.voice.utils.audio_io import save_wav_file
from tau2.voice.utils.text_effects import insert_speech_text

# Generic type variables for streaming mixins
InputMessageType = TypeVar("InputMessageType", bound=Message)
OutputMessageType = TypeVar("OutputMessageType", bound=Message)


class VoiceState(BaseModel):
    """State for voice capabilities (holds noise generator)."""

    model_config = {"arbitrary_types_allowed": True}

    noise_generator: BackgroundNoiseGenerator


VoiceStateType = TypeVar("VoiceStateType", bound=VoiceState)


class VoiceMixin(
    BatchAudioEffectsMixin, Generic[InputMessageType, OutputMessageType, VoiceStateType]
):
    """Mixin to add voice synthesis/transcription capabilities."""

    def __init__(
        self, *args, voice_settings: VoiceSettings = VoiceSettings(), **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.voice_settings = voice_settings

    def transcribe_voice(self, message: InputMessageType) -> InputMessageType:
        """Transcribe audio message to text."""
        if not message.is_audio:
            raise ValueError("Message must be audio.")
        if self.voice_settings.transcription_config is None:
            raise ValueError("Transcription config is not set.")

        message = deepcopy(message)
        audio_data = AudioData(
            data=message.get_audio_bytes(),
            format=message.audio_format,
            audio_path=message.audio_path,
        )
        transcription_result = transcribe_audio(
            audio_data=audio_data,
            config=self.voice_settings.transcription_config,
        )
        if transcription_result.error:
            raise ValueError(f"Transcription failed: {transcription_result.error}")

        message = deepcopy(message)
        message.content = transcription_result.transcript
        gold_transcript = message.audio_script_gold
        if gold_transcript:
            comparison = format_transcript_comparison(
                transcription_result.transcript,
                gold_transcript,
                show_chunks=False,
            )
            logger.info(f"Transcription comparison:\n{comparison}")
        else:
            logger.info(f"Transcribed text: {transcription_result.transcript}")

        if self.voice_settings.output_dir:
            turn_uuid = str(uuid.uuid4())
            audio_dir = self.voice_settings.output_dir / f"turn_{turn_uuid}"
            audio_dir.mkdir(parents=True, exist_ok=True)
            transcribed_text_path = audio_dir / "transcribed_text.txt"
            with open(transcribed_text_path, "w", encoding="utf-8") as f:
                f.write(transcription_result.transcript)
        return message

    def synthesize_voice(
        self,
        message: OutputMessageType,
        state: VoiceStateType,
        effects_turn_idx: int = 0,
        add_background_noise: bool = True,
        add_burst_noise: bool = True,
        add_telephony_format: bool = True,
        add_channel_effects: bool = True,
        apply_speech_effects: bool = True,
    ) -> OutputMessageType:
        """Synthesize text message to audio with per-turn effects."""
        if message.is_audio:
            raise ValueError("Message must be text.")
        if message.is_tool_call():
            raise ValueError("Message must not be a tool call.")
        if not message.has_text_content():
            raise ValueError("Message must have text content.")
        if message.has_audio_content():
            raise ValueError("Message must not have audio content.")

        speech_env = self.voice_settings.speech_environment
        synthesis_config = self.voice_settings.synthesis_config
        provider_config = deepcopy(synthesis_config.provider_config)
        provider_config.voice_id = get_elevenlabs_voice_id(speech_env.persona_name)

        # Generate per-turn effects (complexity overrides already merged into synthesis_config)
        speech_effects, source_effects, channel_effects = generate_turn_effects(
            seed=speech_env.voice_seed,
            turn_idx=effects_turn_idx,
            synthesis_config=synthesis_config,
        )

        text_to_synthesize = message.content
        speech_config = synthesis_config.speech_effects_config
        text_effects_rng = random.Random(speech_env.voice_seed + effects_turn_idx)

        if speech_effects.speech_insert:
            text_to_synthesize = insert_speech_text(
                text_to_synthesize,
                speech_effects.speech_insert,
                rng=text_effects_rng,
                min_words=speech_config.min_words_for_vocal_tics,
                in_turn=True,
            )

        audio_data = synthesize_voice(
            text=text_to_synthesize,
            provider=synthesis_config.provider,
            provider_config=provider_config,
        )

        audio_data = self.apply_batch_effects(
            audio=audio_data,
            speech_effects=speech_effects,
            source_effects=source_effects,
            channel_effects=channel_effects,
            speech_config=speech_config,
            channel_config=synthesis_config.channel_effects_config,
            noise_generator=(
                state.noise_generator
                if (add_background_noise and speech_env.background_noise_file)
                else None
            ),
            apply_speech_effects=apply_speech_effects,
            apply_source_effects=apply_speech_effects and add_burst_noise,
            apply_channel_effects=apply_speech_effects and add_channel_effects,
            apply_telephony=add_telephony_format and speech_env.telephony_enabled,
        )

        message.audio_content = audio_bytes_to_string(audio_data.data)
        message.audio_format = audio_data.format
        message.is_audio = True
        message.audio_script_gold = text_to_synthesize
        message.speech_effects = speech_effects
        message.source_effects = source_effects
        message.channel_effects = channel_effects
        logger.debug(
            f"Voice synthesized: {audio_data.audio_path} with format {audio_data.format}. Duration: {audio_data.duration} seconds"
        )

        audio_path = None
        if self.voice_settings.output_dir:
            turn_uuid = str(uuid.uuid4())
            audio_dir = self.voice_settings.output_dir / f"turn_{turn_uuid}"
            audio_dir.mkdir(parents=True, exist_ok=True)

            text_to_synthesize_path = audio_dir / "text_to_synthesize.txt"
            with open(text_to_synthesize_path, "w", encoding="utf-8") as f:
                f.write(text_to_synthesize)

            audio_path = audio_dir / "speech.wav"
            audio_data.audio_path = audio_path
            save_wav_file(audio_data, str(audio_path))

        message.audio_path = str(audio_path) if audio_path else None
        return message

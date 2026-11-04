"""Shared ElevenLabs SDK operations; callers own review, retries and persistence."""

from typing import TYPE_CHECKING

from tau2.data_model.voice import (
    VoiceCreateRequest,
    VoiceDesignPreview,
    VoiceDesignRequest,
)

if TYPE_CHECKING:
    from elevenlabs import ElevenLabs


def design_voice(
    client: "ElevenLabs", request: VoiceDesignRequest
) -> list[VoiceDesignPreview]:
    """Request previews once without generating audition text or saving a voice."""
    response = client.text_to_voice.design(
        voice_description=request.description,
        text=request.text,
        model_id=request.model,
        loudness=request.loudness,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        auto_generate_text=False,
    )
    return [
        VoiceDesignPreview(
            generated_voice_id=preview.generated_voice_id,
            audio_base_64=preview.audio_base_64,
            media_type=preview.media_type,
        )
        for preview in response.previews
    ]


def create_voice(client: "ElevenLabs", request: VoiceCreateRequest) -> str:
    """Save exactly the requested preview once; never retry an uncertain creation."""
    return client.text_to_voice.create(
        voice_name=request.name,
        voice_description=request.description,
        generated_voice_id=request.generated_voice_id,
    ).voice_id

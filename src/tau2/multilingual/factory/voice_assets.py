"""Resumable voice provisioning with provider-independent review artifacts.

No provider is constructed at import time. Runtime never imports this module.
"""

import base64
import errno
import hashlib
import io
import os
import tempfile
import wave
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from tau2.data_model.voice import VoiceCreateRequest, VoiceDesignRequest
from tau2.multilingual.factory.schema import (
    DraftArtifact,
    FormModel,
)
from tau2.utils.artifacts import (
    ReviewApproval,
    content_digest,
    write_artifact,
)
from tau2.voice.elevenlabs_operations import create_voice, design_voice


class VoiceDesignConfig(FormModel):
    """Explicit recorded provider settings, separate from user-model settings."""

    mode: Annotated[
        Literal["design"], Field(description="Design new provider voices")
    ] = "design"
    provider: Annotated[
        Literal["elevenlabs"],
        Field(description="Voice provider used for asset provisioning."),
    ] = "elevenlabs"
    model: Annotated[
        str,
        Field(
            min_length=1, description="Explicit provider voice-design model identifier."
        ),
    ]
    seed: Annotated[int, Field(description="Explicit deterministic voice-design seed.")]
    loudness: Annotated[
        float, Field(ge=-1, le=1, description="Explicit voice-design loudness setting.")
    ]
    guidance_scale: Annotated[
        float,
        Field(
            ge=0,
            le=100,
            description="Explicit strength of adherence to the voice-design brief.",
        ),
    ]


class ExistingVoiceConfig(FormModel):
    """Explicit native audition settings for voices already present in an account."""

    mode: Annotated[
        Literal["existing"], Field(description="Audition existing voices")
    ] = "existing"
    voice_ids: Annotated[
        dict[str, str],
        Field(description="Reviewed provisioned voice ID by persona label."),
    ]
    model: Annotated[
        str,
        Field(
            min_length=1,
            description="Explicit TTS model used for existing-voice auditions.",
        ),
    ]

    @model_validator(mode="after")
    def distinct_voices(self):
        if len(set(self.voice_ids.values())) != len(self.voice_ids):
            raise ValueError("existing personas require distinct voice IDs")
        return self


class AuditionSpec(FormModel):
    """Reviewed native text to audition a voice already in the account."""

    audition_text: Annotated[
        str, Field(min_length=100, description="Reviewed native audition text")
    ]


class VoiceSpec(AuditionSpec):
    """Acoustic brief and audition text used to design a new provider voice."""

    description: Annotated[
        str, Field(min_length=1, description="Reviewed acoustic design brief")
    ]


VoiceConfig = Annotated[
    VoiceDesignConfig | ExistingVoiceConfig, Field(discriminator="mode")
]


def _validate_voice_specs(
    config: VoiceConfig, specs: dict[str, VoiceSpec | AuditionSpec]
) -> None:
    if isinstance(config, ExistingVoiceConfig):
        if set(config.voice_ids) != set(specs):
            raise ValueError("existing voice IDs must cover exactly the personas")
        if any(isinstance(spec, VoiceSpec) for spec in specs.values()):
            raise ValueError(
                "existing voices need audition text, not design descriptions"
            )
    elif any(not isinstance(spec, VoiceSpec) for spec in specs.values()):
        raise ValueError("voice design requires acoustic descriptions")


class VoicePreview(FormModel):
    """Transient provider reply; never persisted in an authoring artifact."""

    generated_voice_id: Annotated[
        str,
        Field(
            min_length=1,
            description="Provider preview identity used for voice creation.",
        ),
    ]
    audio_base64: Annotated[
        str,
        Field(
            min_length=1, description="Exact provider audition bytes encoded as base64."
        ),
    ]
    media_type: Annotated[
        Literal["audio/mpeg", "audio/wav"],
        Field(description="Supported encoded audio format for audition validation."),
    ]

    @field_validator("audio_base64")
    @classmethod
    def valid_audio(cls, value):
        """Reject malformed base64 before accepting a provider response."""
        if not base64.b64decode(value, validate=True):
            raise ValueError("empty audition audio")
        return value

    def audio(self) -> bytes:
        """Decode and validate audio without rewriting provider bytes."""
        data = base64.b64decode(self.audio_base64, validate=True)
        _validate_audio(data, self.media_type)
        return data


def _validate_audio(data: bytes, media_type: str) -> None:
    """Check the actual encoded payload, including truncated WAV frames."""
    try:
        if media_type == "audio/wav":
            with wave.open(io.BytesIO(data)) as audio:
                if audio.getnframes() < 1 or audio.getnchannels() < 1:
                    raise ValueError("empty audition audio")
                expected = (
                    audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
                )
                if len(audio.readframes(audio.getnframes())) != expected:
                    raise ValueError("truncated audition audio")
        else:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(io.BytesIO(data), format="mp3")
            if len(audio) == 0:
                raise ValueError("empty audition audio")
    except Exception as exc:
        raise ValueError("invalid or empty audition audio") from exc


class AudioReference(FormModel):
    """Content-addressed audition bytes located relative to an asset bundle."""

    sha256: Annotated[
        str,
        Field(
            pattern="^[0-9a-f]{64}$",
            description="SHA256 of the exact encoded audio bytes.",
        ),
    ]
    media_type: Annotated[
        Literal["audio/mpeg", "audio/wav"],
        Field(description="Validated encoded audio format."),
    ]

    @property
    def relative_path(self) -> Path:
        """Derive the only permitted relative filename from hash and format."""
        extension = "wav" if self.media_type == "audio/wav" else "mp3"
        return Path("audio") / f"{self.sha256}.{extension}"

    def read(self, assets_root: Path) -> bytes:
        """Verify existence, digest and decodability every time evidence is used."""
        path = assets_root / self.relative_path
        if path.is_symlink() or not path.resolve().is_relative_to(
            assets_root.resolve()
        ):
            raise ValueError("audition audio escapes its asset bundle")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"missing or unreadable audition audio: {self.relative_path}"
            ) from exc
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"audition audio digest mismatch: {self.relative_path}")
        _validate_audio(data, self.media_type)
        return data


class StoredVoicePreview(FormModel):
    """Durable provider preview identity referencing one stored audition blob."""

    generated_voice_id: Annotated[
        str,
        Field(
            min_length=1,
            description="Provider preview identity used for voice creation.",
        ),
    ]
    audio: Annotated[
        AudioReference, Field(description="Portable verified audition evidence.")
    ]


def _publish_audio(data: bytes, target: Path, *, source: Path | None = None) -> None:
    """Atomically publish immutable bytes, preferring a checked local hardlink."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("audition audio destination cannot be a symlink")
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError("audition audio file differs")
        return
    if source is not None:
        try:
            os.link(source, target)
        except FileExistsError:
            if target.is_symlink() or target.read_bytes() != data:
                raise FileExistsError("audition audio file differs")
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
        else:
            if target.read_bytes() != data:
                raise ValueError("audition audio changed while linking")
            return
    # Copy into the target filesystem, then atomically reserve the final name.
    # os.link never replaces an existing reviewed file (including concurrent writes).
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".audio-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            if temporary.read_bytes() != data:
                raise ValueError("audition audio changed while copying")
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.is_symlink() or target.read_bytes() != data:
                    raise FileExistsError("audition audio file differs")
        finally:
            temporary.unlink(missing_ok=True)


def _store_preview(preview: VoicePreview, assets_root: Path) -> StoredVoicePreview:
    data = preview.audio()
    reference = AudioReference(
        sha256=hashlib.sha256(data).hexdigest(), media_type=preview.media_type
    )
    _publish_audio(data, assets_root / reference.relative_path)
    reference.read(assets_root)
    return StoredVoicePreview(
        generated_voice_id=preview.generated_voice_id, audio=reference
    )


def _load_preview(path: Path, assets_root: Path) -> StoredVoicePreview:
    preview = StoredVoicePreview.model_validate_json(path.read_text())
    preview.audio.read(assets_root)
    return preview


class ProvisionedVoice(FormModel):
    """The accessible provider voice identity, validated before publication."""

    voice_id: Annotated[
        str,
        Field(
            min_length=1,
            description="Provider voice identity; null only in unprovisioned drafts.",
        ),
    ]


class VoiceProvider(Protocol):
    """Injectable boundary used by live CLI and entirely offline tests."""

    def preview(self, spec: VoiceSpec, config: VoiceDesignConfig) -> VoicePreview: ...
    def create(
        self, persona_id: str, preview: StoredVoicePreview, spec: VoiceSpec
    ) -> ProvisionedVoice: ...
    def validate(self, voice_id: str) -> ProvisionedVoice: ...
    def audition(
        self, voice_id: str, spec: AuditionSpec, model: str
    ) -> VoicePreview: ...


class VoiceAssetsRequest(FormModel):
    """Reviewed draft and explicit per-persona acoustic inputs."""

    draft: Annotated[
        DraftArtifact,
        Field(description="Exact validated language draft to review or provision."),
    ]
    specs: Annotated[
        dict[str, VoiceSpec | AuditionSpec],
        Field(description="Persona label to reviewed voice design and audition text."),
    ]
    review: Annotated[
        ReviewApproval,
        Field(description="Digest-bound approval of the relevant staged artifact."),
    ]
    config: Annotated[
        VoiceConfig,
        Field(
            description="Explicit new-voice design or existing-voice audition settings."
        ),
    ]

    @model_validator(mode="after")
    def complete(self):
        _validate_voice_specs(self.config, self.specs)
        self.review.require(self.draft)
        if set(self.specs) != set(self.draft.pack.personas):
            raise ValueError("voice specifications must cover exactly the personas")
        return self


class VoiceCheckpoint(FormModel):
    """One completed provider side effect, retained across later-stage failures."""

    persona_id: Annotated[
        str, Field(description="Exact persona label from the staged language pack.")
    ]
    input_sha256: Annotated[
        str,
        Field(
            description="Digest of all validated stage inputs for resume invalidation."
        ),
    ]
    preview: Annotated[
        StoredVoicePreview,
        Field(
            description="Validated preview identity and content-addressed audition audio."
        ),
    ]
    voice: Annotated[
        ProvisionedVoice, Field(description="Accessible saved provider voice identity.")
    ]


class VoiceAssetsArtifact(FormModel):
    """Validated voice IDs, exact samples and full authoring-input provenance."""

    asset_version: Annotated[
        Literal["voice-assets-v3"],
        Field(description="Provisioning and audition artifact contract version."),
    ] = "voice-assets-v3"

    draft_sha256: Annotated[
        str,
        Field(description="Digest of the precise content draft these assets support."),
    ]
    input_sha256: Annotated[
        str,
        Field(
            description="Digest of all validated stage inputs for resume invalidation."
        ),
    ]
    content_review: Annotated[
        ReviewApproval,
        Field(description="Review of the exact staged language content."),
    ]
    config: Annotated[
        VoiceConfig,
        Field(
            description="Explicit new-voice design or existing-voice audition settings."
        ),
    ]
    specs: Annotated[
        dict[str, VoiceSpec | AuditionSpec],
        Field(description="Persona label to reviewed voice design and audition text."),
    ]
    voices: Annotated[
        dict[str, VoiceCheckpoint],
        Field(
            description="Complete persona-to-voice checkpoints and audition evidence."
        ),
    ]
    parent_sha256: Annotated[
        str | None,
        Field(description="Digest of the prior staged artifact for derived revisions."),
    ] = None

    @model_validator(mode="after")
    def complete(self):
        _validate_voice_specs(self.config, self.specs)
        if set(self.specs) != set(self.voices):
            raise ValueError("incomplete voice assets")
        if self.content_review.content_sha256 != self.draft_sha256:
            raise ValueError("asset review disagrees with draft")
        if len({v.voice.voice_id for v in self.voices.values()}) != len(self.voices):
            raise ValueError("personas must use distinct voices")
        for name, voice in self.voices.items():
            if voice.persona_id != name or voice.input_sha256 != self.input_sha256:
                raise ValueError("voice checkpoint disagrees with assets")
        return self


class ListeningEvidence(FormModel):
    """Exactly what a listener heard and approved; independent of draft revisions.

    Domain-only revisions rebind assets to a new draft without changing any of
    these fields, so the prior listening approval remains valid for them.
    """

    config: Annotated[
        VoiceConfig,
        Field(description="Voice design or existing-voice audition settings."),
    ]
    specs: Annotated[
        dict[str, VoiceSpec | AuditionSpec],
        Field(description="Persona label to reviewed voice design and audition text."),
    ]
    previews: Annotated[
        dict[str, StoredVoicePreview],
        Field(description="Persona label to the exact auditioned sample."),
    ]
    voices: Annotated[
        dict[str, ProvisionedVoice],
        Field(description="Persona label to the provider voice that was heard."),
    ]


def listening_digest(assets: VoiceAssetsArtifact) -> str:
    """Digest bound by the listening approval (voice gate)."""
    return content_digest(
        ListeningEvidence(
            config=assets.config,
            specs=assets.specs,
            previews={name: voice.preview for name, voice in assets.voices.items()},
            voices={name: voice.voice for name, voice in assets.voices.items()},
        )
    )


def require_listening_approval(
    approval: ReviewApproval, assets: VoiceAssetsArtifact
) -> None:
    """Reject a listening approval for different voices or samples."""
    if approval.content_sha256 != listening_digest(assets):
        raise ValueError("listening review does not match these voice samples")


class PendingVoice(FormModel):
    """Durable create intent: an uncertain create is never silently repeated."""

    persona_id: Annotated[
        str, Field(description="Exact persona label from the staged language pack.")
    ]
    input_sha256: Annotated[
        str,
        Field(
            description="Digest of all validated stage inputs for resume invalidation."
        ),
    ]
    preview: Annotated[
        StoredVoicePreview,
        Field(
            description="Validated preview identity and content-addressed audition audio."
        ),
    ]


def _checkpoint(request, name, preview, voice):
    return VoiceCheckpoint(
        persona_id=name,
        input_sha256=content_digest(request),
        preview=preview,
        voice=voice,
    )


def prepare_voices(
    request: VoiceAssetsRequest, output: Path, provider: VoiceProvider
) -> VoiceAssetsArtifact:
    """Resume validated checkpoints; never automatically repeat an uncertain create."""
    request = VoiceAssetsRequest.model_validate(request.model_dump())
    from tau2.multilingual.factory.pipeline import validate_draft

    validate_draft(request.draft)
    output.mkdir(parents=True, exist_ok=True)
    write_artifact(request, output / "request.json")
    if (output / "assets.json").exists():
        result = VoiceAssetsArtifact.model_validate_json(
            (output / "assets.json").read_text()
        )
        if result.input_sha256 != content_digest(request):
            raise ValueError("asset inputs changed")
        return validate_voices(result, output)
    voices = {}
    for index, (name, spec) in enumerate(request.specs.items()):
        # File names never derive from human persona labels.
        checkpoint = output / f"voice-{index}.json"
        pending = output / f"pending-{index}.json"
        preview_path = output / f"preview-{index}.json"
        if checkpoint.exists():
            saved = VoiceCheckpoint.model_validate_json(checkpoint.read_text())
            if saved.persona_id != name or saved.input_sha256 != content_digest(
                request
            ):
                raise ValueError("voice checkpoint belongs to different inputs")
            saved.preview.audio.read(output)
            voices[name] = saved
            continue
        if isinstance(request.config, ExistingVoiceConfig):
            preview = (
                _load_preview(preview_path, output) if preview_path.exists() else None
            )
            voice_id = request.config.voice_ids[name]
            voice = ProvisionedVoice.model_validate(
                provider.validate(voice_id).model_dump()
            )
            if voice.voice_id != voice_id:
                raise ValueError("provider returned another existing voice")
            if preview is None:
                response = VoicePreview.model_validate(
                    provider.audition(voice_id, spec, request.config.model).model_dump()
                )
                preview = _store_preview(response, output)
                write_artifact(preview, preview_path)
            saved = _checkpoint(request, name, preview, voice)
            write_artifact(saved, checkpoint)
            voices[name] = saved
            continue
        if pending.exists():
            intent = PendingVoice.model_validate_json(pending.read_text())
            if intent.persona_id != name or intent.input_sha256 != content_digest(
                request
            ):
                raise ValueError("pending creation belongs to different inputs")
            intent.preview.audio.read(output)
            raise RuntimeError(
                "uncertain voice creation; reconcile with record-created-voice before resuming"
            )
        if preview_path.exists():
            preview = _load_preview(preview_path, output)
        else:
            response = VoicePreview.model_validate(
                provider.preview(spec, request.config).model_dump()
            )
            preview = _store_preview(response, output)
            write_artifact(preview, preview_path)
        intent = PendingVoice(
            persona_id=name, input_sha256=content_digest(request), preview=preview
        )
        # Exclusive create is a cross-process reservation, unlike the idempotent writer.
        with pending.open("x") as handle:
            handle.write(intent.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        voice = ProvisionedVoice.model_validate(
            provider.create(name, preview, spec).model_dump()
        )
        # Persist immediately, before a validation API call can fail.
        saved = _checkpoint(request, name, preview, voice)
        write_artifact(saved, checkpoint)
        voices[name] = saved
    for voice in voices.values():
        voice.preview.audio.read(output)
        checked = ProvisionedVoice.model_validate(
            provider.validate(voice.voice.voice_id).model_dump()
        )
        if checked != voice.voice:
            raise ValueError("provider validated a different voice")
    result = VoiceAssetsArtifact(
        draft_sha256=content_digest(request.draft),
        input_sha256=content_digest(request),
        content_review=request.review,
        specs=request.specs,
        config=request.config,
        voices=voices,
    )
    validate_voices(result, output)
    write_artifact(result, output / "assets.json")
    return result


def recover_voice(
    output: Path, persona_id: str, voice_id: str, provider: VoiceProvider
) -> VoiceCheckpoint:
    """Record an explicitly reconciled account voice after an uncertain create."""
    request = VoiceAssetsRequest.model_validate_json(
        (output / "request.json").read_text()
    )
    index = list(request.specs).index(persona_id)
    pending = PendingVoice.model_validate_json(
        (output / f"pending-{index}.json").read_text()
    )
    if (
        pending.input_sha256 != content_digest(request)
        or pending.persona_id != persona_id
    ):
        raise ValueError("pending creation does not match request")
    pending.preview.audio.read(output)
    voice = ProvisionedVoice.model_validate(provider.validate(voice_id).model_dump())
    if voice.voice_id != voice_id:
        raise ValueError("provider returned another voice")
    result = _checkpoint(request, persona_id, pending.preview, voice)
    write_artifact(result, output / f"voice-{index}.json")
    return result


def validate_voices(
    assets: VoiceAssetsArtifact, assets_root: Path
) -> VoiceAssetsArtifact:
    """Validate an artifact and every referenced sample against an explicit root."""
    assets = VoiceAssetsArtifact.model_validate(assets.model_dump())
    for voice in assets.voices.values():
        voice.preview.audio.read(assets_root)
    return assets


class AuditionSample(FormModel):
    """Compact listening-review index entry, without source artifacts or audio."""

    persona_id: Annotated[str, Field(description="Persona to audition.")]
    voice_id: Annotated[str, Field(description="Provisioned provider voice ID.")]
    filename: Annotated[
        str,
        Field(
            pattern=r"^[0-9]+\.(?:wav|mp3)$",
            description="Relative exported audio filename.",
        ),
    ]
    media_type: Annotated[
        Literal["audio/wav", "audio/mpeg"], Field(description="Encoded sample format.")
    ]
    sha256: Annotated[
        str,
        Field(pattern="^[0-9a-f]{64}$", description="Digest of exact audition bytes."),
    ]


class AuditionIndex(FormModel):
    """Listening-review evidence bound to the exact validated asset artifact."""

    listening_sha256: Annotated[
        str,
        Field(
            pattern="^[0-9a-f]{64}$",
            description="Digest the listening approval binds (see listening_digest).",
        ),
    ]
    samples: Annotated[
        list[AuditionSample],
        Field(description="Persona identities and portable audio references."),
    ]


def audition_samples(
    assets: VoiceAssetsArtifact, output: Path, *, assets_root: Path
) -> list[Path]:
    """Export checked samples as local hardlinks or atomic cross-device copies."""
    assets = validate_voices(assets, assets_root)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    samples = []
    for index, (name, voice) in enumerate(assets.voices.items()):
        reference = voice.preview.audio
        path = output / f"{index}{reference.relative_path.suffix}"
        data = reference.read(assets_root)
        _publish_audio(data, path, source=assets_root / reference.relative_path)
        paths.append(path)
        samples.append(
            AuditionSample(
                persona_id=name,
                voice_id=voice.voice.voice_id,
                filename=path.name,
                media_type=reference.media_type,
                sha256=reference.sha256,
            )
        )
    write_artifact(
        AuditionIndex(listening_sha256=listening_digest(assets), samples=samples),
        output / "index.json",
    )
    return paths


class ElevenLabsVoiceProvider:
    """Thin ElevenLabs SDK adapter for voice design, creation and auditions."""

    def __init__(self):
        from elevenlabs.client import ElevenLabs

        self.client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    def preview(self, spec: VoiceSpec, config: VoiceDesignConfig) -> VoicePreview:
        """Design native-text previews without provider-generated audition prose."""
        previews = design_voice(
            self.client,
            VoiceDesignRequest(
                description=spec.description,
                text=spec.audition_text,
                model=config.model,
                loudness=config.loudness,
                guidance_scale=config.guidance_scale,
                seed=config.seed,
            ),
        )
        if not previews:
            raise ValueError("provider returned no previews")
        preview = previews[0]
        return VoicePreview(
            generated_voice_id=preview.generated_voice_id,
            audio_base64=preview.audio_base_64,
            media_type=preview.media_type,
        )

    def create(
        self, persona_id: str, preview: StoredVoicePreview, spec: VoiceSpec
    ) -> ProvisionedVoice:
        """Save the selected preview to the account exactly once per checkpoint."""
        voice_id = create_voice(
            self.client,
            VoiceCreateRequest(
                name=persona_id,
                description=spec.description,
                generated_voice_id=preview.generated_voice_id,
            ),
        )
        return ProvisionedVoice(voice_id=voice_id)

    def audition(self, voice_id: str, spec: AuditionSpec, model: str) -> VoicePreview:
        """Synthesize reviewed native audition text using an existing account voice."""
        data = b"".join(
            self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=spec.audition_text,
                model_id=model,
                output_format="mp3_44100_128",
            )
        )
        return VoicePreview(
            generated_voice_id=voice_id,
            audio_base64=base64.b64encode(data).decode(),
            media_type="audio/mpeg",
        )

    def validate(self, voice_id: str) -> ProvisionedVoice:
        """Ensure the configured account can access the saved voice."""
        return ProvisionedVoice(voice_id=self.client.voices.get(voice_id).voice_id)


class ReuseAssetsRequest(FormModel):
    """Reuse auditioned voices when adding domains without changing personas."""

    previous_draft: Annotated[
        DraftArtifact,
        Field(
            description="Original reviewed draft used to provision the existing voices."
        ),
    ]
    previous_assets_root: Annotated[
        Path,
        Field(description="Explicit root containing previous content-addressed audio."),
    ]
    previous_assets: Annotated[
        VoiceAssetsArtifact,
        Field(
            description="Validated previous voice artifact eligible for domain-only reuse."
        ),
    ]
    draft: Annotated[
        DraftArtifact,
        Field(description="Exact validated language draft to review or provision."),
    ]
    review: Annotated[
        ReviewApproval,
        Field(description="Digest-bound approval of the relevant staged artifact."),
    ]


def reuse_voices(request: ReuseAssetsRequest, output: Path) -> VoiceAssetsArtifact:
    """Rebind unchanged voices to a domain-only revision in a new asset directory.

    Like prepare-voices, ``output`` is a directory holding ``assets.json`` and the
    content-addressed audio. Listening evidence is unchanged, so the listening
    approval of the previous assets also covers the result.
    """
    request = ReuseAssetsRequest.model_validate(request.model_dump())
    from tau2.multilingual.factory.pipeline import validate_draft

    validate_draft(request.draft)
    previous = validate_voices(request.previous_assets, request.previous_assets_root)
    previous.content_review.require(request.previous_draft)
    request.review.require(request.draft)
    if (
        request.previous_draft.pack.language != request.draft.pack.language
        or request.previous_draft.pack.personas != request.draft.pack.personas
    ):
        raise ValueError("changed persona content requires new voice asset review")
    current = VoiceAssetsRequest(
        draft=request.draft,
        specs=previous.specs,
        review=request.review,
        config=previous.config,
    )
    voices = {
        name: voice.model_copy(update={"input_sha256": content_digest(current)})
        for name, voice in previous.voices.items()
    }
    artifact = VoiceAssetsArtifact(
        draft_sha256=content_digest(request.draft),
        input_sha256=content_digest(current),
        content_review=request.review,
        config=previous.config,
        specs=previous.specs,
        voices=voices,
        parent_sha256=content_digest(previous),
    )
    if listening_digest(artifact) != listening_digest(previous):
        raise ValueError("reused voices must keep their listening evidence")
    output.mkdir(parents=True, exist_ok=True)
    for voice in artifact.voices.values():
        reference = voice.preview.audio
        _publish_audio(
            reference.read(request.previous_assets_root),
            output / reference.relative_path,
            source=request.previous_assets_root / reference.relative_path,
        )
    validate_voices(artifact, output)
    write_artifact(artifact, output / "assets.json")
    return artifact


class FinalizeRequest(FormModel):
    """Validated stage artifacts and exact listening approval for publication."""

    assets_root: Annotated[
        Path,
        Field(
            description="Explicit root containing the reviewed content-addressed audio."
        ),
    ]

    draft: Annotated[DraftArtifact, Field(description="Reviewed draft to publish.")]
    assets: Annotated[
        VoiceAssetsArtifact,
        Field(description="Complete matching provisioned voice assets."),
    ]
    review: Annotated[
        ReviewApproval,
        Field(description="Listening approval bound to listening_digest(assets)."),
    ]

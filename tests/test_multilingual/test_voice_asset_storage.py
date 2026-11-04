"""Offline integrity, deduplication, and recovery of content-addressed auditions."""

import errno
import shutil

import pytest

from tau2.multilingual.factory import pipeline
from tau2.multilingual.factory import voice_assets as storage
from tau2.utils.artifacts import (
    ReviewApproval,
    content_digest,
)
from tests.test_multilingual import test_authoring_pipeline as fixtures
from tests.test_multilingual.test_authoring_pipeline import fake_generate, fixture_audio

draft_request = fixtures.draft_request


@pytest.fixture
def asset_request(draft_request, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    draft = pipeline.draft(draft_request, tmp_path / "draft.json")
    return storage.VoiceAssetsRequest(
        draft=draft,
        specs={
            name: storage.VoiceSpec(
                description="Fixture voice", audition_text="Fixture words. " * 10
            )
            for name in draft.pack.personas
        },
        review=ReviewApproval(
            content_sha256=content_digest(draft), reviewed_by="fixture"
        ),
        config=storage.VoiceDesignConfig(
            model="fixture", seed=1, loudness=0, guidance_scale=10
        ),
    )


class Provider:
    def __init__(self, *, interrupt=False):
        self.previews = self.creates = self.auditions = 0
        self.interrupt = interrupt

    def preview(self, spec, config):
        self.previews += 1
        return storage.VoicePreview(
            generated_voice_id=f"preview-{self.previews}",
            audio_base64=fixture_audio(),
            media_type="audio/wav",
        )

    def create(self, persona_id, preview, spec):
        assert isinstance(preview, storage.StoredVoicePreview)
        self.creates += 1
        if self.interrupt:
            self.interrupt = False
            raise RuntimeError("uncertain create")
        return storage.ProvisionedVoice(voice_id=f"voice-{self.creates}")

    def validate(self, voice_id):
        return storage.ProvisionedVoice(voice_id=voice_id)

    def audition(self, voice_id, spec, model):
        self.auditions += 1
        return storage.VoicePreview(
            generated_voice_id=voice_id,
            audio_base64=fixture_audio(),
            media_type="audio/wav",
        )


def test_audio_once_compact_index_idempotence_and_portability(asset_request, tmp_path):
    provider = Provider()
    root = tmp_path / "assets"
    result = storage.prepare_voices(asset_request, root, provider)
    assert result.asset_version == "voice-assets-v3"
    blobs = list((root / "audio").glob("*"))
    assert len(blobs) == 1  # Different voices can return identical fixture bytes.
    for path in root.glob("*.json"):
        assert "audio_base64" not in path.read_text()
        assert fixture_audio() not in path.read_text()
    assert storage.prepare_voices(asset_request, root, provider) == result
    assert (provider.previews, provider.creates) == (2, 2)
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert storage.validate_voices(result, relocated) == result
    paths = storage.audition_samples(
        result, tmp_path / "audition", assets_root=relocated
    )
    index = storage.AuditionIndex.model_validate_json(
        (tmp_path / "audition/index.json").read_text()
    )
    assert index.listening_sha256 == storage.listening_digest(result)
    assert set(index.model_dump()) == {"listening_sha256", "samples"}
    assert len(index.samples) == len(paths) == 2
    for sample, path in zip(index.samples, paths):
        assert set(sample.model_dump()) == {
            "persona_id",
            "voice_id",
            "filename",
            "media_type",
            "sha256",
        }
        assert (
            path.stat().st_ino
            == (
                relocated
                / next(iter(result.voices.values())).preview.audio.relative_path
            )
            .stat()
            .st_ino
        )
    assert (
        storage.audition_samples(result, tmp_path / "audition", assets_root=relocated)
        == paths
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_or_changed_audio_refuses_every_consumption(
    asset_request, tmp_path, corrupt
):
    provider = Provider()
    root = tmp_path / "assets"
    result = storage.prepare_voices(asset_request, root, provider)
    blob = root / next(iter(result.voices.values())).preview.audio.relative_path
    if corrupt:
        blob.write_bytes(b"corrupted")
    else:
        blob.unlink()
    for consume in [
        lambda: storage.validate_voices(result, root),
        lambda: storage.prepare_voices(asset_request, root, provider),
        lambda: storage.audition_samples(
            result, tmp_path / "audition", assets_root=root
        ),
    ]:
        with pytest.raises(ValueError, match="audio"):
            consume()
    assert (provider.previews, provider.creates) == (2, 2)
    assert not (tmp_path / "audition/index.json").exists()


def test_uncertain_create_recovery_uses_same_blob_without_repeating(
    asset_request, tmp_path
):
    provider = Provider(interrupt=True)
    root = tmp_path / "assets"
    with pytest.raises(RuntimeError, match="uncertain"):
        storage.prepare_voices(asset_request, root, provider)
    with pytest.raises(RuntimeError, match="record-created-voice"):
        storage.prepare_voices(asset_request, root, provider)
    assert (provider.previews, provider.creates) == (1, 1)
    first = next(iter(asset_request.specs))
    recovered = storage.recover_voice(root, first, "voice-1", provider)
    assert recovered.preview.audio.relative_path.is_relative_to("audio")
    result = storage.prepare_voices(asset_request, root, provider)
    assert (provider.previews, provider.creates) == (2, 2)
    assert storage.validate_voices(result, root) == result
    assert len(list((root / "audio").glob("*"))) == 1


def test_corrupt_pending_audio_refuses_recovery_before_provider(
    asset_request, tmp_path
):
    provider = Provider(interrupt=True)
    root = tmp_path / "assets"
    with pytest.raises(RuntimeError):
        storage.prepare_voices(asset_request, root, provider)
    next((root / "audio").glob("*")).write_bytes(b"broken")
    provider.validate = lambda _: pytest.fail("Must check local audio before provider")
    with pytest.raises(ValueError, match="audio"):
        storage.recover_voice(
            root, next(iter(asset_request.specs)), "voice-1", provider
        )
    with pytest.raises(ValueError, match="audio"):
        storage.prepare_voices(asset_request, root, provider)


def test_existing_voice_config_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="distinct voice IDs"):
        storage.ExistingVoiceConfig(
            voice_ids={"male": "shared-voice", "female": "shared-voice"},
            model="fixture",
        )


def test_duplicate_existing_voices_refuse_before_provider_work(asset_request, tmp_path):
    provider = Provider()
    asset_request.config = storage.ExistingVoiceConfig(
        voice_ids={name: f"existing-{i}" for i, name in enumerate(asset_request.specs)},
        model="fixture",
    )
    asset_request.specs = {
        name: storage.AuditionSpec(audition_text=spec.audition_text)
        for name, spec in asset_request.specs.items()
    }
    # Mutable caller input must be checked again at the stage boundary.
    asset_request.config.voice_ids = dict.fromkeys(asset_request.specs, "shared-voice")
    provider.validate = lambda _: pytest.fail("Must reject before provider access")
    output = tmp_path / "duplicate-voices"
    with pytest.raises(ValueError, match="distinct voice IDs"):
        storage.prepare_voices(asset_request, output, provider)
    assert provider.auditions == 0
    assert not output.exists()


def test_existing_voices_and_reuse_copy_evidence_without_provisioning(
    asset_request, tmp_path
):
    provider = Provider()
    asset_request.config = storage.ExistingVoiceConfig(
        voice_ids={name: f"existing-{i}" for i, name in enumerate(asset_request.specs)},
        model="fixture",
    )
    asset_request.specs = {
        name: storage.AuditionSpec(audition_text=spec.audition_text)
        for name, spec in asset_request.specs.items()
    }
    root = tmp_path / "assets"
    result = storage.prepare_voices(asset_request, root, provider)
    assert (provider.previews, provider.creates, provider.auditions) == (0, 0, 2)
    draft = asset_request.draft.model_copy(deep=True)
    draft.pack.revision = "fixture-next"
    request = storage.ReuseAssetsRequest(
        previous_draft=asset_request.draft,
        previous_assets=result,
        previous_assets_root=root,
        draft=draft,
        review=ReviewApproval(
            content_sha256=content_digest(draft), reviewed_by="fixture"
        ),
    )
    output = tmp_path / "reused"
    reused = storage.reuse_voices(request, output)
    assert storage.validate_voices(reused, output) == reused
    assert storage.reuse_voices(request, output) == reused
    assert storage.listening_digest(reused) == storage.listening_digest(result)
    assert len(list((output / "audio").glob("*"))) == 1
    old_bytes = (output / "assets.json").read_bytes()
    next((root / "audio").glob("*")).write_bytes(b"broken")
    with pytest.raises(ValueError, match="audio"):
        storage.reuse_voices(request, output)
    assert (output / "assets.json").read_bytes() == old_bytes


def test_audition_cross_device_copy_fallback(asset_request, tmp_path, monkeypatch):
    root = tmp_path / "assets"
    result = storage.prepare_voices(asset_request, root, Provider())
    original_link = storage.os.link

    def cross_device(source, destination, *args, **kwargs):
        if str(source).startswith(str(root)):
            raise OSError(errno.EXDEV, "cross-device link")
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(storage.os, "link", cross_device)
    paths = storage.audition_samples(result, tmp_path / "audition", assets_root=root)
    assert all(
        path.read_bytes() == next((root / "audio").glob("*")).read_bytes()
        for path in paths
    )
    assert (
        storage.audition_samples(result, tmp_path / "audition", assets_root=root)
        == paths
    )


@pytest.mark.parametrize("corrupt", [False, True])
def test_preview_only_resume_checks_blob_before_create(
    asset_request, tmp_path, monkeypatch, corrupt
):
    provider = Provider()
    root = tmp_path / "assets"
    original_write = storage.write_artifact

    def interrupt_after_preview(value, path):
        result = original_write(value, path)
        if path.name == "preview-0.json":
            raise RuntimeError("interrupted after durable preview")
        return result

    monkeypatch.setattr(storage, "write_artifact", interrupt_after_preview)
    with pytest.raises(RuntimeError, match="durable preview"):
        storage.prepare_voices(asset_request, root, provider)
    assert (provider.previews, provider.creates) == (1, 0)
    monkeypatch.setattr(storage, "write_artifact", original_write)
    if corrupt:
        next((root / "audio").glob("*")).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="audio"):
            storage.prepare_voices(asset_request, root, provider)
        assert (provider.previews, provider.creates) == (1, 0)
    else:
        result = storage.prepare_voices(asset_request, root, provider)
        assert storage.validate_voices(result, root) == result
        assert (provider.previews, provider.creates) == (2, 2)


def test_checkpoint_resume_validates_audio_before_provider(asset_request, tmp_path):
    provider = Provider()
    root = tmp_path / "assets"

    def unavailable(_):
        raise RuntimeError("provider unavailable")

    provider.validate = unavailable
    with pytest.raises(RuntimeError, match="unavailable"):
        storage.prepare_voices(asset_request, root, provider)
    assert (provider.previews, provider.creates) == (2, 2)
    assert (root / "voice-0.json").exists()
    assert not (root / "assets.json").exists()
    next((root / "audio").glob("*")).write_bytes(b"corrupt")
    provider.validate = lambda _: pytest.fail("Must validate local audio first")
    with pytest.raises(ValueError, match="audio"):
        storage.prepare_voices(asset_request, root, provider)
    assert (provider.previews, provider.creates) == (2, 2)


def test_audio_reference_requires_decodable_bytes_even_with_matching_hash(tmp_path):
    import hashlib

    data = b"not really a WAV"
    reference = storage.AudioReference(
        sha256=hashlib.sha256(data).hexdigest(), media_type="audio/wav"
    )
    path = tmp_path / reference.relative_path
    path.parent.mkdir()
    path.write_bytes(data)
    with pytest.raises(ValueError, match="audition audio"):
        reference.read(tmp_path)


@pytest.mark.parametrize("version", ["voice-assets-v1", "voice-assets-v2"])
def test_embedded_old_asset_version_is_rejected(asset_request, tmp_path, version):
    result = storage.prepare_voices(asset_request, tmp_path / "assets", Provider())
    value = result.model_dump()
    value["asset_version"] = version
    with pytest.raises(ValueError, match="asset_version"):
        storage.VoiceAssetsArtifact.model_validate(value)

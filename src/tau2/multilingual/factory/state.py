# Copyright Sierra
"""Persistent factory project state.

One :class:`FactoryProject` per language being onboarded, persisted as plain
JSON at ``data/tau2/multilingual/_factory/<lang>/project.json``. The
``_factory`` folder is a workspace, NOT a language pack: nothing in it is
discovered by the Language Factory loaders.

SAFETY: the pack loader globs ``data/tau2/multilingual/*/pack.yaml``, so
factory drafts are named ``pack_draft.yaml`` and :func:`factory_file_path`
refuses to produce a ``pack.yaml`` path under ``_factory/``.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN

PROJECT_FILENAME = "project.json"
PACK_DRAFT_FILENAME = "pack_draft.yaml"


class FactoryStage(StrEnum):
    """Persistent language-factory stages, in execution order."""

    DRAFT = "draft"
    NATIVENESS = "nativeness"
    FINALIZE = "finalize"


class StageStatus(StrEnum):
    """Closed outcome state for a factory stage."""

    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


def factory_root_dir() -> Path:
    """The ``_factory`` root under the CURRENT ``tau2.utils.DATA_DIR``.

    Resolved dynamically (not snapshotted at import) so all factory paths share
    one root and test fixtures that monkeypatch ``tau2.utils.DATA_DIR`` keep
    factory state inside their temporary data dir.
    """
    import tau2.utils

    return tau2.utils.DATA_DIR / "tau2" / "multilingual" / "_factory"


def factory_project_dir(language: str) -> Path:
    """The factory workspace folder for one language."""
    return factory_root_dir() / language


def factory_file_path(language: str, filename: str) -> Path:
    """A path inside the language's factory workspace, with the pack.yaml guard.

    Raises:
        ValueError: If ``filename`` is ``pack.yaml`` — the Language Factory loader
            discovers ``*/pack.yaml``, so a draft must be named
            ``pack_draft.yaml`` until finalize promotes it out of ``_factory/``.
    """
    if Path(filename).name == "pack.yaml":
        raise ValueError(
            "Refusing to create 'pack.yaml' under _factory/ — the pack loader "
            f"would discover it. Use '{PACK_DRAFT_FILENAME}' for drafts."
        )
    return factory_project_dir(language) / filename


class HumanChecklist(BaseModel):
    """The human-bottleneck checklist: steps automation cannot do.

    Each flag is flipped by a human once the corresponding deliverable exists.
    """

    model_config = ConfigDict(extra="forbid")

    voice_ids: bool = Field(
        default=False,
        description="TTS voice ids picked for every persona (audition winner)",
    )
    noise_wavs: bool = Field(
        default=False,
        description="Locale-specific noise recordings delivered under "
        "data/voice/background_noise_audio_pcm_mono_verified/",
    )
    script_range: bool = Field(
        default=False,
        description="The language's script exists in invariants.SCRIPT_RANGES "
        "(engineer-extended if it is a new script)",
    )
    native_calibration: bool = Field(
        default=False,
        description="A native speaker reviewed personas, prompts, and task "
        "translations",
    )


class FactoryProject(BaseModel):
    """All persistent state for one language's factory onboarding."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(description="ISO 639-1 language code, e.g. 'sw'")
    script: Optional[str] = Field(
        default=None,
        description="ISO 15924 script code for the language, e.g. 'latn'",
    )
    domain: str = Field(
        default=DEFAULT_MULTILINGUAL_DOMAIN,
        description="Domain whose tasks the language is localized for",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description="Model used for factory LLM calls (drafting, translation). "
        "None uses the factory default at call time.",
    )
    llm_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra arguments for factory LLM calls (temperature, ...)",
    )
    pack_revision: int = Field(
        default=0,
        description="Monotonic revision of the pack draft; bumped on every "
        "draft regeneration or edit pass",
    )
    stages: dict[FactoryStage, StageStatus] = Field(
        default_factory=lambda: {stage: StageStatus.PENDING for stage in FactoryStage},
        description="Validated pipeline stage outcomes.",
    )
    checklist: HumanChecklist = Field(default_factory=HumanChecklist)
    guidelines_final_filename: Optional[str] = Field(
        default=None,
        description="The PACK-RELATIVE guidelines path the draft declared for "
        "the FINAL pack (the workspace copy is always guidelines_draft.md; "
        "finalize restores this path). None falls back to the "
        "guidelines/simulation_guidelines_voice_<lang>.md convention — no run "
        "reads the localized guidelines, and finalize writes them into the "
        "pack's guidelines/ subdirectory.",
    )
    guidelines_text_final_filename: Optional[str] = Field(
        default=None,
        description="The PACK-RELATIVE TEXT guidelines path the draft declared "
        "for the FINAL pack (the workspace copy is always "
        "guidelines_text_draft.md; finalize restores this path). None falls "
        "back to the guidelines/simulation_guidelines_text_<lang>.md convention.",
    )

    @field_validator("stages")
    @classmethod
    def require_all_stages(
        cls, stages: dict[FactoryStage, StageStatus]
    ) -> dict[FactoryStage, StageStatus]:
        missing = set(FactoryStage) - set(stages)
        if missing:
            raise ValueError(f"missing factory stages: {sorted(missing)}")
        return stages

    @property
    def project_dir(self) -> Path:
        return factory_project_dir(self.language)

    @property
    def path(self) -> Path:
        return factory_file_path(self.language, PROJECT_FILENAME)

    @property
    def pack_draft_path(self) -> Path:
        return factory_file_path(self.language, PACK_DRAFT_FILENAME)

    def save(self) -> Path:
        """Atomically write project.json, creating the workspace if needed."""
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        try:
            tmp.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def mark_stage(self, stage: FactoryStage, *, success: bool) -> None:
        """Record and persist one stage outcome."""
        self.stages[stage] = StageStatus.DONE if success else StageStatus.FAILED
        self.save()

    @classmethod
    def load(cls, language: str) -> Optional["FactoryProject"]:
        """Load a language's project, or None if it does not exist."""
        path = factory_file_path(language, PROJECT_FILENAME)
        if not path.exists():
            return None
        return cls.model_validate(json.loads(path.read_text()))

    @classmethod
    def load_or_create(cls, language: str, **defaults: Any) -> "FactoryProject":
        """Load the project, creating (and persisting) it if missing."""
        project = cls.load(language)
        if project is not None:
            return project
        project = cls(language=language, **defaults)
        project.save()
        logger.info(f"Created factory project for '{language}' at {project.path}")
        return project

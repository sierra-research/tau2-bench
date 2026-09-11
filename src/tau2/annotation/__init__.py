# Copyright Sierra
"""The annotation factory: human-facing artifacts over judged results.

Third pillar of the multilingual machine (factory → judges → annotation):
turns stored results + judge verdicts + factory translation state into
annotator sheets/workbooks/audit tabs, and ingests the filled sheets back
into agreement metrics with LIVE/shadow gating.

Boundaries: imports ``tau2.judges.export`` (typed verdict records),
``tau2.data_model``, and the factory's typed surface — nothing imports
annotation. Every export is a provenance-bearing artifact
(``tau2.annotation.artifacts``); every sheet contract is generated from the
row models (``tau2.annotation.models``).
"""

from tau2.annotation.artifacts import (
    ArtifactManifest,
    LoadedArtifact,
    ManifestFile,
    PacketEntry,
    read_artifact,
    write_sheet_family,
)
from tau2.annotation.loading import LoadedSim, iter_loaded_sims
from tau2.annotation.models import (
    AuditNuanceRow,
    ColdLabel,
    ColdSheet,
    ColdSheetRow,
    ColdSidecarRow,
    CommunicateJudgeRow,
    ErrorAnalysisRow,
    FactorKeyRow,
    JudgePrecisionRow,
    PrecisionVerdict,
    RealismRow,
    SheetRow,
    TranslationReviewRow,
    VoiceReviewRow,
    YesNo,
)

__all__ = [
    "ArtifactManifest",
    "AuditNuanceRow",
    "ColdLabel",
    "ColdSheet",
    "ColdSheetRow",
    "ColdSidecarRow",
    "CommunicateJudgeRow",
    "ErrorAnalysisRow",
    "FactorKeyRow",
    "JudgePrecisionRow",
    "LoadedArtifact",
    "LoadedSim",
    "ManifestFile",
    "PacketEntry",
    "PrecisionVerdict",
    "RealismRow",
    "SheetRow",
    "TranslationReviewRow",
    "VoiceReviewRow",
    "YesNo",
    "iter_loaded_sims",
    "read_artifact",
    "write_sheet_family",
]

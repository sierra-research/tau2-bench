"""The 2.0 leaderboard package: rewards, judge diagnostics and display metadata.

A package is three or four files in one directory:

- ``benchmark-submission.json``: the frozen frame, every task/trial reward and
  the recomputable summary (``tau2.data_model.benchmark.BenchmarkSubmission``).
- ``benchmark-diagnostics.json``: voice-v2 only; per-cell judge summaries and
  verdict fingerprints (``tau2.judges.submission.BenchmarkDiagnostics``).
- ``benchmark-quality.json``: voice-v2 only; call-level Interaction and
  Experience (Generation quality) per scored call
  (``tau2.scripts.leaderboard.benchmark_quality.BenchmarkQuality``).
- ``benchmark-pass3.json``: voice-v2-entity only; each task's outcome in the
  three crossed conditions and the Pass^3 score the leaderboard shows
  (``tau2.scripts.leaderboard.benchmark_pass3.BenchmarkPass3``).
- ``benchmark-metadata.json``: who submitted what (``PackageMetadata``).

``tau2 submit prepare`` writes a package from a saved matrix; the maintainer-only
historical adapter writes the same package from the frozen 2026-10-02 exports.
The website reads these files directly.
"""

import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.benchmark import BenchmarkSubmission
from tau2.judges import submission as judge_submission
from tau2.judges.submission import DIAGNOSTICS_FILE_NAME, BenchmarkDiagnostics
from tau2.metrics.benchmark import validate_submission
from tau2.scripts.leaderboard.benchmark_pass3 import (
    PASS3_BENCHMARKS,
    PASS3_FILE_NAME,
    BenchmarkPass3,
    check_pass3,
)
from tau2.scripts.leaderboard.benchmark_quality import (
    QUALITY_BENCHMARKS,
    QUALITY_FILE_NAME,
    BenchmarkQuality,
    check_quality,
)
from tau2.scripts.leaderboard.submission import Reference

SUBMISSION_FILE_NAME = "benchmark-submission.json"
METADATA_FILE_NAME = "benchmark-metadata.json"
PACKAGE_FILE_NAMES = (
    SUBMISSION_FILE_NAME,
    DIAGNOSTICS_FILE_NAME,
    QUALITY_FILE_NAME,
    PASS3_FILE_NAME,
    METADATA_FILE_NAME,
)

# How the package's evidence root is laid out, and which reader recomputes it.
# "matrix": a `tau2 run --benchmark ... --save-to` root (`<lang>-<domain>/` cells).
# "historical-voice-v2-2026-10-02": the frozen release export
# (`trajectories/<lang>/<domain>/`) plus its saved judge verdicts.
Evidence = Literal["matrix", "historical-voice-v2-2026-10-02"]
# Package-name suffixes of the domain frames that fill an agent's voice-v2 row.
DOMAIN_PACKAGE_SUFFIXES = {"voice-v2-banking": "banking", "voice-v2-entity": "entity"}
# Release-export directories of frame domains whose names differ there; entity
# scores the export's regular condition; Pass^3 adds the other two conditions.
HISTORICAL_EXPORT_DIRS = {"banking_knowledge": "banking", "intake": "intake/regular"}
# Export directories of the non-regular crossed Pass^3 conditions, by label.
HISTORICAL_PASS3_DIRS = {
    "channel_heavy": "intake/channel-heavy",
    "speech_heavy": "intake/speech-heavy",
}


class PackageContact(BaseModel):
    """Who to contact about the submission."""

    model_config = ConfigDict(extra="forbid")
    email: Annotated[str, Field(min_length=1, description="Contact email.")]
    name: Annotated[str | None, Field(description="Submitter name.")] = None
    github: Annotated[str | None, Field(description="GitHub username.")] = None


class SubmitterMetadata(BaseModel):
    """The fields a submitter supplies with ``--metadata-file``."""

    model_config = ConfigDict(extra="forbid")
    model_name: Annotated[str, Field(min_length=1, description="Displayed name.")]
    model_organization: Annotated[
        str, Field(min_length=1, description="Organization that built the model.")
    ]
    submitting_organization: Annotated[
        str, Field(min_length=1, description="Organization that ran the benchmark.")
    ]
    submission_date: Annotated[date, Field(description="Submission date.")]
    contact_info: PackageContact
    submission_type: Annotated[
        Literal["standard", "custom"],
        Field(description="custom: modified benchmark-side setup (see the guide)."),
    ] = "standard"
    references: Annotated[
        list[Reference], Field(description="Papers, blog posts, model cards.")
    ] = []
    notes: Annotated[
        str | None, Field(description="How the agent was configured and run.")
    ] = None


class PackageMetadata(SubmitterMetadata):
    """Display metadata plus the recorded agent identity and evidence location."""

    schema_version: Literal["benchmark-metadata-v1"] = "benchmark-metadata-v1"
    agent_provider: Annotated[
        str | None,
        Field(description="Recorded audio-native provider; null for text."),
    ]
    agent_model: Annotated[
        str, Field(min_length=1, description="Recorded agent model.")
    ]
    evidence: Annotated[
        Evidence, Field(description="Layout of the saved evidence the package cites.")
    ] = "matrix"
    trajectories: Annotated[
        str | None,
        Field(
            description="S3 prefix of the uploaded evidence; set by maintainers.",
            pattern=r"^s3://[a-z0-9.-]+/.+[^/]$",
        ),
    ] = None


class BenchmarkPackage(BaseModel):
    """A loaded, self-consistent package."""

    model_config = ConfigDict(extra="forbid")
    metadata: PackageMetadata
    submission: BenchmarkSubmission
    diagnostics: BenchmarkDiagnostics | None
    quality: BenchmarkQuality | None = None
    pass3: BenchmarkPass3 | None = None

    def check(self) -> "BenchmarkPackage":
        """Recompute the reward summary and cross-check the package files."""
        validate_submission(self.submission)
        if not self.submission.summary.complete:
            raise ValueError("A leaderboard package requires a complete matrix")
        frame = self.submission.frame
        judged = frame.benchmark_version in judge_submission.JUDGED_BENCHMARKS
        if judged and self.diagnostics is None:
            raise ValueError(f"Package lacks {DIAGNOSTICS_FILE_NAME}")
        if not judged and self.diagnostics is not None:
            raise ValueError(f"{frame.benchmark_version} carries no judge diagnostics")
        if self.diagnostics is not None:
            if (
                self.diagnostics.benchmark_version != frame.benchmark_version
                or self.diagnostics.frame_revision != frame.revision
            ):
                raise ValueError("Diagnostics and rewards cite different frames")
            cells = {(cell.language, cell.domain) for cell in frame.cells}
            judged_cells = [
                (cell.language, cell.domain) for cell in self.diagnostics.cells
            ]
            if len(judged_cells) != len(cells) or set(judged_cells) != cells:
                raise ValueError("Diagnostics must cover every frame cell once")
        # Call quality reads judge verdicts, so only judged frames carry it.
        rated = judged and frame.benchmark_version in QUALITY_BENCHMARKS
        if rated and self.quality is None:
            raise ValueError(f"Package lacks {QUALITY_FILE_NAME}")
        if not rated and self.quality is not None:
            raise ValueError(f"{frame.benchmark_version} carries no call quality")
        if self.quality is not None:
            check_quality(self.quality, frame)
        if self.pass3 is not None:
            if frame.benchmark_version not in PASS3_BENCHMARKS:
                raise ValueError(f"{frame.benchmark_version} carries no Pass^3")
            check_pass3(self.pass3, frame, self.submission.trials)
        voice = frame.benchmark_version.startswith("voice")
        if voice != (self.metadata.agent_provider is not None):
            raise ValueError("agent_provider is required for voice and only for voice")
        return self

    @property
    def name(self) -> str:
        return package_name(self.metadata, self.submission.frame.benchmark_version)

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SUBMISSION_FILE_NAME).write_text(
            self.submission.model_dump_json(indent=2) + "\n"
        )
        if self.diagnostics is not None:
            (directory / DIAGNOSTICS_FILE_NAME).write_text(
                self.diagnostics.model_dump_json(indent=2) + "\n"
            )
        if self.quality is not None:
            (directory / QUALITY_FILE_NAME).write_text(
                self.quality.model_dump_json(indent=2) + "\n"
            )
        if self.pass3 is not None:
            (directory / PASS3_FILE_NAME).write_text(
                self.pass3.model_dump_json(indent=2) + "\n"
            )
        (directory / METADATA_FILE_NAME).write_text(
            self.metadata.model_dump_json(indent=2) + "\n"
        )


def package_name(
    metadata: SubmitterMetadata, benchmark_version: str | None = None
) -> str:
    """``{model_name}_{model_organization}_{submission_date}[_{suffix}]``, slugified.

    One agent's voice-v2 and domain packages share their display metadata so the
    leaderboard groups them; a domain frame's suffix keeps their directories
    apart. voice-v2 and text-v1 names take no suffix.
    """

    def slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

    parts = [
        slug(metadata.model_name),
        slug(metadata.model_organization),
        metadata.submission_date.isoformat(),
    ]
    if benchmark_version in DOMAIN_PACKAGE_SUFFIXES:
        parts.append(DOMAIN_PACKAGE_SUFFIXES[benchmark_version])
    return "_".join(parts)


def load_package(directory: Path) -> BenchmarkPackage:
    """Read and self-check a package; validating it against evidence is separate."""
    for name in (SUBMISSION_FILE_NAME, METADATA_FILE_NAME):
        if not (directory / name).is_file():
            raise ValueError(f"Package {directory} lacks {name}")
    diagnostics_path = directory / DIAGNOSTICS_FILE_NAME
    quality_path = directory / QUALITY_FILE_NAME
    pass3_path = directory / PASS3_FILE_NAME
    return BenchmarkPackage(
        metadata=PackageMetadata.model_validate_json(
            (directory / METADATA_FILE_NAME).read_text()
        ),
        submission=BenchmarkSubmission.model_validate_json(
            (directory / SUBMISSION_FILE_NAME).read_text()
        ),
        diagnostics=BenchmarkDiagnostics.model_validate_json(
            diagnostics_path.read_text()
        )
        if diagnostics_path.is_file()
        else None,
        quality=BenchmarkQuality.model_validate_json(quality_path.read_text())
        if quality_path.is_file()
        else None,
        pass3=BenchmarkPass3.model_validate_json(pass3_path.read_text())
        if pass3_path.is_file()
        else None,
    ).check()


def is_package(directory: Path) -> bool:
    return (directory / SUBMISSION_FILE_NAME).is_file()


def evidence_cells(
    evidence_root: Path, evidence: Evidence, cells: list[tuple[str, str]]
) -> dict[str, Path]:
    """Map each uploaded ``<lang>/<domain>`` path to its local evidence directory."""
    return {
        f"{language}/{domain}": (
            evidence_root / f"{language}-{domain}"
            if evidence == "matrix"
            else evidence_root
            / "trajectories"
            / language
            / HISTORICAL_EXPORT_DIRS.get(domain, domain)
        )
        for language, domain in cells
    }

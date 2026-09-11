# Copyright Sierra
"""Provenance-bearing sheet families: CSVs + manifest written atomically.

An annotation export is a FAMILY of CSVs (the annotator sheet plus sidecars)
and a ``<stem>.manifest.json`` recording what produced them: kind, git sha,
provenance, and a sha256 per file. ``write_sheet_family`` writes them all in
one atomic step — an export cannot exist without its provenance. Ingest
(``read_artifact``) hard-fails without a manifest, verifies checksums on
non-editable members (the annotator must not have touched the answer key),
and validates the edited CSV's header set against the registered row model
for the artifact kind before returning typed rows.

Idempotency (machine principle): ``batch_id`` is CONTENT-DERIVED — the sha256
of (kind, batch_name, per-role content hashes), truncated — so the same
inputs + code version produce the same id. ``created_at`` is the only
nondeterministic field in the manifest.
"""

import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Optional, Sequence, Union

from loguru import logger
from pydantic import BaseModel, Field

from tau2.annotation.models import (
    COLD_META_LABELS,
    COLD_WIDE,
    SHEET_FAMILIES,
    ColdSheet,
    FactorKeyRow,
    SheetRow,
)

ARTIFACT_SCHEMA_VERSION = 1

ArtifactKind = Literal[
    "translation_review",
    "communicate_judge",
    "nativeness_precision",
    "nativeness_cold",
    "packet_error_analysis",
    "packet_user_realism",
    "packet_voice_review",
    "packet_prompt_bed_review",
    "packet_audio_quality",
    "packet_rubric",
    "packet_nativeness_labels",
    "packet_nativeness_adjudication_call",
    "packet_nativeness_adjudication_utterance",
    "packet_judge_calibration",
    "packet_judge_calibration_adjudication",
    "packet_judge_calibration_decisions",
]

# The family's main (annotator-facing) role writes to <stem>.csv; every other
# role writes to <stem>_<role>.csv.
MAIN_ROLE = "sheet"

# CSVs are written utf-8-sig so Excel/Sheets open native-script content
# correctly; readers strip the BOM with encoding="utf-8-sig".
CSV_ENCODING = "utf-8-sig"

# Formula-injection guard: LLM/judge/user-sim content flows into these CSVs,
# and Excel/Sheets execute cells starting with these characters as formulas
# (tab/CR-leading cells can be coerced too). Exported cells starting with one
# are armored with a single leading apostrophe; ingest strips exactly one so
# round-trips are lossless.
FORMULA_GUARD_CHARS = ("=", "+", "-", "@", "\t", "\r")


def armor_cell(value: str) -> str:
    """Neutralize a would-be spreadsheet formula with a leading apostrophe."""
    if value and value[0] in FORMULA_GUARD_CHARS:
        return "'" + value
    return value


def dearmor_cell(value: str) -> str:
    """Strip exactly ONE armoring apostrophe (the inverse of ``armor_cell``)."""
    if len(value) >= 2 and value[0] == "'" and value[1] in FORMULA_GUARD_CHARS:
        return value[1:]
    return value


class ManifestFile(BaseModel):
    """One member of a sheet family."""

    name: Annotated[str, Field(description="File name (relative to the manifest).")]
    sha256: Annotated[str, Field(description="sha256 of the file as exported.")]
    role: Annotated[
        str, Field(description="Family role, e.g. 'sheet' / 'judge_sidecar'.")
    ]
    editable: Annotated[
        bool,
        Field(
            description="True for the annotator-facing sheet (its checksum is "
            "expected to change); False for answer keys/sidecars, whose "
            "checksums are verified at ingest."
        ),
    ]


class PacketEntry(BaseModel):
    """One exported conversation in an HTML annotation packet."""

    task_id: Annotated[str, Field(description="Task id of the exported sim.")]
    sim_id: Annotated[str, Field(description="Simulation id (the dedupe key).")]
    trial: Annotated[Optional[int], Field(description="Trial index, if any.")] = None
    dir_name: Annotated[str, Field(description="Packet-relative page directory name.")]
    experiment: Annotated[
        str, Field(description="Human label of the source run (its dir name).")
    ] = ""
    domain: Annotated[str, Field(description="Domain of the source run.")] = ""
    has_audio: Annotated[
        bool, Field(description="Whether audio.wav was copied into the page dir.")
    ] = False


class AudioQualityEntry(BaseModel):
    """Provenance mapping for one provider-hidden agent-only audio clip."""

    clip_id: Annotated[str, Field(description="Blind clip id shown to the rater.")]
    sim_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Source task id.")]
    trial: Annotated[int, Field(description="Source trial index.")]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    domain: Annotated[str, Field(description="Source domain.")]
    provider: Annotated[str, Field(description="Source audio-native provider.")]
    agent_model: Annotated[str, Field(description="Source agent model.")]
    reasoning_effort: Annotated[
        Optional[str], Field(description="Source arm's reasoning effort.")
    ] = None
    results_path: Annotated[str, Field(description="Source results artifact path.")]
    audio_file: Annotated[str, Field(description="Packet-relative agent-only WAV.")]


class NativenessAgentTurn(BaseModel):
    """One stable, exactly rendered agent turn and its local customer context."""

    index: Annotated[int, Field(ge=0, description="Zero-based judge turn index.")]
    turn_id: Annotated[str, Field(description="Stable packet turn identifier.")]
    text: Annotated[str, Field(description="Exact displayed agent text.")]
    preceding_customer_text: Annotated[
        Optional[str], Field(description="Immediately preceding customer text.")
    ] = None
    interrupted: Annotated[
        bool,
        Field(
            description="A caller barge-in cut this turn short (chunk signal "
            "OR tick-overlap detector), so the displayed text is an estimate "
            "that can stop mid-word relative to the audible audio. Rendered "
            "as a truncation badge so annotators do not file the boundary "
            "mismatch as a delivery defect."
        ),
    ] = False


class RubricPacketEntry(BaseModel):
    """Provenance mapping for one blind combined-rubric annotation item."""

    clip_id: Annotated[str, Field(description="Blind call id shown to the rater.")]
    sim_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Source task id.")]
    trial: Annotated[int, Field(description="Source trial index.")]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    domain: Annotated[str, Field(description="Source domain.")]
    provider: Annotated[str, Field(description="Source audio-native provider.")]
    agent_model: Annotated[str, Field(description="Source agent model.")]
    reasoning_effort: Annotated[
        Optional[str], Field(description="Source arm's reasoning effort.")
    ] = None
    results_path: Annotated[str, Field(description="Source results artifact path.")]
    full_audio_file: Annotated[
        str, Field(description="Packet-relative full-conversation WAV.")
    ]
    agent_audio_file: Annotated[
        str, Field(description="Packet-relative agent-only WAV.")
    ]
    agent_gender: Annotated[
        Optional[str], Field(description="Known agent voice gender, if available.")
    ] = None
    caller_gender: Annotated[
        Optional[str], Field(description="Known caller gender, if available.")
    ] = None
    agent_turns: Annotated[
        list[NativenessAgentTurn],
        Field(
            default_factory=list,
            description="Stable agent turns rendered for nativeness selection.",
        ),
    ]


class CalibrationPacketEntry(BaseModel):
    """One blind call in a judge-calibration annotation packet.

    DELIBERATELY JUDGE-BLIND: this entry ships to raters inside the packet, so
    it carries no provider/model/arm fields, no results paths (run directory
    names encode arms), no judge verdicts, no rewards, and nothing about why
    the call was sampled. The clip→source join (arm, results path, sampling
    coverage) lives ONLY in the builder's coverage sidecar, which is written
    OUTSIDE the packet directory and never distributed.
    """

    clip_id: Annotated[str, Field(description="Blind call id shown to the rater.")]
    sim_id: Annotated[str, Field(description="Source simulation id (join key).")]
    task_id: Annotated[str, Field(description="Source task id (join key).")]
    trial: Annotated[int, Field(description="Source trial index.")] = 0
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    domain: Annotated[str, Field(description="Source domain.")]
    full_audio_file: Annotated[
        str, Field(description="Packet-relative full-conversation WAV.")
    ]
    agent_audio_file: Annotated[
        str, Field(description="Packet-relative agent-only WAV.")
    ]
    agent_gender: Annotated[
        Optional[str], Field(description="Known agent voice gender, if available.")
    ] = None
    caller_gender: Annotated[
        Optional[str], Field(description="Known caller gender, if available.")
    ] = None
    agent_turns: Annotated[
        list[NativenessAgentTurn],
        Field(
            default_factory=list,
            description="Stable agent turns rendered for nativeness selection.",
        ),
    ]


class NativenessAdjudicationEntry(BaseModel):
    """One judge candidate rendered in a call- or utterance-level packet."""

    candidate_id: Annotated[str, Field(description="Content-derived candidate id.")]
    clip_id: Annotated[str, Field(description="Blind source-call id.")]
    sim_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Source task id.")]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    factor_id: Annotated[str, Field(description="Central nativeness factor id.")]
    evaluation_level: Annotated[
        Literal["call", "utterance"], Field(description="Candidate unit level.")
    ]
    agent_turn: Annotated[
        Optional[NativenessAgentTurn],
        Field(description="Flagged agent turn; None for call-level candidates."),
    ] = None
    judge_reasoning: Annotated[str, Field(description="Judge rationale for review.")]
    judge_quote: Annotated[str, Field(description="Judge's exact claimed quote.")]
    judge_severity: Annotated[int, Field(ge=1, le=4)]


class CalibrationDecisionEntry(BaseModel):
    """One judge-positive candidate offered for decision on the calibration
    adjudication page.

    Owner-facing only — the adjudication packet never ships to raters, so this
    entry may (and does) carry judge evidence. Utterance-level candidates pin
    the exact flagged agent turn; call-level candidates carry no turn. The
    browser CSV row this candidate round-trips through is
    ``tau2.annotation.models.CalibrationDecisionRow``.
    """

    candidate_id: Annotated[str, Field(description="Content-derived candidate id.")]
    clip_id: Annotated[str, Field(description="Blind source-call id.")]
    sim_id: Annotated[str, Field(description="Source simulation id.")]
    task_id: Annotated[str, Field(description="Source task id.")]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    judge: Annotated[
        str, Field(description="Judge family (nativeness/delivery/quality/semantic).")
    ]
    factor_id: Annotated[str, Field(description="Factor / axis / dimension id.")]
    evaluation_level: Annotated[
        Literal["call", "utterance"], Field(description="Candidate unit level.")
    ]
    judge_verdict: Annotated[
        str, Field(description="Stored verdict being decided (always 'fail').")
    ]
    judge_evidence: Annotated[str, Field(description="Judge rationale for review.")] = (
        ""
    )
    judge_quote: Annotated[str, Field(description="Judge's exact claimed span.")] = ""
    agent_turn: Annotated[
        Optional[NativenessAgentTurn],
        Field(description="Flagged agent turn; None for call-level candidates."),
    ] = None


class ArtifactManifest(BaseModel):
    """Provenance for one exported sheet family."""

    schema_version: Literal[1] = ARTIFACT_SCHEMA_VERSION
    kind: Annotated[ArtifactKind, Field(description="Which sheet family this is.")]
    batch_id: Annotated[
        str,
        Field(
            description="Content-derived id: sha256(kind, batch_name, file "
            "hashes) truncated to 12 hex chars — identical inputs reproduce it."
        ),
    ]
    batch_name: Annotated[str, Field(description="Human name (the export stem).")]
    created_at: Annotated[
        str, Field(description="Export wall-clock time (ISO 8601, UTC).")
    ]
    git_sha: Annotated[str, Field(description="Repo HEAD at export time.")]
    language: Annotated[
        Optional[str], Field(description="ISO 639-1 language, when single-language.")
    ] = None
    domain: Annotated[Optional[str], Field(description="Domain, when applicable.")] = (
        None
    )
    form: Annotated[
        Optional[str],
        Field(description="Packet form type (packets only; None for sheets)."),
    ] = None
    provenance: Annotated[
        dict,
        Field(
            description="Judge/pipeline configuration that produced the judged "
            "rows (models, prompt hashes, source paths, knobs)."
        ),
    ]
    files: Annotated[list[ManifestFile], Field(description="Family members.")]
    entries: Annotated[
        Optional[list[PacketEntry]],
        Field(
            description="Per-conversation entries (HTML packets only; None for "
            "sheet families). --append dedupes new sims by entry sim_id."
        ),
    ] = None
    audio_quality_entries: Annotated[
        Optional[list[AudioQualityEntry]],
        Field(
            description="Blind clip to source-call mapping for an audio-quality "
            "packet; None for other artifact kinds."
        ),
    ] = None
    rubric_entries: Annotated[
        Optional[list[RubricPacketEntry]],
        Field(
            description="Blind call to source mapping for an interaction or "
            "nativeness rubric packet; None for other artifact kinds."
        ),
    ] = None
    nativeness_adjudication_entries: Annotated[
        Optional[list[NativenessAdjudicationEntry]],
        Field(description="Candidate records in a nativeness adjudication packet."),
    ] = None
    calibration_entries: Annotated[
        Optional[list[CalibrationPacketEntry]],
        Field(
            description="Blind call entries of a judge-calibration packet — "
            "judge-blind by construction (see CalibrationPacketEntry); None "
            "for other artifact kinds."
        ),
    ] = None
    calibration_decision_entries: Annotated[
        Optional[list[CalibrationDecisionEntry]],
        Field(
            description="Judge-positive decision candidates of a calibration "
            "adjudication packet (owner-facing); None for other kinds."
        ),
    ] = None


class SheetPayload(BaseModel):
    """One family member ready to serialize: headers + header-keyed cell rows."""

    headers: list[str]
    rows: list[dict[str, str]]
    editable: Annotated[
        bool, Field(description="Annotator-facing (True) vs answer key (False).")
    ] = True


def payload_from_rows(
    model: type[SheetRow], rows: Sequence[SheetRow], *, editable: bool = True
) -> SheetPayload:
    """Render typed rows into a payload; headers come from the model."""
    return SheetPayload(
        headers=model.headers(),
        rows=[row.to_cells() for row in rows],
        editable=editable,
    )


def payload_from_cold_sheet(sheet: ColdSheet) -> SheetPayload:
    return SheetPayload(headers=sheet.headers(), rows=sheet.to_cells(), editable=True)


def git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def provenance_stamp() -> dict:
    """The write-time provenance every JSON artifact carries."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
    }


def write_json_artifact(path: Path, payload: Union[BaseModel, dict, list]) -> Path:
    """Atomically write a JSON artifact (tmp staging + rename).

    The JSON counterpart of ``write_sheet_family`` for single-file artifacts
    (agreement records, nuance candidates, audit bodies, fit reports): the
    content is staged as ``<name>.tmp`` and renamed into place, so a crashed
    export never leaves a truncated artifact behind. Callers stamp provenance
    into the payload themselves (``provenance_stamp`` for the write-time
    fields; models carry their own).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=2) + "\n"
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _render_csv(payload: SheetPayload) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=payload.headers)
    writer.writeheader()
    for row in payload.rows:
        # Cells (not headers — those are fixed model labels) are armored
        # against spreadsheet formula injection; ingest strips the armor.
        writer.writerow({h: armor_cell(row.get(h, "")) for h in payload.headers})
    return buf.getvalue().encode(CSV_ENCODING)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def role_filename(stem: str, role: str) -> str:
    return f"{stem}.csv" if role == MAIN_ROLE else f"{stem}_{role}.csv"


def manifest_path_for(out_stem: Path) -> Path:
    return out_stem.parent / f"{out_stem.name}.manifest.json"


def derive_batch_id(kind: str, batch_name: str, file_hashes: dict[str, str]) -> str:
    """Content-derived batch id (NOT uuid4 — identical inputs reproduce it)."""
    h = hashlib.sha256()
    h.update(kind.encode())
    h.update(b"\0")
    h.update(batch_name.encode())
    for role, sha in sorted(file_hashes.items()):
        h.update(b"\0")
        h.update(role.encode())
        h.update(b"\0")
        h.update(sha.encode())
    return h.hexdigest()[:12]


def write_sheet_family(
    out_stem: Path,
    *,
    kind: ArtifactKind,
    payloads: dict[str, SheetPayload],
    provenance: dict,
    language: Optional[str] = None,
    domain: Optional[str] = None,
    batch_name: Optional[str] = None,
    entries: Optional[list[PacketEntry]] = None,
) -> Path:
    """Write every family CSV plus the manifest, atomically; returns the manifest.

    All content is rendered and staged as ``*.tmp`` first; only when every
    member staged successfully are the finals renamed into place, manifest
    last — so a family on disk always has complete members and provenance.
    """
    if not payloads:
        raise ValueError("write_sheet_family needs at least one payload")
    registered = SHEET_FAMILIES[kind]
    for role in payloads:
        if role not in registered:
            raise ValueError(f"role '{role}' is not part of the '{kind}' family")
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    batch_name = batch_name or out_stem.name

    rendered: dict[str, bytes] = {
        role: _render_csv(payload) for role, payload in payloads.items()
    }
    hashes = {role: _sha256(data) for role, data in rendered.items()}
    manifest = ArtifactManifest(
        kind=kind,
        batch_id=derive_batch_id(kind, batch_name, hashes),
        batch_name=batch_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        language=language,
        domain=domain,
        provenance=provenance,
        files=[
            ManifestFile(
                name=role_filename(out_stem.name, role),
                sha256=hashes[role],
                role=role,
                editable=payloads[role].editable,
            )
            for role in payloads
        ],
        entries=entries,
    )

    staged: list[tuple[Path, Path]] = []
    manifest_final = manifest_path_for(out_stem)
    manifest_tmp = manifest_final.with_suffix(".tmp")
    try:
        for role, data in rendered.items():
            final = out_stem.parent / role_filename(out_stem.name, role)
            tmp = final.with_suffix(final.suffix + ".tmp")
            tmp.write_bytes(data)
            staged.append((tmp, final))
        manifest_tmp.write_text(manifest.model_dump_json(indent=2) + "\n")
        for tmp, final in staged:
            tmp.replace(final)
        manifest_tmp.replace(manifest_final)  # manifest lands LAST
    except BaseException:
        # A failed export must not leave *.tmp staging litter behind.
        for tmp in [*(tmp for tmp, _ in staged), manifest_tmp]:
            tmp.unlink(missing_ok=True)
        raise
    for role, payload in payloads.items():
        logger.info(
            f"wrote {len(payload.rows)} rows -> "
            f"{out_stem.parent / role_filename(out_stem.name, role)}"
        )
    logger.info(f"manifest ({manifest.batch_id}) -> {manifest_final}")
    return manifest_final


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class LoadedArtifact(BaseModel):
    """A verified sheet family read back from disk, with typed row access."""

    manifest: ArtifactManifest
    directory: Path
    manifest_path: Annotated[
        Optional[Path],
        Field(
            description="Where the manifest was loaded from — lets a caller "
            "read a sibling CSV (e.g. a second annotator's renamed copy) "
            "against the SAME family without re-discovery."
        ),
    ] = None
    role: Annotated[
        Optional[str],
        Field(
            description="The family role of the CSV that was passed to ingest "
            "(None when the manifest itself was passed)."
        ),
    ] = None
    overrides: Annotated[
        dict[str, Path],
        Field(
            default_factory=dict,
            description="Role -> path overrides for filled CSVs that were "
            "renamed/moved away from their exported location.",
        ),
    ]

    def _file(self, role: str) -> ManifestFile:
        for f in self.manifest.files:
            if f.role == role:
                return f
        raise KeyError(
            f"artifact '{self.manifest.batch_name}' ({self.manifest.kind}) has "
            f"no '{role}' member"
        )

    def path_for(self, role: str) -> Path:
        return self.overrides.get(role) or self.directory / self._file(role).name

    def raw_rows(self, role: str) -> list[dict[str, str]]:
        with open(self.path_for(role), newline="", encoding=CSV_ENCODING) as fp:
            return [
                {
                    h: dearmor_cell(v) if isinstance(v, str) else v
                    for h, v in raw.items()
                }
                for raw in csv.DictReader(fp)
            ]

    def rows(self, role: str = MAIN_ROLE) -> list[SheetRow]:
        """Typed rows for a fixed-shape role (use ``cold_sheet`` for the grid)."""
        model = SHEET_FAMILIES[self.manifest.kind][role]
        if model == COLD_WIDE:
            raise ValueError("wide cold sheets load via cold_sheet(), not rows()")
        return [model.from_cells(raw) for raw in self.raw_rows(role)]

    def cold_sheet(self) -> ColdSheet:
        """The wide cold grid + its factor key, unmelted into a ColdSheet."""
        factors = [FactorKeyRow.from_cells(raw) for raw in self.raw_rows("factors_key")]
        return ColdSheet.from_cells(self.raw_rows(MAIN_ROLE), factors)


def _find_manifest(csv_path: Path) -> tuple[Path, str]:
    """Locate the manifest for a filled CSV by EXACT filename match.

    A manifest in the CSV's directory must list the CSV as a family member —
    no guessing (a lone manifest must not claim unrelated CSVs dropped in the
    same directory). A renamed/moved filled CSV needs an explicit --manifest.
    """
    for candidate in sorted(csv_path.parent.glob("*.manifest.json")):
        manifest = ArtifactManifest.model_validate_json(candidate.read_text())
        for f in manifest.files:
            if f.name == csv_path.name:
                return candidate, f.role
    raise FileNotFoundError(
        f"No manifest lists {csv_path.name} — an annotation artifact cannot "
        "be ingested without its .manifest.json (pass --manifest if the "
        "filled CSV was renamed or moved)."
    )


def _resolve_role_by_headers(
    manifest: ArtifactManifest, headers: set[str]
) -> Optional[str]:
    """Which editable role's registered header set matches the CSV's."""
    for f in manifest.files:
        if not f.editable:
            continue
        model = SHEET_FAMILIES[manifest.kind][f.role]
        if model == COLD_WIDE:
            if COLD_META_LABELS <= headers:
                return f.role
        elif set(model.headers()) == headers:
            return f.role
    return None


def _validate_headers(artifact: LoadedArtifact, role: str, headers: set[str]) -> None:
    """The edited CSV must still carry the registered model's exact header set
    (wide cold grids: meta headers + only known factor columns)."""
    kind = artifact.manifest.kind
    model = SHEET_FAMILIES[kind][role]
    if model == COLD_WIDE:
        missing = COLD_META_LABELS - headers
        if missing:
            raise ValueError(
                f"filled cold sheet is missing required columns {sorted(missing)}"
            )
        factors = [
            FactorKeyRow.from_cells(raw) for raw in artifact.raw_rows("factors_key")
        ]
        known = {f.nuance for f in factors} | {f.factor_id for f in factors}
        unknown = headers - COLD_META_LABELS - known
        if unknown:
            raise ValueError(
                f"filled cold sheet has factor columns not in the factors key: "
                f"{sorted(unknown)}"
            )
        # A deleted factor column silently drops that factor from calibration.
        expected = {f.nuance or f.factor_id for f in factors}
        missing_factors = expected - headers
        if missing_factors:
            raise ValueError(
                f"filled cold sheet is missing factor columns "
                f"{sorted(missing_factors)} — every factors_key column must "
                "survive the annotation round trip"
            )
        return
    expected = set(model.headers())
    if headers != expected:
        missing, extra = expected - headers, headers - expected
        raise ValueError(
            f"filled '{role}' sheet header set does not match "
            f"{model.__name__}: missing {sorted(missing)}, extra {sorted(extra)}"
        )


def read_artifact(path: Path, manifest_path: Optional[Path] = None) -> LoadedArtifact:
    """Load + verify a sheet family from a filled CSV (or the manifest itself).

    - Hard-fails without a manifest.
    - Verifies sha256 on every present ``editable=False`` member (the answer
      key must be exactly as exported).
    - Validates the filled CSV's header set against the registered row model.
    """
    path = Path(path)
    role: Optional[str] = None
    if path.name.endswith(".manifest.json"):
        manifest_file = path
    elif manifest_path is not None:
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_file}")
    else:
        manifest_file, role = _find_manifest(path)
    manifest = ArtifactManifest.model_validate_json(manifest_file.read_text())
    directory = manifest_file.parent
    artifact = LoadedArtifact(
        manifest=manifest, directory=directory, manifest_path=manifest_file, role=role
    )

    # Non-editable members must be present and byte-identical to the export —
    # a missing answer key/sidecar is a provenance error, not a skip.
    for f in manifest.files:
        if f.editable:
            continue
        member = directory / f.name
        if not member.exists():
            raise ValueError(
                f"non-editable member {f.name} of artifact "
                f"'{manifest.batch_name}' is missing from {directory} — the "
                "family cannot be verified without it"
            )
        actual = _sha256(member.read_bytes())
        if actual != f.sha256:
            raise ValueError(
                f"checksum mismatch on non-editable member {f.name}: the "
                f"answer key was modified after export (expected {f.sha256}, "
                f"got {actual})"
            )

    if not path.name.endswith(".manifest.json"):
        with open(path, newline="", encoding=CSV_ENCODING) as fp:
            headers = set(csv.DictReader(fp).fieldnames or [])
        if artifact.role is None:
            artifact.role = _resolve_role_by_headers(manifest, headers)
            if artifact.role is None:
                raise ValueError(
                    f"{path.name} matches no editable sheet of the "
                    f"'{manifest.kind}' family (header mismatch)"
                )
        _validate_headers(artifact, artifact.role, headers)
        # Serve the (possibly renamed/moved) filled CSV for its role.
        artifact.overrides[artifact.role] = path
    return artifact

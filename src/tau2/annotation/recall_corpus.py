# Copyright Sierra
"""Materialize a judge-blind random cohort as a standalone subset corpus.

The recall arm of judge calibration needs the judges and the annotation
packet to run over EXACTLY a seeded uniform sample of stored calls — never
over whatever the judges already flagged (an enriched cohort can only
measure recall where some judge already looked). ``build_recall_corpus``
draws that cohort (``CalibrationDrawConfig(rule="random")``) and
materializes it as ordinary dir-format results — per source cell: a subset
``results.json`` + ``simulations/`` copies + ``artifacts/``/``tasks/``
symlinks to the source audio — so every judge verb and the calibration
packet builder consume it unchanged, and fresh verdicts land on the subset
copies, never on the source corpus. The sampling frame lands inside the
corpus dir for provenance; like every frame it is internal-only and never
ships with a packet.
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from tau2.annotation.artifacts import write_json_artifact
from tau2.annotation.calibration_draw import (
    CalibrationDrawConfig,
    CalibrationFrame,
    FrameCall,
    build_calibration_frame,
)
from tau2.annotation.loading import iter_loaded_sims
from tau2.config import (
    DEFAULT_CALIBRATION_CALLS_PER_LANGUAGE,
    DEFAULT_CALIBRATION_SEED,
)
from tau2.data_model.simulation import Results, SimulationRun
from tau2.judges.delivery.disk_audio import find_both_wav

RECALL_FRAME_NAME = "recall_frame.json"


class RecallCorpusOptions(BaseModel):
    """Everything one recall-corpus materialization depends on."""

    results: Annotated[
        list[Path],
        Field(min_length=1, description="Source results dirs / results.json files."),
    ]
    language: Annotated[str, Field(description="ISO 639-1 language code.")]
    n_calls: Annotated[
        int, Field(gt=0, description="Cohort size (seeded uniform draw).")
    ] = DEFAULT_CALIBRATION_CALLS_PER_LANGUAGE
    seed: Annotated[int, Field(description="Draw seed.")] = DEFAULT_CALIBRATION_SEED
    out_dir: Annotated[
        Path,
        Field(
            description="Corpus root to create; one subdir per source cell "
            "(its experiment label). Refused when it already exists — a "
            "materialized cohort is never silently rebuilt over."
        ),
    ]

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        return value.strip().lower()

    def draw_config(self) -> CalibrationDrawConfig:
        return CalibrationDrawConfig(
            rule="random",
            language=self.language,
            n_calls=self.n_calls,
            seed=self.seed,
        )


class RecallCorpusBuild(BaseModel):
    """What one materialization produced."""

    corpus_dir: Path
    frame_path: Path
    n_calls: Annotated[int, Field(ge=0, description="Cohort calls materialized.")]
    cells: Annotated[
        dict[str, int],
        Field(description="Materialized calls per source cell (experiment label)."),
    ]


def _source_sim_dir(results_dir: Path, sim: SimulationRun) -> Path:
    """The source ``sim_<id>`` artifact dir holding the call's audio."""
    wav = find_both_wav(results_dir, sim)
    if wav is None:
        raise FileNotFoundError(
            f"sim {sim.id}: no both.wav under {results_dir} — the draw "
            "requires audio, so a drawn call losing its wav means the source "
            "corpus changed under us"
        )
    sim_dir = wav.parent if wav.parent.name == f"sim_{sim.id}" else wav.parent.parent
    if sim_dir.name != f"sim_{sim.id}":
        raise FileNotFoundError(
            f"sim {sim.id}: unrecognized audio layout at {wav} — expected the "
            "wav inside a sim_<id> dir (optionally under audio/)"
        )
    return sim_dir


def _materialize_cell(
    results_path: str, cohort: list[FrameCall], out_dir: Path
) -> tuple[str, int]:
    """One source cell's subset: results.json + sim copies + audio links."""
    source = Path(results_path)
    results_dir = source.parent
    wanted = {call.sim_id for call in cohort}
    experiment = cohort[0].experiment
    results = Results.load(source)
    subset = [sim for sim in results.simulations if sim.id in wanted]
    missing = wanted - {sim.id for sim in subset}
    if missing:
        raise ValueError(
            f"drawn sims missing from {source}: {sorted(missing)[:5]} — the "
            "source corpus changed between the draw and the materialization"
        )
    results.simulations = subset
    cell_out = out_dir / experiment
    # A fresh dir-format target must exist as a directory before save()
    # resolves it (otherwise it is treated as a metadata FILE path and the
    # simulations/ dir lands next to it in the parent).
    cell_out.mkdir(parents=True, exist_ok=True)
    results.save(cell_out, format="dir")
    for sim in subset:
        src_sim_dir = _source_sim_dir(results_dir, sim)
        rel = src_sim_dir.relative_to(results_dir)
        dest = cell_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(src_sim_dir.resolve(), target_is_directory=True)
    return experiment, len(subset)


def build_recall_corpus(options: RecallCorpusOptions) -> RecallCorpusBuild:
    """Draw the judge-blind cohort and materialize it as a subset corpus.

    Deterministic: same inputs + seed + n_calls → the same cohort. The frame
    (whole eligibility scan, drop counters, descriptive strata coverage) is
    written inside the corpus dir as its provenance record.
    """
    if options.out_dir.exists() and any(options.out_dir.iterdir()):
        raise FileExistsError(
            f"recall corpus already exists: {options.out_dir} — delete it to "
            "re-materialize (judged subsets are never silently rebuilt over)"
        )
    config = options.draw_config()
    frame: CalibrationFrame = build_calibration_frame(
        iter_loaded_sims(options.results),
        config,
        results=options.results,
    )
    cohort = frame.drawn_calls()
    if not cohort:
        raise ValueError(
            "the draw selected no calls — the inputs hold no eligible calls "
            f"for language '{config.language}' (see the frame drop counters)"
        )
    by_cell: dict[str, list[FrameCall]] = {}
    for call in cohort:
        by_cell.setdefault(call.results_path, []).append(call)
    options.out_dir.mkdir(parents=True, exist_ok=True)
    cells: dict[str, int] = {}
    for results_path in sorted(by_cell):
        experiment, count = _materialize_cell(
            results_path, by_cell[results_path], options.out_dir
        )
        cells[experiment] = count
    frame_path = write_json_artifact(options.out_dir / RECALL_FRAME_NAME, frame)
    logger.info(
        f"recall corpus ({config.language}, n={len(cohort)}, seed "
        f"{config.seed}) -> {options.out_dir} "
        f"[{', '.join(f'{k}={v}' for k, v in sorted(cells.items()))}]"
    )
    return RecallCorpusBuild(
        corpus_dir=options.out_dir,
        frame_path=frame_path,
        n_calls=len(cohort),
        cells=cells,
    )

"""Package the frozen 2026-10-02 Voice V2 release exports as 2.0 packages.

Maintainer-only. The release calls predate the current matrix layout and some
recorded condition fields, so ``tau2 submit prepare <matrix-root>`` cannot read
them. This adapter reads the saved exports (``trajectories/<lang>/<domain>/``)
and their saved judge verdicts as raw JSON, checks them against the published
frame, and writes the same package a matrix submission produces. It never runs
a simulation or a judge, and it must never qualify an incomplete new run.

The same export holds the release's banking and entity (intake) calls, which
ran before the ``voice-v2-banking`` and ``voice-v2-entity`` manifests existed
and so record no ``info.benchmark``. For those frames the adapter derives each
cell's binding from this checkout's released cell and checks the recorded pins
against it (see ``_check_recorded_binding``).
"""

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from tau2.data_model.benchmark import (
    BenchmarkFrame,
    BenchmarkSubmission,
    CellFrame,
    TaskTrial,
    TrialStatus,
)
from tau2.data_model.judges import (
    PlanAction,
    PlannedVerdict,
    StoredVerdict,
    SuitePlan,
)
from tau2.data_model.simulation import SimulationRun
from tau2.judges.submission import JUDGED_BENCHMARKS, export_diagnostics
from tau2.metrics.benchmark import summarize
from tau2.runner.pass3 import CROSSED_CONDITIONS, STABLE_INFO_FIELDS
from tau2.runner.presets import binding_revision, load_cell_language, load_frame
from tau2.scripts.leaderboard.benchmark_package import (
    HISTORICAL_EXPORT_DIRS,
    HISTORICAL_PASS3_DIRS,
    BenchmarkPackage,
    PackageMetadata,
    SubmitterMetadata,
    load_package,
)
from tau2.scripts.leaderboard.benchmark_pass3 import (
    PASS3_BENCHMARKS,
    BenchmarkPass3,
    Pass3Task,
    summarize_cell,
    summarize_pass3,
)
from tau2.scripts.leaderboard.benchmark_quality import (
    QUALITY_BENCHMARKS,
    BenchmarkQuality,
    build_quality,
    cell_quality,
    score_call,
)

if TYPE_CHECKING:
    from tau2.multilingual.runtime import LoadedLanguage

EVIDENCE = "historical-voice-v2-2026-10-02"
CORRECTION_FORMAT = "voice-v2-gemini-retry-correction-v1"
# Published presets the release export holds: the 18 core cells, banking, entity.
HISTORICAL_BENCHMARKS = ("voice", "banking", "entity")

# Recorded ``runtime_rules_sha256`` digests of the release's non-English banking
# and intake calls. The digest hashes Python sources, and these differ from
# v2.0.0 only by diffs verified as behavior-neutral for these runs:
# banking_knowledge/localization_bindings.py (error-message wording),
# multilingual/agent_guidance.py (an intake-only constant, so neutral for
# banking), data_model/voice.py (new, unused Voice Design models),
# intake/cli.py (an argument-group label) and multilingual/runtime.py (a helper
# refactor, plus a dropped ``config.benchmark is None`` condition that held for
# every one of these runs). Every other pin must equal the released cell.
ACCEPTED_RUNTIME_RULES = {
    "banking_knowledge": frozenset(
        {
            # Release-run commit 05dc65d2.
            "63d28cf8966e0339f629abee87adf664da57acf0145e754d1faaf319586ddef7",
            # Release-run commit 9aef5bb0.
            "1958ecab6b2f51eed5d143616a420a9d773df17454c0f3da371f315b7927036b",
        }
    ),
    # Release-run commits 9aef5bb0, 92b344b3 and af13daae.
    "intake": frozenset(
        {"357ad3469bb43011116827ade7cb1bcaedbabd915d6d68e0e1e7b53116405f25"}
    ),
}
# Recorded run fields that vary by cell by construction: provenance, the
# language binding (checked against the release separately) and the
# language-resolved elicitation contract versions.
_CELL_FIELDS = frozenset(
    {
        "benchmark",
        "git_commit",
        "language",
        "pack_extension",
        "language_tasks",
        "language_artifacts",
        "elicitation_scoring_version",
        "elicitation_prompt_version",
        "elicitation_catalog_version",
        "elicitation_spell_detection_version",
    }
)


def file_sha256(path: Path) -> str:
    """Hash a saved input without modifying it."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_dir(root: Path, language: str, domain: str) -> Path:
    return root / "trajectories" / language / HISTORICAL_EXPORT_DIRS.get(domain, domain)


def _validate_simulation_files(
    root: Path,
    simulations: list[dict],
    language: str,
    domain: str,
    directory: Path | None = None,
) -> None:
    """Check the directory payloads against the canonical result index."""
    sim_dir = (directory or _cell_dir(root, language, domain)) / "simulations"
    indexed_ids = {entry["id"] for entry in simulations}
    file_ids = {path.stem for path in sim_dir.glob("*.json")}
    if indexed_ids != file_ids:
        raise ValueError(f"{language}/{domain}: simulation files differ from index")
    for entry in simulations:
        payload = json.loads((sim_dir / f"{entry['id']}.json").read_text())
        if any(
            payload.get(field) != entry[field]
            for field in ("id", "task_id", "trial", "termination_reason")
        ):
            raise ValueError(
                f"{language}/{domain}: simulation/index identity mismatch: {entry['id']}"
            )
        reward_info = payload.get("reward_info") or {}
        if reward_info.get("reward") != entry["reward"]:
            raise ValueError(
                f"{language}/{domain}: simulation/index reward mismatch: {entry['id']}"
            )


def _validate_cell_frame(
    data: dict,
    simulations: list[dict],
    language: str,
    domain: str,
    official: BenchmarkFrame,
) -> dict[str, str]:
    """Check release metadata against the frame without inventing later fields."""
    info = data["info"]
    frame = info.get("benchmark") or {}
    cell = next(
        item
        for item in official.cells
        if (item.language, item.domain) == (language, domain)
    )
    if frame.get("benchmark_version") != official.benchmark_version:
        raise ValueError(f"{language}/{domain}: trajectory is not voice-v2")
    if frame.get("frame_revision") != official.revision:
        raise ValueError(f"{language}/{domain}: frame revision mismatch")
    if frame.get("binding_revision") != cell.binding_revision:
        raise ValueError(f"{language}/{domain}: frozen binding mismatch")
    if frame.get("canonical_task_ids") != cell.canonical_task_ids:
        raise ValueError(f"{language}/{domain}: canonical task frame mismatch")
    if frame.get("language") != language or frame.get("domain") != domain:
        raise ValueError(f"{language}/{domain}: benchmark frame identity mismatch")
    if info.get("language") != language:
        raise ValueError(f"{language}/{domain}: recorded language mismatch")
    if info["environment_info"]["domain_name"] != domain:
        raise ValueError(f"{language}/{domain}: recorded domain mismatch")
    selected = set(frame["selected_task_ids"])
    mapped = {
        str(task_id): details["canonical_task_id"]
        for task_id, details in info["language_tasks"].items()
    }
    task_ids = [mapped[str(task["id"])] for task in data["tasks"]]
    if (
        frame["selected_task_ids"] != cell.canonical_task_ids
        or len(task_ids) != len(cell.canonical_task_ids)
        or len(set(task_ids)) != len(task_ids)
        or set(task_ids) != selected
    ):
        raise ValueError(f"{language}/{domain}: selected 50-task frame mismatch")
    if frame["trials"] != official.trials or info["num_trials"] != 1:
        raise ValueError(f"{language}/{domain}: V2 publication requires one trial")
    _check_calls(simulations, mapped, selected, f"{language}/{domain}")
    return mapped


def _check_calls(
    simulations: list[dict], mapped: dict[str, str], selected: set[str], label: str
) -> None:
    """One trial-0 call per selected task; unscored calls are infrastructure errors."""
    expected = {(task_id, 0) for task_id in selected}
    actual = [(mapped[str(sim["task_id"])], sim["trial"]) for sim in simulations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"{label}: missing or duplicate task/trial keys")
    if any(
        sim["reward"] is None and sim["termination_reason"] != "infrastructure_error"
        for sim in simulations
    ):
        raise ValueError(f"{label}: unscored call lacks infrastructure status")


def _check_recorded_binding(info: dict, loaded: "LoadedLanguage", label: str) -> None:
    """The recorded language binding must be the released cell's, pin for pin.

    Checks ``info.language_artifacts`` and every ``info.language_tasks`` entry
    against the cell this checkout releases: pack and domain revisions, every
    ``LanguageArtifactPins`` field and the task-loader options. The one
    exception is ``runtime_rules_sha256``, which may instead be a verified
    digest in ``ACCEPTED_RUNTIME_RULES``.
    """
    pins = loaded.artifact_pins.model_dump(mode="json")
    released_rules = pins.pop("runtime_rules_sha256")
    binding = {
        "language": loaded.pack.language,
        "mode": loaded.mode,
        "pack_extension": loaded.pack_extension,
        "pack_revision": loaded.pack.revision,
        "domain_revision": loaded.binding.revision,
        **pins,
    }
    records = [(info["language_artifacts"], pins)] + [
        (entry, binding) for entry in info["language_tasks"].values()
    ]
    for recorded, expected in records:
        changed = sorted(
            field for field, value in expected.items() if recorded.get(field) != value
        )
        if changed:
            raise ValueError(
                f"{label}: recorded binding differs from the release: "
                + ", ".join(changed)
            )
        rules = recorded.get("runtime_rules_sha256")
        if rules != released_rules and (
            released_rules is None
            or rules not in ACCEPTED_RUNTIME_RULES.get(loaded.domain, ())
        ):
            raise ValueError(f"{label}: unverified runtime_rules_sha256 {rules}")
    if any(
        entry["task_selection"]["task_kwargs"] != loaded.selection.task_kwargs
        for entry in info["language_tasks"].values()
    ):
        raise ValueError(f"{label}: recorded task-loader options differ")


def _recorded_conditions(info: dict, language: str, label: str) -> dict:
    """Recorded run settings shared by every cell; per-cell fields are removed."""
    conditions = {
        field: value for field, value in info.items() if field not in _CELL_FIELDS
    }
    audio = dict(conditions["audio_native_config"])
    if audio.get("language") != language:
        raise ValueError(f"{label}: provider language differs from the cell")
    audio["language"] = None
    conditions["audio_native_config"] = audio
    return conditions


def _derived_cell(
    data: dict, simulations: list[dict], cell: CellFrame, official: BenchmarkFrame
) -> tuple[dict[str, str], dict]:
    """Check a pre-manifest release cell; return its task map and conditions.

    Banking and entity calls record no ``info.benchmark``, so the frame identity
    is derived: the recorded binding must be the released one, the tasks the
    frozen subset (localized IDs map through ``canonical_task_id``) and each
    call trial 0 of the regular speech condition.
    """
    language, domain = cell.language, cell.domain
    label = f"{language}/{domain}"
    info = data["info"]
    if info.get("benchmark") is not None:
        raise ValueError(f"{label}: expected a pre-manifest release cell")
    if info.get("language") != language:
        raise ValueError(f"{label}: recorded language mismatch")
    if info["environment_info"]["domain_name"] != domain:
        raise ValueError(f"{label}: recorded domain mismatch")
    loaded = load_cell_language(language, domain, official.mode)
    if binding_revision(loaded) != cell.binding_revision:
        raise ValueError(f"{label}: this checkout differs from the published binding")
    _check_recorded_binding(info, loaded, label)
    mapped = {
        str(task_id): details["canonical_task_id"]
        for task_id, details in info["language_tasks"].items()
    }
    task_ids = [mapped[str(task["id"])] for task in data["tasks"]]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(
        cell.canonical_task_ids
    ):
        raise ValueError(f"{label}: task set differs from the frozen subset")
    if official.trials != [0] or info["num_trials"] != 1:
        raise ValueError(f"{label}: V2 publication requires one trial")
    if any(
        info.get(field) != "regular"
        for field in (
            "speech_complexity",
            "channel_effects_mode",
            "speech_effects_mode",
        )
    ):
        raise ValueError(f"{label}: only the regular speech condition is scored")
    _check_calls(simulations, mapped, set(cell.canonical_task_ids), label)
    return mapped, _recorded_conditions(info, language, label)


def _historical_pass3(
    root: Path, official: BenchmarkFrame, regular: dict[tuple[str, str], tuple]
) -> BenchmarkPass3:
    """Crossed Pass^3 from the export's three intake conditions per cell.

    ``regular`` maps each cell to its checked regular results, task mapping and
    results path. The other two conditions must be the catalog's (seed and
    effect modes), share every stable run setting and the task list with the
    regular run, and hold one trial-0 call per frame task, as
    ``tau2.runner.pass3.summarize_crossed_pass3`` requires.
    """
    cells = []
    for cell in official.cells:
        language, domain = cell.language, cell.domain
        regular_data, mapping, regular_path = regular[(language, domain)]
        outcomes: dict[str, list[bool | None]] = defaultdict(list)
        sources = {}
        for condition in CROSSED_CONDITIONS:
            label = f"{language}/{domain}/{condition.label}"
            if condition.label == "regular":
                data, path = regular_data, regular_path
            else:
                directory = (
                    root
                    / "trajectories"
                    / language
                    / HISTORICAL_PASS3_DIRS[condition.label]
                )
                path = directory / "results.json"
                data = json.loads(path.read_text())
                _validate_simulation_files(
                    root, data["simulation_index"], language, domain, directory
                )
            info = data["info"]
            if (
                info.get("seed") != condition.seed
                or info.get("voice_sampling_seed") is not None
                or info.get("num_trials") != 1
                or info.get("pass3_condition")
                not in (
                    (None, "regular")
                    if condition.label == "regular"
                    else (condition.label,)
                )
                or info.get("channel_effects_mode") != condition.channel_effects_mode
                or info.get("speech_effects_mode") != condition.speech_effects_mode
            ):
                raise ValueError(f"{label}: effects do not match the Pass^3 catalog")
            if (
                any(
                    info.get(field) != regular_data["info"].get(field)
                    for field in STABLE_INFO_FIELDS
                )
                or data["tasks"] != regular_data["tasks"]
            ):
                raise ValueError(f"{label}: run settings or tasks differ from regular")
            _check_calls(
                data["simulation_index"], mapping, set(cell.canonical_task_ids), label
            )
            for sim in data["simulation_index"]:
                outcomes[mapping[str(sim["task_id"])]].append(
                    None if sim["reward"] is None else sim["reward"] == 1.0
                )
            sources[condition.label] = file_sha256(path)
        tasks = [
            Pass3Task(canonical_task_id=task_id, passes=tuple(outcomes[task_id]))
            for task_id in cell.canonical_task_ids
        ]
        cells.append(summarize_cell(language, domain, sources, tasks))
    return summarize_pass3(official, cells)


def _condition_fingerprint(conditions: dict) -> str:
    """Deterministic fingerprint of recorded settings, tagged with the evidence."""
    payload = {"evidence": EVIDENCE, "recorded_conditions": conditions}
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_source_correction(root: Path) -> None:
    """A retry-correction manifest must still describe the corrected export."""
    path = root / "correction.json"
    if not path.is_file():
        return
    correction = json.loads(path.read_text())
    if correction.get("format") != CORRECTION_FORMAT:
        raise ValueError("unknown source correction format")
    for item in correction["corrections"]:
        cell = _cell_dir(root, item["language"], item["domain"])
        simulation = cell / "simulations" / f"{item['new_index']['id']}.json"
        if file_sha256(cell / "results.json") != item["corrected_results_sha256"]:
            raise ValueError("corrected result differs from source correction")
        if file_sha256(simulation) != item["corrected_simulation_sha256"]:
            raise ValueError("corrected simulation differs from source correction")


def _load_verdicts(
    judges_roots: list[Path], language: str, simulation_id: str
) -> list[StoredVerdict]:
    """Exactly one judge root holds each scored call's saved verdict file."""
    paths = [
        path
        for root in judges_roots
        if (path := root / language / f"{simulation_id}.json").is_file()
    ]
    if len(paths) != 1:
        raise ValueError(
            f"{language}/{simulation_id}: expected one saved verdict file, "
            f"found {len(paths)}"
        )
    rows = [
        StoredVerdict.model_validate(row) for row in json.loads(paths[0].read_text())
    ]
    measures = [row.identity.measure for row in rows]
    if len(measures) != len(set(measures)):
        raise ValueError(f"{paths[0]}: duplicate judge measure")
    if any(row.identity.input.simulation_id != simulation_id for row in rows):
        raise ValueError(f"{paths[0]}: verdict simulation ID mismatch")
    return rows


def _judge_plan(
    scored: list[str], verdicts: dict[str, list[StoredVerdict]], measures: list[str]
) -> SuitePlan:
    """Every saved measure of the language, for every scored call of the cell.

    A call without a saved verdict for one of its language's measures stays
    explicit as not judged; it is never dropped or counted as a failure.
    """
    items = []
    for simulation_id in scored:
        rows = {row.identity.measure: row for row in verdicts[simulation_id]}
        call_input = next(iter(rows.values())).identity.input
        for measure in measures:
            row = rows.get(measure)
            items.append(
                PlannedVerdict(
                    input=call_input,
                    measure=measure,
                    action=PlanAction.REUSE if row else PlanAction.MISSING,
                    identity=row.identity if row else None,
                    reason="Frozen 2026-10-02 release verdict"
                    if row
                    else "No saved release verdict",
                )
            )
    return SuitePlan(items=items)


def _saved_verdicts(
    judges_roots: list[Path], scored_by_cell: dict[tuple[str, str], list[str]]
) -> dict[str, dict[str, list[StoredVerdict]]]:
    """The saved verdict rows of every scored call, by language and call."""
    verdicts: dict[str, dict[str, list[StoredVerdict]]] = defaultdict(dict)
    for (language, _), scored in scored_by_cell.items():
        for simulation_id in scored:
            verdicts[language][simulation_id] = _load_verdicts(
                judges_roots, language, simulation_id
            )
    return verdicts


def _judge_diagnostics(
    official: BenchmarkFrame,
    judges_roots: list[Path],
    scored_by_cell: dict[tuple[str, str], list[str]],
    verdicts: dict[str, dict[str, list[StoredVerdict]]],
):
    """Diagnostics from the saved verdicts of every scored call."""
    for language, by_call in verdicts.items():
        saved = {
            path.stem
            for root in judges_roots
            for path in (root / language).glob("*.json")
        }
        if saved != set(by_call):
            raise ValueError(f"{language}: judge IDs differ from scored calls")
    catalogs = {
        row.identity.catalog_revision
        for by_call in verdicts.values()
        for rows in by_call.values()
        for row in rows
    }
    if len(catalogs) != 1:
        raise ValueError(f"Saved verdicts mix judge catalogs: {sorted(catalogs)}")
    cells = []
    for (language, domain), scored in scored_by_cell.items():
        measures = list(
            dict.fromkeys(
                row.identity.measure
                for rows in verdicts[language].values()
                for row in rows
            )
        )
        stored = [row for call in scored for row in verdicts[language][call]]
        cells.append(
            (
                language,
                domain,
                _judge_plan(scored, verdicts[language], measures),
                stored,
            )
        )
    return export_diagnostics(official, catalogs.pop(), cells)


def _historical_quality(
    source_root: Path,
    official: BenchmarkFrame,
    scored_by_cell: dict[tuple[str, str], list[str]],
    verdicts: dict[str, dict[str, list[StoredVerdict]]],
) -> BenchmarkQuality:
    """Call-level quality from each scored call's saved ticks and verdicts."""
    cells = []
    for (language, domain), scored in scored_by_cell.items():
        directory = _cell_dir(source_root, language, domain) / "simulations"
        calls = []
        for simulation_id in scored:
            simulation = SimulationRun.model_validate_json(
                (directory / f"{simulation_id}.json").read_text()
            )
            rows = {
                row.identity.measure: row for row in verdicts[language][simulation_id]
            }
            calls.append(score_call(simulation, rows))
        cells.append(cell_quality(language, domain, calls))
    return build_quality(official, cells)


def build_historical_package(
    source_root: Path,
    judges_roots: list[Path] | None,
    submitter: SubmitterMetadata,
    benchmark: str = "voice",
) -> BenchmarkPackage:
    """Derive the 2.0 package for one provider's frozen export of one frame.

    ``benchmark`` is the published preset: ``voice`` (the 18 core cells, which
    record their frame and are judged), or ``banking`` / ``entity``, whose
    pre-manifest cells are derived and carry no judge diagnostics.
    """
    if benchmark not in HISTORICAL_BENCHMARKS:
        raise ValueError(
            f"Historical Voice V2 packages one of {', '.join(HISTORICAL_BENCHMARKS)}"
        )
    official = load_frame(benchmark, num_trials=1)
    judged = official.benchmark_version in JUDGED_BENCHMARKS
    if judged != bool(judges_roots):
        raise ValueError(
            f"{official.benchmark_version} "
            + ("requires" if judged else "takes no")
            + " --judges-root"
        )
    _validate_source_correction(source_root)
    trials: list[TaskTrial] = []
    sources: dict[str, str] = {}
    agents: set[tuple[str, str]] = set()
    caller_models: set[str] = set()
    conditions: set[str] = set()
    scored_by_cell: dict[tuple[str, str], list[str]] = {}
    regular_cells: dict[tuple[str, str], tuple] = {}
    for cell in official.cells:
        language, domain = cell.language, cell.domain
        results_path = _cell_dir(source_root, language, domain) / "results.json"
        data = json.loads(results_path.read_text())
        simulations = data["simulation_index"]
        info = data["info"]
        if judged:
            benchmark_info = info["benchmark"]
            audio_config = benchmark_info["conditions"]["audio_native_config"]
            task_mapping = _validate_cell_frame(
                data, simulations, language, domain, official
            )
            fingerprint = benchmark_info["condition_fingerprint"]
        else:
            audio_config = info["audio_native_config"]
            task_mapping, recorded = _derived_cell(data, simulations, cell, official)
            fingerprint = _condition_fingerprint(recorded)
        agents.add((audio_config["provider"], audio_config["model"]))
        caller_models.add(info["user_info"]["llm"])
        conditions.add(fingerprint)
        _validate_simulation_files(source_root, simulations, language, domain)
        sources[f"{language}/{domain}"] = file_sha256(results_path)
        regular_cells[(language, domain)] = (data, task_mapping, results_path)
        trials.extend(
            TaskTrial(
                language=language,
                domain=domain,
                canonical_task_id=task_mapping[str(sim["task_id"])],
                trial=sim["trial"],
                condition_fingerprint=fingerprint,
                binding_revision=cell.binding_revision,
                status=TrialStatus.SCORED
                if sim["reward"] is not None
                else TrialStatus.ERROR,
                reward=sim["reward"],
            )
            for sim in simulations
        )
        scored_by_cell[(language, domain)] = [
            sim["id"] for sim in simulations if sim["reward"] is not None
        ]
    if len(agents) != 1 or len(caller_models) != 1:
        raise ValueError("Voice V2 export mixes agent or caller models")
    if len(conditions) != 1:
        raise ValueError("Voice V2 export mixes recorded run conditions")
    provider, model = agents.pop()
    verdicts = _saved_verdicts(judges_roots, scored_by_cell) if judged else {}
    return BenchmarkPackage(
        metadata=PackageMetadata(
            **submitter.model_dump(),
            agent_provider=provider,
            agent_model=model,
            evidence=EVIDENCE,
        ),
        submission=BenchmarkSubmission(
            frame=official,
            trials=trials,
            summary=summarize(official, trials),
            sources=sources,
        ),
        diagnostics=_judge_diagnostics(official, judges_roots, scored_by_cell, verdicts)
        if judged
        else None,
        quality=_historical_quality(source_root, official, scored_by_cell, verdicts)
        if judged and official.benchmark_version in QUALITY_BENCHMARKS
        else None,
        pass3=_historical_pass3(source_root, official, regular_cells)
        if official.benchmark_version in PASS3_BENCHMARKS
        else None,
    ).check()


def prepare_historical_package(
    source_root: Path,
    judges_roots: list[Path] | None,
    metadata_file: Path,
    output: Path,
    benchmark: str = "voice",
) -> BenchmarkPackage:
    """Write ``<output>/<name>/`` from the saved export; nothing is copied."""
    submitter = SubmitterMetadata.model_validate_json(metadata_file.read_text())
    package = build_historical_package(source_root, judges_roots, submitter, benchmark)
    destination = output / package.name
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    package.write(destination)
    print(f"Saved {destination}")
    return package


def validate_historical_package(
    directory: Path,
    source_root: Path,
    judges_roots: list[Path] | None,
    benchmark: str | None = None,
) -> BenchmarkPackage:
    """Recompute the package from the saved export and verdicts.

    The preset is read from the package's frame; ``benchmark``, if given, must
    name the same one.
    """
    package = load_package(directory)
    if package.metadata.evidence != EVIDENCE:
        raise ValueError(f"{directory} is not a historical Voice V2 package")
    recorded = next(
        (
            name
            for name in HISTORICAL_BENCHMARKS
            if load_frame(name).benchmark_version
            == package.submission.frame.benchmark_version
        ),
        None,
    )
    if recorded is None:
        raise ValueError(f"{directory} packages no frame of the release export")
    if benchmark not in (None, recorded):
        raise ValueError(f"{directory} packages {recorded}, not {benchmark}")
    submitter = SubmitterMetadata.model_validate(
        package.metadata.model_dump(include=set(SubmitterMetadata.model_fields))
    )
    expected = build_historical_package(source_root, judges_roots, submitter, recorded)
    expected.metadata.trajectories = package.metadata.trajectories
    if expected.submission != package.submission:
        raise ValueError("Submission differs from the saved release export")
    if expected.diagnostics != package.diagnostics:
        raise ValueError("Judge diagnostics differ from the saved release verdicts")
    if expected.quality != package.quality:
        raise ValueError("Call quality differs from the saved release calls")
    if expected.pass3 != package.pass3:
        raise ValueError("Pass^3 differs from the saved release calls")
    if expected.metadata != package.metadata:
        raise ValueError("Recorded agent identity differs from the saved export")
    return package

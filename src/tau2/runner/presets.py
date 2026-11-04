"""Resolve published benchmark matrices into ordinary runner configurations."""

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Union

from pydantic import Discriminator, Field, Tag

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_BANKING_RETRIEVAL_CONFIG,
    DEFAULT_BENCHMARK_USER_ARGS,
    DEFAULT_BENCHMARK_USER_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE_BENCHMARK,
    DEFAULT_VOICE_BENCHMARK_MAX_STEPS_SECONDS,
    DEFAULT_VOICE_BENCHMARK_OPENAI_VOICE,
    ELICITATION_DOMAINS,
)
from tau2.data_model.benchmark import (
    BenchmarkFrame,
    BenchmarkManifest,
    BenchmarkModel,
    BenchmarkPlan,
    BenchmarkRunInfo,
    BenchmarkSelection,
    CellFrame,
    call_mode_tag,
    encode_conditions,
)
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import RunConfig, TextRunConfig, VoiceRunConfig

if TYPE_CHECKING:
    from tau2.data_model.language import LanguageTaskInfo, TaskSelection
    from tau2.multilingual.runtime import LoadedLanguage


class BenchmarkRequest(BenchmarkModel):
    """Public Python/CLI request resolved before ordinary single-cell execution."""

    preset: str = Field(
        default=DEFAULT_VOICE_BENCHMARK,
        description="Published benchmark manifest; installed packs do not expand it. "
        "Its call mode must match the template's (voice or text).",
    )
    template: Annotated[
        Union[
            Annotated[VoiceRunConfig, Tag("voice")],
            Annotated[TextRunConfig, Tag("text")],
        ],
        Discriminator(call_mode_tag),
    ] = Field(
        default_factory=lambda: VoiceRunConfig(audio_native_config=AudioNativeConfig()),
        description="Shared VoiceRunConfig or TextRunConfig overrides, preserving unset "
        "fields until benchmark defaults resolve.",
    )
    output: Path = Field(
        description="One matrix output root containing ordinary cell runs."
    )
    languages: list[str] | None = Field(
        default=None,
        description="Released language filter; None selects all released languages.",
    )
    domains: list[str] | None = Field(
        default=None,
        description="Released domain filter; None selects all released domains.",
    )


def resolve_benchmark(request: BenchmarkRequest) -> BenchmarkPlan:
    """Resolve the released benchmark for CLI and Python callers through one path."""
    frame = load_frame(request.preset, num_trials=request.template.num_trials)
    verify_frame(frame, request.preset)
    if template_mode(request.template) != frame.mode:
        raise ValueError(
            f"Benchmark '{request.preset}' runs {frame.mode} calls; "
            f"the template is a {template_mode(request.template)} run config"
        )
    plan = resolve_configs(
        frame,
        request.template,
        request.output,
        languages=request.languages,
        domains=request.domains,
    )
    return prepare_plan(plan, request.output)


def template_mode(config: RunConfig) -> str:
    """The call mode a run config executes."""
    return call_mode_tag(config)


def binding_revision(loaded: "LoadedLanguage") -> str:
    """Fingerprint all runtime content consumed by this domain/language binding.

    The call mode is part of the binding: text and voice calls resolve different
    persona and scaffold content from the same pack.
    """
    from tau2.backchannel import BACKCHANNEL_PROMPT_VERSION
    from tau2.data_model.language import LANGUAGE_PROMPT_VERSION, LanguageTaskInfo
    from tau2.user.guidelines import GUIDELINE_TEMPLATE_VERSION

    content = {
        "mode": loaded.mode,
        "pack_revision": loaded.pack.revision,
        "domain_revision": loaded.binding.revision,
        **loaded.artifact_pins.model_dump(mode="json"),
        "prompt_version": LANGUAGE_PROMPT_VERSION,
        "guideline_template_version": GUIDELINE_TEMPLATE_VERSION,
        "assignment_version": LanguageTaskInfo.model_fields[
            "assignment_version"
        ].default,
    }
    if loaded.mode == "voice":
        content["backchannel_prompt_version"] = BACKCHANNEL_PROMPT_VERSION
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cell_selection(domain: str) -> "TaskSelection | None":
    """The frozen task-loader selection a released cell pins.

    Elicitation cells pin the default strategy's task variant, so a released
    entity frame cannot silently mix strategies.
    """
    if domain not in ELICITATION_DOMAINS:
        return None
    from tau2.data_model.elicitation import ElicitationConfig
    from tau2.data_model.language import TaskSelection

    return TaskSelection(task_kwargs={"strategy": ElicitationConfig().strategy.value})


def load_cell_language(
    language: str, domain: str, mode: str, *, canonical_tasks: bool = False
) -> "LoadedLanguage":
    """Load one released cell's binding exactly as the benchmark pins it.

    Domains outside the base pack resolve through their published extension.
    """
    from tau2.multilingual.runtime import extension_for_domain, load_language

    return load_language(
        language,
        domain,
        mode=mode,
        selection=cell_selection(domain),
        pack_extension=extension_for_domain(language, domain),
        canonical_tasks=canonical_tasks,
    )


def uses_canonical_tasks(benchmark: object | None) -> bool:
    """Whether a benchmark binds the canonical registry tasks (voice-v1).

    voice-v1 skips the English pack's reviewed task replacements (retail and
    telecom ``tasks_file``). Accepts a run config's benchmark request, a recorded
    ``BenchmarkRunInfo`` or a frame, so every reload repeats the planner's rule.
    """
    return getattr(benchmark, "benchmark_version", None) == "voice-v1"


def reload_recorded_call(
    context: "LanguageTaskInfo", domain: str, *, canonical_tasks: bool = False
) -> "LoadedLanguage":
    """Reload exactly the binding a recorded call used, rejecting any drift.

    ``canonical_tasks`` must repeat the planner's choice (``uses_canonical_tasks``).
    Kept outside ``multilingual.runtime``, whose source is part of the pinned
    intake runtime rules digest.
    """
    from tau2.multilingual.runtime import load_language, reload_call_language

    if not canonical_tasks:
        return reload_call_language(context, domain)
    loaded = load_language(
        context.language,
        domain,
        mode=context.mode,
        localize_db=context.scenario_mode != "canonical",
        selection=context.task_selection,
        pack_extension=context.pack_extension,
        canonical_tasks=True,
    )
    if loaded.artifact_pins != context.artifact_pins:
        raise ValueError("published language artifacts differ from the recorded call")
    return loaded


def resolve_configs(
    frame: BenchmarkFrame,
    template: RunConfig,
    output_root: Path,
    *,
    languages: list[str] | None = None,
    domains: list[str] | None = None,
) -> BenchmarkPlan:
    """Resolve metadata mappings and all artifact pins before any provider work.

    The full frame must be published with exact binding revisions first. Extra
    installed languages cannot silently enter a released benchmark. Selection filters only choose
    which cells to run, while the returned frame retains the full denominator.
    """
    from tau2.multilingual.runtime import run_task_selection

    if template.pack_extension is not None:
        raise ValueError(
            "Released benchmarks select their own pack extensions; "
            "use --benchmark none for a custom extension"
        )
    if not template.effective_localize_db:
        raise ValueError(
            "Released benchmarks require regional scenarios; use --benchmark none with --no-localize-db for canonical scenarios in the selected language"
        )
    if template.language != DEFAULT_LANGUAGE:
        raise ValueError(
            "Use the benchmark language filter, not a language-bound template"
        )
    if frame.trials != list(range(len(frame.trials))):
        raise ValueError(
            "Runner trials must be the contiguous indices zero through N-1"
        )
    allowed_languages = {cell.language for cell in frame.cells}
    allowed_domains = {cell.domain for cell in frame.cells}
    for selected, allowed in (
        (languages, allowed_languages),
        (domains, allowed_domains),
    ):
        if selected is not None and (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= allowed
        ):
            raise ValueError(
                "Selection must be a nonempty unique subset of the official matrix; use --benchmark none for custom languages or domains"
            )
    if template.task_set_name is not None or template.task_split_name not in (
        None,
        "base",
    ):
        raise ValueError(
            "Released benchmark tasks are frozen; use --benchmark none for custom task sets or splits"
        )
    if frame.benchmark_version == "voice-v1" and (
        template.task_ids is not None or template.num_tasks is not None
    ):
        raise ValueError(
            "Voice V1 submissions require the complete task frame; "
            "use --benchmark none for task samples"
        )
    if template.task_ids is not None and template.num_tasks is not None:
        raise ValueError("Choose either task_ids or num_tasks, not both")
    if template.task_ids is not None:
        if domains is None or len(domains) != 1:
            raise ValueError("Canonical task IDs require exactly one domain")
        if not template.task_ids or len(set(template.task_ids)) != len(
            template.task_ids
        ):
            raise ValueError("Task IDs must be nonempty and unique")
    configs = []
    selected_cells = []
    for cell in frame.cells:
        # Pin every official binding, even for a filtered execution plan.
        loaded = load_cell_language(
            cell.language,
            cell.domain,
            frame.mode,
            canonical_tasks=uses_canonical_tasks(frame),
        )
        if binding_revision(loaded) != cell.binding_revision:
            raise ValueError(
                f"Frozen binding changed for {cell.language}/{cell.domain}"
            )
        local_ids = {
            metadata.canonical_task_id: task_id
            for task_id, metadata in loaded.metadata.items()
        }
        if not set(cell.canonical_task_ids) <= set(local_ids):
            raise ValueError("Published binding is missing frozen canonical tasks")
        if languages is not None and cell.language not in languages:
            continue
        if domains is not None and cell.domain not in domains:
            continue
        selected_ids = cell.canonical_task_ids
        if template.task_ids is not None:
            if not set(template.task_ids) <= set(selected_ids):
                raise ValueError(
                    "Requested task IDs are outside the frozen subset; use --benchmark none for custom tasks"
                )
            selected_ids = [
                task_id for task_id in selected_ids if task_id in template.task_ids
            ]
        if template.num_tasks is not None:
            if not 1 <= template.num_tasks <= len(selected_ids):
                raise ValueError(
                    f"num_tasks must be between 1 and {len(selected_ids)} for {cell.domain}"
                )
            selected_ids = selected_ids[: template.num_tasks]
        selected_cells.append(
            cell.model_copy(update={"canonical_task_ids": selected_ids})
        )
        values = template.template_values()
        values.setdefault("llm_user", DEFAULT_BENCHMARK_USER_MODEL)
        if values["llm_user"] == DEFAULT_BENCHMARK_USER_MODEL:
            values.setdefault("llm_args_user", DEFAULT_BENCHMARK_USER_ARGS.copy())
        if isinstance(template, VoiceRunConfig):
            audio_values = values["audio_native_config"]
            audio_values.setdefault(
                "model",
                DEFAULT_AUDIO_NATIVE_MODELS[template.audio_native_config.provider],
            )
            if cell.domain not in ELICITATION_DOMAINS:
                # Elicitation calls keep their domain's own call-length default.
                audio_values.setdefault(
                    "max_steps_seconds", DEFAULT_VOICE_BENCHMARK_MAX_STEPS_SECONDS
                )
            if template.audio_native_config.provider == "openai":
                audio_values.setdefault("voice", DEFAULT_VOICE_BENCHMARK_OPENAI_VOICE)
        values.update(
            domain=cell.domain,
            language=cell.language,
            task_set_name=None,
            task_split_name=None,
            task_ids=[local_ids[canonical_id] for canonical_id in selected_ids],
            num_tasks=None,
            num_trials=len(frame.trials),
            auto_resume=True,
            save_to=str(output_root.resolve() / f"{cell.language}-{cell.domain}"),
            benchmark=None,
            pack_extension=loaded.pack_extension,
            language_artifacts=loaded.artifact_pins,
        )
        config = type(template).model_validate(values)
        selection = cell_selection(cell.domain)
        if selection is not None and run_task_selection(config) != selection:
            raise ValueError(
                f"Released {cell.domain} cells use the default elicitation strategy; "
                "use --benchmark none for other strategies"
            )
        if cell.domain == "banking_knowledge" and (
            config.retrieval_config != DEFAULT_BANKING_RETRIEVAL_CONFIG
            or config.retrieval_config_kwargs
        ):
            raise ValueError(
                f"Released banking cells use the {DEFAULT_BANKING_RETRIEVAL_CONFIG} "
                "retrieval config; use --benchmark none for other retrieval settings"
            )
        common = encode_conditions(config)
        condition = hashlib.sha256(
            json.dumps(
                common.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        config.benchmark = BenchmarkRunInfo(
            benchmark_version=frame.benchmark_version,
            frame_revision=frame.revision,
            language=cell.language,
            domain=cell.domain,
            binding_revision=cell.binding_revision,
            canonical_task_ids=cell.canonical_task_ids,
            selected_task_ids=selected_ids,
            trials=frame.trials,
            condition_fingerprint=condition,
            conditions=common,
        )
        configs.append(config)
    conditions = {config.benchmark.condition_fingerprint for config in configs}
    if len(conditions) > 1:
        raise ValueError("Effective agent/user conditions differ across resolved cells")
    return BenchmarkPlan(
        frame=frame,
        selection=BenchmarkSelection(cells=selected_cells),
        partial=selected_cells != frame.cells,
        configs=configs,
    )


class ExecutionSettings(BenchmarkModel):
    """Explicit controller allocation; no independent scheduler or judge enablement."""

    workers: Annotated[
        int, Field(gt=0, description="Existing controller worker process count.")
    ]
    provider_limits: Annotated[
        dict[str, Annotated[int, Field(gt=0)]] | None,
        Field(default=None, description="Existing provider concurrency caps."),
    ]
    global_limit: Annotated[
        int | None,
        Field(default=None, gt=0, description="Existing global concurrent-call cap."),
    ]


def preset_names() -> list[str]:
    """List benchmark manifests installed under the configured data root."""
    from tau2.utils import DATA_DIR

    return sorted(
        path.stem for path in (DATA_DIR / "tau2" / "benchmarks").glob("*.json")
    )


def manifest_path(name: str) -> Path:
    """The manifest file for a benchmark name, confined to the benchmarks directory."""
    import re

    from tau2.utils import DATA_DIR

    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError("Invalid benchmark preset name")
    directory = (DATA_DIR / "tau2" / "benchmarks").resolve()
    path = directory / f"{name}.json"
    if path.is_symlink() or not path.resolve().is_relative_to(directory):
        raise ValueError("Benchmark preset name must not escape its directory")
    return path


def load_manifest(name: str) -> tuple[BenchmarkManifest, Path]:
    """Read one installed benchmark manifest by name, confined to its directory."""
    path = manifest_path(name)
    return BenchmarkManifest.model_validate_json(path.read_text()), path


def load_frame(name: str = "voice", *, num_trials: int = 1) -> BenchmarkFrame:
    """Expand a published manifest from its digest-pinned task subsets."""
    from tau2.task_subsets.store import load_subset, subset_path

    if num_trials < 1:
        raise ValueError("num_trials must be positive")
    manifest, _ = load_manifest(name)
    if set(manifest.task_subsets) != {cell.domain for cell in manifest.cells}:
        raise ValueError("Manifest subsets must match the released domains")
    subsets = {}
    for domain, reference in manifest.task_subsets.items():
        payload = subset_path(reference.name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference.sha256:
            raise ValueError(f"Frozen subset digest changed for {reference.name}")
        subset = load_subset(reference.name)
        if subset.domain != domain:
            raise ValueError("Manifest domain and task subset domain disagree")
        subsets[domain] = subset
    return BenchmarkFrame(
        benchmark_version=manifest.benchmark_version,
        revision=manifest.revision,
        mode=manifest.mode,
        trials=list(range(num_trials)),
        cells=[
            CellFrame(
                **cell.model_dump(), canonical_task_ids=subsets[cell.domain].task_ids
            )
            for cell in manifest.cells
        ],
    )


def verify_frame(frame: BenchmarkFrame, name: str = "voice") -> None:
    """Verify official membership and the source data of each referenced subset."""
    from tau2.task_subsets.store import load_subset, verify_subset

    if frame != load_frame(name, num_trials=len(frame.trials)):
        raise ValueError("Benchmark frame differs from the published manifest")
    manifest, _ = load_manifest(name)
    for reference in manifest.task_subsets.values():
        verify_subset(load_subset(reference.name))


def extend_plan(previous: BenchmarkPlan, requested: BenchmarkPlan) -> BenchmarkPlan:
    """Resume the union of requested tasks while retaining all completed calls."""
    if previous.frame != requested.frame:
        raise ValueError(
            "Cannot resume a different benchmark frame at the same output path"
        )
    fingerprints = {
        config.benchmark.condition_fingerprint
        for config in previous.configs + requested.configs
    }
    if len(fingerprints) != 1:
        raise ValueError(
            "Cannot resume different effective benchmark conditions; use a new output path"
        )
    by_cell = {(config.language, config.domain): config for config in previous.configs}
    for config in requested.configs:
        key = (config.language, config.domain)
        old = by_cell.get(key)
        if old is not None:
            local_ids = dict(zip(old.benchmark.selected_task_ids, old.task_ids))
            local_ids.update(zip(config.benchmark.selected_task_ids, config.task_ids))
            selected = [
                task_id
                for task_id in config.benchmark.canonical_task_ids
                if task_id in local_ids
            ]
            config = config.model_copy(deep=True)
            config.task_ids = [local_ids[task_id] for task_id in selected]
            config.benchmark = config.benchmark.model_copy(
                update={"selected_task_ids": selected}
            )
        by_cell[key] = config
    configs = [
        by_cell[(cell.language, cell.domain)]
        for cell in requested.frame.cells
        if (cell.language, cell.domain) in by_cell
    ]
    cells = [
        CellFrame(
            language=config.language,
            domain=config.domain,
            binding_revision=config.benchmark.binding_revision,
            canonical_task_ids=config.benchmark.selected_task_ids,
        )
        for config in configs
    ]
    return BenchmarkPlan(
        frame=requested.frame,
        selection=BenchmarkSelection(cells=cells),
        partial=cells != requested.frame.cells,
        configs=configs,
    )


def prepare_plan(plan: BenchmarkPlan, output: Path) -> BenchmarkPlan:
    """Resolve cumulative resume scope and validate all input paths before preview/work."""
    from tau2.multilingual.runtime import load_run_language

    plan = BenchmarkPlan.model_validate(plan.model_dump())
    frame_path = output / "benchmark-frame.json"
    if (
        frame_path.exists()
        and BenchmarkFrame.model_validate_json(frame_path.read_text()) != plan.frame
    ):
        raise ValueError(
            "Cannot resume a different benchmark frame at the same output path"
        )
    plan_path = output / "benchmark-plan.json"
    if plan_path.exists():
        previous = BenchmarkPlan.model_validate_json(plan_path.read_text())
        plan = extend_plan(previous, plan)
    elif output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory is not a recorded benchmark; use a new output path"
        )
    configs = []
    for config in plan.configs:
        loaded = load_run_language(config)
        local_ids = {
            entry.canonical_task_id: task_id
            for task_id, entry in loaded.metadata.items()
        }
        expected_ids = [
            local_ids[task_id] for task_id in config.benchmark.selected_task_ids
        ]
        if (
            config.task_ids != expected_ids
            or binding_revision(loaded) != config.benchmark.binding_revision
            or config.language_artifacts != loaded.artifact_pins
        ):
            raise ValueError(
                "Plan tasks or bindings differ from authoritative release data"
            )
        configs.append(
            config.model_copy(
                update={
                    "save_to": str(
                        output.resolve() / f"{config.language}-{config.domain}"
                    )
                }
            )
        )
    return BenchmarkPlan(
        frame=plan.frame,
        selection=plan.selection,
        partial=plan.partial,
        configs=configs,
    )


def execute_plan(
    plan: BenchmarkPlan,
    output: Path,
    settings: ExecutionSettings,
    *,
    dry_run: bool = False,
) -> None:
    """Preflight all conditions; ordinary controller handles processes and resume."""
    from tau2.runner.batch import run_domains

    plan = prepare_plan(plan, output)
    frame_path = output / "benchmark-frame.json"
    plan_path = output / "benchmark-plan.json"
    for config in plan.configs:
        config.validate()
    if dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    frame_path.write_text(plan.frame.model_dump_json(indent=2))
    plan_path.write_text(plan.model_dump_json(indent=2))
    (output / "execution.json").write_text(settings.model_dump_json(indent=2))
    run_domains(
        plan.configs,
        workers=settings.workers,
        provider_limits=settings.provider_limits,
        global_limit=settings.global_limit,
    )


def infer_preset(frame: BenchmarkFrame, name: str | None = None) -> str:
    """Identify a recorded frame by exact published membership and revisions."""
    names = [name] if name is not None else preset_names()
    matches = [
        candidate
        for candidate in names
        if frame == load_frame(candidate, num_trials=len(frame.trials))
    ]
    if len(matches) != 1:
        raise ValueError(
            "Recorded frame must match exactly one installed published benchmark"
        )
    return matches[0]


def pin_manifest(
    name: str,
    *,
    revision: str | None = None,
    source: str | None = None,
    mode: str | None = None,
    benchmark_version: str | None = None,
) -> BenchmarkManifest:
    """Recompute a manifest's subset digests and binding revisions from this checkout.

    Pins are derived, never authored: each subset digest is the SHA-256 of the
    published subset file and each binding revision fingerprints the installed
    pack binding in the manifest's call mode. With ``source``, a new manifest is
    derived from an existing one (same cells and subsets) for another mode.
    """
    from tau2.task_subsets.store import subset_path

    if source is not None:
        target = manifest_path(name)
        if target.exists():
            raise FileExistsError(f"benchmark '{name}' already exists; pin it instead")
        if revision is None or benchmark_version is None or mode is None:
            raise ValueError(
                "deriving a new benchmark requires --revision, --benchmark-version "
                "and --mode"
            )
        base, _ = load_manifest(source)
    else:
        if benchmark_version is not None or mode is not None:
            raise ValueError("--benchmark-version and --mode apply only with --from")
        base, target = load_manifest(name)
    call_mode = mode or base.mode
    subsets = {
        domain: reference.model_copy(
            update={
                "sha256": hashlib.sha256(
                    subset_path(reference.name).read_bytes()
                ).hexdigest()
            }
        )
        for domain, reference in base.task_subsets.items()
    }
    cells = [
        cell.model_copy(
            update={
                "binding_revision": binding_revision(
                    load_cell_language(
                        cell.language,
                        cell.domain,
                        call_mode,
                        canonical_tasks=(benchmark_version or base.benchmark_version)
                        == "voice-v1",
                    )
                )
            }
        )
        for cell in base.cells
    ]
    pins_changed = subsets != base.task_subsets or cells != base.cells
    if source is None and pins_changed and (revision or base.revision) == base.revision:
        raise ValueError(
            f"benchmark '{name}' pins changed but its revision {base.revision!r} "
            "did not; pass --revision with a new release revision"
        )
    manifest = BenchmarkManifest(
        benchmark_version=benchmark_version or base.benchmark_version,
        revision=revision or base.revision,
        mode=call_mode,
        task_subsets=subsets,
        cells=cells,
    )
    target.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest

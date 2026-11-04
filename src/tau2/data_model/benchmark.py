"""Persisted benchmark membership, run provenance, and task-score contracts."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationInfo,
    model_validator,
)

from tau2.config import DOMAIN_TOOLS_VERSION, VOICE_INSTRUCTION_PROMPT_VERSION
from tau2.data_model.cascaded import (
    ElevenLabsTTSConfig,
    resolve_cascaded_config,
)
from tau2.data_model.language import CallMode
from tau2.data_model.run_config import (
    AudioNativeConfig,
    TextRunSettings,
    VoiceRunSettings,
)

if TYPE_CHECKING:
    from tau2.data_model.simulation import RunConfig, TextRunConfig, VoiceRunConfig

Language = Annotated[str, Field(min_length=1)]
Domain = Annotated[str, Field(min_length=1)]
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class BenchmarkModel(BaseModel):
    """Reject unknown fields at persisted scoring boundaries."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CellFrame(BenchmarkModel):
    """The canonical task population and artifact revision for one cell."""

    language: Annotated[Language, Field(description="Released language identifier.")]
    domain: Annotated[Domain, Field(description="Shared benchmark domain.")]
    canonical_task_ids: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Frozen canonical IDs, in frame order.",
        ),
    ]
    binding_revision: Annotated[
        str,
        Field(
            min_length=1, description="Immutable selected task/pack binding revision."
        ),
    ]

    @model_validator(mode="after")
    def unique_tasks(self):
        """A repeated task must not change its weight in the benchmark."""
        if len(set(self.canonical_task_ids)) != len(self.canonical_task_ids):
            raise ValueError("A cell requires distinct canonical task IDs")
        return self


class BenchmarkFrame(BenchmarkModel):
    """Complete official matrix; selection never redefines its denominator."""

    benchmark_version: Annotated[
        str,
        Field(
            min_length=1,
            description="Benchmark generation from the published manifest.",
        ),
    ]
    revision: Annotated[
        str, Field(min_length=1, description="Frozen matrix manifest revision.")
    ]
    mode: Annotated[
        CallMode,
        Field(description="Call mode every cell runs in (text or voice)."),
    ]
    trials: Annotated[
        list[int],
        Field(
            min_length=1,
            description="Explicit nonnegative trial indices owed for every task.",
        ),
    ]
    cells: Annotated[
        list[CellFrame],
        Field(min_length=1, description="Complete released language/domain matrix."),
    ]

    @model_validator(mode="after")
    def complete_matrix(self):
        """Enforce the official frame without inferring coverage from results."""
        actual = {(cell.language, cell.domain) for cell in self.cells}
        if len(actual) != len(self.cells):
            raise ValueError("Benchmark cells must be unique")
        expected = {
            (language, domain)
            for language in {cell.language for cell in self.cells}
            for domain in {cell.domain for cell in self.cells}
        }
        if actual != expected:
            raise ValueError("Every released language must cover every released domain")
        if min(self.trials) < 0 or len(set(self.trials)) != len(self.trials):
            raise ValueError("Trial indices must be distinct and nonnegative")
        for domain in {cell.domain for cell in self.cells}:
            frames = [
                cell.canonical_task_ids for cell in self.cells if cell.domain == domain
            ]
            if any(ids != frames[0] for ids in frames[1:]):
                raise ValueError(
                    "Every language must use the same ordered canonical domain frame"
                )
        return self


class BenchmarkSelection(BenchmarkModel):
    """Explicit execution scope, separate from the released score denominator."""

    cells: list[CellFrame] = Field(
        min_length=1,
        description="Requested cells and canonical tasks, in released order.",
    )

    def validate_against(self, frame: BenchmarkFrame) -> None:
        """Reject duplicate, foreign or mutated population entries."""
        expected = {(cell.language, cell.domain): cell for cell in frame.cells}
        seen = set()
        for cell in self.cells:
            key = (cell.language, cell.domain)
            official = expected.get(key)
            if key in seen or official is None:
                raise ValueError(
                    "Selection contains duplicate or foreign benchmark cells"
                )
            seen.add(key)
            if (
                cell.binding_revision != official.binding_revision
                or cell.canonical_task_ids
                != [
                    task_id
                    for task_id in official.canonical_task_ids
                    if task_id in cell.canonical_task_ids
                ]
            ):
                raise ValueError(
                    "Selection differs from the frozen benchmark tasks or binding"
                )


class TrialStatus(str, Enum):
    """Infrastructure errors are separate from ordinary task failure.

    ``ERROR`` trials leave the score's denominator; ``UNSCORED`` ones block it.
    """

    SCORED = "scored"
    ERROR = "error"
    UNSCORED = "unscored"


class TaskTrial(BenchmarkModel):
    """One ordinary run observation, with compatibility identity attached."""

    language: Annotated[Language, Field(description="Recorded run language.")]
    domain: Annotated[Domain, Field(description="Recorded domain.")]
    canonical_task_id: Annotated[
        str,
        Field(description="Explicit canonical task metadata, not suffix inference."),
    ]
    trial: Annotated[int, Field(ge=0, description="Recorded trial index.")]
    condition_fingerprint: Annotated[
        Fingerprint,
        Field(
            description="Fingerprint of all effective comparison conditions and task evaluator contract."
        ),
    ]
    binding_revision: Annotated[
        str, Field(description="Recorded immutable cell artifact revision.")
    ]
    status: Annotated[
        TrialStatus,
        Field(description="Task score availability or explicit error state."),
    ]
    reward: Annotated[
        float | None,
        Field(
            default=None,
            ge=0,
            le=1,
            allow_inf_nan=False,
            description="Ordinary task reward; zero is a scored failure.",
        ),
    ]

    @model_validator(mode="after")
    def reward_matches_status(self):
        """Never interpret an infrastructure error as a scored zero or pass."""
        if (self.status == TrialStatus.SCORED) != (self.reward is not None):
            raise ValueError("Only scored trials must carry a reward")
        return self


class CellSummary(BenchmarkModel):
    """Score and all denominator components for one matrix cell."""

    language: Annotated[Language, Field(description="Language identifier.")]
    domain: Annotated[Domain, Field(description="Domain identifier.")]
    expected_trials: Annotated[
        int, Field(description="Frozen tasks multiplied by selected trial count.")
    ]
    scored_trials: Annotated[
        int, Field(description="Observed trials with usable ordinary task reward.")
    ]
    error_trials: Annotated[
        int,
        Field(description="Explicit infrastructure errors, excluded from the score."),
    ]
    unscored_trials: Annotated[
        int, Field(description="Stored trials without an evaluated reward.")
    ]
    missing_trials: Annotated[
        int, Field(description="Expected trials absent from input.")
    ]
    complete: Annotated[
        bool,
        Field(
            description="Every expected task/trial is scored or errored, and at least one is scored."
        ),
    ]
    score: Annotated[
        float | None,
        Field(
            description="Equal-task mean over scored trials, only when this cell is complete."
        ),
    ]
    observed_score: Annotated[
        float | None,
        Field(
            description="Partial mean: available trials within tasks, then available tasks equally."
        ),
    ]


class ScoreCoverage(BenchmarkModel):
    """Score and coverage for a language or domain, using equal cell weights."""

    expected_trials: int = Field(description="Task/trial units owed by this group.")
    scored_trials: int = Field(description="Units with usable task rewards.")
    error_trials: int = Field(
        description="Recorded infrastructure errors, excluded from the score."
    )
    unscored_trials: int = Field(description="Stored units without task rewards.")
    missing_trials: int = Field(description="Units absent from saved results.")
    complete: bool = Field(description="Every cell in the group is complete.")
    score: float | None = Field(
        description="Equal-cell score only when the group is complete."
    )
    observed_score: float | None = Field(
        description="Explicitly partial mean of cells with observed scores."
    )


class BenchmarkSummary(BenchmarkModel):
    """Full benchmark score cannot be reported for partial coverage."""

    benchmark_version: Annotated[
        str,
        Field(
            min_length=1,
            description="Benchmark generation from the published manifest.",
        ),
    ]
    frame_revision: Annotated[
        str, Field(description="Frozen matrix revision summarized.")
    ]
    condition_fingerprint: Annotated[
        Fingerprint | None,
        Field(
            description="Common effective condition identity; absent for empty input."
        ),
    ]
    expected_trials: Annotated[int, Field(description="Trials owed across all cells.")]
    scored_trials: Annotated[
        int, Field(description="Trials with usable ordinary rewards.")
    ]
    error_trials: Annotated[
        int,
        Field(description="Explicit infrastructure errors, excluded from scores."),
    ]
    unscored_trials: Annotated[int, Field(description="Stored but unscored trials.")]
    missing_trials: Annotated[
        int, Field(description="Absent expected task/trial units.")
    ]
    complete: Annotated[bool, Field(description="All cells have complete scores.")]
    score: Annotated[
        float | None,
        Field(description="Official equal-cell score; null unless complete."),
    ]
    observed_score: Annotated[
        float | None,
        Field(
            description="Explicitly partial equal mean of cells with any scored observations."
        ),
    ]
    by_language: dict[str, ScoreCoverage] = Field(
        description="Language breakdowns using the declared scope."
    )
    by_domain: dict[str, ScoreCoverage] = Field(
        description="Domain breakdowns using the declared scope."
    )
    cells: Annotated[
        list[CellSummary],
        Field(description="All cell breakdowns, including empty cells."),
    ]


class _CompleteConditions(BenchmarkModel):
    """Persisted conditions must not infer omitted settings from today's defaults."""

    domain_tools_version: str | None = Field(
        default=None,
        description="Observed domain lookup semantics; absent legacy records remain readable but cannot qualify as current runs.",
    )

    @model_validator(mode="after")
    def complete_record(self, info: ValidationInfo):
        """Every recorded setting is explicit, so defaults cannot drift silently."""
        recorded = self.model_dump(mode="json", exclude_unset=True)
        complete = self.model_dump(mode="json")
        if "domain_tools_version" not in self.model_fields_set:
            complete.pop("domain_tools_version")
        if recorded != complete:
            historical_fields = {
                "channel_effects_mode",
                "speech_effects_mode",
                "pass3_condition",
                "voice_sampling_seed",
            }
            missing = complete.keys() - recorded.keys()
            if (
                isinstance(self, VoiceBenchmarkConditions)
                and isinstance(info.context, dict)
                and info.context.get("historical_judge_read")
                and missing <= historical_fields
                and all(complete.get(key) == value for key, value in recorded.items())
            ):
                return self
            raise ValueError("Recorded benchmark conditions are incomplete")
        return self


class VoiceBenchmarkConditions(VoiceRunSettings, _CompleteConditions):
    """Complete effective voice settings, using the ordinary runner's field definitions."""

    voice_instruction_prompt_version: str = Field(
        description="Observed shared voice prompt revision included in the fingerprint."
    )


class TextBenchmarkConditions(TextRunSettings, _CompleteConditions):
    """Complete effective text settings, using the ordinary runner's field definitions."""


def call_mode_tag(value) -> str:
    """Voice conditions carry the audio-native configuration; text conditions never do."""
    if isinstance(value, dict):
        return "voice" if "audio_native_config" in value else "text"
    return "voice" if hasattr(value, "audio_native_config") else "text"


BenchmarkConditions = Annotated[
    Union[
        Annotated[VoiceBenchmarkConditions, Tag("voice")],
        Annotated[TextBenchmarkConditions, Tag("text")],
    ],
    Discriminator(call_mode_tag),
]


def conditions_mode(conditions: "BenchmarkConditions") -> CallMode:
    """The call mode a recorded condition contract describes."""
    return call_mode_tag(conditions)


class BenchmarkRunInfo(BenchmarkModel):
    """The immutable benchmark selection recorded on an ordinary cell run."""

    benchmark_version: Annotated[
        str,
        Field(
            min_length=1,
            description="Released benchmark generation; custom runs omit this object.",
        ),
    ]
    frame_revision: Annotated[
        str, Field(description="Frozen matrix manifest revision.")
    ]
    language: Annotated[Language, Field(description="Selected matrix language.")]
    domain: Annotated[Domain, Field(description="Selected matrix domain.")]
    binding_revision: Annotated[
        str,
        Field(
            description="Hash of exact pack/task/metadata artifacts plus declared revisions."
        ),
    ]
    canonical_task_ids: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Frozen selected canonical domain IDs.",
        ),
    ]
    selected_task_ids: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Canonical tasks requested for this cell; a subset of the frozen population.",
        ),
    ]
    trials: Annotated[list[int], Field(description="Explicit expected trial indices.")]
    condition_fingerprint: Annotated[
        Fingerprint,
        Field(
            description="Hash of effective ordinary run conditions after language resolution."
        ),
    ]
    conditions: Annotated[
        BenchmarkConditions,
        Field(
            description="Exact effective conditions hashed above; independently checked on submission.",
        ),
    ]

    @model_validator(mode="after")
    def validate_selected_tasks(self):
        """A declared sample must be an ordered, unique subset of the full population."""
        if self.selected_task_ids != [
            task_id
            for task_id in self.canonical_task_ids
            if task_id in self.selected_task_ids
        ]:
            raise ValueError(
                "Selected canonical tasks differ from the frozen population"
            )
        return self


class TaskSubsetReference(BenchmarkModel):
    """An immutable task subset already owned by the task-subset package."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Frozen subset name.")
    sha256: Fingerprint = Field(description="Digest of the published subset artifact.")


class CellBinding(BenchmarkModel):
    """A released language/domain binding without duplicated task membership."""

    language: Language = Field(description="Released language identifier.")
    domain: Domain = Field(description="Shared domain identifier.")
    binding_revision: str = Field(
        min_length=1, description="Frozen runtime binding digest."
    )


class BenchmarkManifest(BenchmarkModel):
    """Release data expanded into a frame by the ordinary-run preset loader."""

    benchmark_version: str = Field(
        min_length=1, description="Public benchmark generation."
    )
    revision: str = Field(min_length=1, description="Immutable manifest revision.")
    mode: CallMode = Field(description="Call mode every released cell runs in.")
    task_subsets: dict[str, TaskSubsetReference] = Field(
        min_length=1, description="Pinned subset reference per domain."
    )
    cells: list[CellBinding] = Field(
        min_length=1, description="Ordered released language/domain bindings."
    )


class BenchmarkSubmission(BenchmarkModel):
    """Portable task-score export; optional diagnostics remain separate artifacts."""

    schema_version: Literal["benchmark-submission-v1"] = "benchmark-submission-v1"
    frame: Annotated[
        BenchmarkFrame, Field(description="Exact versioned scoring population.")
    ]
    trials: Annotated[
        list[TaskTrial], Field(description="Validated saved ordinary task rewards.")
    ]
    summary: Annotated[
        BenchmarkSummary,
        Field(description="Recomputable task-success coverage and score."),
    ]
    selection: BenchmarkSelection | None = Field(
        default=None,
        description="Recorded cumulative execution scope, if a plan accompanies the results.",
    )
    requested: BenchmarkSummary | None = Field(
        default=None,
        description="Coverage and scoped score for requested tasks; never the official overall score.",
    )
    sources: Annotated[
        dict[str, str],
        Field(description="Run metadata digests by language/domain; no local paths."),
    ]

    @model_validator(mode="after")
    def validate_selection(self):
        """Validate local membership; score recomputation belongs to the reader."""
        if self.selection is not None:
            self.selection.validate_against(self.frame)
        if (self.selection is None) != (self.requested is None):
            raise ValueError("Requested summary requires a recorded selection")
        return self


def encode_conditions(config: "RunConfig") -> BenchmarkConditions:
    """Capture all shared run settings, normalizing only expected cell variation."""
    from tau2.data_model.simulation import VoiceRunConfig

    if not isinstance(config, VoiceRunConfig):
        common = config.model_dump(
            mode="json", include=set(TextRunSettings.model_fields)
        )
        common["domain_tools_version"] = DOMAIN_TOOLS_VERSION
        return TextBenchmarkConditions.model_validate(common)
    common = config.model_dump(mode="json", include=set(VoiceRunSettings.model_fields))
    common["voice_instruction_prompt_version"] = VOICE_INSTRUCTION_PROMPT_VERSION
    common["domain_tools_version"] = DOMAIN_TOOLS_VERSION
    common["audio_native_config"] = shared_audio_config(
        config.audio_native_config, config.language
    ).model_dump(mode="json")
    return VoiceBenchmarkConditions.model_validate(common)


def shared_audio_config(audio: AudioNativeConfig, language: str) -> AudioNativeConfig:
    """Validate observed cell hints before removing their expected variation."""
    if audio.language != language:
        raise ValueError("Provider language must match the benchmark cell")
    expected_transcription = language if audio.provider == "openai" else None
    if audio.transcription_language != expected_transcription:
        raise ValueError("Wrong cell transcription language")
    common = audio.model_copy(deep=True)
    common.language = None
    common.transcription_language = None
    if audio.provider == "livekit":
        if audio.cascaded_config is None:
            raise ValueError("Missing effective cascaded configuration")
        cascade = resolve_cascaded_config(
            config=audio.cascaded_config,
            preset_name=audio.cascaded_request.preset
            if audio.cascaded_request
            else None,
            language=language,
        )
        defaults = resolve_cascaded_config(
            config=None,
            preset_name=audio.cascaded_request.preset
            if audio.cascaded_request
            else None,
            language=language,
        )
        if cascade.stt.language == defaults.stt.language:
            cascade.stt.language = "en"
        if (
            isinstance(cascade.tts, ElevenLabsTTSConfig)
            and cascade.tts.language == language
        ):
            cascade.tts.language = "en"
        if cascade.preamble:
            cascade.preamble_language = "en"
        common.cascaded_config = cascade
        if (
            common.cascaded_request is not None
            and common.cascaded_request.config is not None
        ):
            common.cascaded_request = common.cascaded_request.model_copy(
                update={"config": cascade.model_copy(deep=True)}
            )
    return common


def cell_audio_config(common: AudioNativeConfig, language: str) -> AudioNativeConfig:
    """Restore a shared descriptor to actual settings for independent validation."""
    audio = common.model_copy(deep=True)
    audio.language = language
    audio.transcription_language = language if audio.provider == "openai" else None
    if audio.provider == "livekit" and audio.cascaded_config is not None:
        cascade = audio.cascaded_config
        defaults = resolve_cascaded_config(
            config=None,
            preset_name=audio.cascaded_request.preset
            if audio.cascaded_request
            else None,
            language=language,
        )
        if cascade.stt.language == "en":
            cascade.stt.language = defaults.stt.language
        if (
            isinstance(cascade.tts, ElevenLabsTTSConfig)
            and cascade.tts.language == "en"
        ):
            cascade.tts.language = language
        if cascade.preamble:
            cascade.preamble_language = language
        if (
            audio.cascaded_request is not None
            and audio.cascaded_request.config is not None
        ):
            audio.cascaded_request = audio.cascaded_request.model_copy(
                update={"config": cascade.model_copy(deep=True)}
            )
    return audio


class BenchmarkPlan(BenchmarkModel):
    """Selected ordinary configs plus the complete official denominator."""

    model_config = ConfigDict(defer_build=True)

    @classmethod
    def model_rebuild(cls, *, _types_namespace=None, **kwargs):
        """Resolve the reciprocal run-config reference after both models are defined."""
        from tau2.data_model.simulation import TextRunConfig, VoiceRunConfig

        return super().model_rebuild(
            _types_namespace={
                **(_types_namespace or {}),
                "VoiceRunConfig": VoiceRunConfig,
                "TextRunConfig": TextRunConfig,
            },
            **kwargs,
        )

    frame: Annotated[
        BenchmarkFrame,
        Field(description="Full frozen official matrix, never narrowed by a filter."),
    ]
    selection: BenchmarkSelection = Field(
        description="Explicit selected cells/tasks; does not alter the official frame."
    )
    partial: Annotated[
        bool,
        Field(
            description="True when selection includes fewer cells or tasks than the released benchmark."
        ),
    ]
    configs: Annotated[
        list[
            Annotated[
                Union[
                    Annotated[VoiceRunConfig, Tag("voice")],
                    Annotated[TextRunConfig, Tag("text")],
                ],
                Discriminator(call_mode_tag),
            ]
        ],
        Field(
            description="Ordinary runner configs (all of the frame's mode), each with explicit task IDs and artifact pins."
        ),
    ]

    @model_validator(mode="after")
    def validate_selection(self):
        """Keep selected configs and declared population in agreement."""
        self.selection.validate_against(self.frame)
        expected = {(cell.language, cell.domain): cell for cell in self.selection.cells}
        actual_keys = [(config.language, config.domain) for config in self.configs]
        if actual_keys != list(expected):
            raise ValueError("Plan configs do not match unique ordered selected cells")
        official = {(cell.language, cell.domain): cell for cell in self.frame.cells}
        fingerprints = set()
        for config in self.configs:
            if call_mode_tag(config) != self.frame.mode:
                raise ValueError("Plan configs must run in the frame's call mode")
            if not config.effective_localize_db:
                raise ValueError("Released benchmark plans require regional scenarios")
            key = (config.language, config.domain)
            cell = expected[key]
            released = official[key]
            record = config.benchmark
            if record is None or record.selected_task_ids != cell.canonical_task_ids:
                raise ValueError("Plan config does not match selected canonical tasks")
            if (
                (record.language, record.domain) != key
                or record.benchmark_version != self.frame.benchmark_version
                or record.frame_revision != self.frame.revision
                or record.binding_revision != released.binding_revision
                or record.canonical_task_ids != released.canonical_task_ids
                or record.trials != self.frame.trials
            ):
                raise ValueError("Plan config identity differs from the released frame")
            if (
                config.num_trials != len(self.frame.trials)
                or config.num_tasks is not None
                or config.task_set_name is not None
                or config.task_split_name is not None
                or config.task_ids is None
                or len(config.task_ids) != len(record.selected_task_ids)
            ):
                raise ValueError(
                    "Plan config selection differs from declared tasks/trials"
                )
            common = encode_conditions(config)
            fingerprint = hashlib.sha256(
                json.dumps(
                    common.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if (
                common != record.conditions
                or fingerprint != record.condition_fingerprint
            ):
                raise ValueError(
                    "Plan config differs from recorded effective conditions"
                )
            fingerprints.add(fingerprint)
        if len(fingerprints) != 1:
            raise ValueError("Plan mixes different effective benchmark conditions")
        if self.partial != (self.selection.cells != self.frame.cells):
            raise ValueError("Plan partial flag does not match selected population")
        return self


class SavedBenchmarkCell(BenchmarkModel):
    """One released cell, including an explicit absence of saved results."""

    cell: Annotated[CellFrame, Field(description="Published cell membership.")]
    results_path: Annotated[
        Path | None, Field(description="Saved ordinary results, or null when absent.")
    ]


class SavedBenchmark(BenchmarkModel):
    """Validated matrix inventory for viewers and diagnostic consumers."""

    root: Annotated[Path, Field(description="Saved matrix directory.")]
    preset: Annotated[str, Field(description="Matching installed published manifest.")]
    submission: Annotated[
        BenchmarkSubmission,
        Field(description="Validated evidence and complete official coverage summary."),
    ]
    cells: Annotated[
        list[SavedBenchmarkCell],
        Field(
            description="Every released cell, in frame order, including missing cells."
        ),
    ]

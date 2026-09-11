# Copyright Sierra
"""Run presets for multilingual experiments.

A :class:`RunPreset` bundles the experiment for one multilingual evaluation:
one arm per language-pack persona on the localized task set. The shared
English reference is its own pack and preset (``multilingual_v1_english``);
language results are compared against it post-hoc.

The headline metric is the LLM communicate-info judge, which non-English runs
activate automatically from the run language. The English baseline arm forces
it on (``TAU2_FORCE_LLM_COMMUNICATE_JUDGE=1``), since English would otherwise
fall back to substring matching.

Presets are GENERATED from the ``experiments`` list of each language's
``pack.yaml`` (see ``tau2.multilingual.loader``) — one preset per entry, one
entry per benchmark domain the pack runs; a language owner never edits this
module. Each entry carries the decisions (preset name, description, domain,
smoke task); everything derivable comes from the conventions encoded here:

- main arm: task set ``<domain>_<suffix>`` with ALL its task ids, persona =
  the bare language code (deterministic, seed-rotated per-task round-robin
  among the pack's personas);
- optional extra arms (e.g. an identity-variant ``*_identity`` set): another
  localized task set, same language-code persona sampling;
- matched conditions: openai / regular / 600 s cap / seed 42 unless the
  block overrides them.

Presets are pure data + command construction (no API calls).
``tau2 run-preset <preset_name>`` executes a preset by spawning one
``tau2 run`` subprocess per arm. Smoke is staged: the ``smoke`` stage runs
1 task per arm; the ``full`` stage runs the complete matrix.
"""

import json
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from tau2.config import (
    DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS,
    FORCE_LLM_COMMUNICATE_JUDGE_ENV,
    MULTILINGUAL_SIMULATIONS_SUBDIR,
    SUBSET_AUTO,
)
from tau2.multilingual.schema import ExperimentSpec
from tau2.multilingual.task_sets import discover_localized_task_files
from tau2.task_subsets import resolve_subset

Stage = Literal["smoke", "full"]


class PresetArm(BaseModel):
    """One experiment arm: a persona on a task set, with optional env overrides."""

    name: str = Field(description="Arm name, used in save paths and reports")
    domain: str = Field(description="Domain to run on (environment)")
    task_set_name: str = Field(description="Registered task set name")
    task_ids: list[str] = Field(description="Full task list for the 'full' stage")
    smoke_task_ids: list[str] = Field(
        description="Reduced task list for the 'smoke' stage (typically 1 task)"
    )
    user_persona_id: str = Field(
        description="Forced persona for the whole arm (--user-persona-id). "
        "Language-pack personas activate the multilingual pipeline; plain "
        "English voice personas keep the default English pipeline."
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables for this arm's subprocess",
    )

    def stage_task_ids(self, stage: Stage) -> list[str]:
        return self.smoke_task_ids if stage == "smoke" else self.task_ids

    @model_validator(mode="after")
    def _validate_arm(self) -> "PresetArm":
        missing = set(self.smoke_task_ids) - set(self.task_ids)
        if missing:
            raise ValueError(
                f"Arm '{self.name}': smoke tasks {sorted(missing)} are not in "
                "the full task list. The full list is the domain's fixed task "
                "subset (data/tau2/task_subsets/), so a redraw that dropped "
                "the pack's smoke_task_stem lands here — repoint the stem in "
                "pack.yaml at a task the subset kept."
            )
        return self


class RunPreset(BaseModel):
    """A named, reproducible multi-arm experiment configuration."""

    name: str
    description: str
    audio_native_provider: str = Field(
        description="Audio-native provider for every arm (matched conditions)"
    )
    audio_native_model: Optional[str] = Field(
        default=None,
        description="Audio-native model override; None uses the provider default",
    )
    speech_complexity: str = Field(
        description="Speech complexity level for every arm (matched conditions)"
    )
    max_steps_seconds: Optional[int] = Field(
        default=None,
        description="Conversation duration cap in simulated seconds for every "
        "arm (matched conditions). Keep well under the audio provider's "
        "session lifetime — OpenAI realtime sessions have been observed "
        "dropping at ~18 min. None uses the global default.",
    )
    seed: int = Field(default=42, description="Seed shared by every arm")
    arms: list[PresetArm]

    def get_arm(self, name: str) -> PresetArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(
            f"Unknown arm '{name}'. Options: {[a.name for a in self.arms]}"
        )

    def save_to(self, arm: PresetArm, stage: Stage) -> str:
        """Save path (under data/simulations/) for one arm at one stage.

        Nested under ``multilingual/`` so the experiment tree is one subtree:
        the top level of data/simulations/ stays for one-off runs, and a
        reorganisation of the multilingual runs is a move of one directory.
        """
        return f"{MULTILINGUAL_SIMULATIONS_SUBDIR}/{self.name}/{stage}_{arm.name}"

    def cli_command(
        self,
        arm: PresetArm,
        stage: Stage,
        num_trials: int = 1,
        max_concurrency: int = 1,
        verbose_logs: bool = True,
        extra_args: Optional[list[str]] = None,
    ) -> list[str]:
        """Build the ``tau2 run`` argv for one arm at one stage.

        The caller is responsible for merging ``arm.env`` into the subprocess
        environment (see ``tau2.multilingual.run_preset_driver``).
        """
        task_ids = arm.stage_task_ids(stage)
        command = [
            "tau2",
            "run",
            "--domain",
            arm.domain,
            "--task-set-name",
            arm.task_set_name,
            "--task-ids",
            *task_ids,
            "--audio-native",
            "--audio-native-provider",
            self.audio_native_provider,
            "--speech-complexity",
            self.speech_complexity,
            "--user-persona-id",
            arm.user_persona_id,
            "--seed",
            str(self.seed),
            "--num-trials",
            str(num_trials),
            "--max-concurrency",
            str(max_concurrency),
            "--save-to",
            self.save_to(arm, stage),
        ]
        if self.audio_native_model is not None:
            command += ["--audio-native-model", self.audio_native_model]
        if self.max_steps_seconds is not None:
            command += ["--max-steps-seconds", str(self.max_steps_seconds)]
        # The per-simulation wall-clock timeout every voice run needs (shared
        # with matrix mode): without it a hung provider session blocks the arm
        # indefinitely.
        command += ["--timeout", str(DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS)]
        if verbose_logs:
            command.append("--verbose-logs")
        if extra_args:
            command += extra_args
        return command


# =============================================================================
# Preset generation from pack.yaml experiment blocks
# =============================================================================


def resolve_main_task_suffix(
    domain: str,
    suffix: str,
    *,
    include_identity_arm: bool = True,
    task_files: Optional[dict] = None,
) -> str:
    """The task-set suffix a run's MAIN arm uses.

    Identity-variant ("_identity") set: when the locale identity swap has been
    generated for this language+domain (see `tau2 factory localize-entities`),
    it REPLACES the plain localized set as the run's arm by default — callers
    speak locale identities and the voice is pinned to each identity's gender.
    Opt out per pack with ``experiment.include_identity_arm: false`` to run the
    plain localized set instead.

    SINGLE SOURCE OF TRUTH for that selection: preset generation and matrix
    mode both resolve through here (see :func:`matrix_task_set_name`), so they
    can never run different task sets for the same language.
    """
    if task_files is None:
        task_files = discover_localized_task_files()
    if include_identity_arm and f"{domain}_{suffix}_identity" in task_files:
        return f"{suffix}_identity"
    return suffix


def matrix_task_set_name(domain: str, lang: str) -> str:
    """The task-set name one matrix cell runs for ``lang``.

    Applies the SAME identity-variant selection as preset generation
    (:func:`resolve_main_task_suffix`), honoring the pack experiment block's
    ``include_identity_arm``.
    """
    from tau2.multilingual.loader import get_language_experiment

    experiment = get_language_experiment(lang, domain)
    suffix = resolve_main_task_suffix(
        domain,
        lang,
        include_identity_arm=(
            experiment.include_identity_arm if experiment is not None else True
        ),
    )
    return f"{domain}_{suffix}"


def _build_preset(language: str, experiment: ExperimentSpec) -> RunPreset:
    """One RunPreset from one pack.yaml ``experiments`` entry.

    The entry arrives already TYPED: the loader validated it as
    ``ExperimentSpec`` at pack-load time (and ``guardrails`` re-validates the
    same model at pack-validation time), so only cross-file problems remain
    here — a referenced task set that does not exist on disk.
    """
    domain = experiment.domain
    suffix = experiment.task_set_suffix
    task_files = discover_localized_task_files()

    # The domain's fixed subset, applied to every arm's full-stage task list.
    # A preset arm enumerates its tasks as --task-ids, which switches the
    # run-time subset default off, so the subset has to be applied HERE or a
    # preset would quietly be the only path that still scores on all 114.
    subset = resolve_subset(domain=domain, requested=SUBSET_AUTO)

    def task_ids(task_suffix: str) -> list[str]:
        name = f"{domain}_{task_suffix}"
        if name not in task_files:
            raise ValueError(
                f"experiments entry for '{language}' references task set "
                f"'{name}' but no {domain}_tasks_{task_suffix}.json exists "
                "under data/tau2/multilingual/"
            )
        with open(task_files[name]) as fp:
            ids = [task["id"] for task in json.load(fp)]
        if subset is None:
            return ids
        # Lenient on purpose: preset generation builds the table for EVERY
        # registered pack, so a language whose localized set does not yet
        # cover the subset must not break the other languages' presets.
        kept = subset.intersect_ids(ids)
        if len(kept) < subset.size:
            logger.warning(
                f"Task set '{name}' supplies {len(kept)} of the "
                f"{subset.size} tasks in subset '{subset.name}'; the "
                f"'{language}' preset arm will run the short list. Localize "
                "the missing tasks before treating its numbers as comparable."
            )
        return kept

    smoke_stem = experiment.smoke_task_stem
    main_suffix = resolve_main_task_suffix(
        domain,
        suffix,
        include_identity_arm=experiment.include_identity_arm,
        task_files=task_files,
    )
    main_ids = task_ids(main_suffix)
    # The English pack is the shared baseline preset. English would otherwise
    # fall back to substring matching, so force the LLM communicate judge on it
    # to match the metric every non-English run activates from its run language;
    # localized arms keep env empty and activate the judge automatically.
    main_env = {FORCE_LLM_COMMUNICATE_JUDGE_ENV: "1"} if language == "en" else {}
    arms = [
        PresetArm(
            name=experiment.main_arm_name,
            domain=domain,
            task_set_name=f"{domain}_{main_suffix}",
            task_ids=main_ids,
            smoke_task_ids=[f"{smoke_stem}_{main_suffix}"],
            # Bare language code: deterministic, seed-rotated per-task
            # round-robin among the pack personas, logged in
            # SpeechEnvironment.persona_id.
            user_persona_id=language,
            env=main_env,
        ),
    ]
    for extra in experiment.extra_arms:
        arms.append(
            PresetArm(
                name=extra.name,
                domain=domain,
                task_set_name=f"{domain}_{extra.task_set_suffix}",
                task_ids=task_ids(extra.task_set_suffix),
                smoke_task_ids=[f"{smoke_stem}_{extra.task_set_suffix}"],
                user_persona_id=language,
            )
        )
    return RunPreset(
        name=experiment.preset_name,
        description=experiment.description,
        audio_native_provider=experiment.audio_native_provider,
        speech_complexity=experiment.speech_complexity,
        max_steps_seconds=experiment.max_steps_seconds,
        seed=experiment.seed,
        arms=arms,
    )


def _generate_run_presets() -> dict[str, "RunPreset"]:
    """Build the full preset table, ISOLATING per-language failures.

    One broken experiments entry (a referenced task set that does not exist,
    a duplicate preset name) must not take down any other preset — including
    the same language's OTHER domains — so each entry is built in isolation,
    skipped with a loud warning, and `tau2 factory validate --lang <lang>`
    pinpoints the defect. (Malformed entries never get this far: the loader
    rejects them at pack load as ``ExperimentSpec`` validation errors.)
    """
    from tau2.multilingual.loader import get_all_language_experiments

    presets: dict[str, RunPreset] = {}
    for language, experiments in get_all_language_experiments().items():
        for experiment in experiments:
            try:
                preset = _build_preset(language, experiment)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    f"skipping run preset for language '{language}' domain "
                    f"'{experiment.domain}' (broken experiments entry — run "
                    f"`tau2 factory validate --lang {language}`): {exc}"
                )
                continue
            if preset.name in presets:
                logger.warning(
                    f"skipping run preset for language '{language}' domain "
                    f"'{experiment.domain}': duplicate preset name "
                    f"'{preset.name}' (already generated by another entry)"
                )
                continue
            presets[preset.name] = preset
    return presets


# Lazily-built, cached preset table. Deliberately NOT built at import time:
# building it loads every language pack, so an eager table would make
# ``import tau2.multilingual.run_presets`` fail or slow down on any data-dir
# problem. Tests reset the cache via ``_reset_presets_cache_for_tests``.
_PRESETS_CACHE: Optional[dict[str, RunPreset]] = None


def get_run_presets() -> dict[str, RunPreset]:
    """The generated preset table (built on first use, then cached)."""
    global _PRESETS_CACHE
    if _PRESETS_CACHE is None:
        _PRESETS_CACHE = _generate_run_presets()
    return dict(_PRESETS_CACHE)


def _reset_presets_cache_for_tests() -> None:
    """Drop the cached table (isolated test envs swap the pack data dir)."""
    global _PRESETS_CACHE
    _PRESETS_CACHE = None


def get_run_preset(name: str) -> RunPreset:
    presets = get_run_presets()
    if name not in presets:
        raise ValueError(f"Unknown run preset '{name}'. Options: {sorted(presets)}")
    return presets[name]

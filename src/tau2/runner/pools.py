# Copyright Sierra
"""Run pools — declarative grids of ordinary benchmark runs.

A **pool** is a (language x arm) grid run under one fixed set of conditions, where
an *arm* is a provider, reasoning effort, and optional pinned model — the
audio-native provider in a voice pool, an agent model in a text pool. Each cell becomes one typed
``RunConfig`` and the shared multiprocess runner executes the whole grid through
one controller.

The controller retries a whole simulation after an ``infrastructure_error`` and
records the final failure if both attempts fail. A 50-task x 2-trial cell
therefore always reaches 100 terminal results, normally all usable and rarely 99
usable plus one honest failure. Re-running the same pool later uses ordinary
resume semantics: the checkpoint drops that failure and emits only its unit.

Cells run side by side at conc 10 per worker, not one lane per provider.
A concurrency sweep (2026-07-28, via the since-retired ``tau2
bench-concurrency`` verb) measured the harness ceiling as per
process, not per ``--max-concurrency``: six processes at conc 10 sustained
626.1 sims/hr at 16ms tick overrun. The shared controller now provides those
processes and applies any provider caps across the whole grid.

The task selection is part of the pool's identity. A stratified subset and a
first-N prefix over the same task set are different task populations (the two
50-task telecom selections share only 20 tasks). Mixing them inside one results
dir silently changes what the numbers mean, so the selection is declared per
pool and recorded, never inferred from a flag at the call site.

Registered pools are the record of what was actually run: ``preference_v1``
(2400 simulations, 6 languages x 4 arms, stratified telecom), the earlier
``preference_v1_prefix`` first-50 pool kept as its own frame, ``airline_v1``
(the same 6 languages x 4 arms over all 50 airline tasks), ``retail_v1``
(the same grid over the stratified retail 50), and ``banking_text_en_v1``
(the first text pool: English banking_knowledge, agent-model arms over the
uniform banking 50). The three ``multilingual_text_*_v1`` pools form the
matched text control: the voice pools' exact 50-task frames, six languages,
and two frontier agent-model arms at pinned reasoning efforts.

That "record of what was run" is why the conditions are spelled per pool rather
than read from the config defaults at census time: the defaults move — the
conversation ceiling went to 2400 s on 2026-08-03 — and a pool whose spec
followed them would silently claim its stored simulations ran under a budget
they never saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from tau2.config import (
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_MATRIX_MAX_STEPS_SECONDS,
    DEFAULT_MATRIX_SEED,
    DEFAULT_MATRIX_SPEECH_COMPLEXITY,
    DEFAULT_MAX_STEPS,
    DEFAULT_MULTILINGUAL_RUN_CONCURRENCY,
    DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    VOICE_TIMEOUT_SAFETY_FACTOR,
)
from tau2.data_model.simulation import (
    AudioNativeConfig,
    NativenessJudgeSettings,
    RunConfig,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.utils import DATA_DIR

# Eight workers x conc 10 reproduces the 80-session envelope used to collect
# the existing pools, now through the generic multiprocess runner.
DEFAULT_POOL_WORKERS = 8

INFRASTRUCTURE_ERROR = "infrastructure_error"

# The conversation ceiling the ALREADY-COLLECTED pools ran under, pinned here
# now that DEFAULT_MATRIX_MAX_STEPS_SECONDS is 2400. A pool spec is not a
# preference, it is a record of the conditions its simulations hold: resuming
# preference_v1 or airline_v1 under a wider budget would put cells of two
# different regimes in one results.json, which is the same defect as mixing two
# task selections. New pools take the default; these three keep what they ran.
#
# The 62 ceiling-truncated telecom calls repaired on 2026-07-31 are the
# deliberate exception, and the reason `tau2 pool run --max-steps-seconds`
# exists: a repair that would only truncate again is not a repair. Widening the
# ceiling for a REFILL stays a per-invocation choice, recorded in the audit
# (docs/multilingual/pool_audit_2026-07-31.md), rather than a silent edit to
# what the pool claims its 2400 simulations ran under.
COLLECTED_POOL_MAX_STEPS_SECONDS = 1200
COLLECTED_POOL_TIMEOUT_SECONDS = int(
    COLLECTED_POOL_MAX_STEPS_SECONDS * VOICE_TIMEOUT_SAFETY_FACTOR
)


class PoolArm(BaseModel):
    """One column of the grid, at a pinned effort.

    In a VOICE pool ``provider`` is the audio-native provider. Effort is
    always pinned explicitly, including gemini's ``high`` — that is gemini's
    unpinned default, but pinning records ``reasoning_effort_source: run``
    instead of leaving it for a later backfill to infer. ``xhigh`` is
    OpenAI-only; the run CLI refuses the pair at config time.

    In a TEXT pool there is no audio provider: the arm is an agent model at a
    pinned effort. ``agent_llm`` carries the model id handed to ``--agent-llm``
    and ``provider`` is the short label used in cell names and result dirs
    (``gpt55``, ``claude``). The effort travels through ``--agent-llm-args``
    as a portable ``reasoning_effort`` — the LLM seam translates it per
    vendor (OpenAI Responses bridge takes it verbatim; Anthropic adaptive
    models map it to ``output_config.effort``, which accepts low/medium/high
    only).
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    reasoning_effort: str
    model: Optional[str] = None
    agent_llm: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.reasoning_effort}"


class TaskSelection(BaseModel):
    """Which tasks a pool's cells run, and how that choice is spelled to ``tau2 run``.

    ``subset`` names a frozen draw checked in under ``data/tau2/task_subsets/``
    (stratified, seeded, verifiable with ``tau2 tasks``). ``prefix`` takes the
    first N of the task set in file order. They are NOT interchangeable: the two
    50-task telecom selections overlap on 20 tasks, so a dir that holds one must
    never be resumed under the other.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["subset", "prefix"]
    subset: Optional[str] = None
    num_tasks: Optional[int] = None

    @model_validator(mode="after")
    def _one_selection(self) -> "TaskSelection":
        if self.kind == "subset" and not self.subset:
            raise ValueError("kind='subset' needs a subset name")
        if self.kind == "prefix" and not self.num_tasks:
            raise ValueError("kind='prefix' needs num_tasks")
        return self

    def describe(self) -> str:
        if self.kind == "subset":
            return f"stratified subset {self.subset}"
        return f"first {self.num_tasks} tasks (prefix)"

    @property
    def size(self) -> int:
        """Number of tasks selected for each ordinary run."""
        if self.kind == "prefix":
            return int(self.num_tasks)
        from tau2.task_subsets import load_subset

        return load_subset(str(self.subset)).size


class PoolSpec(BaseModel):
    """A pool: the grid, the target, and the fixed conditions every cell runs under.

    The conditions are the point. Two cells are comparable only if they differ
    in the arm and nothing else, so they live here as data rather than in a
    caller's flags — and the defaults are the shared ``DEFAULT_MATRIX_*``
    constants, so a pool cannot silently diverge from the rest of the
    multilingual experiment.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    domain: str
    # "voice" cells run --audio-native against an arm's realtime provider;
    # "text" cells run the half-duplex orchestrator against an arm's agent
    # model. The modality is part of the pool's identity, not a launch flag:
    # a voice cell and a text cell of the same grid are different populations.
    modality: Literal["voice", "text"] = "voice"
    # lang -> task set name. Explicit rather than resolved at runtime: the task
    # set is part of what the stored simulations ARE, and a pack that later opts
    # out of its identity variant must not silently repoint an existing pool.
    # test_pools.py asserts each entry still matches the matrix resolver.
    task_sets: dict[str, str]
    # lang -> results root under data/simulations/. Two prefixes appear in the
    # same pool by history: en/es/pt already had a first-50 pool under
    # preference_runs_v1_*, so their stratified cells went to
    # preference_strat50_*, while ko/zh (no earlier pool) kept preference_runs_v1_*.
    roots: dict[str, str]
    arms: tuple[PoolArm, ...]
    tasks: TaskSelection
    num_trials: int
    seed: int = DEFAULT_MATRIX_SEED
    max_concurrency: int = DEFAULT_MULTILINGUAL_RUN_CONCURRENCY
    max_steps_seconds: int = DEFAULT_MATRIX_MAX_STEPS_SECONDS
    timeout_seconds: int = DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS
    speech_complexity: str = DEFAULT_MATRIX_SPEECH_COMPLEXITY
    # Text cells only: the half-duplex turn cap, pinned per pool for the same
    # reason max_steps_seconds is pinned for voice — the default moves.
    max_steps: Optional[int] = None
    # Whether cells pass --user-persona-id <lang>. Localized task sets require
    # it (the run refuses them without a matching persona); a domain whose
    # tasks embed their own characters (banking_knowledge's "Sarah Bosch,
    # management consultant") runs WITHOUT pack personas — injecting one would
    # fight the task's own identity and the DB identities auth verifies.
    use_personas: bool = True
    nativeness_llm_judge: bool = False
    communicate_judge_mode: Literal["auto", "llm", "exact"] = "auto"

    @model_validator(mode="after")
    def _languages_agree(self) -> "PoolSpec":
        missing = set(self.task_sets) ^ set(self.roots)
        if missing:
            raise ValueError(
                f"task_sets and roots disagree on languages: {sorted(missing)}"
            )
        return self

    @model_validator(mode="after")
    def _arms_match_modality(self) -> "PoolSpec":
        for arm in self.arms:
            if self.modality == "text" and not arm.agent_llm:
                raise ValueError(
                    f"text pool {self.name!r}: arm {arm.key} has no agent_llm"
                )
            if self.modality == "voice" and arm.agent_llm:
                raise ValueError(
                    f"voice pool {self.name!r}: arm {arm.key} sets agent_llm; "
                    "voice arms name an audio-native provider, not an agent model"
                )
        return self

    @property
    def languages(self) -> list[str]:
        return list(self.task_sets)

    @property
    def target_per_cell(self) -> int:
        return self.tasks.size * self.num_trials

    @property
    def total_target(self) -> int:
        return self.target_per_cell * len(self.languages) * len(self.arms)

    def save_to(self, language: str, arm: PoolArm) -> str:
        """The ``--save-to`` value for one cell (a path under data/simulations/)."""
        cell = f"{language}_{self.domain}_{arm.provider}_{arm.reasoning_effort}"
        return f"{self.roots[language]}/{cell}"

    def cell_dir(self, language: str, arm: PoolArm) -> Path:
        return DATA_DIR / "simulations" / self.save_to(language, arm)

    def cell_name(self, language: str, arm: PoolArm) -> str:
        return f"{language}/{arm.provider}/{arm.reasoning_effort}"

    def cells(self) -> list[tuple[str, PoolArm]]:
        return [(lang, arm) for lang in self.languages for arm in self.arms]

    def cell_config(
        self, language: str, arm: PoolArm, *, redo_stale_tasks: bool = False
    ) -> RunConfig:
        """Build the typed ``tau2 run`` configuration for one cell.

        ``--auto-resume`` on every cell: a rerun of the same command fills gaps
        instead of redoing work, and without it a stale results.json triggers an
        interactive resume prompt that dies with EOFError under a driver.

        Text cells pin ``--results-format dir``: the census below reads the
        ``simulation_index`` only the dir format writes, and the annotation
        packet builders read the same layout, so one format serves both
        modalities.

        ``--auto-resume`` does NOT cover a task whose text changed under the
        cell: that raises, by design, because silently mixing two vintages of a
        task inside one cell is worse than stopping. ``redo_stale_tasks`` is the
        deliberate opt-in that drops those simulations and re-runs them against
        the current text. It is destructive, so it is never a pool default —
        pass it only when the whole pool is being rebased onto one text.
        """
        selection = (
            {"task_subset": self.tasks.subset}
            if self.tasks.kind == "subset"
            else {"num_tasks": self.tasks.num_tasks}
        )
        shared = dict(
            domain=self.domain,
            task_set_name=self.task_sets[language],
            **selection,
            user_persona_id=language if self.use_personas else None,
            seed=self.seed,
            num_trials=self.num_trials,
            max_concurrency=self.max_concurrency,
            timeout=self.timeout_seconds,
            verbose_logs=True,
            auto_resume=True,
            redo_stale_tasks=redo_stale_tasks,
            save_to=self.save_to(language, arm),
            nativeness_judge=NativenessJudgeSettings(
                llm_judge=self.nativeness_llm_judge
            ),
            communicate_judge_mode=self.communicate_judge_mode,
        )
        if self.modality == "voice":
            return VoiceRunConfig(
                **shared,
                audio_native_config=AudioNativeConfig(
                    provider=arm.provider,
                    model=arm.model or DEFAULT_AUDIO_NATIVE_MODELS[arm.provider],
                    reasoning_effort=arm.reasoning_effort,
                    max_steps_seconds=self.max_steps_seconds,
                ),
                speech_complexity=self.speech_complexity,
            )
        return TextRunConfig(
            **shared,
            llm_agent=str(arm.agent_llm),
            llm_args_agent={"reasoning_effort": arm.reasoning_effort},
            results_format="dir",
            max_steps=(
                self.max_steps if self.max_steps is not None else DEFAULT_MAX_STEPS
            ),
        )


class CellCensus(BaseModel):
    """What one cell holds right now, read from its checkpoint index."""

    usable: int
    infrastructure_errors: int
    duplicates: int
    tasks: int
    entries: int

    @property
    def is_complete(self) -> bool:
        return self.infrastructure_errors == 0 and self.duplicates == 0


def census_cell(spec: PoolSpec, language: str, arm: PoolArm) -> CellCensus:
    """Count a cell from ``results.json``'s ``simulation_index``.

    Counted as DISTINCT ``(task_id, trial)`` pairs, not simulations: a grafted
    cell that re-ran 20 simulations a resume failed to recognise held 120
    simulations for 100 pairs, and counting simulations called it 120/100
    complete while 20 of its 50 tasks carried double weight.

    Read from the index rather than the per-simulation files on purpose. The
    index already carries task_id, trial and termination_reason, and one cell's
    ``simulations/`` dir is ~750 MB of transcript — a full 20-cell census over
    the files parses ~15 GB and pins a core for minutes. These runs are
    GIL-bound and their tick drain runs to a wall-clock deadline, so a monitor
    that takes a core degrades the runs it is watching. The index is written
    incrementally as simulations land, so it stays current mid-run.
    """
    results = spec.cell_dir(language, arm) / "results.json"
    if not results.is_file():
        return CellCensus(
            usable=0, infrastructure_errors=0, duplicates=0, tasks=0, entries=0
        )
    try:
        index = json.loads(results.read_text()).get("simulation_index", []) or []
    except (json.JSONDecodeError, OSError):
        # Mid-write. Report nothing rather than a wrong number; the next round
        # rereads it.
        return CellCensus(
            usable=0, infrastructure_errors=0, duplicates=0, tasks=0, entries=0
        )

    seen: set[tuple] = set()
    infra = duplicates = 0
    tasks: set[str] = set()
    for entry in index:
        if entry.get("termination_reason") == INFRASTRUCTURE_ERROR:
            infra += 1
            continue
        pair = (entry.get("task_id"), entry.get("trial"))
        if pair in seen:
            duplicates += 1
            continue
        seen.add(pair)
        if entry.get("task_id"):
            tasks.add(entry["task_id"])
    return CellCensus(
        usable=len(seen),
        infrastructure_errors=infra,
        duplicates=duplicates,
        tasks=len(tasks),
        entries=len(index),
    )


def census_pool(spec: PoolSpec) -> dict[str, CellCensus]:
    """Every cell's census, keyed by ``<lang>/<provider>/<effort>``."""
    return {
        spec.cell_name(lang, arm): census_cell(spec, lang, arm)
        for lang, arm in spec.cells()
    }


# =============================================================================
# Registry
# =============================================================================

_LANG_NAMES = {
    "en": "english",
    "es": "spanish",
    "pt": "portuguese",
    "ko": "korean",
    "zh": "mandarin",
    "hi": "hindi",
}


def _telecom_task_sets(languages: list[str]) -> dict[str, str]:
    """``telecom_<lang>_identity`` per language; English has no identity variant.

    The identity variants carry the caller-identity task split (name-auth and
    phone-auth callers drawn from the seeded pool); English runs the plain set.
    """
    return {
        lang: "telecom_en" if lang == "en" else f"telecom_{lang}_identity"
        for lang in languages
    }


# The four arms of the preference pools: two providers x two reasoning efforts.
# `minimal` is NOT a symmetric treatment across providers — it roughly halves an
# OpenAI call's length while barely changing Gemini's — which is a finding, not a
# defect in the grid.
PREFERENCE_ARMS = (
    PoolArm(provider="openai", reasoning_effort="xhigh"),
    PoolArm(provider="openai", reasoning_effort="minimal"),
    PoolArm(provider="gemini", reasoning_effort="high"),
    PoolArm(provider="gemini", reasoning_effort="minimal"),
)

# The third provider, as its own one-column grid. The paper's pinned xAI 1.0
# condition sent no explicit reasoning effort, so it cannot join
# PREFERENCE_ARMS' provider x effort factorial. It is a provider contrast, not
# a fifth arm. Keep the historical model pin independent of the moving default.
XAI_ARMS = (
    PoolArm(
        provider="xai",
        reasoning_effort="provider_default",
        model="grok-voice-think-fast-1.0",
    ),
)

# The two STRONG arms of the preference factorial, for ablation pools that
# contrast conditions rather than efforts: one column per provider at its top
# effort. A deliberate subset of PREFERENCE_ARMS (same providers, same pinned
# efforts), so an ablation cell reads directly against the corresponding
# retail_v1 column with only the ablated condition differing.
DBSCRIPT_ARMS = (
    PoolArm(provider="openai", reasoning_effort="xhigh"),
    PoolArm(provider="gemini", reasoning_effort="high"),
)

# The English baseline's reward is only comparable to localized arms if all of
# them score communication with the same semantic judge.
PREFERENCE_COMMUNICATE_JUDGE_MODE = "llm"

# The four arms of the English banking TEXT pool: two agent models x their two
# reasoning extremes. The efforts are NOT a symmetric treatment across models —
# they are each model's own top and bottom setting, same as the voice grid's
# xhigh-vs-high asymmetry. gpt-5.5 on the text API spans none..xhigh (the
# API rejects voice's 'minimal': "Supported values are: 'none', 'low',
# 'medium', 'high', and 'xhigh'"), while Anthropic's adaptive output_config
# accepts exactly low/medium/high, so claude's extremes are high and low.
BANKING_TEXT_ARMS = (
    PoolArm(provider="gpt55", reasoning_effort="xhigh", agent_llm="gpt-5.5"),
    PoolArm(provider="gpt55", reasoning_effort="none", agent_llm="gpt-5.5"),
    PoolArm(provider="claude", reasoning_effort="high", agent_llm="claude-opus-4-8"),
    PoolArm(provider="claude", reasoning_effort="low", agent_llm="claude-opus-4-8"),
)

# The matched multilingual TEXT control. These are agent-model arms, not the
# realtime voice providers above. GPT-5.5 is deliberately run at its top
# reasoning setting; Gemini 3.1 Pro Preview supports low/medium/high thinking
# and is pinned at high so a provider-default change cannot move the arm.
MULTILINGUAL_TEXT_ARMS = (
    PoolArm(provider="gpt55", reasoning_effort="xhigh", agent_llm="gpt-5.5"),
    PoolArm(
        provider="gemini31pro",
        reasoning_effort="high",
        agent_llm="gemini/gemini-3.1-pro-preview",
    ),
)

_PREFERENCE_V1_LANGS = ["en", "es", "pt", "ko", "zh", "hi"]

_AIRLINE_V1_LANGS = ["en", "es", "pt", "ko", "zh", "hi"]

_RETAIL_V1_LANGS = ["en", "es", "pt", "ko", "zh", "hi"]

POOLS: dict[str, PoolSpec] = {
    "preference_v1": PoolSpec(
        name="preference_v1",
        description=(
            "Round-1 preference pool: 6 languages x 4 arms x 100 task/trial "
            "results on the stratified telecom_50 subset (2400 total)."
        ),
        domain="telecom",
        task_sets=_telecom_task_sets(_PREFERENCE_V1_LANGS),
        roots={
            # en/es/pt already held a first-50 prefix pool under
            # preference_runs_v1_*, kept as its own frame, so their stratified
            # cells live under preference_strat50_*. ko/zh/hi had no earlier pool.
            "en": "preference_strat50_english_telecom",
            "es": "preference_strat50_spanish_telecom",
            "pt": "preference_strat50_portuguese_telecom",
            "ko": "preference_runs_v1_korean_telecom",
            "zh": "preference_runs_v1_mandarin_telecom",
            "hi": "preference_runs_v1_hindi_telecom",
        },
        arms=PREFERENCE_ARMS,
        tasks=TaskSelection(kind="subset", subset="telecom_50"),
        num_trials=2,
        max_steps_seconds=COLLECTED_POOL_MAX_STEPS_SECONDS,
        timeout_seconds=COLLECTED_POOL_TIMEOUT_SECONDS,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "preference_v1_prefix": PoolSpec(
        name="preference_v1_prefix",
        description=(
            "The earlier first-50 preference pool (en/es/pt only). A DIFFERENT "
            "task population from preference_v1 — they share 20 of 50 tasks — "
            "kept as its own frame, not resumed into the stratified dirs."
        ),
        domain="telecom",
        task_sets=_telecom_task_sets(["en", "es", "pt"]),
        roots={
            lang: f"preference_runs_v1_{_LANG_NAMES[lang]}_telecom"
            for lang in ("en", "es", "pt")
        },
        arms=PREFERENCE_ARMS,
        tasks=TaskSelection(kind="prefix", num_tasks=50),
        num_trials=2,
        max_steps_seconds=COLLECTED_POOL_MAX_STEPS_SECONDS,
        timeout_seconds=COLLECTED_POOL_TIMEOUT_SECONDS,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "airline_v1": PoolSpec(
        name="airline_v1",
        description=(
            "The airline frame: 6 languages x 4 arms x 100 task/trial results "
            "over all 50 airline tasks, 2 trials each (2400 total). Same arms, "
            "judge mode and run conditions as preference_v1, so an "
            "airline-vs-telecom comparison is a domain contrast only. English is "
            "IN the pool — it is the baseline the localized arms are read "
            "against, and a baseline held in a separate pool is a baseline whose "
            "conditions nothing checks."
        ),
        domain="airline",
        # English runs the plain set (no identity variant to choose between); the
        # localized packs all carry the airline identity variant, so unlike
        # telecom there is no per-language opt-out to encode. The identity split
        # is what makes the caller a person with an account rather than a handle
        # recited from the prompt.
        task_sets={
            lang: "airline_en" if lang == "en" else f"airline_{lang}_identity"
            for lang in _AIRLINE_V1_LANGS
        },
        roots={
            # en's 400 cells were collected before the localized arms existed,
            # under the root the then-English-only pool used. Kept as it lies:
            # moving a run dir strands the absolute results_path recorded in
            # every feature table and manifest built from it.
            "en": "airline_en_v1",
            # The other five have run nothing yet, and they are the first airline
            # cells to measure the English-prose task regime (#516) rather than
            # the retired translated-prose one, so no earlier dir is resumable
            # into them.
            **{
                lang: f"airline_v1_{_LANG_NAMES[lang]}_airline"
                for lang in _AIRLINE_V1_LANGS
                if lang != "en"
            },
        },
        arms=PREFERENCE_ARMS,
        # All 50 airline tasks: the domain has exactly 50, so the prefix IS the
        # whole population and there is no stratified subset to disagree with.
        tasks=TaskSelection(kind="prefix", num_tasks=50),
        num_trials=2,
        max_steps_seconds=COLLECTED_POOL_MAX_STEPS_SECONDS,
        timeout_seconds=COLLECTED_POOL_TIMEOUT_SECONDS,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "retail_v1": PoolSpec(
        name="retail_v1",
        description=(
            "The retail frame: 6 languages x 4 arms x 100 task/trial results "
            "on the stratified retail_50 subset, 2 trials each (2400 total). "
            "Same arms, judge mode and grid as preference_v1 and airline_v1, so "
            "retail-vs-telecom and retail-vs-airline are domain contrasts. The "
            "one condition that differs is the conversation ceiling: retail "
            "runs at the current 2400 s default while the two earlier pools "
            "keep the 1200 s they were collected under. That difference is "
            "real and belongs in any cross-domain comparison — 1200 s cut 62 "
            "telecom calls mid-sentence and scored them 0 before they were "
            "refilled at 2400 s (pool_audit_2026-07-31)."
        ),
        domain="retail",
        # English runs the plain set; the five localized packs carry the retail
        # identity variant, which anchors the caller in prose (name + zip) —
        # retail authenticates by email OR name+zip, not by an account handle
        # the prompt recites.
        task_sets={
            lang: "retail_en" if lang == "en" else f"retail_{lang}_identity"
            for lang in _RETAIL_V1_LANGS
        },
        # Nothing retail has ever run at scale, so all six roots are new and
        # uniform — there is no earlier dir to stay compatible with.
        roots={
            lang: f"retail_v1_{_LANG_NAMES[lang]}_retail" for lang in _RETAIL_V1_LANGS
        },
        arms=PREFERENCE_ARMS,
        # Retail has 114 tasks, so unlike airline the prefix is NOT the
        # population: `--num-tasks 50` would census the early scenario families
        # and zero-sample the late ones. The canonical stratified draw instead.
        tasks=TaskSelection(kind="subset", subset="retail_50"),
        num_trials=2,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "retail_xai_v1": PoolSpec(
        name="retail_xai_v1",
        description=(
            "xAI on the retail frame: the same 6 languages, the same stratified "
            "retail_50 subset, the same 2 trials and 100 results per cell as "
            "retail_v1 — one arm instead of four, because xAI accepts no "
            "explicit reasoning effort (600 total). Held out of the preference "
            "pools, so this is a provider contrast against retail_v1's openai "
            "and gemini columns, not part of their factorial. Separate roots: "
            "any frame drawn off retail_v1 stays exactly what it was when it "
            "was drawn, and xAI is opt-in to a later one."
        ),
        domain="retail",
        task_sets={
            lang: "retail_en" if lang == "en" else f"retail_{lang}_identity"
            for lang in _RETAIL_V1_LANGS
        },
        roots={
            lang: f"retail_xai_v1_{_LANG_NAMES[lang]}_retail"
            for lang in _RETAIL_V1_LANGS
        },
        arms=XAI_ARMS,
        tasks=TaskSelection(kind="subset", subset="retail_50"),
        num_trials=2,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    # ------------------------------------------------------------------
    # The xai_v2 family: the pinned-model redo of the xAI column, one pool
    # per paper domain, 1 trial x the same 50-task selections as the
    # openai/gemini frames. What changed since retail_xai_v1 (2026-08-04/05)
    # and makes those cells non-comparable:
    #   - the model is PINNED to grok-voice-think-fast-1.0 via ?model=
    #     (#859) — the earlier run rode the endpoint default across the
    #     2026-08-05 1.0->2.0 flip;
    #   - the session now carries input.transcription.language_hint mapped
    #     from the run's language pack (XAI_LANGUAGE_HINTS), matching the
    #     language prior openai and gemini sessions always had;
    #   - telecom identity sets carry localized names but CANONICAL phone
    #     numbers (the localized number was invisible to
    #     get_customer_by_phone's line matching).
    # One trial, not two: this column is a provider contrast, not part of
    # the provider x effort factorial, and the budget goes to breadth.
    # ------------------------------------------------------------------
    "airline_xai_v2": PoolSpec(
        name="airline_xai_v2",
        description=(
            "xAI on the airline frame: 6 languages x 1 arm x 50 results over "
            "all 50 airline tasks, 1 trial (300 total), pinned to "
            "grok-voice-think-fast-1.0 with the session language hint set. "
            "Runs at the current default 2400s ceiling (owner's call, "
            "2026-08-27) — wider than airline_v1's collected 1200s."
        ),
        domain="airline",
        task_sets={
            lang: "airline_en" if lang == "en" else f"airline_{lang}_identity"
            for lang in _AIRLINE_V1_LANGS
        },
        roots={
            lang: f"airline_xai_v2_{_LANG_NAMES[lang]}_airline"
            for lang in _AIRLINE_V1_LANGS
        },
        arms=XAI_ARMS,
        tasks=TaskSelection(kind="prefix", num_tasks=50),
        num_trials=1,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "retail_xai_v2": PoolSpec(
        name="retail_xai_v2",
        description=(
            "xAI on the retail frame, redone pinned: 6 languages x 1 arm x 50 "
            "results on the stratified retail_50 subset, 1 trial (300 total), "
            "grok-voice-think-fast-1.0 with the session language hint set. "
            "Supersedes retail_xai_v1, whose cells rode the endpoint default "
            "across the 2026-08-05 model flip and carried no language prior. "
            "Same (default) ceilings as retail_v1."
        ),
        domain="retail",
        task_sets={
            lang: "retail_en" if lang == "en" else f"retail_{lang}_identity"
            for lang in _RETAIL_V1_LANGS
        },
        roots={
            lang: f"retail_xai_v2_{_LANG_NAMES[lang]}_retail"
            for lang in _RETAIL_V1_LANGS
        },
        arms=XAI_ARMS,
        tasks=TaskSelection(kind="subset", subset="retail_50"),
        num_trials=1,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "telecom_xai_v2": PoolSpec(
        name="telecom_xai_v2",
        description=(
            "xAI on the telecom frame: 6 languages x 1 arm x 50 results on "
            "the stratified telecom_50 subset, 1 trial (300 total), "
            "grok-voice-think-fast-1.0 with the session language hint set. "
            "Runs the canonical-phone identity sets (localized names, "
            "unlocalized numbers) — the first xAI telecom pool in this "
            "checkout, v2-numbered to travel with its airline/retail "
            "siblings. Runs at the current default 2400s ceiling (owner's "
            "call, 2026-08-27) — wider than preference_v1's collected 1200s."
        ),
        domain="telecom",
        task_sets=_telecom_task_sets(_PREFERENCE_V1_LANGS),
        roots={
            lang: f"telecom_xai_v2_{_LANG_NAMES[lang]}_telecom"
            for lang in _PREFERENCE_V1_LANGS
        },
        arms=XAI_ARMS,
        tasks=TaskSelection(kind="subset", subset="telecom_50"),
        num_trials=1,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    # ------------------------------------------------------------------
    # The romanization x localization 2x2 (hi/zh retail, retail_30 stems).
    # The four quadrants, on the two axes (DB name script x entity
    # localization):
    #   1. romanized DB + localized entities  = retail_v1's hi/zh cells,
    #      restricted to the retail_30 stems (no new pool needed);
    #   2. native-script DB + localized entities = retail_dbscript_v1;
    #   3. original English DB + original English entities, localized
    #      conversation = retail_unlocalized_v1;
    #   4. (romanized DB + original entities does not exist as data — the
    #      identity build IS the localization, so quadrant 1 is the
    #      romanized arm and quadrant 3 the unlocalized floor.)
    # retail_30 is the PREFIX of retail_50 in its recorded draw order, so
    # every cell here pairs task-for-task with retail_v1's cells on the
    # same stems. Everything else is pinned identical to retail_v1: seed,
    # personas, llm communicate judge, default 2400s ceiling. One trial and
    # the two strong arms only — this is a condition contrast, not another
    # provider x effort factorial.
    # ------------------------------------------------------------------
    "retail_dbscript_v1": PoolSpec(
        name="retail_dbscript_v1",
        description=(
            "Native-script DB quadrant of the romanization x localization "
            "2x2: hi/zh retail on the identity_native task sets, where the "
            "customer DB stores first/last names in Devanagari/Hanzi and the "
            "user simulator spells them natively (hi: akshara-by-akshara; "
            "zh: per-character glosses, never pinyin). 2 languages x 2 strong "
            "arms (openai/xhigh, gemini/high) x 30 tasks x 1 trial = 120. "
            "Pairs with retail_v1's hi/zh cells (romanized DB + localized "
            "entities) restricted to the same retail_30 stems."
        ),
        domain="retail",
        task_sets={lang: f"retail_{lang}_identity_native" for lang in ("hi", "zh")},
        roots={
            lang: f"retail_dbscript_v1_{_LANG_NAMES[lang]}_retail"
            for lang in ("hi", "zh")
        },
        arms=DBSCRIPT_ARMS,
        tasks=TaskSelection(kind="subset", subset="retail_30"),
        num_trials=1,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "retail_unlocalized_v1": PoolSpec(
        name="retail_unlocalized_v1",
        description=(
            "Unlocalized-entities quadrant of the romanization x localization "
            "2x2: hi/zh retail on the PLAIN localized sets (retail_hi, "
            "retail_zh) — conversation in the target language, but the "
            "customer DB and the caller's identity stay the original English "
            "entities (John Smith, not a localized persona). 2 languages x 2 "
            "strong arms (openai/xhigh, gemini/high) x 30 tasks x 1 trial = "
            "120. The floor the identity variants are read against, on the "
            "same retail_30 stems as retail_dbscript_v1 and retail_v1."
        ),
        domain="retail",
        task_sets={lang: f"retail_{lang}" for lang in ("hi", "zh")},
        roots={
            lang: f"retail_unlocalized_v1_{_LANG_NAMES[lang]}_retail"
            for lang in ("hi", "zh")
        },
        arms=DBSCRIPT_ARMS,
        tasks=TaskSelection(kind="subset", subset="retail_30"),
        num_trials=1,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "banking_text_en_v1": PoolSpec(
        name="banking_text_en_v1",
        description=(
            "The first TEXT pool: English banking_knowledge, 4 arms x 100 "
            "task/trial results on the uniform banking_knowledge_50 subset, "
            "2 trials each (400 total). Arms are agent models at their own "
            "reasoning extremes (gpt-5.5 xhigh/minimal, claude-opus-4-8 "
            "high/low) on the domain's default alltools retrieval variant. No "
            "pack personas: banking tasks embed their own characters and the "
            "DB identities auth verifies, so injecting a pack persona would "
            "fight both. No judge override either — banking's reward basis is "
            "DB/ACTION only (zero communicate_info in the frame), so there is "
            "no communicate judge to hold constant."
        ),
        domain="banking_knowledge",
        modality="text",
        task_sets={"en": "banking_knowledge"},
        roots={"en": "banking_text_v1_english_banking"},
        arms=BANKING_TEXT_ARMS,
        tasks=TaskSelection(kind="subset", subset="banking_knowledge_50"),
        num_trials=2,
        # The flat half-duplex wall-clock ceiling; voice's derived
        # steps*safety-factor arithmetic does not apply to a turn loop.
        timeout_seconds=int(DEFAULT_TIMEOUT_SECONDS),
        # Pinned so the pool records the turn cap its simulations ran under,
        # not whatever the default is when a cell is refilled later.
        max_steps=DEFAULT_MAX_STEPS,
        use_personas=False,
    ),
    "multilingual_text_airline_v1": PoolSpec(
        name="multilingual_text_airline_v1",
        description=(
            "Matched multilingual text control for airline: 6 languages x 2 "
            "frontier agent-model arms over all 50 airline tasks, one trial "
            "per task (600 total). The task frame, task sets, personas, and "
            "semantic communication judge match the airline voice pool."
        ),
        domain="airline",
        modality="text",
        task_sets={
            lang: "airline_en" if lang == "en" else f"airline_{lang}_identity"
            for lang in _AIRLINE_V1_LANGS
        },
        roots={
            lang: f"multilingual_text_v1_{_LANG_NAMES[lang]}_airline"
            for lang in _AIRLINE_V1_LANGS
        },
        arms=MULTILINGUAL_TEXT_ARMS,
        # Exact voice frame: airline_v1 uses the complete 50-task population.
        tasks=TaskSelection(kind="prefix", num_tasks=50),
        num_trials=1,
        timeout_seconds=int(DEFAULT_TIMEOUT_SECONDS),
        max_steps=DEFAULT_MAX_STEPS,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "multilingual_text_retail_v1": PoolSpec(
        name="multilingual_text_retail_v1",
        description=(
            "Matched multilingual text control for retail: 6 languages x 2 "
            "frontier agent-model arms on the voice pool's stratified "
            "retail_50 subset, one trial per task (600 total)."
        ),
        domain="retail",
        modality="text",
        task_sets={
            lang: "retail_en" if lang == "en" else f"retail_{lang}_identity"
            for lang in _RETAIL_V1_LANGS
        },
        roots={
            lang: f"multilingual_text_v1_{_LANG_NAMES[lang]}_retail"
            for lang in _RETAIL_V1_LANGS
        },
        arms=MULTILINGUAL_TEXT_ARMS,
        tasks=TaskSelection(kind="subset", subset="retail_50"),
        num_trials=1,
        timeout_seconds=int(DEFAULT_TIMEOUT_SECONDS),
        max_steps=DEFAULT_MAX_STEPS,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
    "multilingual_text_telecom_v1": PoolSpec(
        name="multilingual_text_telecom_v1",
        description=(
            "Matched multilingual text control for telecom: 6 languages x 2 "
            "frontier agent-model arms on the voice pool's stratified "
            "telecom_50 subset, one trial per task (600 total)."
        ),
        domain="telecom",
        modality="text",
        task_sets=_telecom_task_sets(_PREFERENCE_V1_LANGS),
        roots={
            lang: f"multilingual_text_v1_{_LANG_NAMES[lang]}_telecom"
            for lang in _PREFERENCE_V1_LANGS
        },
        arms=MULTILINGUAL_TEXT_ARMS,
        tasks=TaskSelection(kind="subset", subset="telecom_50"),
        num_trials=1,
        timeout_seconds=int(DEFAULT_TIMEOUT_SECONDS),
        max_steps=DEFAULT_MAX_STEPS,
        communicate_judge_mode=PREFERENCE_COMMUNICATE_JUDGE_MODE,
    ),
}


def list_pools() -> list[str]:
    return sorted(POOLS)


def get_pool(name: str) -> PoolSpec:
    try:
        return POOLS[name]
    except KeyError:
        raise SystemExit(
            f"unknown pool {name!r}. Registered: {', '.join(list_pools())}"
        ) from None

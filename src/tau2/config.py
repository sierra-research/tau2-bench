from enum import Enum

# =============================================================================
# SIMULATION DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 3
# Text calls are ordinary request/response LLM traffic and sustain a higher
# per-process ceiling than realtime voice sessions. Eight workers turns on the
# controller/checkpoint path and matches the standard text-run process fanout.
DEFAULT_TEXT_MAX_CONCURRENCY = 10
DEFAULT_TEXT_WORKERS = 8
# Noisy-text entity probes (tau2.multilingual.text_noise): default seed for
# the deterministic corruption draw. Pin --text-noise-seed explicitly when an
# experiment needs a different paired stimulus bank.
DEFAULT_TEXT_NOISE_SEED = 42
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "ERROR"
# Per-simulation wallclock cap for TEXT runs, where max_steps counts turns and
# carries no notion of duration, so this is the only duration bound there. A
# timed-out simulation terminates as `timeout` and --auto-resume can redo it.
# Voice runs do NOT use this value: they derive their guard from their own
# conversation budget — see VOICE_TIMEOUT_SAFETY_FACTOR.
DEFAULT_TIMEOUT_SECONDS = 2400.0

# PATHOLOGY GUARD, not a duration budget. A voice run's real duration budget is
# max_steps_seconds, measured in SIMULATED conversation time and therefore
# identical for every call. Wallclock is not: the same conversation costs more
# wall time when providers are slow, so a fixed wallclock cap silently converts
# provider latency into truncated conversations. Those truncations score 0.0 and
# land disproportionately on the longest calls, which are the hard ones — the
# same selection bias that kept the preference sampler (archived on the
# preference-archive branch) from filtering on call length at all.
#
# Measured on the 2026-07-26 en+pt telecom preference pools (629 simulations):
# a full-length call costs wall ~= -108 + 2.54 * sim_seconds, i.e. ~2940s at a
# 1200s budget and ~3500s worst case (max residual +557s). The old flat 2400s
# therefore bought only ~905s of pt conversation and truncated 15/200 pt sims
# while sparing English, purely because pt runs slower. Anchoring the guard to
# the budget at 5x keeps ~1.7x headroom over the worst full-length call, so it
# can only fire on a genuinely wedged session, while still bounding one.
VOICE_TIMEOUT_SAFETY_FACTOR = 5.0

# =============================================================================
# FIXED TASK SUBSETS (tau2 tasks — the frozen frame every run scores on)
# =============================================================================
# A benchmark run scores on 50 tasks per domain. Which 50 is a design decision,
# not a side effect of file order: `--num-tasks 50` takes the FIRST 50, and the
# domain task files are grouped by scenario family, so a prefix is a census of
# the early families and a zero-sample of the late ones. On telecom that cut
# every one of the 49 mms_issue tasks (43% of the 114-task pool) out of every
# run ever measured. The subsets under data/tau2/task_subsets/ replace the
# prefix with a seeded stratified draw, recorded as an explicit id list.
DEFAULT_TASK_SUBSET_SIZE = 50
# Shares the multilingual experiment seed (DEFAULT_MATRIX_SEED) on purpose:
# one number pins every sampling decision in the paper's runs.
DEFAULT_TASK_SUBSET_SEED = 42
# Domain -> the subset a run uses when --task-subset is left at "auto". A
# domain absent from this map has no canonical subset and runs its full task
# set. `telecom-workflow` is the same task pool under a different policy, so
# it must draw the same tasks or its numbers are not comparable to telecom's.
CANONICAL_TASK_SUBSETS: dict[str, str] = {
    "airline": "airline_50",
    "retail": "retail_50",
    "telecom": "telecom_50",
    "telecom-workflow": "telecom_50",
}
# Sentinel values for --task-subset / RunConfig.task_subset. Here rather than
# in tau2.task_subsets so the run config can default without importing it.
# 'auto' = the domain's canonical subset; 'all' = the whole task set, which is
# what a full-benchmark run (every telecom task, not the fixed 50) asks for.
SUBSET_AUTO = "auto"
SUBSET_ALL = "all"

# =============================================================================
# LLM DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "gpt-5.5"
DEFAULT_LLM_USER = "gpt-5.5"
DEFAULT_LLM_TEMPERATURE_AGENT = 0.0
DEFAULT_LLM_TEMPERATURE_USER = 0.0
# The user simulator reasons at xhigh so it tracks task instructions and entity
# round-trips reliably; overrides the fast `none` fallback below that other
# gpt-5* callers get (temperature is dropped for gpt-5*, see llm_utils).
DEFAULT_LLM_REASONING_EFFORT_USER = "xhigh"
DEFAULT_LLM_ARGS_AGENT = {"temperature": DEFAULT_LLM_TEMPERATURE_AGENT}
DEFAULT_LLM_ARGS_USER = {
    "temperature": DEFAULT_LLM_TEMPERATURE_USER,
    "reasoning_effort": DEFAULT_LLM_REASONING_EFFORT_USER,
}
DEFAULT_LLM_NL_ASSERTIONS = "gpt-5.4-mini-2026-03-17"
DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE = 0.0
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE}

# LLM judge for communicate_info checks. Used automatically for non-English
# runs (see tau2.evaluator.evaluator_communicate); English runs keep exact
# substring matching unless the judge is forced.
DEFAULT_LLM_COMMUNICATE_JUDGE = "gpt-5.4-mini-2026-03-17"
DEFAULT_LLM_COMMUNICATE_JUDGE_TEMPERATURE = 0.0
DEFAULT_LLM_COMMUNICATE_JUDGE_ARGS = {
    "temperature": DEFAULT_LLM_COMMUNICATE_JUDGE_TEMPERATURE
}
# "auto" keeps the language-aware policy: semantic judging for non-English,
# exact matching for English. Other modes explicitly pin the policy for a run.
DEFAULT_COMMUNICATE_JUDGE_MODE = "auto"
# Env var to force the LLM communicate judge regardless of run language.
# Used for the English baseline arm of multilingual experiments (see
# tau2.multilingual.run_presets), so English and non-English runs are scored
# with an identical metric. Lives here so the multilingual pillar never has
# to import the evaluator for a constant.
FORCE_LLM_COMMUNICATE_JUDGE_ENV = "TAU2_FORCE_LLM_COMMUNICATE_JUDGE"

# LLM judge for nativeness factors (register, honorifics, code-switching, …).
# ON by default — see NativenessJudgeSettings.llm_judge. GPT-5 series is a reasoning
# model: it takes `reasoning_effort` natively and rejects `temperature` (litellm
# drops it), so we pass medium reasoning and no temperature.
DEFAULT_LLM_NATIVENESS_JUDGE = "gpt-5.5"
DEFAULT_LLM_NATIVENESS_JUDGE_ARGS: dict = {"reasoning_effort": "medium"}

# Frozen τ-Multilingual combined-utterance-naturalness judge.  This is a
# paper-evidence configuration, deliberately separate from the general
# nativeness default above: calibration selected the xhigh arm and the paper
# replay must never inherit a cheaper ambient default.
DEFAULT_TAU_MULTI_NATURALNESS_JUDGE = "gpt-5.5"
DEFAULT_TAU_MULTI_NATURALNESS_JUDGE_ARGS: dict = {"reasoning_effort": "xhigh"}
DEFAULT_TAU_MULTI_NATURALNESS_CONCURRENCY = 100

# Frozen Hindi gender-agreement factor test. This counterfactual validation
# frame fixes the agent gender to male independently of the source run voice.
DEFAULT_TAU_MULTI_HI_GENDER_JUDGE = "gpt-5.5"
DEFAULT_TAU_MULTI_HI_GENDER_JUDGE_ARGS: dict = {"reasoning_effort": "medium"}
DEFAULT_TAU_MULTI_HI_GENDER_CONCURRENCY = 60

# Universal call-quality rubric.  The semantic judge is opt-in; deterministic
# factors are always evaluated when the quality axis is selected.
DEFAULT_LLM_QUALITY_JUDGE = "gpt-5.5"
# Per-factor defaults live in the closed rubric catalog. These are only global
# overrides, so an empty mapping preserves each factor's calibrated effort.
DEFAULT_LLM_QUALITY_JUDGE_ARGS: dict = {}
DEFAULT_LLM_QUALITY_FACTOR_ARGS: dict[str, dict[str, str]] = {
    "unnecessary_repetition": {"reasoning_effort": "none"},
    "agent_caused_tool_error": {"reasoning_effort": "high"},
    "auth_arg_mismatch": {"reasoning_effort": "high"},
    "incorrect_tool_parameters": {"reasoning_effort": "high"},
    "unnecessary_tool_call": {"reasoning_effort": "xhigh"},
}
DEFAULT_QUALITY_RESPONSE_LATENCY_SECONDS = 3.0
DEFAULT_QUALITY_MONOLOGUE_SECONDS = 45.0

# EVA-X conversation judges. Keep this separate from the universal quality judge:
# EVA-X has continuous/conjunctive semantics and is emitted as a sidecar artifact,
# not folded into SimulationRun.quality_info's binary-factor average.
DEFAULT_EVA_X_JUDGE = "gpt-5.2"
DEFAULT_EVA_X_JUDGE_ARGS: dict = {
    "reasoning_effort": "medium",
    "max_completion_tokens": 16000,
    "timeout": 600,
}
DEFAULT_CONVERSATION_JUDGE_CONCURRENCY = 8

# Multimodal AUDIO judge for delivery (fidelity + intonation) — listens to the
# agent's synthesized speech, unlike the text-only nativeness judge above. Uses
# Gemini 3.1 Pro via the Google AI Studio provider (GEMINI_API_KEY); audio is
# passed as an inline `input_audio` content part. Verified reachable id — note
# `gemini-3-pro-preview` (without the .1) 404s. OFF by default (voice-only, and
# adds real per-utterance Gemini cost); enable with --scores ...,delivery.
DEFAULT_LLM_DELIVERY_JUDGE = "gemini/gemini-3.1-pro-preview"
# max_tokens generous (matching sierra's 8192) so long findings lists never
# truncate mid-JSON, which would surface as a parse ERROR. timeout: a hung
# judge call must not stall a max_concurrency slot (litellm forwards it).
DEFAULT_LLM_DELIVERY_JUDGE_ARGS: dict = {
    "temperature": 0.0,
    "max_tokens": 8192,
    "timeout": 120,
}
# Cost control: fraction of CONVERSATIONS judged (deterministic per sim id;
# sampled sims get ALL their utterances judged so per-sim scores are
# trustworthy — sierra-style), plus a per-conversation utterance cap.
DEFAULT_DELIVERY_SAMPLE_RATE = 0.25
DEFAULT_DELIVERY_MAX_SEGMENTS = 100
# Concurrent judge calls per simulation (sierra runs 10; judge calls are ~5-15s
# each, so sequential execution would add minutes of wall-clock per sim).
DEFAULT_DELIVERY_JUDGE_CONCURRENCY = 8
# Concurrent per-sim judge calls in the streaming export/rejudge pipeline
# (tau2 judges rejudge/export --max-concurrency; I/O-bound litellm calls).
DEFAULT_JUDGE_STREAM_CONCURRENCY = 10
# Delivery coverage for `tau2 judges rejudge --delivery`: 1.0 = judge every
# conversation (a deliberate backfill fills all gaps, unlike the sampled
# runtime default above).
DEFAULT_REJUDGE_DELIVERY_SAMPLE_RATE = 1.0
# `tau2 judges suite` worker-process fan-out. The per-process concurrency
# ceiling is GIL-bound (same reason the run controller fans across worker
# processes), so the suite multiplies DEFAULT_JUDGE_STREAM_CONCURRENCY per
# process by a process count per phase: text (nativeness + quality) judges are
# lightweight requests, audio (delivery) payloads are heavier.
DEFAULT_JUDGE_SUITE_TEXT_PROCESSES = 8
DEFAULT_JUDGE_SUITE_AUDIO_PROCESSES = 5

# Canonical packet-aligned fidelity/intonation judge. The model settings are
# pinned to the calibrated high-reasoning arm; provider defaults are not used.
DEFAULT_LLM_AUDIO_QUALITY_JUDGE = "gemini/gemini-3.1-pro-preview"
DEFAULT_LLM_AUDIO_QUALITY_JUDGE_ARGS: dict = {
    "temperature": 0.0,
    "reasoning_effort": "high",
    "timeout": 600,
    "max_tokens": 32768,
}
DEFAULT_AUDIO_QUALITY_JUDGE_CONCURRENCY = 8

# Generator for AI candidate nuance rows in the Nativeness Audit sheet
# (tau2.annotation.nuance_candidates — fixed versioned prompt through the
# generate() seam). Opus with high adaptive-thinking effort: the stage runs
# once per language and the linguistic depth of the candidates is the product.
DEFAULT_NUANCE_CANDIDATES_MODEL = "claude-opus-4-8"
DEFAULT_NUANCE_CANDIDATES_MODEL_ARGS: dict = {"reasoning_effort": "high"}

# =============================================================================
# LANGUAGE FACTORY DEFAULTS (tau2 factory — overridable via CLI flags)
# =============================================================================
# The shared creative-drafting model (autoform, draft Calls A-D, the pack-block
# backfill verbs). Per-stage reasoning below; a stage without its own constant
# uses generate()'s per-model default.
DEFAULT_FACTORY_MODEL = "claude-opus-4-8"
# Autoform is a lighter creative task than drafting.
DEFAULT_FACTORY_AUTOFORM_REASONING = "medium"
DEFAULT_FACTORY_DRAFT_REASONING = "high"
# Per-language nativeness-rubric authoring is a one-shot linguistic task. The
# reviewed packet records this effective setting, so a model/default change
# cannot hide between language packs.
DEFAULT_FACTORY_NATIVENESS_REASONING = "high"
# Translation loop: translator/fixer on one vendor, the verifier on a
# DIFFERENT vendor (cross-vendor adversarial meaning-judge — a translator
# blind spot can't self-confirm).
DEFAULT_FACTORY_TRANSLATOR_MODEL = "gpt-5.4-2026-03-05"
DEFAULT_FACTORY_TRANSLATOR_REASONING = "medium"
DEFAULT_FACTORY_VERIFIER_MODEL = "claude-opus-4-8"
DEFAULT_FACTORY_VERIFIER_REASONING = "high"
# Acoustic-bed prompt author (outdoor SFX line + TV/kitchen recipe pieces).
DEFAULT_FACTORY_SFX_PROMPT_MODEL = "gpt-5.4-2026-03-05"
# Parity probe arms (paired text-mode runs; cheap models on purpose).
DEFAULT_FACTORY_PARITY_AGENT_MODEL = "gpt-5.4-mini-2026-03-17"
DEFAULT_FACTORY_PARITY_USER_MODEL = "gpt-5.4-mini-2026-03-17"
# Parity probe text-mode trials per task per arm (the noise bound in
# factory.parity scales as 1/sqrt(trials); 8 keeps the probe affordable).
DEFAULT_FACTORY_PARITY_TRIALS = 8
# Translation loop (translate -> verify -> fix): attempts per row before the
# row is flagged, and concurrent rows.
DEFAULT_FACTORY_TRANSLATE_MAX_ATTEMPTS = 3
DEFAULT_FACTORY_TRANSLATE_MAX_WORKERS = 8

# Matched-condition defaults every multilingual voice experiment arm pins:
# the generated run presets (multilingual.schema / run_presets), matrix mode
# (multilingual.run_preset_driver), and annotation's prompt-bed packets all
# share them. Deliberately EXPERIMENT-scoped, diverging from the global
# DEFAULT_SEED=300; seed 42 pins the matrix.
#
# The cap is SIMULATED conversation time, and it went 600 -> 1200 (2026-07-22:
# the hard end of the telecom difficulty dial — 3-4 stacked issues, Hard
# persona — was truncated at ~10 min of call audio in 40%+ of sims) -> 2400
# (2026-08-03).
#
# 1200 was measured to be binding, and to be biting the wrong calls. The
# 2026-07-31 preference-pool audit found 62 simulations terminated `max_steps`
# at exactly 1200 s, ALL of them reward 0, concentrated in telecom
# openai_xhigh (hi 9, pt 8, zh 8, es 4, en 3 per 100) — mid-sentence cuts, not
# failures. Re-run same-config at 2400 s, 0 of the 62 truncated again (30 ran
# past the old mark, longest 3541 s wall) and 15 of them scored above zero,
# mean 0.242. A ceiling that converts a sixth of the calls it touches from 0 to
# a real score was not measuring the agent, it was measuring itself — and it
# did so per-arm, so it also biased the arm contrast.
#
# Retail then makes it worse if left alone: its flows are the longest of the
# three domains — a return or exchange walks an order, its items and a payment
# method one at a time — so a ceiling fitted to telecom's shape is one nobody
# measured against retail's.
#
# What binds ABOVE this level is the provider, not us: OpenAI realtime has a
# documented hard cap of 60 min per session/WebSocket with no renewal API, and
# at wall ~= 2.5x simulated time, 60 min of wall is ~1400 s of budget. A call
# that genuinely needs the full 2400 s will drop its session on OpenAI rather
# than reach the ceiling. That surfaces as an infrastructure_error, which
# --auto-resume and `tau2 pool run` redo — so watch the infra-error count on a
# long-tailed domain: a task that cannot fit in 60 min of wall retries instead
# of settling.
DEFAULT_MATRIX_SPEECH_COMPLEXITY = "regular"
DEFAULT_MATRIX_MAX_STEPS_SECONDS = 2400
DEFAULT_MATRIX_SEED = 42
# Every preset run lands under this subdirectory of data/simulations/, so the
# multilingual experiment tree stays one subtree instead of scattering
# multilingual_v1_* dirs across the top level.
MULTILINGUAL_SIMULATIONS_SUBDIR = "multilingual"
DEFAULT_MULTILINGUAL_DOMAIN = "telecom"
DEFAULT_MULTILINGUAL_PROVIDERS = ("openai", "gemini")
DEFAULT_MULTILINGUAL_RUN_CONCURRENCY = 10
# Anchored to the step budget the matrix actually runs with, so the guard can
# never pre-empt the conversation it is guarding. See VOICE_TIMEOUT_SAFETY_FACTOR.
DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS = int(
    DEFAULT_MATRIX_MAX_STEPS_SECONDS * VOICE_TIMEOUT_SAFETY_FACTOR
)

# =============================================================================
# ANNOTATION FACTORY DEFAULTS (tau2 annotate — overridable via CLI)
# =============================================================================
# LIVE/shadow gate: a judge factor whose adjudicated precision clears this bar
# may score LIVE; anything below runs in shadow until recalibrated.
DEFAULT_ANNOTATION_PRECISION_BAR = 0.85
# calibration-packets (the held-out judge-calibration wave): defect-enriched
# draw size per language, the share of judge-clean control calls in it, and
# the draw seed. Owner sizing 2026-08-20: 30 double-annotated calls per
# language (down from the ~60 the paper protocol first asked for;
# papers/tau-multilingual/METHODS_AND_EXPERIMENTS_V2.md §4). The count is a
# first-class draw parameter (--n-calls); this is only its default.
DEFAULT_CALIBRATION_CALLS_PER_LANGUAGE = 30
DEFAULT_CALIBRATION_CONTROL_FRACTION = 0.25
DEFAULT_CALIBRATION_SEED = DEFAULT_MATRIX_SEED
# The adjudicate-mode per-factor draw: for every judge factor take
# min(cap, #flagged-calls) flagged calls — capped, never padded (owner ruling
# 2026-08-20: "max 20, if 20 don't exist, then whatever exists").
DEFAULT_CALIBRATION_ADJUDICATION_FACTOR_CAP = 20
# Transcript truncation per calibration-sheet row (precision / cold exports).
DEFAULT_ANNOTATION_MAX_TRANS_CHARS = 8000
# Communicate-judge calibration sample size (rows per export).
DEFAULT_ANNOTATION_JUDGE_SAMPLE_N = 50
# Agent-side transcript cap on communicate-judge calibration rows. Generous —
# the criterion-relevant turn must be present for a fair native verdict; rows
# note truncation explicitly when the transcript exceeds it.
DEFAULT_ANNOTATION_AGENT_EXCERPT_MAX_CHARS = 4000

# =============================================================================
# FEATURE-TABLE DEFAULTS (tau2 annotate feature-table)
# =============================================================================
# Tier-1 feature extraction (papers/tau-multilingual/METHODS_AND_EXPERIMENTS.md,
# "Finding the factors").
# A silence counts as "long" above this many seconds of neither side speaking.
DEFAULT_FEATURE_LONG_SILENCE_SECONDS = 3.0
# Consecutive agent speech segments merge into one "turn" when the gap between
# them is at most this long AND carries no floor-taking user speech.
DEFAULT_FEATURE_TURN_MERGE_GAP_SECONDS = 3.0
# Reference rates for backchannel_density_deviation, in continuers per minute
# of call, one per BackchannelLevel (tau2.backchannel) — the deviation is
# measured against the level the CALL'S LANGUAGE PACK declares, because native
# backchannel density is a language norm, not a constant. Derived from the
# knob's own density target (~1 continuer per N substantive sentences: low
# 3-4, medium 2-3, high 1-2), taking midpoints and scaling 1/N so that medium
# keeps the historical single-reference value of 1.0/min (~3-5 continuers over
# a ~4-5 min task call): low = 2.5/3.5, high = 2.5/1.5.
DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_LOW = 0.7
DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_MEDIUM = 1.0
DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_HIGH = 1.7
# Fallback reference rate for calls whose language has no pack (or a pack that
# declares no level) — e.g. an English control run predating the `en` pack.
# All of these are recorded in the feature-table provenance so a revision is
# traceable.
DEFAULT_FEATURE_HUMAN_BACKCHANNEL_PER_MIN = 1.0
DEFAULT_LLM_ENV_INTERFACE = "gpt-5.4-mini-2026-03-17"
DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE = 0.0
DEFAULT_LLM_ENV_INTERFACE_ARGS = {"temperature": DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE}

DEFAULT_LLM_EVAL_USER_SIMULATOR = "claude-opus-4-5"

# GPT-5 series are reasoning models and reject `temperature` (litellm drops it via
# drop_params). Default to no extra reasoning to match the fast, low-latency behavior
# of the previous gpt-4.1 non-reasoning defaults. Injected for gpt-5* in
# tau2.utils.llm_utils.generate when the caller does not set reasoning_effort.
DEFAULT_GPT5_REASONING_EFFORT = "none"

# LLM debug logging
DEFAULT_LLM_LOG_MODE = "latest"  # Options: "all", "latest"

# =============================================================================
# LLM INFRASTRUCTURE (fixed operational constants)
# =============================================================================
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_MIN_WAIT = 1.0  # seconds
DEFAULT_RETRY_MAX_WAIT = 10.0  # seconds
DEFAULT_RETRY_MULTIPLIER = 1.0  # exponential backoff multiplier

# LiteLLM cache
LLM_CACHE_ENABLED = False
DEFAULT_LLM_CACHE_TYPE = "redis"

# Redis (fixed infrastructure config)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""
REDIS_PREFIX = "tau2"
REDIS_CACHE_VERSION = "v1"
REDIS_CACHE_TTL = 60 * 60 * 24 * 30

# Langfuse
USE_LANGFUSE = False

# =============================================================================
# API SERVICE (fixed)
# =============================================================================
API_PORT = 8000

# =============================================================================
# AUDIO CONSTANTS (fixed, protocol-defined)
# =============================================================================
DEFAULT_PCM_SAMPLE_RATE = 16000  # User simulator synthesis rate
DEFAULT_TELEPHONY_RATE = 8000  # API/agent rate (8kHz μ-law, 1 byte/sample)
TELEPHONY_ULAW_SILENCE = b"\x7f"  # μ-law silence byte

# =============================================================================
# VOICE DEFAULTS (overridable via CLI, legacy half-duplex mode)
# =============================================================================
DEFAULT_VOICE_ENABLED = False
DEFAULT_VOICE_SYNTHESIS_PROVIDER = "elevenlabs"
DEFAULT_VOICE_TRANSCRIPTION_MODEL = "nova-3"
DEFAULT_VOICE_MODEL = "eleven_v3"

# Text streaming (legacy)
DEFAULT_TEXT_STREAMING_CHUNK_BY = "words"
DEFAULT_TEXT_STREAMING_CHUNK_SIZE = 1
DEFAULT_TEXT_STREAMING_CONFIG = {
    "chunk_by": DEFAULT_TEXT_STREAMING_CHUNK_BY,
    "chunk_size": DEFAULT_TEXT_STREAMING_CHUNK_SIZE,
}

# =============================================================================
# VOICE USER SIMULATOR (fixed versioning + overridable model)
# =============================================================================
VOICE_USER_SIMULATOR_VERSION = "v1.0"  # fixed, bump on changes
VOICE_USER_SIMULATOR_DECISION_MODEL = "gpt-5.4-mini"  # overridable
DEFAULT_SPEECH_COMPLEXITY = "regular"  # overridable: "control", "regular"

# =============================================================================
# FULL-DUPLEX VOICE DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_AUDIO_NATIVE_AGENT_IMPLEMENTATION = "discrete_time_audio_native_agent"
DEFAULT_AUDIO_NATIVE_USER_IMPLEMENTATION = "voice_streaming_user_simulator"
DEFAULT_AUDIO_NATIVE_PROVIDER = (
    "openai"  # overridable: openai, openai_live, gemini, xai, nova, qwen, livekit
)
DEFAULT_TICK_DURATION_SECONDS = 0.20  # overridable
# Overridable. 2400 aligns the bare `tau2 run` default with the matrix budget
# (DEFAULT_MATRIX_MAX_STEPS_SECONDS — its comment carries the 600 -> 1200 ->
# 2400 measurement history: 1200 was binding and truncated only reward-0
# mid-sentence cuts).
DEFAULT_MAX_STEPS_SECONDS = 2400
DEFAULT_SEND_AUDIO_INSTANT = False  # overridable

# Turn-taking thresholds (overridable, in seconds, converted to ticks at runtime)
DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS = 1.0
DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS = 5.0
DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS = 1.0
DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS = 5.0
DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS = 2.0
DEFAULT_INTEGRATION_DURATION_SECONDS = 0.5
DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS = 4.0
DEFAULT_USE_LLM_BACKCHANNEL = True

# Retry (overridable)
DEFAULT_AUDIO_NATIVE_MAX_RETRIES = 3
DEFAULT_AUDIO_NATIVE_RETRY_DELAY_SECONDS = 5.0

# =============================================================================
# ADAPTER TIMING (fixed operational constants)
# =============================================================================
DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS = 20  # fixed, standard RTP pacing
DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT = 30.0  # fixed
DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT = 5.0  # fixed
DEFAULT_AUDIO_NATIVE_THREAD_JOIN_TIMEOUT = 2.0  # fixed
DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER = 30.0  # fixed
DEFAULT_AUDIO_NATIVE_MAX_INACTIVE_SECONDS = 40.0  # fixed, stall detection

# =============================================================================
# OPENAI PROVIDER (overridable model/voice, fixed API constants)
# =============================================================================
DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime-2"  # overridable
DEFAULT_OPENAI_LIVE_MODEL = (
    "gpt-live-1-diamond-alpha"  # overridable, limited-access alias
)
_LEGACY_OPENAI_REALTIME_MODEL = "gpt-realtime-1.5"
DEFAULT_OPENAI_REALTIME_BASE_URL = "wss://api.openai.com/v1/realtime"  # fixed
DEFAULT_OPENAI_VOICE = "marin"  # overridable; pinned gendered (female) so the
# agent's gender is always known for the nativeness judge (alloy is neutral)
DEFAULT_OPENAI_NOISE_REDUCTION = "near_field"  # fixed: "near_field", "far_field", None
DEFAULT_OPENAI_VAD_THRESHOLD_LOW = 0.2  # fixed
DEFAULT_OPENAI_VAD_THRESHOLD_DEFAULT = 0.5  # fixed
DEFAULT_OPENAI_VAD_THRESHOLD = DEFAULT_OPENAI_VAD_THRESHOLD_DEFAULT
DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE = 24000  # fixed, API-defined
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"  # overridable
DEFAULT_WHISPER_MODEL = "whisper-1"  # fixed

# =============================================================================
# GEMINI PROVIDER (overridable model/voice, fixed API constants)
# =============================================================================
# Auth mode selected from environment:
#   GEMINI_API_KEY -> AI Studio
#   GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_APPLICATION_CREDENTIALS -> Vertex AI
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-live-preview"  # overridable
_LEGACY_GEMINI_MODEL = "gemini-live-2.5-flash-native-audio"
DEFAULT_GEMINI_VOICE = "Zephyr"  # overridable
DEFAULT_GEMINI_PROACTIVE_AUDIO = True  # fixed
DEFAULT_GEMINI_LOCATION = "us-central1"  # fixed
DEFAULT_GEMINI_INPUT_SAMPLE_RATE = 16000  # fixed, API-defined
DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE = 24000  # fixed, API-defined

# =============================================================================
# XAI PROVIDER (overridable voice, fixed API constants)
# =============================================================================
DEFAULT_XAI_REALTIME_BASE_URL = "wss://api.x.ai/v1/realtime"  # fixed
DEFAULT_XAI_VOICE = "Ara"  # overridable: Ara, Rex, Sal, Eve, Leo
# Pinned via the ?model= query param on connect. Never use the floating
# "grok-voice-latest" alias: unpinned runs silently track new releases.
DEFAULT_XAI_MODEL = "grok-voice-think-fast-1.0"  # overridable

# =============================================================================
# NOVA PROVIDER (overridable model/voice, fixed API constants)
# =============================================================================
DEFAULT_NOVA_MODEL = "amazon.nova-2-sonic-v1:0"  # overridable
DEFAULT_NOVA_VOICE = "tiffany"  # overridable: matthew, tiffany, amy
DEFAULT_NOVA_REGION = "us-east-1"  # fixed
DEFAULT_NOVA_INPUT_SAMPLE_RATE = 16000  # fixed, API-defined
DEFAULT_NOVA_OUTPUT_SAMPLE_RATE = 24000  # fixed, API-defined

# =============================================================================
# QWEN PROVIDER (overridable model/voice, fixed API constants)
# =============================================================================
DEFAULT_QWEN_REALTIME_URL = (
    "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"  # fixed
)
# Qwen3.5-Omni realtime models support tool calling over WebSocket
# (the older qwen3-omni-flash-realtime accepted tool configs but never
# invoked them). Flash variant: qwen3.5-omni-flash-realtime.
# Rate limits:
# qwen3.5-omni-plus-realtime; 60 (requests per minute); 100,000 (tokens per minute)
# qwen3.5-omni-plus-realtime-2026-03-15; 60 (requests per minute); 100,000 (tokens per minute)
DEFAULT_QWEN_MODEL = "qwen3.5-omni-plus-realtime"  # overridable
DEFAULT_QWEN_VOICE = "Tina"  # overridable; Qwen3.5-Omni-Realtime default voice
DEFAULT_QWEN_INPUT_SAMPLE_RATE = 16000  # fixed, API-defined
DEFAULT_QWEN_OUTPUT_SAMPLE_RATE = 24000  # fixed, API-defined

# =============================================================================
# PROVIDER REGISTRY (derived from above)
# =============================================================================
DEFAULT_AUDIO_NATIVE_MODELS = {
    "openai": DEFAULT_OPENAI_REALTIME_MODEL,
    "openai_live": DEFAULT_OPENAI_LIVE_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "xai": DEFAULT_XAI_MODEL,
    "nova": DEFAULT_NOVA_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
    "livekit": "dummy",
}


class ReasoningEffort(str, Enum):
    """The effective reasoning effort a run used.

    ``PROVIDER_DEFAULT`` is **not** a level: it records that no reasoning
    setting was sent to the provider at all, so whatever the server defaults to
    on that date applied. It must never be read as equivalent to a named level.

    The floor level is API-specific and the two floors are disjoint: the
    Realtime (voice) API accepts ``minimal`` and no ``none``, while the text
    API for the same model family accepts ``none`` and rejects ``minimal``
    outright. Both are members so a text arm's pinned floor round-trips
    through artifacts; the voice support tables below exclude ``NONE``.
    """

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    PROVIDER_DEFAULT = "provider_default"


# The effort each audio-native provider runs at when the CLI does not pin one.
# Resolved eagerly (see resolve_audio_native_reasoning_effort) so the effective
# value is recorded on the result rather than derived later inside the adapter.
#
# openai is PROVIDER_DEFAULT, and that is not shorthand for any level: it means
# no `reasoning` block is sent at all. OpenAI documents the levels as
# minimal|low|medium|high|xhigh "with low as the default", but unpinned is NOT
# interchangeable with pinning `low` — pinning imposes a hard token budget that
# the default does not. Probing gpt-realtime-2 over the Realtime websocket
# (text-only, reasoning tokens off `response.done`, N=10 on one hard
# combinatorics prompt) on 2026-07-25: pinned `minimal` = exactly 56 tokens
# 10/10, pinned `low` = exactly 518 tokens 10/10, UNPINNED = 3753..7250, median
# 5320 — an order of magnitude above pinned `low`, with zero overlap. So an
# unpinned openai run must be recorded as `provider_default`, never mapped onto
# a level; and pinning a level changes behaviour even when you pin the
# documented default.
DEFAULT_AUDIO_NATIVE_REASONING_EFFORT: dict[str, ReasoningEffort] = {
    "openai": ReasoningEffort.PROVIDER_DEFAULT,
    "openai_live": ReasoningEffort.PROVIDER_DEFAULT,
    "gemini": ReasoningEffort.HIGH,
    "xai": ReasoningEffort.PROVIDER_DEFAULT,
    "nova": ReasoningEffort.PROVIDER_DEFAULT,
    "qwen": ReasoningEffort.PROVIDER_DEFAULT,
    # livekit's effort lives on its cascaded LLM config, not this knob.
    "livekit": ReasoningEffort.PROVIDER_DEFAULT,
}


# What each provider will actually accept. The levels are NOT a shared
# vocabulary: `xhigh` is OpenAI-only, and Gemini Live rejects it at connect
# ("Invalid value at 'setup.generation_config.thinking_config.thinking_level'")
# — but the google-genai SDK builds the enum anyway from an unknown name behind
# a UserWarning nobody reads, so an unchecked `--reasoning-effort xhigh
# --audio-native-provider gemini` dies on an opaque websocket 1007 minutes into
# a run. Checked here instead, at config-construction time.
# xai/nova/qwen reject the knob outright; livekit takes its effort from the
# cascaded LLM config, so for all four only the "send nothing" sentinel is
# valid.
_OPENAI_EFFORTS = frozenset(ReasoningEffort) - {ReasoningEffort.NONE}
_GEMINI_EFFORTS = frozenset(_OPENAI_EFFORTS - {ReasoningEffort.XHIGH})
_NO_EFFORTS = frozenset({ReasoningEffort.PROVIDER_DEFAULT})
SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS: dict[str, frozenset[ReasoningEffort]] = {
    "openai": _OPENAI_EFFORTS,
    "openai_live": _NO_EFFORTS,
    "gemini": _GEMINI_EFFORTS,
    "xai": _NO_EFFORTS,
    "nova": _NO_EFFORTS,
    "qwen": _NO_EFFORTS,
    "livekit": _NO_EFFORTS,
}


# THE policy for an unresolved (``None``) reasoning effort, one rule applied in
# four places so the invariant has a single owner:
#
#   `None` means "the caller pinned nothing", and it is resolved exactly once,
#   at a CONSTRUCTION BOUNDARY — the CLI, `AudioNativeConfig`, or a direct
#   `DiscreteTimeAudioNativeAgent(...)`. What comes out of the boundary is a
#   value the run genuinely uses, so it is recorded with source `run`, never
#   `inferred`. BELOW the boundary nothing resolves: `create_adapter` takes an
#   already-resolved value and `require_resolved_reasoning_effort` refuses a
#   None, because a default applied down there would be applied after the
#   config was recorded and results.info would name an effort the run did not
#   send.
#
# `inferred` is reserved for values no run ever resolved: a stored `null` read
# back off disk (AudioNativeConfig's before-validator) or written by the
# now-retired `tau2 backfill-reasoning-effort` repair pass. A Python keyword
# `None` on a fresh construction is not a legacy null and must not be recorded
# as one.
def resolve_audio_native_reasoning_effort(
    provider: str, requested: "ReasoningEffort | str | None"
) -> ReasoningEffort:
    """The effective reasoning effort for ``provider``, validated against it.

    Idempotent: an already-resolved value is returned unchanged. ``None`` (no
    CLI pin) falls back to DEFAULT_AUDIO_NATIVE_REASONING_EFFORT. An effort the
    provider does not support raises here rather than at websocket connect.
    Call this at a construction boundary — never after the config has been
    recorded, and never below one (see require_resolved_reasoning_effort).
    """
    if provider not in DEFAULT_AUDIO_NATIVE_REASONING_EFFORT:
        raise ValueError(
            f"Unknown audio-native provider '{provider}': no default reasoning "
            f"effort. Known providers: "
            f"{sorted(DEFAULT_AUDIO_NATIVE_REASONING_EFFORT)}"
        )
    if requested is None:
        return DEFAULT_AUDIO_NATIVE_REASONING_EFFORT[provider]
    effort = ReasoningEffort(requested)
    supported = SUPPORTED_AUDIO_NATIVE_REASONING_EFFORTS[provider]
    if effort not in supported:
        raise ValueError(
            f"Provider '{provider}' does not support reasoning effort "
            f"'{effort.value}'. It accepts: "
            f"{sorted(e.value for e in supported)} "
            f"('provider_default' sends no reasoning setting at all)."
        )
    return effort


def require_resolved_reasoning_effort(
    provider: str, effort: "ReasoningEffort | str | None", *, caller: str
) -> ReasoningEffort:
    """Assert ``effort`` was already resolved by a construction boundary.

    The below-the-boundary half of the policy above: no default is applied
    here, because applying one at this depth happens after the run config has
    been recorded and leaves results.info claiming an effort the run never
    sent. A ``None`` is the caller's bug, and it is loud.

    Only the None case is judged; ``provider`` names the lane in the message
    and is not re-validated. Whether it accepts this particular level was
    settled at the boundary (``resolve_audio_native_reasoning_effort`` via
    ``AudioNativeConfig``) and is re-checked by the provider adapter that would
    have to send it, so re-deciding it here would only add a third wording of
    the same rejection.
    """
    if effort is None:
        raise ValueError(
            f"{caller} requires an already-resolved reasoning_effort for "
            f"provider '{provider}'; got None. Call "
            "tau2.config.resolve_audio_native_reasoning_effort() at "
            "config-construction time so the effective value is recorded on "
            "the result."
        )
    return ReasoningEffort(effort)


AUDIO_NATIVE_PROVIDER_TYPES = {
    "openai": "audio_native",
    "openai_live": "audio_native",
    "gemini": "audio_native",
    "xai": "audio_native",
    "nova": "audio_native",
    "qwen": "audio_native",
    "livekit": "cascaded",
}

# =============================================================================
# DISPLAY
# =============================================================================
TERM_DARK_MODE = True

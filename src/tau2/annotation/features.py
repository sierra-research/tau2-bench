# Copyright Sierra
"""Deterministic per-call feature extraction (``tau2 annotate feature-table``).

Builds the feature table: one row per saved simulation, deterministic
features computed offline from tick timestamps, ``contains_speech`` flags,
``turn_taking_action`` events, and inline tool calls/results. Exact, free, no
judge. The judged columns are slots here and are filled from stored judge
verdicts by ``attach_judged_features``.

Extraction is deliberately wide: it is a one-time pass over a pool, so
nothing is left out for being unused today — see ``EXTRACTED_FEATURES``.

Language-relative references: where a feature compares against a human norm
rather than a raw count, the norm is the CALL'S LANGUAGE norm, not a global
one — ``backchannel_density_deviation`` measures against the rate the
language pack's ``BackchannelLevel`` declares (see
``resolve_backchannel_reference_per_min``).

Time base: features are computed on the DISCRETE tick clock
(``tick index × tick_duration_seconds``), not wall-clock timestamps — the
composed call audio an annotator hears is exactly that clock, while wall time
includes compute stalls that are not part of the rendered call.

Judged features (nativeness, phrasing, intonation, fidelity, repetition) have
NAMED NULLABLE SLOTS here but are never computed by this extractor — they are
filled from a sim's stored judge verdicts by ``attach_judged_features``.

ARM IDENTITY. A row records ``provider``, ``agent_model`` AND
``reasoning_effort``, because all three together are what distinguishes one
benchmark configuration from another (``Arm``). Effort is not decoration: the
English pool is a 2x2 of provider x reasoning effort, and both openai cells
report the same ``agent_model`` string, so dropping effort would type the
study's primary systematic contrast as a stochastic rerun. Extraction REFUSES
an audio-native run whose effort was never recorded rather than defaulting it
— see ``_resolve_reasoning_effort``.

TEXT RUNS. A run with no ``audio_native_config`` is a half-duplex text run;
its rows carry ``modality="text"``, an empty ``provider``, and a reasoning
effort read from ``agent_info.llm_args`` (recorded verbatim by the runner, so
— unlike the voice config's in-memory default — an absent key truthfully
means "not pinned" and never needs the refusal path). Text sims have no ticks
BY DESIGN, so the quality gate requires messages instead, and the perceptual
timing features stay None while four message-based economy features
(``turn_count`` plus the ``agent_msg_len_p90`` / ``agent_char_ratio`` /
``formatting_density`` trio) are computed from the transcript.
"""

import hashlib
import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Iterable, Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from tau2.annotation.artifacts import git_sha, write_json_artifact
from tau2.annotation.loading import discover_results_files
from tau2.backchannel import BackchannelLevel
from tau2.config import (
    DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_HIGH,
    DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_LOW,
    DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_MEDIUM,
    DEFAULT_FEATURE_HUMAN_BACKCHANNEL_PER_MIN,
    DEFAULT_FEATURE_LONG_SILENCE_SECONDS,
    DEFAULT_FEATURE_TURN_MERGE_GAP_SECONDS,
    ReasoningEffort,
)
from tau2.data_model.message import Tick, ToolCall
from tau2.data_model.simulation import (
    AudioNativeConfig,
    Info,
    ReasoningEffortSource,
    Results,
    SimulationRun,
    TerminationReason,
    resolve_tick_duration_seconds,
)
from tau2.utils.utils import get_now

#: 3.0.0 (was 2.0.0): ``auth_arg_mismatch`` counts a different set of results.
#: A no-match identity lookup reaches the transcript by TWO transports — an
#: empty non-error result and a not-found ERROR (see ``AuthLookupOutcome``) —
#: and 2.0.0 only recognized the first, so it scored zero mismatches for every
#: auth tool that raises: both of airline's, both of retail's, and two of
#: telecom's three. Major, not minor: a 2.x table's ``auth_arg_mismatch``
#: column is a systematic undercount rather than a missing one, so constants
#: fitted on it are wrong instead of absent, and no reader could tell from the
#: schema. Retail also enters the auth seam at this version.
#:
#: 3.1.0: text-mode extraction. Three message-based economy columns
#: (``agent_msg_len_p90``, ``agent_char_ratio``, ``formatting_density``) and
#: ``turn_count`` now fill on tick-less sims; the sim-quality gate becomes
#: modality-aware (text sims gate on messages, not ticks); rows and source
#: runs carry ``modality``; banking_knowledge enters the auth seam (its
#: finders signal no-match on a THIRD transport, a non-error "No records
#: found" in prose). Minor, not major: every 3.0.0 column keeps its meaning
#: and its values on voice rows.
FEATURE_EXTRACTOR_VERSION = "3.1.0"

# ---------------------------------------------------------------------------
# Per-domain auth-tool seam
# ---------------------------------------------------------------------------

# Identity-verification tools per domain: the tools whose job is to turn what
# the caller SAID into a record. Add a domain here to light up time_to_auth /
# auth_attempt_count / auth_arg_mismatch for it.
#
# The three pool domains authenticate on three different shapes, which is the
# point of measuring across them:
#   * telecom — both sides of the name-DOB / phone split, plus direct
#     customer-id lookup.
#   * airline — a user id, and here ``get_user_details`` IS the auth step: the
#     id is the credential and the lookup is what verifies it.
#   * retail — name+zip or email, spoken aloud in prose (CallerIdentityKind.
#     PROSE_NAME_ZIP). ``get_user_details`` is NOT in retail's set even though
#     retail has a tool by that name: it takes a ``user_id`` the agent can only
#     have obtained from one of the two finders, so it is a post-identification
#     lookup. The policy is explicit that locating the id via email or
#     name+zip "has to be done even when the user already provides the user
#     id" — so a caller-supplied id does not authenticate anyone, and counting
#     the lookup that consumes it would book auth as succeeding before it did.
#   * banking_knowledge — the text pool's domain. Verification is 2-of-4
#     fields (DOB, email, phone, address) compared against the record one of
#     the three finders fetched from what the customer stated (an id, a name,
#     an email), so the finders are the auth step: a mis-stated identity comes
#     back as a no-match. ``log_verification`` is NOT in the set — it is the
#     write-side AUDIT record of a verification already done, a post-
#     identification write for the same reason retail's ``get_user_details``
#     is a post-identification read.
AUTH_TOOLS_BY_DOMAIN: dict[str, frozenset[str]] = {
    "telecom": frozenset(
        {"get_customer_by_phone", "get_customer_by_name", "get_customer_by_id"}
    ),
    "airline": frozenset({"get_user_details"}),
    "retail": frozenset({"find_user_id_by_name_zip", "find_user_id_by_email"}),
    "banking_knowledge": frozenset(
        {
            "get_user_information_by_id",
            "get_user_information_by_name",
            "get_user_information_by_email",
        }
    ),
}

# A non-error auth-tool result whose content is one of these is a lookup that
# found NOBODY (e.g. telecom get_customer_by_name returns [] on no match) —
# not a successful identification.
_EMPTY_RESULT_CONTENTS = frozenset({"", "[]", "null", "None"})

# The third transport of "found nobody": a NON-error result that says so in
# prose. Banking's finders route through query_database_tool, which returns
# "No records found in 'users'." with error=False on a miss. Anchored at the
# start deliberately — a successful lookup opens with "Found N record(s)" and
# must never trip this.
_NO_RECORDS_RESULT = re.compile(r"^no records found", re.IGNORECASE)

# The other transport a "found nobody" arrives on. Every auth tool outside
# telecom's get_customer_by_name signals no-match by RAISING — retail's two
# finders raise ``ValueError("User not found")``, airline's get_user_details
# and telecom's phone/id lookups raise their own not-found — and
# Environment.get_response turns any raised exception into
# ``ToolMessage(content=f"Error: {e}", error=True)``. So the same event, an
# argument matching no record, is an empty result in one domain and an error
# string in the next.
#
# Matching the message text is a real coupling to the domain tools, and it is
# the narrowest one available: the alternative is a structured error type,
# which would change the text the AGENT sees mid-study and invalidate every
# recorded run. What keeps the pattern honest is not this comment but
# ``test_every_registered_auth_tool_signals_no_match_recognizably``, which
# calls each registered auth tool with arguments matching nothing and asserts
# the classification — so a domain that changes its wording fails a test here
# instead of silently zeroing the column.
_NOT_FOUND_ERROR = re.compile(r"not\s+found", re.IGNORECASE)


class AuthLookupOutcome(str, Enum):
    """What one identity-verification tool result actually says.

    Three outcomes, and the middle one is the whole reason the enum exists.
    A bool ("did it succeed?") collapses NO_MATCH into TOOL_FAILURE, and those
    are opposite readings of a call: a not-found means the agent HEARD wrong,
    a tool failure means the harness broke and says nothing about hearing.
    """

    IDENTIFIED = "identified"
    NO_MATCH = "no_match"
    TOOL_FAILURE = "tool_failure"


def classify_auth_lookup(content: Optional[str], error: bool) -> AuthLookupOutcome:
    """Classify one identity-tool result.

    ``NO_MATCH`` is the mis-hearing signal, and it arrives three ways: a
    clean empty result, a non-error "no records found" in prose (banking),
    or a not-found error. ``TOOL_FAILURE`` is every other error — a bad
    argument shape, a harness fault — and is deliberately NOT counted as a
    mismatch, because it carries no evidence about what the agent heard.
    """
    text = (content or "").strip()
    if not error:
        return (
            AuthLookupOutcome.NO_MATCH
            if text in _EMPTY_RESULT_CONTENTS or _NO_RECORDS_RESULT.match(text)
            else AuthLookupOutcome.IDENTIFIED
        )
    return (
        AuthLookupOutcome.NO_MATCH
        if _NOT_FOUND_ERROR.search(text)
        else AuthLookupOutcome.TOOL_FAILURE
    )


# The user simulator's turn-taking action literal marking a continuer
# (see BasicActionType in tau2.agent.base.streaming).
_BACKCHANNEL_ACTION = "backchannel"

# A user speech segment is classified as a backchannel utterance when a
# backchannel decision fired within it or within this many ticks before its
# onset (synthesis can lag the decision tick).
_BACKCHANNEL_LOOKBACK_TICKS = 5

# Expected native continuer rate (per minute of call) for each density level a
# language pack can declare. Closed catalog over BackchannelLevel — coverage-
# guarded in tests, so a new level cannot ship without a reference rate.
BACKCHANNEL_REFERENCE_PER_MIN: dict[BackchannelLevel, float] = {
    BackchannelLevel.LOW: DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_LOW,
    BackchannelLevel.MEDIUM: DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_MEDIUM,
    BackchannelLevel.HIGH: DEFAULT_FEATURE_BACKCHANNEL_PER_MIN_HIGH,
}

# Terminations that fail the sim-quality gate (recorded per row as
# ``gate_passed`` / ``gate_reasons``).
GATE_FAIL_TERMINATIONS = frozenset(
    {
        TerminationReason.TOO_MANY_ERRORS,
        TerminationReason.AGENT_ERROR,
        TerminationReason.USER_ERROR,
        TerminationReason.INFRASTRUCTURE_ERROR,
        TerminationReason.CONTEXT_WINDOW_EXCEEDED,
        TerminationReason.UNEXPECTED_ERROR,
    }
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class CallFeatures(BaseModel):
    """Named feature values for one call. All floats; None = not extractable
    (tick-less sim, no auth tools for the domain, judged slot not yet filled).
    """

    # --- Tier 1: latency & silence ---
    response_latency_p50: Annotated[
        Optional[float],
        Field(
            description="Median response latency (s) over exchanges: last tick "
            "with user speech -> first subsequent tick with agent speech."
        ),
    ] = None
    response_latency_p90: Annotated[
        Optional[float], Field(description="p90 response latency (s).")
    ] = None
    long_silence_count: Annotated[
        Optional[float],
        Field(
            description="Silences longer than the configured threshold "
            "(default 3 s) with neither side speaking."
        ),
    ] = None
    long_silence_count_per_min: Annotated[
        Optional[float],
        Field(
            description="``long_silence_count`` per minute of call. The rate, "
            "not the count, is what a caller experiences: four dead pauses in "
            "a two-minute call is a different call from four in ten minutes."
        ),
    ] = None
    dead_air_ratio: Annotated[
        Optional[float],
        Field(description="Total no-speech time / call duration."),
    ] = None
    # --- Tier 1: turn-taking ---
    agent_interruption_count: Annotated[
        Optional[float],
        Field(
            description="Agent speech onsets while floor-taking user speech "
            "is in progress."
        ),
    ] = None
    user_barge_in_count: Annotated[
        Optional[float],
        Field(
            description="Floor-taking (non-backchannel) user speech onsets "
            "while agent speech is in progress. Sim-policy-driven but "
            "triggered by agent behavior — an impatience proxy."
        ),
    ] = None
    overlap_time_total: Annotated[
        Optional[float], Field(description="Seconds of simultaneous speech.")
    ] = None
    overlap_time_total_per_min: Annotated[
        Optional[float],
        Field(
            description="``overlap_time_total`` per minute of call — talking "
            "over each other twice a minute, not twelve seconds in total."
        ),
    ] = None
    backchannel_count: Annotated[
        Optional[float],
        Field(description="User backchannel events (turn-taking decisions)."),
    ] = None
    backchannel_density_deviation: Annotated[
        Optional[float],
        Field(
            description="|observed - native-reference| backchannels per minute "
            "of call, where the reference is the rate expected for the CALL'S "
            "LANGUAGE (its pack's BackchannelLevel; the configured fallback "
            "when the language has no pack). Sim-instrument control as much as "
            "a feature."
        ),
    ] = None
    # --- Tier 1: task-flow friction ---
    time_to_auth: Annotated[
        Optional[float],
        Field(
            description="Call start -> first successful identity-verification "
            "tool result (s). None when auth never succeeded or the domain "
            "has no registered auth tools."
        ),
    ] = None
    auth_attempt_count: Annotated[
        Optional[float],
        Field(
            description="Identity tool calls up to AND INCLUDING the first "
            "successful one (so a first-try success counts 1); total identity "
            "calls when none succeeded."
        ),
    ] = None
    auth_arg_mismatch: Annotated[
        Optional[float],
        Field(
            description="Identity-tool calls that identified nobody — the "
            "agent supplied arguments matching no record. This is the "
            "mis-hearing feature: a name, a date of birth, a phone number, a "
            "zip code or an email address heard wrong comes back as a "
            "not-found. It arrives on two transports and both count — an "
            "empty non-error result (telecom's name+DOB lookup) and a "
            "not-found ERROR (every other auth tool in the pool, which "
            "raises) — see ``classify_auth_lookup``. A tool failure that is "
            "not a not-found does NOT count: it says nothing about what the "
            "agent heard. None when the domain has no auth tools."
        ),
    ] = None
    agent_tool_calls: Annotated[
        Optional[float],
        Field(description="Total agent tool calls over the call."),
    ] = None
    time_to_resolution: Annotated[
        Optional[float], Field(description="Total call duration (s).")
    ] = None
    tool_error_count: Annotated[
        Optional[float],
        Field(description="Agent tool calls that returned an error."),
    ] = None
    repeated_tool_call_count: Annotated[
        Optional[float],
        Field(
            description="Agent tool calls repeating an earlier call with the "
            "same tool and identical arguments (spin)."
        ),
    ] = None
    # --- Tier 1: speech economy ---
    agent_talk_ratio: Annotated[
        Optional[float],
        Field(
            description="Agent speech time / total speech time (monologue detector)."
        ),
    ] = None
    agent_turn_len_p90: Annotated[
        Optional[float],
        Field(description="p90 agent turn length (s); rambling detector."),
    ] = None
    turn_count: Annotated[
        Optional[float],
        Field(
            description="Total exchanges. Voice: user turn followed by an "
            "agent response — the same events response latency is measured "
            "on. Text: agent messages carrying content (half-duplex "
            "alternation makes each one a response)."
        ),
    ] = None
    # --- Tier 1: text economy (message-based; None on voice sims) ---
    agent_msg_len_p90: Annotated[
        Optional[float],
        Field(
            description="p90 agent message length in characters — the "
            "wall-of-text detector, text's analog of ``agent_turn_len_p90``."
        ),
    ] = None
    agent_char_ratio: Annotated[
        Optional[float],
        Field(
            description="Agent content characters / total content characters "
            "(agent + user) — the monologue detector, text's analog of "
            "``agent_talk_ratio``."
        ),
    ] = None
    formatting_density: Annotated[
        Optional[float],
        Field(
            description="Fraction of non-empty agent message lines that are "
            "block-formatted (bullets, numbered lists, headings, tables, "
            "code fences, blockquotes). A support chat delivered as a "
            "document is a different experience from one delivered as "
            "conversation, whichever way a fit ends up signing it."
        ),
    ] = None
    # --- Judged slots (never computed here, never sampled) ---
    # Filled from a sim's stored judge verdicts by ``attach_judged_features``.
    # One slot per judged AXIS, not per factor: the nativeness judge scores
    # many per-language factors and they enter the fit as one number, because
    # a per-factor column would be mostly empty (a language only carries its
    # own factors) and no experiment reads factors individually.
    judged_nativeness: Annotated[
        Optional[float],
        Field(description="Nativeness judge; agent speech sounds native."),
    ] = None
    judged_phrasing: Annotated[
        Optional[float],
        Field(description="How a native speaker would have put it."),
    ] = None
    judged_intonation: Annotated[
        Optional[float],
        Field(description="Delivery judge; stress, tone, prosody."),
    ] = None
    judged_fidelity: Annotated[
        Optional[float],
        Field(description="Delivery judge; the words spoken are the words meant."),
    ] = None
    judged_repetition: Annotated[
        Optional[float],
        Field(description="Agent restating what it already said."),
    ] = None


# EXTRACTED_FEATURES is everything the extractor computes. It is deliberately
# wide: extraction is a one-time pass over the pool and a feature dropped here
# can only be recovered by re-running it, so nothing is left out for being
# unused today. Guarded by a coverage test against the CallFeatures model, so
# it cannot drift from the schema.
EXTRACTED_FEATURES: tuple[str, ...] = (
    "response_latency_p50",
    "response_latency_p90",
    "long_silence_count",
    "long_silence_count_per_min",
    "dead_air_ratio",
    "agent_interruption_count",
    "user_barge_in_count",
    "overlap_time_total",
    "overlap_time_total_per_min",
    "backchannel_count",
    "backchannel_density_deviation",
    "time_to_auth",
    "auth_attempt_count",
    "auth_arg_mismatch",
    "agent_tool_calls",
    "time_to_resolution",
    "tool_error_count",
    "repeated_tool_call_count",
    "agent_talk_ratio",
    "agent_turn_len_p90",
    "turn_count",
    "agent_msg_len_p90",
    "agent_char_ratio",
    "formatting_density",
)
JUDGED_FEATURE_SLOTS: tuple[str, ...] = (
    "judged_nativeness",
    "judged_phrasing",
    "judged_intonation",
    "judged_fidelity",
    "judged_repetition",
)


class FeatureExtractionConfig(BaseModel):
    """Extraction knobs, recorded in the table's provenance."""

    long_silence_threshold_s: Annotated[
        float, Field(description="Silence length (s) counted as long.")
    ] = DEFAULT_FEATURE_LONG_SILENCE_SECONDS
    turn_merge_gap_s: Annotated[
        float,
        Field(
            description="Max agent-silence gap (s) merged into one agent turn "
            "when no floor-taking user speech intervenes."
        ),
    ] = DEFAULT_FEATURE_TURN_MERGE_GAP_SECONDS
    backchannel_per_min_by_level: Annotated[
        dict[BackchannelLevel, float],
        Field(
            default_factory=lambda: dict(BACKCHANNEL_REFERENCE_PER_MIN),
            description="Native continuer rate (per minute) expected at each "
            "backchannel density level; the deviation's reference is the entry "
            "for the level the call's language pack declares.",
        ),
    ]
    human_backchannel_per_min: Annotated[
        float,
        Field(
            description="Fallback reference rate for calls whose language has "
            "no pack (or whose pack declares no level)."
        ),
    ] = DEFAULT_FEATURE_HUMAN_BACKCHANNEL_PER_MIN
    langs: Annotated[
        Optional[list[str]],
        Field(description="Restrict to these ISO 639-1 codes (None = all)."),
    ] = None
    domain: Annotated[
        Optional[str], Field(description="Restrict to one domain (None = all).")
    ] = None
    max_sims: Annotated[
        Optional[int],
        Field(description="Cap on extracted sims (dry runs / smokes)."),
    ] = None

    @field_validator("backchannel_per_min_by_level")
    @classmethod
    def _every_level_has_a_rate(
        cls, value: dict[BackchannelLevel, float]
    ) -> dict[BackchannelLevel, float]:
        """A partial map would silently score some languages against the
        fallback rate — reject it instead."""
        missing = sorted(
            level.value for level in BackchannelLevel if level not in value
        )
        if missing:
            raise ValueError(f"no backchannel reference rate for level(s): {missing}")
        return value


class CallFeatureRow(BaseModel):
    """One call: identity + gate + features."""

    results_path: Annotated[str, Field(description="Path of the run's results.json.")]
    experiment_label: Annotated[
        str, Field(description="Human label of the source run (its dir name).")
    ]
    sim_id: Annotated[str, Field(description="Simulation id.")]
    task_id: Annotated[str, Field(description="Task id.")]
    trial: Annotated[int, Field(description="Trial index.")]
    language: Annotated[str, Field(description="ISO 639-1 language of the call.")]
    domain: Annotated[str, Field(description="Domain of the run.")]
    modality: Annotated[
        Literal["voice", "text"],
        Field(
            description="How the call was conducted: 'voice' for an "
            "audio-native run, 'text' for a half-duplex text run. Run-level "
            "(a run either has an audio_native_config or it does not) and "
            "explicit rather than inferred from the empty provider, so a "
            "filter over it reads as what it means. Defaults to 'voice' "
            "because every table written before the field existed is one."
        ),
    ] = "voice"
    provider: Annotated[
        str,
        Field(
            description="Audio-native provider the agent spoke with "
            "('' on text rows — there is no speech stack in the loop)."
        ),
    ]
    agent_model: Annotated[str, Field(description="Agent model of the run.")]
    reasoning_effort: Annotated[
        Optional[ReasoningEffort],
        Field(
            description="The reasoning effort the run actually ran at. Part "
            "of the ARM (see ``Arm``), never defaulted. Voice rows read it "
            "from the run's AudioNativeConfig, and a voice run that recorded "
            "no effort is REFUSED by the extractor. Text rows read it from "
            "``agent_info.llm_args``, recorded verbatim by the runner — "
            "there None truthfully means the run pinned no effort, which is "
            "a different statement from 'unknown' and is why the field is "
            "nullable rather than sentinel-valued."
        ),
    ] = None
    gate_passed: Annotated[
        bool,
        Field(description="Sim-quality gate: clean termination + ticks present."),
    ]
    gate_reasons: Annotated[
        list[str],
        Field(default_factory=list, description="Why the gate failed, if it did."),
    ]
    duration_s: Annotated[
        Optional[float],
        Field(
            description="Call duration (s) on the tick clock; falls back to "
            "the sim's recorded wall-clock duration for tick-less sims."
        ),
    ] = None
    reward: Annotated[
        Optional[float],
        Field(
            description="Task reward as scored by the environment. Carried on "
            "the row so downstream analysis never reopens a results.json — a "
            "table that has drifted from its runs should fail loudly at "
            "load, not be silently repaired later."
        ),
    ] = None
    features: Annotated[CallFeatures, Field(description="The feature values.")]


class Arm(BaseModel):
    """A benchmark ARM: the configuration a call was produced under.

    Identity is ``(provider, agent_model, reasoning_effort)``. All three are
    load-bearing, and the third was the one that used to be missing.

    The English preference pool is a 2x2 of provider x reasoning effort —
    ``openai@xhigh``, ``openai@minimal``, ``gemini@high``, ``gemini@minimal``.
    Both openai cells report the SAME ``agent_model`` string
    (``openai:gpt-realtime-2``), so on a ``(provider, agent_model)`` identity
    an xhigh-vs-minimal pair typed as ``rerun`` — "pure stochastic variation"
    — when it is the study's primary systematic contrast. Two calls are the
    same arm only when the provider, the model AND the effort all match.

    Text rows fit the same identity: provider is ``''``, and the effort is
    the one pinned in ``agent_info.llm_args`` — the banking text pool is a
    2x2 of model x effort where both cells of each model report the same
    ``agent_model`` string, so effort carries that contrast exactly as it
    does for voice. ``reasoning_effort`` is ``None`` only for a text run
    that pinned no effort knob at all; an audio-native run that failed to
    record one never reaches this model (``build_feature_table`` refuses it).
    """

    model_config = {"frozen": True}

    provider: Annotated[str, Field(description="Audio-native provider.")]
    agent_model: Annotated[
        str, Field(description="Agent model id of the run ('' when unrecorded).")
    ]
    reasoning_effort: Annotated[
        Optional[ReasoningEffort],
        Field(description="Effort the run ran at (None = not audio-native)."),
    ] = None

    @classmethod
    def of(cls, row: "CallFeatureRow") -> "Arm":
        return cls(
            provider=row.provider,
            agent_model=row.agent_model,
            reasoning_effort=row.reasoning_effort,
        )

    @property
    def label(self) -> str:
        """Stable human-readable arm label, e.g. ``openai:gpt-realtime-2@xhigh``.

        Model ids recorded by the runner usually already carry their provider
        prefix (``openai:gpt-realtime-2``); the prefix is not repeated. The
        effort is suffixed with ``@`` so the two openai cells of the 2x2 read
        apart at a glance in every artifact that keys on arm labels.
        """
        model = self.agent_model.strip()
        if not model:
            base = self.provider
        elif not self.provider:
            base = model  # a text run: no provider to qualify the model with
        else:
            head = model.split(":", 1)[0].split("/", 1)[0]
            base = model if head == self.provider else f"{self.provider}/{model}"
        if self.reasoning_effort is None:
            return base
        return f"{base}@{self.reasoning_effort.value}"


def arm_pair_label(arm_a: Arm, arm_b: Arm) -> str:
    """Order-independent label for a pair of arms, e.g. ``'a+b'``."""
    return "+".join(sorted((arm_a.label, arm_b.label)))


class SourceRun(BaseModel):
    """Provenance for one source results file."""

    results_path: str
    experiment_label: str
    domain: str
    modality: Annotated[
        Literal["voice", "text"],
        Field(
            description="Whether the run was audio-native or half-duplex "
            "text, so a table's modality mix is readable from its "
            "provenance alone. Defaults to 'voice': every table written "
            "before the field existed is one."
        ),
    ] = "voice"
    run_provider: Annotated[
        str, Field(description="Run-level audio-native provider ('' for text).")
    ]
    agent_model: str
    reasoning_effort: Annotated[
        Optional[ReasoningEffort],
        Field(
            description="Run-level reasoning effort (voice: from the "
            "audio-native config; text: the effort pinned in "
            "``agent_info.llm_args``, None when none was). The third "
            "component of the arm identity, recorded here so a table's "
            "arm mix is readable from its provenance alone."
        ),
    ] = None
    run_git_commit: Annotated[
        str, Field(description="Git commit recorded on the run itself.")
    ]
    consolidation_id: Annotated[
        str,
        Field(
            description="sha256 (12 hex) of the ``consolidation.json`` sitting "
            "beside this results file, or '' if there is none. A consolidated "
            "arm was assembled from several source runs, and the sidecar is "
            "the only record of WHICH — pinning its content id here means the "
            "chain from a sampled pair back to the original run dirs cannot "
            "be broken by an edit to the sidecar."
        ),
    ] = ""
    n_calls: Annotated[int, Field(description="Rows extracted from this run.")]


class FeatureTable(BaseModel):
    """The provenance-bearing feature-table artifact.

    THE contract between the extractor and the pair sampler (and any future
    fit workstream): rows are keyed by (results_path, sim_id); ``table_id``
    is content-derived so a manifest can pin exactly which table it sampled.

    ``schema_version`` 2 (was 1) because rows carry ``reasoning_effort``.
    Nullable fields are usually additive, but this one is not: a version-1
    table would validate here with ``reasoning_effort=None`` on every row,
    which reads as "not audio-native" and merges the 2x2's two openai arms
    into one. Refusing to load it is the point.

    ``modality`` and the text-economy columns arrive WITHOUT a bump,
    deliberately: every table written before them was a voice table, so the
    defaults they load with (``modality="voice"``, text columns None) are
    true statements about those rows, not gaps wearing a default.
    """

    schema_version: Literal[2] = 2
    extractor_version: Annotated[
        str, Field(description="Version of the feature definitions/extractor.")
    ] = FEATURE_EXTRACTOR_VERSION
    table_id: Annotated[
        str,
        Field(
            description="Content-derived id: sha256(extractor_version, config, "
            "rows) truncated to 12 hex chars — identical inputs reproduce it."
        ),
    ]
    created_at: Annotated[str, Field(description="Extraction wall-clock time.")]
    git_sha: Annotated[str, Field(description="Repo HEAD at extraction time.")]
    config: Annotated[
        FeatureExtractionConfig, Field(description="Extraction knobs used.")
    ]
    source_runs: Annotated[list[SourceRun], Field(description="Per-run provenance.")]
    rows: Annotated[list[CallFeatureRow], Field(description="One row per call.")]


def load_feature_table(path: Path) -> FeatureTable:
    """Load + validate a feature-table artifact."""
    return FeatureTable.model_validate_json(Path(path).read_text())


# ---------------------------------------------------------------------------
# Tick timeline primitives
# ---------------------------------------------------------------------------


def _speech_flags(ticks: list[Tick]) -> tuple[list[bool], list[bool], set[int]]:
    """Per-tick (agent_speaking, user_speaking) + backchannel decision ticks."""
    agent = [t.agent_chunk is not None and t.agent_chunk.contains_speech for t in ticks]
    user = [t.user_chunk is not None and t.user_chunk.contains_speech for t in ticks]
    bc_ticks: set[int] = set()
    prev_was_bc = False
    for i, t in enumerate(ticks):
        action = (
            t.user_chunk.turn_taking_action.action
            if t.user_chunk is not None and t.user_chunk.turn_taking_action
            else None
        )
        is_bc = action == _BACKCHANNEL_ACTION
        if is_bc and not prev_was_bc:  # dedupe consecutive decision ticks
            bc_ticks.add(i)
        prev_was_bc = is_bc
    return agent, user, bc_ticks


def _segments(flags: list[bool]) -> list[tuple[int, int]]:
    """Maximal contiguous True runs as inclusive (start, end) index pairs."""
    segments: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(flags) - 1))
    return segments


def _backchannel_segments(
    user_segments: list[tuple[int, int]], bc_ticks: set[int]
) -> set[tuple[int, int]]:
    """User segments that are backchannel utterances (decision fired inside
    the segment or within the lookback window before its onset)."""
    marked: set[tuple[int, int]] = set()
    for start, end in user_segments:
        window = range(max(0, start - _BACKCHANNEL_LOOKBACK_TICKS), end + 1)
        if any(i in bc_ticks for i in window):
            marked.add((start, end))
    return marked


def _percentile(values: list[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile (numpy 'linear'); None on empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class _Exchange(BaseModel):
    """One user-turn -> agent-response event."""

    user_last_tick: int
    agent_onset_tick: int
    latency_s: float


def _exchanges(
    agent: list[bool],
    user_floor: list[bool],
    dur: float,
) -> list[_Exchange]:
    """User-turn -> agent-response events + tick-clock latencies.

    An agent speech onset at tick ``s`` is a response when floor-taking user
    speech occurred after the previous agent speech ended and the user is not
    still speaking at ``s`` (that overlap is an interruption, not a response).
    Latency is the tick-index delta to the last user speech tick: the last tick
    with user speech to the first subsequent tick with agent speech.
    Backchannel-only user segments never initiate an exchange.
    """
    exchanges: list[_Exchange] = []
    last_user: Optional[int] = None
    prev_agent_end = -1
    agent_segments = _segments(agent)
    for start, end in agent_segments:
        last_user = None
        for i in range(prev_agent_end + 1, start):
            if user_floor[i]:
                last_user = i
        if last_user is not None and not user_floor[start]:
            exchanges.append(
                _Exchange(
                    user_last_tick=last_user,
                    agent_onset_tick=start,
                    latency_s=(start - last_user) * dur,
                )
            )
        prev_agent_end = end
    return exchanges


def _merge_agent_turns(
    agent_segments: list[tuple[int, int]],
    user_floor: list[bool],
    dur: float,
    merge_gap_s: float,
) -> list[float]:
    """Agent turn durations (s): consecutive segments merge when the gap is
    short and carries no floor-taking user speech."""
    turns: list[tuple[int, int]] = []
    for start, end in agent_segments:
        if turns:
            prev_start, prev_end = turns[-1]
            gap_ticks = start - prev_end - 1
            gap_has_user = any(user_floor[i] for i in range(prev_end + 1, start))
            if gap_ticks * dur <= merge_gap_s and not gap_has_user:
                turns[-1] = (prev_start, end)
                continue
        turns.append((start, end))
    return [(end - start + 1) * dur for start, end in turns]


# ---------------------------------------------------------------------------
# Tool-flow features
# ---------------------------------------------------------------------------


def _tool_call_key(call: ToolCall) -> tuple[str, str]:
    return call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)


class _ToolFlow(BaseModel):
    tool_error_count: float
    repeated_tool_call_count: float
    agent_tool_calls: float
    auth_attempt_count: Optional[float] = None
    auth_arg_mismatch: Optional[float] = None
    time_to_auth: Optional[float] = None


def _tool_flow_from_ticks(
    ticks: list[Tick], auth_tools: Optional[frozenset[str]], dur: float
) -> _ToolFlow:
    """Tool friction features from inline tick tool activity (agent side)."""
    call_names: dict[str, str] = {}
    errors = 0
    repeats = 0
    total_calls = 0
    mismatches = 0
    seen: set[tuple[str, str]] = set()
    auth_calls_by_tick: list[int] = []  # tick index per auth call, in order
    first_auth_success_tick: Optional[int] = None

    for i, tick in enumerate(ticks):
        for call in tick.agent_tool_calls:
            total_calls += 1
            if call.id:
                call_names[call.id] = call.name
            key = _tool_call_key(call)
            if key in seen:
                repeats += 1
            seen.add(key)
            if auth_tools and call.name in auth_tools:
                auth_calls_by_tick.append(i)
    for i, tick in enumerate(ticks):
        for result in tick.agent_tool_results:
            # A not-found auth result counts here AND in auth_arg_mismatch,
            # deliberately: it is factually an error result, and the two
            # features answer different questions.
            if result.error:
                errors += 1
            name = call_names.get(result.id)
            if auth_tools is None or name not in auth_tools:
                continue
            outcome = classify_auth_lookup(result.content, result.error)
            if outcome is AuthLookupOutcome.NO_MATCH:
                mismatches += 1
            elif (
                outcome is AuthLookupOutcome.IDENTIFIED
                and first_auth_success_tick is None
            ):
                first_auth_success_tick = i

    auth_attempts: Optional[float] = None
    time_to_auth: Optional[float] = None
    if auth_tools is not None:
        if first_auth_success_tick is not None:
            time_to_auth = first_auth_success_tick * dur
            auth_attempts = float(
                sum(1 for t in auth_calls_by_tick if t <= first_auth_success_tick)
            )
        else:
            auth_attempts = float(len(auth_calls_by_tick))
    return _ToolFlow(
        tool_error_count=float(errors),
        repeated_tool_call_count=float(repeats),
        agent_tool_calls=float(total_calls),
        auth_attempt_count=auth_attempts,
        auth_arg_mismatch=(float(mismatches) if auth_tools is not None else None),
        time_to_auth=time_to_auth,
    )


def _tool_flow_from_messages(
    sim: SimulationRun, auth_tools: Optional[frozenset[str]]
) -> _ToolFlow:
    """Count-only tool features for tick-less (half-duplex) sims; the timing
    features stay None (there is no perceptual tick clock to measure on)."""
    call_names: dict[str, str] = {}
    errors = 0
    repeats = 0
    total_calls = 0
    mismatches = 0
    seen: set[tuple[str, str]] = set()
    auth_call_count = 0
    auth_succeeded = False
    for msg in sim.get_messages():
        calls = getattr(msg, "tool_calls", None)
        for call in calls or []:
            if call.requestor != "assistant":
                continue
            total_calls += 1
            if call.id:
                call_names[call.id] = call.name
            key = _tool_call_key(call)
            if key in seen:
                repeats += 1
            seen.add(key)
            if auth_tools and call.name in auth_tools:
                if not auth_succeeded:
                    auth_call_count += 1
        if getattr(msg, "role", None) == "tool" and msg.requestor == "assistant":
            if msg.error:
                errors += 1
            name = call_names.get(msg.id)
            if auth_tools is not None and name in auth_tools:
                outcome = classify_auth_lookup(msg.content, msg.error)
                if outcome is AuthLookupOutcome.NO_MATCH:
                    mismatches += 1
                elif outcome is AuthLookupOutcome.IDENTIFIED:
                    auth_succeeded = True
    return _ToolFlow(
        tool_error_count=float(errors),
        repeated_tool_call_count=float(repeats),
        agent_tool_calls=float(total_calls),
        auth_attempt_count=(float(auth_call_count) if auth_tools is not None else None),
        auth_arg_mismatch=(float(mismatches) if auth_tools is not None else None),
        time_to_auth=None,
    )


# ---------------------------------------------------------------------------
# Text-economy features (message-based)
# ---------------------------------------------------------------------------

# A line that opens a block-formatting construct: bullet, numbered list,
# heading, table row, code fence, blockquote. Block-level only, deliberately —
# inline emphasis (**bold**) inside a conversational sentence is not the
# document-shaped answer this feature measures, and matching it would count
# ordinary chat as formatted.
_FORMATTED_LINE = re.compile(r"^\s*(?:[-*•]\s|\d+[.)]\s|#{1,6}\s|\||```|>\s)")


class _TextEconomy(BaseModel):
    turn_count: Optional[float] = None
    agent_msg_len_p90: Optional[float] = None
    agent_char_ratio: Optional[float] = None
    formatting_density: Optional[float] = None


def _text_economy_from_messages(sim: SimulationRun) -> _TextEconomy:
    """Message-based economy features for a half-duplex (tick-less) sim.

    Counts CONTENT messages only: an assistant message that carries nothing
    but tool calls is plumbing the user never sees, not a turn. Content is
    measured in characters rather than tokens — the annotator scrolling a
    wall of text experiences its length, not its tokenization.
    """
    agent_texts: list[str] = []
    user_chars = 0
    for msg in sim.get_messages():
        content = getattr(msg, "content", None)
        if not isinstance(content, str) or not content.strip():
            continue
        role = getattr(msg, "role", None)
        if role == "assistant":
            agent_texts.append(content)
        elif role == "user":
            user_chars += len(content)
    if not agent_texts and user_chars == 0:
        return _TextEconomy()
    agent_chars = sum(len(text) for text in agent_texts)
    total_chars = agent_chars + user_chars
    lines = [line for text in agent_texts for line in text.splitlines() if line.strip()]
    formatted = sum(1 for line in lines if _FORMATTED_LINE.match(line))
    return _TextEconomy(
        turn_count=float(len(agent_texts)),
        agent_msg_len_p90=_percentile([float(len(t)) for t in agent_texts], 0.9),
        agent_char_ratio=(agent_chars / total_chars if total_chars > 0 else None),
        formatting_density=(formatted / len(lines) if lines else None),
    )


# ---------------------------------------------------------------------------
# Per-call extraction
# ---------------------------------------------------------------------------


def resolve_backchannel_reference_per_min(
    language: str, config: FeatureExtractionConfig
) -> float:
    """Native continuer rate (per minute) to measure ``language`` against.

    Read from the language pack's ``BackchannelLevel`` — the single source of
    truth for how densely a native speaker of this language backchannels.
    Every shipped pack declares ``medium`` as of 2026-07-26 (one density knob,
    one setting: the per-language tuning was guesswork, and the split let a
    cross-language comparison read a difference that came from the knob), so
    today this resolves uniformly. It stays level-driven rather than constant
    because scoring against one global rate would book correct native behavior
    as a defect the moment any pack moves off medium, and miss
    under-backchanneling there.
    Falls back to the configured single reference when the language has no pack
    (English control runs predating the ``en`` pack) or its pack declares no
    level.
    """
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack(language.lower())
    level = pack.backchannel_level if pack is not None else None
    if level is None:
        return config.human_backchannel_per_min
    return config.backchannel_per_min_by_level[level]


def extract_call_features(
    sim: SimulationRun,
    *,
    domain: str,
    language: str,
    info: Optional[Info] = None,
    config: Optional[FeatureExtractionConfig] = None,
    backchannel_reference_per_min: Optional[float] = None,
) -> CallFeatures:
    """All Tier-1 features for one sim (tick clock when ticks exist).

    ``language`` is the call's ISO 639-1 code (see ``_sim_language``); it fixes
    the native backchannel-density reference the deviation is measured against.

    ``backchannel_reference_per_min`` lets a caller supply that reference
    already resolved. It is a property of the LANGUAGE, not of the call, and
    resolving it costs a language-pack registry lookup — a per-simulation cost
    (1581 of them in one table build) for a value that changes once per
    language, over a registry with a known first-touch race. ``build_feature_
    table`` resolves it once per language; None keeps the direct callers
    (tests, ad-hoc extraction) working unchanged.
    """
    cfg = config or FeatureExtractionConfig()
    auth_tools = AUTH_TOOLS_BY_DOMAIN.get(domain)

    if not sim.ticks:
        flow = _tool_flow_from_messages(sim, auth_tools)
        economy = _text_economy_from_messages(sim)
        return CallFeatures(
            tool_error_count=flow.tool_error_count,
            repeated_tool_call_count=flow.repeated_tool_call_count,
            agent_tool_calls=flow.agent_tool_calls,
            auth_attempt_count=flow.auth_attempt_count,
            auth_arg_mismatch=flow.auth_arg_mismatch,
            turn_count=economy.turn_count,
            agent_msg_len_p90=economy.agent_msg_len_p90,
            agent_char_ratio=economy.agent_char_ratio,
            formatting_density=economy.formatting_density,
        )

    ticks = sim.ticks
    dur = resolve_tick_duration_seconds(ticks, info)
    n = len(ticks)
    total_s = n * dur

    agent, user, bc_ticks = _speech_flags(ticks)
    user_segments = _segments(user)
    bc_segments = _backchannel_segments(user_segments, bc_ticks)
    user_floor = list(user)
    for start, end in bc_segments:
        for i in range(start, end + 1):
            user_floor[i] = False

    agent_segments = _segments(agent)
    exchanges = _exchanges(agent, user_floor, dur)
    latencies = [e.latency_s for e in exchanges]

    # Silence & overlap
    neither = [not a and not u for a, u in zip(agent, user)]
    long_silences = sum(
        1
        for start, end in _segments(neither)
        if (end - start + 1) * dur > cfg.long_silence_threshold_s
    )
    dead_air = sum(neither) * dur
    overlap = sum(1 for a, u in zip(agent, user) if a and u) * dur

    # Interruptions (onset while the other side's speech is in progress)
    agent_interruptions = sum(1 for start, _ in agent_segments if user_floor[start])
    user_barge_ins = sum(
        1 for seg in user_segments if seg not in bc_segments and agent[seg[0]]
    )

    # Backchannel density vs the reference rate native to this language
    bc_count = float(len(bc_ticks))
    minutes = total_s / 60.0
    reference_per_min = (
        backchannel_reference_per_min
        if backchannel_reference_per_min is not None
        else resolve_backchannel_reference_per_min(language, cfg)
    )
    bc_deviation = abs(bc_count / minutes - reference_per_min) if minutes > 0 else None

    # Speech economy
    agent_speech_s = sum(agent) * dur
    user_speech_s = sum(user) * dur
    talk_denominator = agent_speech_s + user_speech_s
    talk_ratio = agent_speech_s / talk_denominator if talk_denominator > 0 else None
    turn_lengths = _merge_agent_turns(
        agent_segments, user_floor, dur, cfg.turn_merge_gap_s
    )

    flow = _tool_flow_from_ticks(ticks, auth_tools, dur)

    return CallFeatures(
        response_latency_p50=_percentile(latencies, 0.5),
        response_latency_p90=_percentile(latencies, 0.9),
        long_silence_count=float(long_silences),
        long_silence_count_per_min=(long_silences / minutes if minutes > 0 else None),
        dead_air_ratio=(dead_air / total_s) if total_s > 0 else None,
        agent_interruption_count=float(agent_interruptions),
        user_barge_in_count=float(user_barge_ins),
        overlap_time_total=overlap,
        overlap_time_total_per_min=(overlap / minutes if minutes > 0 else None),
        backchannel_count=bc_count,
        backchannel_density_deviation=bc_deviation,
        time_to_auth=flow.time_to_auth,
        auth_attempt_count=flow.auth_attempt_count,
        auth_arg_mismatch=flow.auth_arg_mismatch,
        agent_tool_calls=flow.agent_tool_calls,
        time_to_resolution=total_s,
        tool_error_count=flow.tool_error_count,
        repeated_tool_call_count=flow.repeated_tool_call_count,
        agent_talk_ratio=talk_ratio,
        agent_turn_len_p90=_percentile(turn_lengths, 0.9),
        turn_count=float(len(exchanges)),
    )


def _gate(
    sim: SimulationRun, *, modality: Literal["voice", "text"] = "voice"
) -> tuple[bool, list[str]]:
    """Sim-quality gate, judged against what the MODALITY promises.

    A voice sim without ticks is a broken recording — there is no call to
    listen to. A text sim never has ticks, by design; its transcript IS the
    messages, so that is what must be present.
    """
    reasons: list[str] = []
    if sim.termination_reason in GATE_FAIL_TERMINATIONS:
        reasons.append(f"termination:{sim.termination_reason.value}")
    if modality == "voice":
        if not sim.ticks:
            reasons.append("no_ticks")
    elif not sim.get_messages():
        reasons.append("no_messages")
    return (not reasons), reasons


def _sim_language(sim: SimulationRun, domain: str) -> str:
    """A sim's run language: judged language if stored, else the task-id suffix.

    Older English runs may not carry ``nativeness_info``, and the run-level Info
    holds no reliable language, so the localized task-set suffix on the task id
    (``..._es``, ``..._pt_identity``) is the deterministic fallback; base English
    task ids have no suffix and mean English.
    """
    if sim.nativeness_info is not None and sim.nativeness_info.language:
        return sim.nativeness_info.language.lower()
    from tau2.multilingual.english_prompts import task_language

    return (task_language(str(sim.task_id), domain) or "en").lower()


def _resolve_reasoning_effort(
    audio_cfg: Optional[AudioNativeConfig],
    *,
    provider: str,
    results_file: Path,
    agent_llm_args: Optional[dict] = None,
) -> Optional[ReasoningEffort]:
    """The arm's reasoning effort, or a loud refusal.

    Three cases, and the middle one is the whole reason this function exists.

    * **No audio-native config and no provider** — a text run. Its effort
      knob, if any, lives in ``agent_info.llm_args``, which the runner
      records VERBATIM: an absent key truthfully means the run pinned
      nothing (``None``), so text never needs the refusal path — there is no
      in-memory default that could masquerade as a recorded value. A pinned
      value that names no known level still refuses: it is part of the arm
      identity and cannot be quietly dropped.
    * **Effort never recorded by the run** — ``AudioNativeConfig`` resolves a
      stored ``null`` to the provider default IN MEMORY and marks the source
      ``inferred`` with no backfill stamp. Reading that value would silently
      merge two arms into one: an unpinned ``openai`` run and a pinned
      ``openai@minimal`` run would come back with the same effort and type as
      a stochastic rerun of each other. Refuse instead.
    * **Recorded (``run``) or backfilled (``inferred`` + a backfill stamp)** —
      trustworthy, and in the backfilled case traceable to the verb and basis
      that wrote it.

    A provider with no audio-native config at all is the same failure wearing
    a different hat: the call was audio-native (something recorded a provider
    for it) but its arm is not identifiable from the artifact.
    """
    if audio_cfg is None:
        if not provider:
            pinned = (agent_llm_args or {}).get("reasoning_effort")
            if pinned is None:
                return None
            try:
                return ReasoningEffort(pinned)
            except ValueError:
                raise ValueError(
                    f"{results_file}: the text run pins reasoning_effort "
                    f"'{pinned}' in agent llm_args, which is not a known "
                    "level. It is part of the ARM identity, so it cannot be "
                    "dropped or defaulted — add the level to ReasoningEffort "
                    "if it is real."
                ) from None
        raise ValueError(
            f"{results_file}: the run records audio-native provider "
            f"'{provider}' but no audio_native_config, so its reasoning "
            "effort — part of the ARM identity — cannot be read. Refusing to "
            "default it: a missing effort collapses two arms into one and "
            "types a systematic contrast as a stochastic rerun. This run "
            "predates recorded reasoning-effort and was never backfilled "
            "(the backfill verb is retired), so the value cannot be trusted."
        )
    unrecorded = (
        audio_cfg.reasoning_effort_source is ReasoningEffortSource.INFERRED
        and audio_cfg.reasoning_effort_backfill is None
    )
    if unrecorded:
        raise ValueError(
            f"{results_file}: reasoning_effort was never recorded by this run "
            f"(provider '{audio_cfg.provider}'; the loader inferred "
            f"'{audio_cfg.reasoning_effort.value}' from the provider default "
            "table). Refusing to default it: a missing effort collapses two "
            "arms into one and types a systematic contrast as a stochastic "
            "rerun. This run predates recorded reasoning-effort and was never "
            "backfilled (the backfill verb is retired), so the value cannot "
            "be trusted."
        )
    return audio_cfg.reasoning_effort


#: Written beside ``results.json`` by ``tau2 annotate consolidate-pool``.
CONSOLIDATION_SIDECAR = "consolidation.json"


def _consolidation_id(results_file: Path) -> str:
    """Content id of the consolidation sidecar beside a run, or ''."""
    sidecar = Path(results_file).parent / CONSOLIDATION_SIDECAR
    if not sidecar.is_file():
        return ""
    return hashlib.sha256(sidecar.read_bytes()).hexdigest()[:12]


def _derive_table_id(
    config: FeatureExtractionConfig, rows: list[CallFeatureRow]
) -> str:
    h = hashlib.sha256()
    h.update(FEATURE_EXTRACTOR_VERSION.encode())
    h.update(b"\0")
    h.update(config.model_dump_json().encode())
    for row in rows:
        h.update(b"\0")
        h.update(row.model_dump_json().encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Judged slots
# ---------------------------------------------------------------------------

#: Judged slots with no judge behind them yet. Named here rather than left as
#: quiet Nones, so a consumer can tell "no judge exists" from "the judge did
#: not run on this call".
UNJUDGED_SLOTS: tuple[str, ...] = ("judged_phrasing", "judged_repetition")


def attach_judged_features(features: CallFeatures, sim: SimulationRun) -> CallFeatures:
    """Copy a sim's stored judge verdicts into the judged feature slots.

    Read, never re-judged: the judges run in the pipeline and persist their
    scores on the simulation, and extraction is a cheap deterministic pass that
    must not make LLM calls. A sim whose judges never ran keeps its Nones,
    which the fit drops rather than imputing as the pool mean.
    """
    nativeness = sim.nativeness_info
    delivery = sim.delivery_info
    return features.model_copy(
        update={
            "judged_nativeness": nativeness.score if nativeness else None,
            "judged_fidelity": delivery.fidelity_score if delivery else None,
            "judged_intonation": delivery.intonation_score if delivery else None,
        }
    )


# ---------------------------------------------------------------------------
# Table extraction over runs
# ---------------------------------------------------------------------------


def build_feature_table(
    paths: Iterable[Path | str],
    config: Optional[FeatureExtractionConfig] = None,
) -> FeatureTable:
    """One feature row per sim across every results file under ``paths``."""
    cfg = config or FeatureExtractionConfig()
    wanted_langs = {lang.lower() for lang in cfg.langs} if cfg.langs else None
    # One registry lookup per LANGUAGE, not per simulation: the native
    # backchannel reference is constant within a language, and the pack
    # registry has a known first-touch race under concurrency.
    backchannel_reference: dict[str, float] = {}
    rows: list[CallFeatureRow] = []
    source_runs: list[SourceRun] = []
    results_files = [f for p in paths for f in discover_results_files(Path(p))]
    for i, results_file in enumerate(results_files, 1):
        logger.info(f"extracting features [{i}/{len(results_files)}]: {results_file}")
        meta = Results.load_metadata(results_file)
        domain = meta.info.environment_info.domain_name
        if cfg.domain and domain != cfg.domain:
            logger.info(f"skipping (domain {domain} != {cfg.domain}): {results_file}")
            continue
        audio_cfg = meta.info.audio_native_config
        modality: Literal["voice", "text"] = (
            "voice" if audio_cfg is not None else "text"
        )
        run_provider = audio_cfg.provider if audio_cfg is not None else ""
        agent_model = str(meta.info.agent_info.llm or "")
        agent_llm_args = meta.info.agent_info.llm_args
        # Resolved once per RUN, before any sim is read, so an unrecorded
        # effort fails the extraction immediately rather than after a few
        # hundred rows. Only the no-config case can still vary per sim (a sim
        # may carry an ``agent_provider`` its run-level config does not).
        run_effort = _resolve_reasoning_effort(
            audio_cfg,
            provider=run_provider,
            results_file=results_file,
            agent_llm_args=agent_llm_args,
        )
        experiment_label = results_file.parent.name
        n_before = len(rows)
        for sim in Results.iter_simulations(results_file):
            if cfg.max_sims is not None and len(rows) >= cfg.max_sims:
                break
            language = _sim_language(sim, domain)
            if wanted_langs is not None and language not in wanted_langs:
                continue
            gate_passed, gate_reasons = _gate(sim, modality=modality)
            if language not in backchannel_reference:
                backchannel_reference[language] = resolve_backchannel_reference_per_min(
                    language, cfg
                )
            features = extract_call_features(
                sim,
                domain=domain,
                language=language,
                info=meta.info,
                config=cfg,
                backchannel_reference_per_min=backchannel_reference[language],
            )
            features = attach_judged_features(features, sim)
            duration_s = (
                features.time_to_resolution
                if features.time_to_resolution is not None
                else sim.duration
            )
            provider = sim.agent_provider or run_provider
            effort = (
                run_effort
                if audio_cfg is not None
                else _resolve_reasoning_effort(
                    None,
                    provider=provider,
                    results_file=results_file,
                    agent_llm_args=agent_llm_args,
                )
            )
            rows.append(
                CallFeatureRow(
                    results_path=str(results_file),
                    experiment_label=experiment_label,
                    sim_id=sim.id,
                    task_id=str(sim.task_id),
                    trial=sim.trial or 0,
                    language=language,
                    domain=domain,
                    modality=modality,
                    provider=provider,
                    agent_model=agent_model,
                    reasoning_effort=effort,
                    gate_passed=gate_passed,
                    gate_reasons=gate_reasons,
                    duration_s=duration_s,
                    reward=(
                        sim.reward_info.reward if sim.reward_info is not None else None
                    ),
                    features=features,
                )
            )
        source_runs.append(
            SourceRun(
                results_path=str(results_file),
                experiment_label=experiment_label,
                domain=domain,
                modality=modality,
                run_provider=run_provider,
                agent_model=agent_model,
                reasoning_effort=run_effort,
                run_git_commit=meta.info.git_commit,
                consolidation_id=_consolidation_id(results_file),
                n_calls=len(rows) - n_before,
            )
        )
        if cfg.max_sims is not None and len(rows) >= cfg.max_sims:
            break
    rows.sort(key=lambda r: (r.language, r.task_id, r.provider, r.trial, r.sim_id))
    return FeatureTable(
        table_id=_derive_table_id(cfg, rows),
        created_at=get_now(),
        git_sha=git_sha(),
        config=cfg,
        source_runs=source_runs,
        rows=rows,
    )


def export_feature_table(
    paths: Iterable[Path | str],
    out: Path,
    config: Optional[FeatureExtractionConfig] = None,
) -> Path:
    """Build + atomically write the feature-table artifact; returns the path."""
    table = build_feature_table(paths, config)
    if not table.rows:
        raise ValueError(
            "no calls extracted — check the results paths and the "
            "--langs/--domain filters"
        )
    path = write_json_artifact(Path(out), table)
    gated = sum(1 for r in table.rows if r.gate_passed)
    logger.info(
        f"feature table {table.table_id}: {len(table.rows)} calls "
        f"({gated} gate-passing) -> {path}"
    )
    return path

# Copyright Sierra
"""Domain -> complication-sampler dispatch (runner seam).

The runner stays domain-agnostic: a domain that supports scripted caller
complications registers its sampler here, and ``user_prompt_task`` /
the orchestrator builders consult :func:`sample_run_complication` without
importing anything domain-specific. Today only intake has a sampler (see
``tau2.domains.intake.complications``).

The same seam owns the domain's pre-synthesis pronunciation map
(:func:`run_pronunciation_map`): the voice-orchestrator builder hands the
map to the voice USER simulator's settings, and domains without a builder
get None — synthesis stays identity everywhere but intake.

Sampling is deterministic from ``(run seed, task id, catalog version)`` plus
the run's complication profile, rate override, and channel (text vs voice —
channel-bound kinds like intake's voice-only ``mispronounced_term`` are
infeasible off-channel); this module memoizes the draw per run so the
prompt-injection site and the provenance-stamping site read the SAME object
and the trigger is logged once. It also owns the profile's task selection
(:func:`profile_task_filter`): the HARD profile restricts a run to hard-tier
tasks at task-selection level (``resolve_tasks``), never by editing task
data.
"""

from typing import Callable, Literal, Optional, Protocol

from loguru import logger

from tau2.data_model.simulation import ComplicationProfile, SampledComplication
from tau2.data_model.tasks import Task

Channel = Literal["text", "voice"]


class _RunConfigLike(Protocol):
    domain: str
    seed: Optional[int]
    complication_profile: ComplicationProfile
    complication_rate: Optional[float]


def run_channel(config: _RunConfigLike) -> Channel:
    """The run's modality, read off the config's concrete type.

    The RunConfig subclass IS the channel at the build layer: a
    ``TextRunConfig`` only ever builds a half-duplex text orchestrator and a
    ``VoiceRunConfig`` only ever builds a full-duplex voice one. Anything else
    fails loud — complication feasibility is channel-bound, so an
    indeterminable channel must never silently sample as text.
    """
    from tau2.data_model.simulation import TextRunConfig, VoiceRunConfig

    if isinstance(config, VoiceRunConfig):
        return "voice"
    if isinstance(config, TextRunConfig):
        return "text"
    raise ValueError(
        f"Cannot determine the run channel from config type "
        f"{type(config).__name__}: complication sampling is channel-bound "
        "(expected TextRunConfig or VoiceRunConfig)"
    )


def _intake_sampler(
    run_seed: int,
    task: Task,
    profile: ComplicationProfile,
    rate_override: Optional[float],
    channel: Channel,
) -> Optional[SampledComplication]:
    from tau2.domains.intake.complications import sample_complication

    return sample_complication(
        run_seed, task, profile=profile, rate_override=rate_override, channel=channel
    )


# Domains that support scripted complications. Samplers share the signature
# (run_seed, task, profile, rate_override, channel) ->
# Optional[SampledComplication] and must be deterministic in
# (run_seed, task.id, profile, rate_override, channel, their catalog version).
DOMAIN_COMPLICATION_SAMPLERS: dict[
    str,
    Callable[
        [int, Task, ComplicationProfile, Optional[float], Channel],
        Optional[SampledComplication],
    ],
] = {
    "intake": _intake_sampler,
    # The free-strategy arm shares the intake catalog verbatim: task ids are
    # identical, so the (seed, task) draw is byte-identical across arms.
    "intake_free": _intake_sampler,
}

# Memoized draws for this process: (domain, seed, profile, rate_override,
# channel, task_id) -> draw. Purely deterministic values, so the cache only
# saves recomputation and keeps the "complication armed" log line to one per
# simulation build.
_SAMPLED: dict[tuple, Optional[SampledComplication]] = {}


def sample_run_complication(
    config: _RunConfigLike, task: Task
) -> Optional[SampledComplication]:
    """The task's scripted complication under this run config, or None.

    None when the run is explicitly clean (``--complication-rate 0``), the
    domain has no registered sampler, or the profile's categorical draw lands
    on "none" for this task. A domain WITH a sampler is armed by default —
    its profile's per-kind rates decide the triggers. An explicit rate
    override on a domain with NO sampler fails loud: an armed run silently
    running clean is exactly the confusion the provenance exists to prevent
    (an un-overridden profile on such a domain is just the field's default
    and runs clean quietly). The channel is derived from the config's
    concrete type (:func:`run_channel`) and threaded to the sampler, which
    gates channel-bound kinds.
    """
    domain = config.domain
    profile = config.complication_profile
    rate_override = getattr(config, "complication_rate", None)
    sampler = DOMAIN_COMPLICATION_SAMPLERS.get(domain)
    if sampler is None:
        if rate_override:
            raise ValueError(
                f"--complication-rate {rate_override} was set but domain "
                f"{domain!r} has no complication sampler registered "
                "(tau2.runner.complications.DOMAIN_COMPLICATION_SAMPLERS)"
            )
        return None
    if rate_override is not None and rate_override <= 0.0:
        return None
    channel = run_channel(config)
    run_seed = config.seed if config.seed is not None else 0
    key = (domain, run_seed, profile, rate_override, channel, task.id)
    if key not in _SAMPLED:
        sampled = sampler(run_seed, task, profile, rate_override, channel)
        if sampled is not None:
            logger.info(
                f"Complication armed for task {task.id}: kind={sampled.kind} "
                f"params={sampled.params} injected={sampled.injected}"
            )
        _SAMPLED[key] = sampled
    return _SAMPLED[key]


def _intake_hard_tier(task: Task) -> bool:
    from tau2.domains.intake.complications import is_hard_tier

    return is_hard_tier(task.id)


# Domains whose task ids encode an entity tier, for profile-imposed task
# selection. The HARD profile keeps exactly the tasks this predicate accepts.
DOMAIN_HARD_TIER_PREDICATES: dict[str, Callable[[Task], bool]] = {
    "intake": _intake_hard_tier,
    "intake_free": _intake_hard_tier,
}


def profile_task_filter(config: _RunConfigLike) -> Optional[Callable[[Task], bool]]:
    """The task predicate this run's complication profile imposes, or None.

    The DEFAULT profile imposes none (the domain's full frozen set runs —
    intake's is 100 easy / 100 hard by construction, so the default 50/50
    tier mix needs no subsampling). HARD restricts the run to hard-tier
    tasks; a HARD profile on a domain without a tier predicate fails loud
    rather than silently running the full set.
    """
    if config.complication_profile is not ComplicationProfile.HARD:
        return None
    predicate = DOMAIN_HARD_TIER_PREDICATES.get(config.domain)
    if predicate is None:
        raise ValueError(
            f"--complication-profile hard was set but domain "
            f"{config.domain!r} has no hard-tier predicate registered "
            "(tau2.runner.complications.DOMAIN_HARD_TIER_PREDICATES)"
        )
    return predicate


def _intake_pronunciation_map(
    complication: Optional[SampledComplication],
) -> dict[str, str]:
    from tau2.domains.intake.pronunciation_map import build_pronunciation_map

    return build_pronunciation_map(complication)


# Domains whose voice runs apply a pre-synthesis pronunciation substitution.
# Builders take the simulation's sampled complication (a mispronounced_term
# draw overrides exactly the drawn term) and return the token -> respelling
# map the voice user's synthesis text goes through.
DOMAIN_PRONUNCIATION_MAP_BUILDERS: dict[
    str, Callable[[Optional[SampledComplication]], dict[str, str]]
] = {
    "intake": _intake_pronunciation_map,
    # The staged variant keeps the default readings (its runs are clean —
    # no complication sampler is registered for it — so the builder only
    # ever sees complication=None).
    "intake_staged": _intake_pronunciation_map,
    # The free-strategy arm keeps the full intake behavior: default readings
    # always, mispronounced_term draws override exactly the drawn term.
    "intake_free": _intake_pronunciation_map,
}


def run_pronunciation_map(
    config: _RunConfigLike, complication: Optional[SampledComplication]
) -> Optional[dict[str, str]]:
    """The run's pre-synthesis pronunciation map, or None.

    None for domains without a registered builder — the voice seam treats a
    missing/empty map as identity, so non-intake domains are untouched by
    construction. Intake returns an EMPTY map unless the simulation drew
    ``mispronounced_term`` (default TTS readings always — design doc sec. 3,
    owner decision 2026-08-26); the drawn term is the map's only entry.
    """
    builder = DOMAIN_PRONUNCIATION_MAP_BUILDERS.get(config.domain)
    if builder is None:
        return None
    return builder(complication)


def _intake_free_spell_detector(task: Task):
    from tau2.domains.intake.spell_events import build_spell_event_detector

    return build_spell_event_detector(task)


# Domains whose voice USER simulator stamps deterministic ``spell_out``
# effect-timeline events when the sim's pre-synthesis utterance text renders
# a character-by-character spell-out of a tracked entity field (ground truth
# for the caller-effort ledger — tau2.metrics.caller_effort). Builders take
# the run's task and return an object whose ``detect(text)`` yields
# ``SpellOutDetection`` records. ONLY the free-strategy arm registers one:
# canonical intake runs stamp nothing and stay byte-identical.
DOMAIN_SPELL_EVENT_DETECTOR_BUILDERS: dict[str, Callable[[Task], object]] = {
    "intake_free": _intake_free_spell_detector,
}


def run_spell_event_detector(config: _RunConfigLike, task: Task):
    """The run's spell-out detector, or None for domains without a builder."""
    builder = DOMAIN_SPELL_EVENT_DETECTOR_BUILDERS.get(config.domain)
    if builder is None:
        return None
    return builder(task)

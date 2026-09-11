# Copyright Sierra
"""Attach judge axes to a simulation: applicability gating + result stamping.

``attach_quality`` / ``attach_nativeness`` / ``attach_delivery`` are the run-time entry points the
runner (and post-hoc rescoring) call per simulation. They own the applicability
gating (language resolution, defaults) and hand the harnesses the settings that
get stamped — judge_model / judge_args / judge_prompt_version — onto the
resulting ``*Info`` models. The scoring axes are decoupled siblings: neither
ever folds into reward / pass@1.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from tau2.data_model.simulation import (
    DeliveryJudgeSettings,
    NativenessJudgeSettings,
    QualityJudgeSettings,
    SimulationRun,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import get_simulation_language_info
from tau2.judges.delivery.harness import evaluate_delivery
from tau2.judges.nativeness.agent_voice import (
    ParticipantGender,
    normalize_participant_gender,
)
from tau2.judges.nativeness.agent_voice import (
    agent_context as build_agent_context,
)
from tau2.judges.nativeness.caller_identity import (
    CallerNameContext,
    resolve_korean_caller_name,
)
from tau2.judges.nativeness.harness import evaluate_nativeness
from tau2.judges.quality.harness import evaluate_quality
from tau2.multilingual.factory.entity_localization import caller_gender_for_task
from tau2.voice.voice_gender import resolve_agent_gender


class ResolvedParticipants(BaseModel):
    """Participant identity context resolved for one simulation's judging."""

    provider: Optional[str] = Field(
        default=None, description="Audio-native provider the agent voice ran on."
    )
    voice: Optional[str] = Field(
        default=None, description="Pinned provider voice id, when recorded."
    )
    agent_gender: Optional[ParticipantGender] = Field(
        default=None,
        description="Agent voice gender from the reviewed provider-voice "
        "catalog; never inferred from a name.",
    )
    caller_gender: Optional[ParticipantGender] = Field(
        default=None,
        description="Caller persona gender from run context, the recorded "
        "persona tag, or the reviewed task sidecar; never inferred from a name.",
    )
    caller_name: Optional[CallerNameContext] = Field(
        default=None,
        description="Typed localized Korean caller name; local prechecks only.",
    )


def resolve_participants(
    simulation: SimulationRun,
    task: Task,
    *,
    language: Optional[str],
    agent_provider: Optional[str] = None,
    agent_voice: Optional[str] = None,
    domain: Optional[str] = None,
    caller_gender: Optional[str] = None,
) -> ResolvedParticipants:
    """Resolve the agent/caller gender context the nativeness judging uses.

    One resolution rule shared by run-time judging (``attach_nativeness``) and
    post-hoc factor rechecks: per-sim recorded provider/voice first, falling
    back to the caller-supplied run-level values; caller gender from explicit
    run context, then the recorded persona tag, then the reviewed task sidecar.
    Missing values remain unknown — nothing is inferred from names.
    """
    provider = simulation.agent_provider or agent_provider
    voice = simulation.agent_voice or agent_voice
    customer_gender = caller_gender
    if customer_gender is None and simulation.speech_environment is not None:
        customer_gender = simulation.speech_environment.persona_tags.get("gender")
    if customer_gender is None and domain and language:
        customer_gender = caller_gender_for_task(task.id, language, domain)
    return ResolvedParticipants(
        provider=provider,
        voice=voice,
        agent_gender=normalize_participant_gender(
            resolve_agent_gender(provider, voice)
        ),
        caller_gender=normalize_participant_gender(customer_gender),
        caller_name=resolve_korean_caller_name(task, language, domain),
    )


def attach_quality(
    simulation: SimulationRun,
    task: Task,
    *,
    domain: Optional[str] = None,
    settings: Optional[QualityJudgeSettings] = None,
) -> None:
    """Compute and attach universal call quality, separately from reward."""
    simulation.quality_info = evaluate_quality(
        simulation,
        task,
        domain=domain,
        settings=settings or QualityJudgeSettings(),
    )


def attach_nativeness(
    simulation: SimulationRun,
    task: Task,
    settings: Optional[NativenessJudgeSettings] = None,
    agent_provider: Optional[str] = None,
    agent_voice: Optional[str] = None,
    domain: Optional[str] = None,
    caller_gender: Optional[str] = None,
) -> None:
    """Compute and attach the nativeness score (a decoupled axis, not reward).

    No-op only when the run has no resolved language. English runs retain the
    deterministic interaction factors; language-specific LLM factors still
    come only from multilingual packs.
    Deterministic factors selected by the language pack always run; the LLM
    judge is OPT-IN (``settings=None`` means default
    ``NativenessJudgeSettings``, whose ``llm_judge`` is False — enable with
    ``NativenessJudgeSettings(llm_judge=True)``).

    The judge evaluates the AGENT's speech. Agent gender (resolved from the
    provider voice) is used for first-person agreement; caller gender (resolved
    from explicit run context, the recorded persona tag, or the reviewed task
    sidecar) is used only for direct address/reference. Korean caller-name
    context is resolved only from typed localized identity metadata and stays
    local to the deterministic checker. Missing values remain unknown.
    """
    settings = settings if settings is not None else NativenessJudgeSettings()
    language, script = get_simulation_language_info(simulation)
    # Prefer the provider/voice recorded on the run; fall back to caller-supplied
    # values (older results predating those fields, threaded from results.info).
    participants = resolve_participants(
        simulation,
        task,
        language=language,
        agent_provider=agent_provider,
        agent_voice=agent_voice,
        domain=domain,
        caller_gender=caller_gender,
    )
    agent_context = (
        build_agent_context(
            participants.provider, participants.voice, participants.caller_gender
        )
        if settings.llm_judge
        else None
    )
    simulation.nativeness_info = evaluate_nativeness(
        simulation,
        task,
        language,
        script,
        settings=settings,
        agent_context=agent_context,
        agent_gender=participants.agent_gender,
        caller_gender=participants.caller_gender,
        caller_name=participants.caller_name,
    )


def attach_delivery(
    simulation: SimulationRun,
    *,
    settings: Optional[DeliveryJudgeSettings] = None,
) -> None:
    """Compute and attach audio delivery scores (fidelity + intonation).

    A perceptual-quality axis decoupled from reward. No-op for non-voice runs
    (no ticks → leaves ``delivery_info`` None). Runs the combined multimodal
    Gemini judge over a deterministic sample of the agent's spoken utterances
    (``settings=None`` means default ``DeliveryJudgeSettings``). Language is
    resolved for context only; delivery is evaluated for any language
    (including English).
    """
    language, _script = get_simulation_language_info(simulation)
    locale = (
        simulation.speech_environment.locale
        if simulation.speech_environment is not None
        else None
    )
    simulation.delivery_info = evaluate_delivery(
        simulation,
        language,
        locale=locale,
        settings=settings,
        seed=simulation.seed or 0,
    )


def attach_delivery_from_disk(
    simulation: SimulationRun,
    results_dir: Path,
    *,
    settings: Optional[DeliveryJudgeSettings] = None,
) -> None:
    """Compute and attach delivery scores for a STORED run, from disk audio.

    Same judge, same scoring as ``attach_delivery`` — only the audio source
    differs: per-utterance agent segments sliced from the run's stereo
    ``both.wav`` right channel by tick span (see ``delivery.disk_audio``).
    Full-duplex only; raises ``DiskAudioError`` loudly for half-duplex/mono
    layouts, missing audio, or a timeline that contradicts the ticks.
    """
    # Lazy import: keeps numpy/audio_io off the inline judging path.
    from tau2.judges.delivery.disk_audio import SimAgentAudio

    if not simulation.ticks:
        simulation.delivery_info = None
        return
    source = SimAgentAudio(simulation, results_dir)
    language, _script = get_simulation_language_info(simulation)
    locale = (
        simulation.speech_environment.locale
        if simulation.speech_environment is not None
        else None
    )
    simulation.delivery_info = evaluate_delivery(
        simulation,
        language,
        locale=locale,
        settings=settings,
        seed=simulation.seed or 0,
        wav_source=source.wav_b64_for,
    )

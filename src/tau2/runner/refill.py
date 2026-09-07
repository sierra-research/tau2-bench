"""Reconstruct the recorded run configuration for an in-place refill.

The public elicitation release needs a small, typed seam for rebuilding the
configuration that produced a checkpoint. It deliberately reports fields
that older ``Info`` records did not preserve instead of pretending those
values are known.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from tau2.config import DEFAULT_COMPLICATION_PROFILE
from tau2.data_model.simulation import (
    ComplicationProfile,
    Info,
    RunConfig,
    TextRunConfig,
    VoiceRunConfig,
)


class ReconstructedConfig(BaseModel):
    """A run config reconstructed from stored metadata and its known gaps."""

    config: RunConfig
    unrecorded: list[str] = Field(default_factory=list)


def reconstruct_config(
    info: Info,
    *,
    run_dir: str | Path,
    task_ids: list[str],
) -> ReconstructedConfig:
    """Rebuild the reproducibility-relevant settings stored in ``Info``."""
    unrecorded: list[str] = []
    if not info.user_info.llm:
        raise ValueError("The results do not record a user-simulator model")

    if info.complication_profile is None:
        unrecorded.append("complication_profile")
        profile = ComplicationProfile(DEFAULT_COMPLICATION_PROFILE)
        complication_rate = info.complication_rate
        if info.complication_rate is None:
            unrecorded.append("complication_rate")
    else:
        profile = ComplicationProfile(info.complication_profile)
        # None is authoritative for modern records: use the profile rates.
        complication_rate = info.complication_rate

    shared = dict(
        domain=info.environment_info.domain_name,
        task_ids=list(task_ids),
        llm_user=info.user_info.llm,
        llm_args_user=dict(info.user_info.llm_args or {}),
        num_trials=info.num_trials,
        max_errors=info.max_errors,
        seed=info.seed,
        save_to=str(run_dir),
        auto_resume=True,
        retrieval_config=info.retrieval_config,
        retrieval_config_kwargs=info.retrieval_config_kwargs,
        complication_profile=profile,
        complication_rate=complication_rate,
    )
    if info.audio_native_config is not None:
        config: RunConfig = VoiceRunConfig(
            **shared,
            audio_native_config=info.audio_native_config.model_copy(deep=True),
            speech_complexity=info.speech_complexity or "regular",
            channel_effects_mode=info.channel_effects_mode or "regular",
            speech_effects_mode=info.speech_effects_mode or "regular",
        )
        for field, recorded in (
            ("speech_complexity", info.speech_complexity),
            ("channel_effects_mode", info.channel_effects_mode),
            ("speech_effects_mode", info.speech_effects_mode),
        ):
            if recorded is None:
                unrecorded.append(field)
    else:
        if not info.agent_info.llm:
            raise ValueError("The results do not record a text-agent model")
        config = TextRunConfig(
            **shared,
            agent=info.agent_info.implementation,
            llm_agent=info.agent_info.llm,
            llm_args_agent=dict(info.agent_info.llm_args or {}),
            user=info.user_info.implementation,
            max_steps=info.max_steps,
        )
    return ReconstructedConfig(config=config, unrecorded=unrecorded)

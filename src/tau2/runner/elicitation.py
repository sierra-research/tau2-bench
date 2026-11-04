"""Explicit elicitation integration with the ordinary simulation runner."""

from tau2.data_model.elicitation import SampledComplication
from tau2.data_model.simulation import RunConfig, VoiceRunConfig
from tau2.data_model.tasks import StructuredUserInstructions, Task
from tau2.domains.intake.localized import SelectedIntakeContext


def sample_run_complication(
    config: RunConfig, task: Task
) -> SampledComplication | None:
    """Sample a domain-owned stimulus without changing other domains or task data."""
    if config.elicitation is None:
        return None
    from tau2.domains.intake.complications import sample_complication

    if config.language is not None and config.language != "en":
        from tau2.domains.intake.localized import sample_localized_complication

        context = localized_intake_context(config, task)
        return sample_localized_complication(
            config.seed or 0,
            task,
            context.data.for_task(context.canonical_task_id),
            rate_override=config.elicitation.rate,
            channel="voice" if isinstance(config, VoiceRunConfig) else "text",
        )

    return sample_complication(
        config.seed if config.seed is not None else 0,
        task,
        rate_override=config.elicitation.rate,
        channel="voice" if isinstance(config, VoiceRunConfig) else "text",
    )


def user_prompt_task(config: RunConfig, task: Task) -> Task:
    """Apply the sampled caller instruction to a private task copy only."""
    sampled = sample_run_complication(config, task)
    if sampled is None or not sampled.injected:
        return task
    prompted = task.model_copy(deep=True)
    instructions = prompted.user_scenario.instructions
    if isinstance(instructions, StructuredUserInstructions):
        instructions.task_instructions += f"\n\n{sampled.line}"
    else:
        prompted.user_scenario.instructions = f"{instructions}\n\n{sampled.line}"
    return prompted


def localized_intake_context(config: RunConfig, task: Task) -> SelectedIntakeContext:
    """Resolve pinned selected stimuli; never fall back to English bank entries."""
    from tau2.domains.intake.regional import IntakeScenarioBundle
    from tau2.multilingual.runtime import pin_language_artifacts

    loaded = pin_language_artifacts(config, update=False)
    if (
        loaded is None
        or not isinstance(loaded.scenario, IntakeScenarioBundle)
        or loaded.scenario.intake_localization is None
    ):
        raise ValueError(
            "Localized intake requires a reviewed regional bundle with selected stimuli"
        )
    return SelectedIntakeContext(
        data=loaded.scenario.intake_localization,
        canonical_task_id=loaded.metadata[task.id].canonical_task_id,
    )

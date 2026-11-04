"""Runtime contracts for reviewed selected intake stimuli; no factory dependencies."""

import hashlib
import random
import string
from typing import Annotated, Literal

from pydantic import Field, model_validator

from tau2.data_model.elicitation import (
    ComplicationKind,
    ComplicationParams,
    SampledComplication,
)
from tau2.data_model.tasks import Task
from tau2.domains.intake.date_display import DateDisplayFormat
from tau2.domains.intake.folds import FoldKind, fold_value
from tau2.domains.intake.time_display import TimeDisplayFormat, display_time
from tau2.utils.pydantic_utils import BaseModelNoExtra

LOCALIZED_STIMULUS_VERSION = "intake-selected-stimuli-v1"
NATIVE_SCORING_VERSION = "intake-native-decimal-digits-v2"


class SelectedPronunciation(BaseModelNoExtra):
    """A reviewed token and its exact TTS distortion, with no ASCII assumption."""

    token: Annotated[
        str,
        Field(min_length=1, description="Exact native token in the measured value."),
    ]
    reference: Annotated[
        str,
        Field(
            min_length=1,
            description="Pronunciation reference for native review; never an accepted answer.",
        ),
    ]
    distorted: Annotated[
        str,
        Field(
            min_length=1,
            description="TTS-ready mistaken pronunciation, reviewed in audio.",
        ),
    ]
    boundary: Annotated[
        Literal["word", "grapheme"],
        Field(
            default="word",
            description="Reviewed matching boundary; grapheme permits names in unsegmented native speech without splitting a written cluster.",
        ),
    ]
    reason: Annotated[
        str,
        Field(min_length=1, description="Explanation of the intended audible error."),
    ]

    @model_validator(mode="after")
    def distinct(self):
        if self.token.casefold() == self.distorted.casefold():
            raise ValueError(
                "Pronunciation distortion must differ from the source token"
            )
        return self


class SelectedStimulus(BaseModelNoExtra):
    """Only the dependencies of one frozen measured task slot."""

    canonical_task_id: Annotated[
        str, Field(description="Pinned task identity, independent of locale suffixes.")
    ]
    family: Annotated[
        Literal[
            "person_names",
            "codes",
            "phones",
            "dates",
            "times",
            "addresses",
            "properties",
            "emails",
            "medications",
            "coined",
        ],
        Field(description="Frozen measured family."),
    ]
    tier: Annotated[
        Literal["easy", "hard"],
        Field(description="Frozen difficulty assignment requiring native review."),
    ]
    field_name: Annotated[
        str, Field(description="Protocol field collected by this task.")
    ]
    written: Annotated[
        str,
        Field(
            min_length=1,
            description="Mapped written answer, identical to the golden submission.",
        ),
    ]
    date_display_format: DateDisplayFormat | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Deterministic caller-facing date format; written remains the exact ISO golden submission.",
    )
    time_display_format: TimeDisplayFormat | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Deterministic caller-facing clock notation; written remains the exact H:MM AM/PM golden submission.",
    )
    caller_spoken: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description="Source-bound initial time expression, separate from the full written clarification and canonical gold.",
    )
    decoys: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Only this selected entry's plausible wrong neighbors.",
        ),
    ]
    spelling_styles: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Reviewed locale spell-out style instructions; an empty string disables an inapplicable source style.",
        ),
    ]
    omission_instruction: Annotated[
        str,
        Field(
            default="",
            description="Reviewed instruction for initially omitting a component; empty means inapplicable.",
        ),
    ]
    pronunciations: Annotated[
        list[SelectedPronunciation],
        Field(
            default_factory=list,
            description="Only used distortion-bearing tokens; ordinary TTS receives written text.",
        ),
    ]


class IntakeLocalizationData(BaseModelNoExtra):
    """Reviewable domain guidance and compact runtime dependencies of the 200 slots."""

    version: Annotated[
        Literal["intake-selected-stimuli-v1"],
        Field(
            default=LOCALIZED_STIMULUS_VERSION,
            description="Explicit compact stimulus contract revision.",
        ),
    ]
    caller_greeting: Annotated[
        str,
        Field(
            min_length=1,
            description="Natural local equivalent of answering a phone, without identifying the caller.",
        ),
    ]
    agent_opener_template: Annotated[
        str,
        Field(
            min_length=1,
            description="Localized callback opener with exactly one {org} placeholder.",
        ),
    ]
    clarification_guidance: Annotated[
        str,
        Field(
            min_length=1,
            description="Script-specific written clarification conventions, supplied only to the caller.",
        ),
    ]
    digit_words: Annotated[
        dict[str, str],
        Field(
            description="ASCII digits to reviewed spoken names; used only by effort detection."
        ),
    ]
    stimuli: Annotated[
        list[SelectedStimulus],
        Field(
            min_length=200,
            max_length=200,
            description="Exact dependencies of the frozen task selection, never complete banks.",
        ),
    ]

    @model_validator(mode="after")
    def validate_contract(self):
        fields = [
            field
            for _, field, spec, conversion in string.Formatter().parse(
                self.agent_opener_template
            )
            if field is not None
        ]
        if fields != ["org"] or "{org}" not in self.agent_opener_template:
            raise ValueError(
                "Intake callback opener must contain exactly one plain {org} placeholder"
            )
        if any(
            value not in "0123456789" or len(value) != 1 for value in self.digit_words
        ):
            raise ValueError("Digit-word keys must be single ASCII digits")
        if len({row.canonical_task_id for row in self.stimuli}) != 200:
            raise ValueError("Intake stimuli must cover 200 distinct canonical tasks")
        for row in self.stimuli:
            if not row.canonical_task_id.startswith(f"intake_{row.family}_{row.tier}_"):
                raise ValueError(
                    "Selected stimulus family/tier disagrees with its canonical identity"
                )
        return self

    def for_task(self, canonical_task_id: str) -> SelectedStimulus:
        """Resolve explicit source identity; never guess from a localized task suffix."""
        for row in self.stimuli:
            if row.canonical_task_id == canonical_task_id:
                return row
        raise ValueError(f"No reviewed selected stimulus for {canonical_task_id}")


def sample_localized_complication(
    run_seed: int,
    task: Task,
    stimulus: SelectedStimulus,
    *,
    rate_override: float | None,
    channel: Literal["voice", "text"],
) -> SampledComplication | None:
    """Sample the compact reviewed inventory, preserving inapplicable mass as clean.

    Reviewed style and omission instructions may explicitly be empty when the
    source phenomenon is not applicable. Their probability mass remains clean.
    Restart errors operate on Unicode graphemes of the exact written value.
    """
    from tau2.domains.intake.complications import effective_kind_rates

    rates = effective_kind_rates(rate_override, stimulus.family)
    digest = hashlib.sha256(
        f"{LOCALIZED_STIMULUS_VERSION}:{run_seed}:{stimulus.canonical_task_id}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    point = rng.random()
    selected = None
    cumulative = 0.0
    for kind, rate in rates.items():
        cumulative += rate
        if point < cumulative:
            selected = kind
            break
    if selected is None:
        return None
    from tau2.domains.intake.date_display import display_date

    caller_written = (
        display_date(stimulus.written, stimulus.date_display_format)
        if stimulus.date_display_format and stimulus.family == "dates"
        else stimulus.written
    )
    if stimulus.time_display_format and stimulus.family == "times":
        caller_written = display_time(stimulus.written, stimulus.time_display_format)
    actions = {
        action.func_name: action.arguments
        for action in task.initial_state.initialization_actions
    }
    fields = actions["seed_record"]["record"]["fields"]
    measured = next(field for field in fields if field["name"] == stimulus.field_name)
    if (
        stimulus.written
        != task.evaluation_criteria.actions[0].arguments["fields"][stimulus.field_name]
    ):
        raise ValueError("Selected stimulus no longer matches golden task value")
    if selected is ComplicationKind.SPELLING_STYLE and channel == "voice":
        styles = [line for line in stimulus.spelling_styles if line.strip()]
        if not styles:
            return None
        return SampledComplication(
            kind=selected,
            line=rng.choice(styles),
            params=ComplicationParams(style="reviewed_locale"),
            catalog_version=LOCALIZED_STIMULUS_VERSION,
        )
    if selected is ComplicationKind.LAZY_OMISSION:
        if not stimulus.omission_instruction.strip():
            return None
        return SampledComplication(
            kind=selected,
            line=stimulus.omission_instruction,
            params=ComplicationParams(omitted_component="reviewed_locale"),
            catalog_version=LOCALIZED_STIMULUS_VERSION,
        )
    if selected is ComplicationKind.SPELL_CORRECTION and channel == "voice":
        import regex

        units = [
            unit
            for unit in regex.findall(r"\X", caller_written)
            if any(character.isalnum() for character in unit)
        ]
        if len(units) < 6 or len(set(units)) < 2:
            return None
        index = rng.randrange(1, len(units) - 1)
        wrong = rng.choice([unit for unit in units if unit != units[index]])
        prefix = units[:index] + [wrong]
        return SampledComplication(
            kind=selected,
            line=f"Scripted written-clarification restart: only if asked to spell or clarify the exact written units, begin with this incorrect partial sequence: {', '.join(prefix)}. Break off, correct yourself naturally in the conversation language, then give the complete correct sequence from get_entity. Never use the incorrect sequence as your ordinary spoken answer or confirm it.",
            params=ComplicationParams(
                operator="substitute_grapheme",
                error_index=index,
                extra_units=len(prefix),
                sequence_length=len(units),
                faltered_partial=", ".join(prefix),
            ),
            catalog_version=LOCALIZED_STIMULUS_VERSION,
        )
    if selected is ComplicationKind.SELF_CORRECTION:
        decoys = [
            value
            for value in stimulus.decoys
            if fold_value(FoldKind(measured["fold"]), value)
            != fold_value(FoldKind(measured["fold"]), caller_written)
        ]
        if not decoys:
            raise ValueError("Selected intake decoys collapse to the correct answer")
        decoy = rng.choice(decoys)
        return SampledComplication(
            kind=selected,
            line=f'Scripted slip: on the first request for the missing value, say "{decoy}" once, then immediately correct yourself naturally in the conversation language and give the value from get_entity. Never spell, confirm or repeat the wrong value. This single supplied slip is the only exception to tool grounding.',
            params=ComplicationParams(decoy_value=decoy),
            catalog_version=LOCALIZED_STIMULUS_VERSION,
        )
    if selected is ComplicationKind.WRONG_SLOT:
        context = [field for field in fields if field["name"] != stimulus.field_name]
        if not context:
            return None
        wrong = rng.choice(context).copy()
        if stimulus.date_display_format and wrong["fold"] == FoldKind.DATE.value:
            wrong["value"] = display_date(wrong["value"], stimulus.date_display_format)
        if stimulus.time_display_format and wrong["fold"] == FoldKind.TIME.value:
            wrong["value"] = display_time(wrong["value"], stimulus.time_display_format)
        return SampledComplication(
            kind=selected,
            line=f'Scripted misunderstanding: on the first request for the missing value, answer with this other personal fact instead: "{wrong["value"]}". When the agent asks again, fetch and answer the requested field accurately. This supplied wrong-slot value is the only exception to tool grounding.',
            params=ComplicationParams(
                off_slot_field=wrong["name"], off_slot_value=wrong["value"]
            ),
            catalog_version=LOCALIZED_STIMULUS_VERSION,
        )
    if (
        selected is ComplicationKind.MISPRONOUNCED_TERM
        and channel == "voice"
        and stimulus.pronunciations
    ):
        pronunciation = rng.choice(stimulus.pronunciations)
        return SampledComplication(
            kind=selected,
            line=pronunciation.reason,
            params=ComplicationParams(
                term=pronunciation.token, respelling=pronunciation.reference
            ),
            catalog_version=LOCALIZED_STIMULUS_VERSION,
            injected=False,
            mispronunciation=pronunciation.distorted,
        )
    return None


class SelectedIntakeContext(BaseModelNoExtra):
    """One source-checked selection at the runtime call boundary."""

    data: Annotated[
        IntakeLocalizationData,
        Field(description="Reviewed domain content from the pinned scenario."),
    ]
    canonical_task_id: Annotated[
        str, Field(description="Explicit metadata mapping for the selected task.")
    ]

"""Typed configuration and recorded stimuli for elicitation runs."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from tau2.config import (
    DEFAULT_ELICITATION_COMPLICATION_RATE,
    DEFAULT_ELICITATION_STRATEGY,
)


class ElicitationStrategy(str, Enum):
    """Collection strategies within the intake domain."""

    AGENT_DIRECTED = "agent_directed"
    SCAFFOLDED = "scaffolded"


class ElicitationConfig(BaseModel):
    """Elicitation settings resolved and recorded by the run configuration."""

    model_config = ConfigDict(extra="forbid")

    strategy: Annotated[
        ElicitationStrategy,
        Field(
            default=ElicitationStrategy(DEFAULT_ELICITATION_STRATEGY),
            description="Agent-directed collection by default; scaffolded is an ablation.",
        ),
    ]
    rate: Annotated[
        float | None,
        Field(
            default=DEFAULT_ELICITATION_COMPLICATION_RATE,
            ge=0,
            le=1,
            description="Rescale the bank rate vector to this total trigger probability. "
            "Zero is clean; None uses unscaled catalog rates. The kind mix is preserved and "
            "task-infeasible kinds still contribute their mass to no complication.",
        ),
    ]


class ComplicationKind(str, Enum):
    """Closed catalog of scripted elicitation caller behaviors."""

    SELF_CORRECTION = "self_correction"
    SPELLING_STYLE = "spelling_style"
    LAZY_OMISSION = "lazy_omission"
    WRONG_SLOT = "wrong_slot"
    MISPRONOUNCED_TERM = "mispronounced_term"
    SPELL_CORRECTION = "spell_correction"


class ComplicationParams(BaseModel):
    """Typed parameters filled by the fixed complication templates."""

    model_config = ConfigDict(extra="forbid")

    decoy_value: Annotated[
        str | None,
        Field(
            default=None,
            description="The initial incorrect value before a self-correction.",
        ),
    ]
    style: Annotated[
        str | None,
        Field(default=None, description="The selected spelling-style catalog entry."),
    ]
    omitted_component: Annotated[
        str | None,
        Field(
            default=None, description="The component initially omitted from the value."
        ),
    ]
    off_slot_field: Annotated[
        str | None,
        Field(
            default=None,
            description="The context field answered instead of the requested field.",
        ),
    ]
    off_slot_value: Annotated[
        str | None,
        Field(
            default=None,
            description="The incorrect-slot value spoken on the first ask.",
        ),
    ]
    term: Annotated[
        str | None,
        Field(
            default=None, description="The entity token intentionally mispronounced."
        ),
    ]
    respelling: Annotated[
        str | None,
        Field(
            default=None, description="The bank's canonical pronunciation reference."
        ),
    ]
    operator: Annotated[
        str | None,
        Field(
            default=None,
            description="The selected pronunciation or spell-correction operator.",
        ),
    ]
    error_index: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="Index of the incorrect or omitted spelling unit.",
        ),
    ]
    extra_units: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="Additional units spoken before correcting a spell-out.",
        ),
    ]
    faltered_partial: Annotated[
        str | None,
        Field(
            default=None, description="The exact erroneous initial spelling sequence."
        ),
    ]
    marker: Annotated[
        str | None,
        Field(
            default=None,
            description="The fixed correction phrase introducing the restart.",
        ),
    ]
    sequence_length: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Number of units in the complete correct spelling.",
        ),
    ]


class SampledComplication(BaseModel):
    """One reproducible caller stimulus recorded on the simulation result."""

    model_config = ConfigDict(extra="forbid")

    kind: Annotated[
        ComplicationKind, Field(description="The sampled scripted behavior.")
    ]
    line: Annotated[
        str,
        Field(
            description="Exact prompt addition, or descriptive text for a prompt-silent pronunciation stimulus."
        ),
    ]
    params: Annotated[
        ComplicationParams,
        Field(description="Typed parameters selected by the seeded sampler."),
    ]
    catalog_version: Annotated[
        str, Field(description="Version of the calibrated catalog and sampling rules.")
    ]
    injected: Annotated[
        bool,
        Field(
            default=True, description="Whether line was appended to the caller prompt."
        ),
    ]
    mispronunciation: Annotated[
        str | None,
        Field(
            default=None,
            description="Exact TTS substitution for the drawn term, if any.",
        ),
    ]

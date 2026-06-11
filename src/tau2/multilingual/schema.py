# Copyright Sierra
"""Schema for Language Packs and multilingual personas.

These models are the frozen interface between the shared infrastructure and
per-language content owners. A language owner authors instances of these
models (one ``LanguagePack`` per language) and never edits shared code.

Field names are language-agnostic by design (``tv_form_default``, not
``aap_default``; ``cultural_register`` is a free string) so that future
languages (e.g. Mandarin) slot in without schema changes.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from tau2.data_model.persona import PersonaConfig


class BackchannelRepertoire(BaseModel):
    """Language/persona-specific backchannel phrases and firing behavior.

    Consumed by the streaming user simulator's backchannel subsystem (the
    parameterization of that subsystem lands separately; this schema is the
    contract). When no repertoire is active, the simulator uses the English
    module-global defaults in ``voice_config.py``.
    """

    id: str = Field(description="Unique repertoire id within the pack")
    phrases: list[str] = Field(
        description="Backchannel continuer phrases, in the pack's language/script"
    )
    decision_prompt: Optional[str] = Field(
        default=None,
        description="Override for the LLM backchannel decision prompt. None falls "
        "back to the pack-level override, then to the English default.",
    )
    poisson_rate: Optional[float] = Field(
        default=None,
        description="Override for the Poisson backchannel rate (events/sec of agent "
        "speech). None falls back to the global default.",
    )


class SideTalkRepertoire(BaseModel):
    """Language/persona-specific out-of-turn (side-talk) phrases.

    Localizes ``NON_DIRECTED_PHRASES`` (e.g. speaking to family or a driver,
    not to the agent).
    """

    id: str = Field(description="Unique repertoire id within the pack")
    phrases: list[str] = Field(
        description="Non-directed speech phrases, in the pack's language/script"
    )
    events_per_minute: Optional[float] = Field(
        default=None,
        description="Override for out-of-turn speech rate. None falls back to the "
        "global default.",
    )


class AcousticPreset(BaseModel):
    """A locale-specific acoustic environment (background + burst noise files).

    File names are relative to the verified background-noise data directories,
    and may include a subdirectory (e.g. ``india/india_outdoor_traffic.wav``).
    Audio files themselves are language-owner deliverables.
    """

    id: str = Field(description="Unique preset id within the pack")
    display_name: str = Field(description="Human-readable name for visualizations")
    background_noise_files: list[str] = Field(
        default_factory=list,
        description="Continuous background noise files (relative to the continuous "
        "noise dir)",
    )
    burst_noise_files: list[str] = Field(
        default_factory=list,
        description="Burst noise files (relative to the burst noise dir)",
    )


class MultilingualPersonaConfig(PersonaConfig):
    """A language-pack persona: identity, language metadata, and behavior knobs.

    Extends the runtime :class:`PersonaConfig` (verbosity, interrupt tendency)
    with language-aware fields. The persona's full prompt contribution flows
    through the existing ``<PERSONA_GUIDELINES>`` slot via
    :meth:`to_guidelines_text` — no user-simulator changes are required for a
    persona to take effect.

    A registered persona also becomes a ``VoicePersona`` (for TTS voice-id
    resolution) automatically; see ``tau2.multilingual.registry``.
    """

    persona_id: str = Field(
        description="Globally unique persona id, e.g. 'priya_hindi_v1'. Must not "
        "collide with existing voice persona names."
    )
    display_name: str = Field(description="Human-readable name, e.g. 'Priya'")
    short_description: str = Field(
        description="One-line description for logs and visualizations"
    )
    language: str = Field(description="ISO 639-1 language code, e.g. 'hi'")
    locale: Optional[str] = Field(
        default=None, description="Region/locale tag, e.g. 'IN-MH'"
    )
    script: Optional[str] = Field(
        default=None, description="ISO 15924 script code for the matrix language, "
        "e.g. 'deva'"
    )
    matrix_language: Optional[str] = Field(
        default=None,
        description="The grammatical matrix language when code-switching, if "
        "different from `language`",
    )
    cs_density_target: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Target fraction of embedded-language (e.g. English) insertions. "
        "Prompt-authoring target only; nothing measures it post-hoc.",
    )
    tv_form_default: Optional[str] = Field(
        default=None,
        description="Default T-V politeness form toward the agent, e.g. 'aap'",
    )
    cultural_register: Optional[str] = Field(
        default=None,
        description="Free-text cultural/religious register, e.g. 'hindu_neutral', "
        "'muslim'",
    )
    pragmatics_clauses: list[str] = Field(
        default_factory=list,
        description="Freeform prompt-injectable clauses describing pragmatics: "
        "indirect-refusal patterns, code-switch triggers, frustration patterns, "
        "greeting/closing rituals.",
    )
    agent_capability_expectation: Optional[str] = Field(
        default=None,
        description="The persona's prior on the agent's fluency in their language, "
        "and how they react when it is not met.",
    )
    backchannel_repertoire_id: Optional[str] = Field(
        default=None,
        description="Id of a BackchannelRepertoire in the same pack",
    )
    burst_repertoire_id: Optional[str] = Field(
        default=None,
        description="Id of a SideTalkRepertoire in the same pack",
    )
    voice_id: Optional[str] = Field(
        default=None,
        description="ElevenLabs voice id for this persona. Overridable at runtime "
        "via TAU2_VOICE_ID_<PERSONA_ID_UPPER>.",
    )
    tts_voice_prompt: str = Field(
        default="",
        description="Speaker-identity prompt (accent, age, tone, pacing). Used as "
        "the VoicePersona prompt and injected into the user-simulator persona "
        "guidelines.",
    )
    acoustic_preset_id: Optional[str] = Field(
        default=None,
        description="Id of an AcousticPreset in the same pack",
    )

    def to_guidelines_text(self) -> Optional[str]:
        """Persona guidelines for the ``<PERSONA_GUIDELINES>`` system-prompt slot.

        Extends the base PersonaConfig guidelines (verbosity etc.) with a
        language/identity block assembled from this persona's fields.
        """
        sections: list[str] = []
        base = super().to_guidelines_text()
        if base:
            sections.append(base)

        lines: list[str] = ["## PERSONA AND LANGUAGE"]
        if self.tts_voice_prompt:
            lines.append(self.tts_voice_prompt.strip())
        lang_bits = [f"You speak {self.language}"]
        if self.matrix_language:
            lang_bits.append(f"with {self.matrix_language} as your matrix language")
        if self.script:
            lang_bits.append(f"written in the '{self.script}' script")
        lines.append(" ".join(lang_bits) + ".")
        if self.cs_density_target is not None:
            lines.append(
                f"Roughly {round(self.cs_density_target * 100)}% of your words are "
                "English insertions (code-switching); the rest stay in your "
                "language. Code-switch the way a real speaker of your profile "
                "does — never translate whole sentences."
            )
        if self.tv_form_default:
            lines.append(
                f"You default to the '{self.tv_form_default}' politeness form when "
                "addressing the agent."
            )
        if self.cultural_register:
            lines.append(
                f"Your cultural register is '{self.cultural_register}'; greetings, "
                "exclamations, and religious phrasing follow it naturally."
            )
        if self.agent_capability_expectation:
            lines.append(self.agent_capability_expectation.strip())
        for clause in self.pragmatics_clauses:
            lines.append(clause.strip())

        sections.append("\n\n".join(lines))
        return "\n\n".join(sections)


class LanguagePack(BaseModel):
    """All language-specific content for one language, bundled.

    Shared code consumes packs only through the registry
    (``tau2.multilingual.registry``); language owners author exactly one pack
    under ``tau2/multilingual/languages/<lang>/`` and register it.
    """

    language: str = Field(description="ISO 639-1 language code, e.g. 'hi'")
    display_name: str = Field(description="Human-readable language name")
    guidelines_voice_path: Optional[Path] = Field(
        default=None,
        description="Path to the localized voice user-simulator guidelines markdown "
        "(re-authored, not translated). None falls back to the English guidelines.",
    )
    personas: dict[str, MultilingualPersonaConfig] = Field(
        default_factory=dict,
        description="Personas keyed by persona_id",
    )
    backchannel_repertoires: dict[str, BackchannelRepertoire] = Field(
        default_factory=dict
    )
    side_talk_repertoires: dict[str, SideTalkRepertoire] = Field(default_factory=dict)
    acoustic_presets: dict[str, AcousticPreset] = Field(default_factory=dict)
    backchannel_decision_prompt: Optional[str] = Field(
        default=None,
        description="Pack-level override for the LLM backchannel decision prompt "
        "(language-appropriate examples and density). Persona repertoires may "
        "override it further.",
    )
    agent_language_clause: Optional[str] = Field(
        default=None,
        description="Agent-side system-prompt clause describing how to respond in "
        "this language (script conventions, mirroring the user's register, keeping "
        "tool calls in English).",
    )
    default_backchannel_poisson_rate: Optional[float] = Field(
        default=None,
        description="Language-level default backchannel Poisson rate. None keeps "
        "the global default.",
    )
    default_out_of_turn_events_per_minute: Optional[float] = Field(
        default=None,
        description="Language-level default side-talk rate. None keeps the global "
        "default.",
    )

    @model_validator(mode="after")
    def _validate_pack(self) -> "LanguagePack":
        for persona_id, persona in self.personas.items():
            if persona.persona_id != persona_id:
                raise ValueError(
                    f"Persona key '{persona_id}' does not match its persona_id "
                    f"'{persona.persona_id}'"
                )
            if persona.language != self.language:
                raise ValueError(
                    f"Persona '{persona_id}' has language '{persona.language}' but "
                    f"the pack language is '{self.language}'"
                )
            if (
                persona.backchannel_repertoire_id is not None
                and persona.backchannel_repertoire_id
                not in self.backchannel_repertoires
            ):
                raise ValueError(
                    f"Persona '{persona_id}' references unknown backchannel "
                    f"repertoire '{persona.backchannel_repertoire_id}'"
                )
            if (
                persona.burst_repertoire_id is not None
                and persona.burst_repertoire_id not in self.side_talk_repertoires
            ):
                raise ValueError(
                    f"Persona '{persona_id}' references unknown side-talk "
                    f"repertoire '{persona.burst_repertoire_id}'"
                )
            if (
                persona.acoustic_preset_id is not None
                and persona.acoustic_preset_id not in self.acoustic_presets
            ):
                raise ValueError(
                    f"Persona '{persona_id}' references unknown acoustic preset "
                    f"'{persona.acoustic_preset_id}'"
                )
        if self.guidelines_voice_path is not None and not Path(
            self.guidelines_voice_path
        ).exists():
            raise ValueError(
                f"Guidelines file does not exist: {self.guidelines_voice_path}"
            )
        return self

    def get_persona(self, persona_id: str) -> Optional[MultilingualPersonaConfig]:
        return self.personas.get(persona_id)

    def get_backchannel_repertoire(
        self, persona: MultilingualPersonaConfig
    ) -> Optional[BackchannelRepertoire]:
        if persona.backchannel_repertoire_id is None:
            return None
        return self.backchannel_repertoires.get(persona.backchannel_repertoire_id)

    def get_side_talk_repertoire(
        self, persona: MultilingualPersonaConfig
    ) -> Optional[SideTalkRepertoire]:
        if persona.burst_repertoire_id is None:
            return None
        return self.side_talk_repertoires.get(persona.burst_repertoire_id)

    def get_acoustic_preset(
        self, persona: MultilingualPersonaConfig
    ) -> Optional[AcousticPreset]:
        if persona.acoustic_preset_id is None:
            return None
        return self.acoustic_presets.get(persona.acoustic_preset_id)

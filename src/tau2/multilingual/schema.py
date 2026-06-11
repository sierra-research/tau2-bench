# Copyright Sierra
"""Schema for Language Packs and multilingual personas.

These models are the frozen interface between the shared infrastructure and
per-language content owners. A language owner authors instances of these
models (one ``LanguagePack`` per language) and never edits shared code.

The schema is deliberately small: typed fields exist only where code branches
on them (registry keys, voice/preset references, injectable phrase lists).
All behavioral language content — register, code-switching, politeness,
rituals — lives in the author-written ``tts_voice_prompt`` and
``pragmatics_clauses``, because free text is the only representation that
generalizes across languages. Persona differences are expressed through those
prompts plus per-persona phrase lists; language-wide behavior (decision
prompts, firing rates) lives on the pack.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from tau2.data_model.persona import PersonaConfig


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
        default=None,
        description="ISO 3166-2 subdivision code, e.g. 'IN-MH' (Maharashtra), "
        "'IN-TG' (Telangana). Trajectory-tagging metadata only.",
    )
    script: Optional[str] = Field(
        default=None, description="ISO 15924 script code for the matrix language, "
        "e.g. 'deva'"
    )
    pragmatics_clauses: list[str] = Field(
        default_factory=list,
        description="Freeform prompt-injected clauses, authored natively by the "
        "language owner. This is where ALL behavioral language content lives: "
        "register, code-switching behavior, politeness forms, greeting/closing "
        "rituals, indirect-refusal patterns, frustration patterns, and the "
        "persona's expectation of the agent's language fluency.",
    )
    backchannel_phrases: Optional[list[str]] = Field(
        default=None,
        description="Backchannel continuer phrases for this persona, in the "
        "persona's language/script. None falls back to the English defaults.",
    )
    side_talk_phrases: Optional[list[str]] = Field(
        default=None,
        description="Out-of-turn (side-talk) phrases for this persona — speech "
        "directed at family, a driver, etc., not at the agent. None falls back "
        "to the English defaults.",
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

        Extends the base PersonaConfig guidelines (verbosity etc.) with the
        author's verbatim persona content: the speaker-identity prompt and the
        pragmatics clauses. No prose is generated from metadata fields — the
        language owner controls all language/register phrasing natively.
        """
        sections: list[str] = []
        base = super().to_guidelines_text()
        if base:
            sections.append(base)

        lines: list[str] = []
        if self.tts_voice_prompt:
            lines.append(self.tts_voice_prompt.strip())
        for clause in self.pragmatics_clauses:
            lines.append(clause.strip())
        if lines:
            sections.append("\n\n".join(["## PERSONA AND LANGUAGE"] + lines))

        return "\n\n".join(sections) if sections else None


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
    acoustic_presets: dict[str, AcousticPreset] = Field(default_factory=dict)
    backchannel_decision_prompt: Optional[str] = Field(
        default=None,
        description="Language-level override for the LLM backchannel decision "
        "prompt (language-appropriate examples and density). Must contain a "
        "{conversation_history} placeholder. None keeps the English default.",
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

    def get_acoustic_preset(
        self, persona: MultilingualPersonaConfig
    ) -> Optional[AcousticPreset]:
        if persona.acoustic_preset_id is None:
            return None
        return self.acoustic_presets.get(persona.acoustic_preset_id)

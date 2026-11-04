"""Versioned English elicitation prompts, selected only by domain runtime wiring."""

from tau2.config import AUDIO_NATIVE_PROVIDER_TYPES
from tau2.data_model.persona import PersonaConfig
from tau2.domains.intake.data_model import CallbackOrder
from tau2.domains.intake.utils import INTAKE_DATA_DIR

ELICITATION_PROMPT_VERSION = "1.5.0"
OUTBOUND_ROLE_REMINDER = "REMINDER: You are the CUSTOMER answering a call from a company's agent. Respond as the customer would, providing only information from your scenario or tools. Do NOT respond as the customer service agent."
_SPELL_OFFER_LINE = "- If the agent asks you to repeat your name, email, or other personal details, offer to spell it out letter by letter (as shown in examples above).\n"

AGENT_DIRECTED_AUDIO_NATIVE_VOICE_INSTRUCTION = """
You are a customer service agent handling a VOICE CALL with a customer.

# Important Voice Call Considerations

1. Respond naturally and conversationally as you would in a real phone call

2. Try to be helpful and always follow the policy.
""".strip()

AGENT_DIRECTED_CASCADED_MODEL_INSTRUCTION = """
You are a customer service agent handling a VOICE CALL with a customer.

# Important Voice Call Considerations

1. For the conversation, you will see transcribed speech, not written text. Expect:
- Misspellings of names, emails, or technical terms
- Missing or incorrect punctuation (periods, commas)
- Run-on sentences or incomplete thoughts

2. Respond naturally and conversationally as you would in a real phone call. Do not use bullets (numbered or unnumbered) or markdown formatting.

3. Try to be helpful and always follow the policy.
""".strip()


def render_callback_order(order: CallbackOrder) -> str:
    """Render only the agent-visible work order, without record or caller values."""
    return f"# Callback order\n\n{order.model_dump_json(indent=2)}"


def get_user_guidelines(
    is_voice: bool, agent_directed: bool, persona: PersonaConfig | None = None
) -> str:
    """Load the outbound caller frame for an elicitation user simulator."""
    from tau2.data_model.language import RuntimePersona

    if isinstance(persona, RuntimePersona) and persona.language != "en":
        return _LOCALIZED_OUTBOUND_GUIDELINES.format(
            language=persona.display_language, channel="voice" if is_voice else "text"
        )
    name = (
        "simulation_guidelines_voice_tools_outbound.md"
        if is_voice
        else "simulation_guidelines_tools_outbound.md"
    )
    guidelines = (INTAKE_DATA_DIR / name).read_text()
    if is_voice and agent_directed:
        if _SPELL_OFFER_LINE not in guidelines:
            raise ValueError(
                "Outbound spelling guidance changed; update the agent-directed prompt."
            )
        guidelines = guidelines.replace(_SPELL_OFFER_LINE, "")
    return guidelines


def get_agent_instructions(provider: str, agent_directed: bool) -> str | None:
    """Provide an unprescribed strategy only for the agent-directed arm."""
    if not agent_directed:
        return None
    if AUDIO_NATIVE_PROVIDER_TYPES.get(provider) == "cascaded":
        return AGENT_DIRECTED_CASCADED_MODEL_INSTRUCTION
    return AGENT_DIRECTED_AUDIO_NATIVE_VOICE_INSTRUCTION


LOCALIZED_ELICITATION_PROMPT_VERSION = "intake-native-prompts-v1"
_LOCALIZED_OUTBOUND_GUIDELINES = """You are the CUSTOMER answering an outbound {channel} call from a company's agent. Speak in {language}, using the scenario's greeting. You did not initiate this call.
Answer only what the agent asks. Fetch each factual value with get_entity before saying it; tools and measurement notes are private. Never invent missing values or expose internal field names. Follow the scenario's cooperation, spelling, read-back and stopping rules.
Use natural speech in the selected language. A request to repeat is not a request to spell. Give written clarification only when asked, using the scenario's script-specific conventions. Preserve the exact written identity, units and numeric meaning; never volunteer a transliteration. When explicitly rendering separate written units, separate those units with comma and space for measurement.
<PERSONA_GUIDELINES>
Use ###STOP### only after the caller has finished the call, as specified by the scenario.
"""

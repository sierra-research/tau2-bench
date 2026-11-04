"""Versioned domain agent guidance, shared by text and voice prompt composition."""

from tau2.data_model.language import DomainAgentGuidance

BANKING_AGENT_GUIDANCE_VERSION = "banking-english-knowledge-v1"
BANKING_AGENT_GUIDANCE = (
    "The knowledge base is in English. Use it to answer the customer's questions, "
    "but always respond to the customer in {target_language}."
)


INTAKE_AGENT_GUIDANCE_VERSION = "intake-written-form-v2"
INTAKE_AGENT_GUIDANCE = (
    "When a spoken value could have more than one written form, and the field "
    "description does not specify the required script or numeral system, ask "
    "which written form the person intends. Follow any format explicitly "
    "required by the field description; otherwise preserve the person's "
    "clarified written form. Do not infer the script from pronunciation alone. "
    "Use the target language’s script for medication names; preserve the stated strength and units."
)

_ENGLISH_TRANSFER_LINE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
_TRANSFER_LINES = {
    "es": "Le voy a comunicar con un agente. Por favor, permanezca en la línea.",
    "hi": "आपको एक प्रतिनिधि से जोड़ा जा रहा है। कृपया प्रतीक्षा करें।",
    "ko": "상담원에게 연결해 드리겠습니다. 잠시만 기다려 주세요.",
    "pt": "Vou transferir você para um atendente. Por favor, aguarde.",
    "zh": "现在为您转接人工客服，请稍候。",
}


def localized_transfer_policy(policy: str, domain: str, language: str) -> str:
    """Use the call language for the policy's spoken transfer line."""
    if (
        domain not in {"airline", "retail", "telecom", "telecom-workflow"}
        or language == "en"
    ):
        return policy
    line = _TRANSFER_LINES.get(language)
    if line is not None:
        return policy.replace(_ENGLISH_TRANSFER_LINE, line)
    return policy.replace(
        f"send the message '{_ENGLISH_TRANSFER_LINE}' to the user",
        "tell the user in the call's language that they are being transferred "
        "to a human agent and should hold",
    )


def domain_agent_guidance(
    domain: str, target_language: str
) -> DomainAgentGuidance | None:
    """Render the domain's instruction in addition to the shared language clause."""
    if domain == "intake":
        return DomainAgentGuidance(
            text=INTAKE_AGENT_GUIDANCE,
            version=INTAKE_AGENT_GUIDANCE_VERSION,
        )
    if domain != "banking_knowledge":
        return None
    if not target_language.strip():
        raise ValueError("Domain agent guidance requires the pack display language")
    return DomainAgentGuidance(
        text=BANKING_AGENT_GUIDANCE.format(target_language=target_language),
        version=BANKING_AGENT_GUIDANCE_VERSION,
    )

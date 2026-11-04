"""The spoken transfer instruction follows the resolved call language."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.multilingual.agent_guidance import localized_transfer_policy
from tau2.runner.build import agent_domain_policy

DATA = Path(__file__).resolve().parents[2] / "data" / "tau2" / "domains"
SOURCE_LINE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
POLICIES = {
    "airline": DATA / "airline" / "policy.md",
    "retail": DATA / "retail" / "policy.md",
    "telecom": DATA / "telecom" / "main_policy.md",
    "telecom-workflow": DATA / "telecom" / "main_policy.md",
}
LOCALIZED_LINES = {
    "es": "Le voy a comunicar con un agente. Por favor, permanezca en la línea.",
    "hi": "आपको एक प्रतिनिधि से जोड़ा जा रहा है। कृपया प्रतीक्षा करें।",
    "ko": "상담원에게 연결해 드리겠습니다. 잠시만 기다려 주세요.",
    "pt": "Vou transferir você para um atendente. Por favor, aguarde.",
    "zh": "现在为您转接人工客服，请稍候。",
}


@pytest.mark.parametrize("domain", POLICIES)
@pytest.mark.parametrize("language,line", LOCALIZED_LINES.items())
def test_transfer_line_is_localized_in_agent_policy(domain, language, line):
    source = POLICIES[domain].read_text()
    environment = SimpleNamespace(
        get_policy=lambda: source,
        get_domain_name=lambda: domain,
    )

    prompt = agent_domain_policy(
        environment, "Respond in the call language.", language=language
    )

    assert SOURCE_LINE not in prompt
    assert f"send the message '{line}' to the user" in prompt
    assert "transfer_to_human_agents" in prompt
    assert prompt.endswith("Respond in the call language.")


def test_english_transfer_line_stays_unchanged():
    source = POLICIES["airline"].read_text()
    assert localized_transfer_policy(source, "airline", "en") == source


def test_other_domain_policy_stays_unchanged():
    source = POLICIES["airline"].read_text()
    assert localized_transfer_policy(source, "intake", "es") == source


def test_new_language_uses_localized_instruction_instead_of_english_line():
    source = POLICIES["airline"].read_text()
    policy = localized_transfer_policy(source, "airline", "fr")
    assert SOURCE_LINE not in policy
    assert "tell the user in the call's language" in policy

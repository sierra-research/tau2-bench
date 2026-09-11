# Copyright Sierra
"""The <PERSONA_GUIDELINES> slot must be filled exactly once.

The guidelines both HAVE the slot (a line consisting of exactly the
placeholder) and MENTION the placeholder in prose. A plain ``str.replace``
matched the prose mention too, splicing the persona + localization block into
the middle of that sentence and duplicating it in every run, in every
language, in both simulators. Caught by the es prompt+bed review packet.
"""

import pytest

from tau2.multilingual.registry import list_language_packs
from tau2.user.user_simulator import UserSimulator, substitute_persona_slot
from test_multilingual.conftest import first_persona, make_voice_sim

PERSONA_HEADER = "## PERSONA AND LANGUAGE"


class TestSubstitutePersonaSlot:
    def test_only_the_standalone_line_is_a_slot(self):
        guidelines = (
            "Intro mentioning `<PERSONA_GUIDELINES>` in prose.\n"
            "<PERSONA_GUIDELINES>\n"
            "Outro.\n"
        )
        filled = substitute_persona_slot(guidelines, "PERSONA BLOCK")
        assert filled.count("PERSONA BLOCK") == 1
        assert "prose.\nPERSONA BLOCK\nOutro." in filled
        assert "`<PERSONA_GUIDELINES>`" in filled  # prose mention untouched

    def test_replacement_block_is_not_backslash_processed(self):
        filled = substitute_persona_slot(
            "<PERSONA_GUIDELINES>\n", r"speak \1 naturally \g<0>"
        )
        assert filled == "speak \\1 naturally \\g<0>\n"


@pytest.mark.parametrize("language", sorted(list_language_packs()))
class TestSingleInjectionEveryPack:
    def test_text_prompt_injects_persona_once(self, language):
        sim = UserSimulator(
            llm="dummy",
            instructions="scenario",
            persona_config=first_persona(language),
        )
        prompt = sim.system_prompt
        assert prompt.count(PERSONA_HEADER) == 1
        assert "<PERSONA_GUIDELINES>" not in [
            line.strip() for line in prompt.splitlines()
        ]

    def test_voice_prompt_injects_persona_once(self, language):
        sim = make_voice_sim(first_persona(language))
        prompt = sim.system_prompt
        assert prompt.count(PERSONA_HEADER) == 1
        assert "<PERSONA_GUIDELINES>" not in [
            line.strip() for line in prompt.splitlines()
        ]

# Copyright Sierra
"""Hindi-specific pack content tests.

Generic pack validity (guidelines file, backchannel level, agent-language
clause, native-script content, sampling flow, English regression) is covered
for ALL languages by test_pack_invariants.py / test_language_pack.py /
test_language_persona_sampling.py. This file keeps only the Hindi-unique
contracts: gender-agreement mechanics and the localization-block move of the
Indian spell-letter convention.
"""

import re

from tau2.multilingual.registry import (
    get_language_pack,
    get_localization_guidelines,
    get_multilingual_persona,
)
from tau2.multilingual.schema import MultilingualPersonaConfig
from tau2.user.user_simulator import get_global_user_sim_guidelines_voice

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

RISHIKA_ID = "rishika_hindi_v1"
IMRAN_ID = "imran_hindi_v1"


def test_pack_discovered():
    pack = get_language_pack("hi")
    assert pack is not None
    assert pack.language == "hi"
    assert pack.display_name == "Hindi"
    assert set(pack.personas) == {RISHIKA_ID, IMRAN_ID}


class TestGenderMechanics:
    def test_guidelines_text_carries_gender_clause(self):
        # The assigned persona's gender drives first-person grammatical
        # agreement in the live user-sim prompt (it does not flow via
        # translation, which stays persona-neutral).
        for persona_id, gender in [(RISHIKA_ID, "female"), (IMRAN_ID, "male")]:
            persona = get_multilingual_persona(persona_id)[1]
            text = persona.to_guidelines_text()
            assert "first-person gender agreement" in text
            assert f"You are {gender}." in text

    def test_gender_clause_omitted_when_no_gender_tag(self):
        # Packs/personas without a gender tag emit no gender clause (the clause
        # is gated; gender is not a required tag dimension).
        persona = MultilingualPersonaConfig(
            persona_id="x_xx_v1",
            display_name="X",
            short_description="A speaker.",
            language="xx",
            script="latn",
            tts_voice_prompt="A speaker.",
            tags={},
        )
        text = persona.to_guidelines_text() or ""
        assert "first-person gender agreement" not in text

    def test_translation_guidance_carries_gender_mechanics(self):
        """Gender inflection is Hindi-specific, so the full gender mechanics
        live in the pack's translation_guidance, not in the shared templates:
        the generative rule (default/unmarked masculine, don't bake a gender,
        runtime-injected) and the role-neutral judging rule (the default form is
        CORRECT and must NOT be flagged; only INCONSISTENT gender is a defect)."""
        pack = get_language_pack("hi")
        guidance = pack.translation_guidance
        assert guidance is not None
        lower = guidance.lower()
        # Generative mechanics: default/unmarked masculine, runtime-injected.
        assert "unmarked" in lower or "default" in lower
        assert "runtime" in lower
        # Judging rule: default form is correct; only inconsistent gender is a defect.
        assert "correct" in lower
        assert "inconsisten" in lower


def test_spell_letter_convention_lives_in_localization_block():
    # Indian spell-letter convention (not NATO) lives in the pack's
    # localization block: the English guidelines every run reads carry no
    # phone/amount/spelling bullets; the block appended at the persona slot does.
    text = get_global_user_sim_guidelines_voice()
    assert "Bombay" not in text
    block = get_localization_guidelines("hi", mode="voice")
    assert block is not None and "Bombay" in block

# Copyright Sierra
"""Tests for gender-aware Voice Design audition.

Synthetic/fake only — no real ElevenLabs calls. Persona voices are auditioned
in their own grammatical gender so first-person verb agreement in the audition
text matches the persona, and each persona in a pack is auditioned with the
text matching its own gender.
"""

from __future__ import annotations

import textwrap


class TestGenderAwareAudition:
    def test_hindi_male_and_female_variants_differ(self):
        from tau2.multilingual.factory.voice_generation import audition_text_for

        male = audition_text_for("hi", "male")
        female = audition_text_for("hi", "female")
        assert male != female
        # Gendered first-person verb agreement is the whole point.
        assert "जा रहा हूँ" in male and "चाहता हूँ" in male
        assert "जा रही हूँ" in female and "चाहती हूँ" in female
        assert len(male) >= 100 and len(female) >= 100

    def test_unknown_gender_falls_back_to_male_variant(self):
        from tau2.multilingual.factory.voice_generation import audition_text_for

        assert audition_text_for("hi", None) == audition_text_for("hi", "male")
        assert audition_text_for("hi", "nonbinary") == audition_text_for("hi", "male")

    def test_unlisted_language_uses_english_default(self):
        from tau2.multilingual.factory.voice_generation import (
            _DEFAULT_AUDITION_TEXT,
            audition_text_for,
        )

        assert audition_text_for("xx", "female") == _DEFAULT_AUDITION_TEXT


class _FakePreview:
    def __init__(self, gid):
        self.generated_voice_id = gid


class _FakeDesignResult:
    def __init__(self, previews):
        self.previews = previews


class _FakeVoice:
    def __init__(self, vid):
        self.voice_id = vid


class _FakeTextToVoice:
    """Records the audition `text` passed to design() for each persona."""

    def __init__(self, recorder):
        self._rec = recorder

    def design(self, *, voice_description, text, **kw):
        self._rec.append(text)
        return _FakeDesignResult([_FakePreview("gen-123")])

    def create(self, *, voice_name, voice_description, generated_voice_id):
        return _FakeVoice("voice-abc")


class _FakeEleven:
    def __init__(self, recorder):
        self.text_to_voice = _FakeTextToVoice(recorder)


class TestPackVoicesAuditionPerPersona:
    def test_each_persona_auditioned_in_its_own_gender(self, tmp_path, monkeypatch):
        from tau2.multilingual.factory import voice_generation as vg

        pack = tmp_path / "pack.yaml"
        pack.write_text(
            textwrap.dedent(
                """\
                personas:
                  rishika_test:
                    display_name: Rishika
                    tts_voice_prompt: "a calm professional voice"
                    tags:
                      gender: female
                  imran_test:
                    display_name: Imran
                    tts_voice_prompt: "a patient older voice"
                    tags:
                      gender: male
                """
            )
        )
        monkeypatch.setattr(vg, "_pack_path", lambda language: pack)

        recorded: list[str] = []
        results = vg.generate_pack_voices(
            "hi", client=_FakeEleven(recorded), force=True
        )

        assert [r.status for r in results] == ["created", "created"]
        # Insertion order: female persona first, male second.
        assert recorded[0] == vg.audition_text_for("hi", "female")
        assert recorded[1] == vg.audition_text_for("hi", "male")

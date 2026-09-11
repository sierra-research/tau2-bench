# Copyright Sierra
"""Tests for the factory guardrails (and the factory state safety rails).

The guardrails are pure functions over pack files: the real hi pack must
pass ``validate_final_pack``, and each hand-rolled broken draft must fail
with a readable problem string naming what is wrong. No LLM is involved
anywhere (FactoryLLM is never instantiated): the guardrails are deterministic
by design so they can gate model output.
"""

import re
from pathlib import Path

import pytest
import yaml

from tau2.multilingual.factory.guardrails import (
    GuardrailReport,
    validate_final_pack,
    validate_pack_draft,
)
from tau2.multilingual.factory.state import (
    PACK_DRAFT_FILENAME,
    FactoryProject,
    FactoryStage,
    factory_file_path,
)

GUIDELINES_FILENAME = "simulation_guidelines_voice_xg.md"


def base_pack() -> dict:
    """A minimal draft that passes every guardrail."""
    return {
        "language": "xg",
        "display_name": "Guardlang",
        "guidelines_voice_path": GUIDELINES_FILENAME,
        "personas": {
            "gala_xg_v1": {
                "persona_id": "gala_xg_v1",
                "display_name": "Gala",
                "short_description": "Synthetic guardrail-test persona",
                "language": "xg",
                "script": "deva",
                "pragmatics_clauses": ["You speak plainly."],
                "acoustic_preset_id": "factory_test_env",
                "tags": {
                    "code_switch": "low",
                    "formality": "medium",
                    "english_tolerance": "medium",
                    "age_band": "30s",
                    "gender": "female",
                    "register": "student",
                    "closing_style": "brisk",
                },
            }
        },
        "acoustic_presets": {
            "factory_test_env": {
                "id": "factory_test_env",
                "display_name": "Factory Test Env",
                # Real verified files, resolved exactly as the runtime does.
                "background_noise_files": ["hi_IN/busy_street_iphone_mic.wav"],
                "burst_noise_files": ["car_horn.wav"],
            }
        },
        "backchannel_level": "medium",
    }


def validate(tmp_path: Path, data: dict) -> GuardrailReport:
    (tmp_path / GUIDELINES_FILENAME).write_text(
        "# Guidelines\n<PERSONA_GUIDELINES>\n###STOP###\n"
    )
    (tmp_path / PACK_DRAFT_FILENAME).write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    )
    return validate_pack_draft(tmp_path)


def validate_with_guidelines(tmp_path: Path, guidelines_text: str) -> GuardrailReport:
    """Validate a base draft whose guidelines body is ``guidelines_text``."""
    (tmp_path / GUIDELINES_FILENAME).write_text(guidelines_text)
    (tmp_path / PACK_DRAFT_FILENAME).write_text(
        yaml.safe_dump(base_pack(), allow_unicode=True, sort_keys=False)
    )
    return validate_pack_draft(tmp_path)


def assert_fails_with(report: GuardrailReport, needle: str):
    assert not report.ok
    assert any(needle in problem for problem in report.problems), (
        f"no problem mentions '{needle}':\n" + "\n".join(report.problems)
    )


class TestFinalPacks:
    """The shipped packs pass the full factory guardrail suite."""

    @pytest.mark.parametrize("language", ["hi"])
    def test_real_pack_passes(self, language):
        report = validate_final_pack(language)
        assert report.ok, "\n".join(report.problems)

    def test_unknown_language_reports_missing_file(self):
        report = validate_final_pack("nope")
        assert_fails_with(report, "does not exist")


class TestDraftGuardrails:
    def test_base_draft_passes(self, tmp_path):
        report = validate(tmp_path, base_pack())
        assert report.ok, "\n".join(report.problems)

    def test_draft_rejects_english_skeleton_guidelines(self, tmp_path):
        """A draft that leaves the INSTRUCTIONS in English (only examples
        localized) violates the full-translation contract and must fail."""
        report = validate_with_guidelines(
            tmp_path,
            "# Guidelines\n"
            "## Information Disclosure\n"
            "- Only share information that is explicitly provided.\n"
            "- Make the agent work for information.\n"
            "- Start with minimal information.\n"
            "<PERSONA_GUIDELINES>\n###STOP###\n",
        )
        assert_fails_with(report, "English instruction prose")

    def test_draft_accepts_fully_localized_guidelines(self, tmp_path):
        """The same structure, but instructions rendered in the target language
        (no English instruction markers), passes."""
        report = validate_with_guidelines(
            tmp_path,
            "# दिशानिर्देश\n"
            "## जानकारी साझा करना\n"
            "- केवल वही जानकारी साझा करें जो स्पष्ट रूप से दी गई हो।\n"
            "- एजेंट को जानकारी के लिए मेहनत कराएँ।\n"
            "<PERSONA_GUIDELINES>\n###STOP###\n",
        )
        assert report.ok, "\n".join(report.problems)

    def test_missing_draft_file(self, tmp_path):
        report = validate_pack_draft(tmp_path)
        assert_fails_with(report, "does not exist")

    def test_not_a_mapping(self, tmp_path):
        (tmp_path / PACK_DRAFT_FILENAME).write_text("- just\n- a list\n")
        report = validate_pack_draft(tmp_path)
        assert_fails_with(report, "not a YAML mapping")

    def test_unknown_top_level_key(self, tmp_path):
        data = base_pack()
        data["surprise_field"] = True
        assert_fails_with(
            validate(tmp_path, data), "unknown top-level key 'surprise_field'"
        )

    def test_persona_key_persona_id_mismatch(self, tmp_path):
        data = base_pack()
        data["personas"]["wrong_key"] = data["personas"].pop("gala_xg_v1")
        assert_fails_with(validate(tmp_path, data), "does not match its persona_id")

    def test_dangling_acoustic_preset_id(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["acoustic_preset_id"] = "no_such_preset"
        assert_fails_with(
            validate(tmp_path, data), "unknown acoustic preset 'no_such_preset'"
        )

    def test_nonexistent_top_level_noise_wav_lists_available_files(self, tmp_path):
        """A missing TOP-LEVEL/shared bed (no subfolder) is a HARD error: it
        must already exist in the repo."""
        data = base_pack()
        data["acoustic_presets"]["factory_test_env"]["background_noise_files"] = [
            "underwater_market.wav"
        ]
        report = validate(tmp_path, data)
        assert_fails_with(report, "'underwater_market.wav' not found")
        # The message helps the author by listing what IS available.
        assert any(
            "hi_IN/busy_street_iphone_mic.wav" in problem for problem in report.problems
        )

    def test_missing_locale_bed_is_a_nonfatal_note_not_a_problem(self, tmp_path):
        """A missing CONTINUOUS bed under a LOCALE subfolder is an authoring
        leftover (bed production is retired) — non-fatal, surfaced as a note,
        so draft and finalize still pass (the ordering trap)."""
        data = base_pack()
        data["acoustic_presets"]["factory_test_env"]["background_noise_files"] = [
            "zz/busy_street_iphone_mic.wav"
        ]
        report = validate(tmp_path, data)
        assert report.ok, "\n".join(report.problems)
        assert not any("zz/busy_street_iphone_mic.wav" in p for p in report.problems)
        assert any("zz/busy_street_iphone_mic.wav" in note for note in report.notes)
        assert any("locale-bed production is retired" in note for note in report.notes)

    def test_missing_locale_burst_stays_fatal(self, tmp_path):
        """Bursts are always shared/top-level; even a subfolder'd missing burst
        stays a HARD error (only continuous beds get the pending-bed grace)."""
        data = base_pack()
        data["acoustic_presets"]["factory_test_env"]["burst_noise_files"] = [
            "ro/car_horn.wav"
        ]
        report = validate(tmp_path, data)
        assert_fails_with(report, "'ro/car_horn.wav' not found")

    def test_unknown_script_code(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["script"] = "qaax"
        assert_fails_with(
            validate(tmp_path, data),
            "extend SCRIPT_RANGES in tau2.multilingual.invariants",
        )

    def test_long_non_directed_phrase_fails(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["non_directed_phrases"] = [
            "एक मिनट रुको, मैं अभी फ़ोन पर बात कर रही हूँ"
        ]
        assert_fails_with(validate(tmp_path, data), "the budget is 4 words")

    def test_too_many_non_directed_phrases_fails(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["non_directed_phrases"] = ["एक मिनट"] * 5
        assert_fails_with(validate(tmp_path, data), "carries 5 non_directed_phrases")

    def test_clipped_non_directed_phrases_pass(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["non_directed_phrases"] = [
            "एक मिनट रुको",
            "बेटा, ग्राहक संभालो",
        ]
        assert validate(tmp_path, data).ok

    def test_long_backchannel_phrase_fails(self, tmp_path):
        """A continuer that grew into an utterance is the same defect the
        count bound guards against, arriving by a different route."""
        data = base_pack()
        data["personas"]["gala_xg_v1"]["backchannel_phrases"] = ["जी हाँ बिल्कुल ठीक"]
        assert_fails_with(validate(tmp_path, data), "the budget is 2 words")

    def test_too_many_backchannel_phrases_fails(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["backchannel_phrases"] = ["हम्म"] * 3
        assert_fails_with(validate(tmp_path, data), "carries 3 backchannel_phrases")

    def test_pure_continuer_passes(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["backchannel_phrases"] = ["हम्म"]
        assert validate(tmp_path, data).ok

    def test_unknown_tag_dimension(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["tags"]["spirit_animal"] = "owl"
        assert_fails_with(
            validate(tmp_path, data), "unknown tag dimension 'spirit_animal'"
        )

    def test_invalid_tag_value(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["tags"]["formality"] = "extreme"
        assert_fails_with(
            validate(tmp_path, data), "tag 'formality' has invalid value 'extreme'"
        )

    def test_missing_required_tag_dimension(self, tmp_path):
        data = base_pack()
        del data["personas"]["gala_xg_v1"]["tags"]["english_tolerance"]
        assert_fails_with(
            validate(tmp_path, data),
            "missing required tag dimension 'english_tolerance'",
        )

    def test_missing_gender_tag_fails(self, tmp_path):
        """gender is REQUIRED: the first-person gender-agreement clause in
        to_guidelines_text only fires when tags.gender is male/female, so a
        draft missing it would never reach the user simulator (bugbot #303)."""
        data = base_pack()
        del data["personas"]["gala_xg_v1"]["tags"]["gender"]
        assert_fails_with(
            validate(tmp_path, data),
            "missing required tag dimension 'gender'",
        )

    def test_all_problems_collected_at_once(self, tmp_path):
        """The report is exhaustive, not fail-fast."""
        data = base_pack()
        data["surprise_field"] = True
        data["personas"]["gala_xg_v1"]["script"] = "qaax"
        del data["personas"]["gala_xg_v1"]["tags"]["age_band"]
        report = validate(tmp_path, data)
        assert len(report.problems) >= 3


class TestDisplayNameGenderGuardrail:
    """display_name ↔ tags.gender consistency (Olivia♂ / Noah♀ insurance)."""

    def test_known_female_name_tagged_male_fails(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["display_name"] = "Olivia"
        data["personas"]["gala_xg_v1"]["tags"]["gender"] = "male"
        assert_fails_with(
            validate(tmp_path, data),
            "known female given name but tags.gender is 'male'",
        )

    def test_known_male_name_tagged_female_fails(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["display_name"] = "Noah"
        data["personas"]["gala_xg_v1"]["tags"]["gender"] = "female"
        assert_fails_with(
            validate(tmp_path, data),
            "known male given name but tags.gender is 'female'",
        )

    def test_diacritics_normalized_before_lookup(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["display_name"] = "Étienne"
        data["personas"]["gala_xg_v1"]["tags"]["gender"] = "female"
        assert_fails_with(
            validate(tmp_path, data),
            "known male given name but tags.gender is 'female'",
        )

    def test_unknown_name_passes(self, tmp_path):
        # 'Gala' is not in the catalog — the guardrail stays silent (the base
        # pack passing at all also covers this, but keep it explicit).
        data = base_pack()
        assert "Gala" == data["personas"]["gala_xg_v1"]["display_name"]
        report = validate(tmp_path, data)
        assert not any("given name" in p for p in report.problems)

    def test_matching_name_and_tag_pass(self, tmp_path):
        data = base_pack()
        data["personas"]["gala_xg_v1"]["display_name"] = "Olivia"
        assert data["personas"]["gala_xg_v1"]["tags"]["gender"] == "female"
        report = validate(tmp_path, data)
        assert report.ok, "\n".join(report.problems)

    def test_every_shipped_pack_persona_is_consistent(self):
        """Coverage sweep: no shipped pack ships an Olivia-tagged-male."""
        import yaml as _yaml

        from tau2.multilingual.factory.name_genders import known_given_name_gender
        from tau2.multilingual.localize_lib import multilingual_data_dir

        pack_paths = sorted(multilingual_data_dir().glob("*/pack.yaml"))
        assert pack_paths
        for pack_path in pack_paths:
            data = _yaml.safe_load(pack_path.read_text())
            personas = data.get("personas") or {}
            if isinstance(personas, list):
                personas = {p["persona_id"]: p for p in personas}
            for pid, persona in personas.items():
                tag_gender = (persona.get("tags") or {}).get("gender")
                catalog = known_given_name_gender(persona.get("display_name") or "")
                if catalog is not None and tag_gender in ("male", "female"):
                    assert catalog == tag_gender, (
                        f"{pack_path.parent.name}/{pid}: display_name "
                        f"'{persona.get('display_name')}' is {catalog} but "
                        f"tags.gender is {tag_gender}"
                    )


# Scripts whose presence in note PROSE (outside quotes) proves the note was
# not written in English. Latin-script languages cannot be caught this way, so
# the persona check below is the guard that applies to every pack.
_NATIVE_SCRIPT_RANGES = (
    ("؀", "ۿ"),  # Arabic
    ("ऀ", "ॿ"),  # Devanagari
    ("Ѐ", "ӿ"),  # Cyrillic
    ("぀", "ヿ"),  # Hiragana/Katakana
    ("一", "鿿"),  # Han
    ("가", "힯"),  # Hangul
)


# Every quote pair the packs use to cite a native form.
_QUOTED = re.compile(
    "|".join(
        (
            r"'[^']*'",
            r'"[^"]*"',
            r"‘[^’]*’",
            r"“[^”]*”",
            r"«[^»]*»",
            r"「[^」]*」",
        )
    )
)


def _strip_quoted(text: str) -> str:
    """Drop quoted spans — a cited native form is DATA, not prose."""
    return _QUOTED.sub("", text)


class TestShippedGlossaryNotes:
    """Glossary notes are English prose, persona-neutral, native forms quoted.

    The notes render into the unconditionally-English prompt scaffold
    (``LocalizationPackConfig.to_prompt_text`` -> ``english_prompt_string``),
    so a target-language note puts two languages on one prompt line. Seven of
    the twelve two-domain packs had drifted to target-language notes in ONE of
    their two domains before this was pinned; several also named the casual
    persona, which the persona-neutrality rule forbids.
    """

    def _glossaries(self):
        import yaml as _yaml

        from tau2.multilingual.localize_lib import multilingual_data_dir

        pack_paths = sorted(multilingual_data_dir().glob("*/pack.yaml"))
        assert pack_paths
        for pack_path in pack_paths:
            data = _yaml.safe_load(pack_path.read_text())
            localization = data.get("localization") or {}
            for domain, entries in (
                localization.get("domain_glossaries") or {}
            ).items():
                yield pack_path, data, domain, entries

    def test_notes_never_name_a_persona(self):
        """A note may name a REGISTER; naming the speaker is the defect."""
        for pack_path, data, domain, entries in self._glossaries():
            personas = data.get("personas") or {}
            if isinstance(personas, list):
                personas = {p["persona_id"]: p for p in personas}
            names = {
                persona["display_name"].split()[0]
                for persona in personas.values()
                if persona.get("display_name")
            }
            for entry in entries:
                note = entry.get("note")
                if not note:
                    continue
                named = sorted(n for n in names if n in note)
                assert not named, (
                    f"{pack_path.parent.name}/{domain}/{entry['term_id']}: "
                    f"note names persona(s) {named} — say 'casual register' "
                    f"instead: {note!r}"
                )

    def test_notes_are_not_target_language_prose(self):
        """The note's PROSE is English; a cited native form is not prose.

        Measured as a ratio rather than "no native characters at all": plenty
        of legitimate English notes cite a native form without quoting it
        ("established loanword — 便 is used for a specific numbered flight").
        Those read as English. What this rejects is a note whose prose, once
        cited forms are removed, is still mostly non-Latin — i.e. the note was
        written in the target language.
        """
        for pack_path, _data, domain, entries in self._glossaries():
            for entry in entries:
                note = entry.get("note")
                if not note:
                    continue
                prose = _strip_quoted(note)
                native = sum(
                    any(lo <= ch <= hi for lo, hi in _NATIVE_SCRIPT_RANGES)
                    for ch in prose
                )
                latin = sum(ch.isascii() and ch.isalpha() for ch in prose)
                assert native < latin or native == 0, (
                    f"{pack_path.parent.name}/{domain}/{entry['term_id']}: "
                    f"note prose is target-language ({native} native chars vs "
                    f"{latin} Latin) — write the note in English and quote any "
                    f"cited form: {note!r}"
                )


class TestFactoryStateSafety:
    """The _factory workspace must never grow a discoverable pack.yaml."""

    def test_pack_yaml_path_is_refused(self):
        with pytest.raises(ValueError, match="pack_draft.yaml"):
            factory_file_path("xg", "pack.yaml")

    def test_draft_filename_is_allowed(self):
        path = factory_file_path("xg", PACK_DRAFT_FILENAME)
        assert path.name == PACK_DRAFT_FILENAME
        assert "_factory" in str(path)

    def test_project_round_trip(self, factory_dir):
        project = FactoryProject.load_or_create("xg", script="deva", domain="airline")
        assert project.path.exists()
        assert project.pack_revision == 0
        assert all(status == "pending" for status in project.stages.values())

        project.pack_revision += 1
        project.checklist.voice_ids = True
        project.mark_stage(FactoryStage.DRAFT, success=True)

        loaded = FactoryProject.load("xg")
        assert loaded == project
        # load_or_create must not clobber an existing project.
        assert FactoryProject.load_or_create("xg") == project

    def test_project_rejects_unknown_stage(self):
        with pytest.raises(ValueError):
            FactoryProject(language="xg", stages={"oops": "done"})

    def test_project_rejects_missing_stage(self):
        with pytest.raises(ValueError, match="missing factory stages"):
            FactoryProject(language="xg", stages={"draft": "done"})

    def test_load_missing_project_returns_none(self, factory_dir):
        assert FactoryProject.load("zz") is None

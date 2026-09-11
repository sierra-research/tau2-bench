# Copyright Sierra
"""Move #1 + #3 tests: deterministic pack assembly and cross-consistency checks.

Covers the code-generated pack fields (``pack_assembly``) — experiment-block
derivation, agent_language_clause templating, firing-rate defaults + overrides,
acoustic-preset scaffolding with locale-namespaced beds — plus the deterministic
cross-consistency checks. HARD guardrails (run in ``_validate_pack_file``,
fail the draft) are asserted via ``validate_pack_draft``; quality heuristics
that tolerate legitimate variation are asserted as TEST-only invariants over a
fixture pack, not as draft-failing guardrails.
"""

from __future__ import annotations

import yaml

from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN
from tau2.multilingual.factory import pack_assembly
from tau2.multilingual.factory.author_form import AuthorForm
from tau2.multilingual.factory.guardrails import _validate_pack_file

SW_FORM = AuthorForm.model_validate(
    {
        "language": "sw",
        "display_name": "Swahili",
        "script": "latn",
        "locale_notes": "Coastal vs inland accents.",
        "personas": [
            {"name": "Amara", "sketch": "Formal 50s coastal speaker."},
            {"name": "Baraka", "sketch": "Casual 20s urban speaker."},
        ],
        "preferences": "Keep it natural.",
        # Locale beds are OPT-IN (default off -> no acoustic scaffolding);
        # these tests exercise the opted-in scaffolding shape.
        "locale_beds": True,
    }
)


# The same form WITHOUT the locale-beds opt-in: the shipped default. No
# acoustic presets are scaffolded and personas carry no acoustic wiring.
SW_FORM_NO_BEDS = AuthorForm.model_validate(
    SW_FORM.model_dump(mode="json") | {"locale_beds": False}
)


class TestExperimentDerivation:
    def test_preset_name_and_suffix_derived(self):
        # The default domain (telecom) is not the legacy unsuffixed one, so
        # the arm name carries the ``_<domain>`` suffix.
        exp = pack_assembly.build_experiment_block(SW_FORM)
        assert exp["preset_name"] == "multilingual_v1_swahili_telecom"
        assert exp["main_arm_name"] == "swahili_telecom"
        assert exp["task_set_suffix"] == "sw"

    def test_legacy_domain_keeps_bare_arm_name(self):
        exp = pack_assembly.build_experiment_block(
            SW_FORM, domain=pack_assembly.UNSUFFIXED_ARM_DOMAIN
        )
        assert exp["main_arm_name"] == "swahili"
        assert exp["preset_name"] == "multilingual_v1_swahili"

    def test_multiword_display_name_first_token(self):
        form = SW_FORM.model_copy(update={"display_name": "Mandarin Chinese"})
        exp = pack_assembly.build_experiment_block(form)
        assert exp["main_arm_name"] == "mandarin_telecom"
        assert exp["preset_name"] == "multilingual_v1_mandarin_telecom"

    def test_genuine_decisions_come_from_args_not_invented(self):
        exp = pack_assembly.build_experiment_block(
            SW_FORM,
            domain="retail",
            smoke_task_stem="7",
        )
        assert exp["domain"] == "retail"
        assert exp["smoke_task_stem"] == "7"
        # The shared English baseline is its own pack/preset — no per-language
        # baseline persona is emitted here.
        assert "baseline_persona_id" not in exp

    def test_documented_defaults_when_unspecified(self):
        exp = pack_assembly.build_experiment_block(SW_FORM)
        assert exp["domain"] == DEFAULT_MULTILINGUAL_DOMAIN
        assert exp["smoke_task_stem"] == pack_assembly.default_smoke_task_stem_for(
            DEFAULT_MULTILINGUAL_DOMAIN
        )

    def test_typed_form_field_overrides_main_arm_name(self):
        """Overrides ride the TYPED AuthorForm fields (no prose-regex channel)."""
        form = SW_FORM.model_copy(update={"main_arm_name": "kiswahili"})
        exp = pack_assembly.build_experiment_block(form)
        assert exp["main_arm_name"] == "kiswahili_telecom"
        assert exp["preset_name"] == "multilingual_v1_kiswahili_telecom"

    def test_typed_form_fields_override_domain_and_smoke(self):
        form = SW_FORM.model_copy(update={"domain": "retail", "smoke_task_stem": "7"})
        exp = pack_assembly.build_experiment_block(form)
        assert exp["domain"] == "retail"
        assert exp["smoke_task_stem"] == "7"
        # A prose mention in preferences is just prose — it must NOT override.
        prose = SW_FORM.model_copy(update={"preferences": "domain: retail"})
        assert pack_assembly.build_experiment_block(prose)["domain"] == (
            DEFAULT_MULTILINGUAL_DOMAIN
        )

    def test_preset_name_derived_via_shared_helper(self):
        """The generator's preset_name == preset_name_for(main_arm_name).

        This is the single-source-of-truth contract: the guardrail derives the
        expected name through the SAME helper, so the convention is defined once
        and the two cannot drift (see test_guardrail_uses_same_preset_helper)."""
        exp = pack_assembly.build_experiment_block(SW_FORM)
        assert exp["preset_name"] == pack_assembly.preset_name_for(exp["main_arm_name"])


class TestAgentLanguageClause:
    def test_clause_names_display_name_and_script(self):
        clause = pack_assembly.build_agent_language_clause(SW_FORM)
        assert "Swahili" in clause
        assert "Latin script" in clause
        # The constant tail is preserved.
        assert "mirror the user's code-switching register" in clause
        assert "tool calls" in clause

    def test_known_scripts_get_named(self):
        hi = SW_FORM.model_copy(update={"display_name": "Hindi", "script": "deva"})
        assert "Devanagari script" in pack_assembly.build_agent_language_clause(hi)
        zh = SW_FORM.model_copy(update={"display_name": "Mandarin", "script": "hans"})
        assert (
            "Simplified Chinese characters"
            in pack_assembly.build_agent_language_clause(zh)
        )


class TestFiringRates:
    def test_defaults_applied(self):
        fields = pack_assembly.build_deterministic_pack_fields(SW_FORM)
        assert (
            fields["default_out_of_turn_events_per_minute"]
            == pack_assembly.DEFAULT_OUT_OF_TURN_EVENTS_PER_MINUTE
        )

    def test_typed_form_field_overrides_rates(self):
        form = SW_FORM.model_copy(update={"default_out_of_turn_events_per_minute": 1.7})
        fields = pack_assembly.build_deterministic_pack_fields(form)
        assert fields["default_out_of_turn_events_per_minute"] == 1.7


class TestAcousticScaffolding:
    def test_locale_namespaced_continuous_beds(self):
        presets = pack_assembly.build_acoustic_presets(SW_FORM)
        outdoor = presets["sw_outdoor_traffic"]
        household = presets["sw_household_multi_gen"]
        # Locale-flavored continuous beds are under the <lang>/ subfolder.
        assert outdoor["background_noise_files"] == ["sw/busy_street_iphone_mic.wav"]
        assert household["background_noise_files"] == [
            "sw/medium_size_room_tv_news_iphone_mic.wav"
        ]

    def test_office_bed_and_bursts_are_shared_top_level(self):
        presets = pack_assembly.build_acoustic_presets(SW_FORM)
        office = presets["sw_office"]
        # Generic office bed reuses the shared top-level English file.
        assert office["background_noise_files"] == ["people_talking.wav"]
        # Every burst is a shared top-level English file (no subfolder).
        for preset in presets.values():
            for burst in preset["burst_noise_files"]:
                assert "/" not in burst

    def test_default_form_scaffolds_no_acoustics(self):
        # The shipped default: no locale-beds opt-in -> no presets, no
        # acoustic_presets key in the assembled fields at all (the runtime
        # then uses the stock shared-English environment selection).
        assert pack_assembly.build_acoustic_presets(SW_FORM_NO_BEDS) == {}
        assert pack_assembly.acoustic_preset_ids(SW_FORM_NO_BEDS) == []
        fields = pack_assembly.build_deterministic_pack_fields(SW_FORM_NO_BEDS)
        assert "acoustic_presets" not in fields


class TestExperimentGuardrail:
    """HARD guardrail: experiment naming invariants fail the draft."""

    def _write_pack(self, tmp_path, *, mutate=None):
        fields = pack_assembly.build_deterministic_pack_fields(SW_FORM)
        # Minimal but valid creative content so only the experiment is at issue.
        fields["personas"] = {
            "amara_sw_v1": {
                "persona_id": "amara_sw_v1",
                "display_name": "Amara",
                "short_description": "x",
                "language": "sw",
                "script": "latn",
                "acoustic_preset_id": "sw_office",
                "tags": {
                    "code_switch": "low",
                    "formality": "high",
                    "english_tolerance": "low",
                    "age_band": "50s",
                    "gender": "female",
                },
            }
        }
        guidelines = tmp_path / "guidelines_draft.md"
        guidelines.write_text("<PERSONA_GUIDELINES>\n###STOP###\n")
        fields["guidelines_voice_path"] = "guidelines_draft.md"
        fields["backchannel_level"] = "medium"
        if mutate:
            mutate(fields)
        pack_path = tmp_path / "pack_draft.yaml"
        pack_path.write_text(
            yaml.safe_dump(fields, sort_keys=False, allow_unicode=True)
        )
        return pack_path

    def test_clean_assembly_passes(self, tmp_path):
        report = _validate_pack_file(self._write_pack(tmp_path))
        assert report.ok, "\n".join(report.problems)

    def test_bad_preset_name_fails(self, tmp_path):
        def mutate(fields):
            fields["experiments"][0]["preset_name"] = "wrong_name"

        report = _validate_pack_file(self._write_pack(tmp_path, mutate=mutate))
        assert not report.ok
        assert any("preset_name" in p for p in report.problems)

    def test_guardrail_uses_same_preset_helper(self):
        """The guardrail derives its expected preset_name through the SAME
        helper as the generator (single source of truth, no re-hardcoded
        pattern). Asserting the imported symbol is the very same object catches
        any future copy that would let the convention drift."""
        import tau2.multilingual.factory.guardrails as guardrails

        assert guardrails.preset_name_for is pack_assembly.preset_name_for

    def test_missing_required_experiment_key_fails(self, tmp_path):
        """A block missing any required ExperimentSpec field fails validate —
        run presets cannot be generated from it, and the failure must surface
        at pack-validation time, not as a skipped preset at run time."""

        def mutate(fields):
            del fields["experiments"][0]["smoke_task_stem"]
            del fields["experiments"][0]["description"]

        report = _validate_pack_file(self._write_pack(tmp_path, mutate=mutate))
        assert not report.ok
        # pydantic reports one problem line per missing field.
        for key in ("smoke_task_stem", "description"):
            assert any(key in p for p in report.problems)

    def test_bad_task_set_suffix_fails(self, tmp_path):
        def mutate(fields):
            fields["experiments"][0]["task_set_suffix"] = "xx"

        report = _validate_pack_file(self._write_pack(tmp_path, mutate=mutate))
        assert not report.ok
        assert any("task_set_suffix" in p for p in report.problems)

    def test_dangling_acoustic_preset_id_fails(self, tmp_path):
        def mutate(fields):
            fields["personas"]["amara_sw_v1"]["acoustic_preset_id"] = "nope"

        report = _validate_pack_file(self._write_pack(tmp_path, mutate=mutate))
        assert not report.ok
        assert any("nope" in p for p in report.problems)

    def test_guidelines_missing_persona_slot_fails(self, tmp_path):
        pack_path = self._write_pack(tmp_path)
        # Overwrite the guidelines body to drop the slot.
        (tmp_path / "guidelines_draft.md").write_text("no slot here\n###STOP###\n")
        report = _validate_pack_file(pack_path)
        assert not report.ok
        assert any("PERSONA_GUIDELINES" in p for p in report.problems)

    def test_guidelines_missing_control_tokens_fails(self, tmp_path):
        pack_path = self._write_pack(tmp_path)
        (tmp_path / "guidelines_draft.md").write_text("<PERSONA_GUIDELINES>\n")
        report = _validate_pack_file(pack_path)
        assert not report.ok
        assert any("control token" in p for p in report.problems)

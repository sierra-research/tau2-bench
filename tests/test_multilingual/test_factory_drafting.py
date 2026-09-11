# Copyright Sierra
"""Tests for two-call pack drafting and finalize (FakeFactoryLLM only).

Every LLM interaction is scripted through ``FakeFactoryLLM``. Drafting is two
chained creative calls — Call A (``factory_draft_personas``, JSON: persona
content + translation_guidance) then Call B (``factory_draft_artifacts``, one
fenced block: the guidelines markdown) — plus the
deterministic code-generated fields (experiment / clause / rates / acoustics)
that ``pack_assembly`` produces. TARGETED repair re-runs only the call a
guardrail failure localizes to (``factory_draft_personas_repair`` /
``factory_draft_artifacts_repair``). The toy Toylang pack supplies the scripted
creative outputs; the broken-pack catalog provides invalid variants.
"""

from pathlib import Path

import pytest
import yaml

import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN
from tau2.multilingual.factory.author_form import (
    AuthorForm,
    FactoryDraftError,
    extract_fenced_blocks,
    parse_author_form,
)
from tau2.multilingual.factory.drafting import (
    DRAFT_ARTIFACTS_CALL_NAME,
    DRAFT_PERSONAS_CALL_NAME,
    DRAFT_TEXT_ARTIFACTS_CALL_NAME,
    GUIDELINES_DRAFT_FILENAME,
    GUIDELINES_TEXT_DRAFT_FILENAME,
    REPAIR_ARTIFACTS_CALL_NAME,
    REPAIR_PERSONAS_CALL_NAME,
    REPAIR_TEXT_ARTIFACTS_CALL_NAME,
    _parse_personas_response,
    run_draft,
)
from tau2.multilingual.factory.finalize import finalize
from tau2.multilingual.factory.localization_drafting import (
    DRAFT_LOCALIZATION_CALL_NAME,
)
from tau2.multilingual.factory.state import (
    PACK_DRAFT_FILENAME,
    FactoryProject,
    FactoryStage,
    StageStatus,
)
from tau2.multilingual.tags import REQUIRED_TAG_DIMENSIONS
from test_multilingual.factory_testing.draft_scripts import (
    TOY_FORM,
    TOY_FORM_OBJECT,
    TOY_PRESET_IDS,
    TOY_TEXT_GUIDELINES,
    artifacts_response,
    draft_scripts,
    guidelines_text,
    localization_response,
    personas_response,
    text_artifacts_response,
    toy_personas_for_call_a,
)
from test_multilingual.factory_testing.fake_llm import FakeFactoryLLM
from test_multilingual.factory_testing.toy_language import (
    TOY_GUIDELINES_FILENAME,
    toy_pack_yaml_dict,
)


def broken_artifacts_response() -> str:
    """A Call-B response whose guidelines fail a guardrail (no PERSONA slot)."""
    broken_md = guidelines_text().replace("<PERSONA_GUIDELINES>", "REMOVED")
    return f"```markdown\n{broken_md}```\n"


@pytest.fixture
def form_path(tmp_path) -> Path:
    path = tmp_path / "toylang_form.md"
    path.write_text(TOY_FORM)
    return path


class TestAuthorForm:
    def test_markdown_form_parses(self, form_path):
        form = parse_author_form(form_path)
        assert form.language == "tl"
        assert form.script == "latn"
        assert [p.name for p in form.personas] == ["Tessa", "Orin"]

    def test_plain_yaml_form_parses(self, tmp_path):
        path = tmp_path / "form.yaml"
        path.write_text(extract_fenced_blocks(TOY_FORM)["yaml"][0])
        assert parse_author_form(path) == parse_author_form_reference()

    def test_missing_file_is_readable(self, tmp_path):
        with pytest.raises(FactoryDraftError, match="does not exist"):
            parse_author_form(tmp_path / "nope.md")

    def test_validation_errors_name_the_fields(self, tmp_path):
        path = tmp_path / "bad_form.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "language": "tl",
                    "display_name": "Toylang",
                    "script": "latn",
                    # only one persona: violates min_length=2
                    "personas": [{"name": "Solo", "sketch": "alone"}],
                }
            )
        )
        with pytest.raises(FactoryDraftError) as excinfo:
            parse_author_form(path)
        message = str(excinfo.value)
        assert "personas" in message
        assert "factory_form_example.md" in message

    def test_non_mapping_form_is_readable(self, tmp_path):
        path = tmp_path / "list_form.yaml"
        path.write_text("- not\n- a\n- mapping\n")
        with pytest.raises(FactoryDraftError, match="YAML mapping"):
            parse_author_form(path)

    def test_non_dict_persona_raises_not_silently_dropped(self):
        """A non-dict persona entry in Call A must fail loudly rather than be
        silently skipped (which would yield too few personas; bugbot #305)."""
        good = toy_personas_for_call_a()[0]
        data = {"personas": [good, "i am not a persona dict"]}
        with pytest.raises(FactoryDraftError, match="non-dict persona at index 1"):
            _parse_personas_response(data, call_name=DRAFT_PERSONAS_CALL_NAME)


def parse_author_form_reference() -> AuthorForm:
    return AuthorForm.model_validate(
        yaml.safe_load(extract_fenced_blocks(TOY_FORM)["yaml"][0])
    )


def test_guidelines_voice_path_problem_routes_to_call_a():
    """Bugbot #305: a `guidelines_voice_path` pack-field schema error must NOT be
    misrouted to Call-B-only repair (the old bare 'guidelines' marker matched it
    and skipped Call A, leaving the pack problem unfixed). Genuine guidelines-body
    problems still route to Call B only."""
    from tau2.multilingual.factory.drafting import (
        _problem_is_text_artifact,
        _problem_is_voice_artifact,
        _repair_targets,
    )

    # A guidelines_voice_path pack-field schema error is a pack (Call A) problem,
    # not a Call-B artifact problem: it must re-run everything.
    pack_problem = "schema: guidelines_voice_path: Input should be a valid string"
    assert _problem_is_voice_artifact(pack_problem) is False
    assert _problem_is_text_artifact(pack_problem) is False
    assert _repair_targets([pack_problem]) == (True, True, True)

    # A voice guidelines-body problem re-runs only Call B.
    slot_problem = (
        "guidelines file guidelines_draft.md is missing the literal "
        "<PERSONA_GUIDELINES> slot"
    )
    assert _problem_is_voice_artifact(slot_problem) is True
    assert _problem_is_text_artifact(slot_problem) is False
    assert _repair_targets([slot_problem]) == (False, True, False)

    # A TEXT guidelines-body problem re-runs only Call C.
    text_slot_problem = (
        "guidelines file guidelines_text_draft.md is missing the literal "
        "<PERSONA_GUIDELINES> slot"
    )
    assert _problem_is_text_artifact(text_slot_problem) is True
    assert _problem_is_voice_artifact(text_slot_problem) is False
    assert _repair_targets([text_slot_problem]) == (False, False, True)


class TestRunDraft:
    def test_valid_draft_written_and_clean(self, factory_dir, form_path):
        llm = FakeFactoryLLM(draft_scripts())
        outcome = run_draft("tl", form_path, llm)

        assert outcome.report.ok, "\n".join(outcome.report.problems)
        assert not outcome.repair_attempted
        # Exactly the four chained creative calls, in order (personas, voice
        # guidelines, text guidelines, localization).
        assert [c.call_name for c in llm.calls] == [
            DRAFT_PERSONAS_CALL_NAME,
            DRAFT_ARTIFACTS_CALL_NAME,
            DRAFT_TEXT_ARTIFACTS_CALL_NAME,
            DRAFT_LOCALIZATION_CALL_NAME,
        ]

        # Files land in the workspace; the draft references the draft
        # guidelines so it validates self-contained.
        assert outcome.pack_draft_path == factory_dir / "tl" / PACK_DRAFT_FILENAME
        written = yaml.safe_load(outcome.pack_draft_path.read_text())
        assert written["guidelines_voice_path"] == GUIDELINES_DRAFT_FILENAME
        assert written["guidelines_text_path"] == GUIDELINES_TEXT_DRAFT_FILENAME
        assert outcome.guidelines_draft_path.read_text() == guidelines_text()
        assert outcome.text_guidelines_draft_path.read_text() == TOY_TEXT_GUIDELINES

        # Revision bumped; declared final guidelines filenames recorded —
        # under the pack's guidelines/ subdir.
        assert outcome.pack_revision == 1
        project = FactoryProject.load("tl")
        assert project.pack_revision == 1
        assert project.stages[FactoryStage.DRAFT] == StageStatus.DONE
        assert project.guidelines_final_filename == (
            f"guidelines/{TOY_GUIDELINES_FILENAME}"
        )
        assert project.guidelines_text_final_filename == (
            "guidelines/simulation_guidelines_text_tl.md"
        )

    def test_deterministic_fields_assembled(self, factory_dir, form_path):
        """Move #1: the experiment / clause / rates / acoustics come from code,
        not the creative calls, and are present + consistent on disk."""
        llm = FakeFactoryLLM(draft_scripts())
        outcome = run_draft("tl", form_path, llm)
        assert outcome.report.ok, "\n".join(outcome.report.problems)
        written = yaml.safe_load(outcome.pack_draft_path.read_text())

        # Experiment block: derived naming + language suffix (the default
        # domain is telecom, which carries the ``_<domain>`` arm suffix).
        assert (
            written["experiments"][0]["preset_name"]
            == "multilingual_v1_toylang_telecom"
        )
        assert written["experiments"][0]["main_arm_name"] == "toylang_telecom"
        assert written["experiments"][0]["task_set_suffix"] == "tl"
        assert written["experiments"][0]["domain"] == DEFAULT_MULTILINGUAL_DOMAIN
        # No per-language baseline persona: the English baseline is its own pack.
        assert "baseline_persona_id" not in written["experiments"][0]

        # Agent language clause templated from display_name + script.
        assert "Toylang" in written["agent_language_clause"]
        assert "Latin script" in written["agent_language_clause"]

        # Rates + acoustic scaffolding are deterministic.
        assert written["default_out_of_turn_events_per_minute"] is not None
        assert set(written["acoustic_presets"]) == set(TOY_PRESET_IDS)
        # Locale-flavored continuous bed is namespaced under the LOCALE subfolder
        # (language_COUNTRY, derived from the personas' locale codes) so the path
        # matches the locale-bed naming convention.
        outdoor = written["acoustic_presets"]["tl_outdoor_traffic"]
        assert outdoor["background_noise_files"] == ["tl_TL/busy_street_iphone_mic.wav"]

    def test_lang_form_mismatch_refused(self, factory_dir, form_path):
        llm = FakeFactoryLLM({})
        with pytest.raises(FactoryDraftError, match="mismatched"):
            run_draft("zz", form_path, llm)
        assert llm.calls == []

    def test_redraft_refreshes_project_script_and_domain(self, factory_dir, form_path):
        """M1: load_or_create ignores its defaults when the project exists, so
        a re-draft that corrects the script (or domain) must refresh the
        persisted project explicitly — the form is the source of truth."""
        stale = FactoryProject.load_or_create("tl", script="deva", domain="retail")
        assert stale.script == "deva" and stale.domain == "retail"

        run_draft("tl", form_path, FakeFactoryLLM(draft_scripts()))

        project = FactoryProject.load("tl")
        assert project.script == "latn"  # the form's script
        assert project.domain == DEFAULT_MULTILINGUAL_DOMAIN

    def test_artifact_failure_reruns_only_call_b(self, factory_dir, form_path):
        """Targeted repair: a guidelines (Call B) failure re-runs only the
        artifacts call — not the persona call."""
        broken_md = guidelines_text().replace("<PERSONA_GUIDELINES>", "REMOVED")
        broken_artifacts = f"```markdown\n{broken_md}```\n"
        llm = FakeFactoryLLM(
            {
                DRAFT_PERSONAS_CALL_NAME: [personas_response()],
                DRAFT_ARTIFACTS_CALL_NAME: [broken_artifacts],
                DRAFT_TEXT_ARTIFACTS_CALL_NAME: [text_artifacts_response()],
                DRAFT_LOCALIZATION_CALL_NAME: [localization_response()],
                REPAIR_ARTIFACTS_CALL_NAME: [artifacts_response()],  # clean repair
            }
        )
        outcome = run_draft("tl", form_path, llm)

        assert outcome.repair_attempted
        assert outcome.report.ok, "\n".join(outcome.report.problems)
        # Only Call B re-ran; Calls A, C, and D were NOT repeated.
        assert [c.call_name for c in llm.calls] == [
            DRAFT_PERSONAS_CALL_NAME,
            DRAFT_ARTIFACTS_CALL_NAME,
            DRAFT_TEXT_ARTIFACTS_CALL_NAME,
            DRAFT_LOCALIZATION_CALL_NAME,
            REPAIR_ARTIFACTS_CALL_NAME,
        ]
        # The repair prompt carried the guardrail finding.
        repair_prompt = llm.prompts_for(REPAIR_ARTIFACTS_CALL_NAME)[0]
        assert "PERSONA_GUIDELINES" in repair_prompt
        assert "Guardrail problems" in repair_prompt

    def test_persona_failure_reruns_call_a_b_and_c(self, factory_dir, form_path):
        """Targeted repair: a persona/pack problem re-runs Call A, then
        re-derives Call B AND Call C from the repaired personas (both depend on
        Call A's personas)."""
        # A schema-valid gender/name mismatch is a Call A guardrail problem.
        broken = toy_pack_yaml_dict()
        broken["personas"]["tessa_tl_v1"]["display_name"] = "Olivia"
        broken["personas"]["tessa_tl_v1"]["tags"]["gender"] = "male"
        llm = FakeFactoryLLM(
            draft_scripts(
                broken,
                repair_personas=[personas_response()],  # clean personas
                repair_artifacts=[artifacts_response()],
                repair_text_artifacts=[text_artifacts_response()],
            )
        )
        outcome = run_draft("tl", form_path, llm)

        assert outcome.repair_attempted
        assert outcome.report.ok, "\n".join(outcome.report.problems)
        # Call A re-ran, and Calls B and C were re-derived from it; the
        # schema-validated localization block (Call D) carried through.
        assert [c.call_name for c in llm.calls] == [
            DRAFT_PERSONAS_CALL_NAME,
            DRAFT_ARTIFACTS_CALL_NAME,
            DRAFT_TEXT_ARTIFACTS_CALL_NAME,
            DRAFT_LOCALIZATION_CALL_NAME,
            REPAIR_PERSONAS_CALL_NAME,
            REPAIR_ARTIFACTS_CALL_NAME,
            REPAIR_TEXT_ARTIFACTS_CALL_NAME,
        ]

    def test_persona_repair_preserves_omitted_guidance_fields(
        self, factory_dir, form_path
    ):
        """Bugbot #305: a persona-only repair fragment may omit
        translation_guidance / agent_greeting. The prior values must carry
        forward so assembly does not erase previously-good content."""
        broken = toy_pack_yaml_dict()
        broken["personas"]["tessa_tl_v1"]["display_name"] = "Olivia"
        broken["personas"]["tessa_tl_v1"]["tags"]["gender"] = "male"
        # The repaired fragment fixes the tag but OMITS the guidance keys.
        repaired_fragment = {"personas": toy_personas_for_call_a()}
        llm = FakeFactoryLLM(
            draft_scripts(
                broken,
                repair_personas=[repaired_fragment],
                repair_artifacts=[artifacts_response()],
                repair_text_artifacts=[text_artifacts_response()],
            )
        )
        outcome = run_draft("tl", form_path, llm)
        assert outcome.report.ok, "\n".join(outcome.report.problems)
        written = yaml.safe_load(outcome.pack_draft_path.read_text())
        expected = personas_response(broken)
        assert written["translation_guidance"] == expected["translation_guidance"]
        assert written["agent_greeting"] == expected["agent_greeting"]

    def test_artifact_repair_preserves_omitted_guidelines(self, factory_dir, form_path):
        """A Call-B repair that returns a BLANK markdown block must keep the
        prior (valid) guidelines rather than dropping them. Triggered by a
        guidelines-body problem, so only the markdown is re-derived."""
        broken_md = guidelines_text().replace("<PERSONA_GUIDELINES>", "REMOVED")
        initial_artifacts = f"```markdown\n{broken_md}```\n"
        # Repair returns a blank markdown block (whitespace only).
        repair_resp = "```markdown\n   \n```\n"
        llm = FakeFactoryLLM(
            {
                DRAFT_PERSONAS_CALL_NAME: [personas_response()],
                DRAFT_ARTIFACTS_CALL_NAME: [initial_artifacts],
                DRAFT_TEXT_ARTIFACTS_CALL_NAME: [text_artifacts_response()],
                DRAFT_LOCALIZATION_CALL_NAME: [localization_response()],
                REPAIR_ARTIFACTS_CALL_NAME: [repair_resp],
            }
        )
        outcome = run_draft("tl", form_path, llm)
        # The guidelines problem is artifact-only -> only Call B re-ran.
        assert [c.call_name for c in llm.calls] == [
            DRAFT_PERSONAS_CALL_NAME,
            DRAFT_ARTIFACTS_CALL_NAME,
            DRAFT_TEXT_ARTIFACTS_CALL_NAME,
            DRAFT_LOCALIZATION_CALL_NAME,
            REPAIR_ARTIFACTS_CALL_NAME,
        ]
        # The prior (broken) guidelines were carried forward, not blanked.
        assert outcome.guidelines_draft_path.read_text().strip() == broken_md.strip()

    def test_remaining_problems_surfaced_after_repair(self, factory_dir, form_path):
        llm = FakeFactoryLLM(
            {
                DRAFT_PERSONAS_CALL_NAME: [personas_response()],
                DRAFT_ARTIFACTS_CALL_NAME: [broken_artifacts_response()],
                DRAFT_TEXT_ARTIFACTS_CALL_NAME: [text_artifacts_response()],
                DRAFT_LOCALIZATION_CALL_NAME: [localization_response()],
                REPAIR_ARTIFACTS_CALL_NAME: [
                    broken_artifacts_response()
                ],  # still broken
            }
        )
        outcome = run_draft("tl", form_path, llm)
        assert outcome.repair_attempted
        assert not outcome.report.ok
        assert any(
            "PERSONA_GUIDELINES" in problem for problem in outcome.report.problems
        )
        # Exactly one repair round — never a loop.
        llm.assert_called(REPAIR_ARTIFACTS_CALL_NAME, times=1)
        assert (
            FactoryProject.load("tl").stages[FactoryStage.DRAFT] == StageStatus.FAILED
        )

    def test_multi_round_repair_converges(self, factory_dir, form_path):
        # Broken draft + two broken Call-B repairs + a clean third; budget 3.
        llm = FakeFactoryLLM(
            {
                DRAFT_PERSONAS_CALL_NAME: [personas_response()],
                DRAFT_ARTIFACTS_CALL_NAME: [broken_artifacts_response()],
                DRAFT_TEXT_ARTIFACTS_CALL_NAME: [text_artifacts_response()],
                DRAFT_LOCALIZATION_CALL_NAME: [localization_response()],
                REPAIR_ARTIFACTS_CALL_NAME: [
                    broken_artifacts_response(),
                    broken_artifacts_response(),
                    artifacts_response(),  # clean on the third round
                ],
            }
        )
        outcome = run_draft("tl", form_path, llm, max_repair_rounds=3)
        assert outcome.report.ok, "\n".join(outcome.report.problems)
        assert outcome.repair_rounds == 3
        llm.assert_called(REPAIR_ARTIFACTS_CALL_NAME, times=3)

    def test_multi_round_repair_respects_budget(self, factory_dir, form_path):
        # Same scripted responses, but a budget of 2 stops before the clean
        # third repair — the draft is surfaced as still-failing.
        llm = FakeFactoryLLM(
            {
                DRAFT_PERSONAS_CALL_NAME: [personas_response()],
                DRAFT_ARTIFACTS_CALL_NAME: [broken_artifacts_response()],
                DRAFT_TEXT_ARTIFACTS_CALL_NAME: [text_artifacts_response()],
                DRAFT_LOCALIZATION_CALL_NAME: [localization_response()],
                REPAIR_ARTIFACTS_CALL_NAME: [
                    broken_artifacts_response(),
                    broken_artifacts_response(),
                    artifacts_response(),
                ],
            }
        )
        outcome = run_draft("tl", form_path, llm, max_repair_rounds=2)
        assert not outcome.report.ok
        assert outcome.repair_rounds == 2
        llm.assert_called(REPAIR_ARTIFACTS_CALL_NAME, times=2)

    def test_emitted_voice_ids_stripped_and_reported(self, factory_dir, form_path):
        # The toy pack carries voice ids; the drafter must strip them.
        llm = FakeFactoryLLM(draft_scripts())
        outcome = run_draft("tl", form_path, llm)
        assert outcome.stripped_voice_ids == {
            "tessa_tl_v1": "fake_voice_id_tessa",
            "orin_tl_v1": "fake_voice_id_orin",
        }
        written = yaml.safe_load(outcome.pack_draft_path.read_text())
        assert all(
            "voice_id" not in persona for persona in written["personas"].values()
        )

    def test_unparseable_artifacts_response_is_readable(self, factory_dir, form_path):
        scripts = draft_scripts()
        scripts[DRAFT_ARTIFACTS_CALL_NAME] = ["no fenced blocks here"]
        llm = FakeFactoryLLM(scripts)
        with pytest.raises(FactoryDraftError, match="```markdown"):
            run_draft("tl", form_path, llm)

    def test_assembled_pack_carries_form_backchannel_level(
        self, factory_dir, form_path
    ):
        """The density knob comes from the author form, not a drafted prompt:
        the assembled pack declares the form's backchannel_level and never a
        retired per-pack backchannel_decision_prompt."""
        llm = FakeFactoryLLM(draft_scripts())
        outcome = run_draft("tl", form_path, llm)
        written = yaml.safe_load(outcome.pack_draft_path.read_text())
        # TOY_FORM omits backchannel_level, so the AuthorForm default applies.
        assert written["backchannel_level"] == TOY_FORM_OBJECT.backchannel_level.value
        assert "backchannel_decision_prompt" not in written


class TestDraftPromptContents:
    """The two creative prompts each embed their portion of the contracts."""

    @pytest.fixture
    def prompts(self, factory_dir, form_path) -> tuple[str, str]:
        llm = FakeFactoryLLM(draft_scripts())
        run_draft("tl", form_path, llm)
        return (
            llm.prompts_for(DRAFT_PERSONAS_CALL_NAME)[0],
            llm.prompts_for(DRAFT_ARTIFACTS_CALL_NAME)[0],
        )

    @pytest.fixture
    def text_artifacts_prompt(self, factory_dir, form_path) -> str:
        llm = FakeFactoryLLM(draft_scripts())
        run_draft("tl", form_path, llm)
        return llm.prompts_for(DRAFT_TEXT_ARTIFACTS_CALL_NAME)[0]

    def test_call_b_embeds_the_es_exemplar_body(self, prompts):
        """The exemplar guidelines are pack-relative and live under the pack's
        guidelines/ subdir; resolving them by BASENAME silently substituted a
        '(not found)' placeholder and degraded this fixed prompt — and Call B's
        body grounds Call D, the live localization block."""
        _, artifacts_prompt = prompts
        assert "Pautas de Simulación de Llamadas de Voz (Español)" in artifacts_prompt
        assert "not found" not in artifacts_prompt

    def test_call_c_embeds_the_es_text_exemplar_body(self, text_artifacts_prompt):
        assert (
            "Pautas de Simulación de Chat de Texto (Español)" in text_artifacts_prompt
        )
        assert "not found" not in text_artifacts_prompt

    def test_personas_prompt_field_docs_rendered_from_schema(self, prompts):
        # Field(description=...) strings, walked from model_fields — these
        # exist nowhere else, so their presence proves the rendering.
        personas_prompt, _ = prompts
        assert "Globally unique persona id" in personas_prompt

    def test_exemplar_pack_verbatim_in_personas_prompt(self, prompts):
        personas_prompt, _ = prompts
        assert "rishika_hindi_v1" in personas_prompt  # hi pack
        assert "imran_hindi_v1" in personas_prompt

    def test_tag_vocabulary_in_personas_prompt(self, prompts):
        personas_prompt, _ = prompts
        assert "code_switch" in personas_prompt
        assert "english_tolerance" in personas_prompt
        assert str(REQUIRED_TAG_DIMENSIONS) in personas_prompt

    def test_tag_vocabulary_instructs_omitting_optional_dimensions(self, prompts):
        """Optional dimensions (e.g. register) must be OMITTED when nothing
        fits rather than force-fit to the nearest value (PR #305), while the
        required dimensions stay mandatory and inventing values stays
        forbidden."""
        personas_prompt, _ = prompts
        assert "OPTIONAL" in personas_prompt
        assert "OMIT" in personas_prompt
        assert "MUST carry ALL" in personas_prompt
        assert str(REQUIRED_TAG_DIMENSIONS) in personas_prompt
        assert "Never invent" in personas_prompt
        assert "gig_worker" in personas_prompt

    def test_personas_prompt_carries_form_and_voice_bottleneck(self, prompts):
        personas_prompt, _ = prompts
        assert "Fast casual 20s speaker" in personas_prompt  # the form itself
        assert "DO NOT emit it" in personas_prompt  # voice_id bottleneck

    def test_artifacts_prompt_carries_guidelines_contract(self, prompts):
        _, artifacts_prompt = prompts
        assert "<PERSONA_GUIDELINES>" in artifacts_prompt
        assert "###STOP###" in artifacts_prompt

    def test_personas_prompt_constrains_acoustic_preset_ids(self, prompts):
        """Move #1: the model only WIRES personas to the deterministic
        scaffolded preset ids — it does not free-author file paths."""
        personas_prompt, _ = prompts
        for preset_id in TOY_PRESET_IDS:
            assert preset_id in personas_prompt
        # The model is told to pick from the scaffolded ids, not invent paths.
        assert "invent ids or file paths" in personas_prompt

    def test_artifacts_prompt_sees_persona_backchannel_phrases(self, prompts):
        """Move #2 coherence seam: Call B sees Call A's personas (incl. their
        backchannel phrases) so the guidelines stay consistent with them."""
        _, artifacts_prompt = prompts
        # Tessa's backchannel phrase from the toy pack.
        assert "ja ja" in artifacts_prompt


class TestFinalize:
    def test_refuses_on_dirty_guardrails(self, factory_dir):
        workspace = factory_dir / "tl"
        workspace.mkdir(parents=True)
        (workspace / PACK_DRAFT_FILENAME).write_text("language: tl\n")  # invalid
        with pytest.raises(FactoryDraftError, match="guardrail"):
            finalize("tl")

    def test_refuses_on_missing_draft(self, factory_dir):
        with pytest.raises(FactoryDraftError, match="does not exist"):
            finalize("tl")

    @pytest.fixture
    def finalized_env(self, isolated_pack_env, monkeypatch, form_path):
        """A clean draft in an isolated env, with finalize pointed at it.

        ``isolated_pack_env`` patches ``DATA_DIR``, so ``factory_root_dir()``
        and every multilingual path helper already resolve under the
        temporary workspace.
        """
        llm = FakeFactoryLLM(draft_scripts())
        outcome = run_draft("tl", form_path, llm)
        assert outcome.report.ok, "\n".join(outcome.report.problems)
        return isolated_pack_env

    def test_refuses_existing_pack_without_force(self, finalized_env):
        # isolated_pack_env pre-installs the toy tl pack, so pack.yaml exists.
        with pytest.raises(FactoryDraftError, match="--force"):
            finalize("tl")

    def test_round_trip_draft_finalize_load(self, finalized_env):
        outcome = finalize("tl", force=True)
        assert outcome.report.ok, "\n".join(outcome.report.problems)
        assert outcome.pack_path == finalized_env.multilingual_dir / "tl" / "pack.yaml"
        # Guidelines promoted under the DECLARED path (guidelines/ subdir), not
        # the draft name.
        assert outcome.guidelines_path.name == TOY_GUIDELINES_FILENAME
        assert outcome.guidelines_path.parent.name == "guidelines"
        assert outcome.guidelines_path.exists()
        final = yaml.safe_load(outcome.pack_path.read_text())
        assert final["guidelines_voice_path"] == f"guidelines/{TOY_GUIDELINES_FILENAME}"
        assert any("voice_ids" in line for line in outcome.checklist_lines)

        # The finalized pack loads through the REAL loader/registry.
        ml_loader._reset_for_tests()
        ml_registry._reset_for_tests()
        pack = ml_registry.get_language_pack("tl")
        assert pack is not None
        assert pack.display_name == "Toylang"
        assert sorted(pack.personas) == ["orin_tl_v1", "tessa_tl_v1"]
        assert Path(pack.guidelines_voice_path).exists()
        # The experiment block survived the round trip too.
        assert (
            ml_loader.get_language_experiment("tl", "telecom").preset_name
            == "multilingual_v1_toylang_telecom"
        )

    def test_refinalize_keeps_generate_assets_voice_pins(self, finalized_env):
        """H2: a --force re-finalize must not destroy the voice_ids that
        generate-assets pinned into the FINAL pack (the draft never has them)."""
        from tau2.multilingual.factory.pack_text_edit import set_persona_voice_id

        finalize("tl", force=True)
        pack_path = finalized_env.multilingual_dir / "tl" / "pack.yaml"
        # Simulate generate-assets: pin designed voices into the FINAL pack and
        # flip the checklist.
        text = pack_path.read_text()
        text = set_persona_voice_id(text, "tessa_tl_v1", "pinned_voice_tessa")
        text = set_persona_voice_id(text, "orin_tl_v1", "pinned_voice_orin")
        pack_path.write_text(text)
        project = FactoryProject.load("tl")
        project.checklist.voice_ids = True
        project.save()

        finalize("tl", force=True)

        final = yaml.safe_load(pack_path.read_text())
        assert final["personas"]["tessa_tl_v1"]["voice_id"] == "pinned_voice_tessa"
        assert final["personas"]["orin_tl_v1"]["voice_id"] == "pinned_voice_orin"
        assert FactoryProject.load("tl").checklist.voice_ids is True

    def test_refinalize_with_renamed_persona_resets_voice_checklist(
        self, finalized_env
    ):
        """H2: a pin whose persona no longer exists in the new draft is dropped
        LOUDLY — checklist.voice_ids resets so the human bottleneck re-engages."""
        pack_path = finalized_env.multilingual_dir / "tl" / "pack.yaml"
        finalize("tl", force=True)

        # Pin a voice for a persona the draft does not have (as after a rename).
        data = yaml.safe_load(pack_path.read_text())
        data["personas"]["ghost_tl_v1"] = {
            "persona_id": "ghost_tl_v1",
            "voice_id": "pinned_voice_ghost",
        }
        pack_path.write_text(yaml.safe_dump(data, sort_keys=False))
        project = FactoryProject.load("tl")
        project.checklist.voice_ids = True
        project.save()

        finalize("tl", force=True)

        final = yaml.safe_load(pack_path.read_text())
        assert "ghost_tl_v1" not in final["personas"]
        assert FactoryProject.load("tl").checklist.voice_ids is False

    def test_finalized_pack_retires_nativeness_author_this_scaffold(
        self, finalized_env
    ):
        """Nativeness is now a reviewed factory stage, not a hand-fill comment."""
        outcome = finalize("tl", force=True)
        text = outcome.pack_path.read_text()
        assert "nativeness scoring (AUTHOR THIS" not in text
        assert "# nativeness:" not in text
        assert "nativeness" not in yaml.safe_load(text)

    def test_finalized_pack_keeps_delivery_scaffold_comment(self, finalized_env):
        """L5: same as L4 for the delivery scaffold — a finalized pack without a
        real delivery section carries the commented catalog menu so the author
        gets delivery-metric setup at the same moment as nativeness."""
        from tau2.multilingual.delivery_catalog import DELIVERY_FACTOR_CATALOG

        outcome = finalize("tl", force=True)
        text = outcome.pack_path.read_text()
        assert "delivery scoring (audio-layer; AUTHOR THIS" in text
        assert "# delivery:" in text
        # The scaffold lists the full closed catalog menu.
        for factor_id in DELIVERY_FACTOR_CATALOG:
            assert factor_id in text
        # It is a comment, not data: the loaded mapping has no delivery key.
        assert "delivery" not in yaml.safe_load(text)


class TestEmptyGuidelinesPreserved:
    """A Call-B (artifacts) repair returning an empty markdown fence must not
    overwrite the prior good guidelines (two-call architecture)."""

    def _personas_json(self, lang: str = "sw") -> dict:
        return {
            "personas": [
                {
                    "persona_id": f"amara_{lang}_v1",
                    "display_name": "Amara",
                    "short_description": "Salesperson.",
                    "language": lang,
                    "script": "latn",
                    "pragmatics_clauses": ["Speak Swahili."],
                    "backchannel_phrases": ["ndiyo", "sawa"],
                    "acoustic_preset_id": f"{lang}_office",
                    "tags": {
                        "code_switch": "low",
                        "formality": "high",
                        "english_tolerance": "low",
                        "age_band": "50s",
                        "gender": "female",
                    },
                },
                {
                    "persona_id": f"baraka_{lang}_v1",
                    "display_name": "Baraka",
                    "short_description": "Student with urban accent.",
                    "language": lang,
                    "script": "latn",
                    "pragmatics_clauses": ["Speak fast urban Swahili."],
                    "backchannel_phrases": ["eeh", "poa"],
                    "acoustic_preset_id": f"{lang}_outdoor_traffic",
                    "tags": {
                        "code_switch": "high",
                        "formality": "low",
                        "english_tolerance": "high",
                        "age_band": "20s",
                        "gender": "male",
                    },
                },
            ],
            "translation_guidance": "Swahili guidance.",
            "agent_greeting": "Habari!",
        }

    def _artifacts_response(self, guidelines: str) -> str:
        return f"```markdown\n{guidelines}```"

    def test_empty_fence_preserves_prior_guidelines(
        self, factory_dir, tmp_path, monkeypatch
    ):
        import tau2.multilingual.factory.drafting as drafting_mod
        from tau2.multilingual.factory.guardrails import GuardrailReport

        GOOD_GUIDELINES = (
            "# Swahili Guidelines\n\nThis has <PERSONA_GUIDELINES> slot.\n###STOP###\n"
        )
        EMPTY_FENCE = self._artifacts_response("   \n")

        # Guardrails first fail (triggering an artifact repair) — the failure is
        # localized to the VOICE guidelines (named by file, as real guardrail
        # messages are) so only Call B re-runs — then pass.
        failing_report = GuardrailReport(
            ok=False,
            problems=[
                "guidelines file guidelines_draft.md is missing the literal "
                "<PERSONA_GUIDELINES> slot"
            ],
        )
        passing_report = GuardrailReport(ok=True, problems=[])
        call_count = {"n": 0}

        def fake_validate(workspace):
            call_count["n"] += 1
            return failing_report if call_count["n"] == 1 else passing_report

        monkeypatch.setattr(drafting_mod, "validate_pack_draft", fake_validate)

        llm = FakeFactoryLLM(
            scripts={
                DRAFT_PERSONAS_CALL_NAME: [self._personas_json()],
                DRAFT_ARTIFACTS_CALL_NAME: [self._artifacts_response(GOOD_GUIDELINES)],
                DRAFT_TEXT_ARTIFACTS_CALL_NAME: [
                    self._artifacts_response(
                        "# Swahili Text Guidelines\n\nTyped chat, "
                        "<PERSONA_GUIDELINES> slot.\n###STOP###\n"
                    )
                ],
                DRAFT_LOCALIZATION_CALL_NAME: [localization_response()],
                # The repair returns an empty markdown fence.
                REPAIR_ARTIFACTS_CALL_NAME: [EMPTY_FENCE],
            }
        )

        form_data = {
            "language": "sw",
            "display_name": "Swahili",
            "script": "latn",
            "personas": [
                {"name": "Amara", "sketch": "A salesperson who speaks Swahili."},
                {"name": "Baraka", "sketch": "A student with urban accent."},
            ],
        }
        form_path = tmp_path / "form.yaml"
        form_path.write_text(yaml.safe_dump(form_data))

        outcome = run_draft("sw", form_path, llm, model="test-model")

        # The guidelines file must NOT be empty — the prior guidelines survive.
        guidelines_text = outcome.guidelines_draft_path.read_text()
        assert GOOD_GUIDELINES.strip() in guidelines_text, (
            f"repair with empty markdown fence overwrote prior guidelines. "
            f"Got: {guidelines_text!r}"
        )


class TestCallDDomainThreading:
    def test_call_d_receives_the_resolved_domain(
        self, factory_dir, form_path, monkeypatch
    ):
        """`--domain` must reach Call D: without it the glossary is drafted
        against the DEFAULT domain's catalog and the draft dead-ends on the
        missing-glossary guardrail (after a wasted LLM call)."""
        import tau2.multilingual.factory.drafting as drafting_mod

        seen: dict = {}
        real = drafting_mod.draft_localization

        def spy(language, display_name, script, **kwargs):
            seen["domain"] = kwargs.get("domain")
            # Redirect back to the default domain so the toy Call-D script
            # (which covers the DEFAULT domain's catalog) still validates —
            # this test is about what Call D RECEIVES, not airline fixture
            # data.
            kwargs["domain"] = DEFAULT_MULTILINGUAL_DOMAIN
            return real(language, display_name, script, **kwargs)

        monkeypatch.setattr(drafting_mod, "draft_localization", spy)
        llm = FakeFactoryLLM(draft_scripts())
        # max_repair_rounds=0: the redirected draft legitimately fails the
        # airline-glossary guardrail afterwards; this test pins the threading,
        # not the (fixture-less) airline repair flow.
        run_draft(
            "tl",
            form_path,
            llm,
            domain="airline",
            smoke_task_stem="3",
            max_repair_rounds=0,
        )
        assert seen["domain"] == "airline"

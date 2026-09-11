# Copyright Sierra
"""Tests for `tau2 factory draft-continuers`.

All LLM interaction is scripted through ``FakeFactoryLLM``. The verb's
contract: ONE call for the whole pack, shown every continuer-candidate
surface it might SELECT from, replying with a per-persona inventory that
declares its provenance and lands inside the backchannel bounds in the pack's
script — then a SURGICAL replacement of only the ``backchannel_phrases``
blocks (plus one provenance marker) in the shipped pack.yaml.
"""

import pytest
import yaml

from tau2.multilingual.factory import backfill_lib
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.continuer_draft import (
    CONTINUER_DRAFT_PROMPT_VERSION,
    DRAFT_CONTINUERS_CALL_NAME,
    _pack_palettes,
    build_draft_continuers_prompt,
    continuer_draft_prompt_sha256,
    draft_pack_continuers,
    run_draft_continuers,
)
from tau2.multilingual.factory.draft_prompts import (
    BACKCHANNEL_CONTINUER_PURITY_RULE,
)
from test_multilingual.factory_testing.fake_llm import FakeFactoryLLM

# Two personas carrying the pre-fix contaminated inventory (acknowledgments,
# not continuers); Devanagari so the script guard is meaningful; an inline
# comment and a trailing key that must survive byte-identical.
PACK_YAML = """\
language: tl
display_name: Toylang
backchannel_level: low  # an inline comment that must survive byte-identical
personas:
  tessa_tl_v1:
    persona_id: tessa_tl_v1
    display_name: Tessa
    short_description: High-register speaker
    language: tl
    script: deva
    pragmatics_clauses:
    - You address the agent respectfully ('आप') at all times.
    backchannel_phrases:
    - ठीक है
    - समझ गया
    - बिल्कुल
    non_directed_phrases:
    - एक मिनट रुको
    tags:
      gender: female
  tom_tl_v1:
    persona_id: tom_tl_v1
    display_name: Tom
    short_description: Casual speaker
    language: tl
    script: deva
    pragmatics_clauses:
    - You open casually with 'हेलो'.
    backchannel_phrases:
    - अच्छा
    - हाँ हाँ
    non_directed_phrases:
    - बस आ रहा हूँ
    tags:
      gender: male
agent_greeting: नमस्ते!
"""

GOOD_REPLY = {
    "personas": {
        "tessa_tl_v1": {
            "phrases": ["हम्म"],
            "provenance": "authored",
            "note": "Nothing in the material is a pure continuer.",
        },
        "tom_tl_v1": {
            "phrases": ["हम्म"],
            "provenance": "authored",
            "note": "Same canonical hum.",
        },
    }
}


@pytest.fixture
def shipped_pack(tmp_path, monkeypatch):
    pack_dir = tmp_path / "tl"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.yaml").write_text(PACK_YAML)
    monkeypatch.setattr(backfill_lib, "multilingual_data_dir", lambda: tmp_path)
    return pack_dir / "pack.yaml"


def scripted_llm(reply=None) -> FakeFactoryLLM:
    return FakeFactoryLLM({DRAFT_CONTINUERS_CALL_NAME: [reply or GOOD_REPLY]})


def toy_personas() -> dict:
    return yaml.safe_load(PACK_YAML)["personas"]


def draft(llm) -> dict:
    return draft_pack_continuers(
        language="tl",
        display_name="Toylang",
        personas=toy_personas(),
        palettes={"conversational_confirmations": ["ठीक है", "बिल्कुल"]},
        llm=llm,
    )


class TestDraftPackContinuers:
    def test_valid_reply_accepted_with_provenance(self):
        llm = scripted_llm()
        drafted = draft(llm)
        assert drafted["tessa_tl_v1"].phrases == ["हम्म"]
        assert drafted["tom_tl_v1"].provenance == "authored"
        # ONE call for the whole pack — the select-first rule is pack-wide.
        llm.assert_called(DRAFT_CONTINUERS_CALL_NAME, times=1)

    def test_missing_persona_fails_loud(self):
        reply = {"personas": {"tessa_tl_v1": GOOD_REPLY["personas"]["tessa_tl_v1"]}}
        with pytest.raises(FactoryDraftError, match="missing persona"):
            draft(scripted_llm(reply))

    def test_unknown_persona_fails_loud(self):
        reply = {
            "personas": dict(
                GOOD_REPLY["personas"],
                ghost_tl_v1={
                    "phrases": ["हम्म"],
                    "provenance": "authored",
                    "note": "",
                },
            )
        }
        with pytest.raises(FactoryDraftError, match="unknown persona"):
            draft(scripted_llm(reply))

    def test_too_many_phrases_rejected(self):
        """The count bound: the random draw must not reach for an inventory."""
        reply = {
            "personas": {
                "tessa_tl_v1": {
                    "phrases": ["हम्म", "अच्छा", "जी"],
                    "provenance": "selected",
                    "note": "",
                },
                "tom_tl_v1": GOOD_REPLY["personas"]["tom_tl_v1"],
            }
        }
        with pytest.raises(FactoryDraftError, match="at most 2 are allowed"):
            draft(scripted_llm(reply))

    def test_turn_level_response_length_rejected(self):
        """The length bound: a hum, not an utterance."""
        reply = {
            "personas": {
                "tessa_tl_v1": {
                    "phrases": ["जी हाँ बिल्कुल ठीक"],
                    "provenance": "authored",
                    "note": "",
                },
                "tom_tl_v1": GOOD_REPLY["personas"]["tom_tl_v1"],
            }
        }
        with pytest.raises(FactoryDraftError, match="the budget is 2 words"):
            draft(scripted_llm(reply))

    def test_answering_in_english_is_rejected(self):
        reply = {
            "personas": {
                "tessa_tl_v1": {
                    "phrases": ["mm-hmm"],
                    "provenance": "authored",
                    "note": "",
                },
                "tom_tl_v1": GOOD_REPLY["personas"]["tom_tl_v1"],
            }
        }
        with pytest.raises(FactoryDraftError, match="not in the pack's script"):
            draft(scripted_llm(reply))

    def test_empty_inventory_rejected(self):
        reply = {
            "personas": {
                "tessa_tl_v1": {"phrases": [], "provenance": "selected", "note": ""},
                "tom_tl_v1": GOOD_REPLY["personas"]["tom_tl_v1"],
            }
        }
        with pytest.raises(FactoryDraftError, match="returned no continuer"):
            draft(scripted_llm(reply))

    def test_undeclared_provenance_rejected(self):
        """Provenance is the whole audit trail: selected-vs-authored decides
        whether a native reviewer is looking at pack material or new content."""
        reply = {
            "personas": {
                "tessa_tl_v1": {"phrases": ["हम्म"], "provenance": "invented"},
                "tom_tl_v1": GOOD_REPLY["personas"]["tom_tl_v1"],
            }
        }
        with pytest.raises(FactoryDraftError, match="invalid continuer inventory"):
            draft(scripted_llm(reply))

    def test_mixed_scripts_fail_before_the_llm_call(self):
        personas = toy_personas()
        personas["tom_tl_v1"]["script"] = "latn"
        llm = scripted_llm()
        with pytest.raises(FactoryDraftError, match="needs exactly one"):
            draft_pack_continuers(
                language="tl",
                display_name="Toylang",
                personas=personas,
                palettes={},
                llm=llm,
            )
        llm.assert_called(DRAFT_CONTINUERS_CALL_NAME, times=0)


class TestPrompt:
    def test_prompt_shows_every_selectable_surface(self):
        prompt = build_draft_continuers_prompt(
            language="tl",
            display_name="Toylang",
            script="deva",
            personas=toy_personas(),
            palettes={"conversational_confirmations": ["ठीक है", "बिल्कुल"]},
        )
        # The purity rule is shared verbatim with the persona-drafting call.
        assert BACKCHANNEL_CONTINUER_PURITY_RULE in prompt
        # Select-first is only honourable if every candidate surface is shown.
        assert "समझ गया" in prompt  # tessa's current list
        assert "हाँ हाँ" in prompt  # tom's current list
        assert "conversational_confirmations" in prompt
        assert "at most 2 entries" in prompt
        assert "at most 2 words" in prompt
        assert "'deva'" in prompt

    def test_prompt_reports_absent_palettes_rather_than_omitting_them(self):
        prompt = build_draft_continuers_prompt(
            language="tl",
            display_name="Toylang",
            script="deva",
            personas=toy_personas(),
            palettes={},
        )
        assert "(none)" in prompt

    def test_prompt_provenance_is_stable_and_versioned(self):
        assert CONTINUER_DRAFT_PROMPT_VERSION == "v1"
        assert len(continuer_draft_prompt_sha256()) == 64
        assert continuer_draft_prompt_sha256() == continuer_draft_prompt_sha256()

    def test_palettes_read_only_the_continuer_candidate_kinds(self):
        data = {
            "localization": {
                "guideline_examples": [
                    {"kind": "disfluency_fillers", "utterances": ["ehm", "boh"]},
                    {"kind": "restart_example", "utterances": ["Volevo—aspetti..."]},
                    {"kind": "conversational_confirmations", "utterances": ["ok"]},
                ]
            }
        }
        assert _pack_palettes(data) == {
            "disfluency_fillers": ["ehm", "boh"],
            "conversational_confirmations": ["ok"],
        }

    def test_palettes_tolerate_a_pack_without_localization(self):
        assert _pack_palettes({}) == {}


class TestRunDraftContinuers:
    def test_surgical_write_touches_only_backchannel_blocks(self, shipped_pack):
        outcome = run_draft_continuers("tl", llm=scripted_llm())
        assert outcome.written
        assert outcome.personas["tessa_tl_v1"].phrases == ["हम्म"]
        new_text = shipped_pack.read_text()

        data = yaml.safe_load(new_text)
        assert data["personas"]["tessa_tl_v1"]["backchannel_phrases"] == ["हम्म"]
        assert data["personas"]["tom_tl_v1"]["backchannel_phrases"] == ["हम्म"]

        # Everything OUTSIDE the backchannel blocks and the marker is identical.
        assert (
            "backchannel_level: low  # an inline comment that must survive "
            "byte-identical" in new_text
        )
        assert "agent_greeting: नमस्ते!" in new_text
        assert "    tags:\n      gender: female" in new_text
        # The sibling short-phrase inventory is NOT touched.
        assert "    non_directed_phrases:\n    - एक मिनट रुको" in new_text
        # Provenance marker sits above the personas block and names the source.
        assert new_text.index("# --- persona backchannel continuers") < new_text.index(
            "personas:"
        )
        assert "tessa_tl_v1: authored" in new_text
        assert CONTINUER_DRAFT_PROMPT_VERSION in new_text
        assert continuer_draft_prompt_sha256()[:12] in new_text

    def test_skip_when_already_drafted_force_redoes(self, shipped_pack):
        run_draft_continuers("tl", llm=scripted_llm())
        skipped = run_draft_continuers("tl", llm=scripted_llm())
        assert not skipped.written

        redone = run_draft_continuers("tl", force=True, llm=scripted_llm())
        assert redone.written
        # The old marker was stripped: exactly one marker block remains.
        assert (
            shipped_pack.read_text().count("# --- persona backchannel continuers") == 1
        )

    def test_missing_pack_fails_loud(self, tmp_path, monkeypatch):
        monkeypatch.setattr(backfill_lib, "multilingual_data_dir", lambda: tmp_path)
        with pytest.raises(FactoryDraftError, match="no shipped pack"):
            run_draft_continuers("zz", llm=scripted_llm())

    def test_unlocatable_block_refuses_a_partial_rewrite(self, shipped_pack):
        text = shipped_pack.read_text().replace(
            "    backchannel_phrases:\n    - अच्छा\n    - हाँ हाँ\n",
            '    backchannel_phrases: ["अच्छा", "हाँ हाँ"]\n',
        )
        shipped_pack.write_text(text)
        with pytest.raises(FactoryDraftError, match="could not locate"):
            run_draft_continuers("tl", llm=scripted_llm())

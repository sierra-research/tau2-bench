# Copyright Sierra
"""Tests for the localization-drafting verb (`tau2 factory draft-localization`).

All LLM interaction is scripted through ``FakeFactoryLLM``. The verb's
contract: draft the ``localization:`` block through the fixed prompt, validate
it against the closed catalogs + full-coverage requirement, and APPEND ONLY the
new top-level key to the shipped pack.yaml — every other line stays
byte-identical (concurrent edits to other pack keys must rebase trivially).
A pack whose block predates the speech-convention fields is UPGRADED in
place: only the missing fields are drafted in; reviewed values are preserved
verbatim.
"""

import pytest
import yaml

from tau2.multilingual.factory import backfill_lib
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.localization_drafting import (
    DRAFT_LOCALIZATION_CALL_NAME,
    LOCALIZATION_PROMPT_VERSION,
    build_localization_prompt,
    draft_localization,
    localization_prompt_sha256,
    run_backfill_localization,
)
from tau2.multilingual.factory.pack_text_edit import set_nested_block
from tau2.multilingual.localization_catalog import (
    SYMBOL_CATALOG,
    get_domain_term_catalog,
)
from tau2.multilingual.schema import LanguagePack
from test_multilingual.factory_testing.draft_scripts import localization_response
from test_multilingual.factory_testing.fake_llm import FakeFactoryLLM

MINIMAL_PACK_YAML = """\
language: tl
display_name: Toylang
backchannel_level: low  # an inline comment that must survive byte-identical
translation_guidance: 'Toy guidance: decimal comma; Funkalphabet.'
guidelines_voice_path: guidelines_voice_tl.md
personas:
  tessa_tl_v1:
    persona_id: tessa_tl_v1
    display_name: Tessa
    short_description: Fast casual 20s speaker
    language: tl
    pragmatics_clauses:
    - Speak fast casual Toylang.
"""

TOY_GUIDELINES = """\
# Toy voice guidelines

Toy phone convention: read in pairs, 'nul zes' style.

<PERSONA_GUIDELINES>
"""


@pytest.fixture
def shipped_pack(tmp_path, monkeypatch):
    """A shipped tl pack in an isolated multilingual dir."""
    pack_dir = tmp_path / "tl"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack.yaml").write_text(MINIMAL_PACK_YAML)
    (pack_dir / "guidelines_voice_tl.md").write_text(TOY_GUIDELINES)
    monkeypatch.setattr(backfill_lib, "multilingual_data_dir", lambda: tmp_path)
    return pack_dir / "pack.yaml"


def scripted_llm(responses=None) -> FakeFactoryLLM:
    return FakeFactoryLLM(
        {DRAFT_LOCALIZATION_CALL_NAME: responses or [localization_response()]}
    )


def airline_response() -> dict:
    """The scripted LLM reply for an airline drafting call: same toy block,
    with the flat domain_glossary covering the AIRLINE term catalog (the
    default response covers the default domain's — telecom's)."""
    response = localization_response()
    response["domain_glossary"] = [
        {"term_id": term_id, "native": f"toy-{d.english}"}
        for term_id, d in get_domain_term_catalog("airline").items()
    ]
    return response


class TestDraftLocalization:
    def test_valid_response_validates(self):
        llm = scripted_llm()
        config = draft_localization("tl", "Toylang", "latn", llm=llm)
        assert {r.symbol for r in config.symbol_readouts} == set(SYMBOL_CATALOG)
        # The flat drafted glossary is filed under the drafting domain.
        assert set(config.domain_glossaries) == {"telecom"}
        assert {g.term_id for g in config.domain_glossaries["telecom"]} == set(
            get_domain_term_catalog("telecom")
        )
        llm.assert_called(DRAFT_LOCALIZATION_CALL_NAME, times=1)

    def test_drafts_for_the_requested_domain(self):
        llm = scripted_llm([airline_response()])
        config = draft_localization("tl", "Toylang", "latn", domain="airline", llm=llm)
        assert set(config.domain_glossaries) == {"airline"}
        assert {g.term_id for g in config.domain_glossaries["airline"]} == set(
            get_domain_term_catalog("airline")
        )
        # The prompt grounds the model on the airline catalog, not telecom's
        # (the term-listing lines, not the domain-independent field docs —
        # those name both domains as examples).
        prompt = llm.prompts_for(DRAFT_LOCALIZATION_CALL_NAME)[0]
        assert "checked bag" in prompt
        assert "— 'data roaming'" not in prompt

    def test_schema_invalid_response_fails_loud(self):
        bad = localization_response()
        bad["symbol_readouts"][0]["symbol"] = "%"  # not in the closed catalog
        llm = scripted_llm([bad, localization_response()])
        with pytest.raises(FactoryDraftError, match="invalid valid localization"):
            draft_localization("tl", "Toylang", "latn", llm=llm)
        llm.assert_called(DRAFT_LOCALIZATION_CALL_NAME, times=1)

    def test_incomplete_coverage_fails_loud(self):
        incomplete = localization_response()
        incomplete["domain_glossary"] = incomplete["domain_glossary"][:3]
        llm = scripted_llm([incomplete])
        with pytest.raises(FactoryDraftError, match="missing domain_glossary"):
            draft_localization("tl", "Toylang", "latn", llm=llm)
        llm.assert_called(DRAFT_LOCALIZATION_CALL_NAME, times=1)

    def test_prompt_carries_catalogs_guidance_and_personas(self):
        prompt = build_localization_prompt(
            "tl",
            "Toylang",
            "latn",
            translation_guidance="Decimal comma; Funkalphabet.",
            persona_summaries=["Fast casual 20s speaker"],
            speech_guidelines="Toy phone convention: read in pairs.",
        )
        for symbol, d in SYMBOL_CATALOG.items():
            assert f"'{symbol}' — {d.english_name}" in prompt
        for term_id, t in get_domain_term_catalog("telecom").items():
            assert f"{term_id} — '{t.english}'" in prompt
        # Field docs rendered from the validating models (cannot drift).
        assert "Native verbalizations of the symbol" in prompt
        assert "Decimal comma; Funkalphabet." in prompt
        assert "Fast casual 20s speaker" in prompt
        # The grounded failure modes anchor the examples requirement.
        assert "May 십칠" in prompt and "May21" in prompt
        # v2: speech conventions + the guidelines grounding slot.
        assert "phone_number_readout" in prompt
        assert "amount_readout" in prompt
        assert "spelling_alphabet" in prompt
        assert "spoken_value_examples" in prompt
        assert "Toy phone convention: read in pairs." in prompt
        # The persona-neutrality contract is stated explicitly.
        assert "PERSONA-ATTITUDE NEUTRALITY" in prompt

    def test_prompt_carries_guideline_example_catalog_and_backchannels(self):
        from tau2.multilingual.localization_catalog import (
            GUIDELINE_EXAMPLE_CATALOG,
        )

        prompt = build_localization_prompt(
            "tl",
            "Toylang",
            "latn",
            backchannel_phrases=["toy-ja", "toy-oke"],
        )
        # v3: every catalog example kind (with its English default) is shown.
        assert "guideline_examples" in prompt
        for kind_id in GUIDELINE_EXAMPLE_CATALOG:
            assert kind_id in prompt
        assert '"um"; "uh"; "you know"; "like"; "I mean"' in prompt
        # v3: the confirmations palette is grounded on the personas'
        # backchannel phrases and must agree with them.
        assert '"toy-ja", "toy-oke"' in prompt
        assert "conversational_confirmations" in prompt
        # v3: the email/URL comma convention (flow, not per-token commas).
        assert "COMMA CONVENTION" in prompt
        assert "CONTINUOUS FLOW" in prompt
        # v5: the gendered guideline-example variant is specified, so a
        # re-draft cannot silently drop a pack's male realizations.
        assert "SPEAKER GENDER in guideline_examples" in prompt
        assert "male_utterances" in prompt

    def test_prompt_version_and_hash_are_stable_contracts(self):
        # v7: glossary `note` values are written in English (any native form
        # they cite quoted inside) and name a register, never a persona.
        assert LOCALIZATION_PROMPT_VERSION == "v7"
        assert len(localization_prompt_sha256()) == 64


class TestStripLocalizationBlock:
    def test_root_level_lines_after_block_survive_force_strip(self):
        """Only the block's own header comments and indented body are dropped;
        a root-level comment after the block (with or without a following key)
        is not part of it."""

        def _strip_localization_block(text: str) -> str:
            return backfill_lib.strip_top_level_block(
                text, key="localization", marker="# --- spoken localization"
            )

        block = (
            "# --- spoken localization (symbols / date-numeral rule) ---\n"
            "# Drafted by `tau2 factory draft-localization` (prompt v1).\n"
            "localization:\n"
            "  symbol_readouts:\n"
            "    - symbol: '@'\n"
            "      native: arroba\n"
            "\n"
        )
        followed_by_key = (
            f"language: tl\n\n{block}# root comment that must survive\n"
            "experiment: true\n"
        )
        stripped = _strip_localization_block(followed_by_key)
        assert "# root comment that must survive" in stripped
        assert "experiment: true" in stripped
        assert "localization:" not in stripped
        assert "spoken localization" not in stripped
        assert stripped.startswith("language: tl\n")

        at_eof = f"language: tl\n\n{block}# trailing note only\n"
        stripped = _strip_localization_block(at_eof)
        assert "# trailing note only" in stripped
        assert "localization:" not in stripped


class TestBackfillLocalization:
    def test_appends_only_the_new_key(self, shipped_pack):
        original = shipped_pack.read_text()
        outcome = run_backfill_localization("tl", llm=scripted_llm())
        assert outcome.written and not outcome.replaced
        new_text = shipped_pack.read_text()
        # Every original line is byte-identical (append-only write).
        assert new_text.startswith(original.rstrip("\n"))
        data = yaml.safe_load(new_text)
        assert {r["symbol"] for r in data["localization"]["symbol_readouts"]} == set(
            SYMBOL_CATALOG
        )
        # Provenance marker comment travels with the block.
        assert LOCALIZATION_PROMPT_VERSION in new_text
        assert localization_prompt_sha256()[:12] in new_text
        # The merged pack still validates as a LanguagePack.
        data.pop("experiments", None)
        data["guidelines_voice_path"] = str(
            shipped_pack.parent / data["guidelines_voice_path"]
        )
        LanguagePack.model_validate(data)

    def test_skips_when_block_present(self, shipped_pack):
        run_backfill_localization("tl", llm=scripted_llm())
        after_first = shipped_pack.read_text()
        llm = scripted_llm()
        outcome = run_backfill_localization("tl", llm=llm)
        assert not outcome.written
        assert shipped_pack.read_text() == after_first
        assert llm.calls == []  # reuse-by-default: no LLM spend

    def test_force_replaces_block_once(self, shipped_pack):
        run_backfill_localization("tl", llm=scripted_llm())
        replacement = localization_response()
        replacement["domain_glossary"] = [
            {"term_id": g["term_id"], "native": f"v2-{g['native']}"}
            for g in replacement["domain_glossary"]
        ]
        outcome = run_backfill_localization(
            "tl", force=True, llm=scripted_llm([replacement])
        )
        assert outcome.written and outcome.replaced
        text = shipped_pack.read_text()
        assert text.count("\nlocalization:") == 1
        data = yaml.safe_load(text)
        assert all(
            g["native"].startswith("v2-")
            for g in data["localization"]["domain_glossaries"]["telecom"]
        )
        # The non-localization prefix is still intact (incl. inline comment).
        assert "an inline comment that must survive byte-identical" in text

    def test_missing_pack_fails_loud(self, shipped_pack):
        with pytest.raises(FactoryDraftError, match="no shipped pack"):
            run_backfill_localization("zz", llm=scripted_llm())

    def test_prompt_grounded_on_pack_content(self, shipped_pack):
        llm = scripted_llm()
        run_backfill_localization("tl", llm=llm)
        prompt = llm.prompts_for(DRAFT_LOCALIZATION_CALL_NAME)[0]
        assert "Toy guidance: decimal comma; Funkalphabet." in prompt
        assert "Fast casual 20s speaker" in prompt
        # The pack's localized voice guidelines ground the speech conventions.
        assert "Toy phone convention: read in pairs, 'nul zes' style." in prompt


class TestBackfillUpgrade:
    """A pack whose reviewed block predates the speech-convention fields is
    upgraded in place: the missing fields are drafted in; every reviewed
    field keeps its exact existing values; the rest of pack.yaml stays
    byte-identical."""

    V1_FIELDS = (
        "symbol_readouts",
        "date_examples",
        "number_examples",
        "domain_glossaries",
    )

    @pytest.fixture
    def pack_with_v1_block(self, shipped_pack):
        """Simulate a shipped pack whose block was drafted by prompt v1
        (no speech-convention fields), with owner-reviewed edits."""
        response = localization_response()
        v1_block = {
            field: response[field]
            for field in ("symbol_readouts", "date_examples", "number_examples")
        }
        v1_block["domain_glossaries"] = {"telecom": response["domain_glossary"]}
        # An owner-reviewed tweak that MUST survive the upgrade untouched.
        v1_block["domain_glossaries"]["telecom"][0]["native"] = "reviewed-by-owner"
        shipped_pack.write_text(
            shipped_pack.read_text()
            + "\n# --- spoken localization (symbols / date-numeral rule / domain glossary) ---\n"
            + "# Drafted by `tau2 factory draft-localization` (prompt v1).\n"
            + yaml.safe_dump(
                {"localization": v1_block}, sort_keys=False, allow_unicode=True
            )
        )
        return shipped_pack

    def test_upgrade_adds_only_missing_fields(self, pack_with_v1_block):
        llm = scripted_llm()
        outcome = run_backfill_localization("tl", llm=llm)
        assert outcome.written and outcome.upgraded and not outcome.replaced
        assert set(outcome.upgraded_fields) == {
            "phone_number_readout",
            "amount_readout",
            "spelling_alphabet",
            "spoken_value_examples",
            "guideline_examples",
        }
        data = yaml.safe_load(pack_with_v1_block.read_text())
        block = data["localization"]
        # The drafted speech conventions landed...
        assert block["phone_number_readout"]["examples"]
        assert block["spelling_alphabet"]["avoid"]
        assert len(block["spoken_value_examples"]) == 6
        # ...as did the v3 guideline example palettes (full catalog).
        from tau2.multilingual.localization_catalog import (
            GUIDELINE_EXAMPLE_CATALOG,
        )

        assert {e["kind"] for e in block["guideline_examples"]} == set(
            GUIDELINE_EXAMPLE_CATALOG
        )
        # ...and every reviewed field kept its exact values (the drafted
        # replacements for them were discarded).
        expected = localization_response()
        expected["domain_glossaries"] = {"telecom": expected.pop("domain_glossary")}
        expected["domain_glossaries"]["telecom"][0]["native"] = "reviewed-by-owner"
        for field in self.V1_FIELDS:
            assert block[field] == expected[field], f"{field} was re-rolled"
        # The non-localization prefix stays byte-identical.
        text = pack_with_v1_block.read_text()
        assert text.startswith(MINIMAL_PACK_YAML.rstrip("\n"))
        assert "an inline comment that must survive byte-identical" in text
        assert text.count("\nlocalization:") == 1
        # The upgrade provenance marker replaces the v1 marker.
        assert "all previously-reviewed fields preserved verbatim" in text

    def test_second_run_skips_after_upgrade(self, pack_with_v1_block):
        run_backfill_localization("tl", llm=scripted_llm())
        after_upgrade = pack_with_v1_block.read_text()
        llm = scripted_llm()
        outcome = run_backfill_localization("tl", llm=llm)
        assert not outcome.written
        assert pack_with_v1_block.read_text() == after_upgrade
        assert llm.calls == []  # reuse-by-default: no LLM spend

    def test_new_domain_glossary_upgrades_in_place(self, shipped_pack):
        """A pack with a complete telecom block gains an airline glossary via
        the SAME upgrade path: only domain_glossaries.airline is added; the
        reviewed telecom glossary (and every other field) is untouched."""
        run_backfill_localization("tl", llm=scripted_llm())
        before = yaml.safe_load(shipped_pack.read_text())["localization"]
        outcome = run_backfill_localization(
            "tl", domain="airline", llm=scripted_llm([airline_response()])
        )
        assert outcome.written and outcome.upgraded and not outcome.replaced
        assert outcome.upgraded_fields == ["domain_glossaries[airline]"]
        block = yaml.safe_load(shipped_pack.read_text())["localization"]
        assert set(block["domain_glossaries"]) == {"airline", "telecom"}
        assert (
            block["domain_glossaries"]["telecom"]
            == (before["domain_glossaries"]["telecom"])
        )
        assert {g["term_id"] for g in block["domain_glossaries"]["airline"]} == set(
            get_domain_term_catalog("airline")
        )
        # Every non-glossary field kept its exact reviewed values.
        for field, value in before.items():
            if field != "domain_glossaries":
                assert block[field] == value, f"{field} was re-rolled"
        # A third run (either domain) is a no-op.
        llm = scripted_llm()
        assert not run_backfill_localization("tl", llm=llm).written
        assert not run_backfill_localization("tl", domain="airline", llm=llm).written
        assert llm.calls == []

    def test_invalid_existing_block_fails_loud(self, shipped_pack):
        shipped_pack.write_text(
            shipped_pack.read_text()
            + "\nlocalization:\n  symbol_readouts:\n  - symbol: '%'\n    spoken: [pct]\n"
        )
        with pytest.raises(FactoryDraftError, match="invalid localization block"):
            run_backfill_localization("tl", llm=scripted_llm())


class TestForceIsDomainScoped:
    """``--force`` re-rolls the REQUESTED domain's glossary and nothing else.

    The verb is ``--domain``-scoped, so its override is too. Everything else
    in the block — other domains' glossaries AND every non-glossary field
    (symbol readouts, date/number examples, speech conventions, guideline
    palettes) — is reviewed native content that a glossary re-draft has no
    business re-rolling. Before this was scoped, `--force` silently replaced
    all of it with fresh LLM output.
    """

    @staticmethod
    def _two_domain_pack(pack):
        """A complete block with reviewed telecom + airline glossaries."""
        run_backfill_localization("tl", llm=scripted_llm())
        run_backfill_localization(
            "tl", domain="airline", llm=scripted_llm([airline_response()])
        )
        return yaml.safe_load(pack.read_text())["localization"]

    @staticmethod
    def _v2(response: dict) -> dict:
        """The scripted reply marked so a re-roll is visible everywhere."""
        response["domain_glossary"] = [
            {"term_id": g["term_id"], "native": f"v2-{g['native']}"}
            for g in response["domain_glossary"]
        ]
        response["symbol_readouts"] = [
            {**r, "spoken": [f"v2-{s}" for s in r["spoken"]]}
            for r in response["symbol_readouts"]
        ]
        return response

    def test_only_the_requested_glossary_is_rerolled(self, shipped_pack):
        before = self._two_domain_pack(shipped_pack)

        outcome = run_backfill_localization(
            "tl",
            domain="airline",
            force=True,
            llm=scripted_llm([self._v2(airline_response())]),
        )
        assert outcome.written and outcome.replaced and outcome.upgraded
        assert outcome.upgraded_fields == ["domain_glossaries[airline]"]

        block = yaml.safe_load(shipped_pack.read_text())["localization"]
        # The requested domain's glossary IS re-rolled...
        assert all(
            g["native"].startswith("v2-") for g in block["domain_glossaries"]["airline"]
        )
        # ...the other domain's is not...
        assert (
            block["domain_glossaries"]["telecom"]
            == before["domain_glossaries"]["telecom"]
        )
        # ...and no other reviewed field is touched, though the drafted reply
        # carried a fresh (v2-) value for every one of them.
        for field, value in before.items():
            if field != "domain_glossaries":
                assert block[field] == value, f"{field} was re-rolled by --force"

    def test_provenance_names_the_redrafted_glossary(self, shipped_pack):
        self._two_domain_pack(shipped_pack)
        run_backfill_localization(
            "tl",
            domain="airline",
            force=True,
            llm=scripted_llm([airline_response()]),
        )
        text = shipped_pack.read_text()
        assert (
            "domain_glossaries[airline] re-drafted by prompt "
            f"{LOCALIZATION_PROMPT_VERSION} sha256:{localization_prompt_sha256()[:12]}"
            in text
        )
        assert "all previously-reviewed fields preserved verbatim" in text
        assert text.count("\nlocalization:") == 1

    def test_whole_block_reroll_is_the_delete_and_redraft_path(self, shipped_pack):
        """No flag re-rolls a language's whole block; dropping the key does."""
        self._two_domain_pack(shipped_pack)
        text = shipped_pack.read_text()
        stripped = backfill_lib.strip_top_level_block(
            text, key="localization", marker="# --- spoken localization"
        )
        shipped_pack.write_text(stripped)
        outcome = run_backfill_localization(
            "tl", llm=scripted_llm([self._v2(localization_response())])
        )
        assert outcome.written and not outcome.upgraded and not outcome.replaced
        block = yaml.safe_load(shipped_pack.read_text())["localization"]
        assert all(
            s.startswith("v2-") for r in block["symbol_readouts"] for s in r["spoken"]
        )
        # A fresh draft knows only the domain it drafted.
        assert set(block["domain_glossaries"]) == {"telecom"}


class TestInPlaceEditPreservesAuthoredComments:
    """An edit of an EXISTING block must not delete the block's comments.

    A shipped ``localization:`` block accumulates hand-written linguistic notes —
    measured-defect rationale, ``PENDING NATIVE REVIEW`` markers, the
    native-language reasoning behind a TTS respelling. Re-dumping the
    whole block silently deleted every one of them (caught 2026-07-29 on the
    airline glossary re-draft: ko, zh and pt each lost a note). The merge path
    splices field by field instead.
    """

    NOTE = (
        "  # Authored review note: measured in 71 of 300 sims.\n"
        "  # PENDING NATIVE REVIEW.\n"
    )

    def _annotated_two_domain_pack(self, pack) -> str:
        """A complete two-domain block with an authored comment inside it."""
        run_backfill_localization("tl", llm=scripted_llm())
        run_backfill_localization(
            "tl", domain="airline", llm=scripted_llm([airline_response()])
        )
        text = pack.read_text()
        # Annotate a field the airline re-draft has no business touching.
        anchor = "  spelling_alphabet:\n"
        assert anchor in text
        pack.write_text(text.replace(anchor, self.NOTE + anchor))
        return pack.read_text()

    def test_the_comment_survives_a_forced_glossary_redraft(self, shipped_pack):
        self._annotated_two_domain_pack(shipped_pack)
        run_backfill_localization(
            "tl",
            domain="airline",
            force=True,
            llm=scripted_llm([airline_response()]),
        )
        text = shipped_pack.read_text()
        assert "PENDING NATIVE REVIEW." in text
        assert "measured in 71 of 300 sims" in text
        # And it still sits above the key it documents.
        lines = text.splitlines()
        assert lines[lines.index("  # PENDING NATIVE REVIEW.") + 1] == (
            "  spelling_alphabet:"
        )

    def test_only_the_glossary_lines_change(self, shipped_pack):
        before = self._annotated_two_domain_pack(shipped_pack)
        run_backfill_localization(
            "tl",
            domain="airline",
            force=True,
            llm=scripted_llm([TestForceIsDomainScoped._v2(airline_response())]),
        )
        after = shipped_pack.read_text()

        # Splice the SAME placeholder glossary into both and drop the
        # provenance comment: if nothing else moved, the two texts are then
        # byte-identical — no key reordering, no schema-default churn, no
        # re-folded lines anywhere in the block.
        def normalized(text: str) -> str:
            text = set_nested_block(
                text,
                path=("localization", "domain_glossaries", "airline"),
                new_value=[{"term_id": "flight", "native": "placeholder"}],
            )
            return "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))

        assert normalized(before) == normalized(after)

    def test_a_first_write_still_appends_the_whole_block(self, shipped_pack):
        """The fresh-write path is unchanged — there are no comments to keep
        in a block that does not exist yet."""
        outcome = run_backfill_localization("tl", llm=scripted_llm())
        assert outcome.written and not outcome.upgraded
        text = shipped_pack.read_text()
        assert text.count("\nlocalization:") == 1
        assert "# --- spoken localization" in text

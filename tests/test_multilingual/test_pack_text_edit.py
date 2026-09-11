# Copyright Sierra
"""Surgical pack.yaml text edits.

``replace_persona_list_blocks`` is the shared splice behind every redraft verb
(``draft-continuers`` today; the retired one-shot redraft verbs before it). Its
contract is narrow and load-bearing: replace ONE list-valued key inside each
named persona, leave every other byte alone, and refuse a partial rewrite
rather than half-edit a pack. These tests pin the awkward shapes — a blank
line or a comment sitting inside the list — where "stop at the first
surprising line" would silently strand the items after it and let YAML fold
them back into the key that was just rewritten.
"""

import pytest
import yaml

from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.pack_text_edit import (
    insert_marker_before_key,
    insert_marker_before_personas,
    replace_nested_scalar,
    replace_persona_list_blocks,
    set_nested_block,
    strip_marker_block,
)


def pack(persona_block: str) -> str:
    return (
        "language: tl\n"
        "personas:\n"
        "  tessa_tl_v1:\n"
        "    persona_id: tessa_tl_v1\n"
        f"{persona_block}"
        "    tags:\n"
        "      gender: female\n"
        "agent_greeting: hi!\n"
    )


def phrases_of(text: str, key: str = "backchannel_phrases") -> list[str]:
    return yaml.safe_load(text)["personas"]["tessa_tl_v1"][key]


def replace(text: str, new: list[str]) -> str:
    return replace_persona_list_blocks(
        text, key="backchannel_phrases", new_by_persona={"tessa_tl_v1": new}
    )


class TestReplacePersonaListBlocks:
    def test_plain_list_is_replaced_and_siblings_survive(self):
        out = replace(
            pack("    backchannel_phrases:\n    - old\n    - older\n"), ["new"]
        )
        assert phrases_of(out) == ["new"]
        assert "    tags:\n      gender: female\n" in out
        assert "agent_greeting: hi!\n" in out

    def test_blank_line_inside_the_list_does_not_strand_later_items(self):
        """The whole list is replaced, not the part before the blank — items
        left behind would fold straight back into the rewritten key."""
        out = replace(
            pack("    backchannel_phrases:\n    - old\n\n    - stale\n"), ["new"]
        )
        assert phrases_of(out) == ["new"]
        assert "stale" not in out

    def test_comment_inside_the_list_does_not_strand_later_items(self):
        out = replace(
            pack("    backchannel_phrases:\n    - old\n    # a note\n    - stale\n"),
            ["new"],
        )
        assert phrases_of(out) == ["new"]
        assert "stale" not in out
        assert "# a note" not in out  # the note described the replaced items

    def test_trailing_blank_line_after_the_list_is_preserved(self):
        """Only the block changes: a separator line outside it is not the
        verb's to delete."""
        out = replace(
            pack("    backchannel_phrases:\n    - old\n\n"),
            ["new"],
        )
        assert phrases_of(out) == ["new"]
        assert "    - new\n\n    tags:" in out

    def test_a_top_level_comment_after_the_list_is_not_swallowed(self):
        """A comment at column 0 ends the persona block — scanning past it
        would consume the next persona's lines."""
        text = (
            "language: tl\n"
            "personas:\n"
            "  tessa_tl_v1:\n"
            "    backchannel_phrases:\n"
            "    - old\n"
            "\n"
            "# --- a provenance marker ---\n"
            "agent_greeting: hi!\n"
        )
        out = replace(text, ["new"])
        assert phrases_of(out) == ["new"]
        assert "# --- a provenance marker ---\n" in out

    def test_empty_block_is_filled_without_touching_the_next_key(self):
        out = replace(pack("    backchannel_phrases:\n"), ["new"])
        assert phrases_of(out) == ["new"]
        assert "      gender: female" in out

    def test_deeper_indented_continuation_lines_are_consumed(self):
        out = replace(
            pack("    backchannel_phrases:\n    - >-\n        a folded old entry\n"),
            ["new"],
        )
        assert phrases_of(out) == ["new"]
        assert "folded" not in out

    def test_unlocatable_persona_refuses_a_partial_rewrite(self):
        text = pack("    backchannel_phrases:\n    - old\n")
        with pytest.raises(FactoryDraftError, match="refusing a partial rewrite"):
            replace_persona_list_blocks(
                text,
                key="backchannel_phrases",
                new_by_persona={"tessa_tl_v1": ["new"], "ghost_tl_v1": ["new"]},
            )

    def test_only_the_named_key_is_touched(self):
        text = pack(
            "    pragmatics_clauses:\n"
            "    - You are polite.\n"
            "    backchannel_phrases:\n"
            "    - old\n"
        )
        out = replace(text, ["new"])
        assert phrases_of(out, "pragmatics_clauses") == ["You are polite."]
        assert phrases_of(out) == ["new"]


# A pack accumulates one marker per verb that has written to it, stacked
# contiguously above `personas:` — five to seven in the shipped packs.
STACKED_MARKERS = (
    "language: tl\n"
    "# --- persona pragmatics (conditional-realization form) ---\n"
    "# pragmatics_clauses redrafted by `tau2 factory redraft-pragmatics`.\n"
    "# Native content pending per-language owner review.\n"
    "# --- persona out-of-turn speech (away-from-the-mic length) ---\n"
    "# non_directed_phrases clipped to away-from-the-mic length.\n"
    "# --- persona backchannel continuers (pure continuers) ---\n"
    "# backchannel_phrases set by `tau2 factory draft-continuers`.\n"
    "personas:\n"
    "  tessa_tl_v1:\n"
    "    persona_id: tessa_tl_v1\n"
)


class TestStripMarkerBlock:
    def test_replacing_one_marker_leaves_the_stack_below_it_intact(self):
        """A `--force` re-run of one verb must not silently drop the other
        verbs' provenance just because their markers sit underneath."""
        out = strip_marker_block(STACKED_MARKERS, "# --- persona pragmatics")
        assert "# --- persona pragmatics" not in out
        assert "redraft-pragmatics" not in out
        assert "# --- persona out-of-turn speech" in out
        assert "# --- persona backchannel continuers" in out
        assert "non_directed_phrases clipped" in out
        assert "draft-continuers" in out

    def test_stripping_the_last_marker_stops_at_the_content(self):
        out = strip_marker_block(
            STACKED_MARKERS, "# --- persona backchannel continuers"
        )
        assert "draft-continuers" not in out
        assert "# --- persona pragmatics" in out
        assert "personas:\n  tessa_tl_v1:\n" in out

    def test_strip_then_reinsert_is_idempotent(self):
        marker = (
            "# --- persona backchannel continuers (pure continuers) ---\n"
            "# backchannel_phrases set by `tau2 factory draft-continuers`.\n"
        )
        once = insert_marker_before_personas(
            strip_marker_block(STACKED_MARKERS, "# --- persona backchannel continuers"),
            marker,
        )
        twice = insert_marker_before_personas(
            strip_marker_block(once, "# --- persona backchannel continuers"), marker
        )
        assert once == twice
        assert twice.count("# --- ") == 3

    def test_an_unknown_header_changes_nothing(self):
        assert strip_marker_block(STACKED_MARKERS, "# --- nothing here") == (
            STACKED_MARKERS
        )


# A localization block shaped like the shipped packs: a wrapped scalar, an
# authored comment inside the block, and a sibling list of mappings after it.
LOCALIZATION_PACK = """\
language: tl
localization:
  spelling_alphabet:
    # Authored review note that must survive the edit.
    rule: Spell with tl anchors.
    avoid: Do NOT use the NATO/anglo phonetic alphabet ('Alpha, Bravo, Charlie'
      or 'B for boy'); tl spells names another way.
    anchor_examples:
    - written: A
      spoken: A come Ancona
  amount_readout:
    rule: The currency word goes last.
agent_greeting: hi!
"""

AVOID_PATH = ("localization", "spelling_alphabet", "avoid")


class TestReplaceNestedScalar:
    """The primitive behind the 2026-07-27 `avoid` label correction: replace
    ONE nested scalar and leave every other byte — comments included — alone.
    ``backfill_lib``'s whole-block re-dump cannot do that (it drops the
    block's comments)."""

    NEW = "The NATO/anglo phonetic alphabet ('Alpha, Bravo, Charlie')."

    def test_replaces_a_wrapped_scalar_and_keeps_everything_else(self):
        out = replace_nested_scalar(
            LOCALIZATION_PACK, path=AVOID_PATH, new_value=self.NEW
        )
        data = yaml.safe_load(out)
        assert data["localization"]["spelling_alphabet"]["avoid"] == self.NEW
        # The authored comment, the sibling keys, and the list survive.
        assert "# Authored review note that must survive the edit." in out
        assert data["localization"]["spelling_alphabet"]["rule"] == (
            "Spell with tl anchors."
        )
        assert data["localization"]["spelling_alphabet"]["anchor_examples"] == [
            {"written": "A", "spoken": "A come Ancona"}
        ]
        assert data["localization"]["amount_readout"]["rule"] == (
            "The currency word goes last."
        )
        assert data["agent_greeting"] == "hi!"
        assert "Do NOT use" not in out

    def test_only_the_targeted_lines_change(self):
        out = replace_nested_scalar(
            LOCALIZATION_PACK, path=AVOID_PATH, new_value=self.NEW
        )
        before = LOCALIZATION_PACK.splitlines()
        after = out.splitlines()
        untouched_before = [ln for ln in before if "NATO" not in ln and "boy" not in ln]
        untouched_after = [ln for ln in after if "NATO" not in ln and "boy" not in ln]
        assert untouched_before == untouched_after

    def test_round_trips_at_the_same_indent(self):
        out = replace_nested_scalar(
            LOCALIZATION_PACK, path=AVOID_PATH, new_value=self.NEW
        )
        assert "    avoid: " in out
        again = replace_nested_scalar(out, path=AVOID_PATH, new_value=self.NEW)
        assert again == out  # idempotent

    def test_missing_path_raises(self):
        with pytest.raises(FactoryDraftError, match="could not locate"):
            replace_nested_scalar(
                LOCALIZATION_PACK,
                path=("localization", "spelling_alphabet", "nope"),
                new_value="x",
            )

    def test_a_key_holding_a_block_is_not_a_scalar_edit(self):
        with pytest.raises(FactoryDraftError, match="not an inline scalar"):
            replace_nested_scalar(
                LOCALIZATION_PACK,
                path=("localization", "spelling_alphabet"),
                new_value="x",
            )

    def test_a_same_named_key_under_a_different_parent_is_not_matched(self):
        """`rule` exists under both spelling_alphabet and amount_readout — the
        path disambiguates, so no ambiguity error and no wrong edit."""
        out = replace_nested_scalar(
            LOCALIZATION_PACK,
            path=("localization", "amount_readout", "rule"),
            new_value="Currency first.",
        )
        data = yaml.safe_load(out)
        assert data["localization"]["amount_readout"]["rule"] == "Currency first."
        assert data["localization"]["spelling_alphabet"]["rule"] == (
            "Spell with tl anchors."
        )


# A localization block shaped like the shipped packs where the domain-scoped
# `draft-localization --force` edit lands: per-domain glossaries, an authored
# review note documenting the key BELOW it, and an empty inline mapping.
GLOSSARY_PACK = """\
language: tl
localization:
  domain_glossaries:
    airline:
    - term_id: flight
      native: volo
      note: also 'aereo' in casual speech
    - term_id: refund
      native: rimborso
    telecom:
    - term_id: data_plan
      native: piano dati
      note: 'casual: piano'
  # Authored review note. Measured defect: the caller applied this to ITSELF
  # in 71 of 300 sims. PENDING NATIVE REVIEW.
  honorific_self_reference:
  - wrong: lei
    right: io
  digit_names: {}
agent_greeting: hi!
"""

AIRLINE_PATH = ("localization", "domain_glossaries", "airline")
NEW_AIRLINE = [
    {"term_id": "flight", "native": "il volo"},
    {"term_id": "refund", "native": "il rimborso", "note": "re-drafted"},
]


class TestSetNestedBlock:
    """The primitive behind a domain-scoped `draft-localization --force`.

    A shipped `localization:` block carries authored review notes as YAML
    comments; `backfill_lib`'s whole-block re-dump deletes them. Every case
    here is one that re-dump got wrong.
    """

    def test_replaces_one_glossary_and_keeps_its_siblings_verbatim(self):
        out = set_nested_block(GLOSSARY_PACK, path=AIRLINE_PATH, new_value=NEW_AIRLINE)
        data = yaml.safe_load(out)
        assert data["localization"]["domain_glossaries"]["airline"] == NEW_AIRLINE
        # The OTHER domain's glossary is untouched, lines included.
        assert "      note: 'casual: piano'\n" in out
        assert data["localization"]["domain_glossaries"]["telecom"] == [
            {"term_id": "data_plan", "native": "piano dati", "note": "casual: piano"}
        ]

    def test_the_authored_review_note_survives(self):
        out = set_nested_block(GLOSSARY_PACK, path=AIRLINE_PATH, new_value=NEW_AIRLINE)
        assert "PENDING NATIVE REVIEW." in out
        assert "in 71 of 300 sims" in out
        # It still documents the key it was written above, not a stray tail.
        lines = out.splitlines()
        note = lines.index(
            "  # Authored review note. Measured defect: the caller "
            "applied this to ITSELF"
        )
        assert lines[note + 2] == "  honorific_self_reference:"

    def test_untargeted_lines_are_byte_identical(self):
        out = set_nested_block(GLOSSARY_PACK, path=AIRLINE_PATH, new_value=NEW_AIRLINE)
        # Everything outside the airline glossary's own lines is unchanged,
        # in the same order — no key reordering, no normalization churn.
        assert out.split("    telecom:")[1] == GLOSSARY_PACK.split("    telecom:")[1]
        assert out.startswith("language: tl\nlocalization:\n  domain_glossaries:\n")

    def test_re_splicing_the_existing_value_is_a_no_op(self):
        """The identity edit must not perturb a single byte — otherwise every
        re-run rewrites lines it did not mean to touch."""
        current = yaml.safe_load(GLOSSARY_PACK)["localization"]["domain_glossaries"][
            "airline"
        ]
        assert (
            set_nested_block(GLOSSARY_PACK, path=AIRLINE_PATH, new_value=current)
            == GLOSSARY_PACK
        )

    def test_a_missing_key_is_inserted_at_the_end_of_its_parent(self):
        out = set_nested_block(
            GLOSSARY_PACK,
            path=("localization", "domain_glossaries", "retail"),
            new_value=[{"term_id": "order", "native": "ordine"}],
        )
        data = yaml.safe_load(out)
        assert list(data["localization"]["domain_glossaries"]) == [
            "airline",
            "telecom",
            "retail",
        ]
        # Inserted inside the parent, above the comment that follows it.
        assert out.index("    retail:") < out.index("# Authored review note")
        assert "PENDING NATIVE REVIEW." in out

    def test_an_empty_inline_mapping_is_reopened_to_take_a_child(self):
        out = set_nested_block(
            GLOSSARY_PACK,
            path=("localization", "digit_names", "0"),
            new_value="zero",
        )
        assert yaml.safe_load(out)["localization"]["digit_names"] == {"0": "zero"}
        assert "digit_names: {}" not in out

    def test_a_missing_parent_is_a_loud_error(self):
        with pytest.raises(FactoryDraftError, match="refusing to invent"):
            set_nested_block(
                GLOSSARY_PACK,
                path=("localization", "nope", "airline"),
                new_value=[],
            )

    def test_an_absent_top_level_key_is_not_spliced(self):
        with pytest.raises(FactoryDraftError, match="backfill_lib"):
            set_nested_block(GLOSSARY_PACK, path=("delivery",), new_value={})

    def test_a_scalar_parent_cannot_take_a_child(self):
        with pytest.raises(FactoryDraftError, match="inline scalar"):
            set_nested_block(
                GLOSSARY_PACK,
                path=("agent_greeting", "nested"),
                new_value="x",
            )

    def test_folding_matches_the_depth_the_key_sits_at(self):
        """A spliced value is dumped alone and then indented, so the emitter
        must fold early enough that the result wraps where a whole-file dump
        would. Otherwise an identity re-splice reflows long lines."""
        long_note = "formal speakers may say 'identificazione utente'; casual say ID"
        out = set_nested_block(
            GLOSSARY_PACK,
            path=AIRLINE_PATH,
            new_value=[{"term_id": "flight", "native": "volo", "note": long_note}],
        )
        assert all(len(ln) <= 80 for ln in out.splitlines())
        assert (
            yaml.safe_load(out)["localization"]["domain_glossaries"]["airline"][0][
                "note"
            ]
            == long_note
        )


class TestInsertMarkerBeforeKey:
    def test_inserts_above_the_named_key(self):
        out = insert_marker_before_key(
            GLOSSARY_PACK, marker="# --- provenance ---\n", key="localization"
        )
        assert out.index("# --- provenance ---") < out.index("localization:")
        assert yaml.safe_load(out) == yaml.safe_load(GLOSSARY_PACK)

    def test_an_absent_key_is_a_loud_error(self):
        with pytest.raises(FactoryDraftError, match="no top-level `personas:`"):
            insert_marker_before_key(GLOSSARY_PACK, marker="# x\n", key="personas")

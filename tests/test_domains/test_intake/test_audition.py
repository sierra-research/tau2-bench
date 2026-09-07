"""Tests for the pronunciation-audition plan and its rendered indexes
(the wav synthesis itself is exercised by the verb's --limit smoke)."""

from tau2.domains.intake.tasks.audition import (
    _render_index,
    _render_listen_html,
    build_audition_plan,
)


def test_plan_covers_every_mispronounced_entry_riskiest_first():
    items = build_audition_plan()
    groups = []
    for item in items:
        if item.group not in groups:
            groups.append(item.group)
    assert groups == [
        "person_names (hard)",
        "medications (hard)",
        "medications (easy)",
    ]
    # Every item is a gold (default reading) / mispronounced pair; the gold
    # cut is the untouched value (default readings always).
    for item in items:
        assert [cut.kind for cut in item.cuts] == ["gold", "mispronounced"]
        assert item.cuts[0].text == item.label
        assert item.cuts[1].text != item.label


def test_mispronounced_cuts_are_tts_rendered():
    """Every mispronounced cut carries each variant dehyphenated and
    lowercased — the caps stress marker and syllable hyphens are reference
    notation and never reach TTS."""
    from tau2.domains.intake.tasks.banks import load_banks

    banks = load_banks()
    variants: dict[str, list[str]] = {}
    for bank in ("person_names", "medications"):
        for entry in getattr(banks, bank):
            for p in entry.pronunciations or []:
                if p.mispronounced is not None:
                    variants.setdefault(entry.value, []).append(p.mispronounced)
    checked = 0
    for item in build_audition_plan():
        text = item.cuts[1].text
        for variant in variants[item.label]:
            rendered = variant.replace("-", "").lower()
            assert rendered in text, (item.label, variant)
            checked += 1
    assert checked > 100


def test_listen_html_renders_players_and_flags():
    items = build_audition_plan()
    html = _render_listen_html(items)
    assert html.count("<audio") == sum(len(item.cuts) for item in items)
    assert 'data-k="' in html
    assert "Copy flags" in html
    assert "intake_audition_flags_banks" in html
    assert items[0].cuts[0].filename in html


def test_index_md_lists_every_cut():
    items = build_audition_plan()
    index = _render_index(items)
    for item in items[:5]:
        for cut in item.cuts:
            assert cut.filename in index

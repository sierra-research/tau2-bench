# Copyright Sierra
"""HTML annotation packets: manifest provenance, PACKET_CONFIG contract,
--append dedupe semantics, index/entries agreement, tick renderer."""

import re
from pathlib import Path

import pytest
from markupsafe import escape  # jinja's escaper — matches what the template emits

from tau2.annotation.artifacts import ArtifactManifest
from tau2.annotation.models import RealismRow, VoiceReviewRow
from tau2.annotation.packets.builder import PacketBuildOptions, build_packet
from tau2.annotation.packets.forms import FORM_ROW_MODELS, FormType
from tau2.annotation.packets.transcript import (
    format_time_ms,
    generate_message_rows,
    generate_tick_rows,
)
from test_annotation.conftest import (
    FAKE_WAV,
    hi_sim,
    make_packet_results_dir,
    packet_config,
    packet_ticks,
)


def build(run_dir, out_dir, form=FormType.VOICE_REVIEW, **kwargs):
    return build_packet(
        PacketBuildOptions(
            form=form,
            batch_name="round1",
            results=[run_dir],
            out_dir=out_dir,
            **kwargs,
        )
    )


@pytest.fixture
def built(tmp_path):
    run_dir = make_packet_results_dir(tmp_path)
    out_dir = tmp_path / "packet"
    manifest_path = build(run_dir, out_dir)
    return run_dir, out_dir, manifest_path


def read_manifest(out_dir) -> ArtifactManifest:
    return ArtifactManifest.model_validate_json((out_dir / "manifest.json").read_text())


# ---------------------------------------------------------------------------
# Build: manifest, entries, audio, batch id
# ---------------------------------------------------------------------------


def test_build_writes_pages_index_and_manifest(built):
    _, out_dir, manifest_path = built
    assert manifest_path == out_dir / "manifest.json"
    assert (out_dir / "index.html").exists()
    assert (out_dir / "task_t1_sim_s1" / "index.html").exists()
    assert (out_dir / "task_t2_sim_s2" / "index.html").exists()

    manifest = read_manifest(out_dir)
    assert manifest.kind == "packet_voice_review"
    assert manifest.form == "voice_review"
    assert manifest.batch_name == "round1"
    assert manifest.domain == "airline"
    assert manifest.provenance["rubric_version"]

    by_id = {e.sim_id: e for e in manifest.entries}
    assert set(by_id) == {"s1", "s2"}
    e1 = by_id["s1"]
    assert e1.task_id == "t1"
    assert e1.trial == 0
    assert e1.dir_name == "task_t1_sim_s1"
    assert e1.experiment == "hi_voice_run"
    assert e1.domain == "airline"
    assert e1.has_audio is True
    assert by_id["s2"].has_audio is False


def test_audio_copied_when_present(built):
    _, out_dir, _ = built
    copied = out_dir / "task_t1_sim_s1" / "audio.wav"
    assert copied.read_bytes() == FAKE_WAV
    assert not (out_dir / "task_t2_sim_s2" / "audio.wav").exists()
    # The page with audio embeds the player; the one without doesn't.
    assert 'id="mainAudio"' in (out_dir / "task_t1_sim_s1" / "index.html").read_text()
    assert (
        'id="mainAudio"' not in (out_dir / "task_t2_sim_s2" / "index.html").read_text()
    )


def test_batch_id_is_content_derived(tmp_path):
    run_a = make_packet_results_dir(tmp_path / "a")
    run_b = make_packet_results_dir(tmp_path / "b")  # same sims, same name
    build(run_a, tmp_path / "packet_a")
    build(run_b, tmp_path / "packet_b")
    id_a = read_manifest(tmp_path / "packet_a").batch_id
    id_b = read_manifest(tmp_path / "packet_b").batch_id
    assert id_a == id_b  # identical inputs reproduce the id

    run_c = make_packet_results_dir(tmp_path / "c", sim_ids=("s7", "s8"))
    build(run_c, tmp_path / "packet_c")
    assert read_manifest(tmp_path / "packet_c").batch_id != id_a


def test_existing_dir_without_append_errors(built, tmp_path):
    run_dir, out_dir, _ = built
    with pytest.raises(FileExistsError, match="--append"):
        build(run_dir, out_dir)


def test_max_items_caps_export(tmp_path):
    run_dir = make_packet_results_dir(tmp_path)
    out_dir = tmp_path / "packet"
    build(run_dir, out_dir, max_items=1)
    assert len(read_manifest(out_dir).entries) == 1


# ---------------------------------------------------------------------------
# PACKET_CONFIG: model-generated csv_headers on every page
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", list(FormType))
def test_packet_config_carries_model_headers(tmp_path, form):
    run_dir = make_packet_results_dir(tmp_path)
    out_dir = tmp_path / f"packet_{form.value}"
    build(run_dir, out_dir, form=form)
    manifest = read_manifest(out_dir)

    expected_headers = FORM_ROW_MODELS[form].headers()
    index_cfg = packet_config(out_dir / "index.html")
    assert index_cfg["csv_headers"] == expected_headers
    assert index_cfg["form"] == form.value
    assert index_cfg["batch_id"] == manifest.batch_id
    assert index_cfg["batch_name"] == "round1"

    sim_cfg = packet_config(out_dir / "task_t1_sim_s1" / "index.html")
    assert sim_cfg["csv_headers"] == expected_headers
    assert sim_cfg["sim_id"] == "s1"
    assert sim_cfg["task_id"] == "t1"
    assert sim_cfg["trial"] == 0
    assert sim_cfg["batch_id"] == manifest.batch_id
    if form is FormType.ERROR_ANALYSIS:
        assert sim_cfg["dims"] == []
    else:
        assert [d["id"] for d in sim_cfg["dims"]] == RealismRow.headers()[5:13]


def test_realism_dims_lockstep_with_rubric():
    from tau2.annotation.models import REALISM_DIMENSION_IDS
    from tau2.annotation.packets.forms import load_rubric

    assert [d.id for d in load_rubric().dimensions] == REALISM_DIMENSION_IDS
    assert RealismRow.headers()[5:13] == REALISM_DIMENSION_IDS
    assert VoiceReviewRow.headers()[8:16] == REALISM_DIMENSION_IDS


def test_voice_review_headers_compose_findings_dims_and_experience():
    headers = VoiceReviewRow.headers()
    assert headers[:5] == ["batch", "rater", "task_id", "simulation_id", "trial"]
    assert headers[5:8] == ["error_source", "error_type", "notes"]
    # The caller-experience block sits between the realism block and the
    # client-state tail.
    assert headers[-8:] == [
        "caller_experience",
        "experience_breaking_point",
        "experience_breaking_tick",
        "experience_factors",
        "primary_factor",
        "experience_notes",
        "completed",
        "created_at",
    ]
    assert "speech_accuracy" in headers and "phrasing_naturalness" in headers


def test_voice_review_page_renders_experience_section(built):
    """The voice_review page carries the caller-experience section and its
    PACKET_CONFIG block; the error_analysis page carries neither."""
    _, out_dir, _ = built
    sim_page = out_dir / "task_t1_sim_s1" / "index.html"
    html = sim_page.read_text()
    assert 'name="caller_experience"' in html
    assert 'name="experience_breaking_point"' in html
    assert 'name="primary_factor"' in html
    assert 'data-factor="sim_multilingual_broken"' in html

    cfg = packet_config(sim_page)
    from tau2.annotation.models import ExperienceFactor

    assert cfg["experience"]["factor_ids"] == [m.value for m in ExperienceFactor]
    assert cfg["experience"]["elaborate_ids"] == ["sim_multilingual_broken", "other"]
    assert cfg["experience"]["breaking_points"] == ["overall", "tick"]


def test_experience_factor_table_renders_every_factor(built):
    """Every taxonomy member gets a contributed checkbox, a primary radio, and
    its display label — a factor that reaches PACKET_CONFIG but not the table
    would be selectable in the CSV and invisible in the form."""
    from tau2.annotation.models import ExperienceFactor

    _, out_dir, _ = built
    html = (out_dir / "task_t1_sim_s1" / "index.html").read_text()
    for member in ExperienceFactor:
        assert f'class="exp-factor-check" data-factor="{member.value}"' in html
        assert f'class="exp-factor-primary" value="{member.value}"' in html
        assert escape(member.display) in html


def test_repetition_factor_and_its_overlap_guidance_render(built):
    """The new symptom factor and the sentence telling annotators to pair it
    with its cause both reach the page annotators actually read."""
    _, out_dir, _ = built
    html = (out_dir / "task_t1_sim_s1" / "index.html").read_text()
    assert 'data-factor="repetition"' in html
    assert "Had to repeat or re-explain (said the same thing twice" in html
    assert "Factors overlap on purpose — check the symptom and its cause." in html
    assert "checking both is the correct answer, not a double-count" in html
    assert "Make primary whichever dominated the call" in html


def test_error_analysis_page_has_no_experience_section(tmp_path):
    run_dir = make_packet_results_dir(tmp_path)
    out_dir = tmp_path / "packet_ea"
    build(run_dir, out_dir, form=FormType.ERROR_ANALYSIS)
    sim_page = out_dir / "task_t1_sim_s1" / "index.html"
    assert 'name="caller_experience"' not in sim_page.read_text()
    assert "experience" not in packet_config(sim_page)


# ---------------------------------------------------------------------------
# Provenance: what the build read, identified rather than merely located
# ---------------------------------------------------------------------------


def test_build_records_the_identity_of_every_results_source(built):
    """A path is a location; provenance must also say WHICH run was read, so
    the source can be traced after the tree moves (tau2 annotate
    source-audit)."""
    from tau2.annotation.provenance import build_records
    from tau2.data_model.run_identity import compute_run_identity

    run_dir, out_dir, _ = built
    (build_record,) = build_records(read_manifest(out_dir).provenance)
    (source,) = build_record.results
    assert source.path == str(run_dir)
    assert source.identity == compute_run_identity(run_dir)


def test_a_moved_source_run_is_repaired_from_its_recorded_identity(built, tmp_path):
    """End to end: build a packet, reorganise the run tree, audit it back."""
    import shutil

    from tau2.annotation.provenance import build_records
    from tau2.annotation.source_audit import SourceStatus, audit_sources

    run_dir, out_dir, _ = built
    annotations_root = tmp_path / "annotations"
    annotations_root.mkdir()
    shutil.move(str(out_dir), str(annotations_root / out_dir.name))
    simulations_root = tmp_path / "simulations"
    moved = simulations_root / "multilingual" / run_dir.name
    moved.parent.mkdir(parents=True)
    shutil.move(str(run_dir), str(moved))

    report = audit_sources(annotations_root, simulations_root, apply=True)

    assert [c.status for c in report.checks] == [SourceStatus.RELOCATED]
    manifest = read_manifest(annotations_root / out_dir.name)
    (build_record,) = build_records(manifest.provenance)
    assert build_record.results[0].path == str(moved)


# ---------------------------------------------------------------------------
# --append: dedupe by sim id, preserve prior labels, reject form mismatch
# ---------------------------------------------------------------------------


def test_append_same_results_keeps_entries_and_prior_builds(built):
    run_dir, out_dir, _ = built
    before = read_manifest(out_dir)
    build(run_dir, out_dir, append=True)
    after = read_manifest(out_dir)
    assert after.batch_id == before.batch_id
    assert [e.sim_id for e in after.entries] == [e.sim_id for e in before.entries]
    # provenance is an append-preserving builds list: the prior build record
    # survives verbatim and the append records its own invocation.
    prior_builds = before.provenance["builds"]
    assert after.provenance["builds"][: len(prior_builds)] == prior_builds
    assert len(after.provenance["builds"]) == len(prior_builds) + 1


def test_append_only_new_run_preserves_prior_provenance(built, tmp_path):
    run_dir, out_dir, _ = built
    run2 = make_packet_results_dir(
        tmp_path / "second", name="hi_voice_run2", sim_ids=("s3", "s4")
    )
    before = read_manifest(out_dir)
    # Append passing ONLY the new run: run-A's build record (with its results
    # paths) must survive — --append never rewrites history.
    build(run2, out_dir, append=True)
    after = read_manifest(out_dir)
    assert after.provenance["builds"][0] == before.provenance["builds"][0]
    assert str(run_dir) in str(before.provenance["builds"][0])
    assert str(run2) in str(after.provenance["builds"][-1])
    assert {e.sim_id for e in after.entries} == {"s1", "s2", "s3", "s4"}


def test_append_new_run_dedupes_and_preserves_prior_labels(built, tmp_path):
    run_dir, out_dir, _ = built
    run2 = make_packet_results_dir(
        tmp_path / "second", name="hi_voice_run2", sim_ids=("s3", "s4")
    )
    before = read_manifest(out_dir)
    # Appending BOTH runs: run_dir's sims dedupe away; run2's export fresh.
    build_packet(
        PacketBuildOptions(
            form=FormType.VOICE_REVIEW,
            batch_name="round1",
            results=[run_dir, run2],
            out_dir=out_dir,
            append=True,
        )
    )
    after = read_manifest(out_dir)
    assert after.batch_id == before.batch_id
    by_id = {e.sim_id: e for e in after.entries}
    assert set(by_id) == {"s1", "s2", "s3", "s4"}
    # Prior entries keep their REAL experiment/domain labels.
    assert by_id["s1"].experiment == "hi_voice_run"
    assert by_id["s3"].experiment == "hi_voice_run2"
    assert (out_dir / "task_t1_sim_s3" / "index.html").exists()


def test_append_max_items_caps_new_sims_not_already_exported_rows(built, tmp_path):
    """Prior sims must not consume the append cap before eligible new sims."""
    run_dir, out_dir, _ = built
    run2 = make_packet_results_dir(
        tmp_path / "second", name="hi_voice_run2", sim_ids=("s3", "s4", "s5")
    )

    build_packet(
        PacketBuildOptions(
            form=FormType.VOICE_REVIEW,
            batch_name="round1",
            results=[run_dir, run2],
            out_dir=out_dir,
            append=True,
            max_items=2,
        )
    )

    assert {e.sim_id for e in read_manifest(out_dir).entries} == {
        "s1",
        "s2",
        "s3",
        "s4",
    }


def test_overlapping_results_dedupe_within_one_build(tmp_path):
    """Passing the same run (or runs sharing sims) twice in ONE invocation
    must export each sim once, not duplicate pages/entries."""
    run_dir = make_packet_results_dir(tmp_path)
    out_dir = tmp_path / "packet"
    build_packet(
        PacketBuildOptions(
            form=FormType.VOICE_REVIEW,
            batch_name="round1",
            results=[run_dir, run_dir],
            out_dir=out_dir,
        )
    )
    manifest = read_manifest(out_dir)
    sim_ids = [e.sim_id for e in manifest.entries]
    assert sorted(sim_ids) == ["s1", "s2"]  # no duplicates
    assert len(list(out_dir.glob("task_*_sim_*"))) == 2


def test_append_rejects_mismatched_form(built):
    run_dir, out_dir, _ = built
    with pytest.raises(ValueError, match="one form per batch"):
        build(run_dir, out_dir, form=FormType.USER_REALISM, append=True)


def test_append_to_a_packet_without_run_identities_says_rebuild(built):
    """Packets are regenerable: the pre-identity provenance shape gets an
    actionable rebuild error, not a raw validation traceback."""
    import json

    run_dir, out_dir, manifest_path = built
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["builds"] = [{"results": [str(run_dir)]}]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="rebuild it from scratch"):
        build(run_dir, out_dir, append=True)


def test_append_requires_manifest(tmp_path):
    run_dir = make_packet_results_dir(tmp_path)
    out_dir = tmp_path / "packet"
    out_dir.mkdir()
    (out_dir / "stray.txt").write_text("not a packet")
    with pytest.raises(FileNotFoundError, match="manifest"):
        build(run_dir, out_dir, append=True)


# ---------------------------------------------------------------------------
# Prev/next navigation (shared _nav.html.j2 macro over the index order)
# ---------------------------------------------------------------------------

NAV_BLOCK_RE = re.compile(r'<nav class="packet-nav">.*?</nav>', re.DOTALL)
STYLE_RE = re.compile(r"<style>.*?</style>", re.DOTALL)


def nav_blocks(page: Path) -> list[str]:
    return NAV_BLOCK_RE.findall(page.read_text())


def page_text(page: Path) -> str:
    """Page HTML minus the embedded stylesheet (which mentions the nav class
    names in selectors on every page, nav bar or not)."""
    return STYLE_RE.sub("", page.read_text())


def make_three_sim_run(tmp_path, name="hi_voice_run"):
    from fixtures_runs import make_hi_results

    return make_hi_results(
        tmp_path,
        [hi_sim("s1", "t1"), hi_sim("s2", "t2"), hi_sim("s3", "t3")],
        name=name,
    )


# The FCE (error_analysis) packet is the headline consumer; the others share
# the same render path.
@pytest.mark.parametrize("form", [FormType.ERROR_ANALYSIS, FormType.VOICE_REVIEW])
def test_nav_links_follow_index_order(tmp_path, form):
    """First page: disabled Previous; middle: both neighbors; last: 'Done —
    back to index'. Order = the index listing's dir-name sort."""
    run_dir = make_three_sim_run(tmp_path)
    out_dir = tmp_path / f"packet_{form.value}"
    build(run_dir, out_dir, form=form)

    p1 = page_text(out_dir / "task_t1_sim_s1" / "index.html")
    p2 = page_text(out_dir / "task_t2_sim_s2" / "index.html")
    p3 = page_text(out_dir / "task_t3_sim_s3" / "index.html")

    # First: disabled Previous, Next -> the second page.
    assert "packet-nav-disabled" in p1
    assert 'href="../task_t2_sim_s2/index.html"' in p1
    assert "Call 1 of 3" in p1
    assert "Done — back to index" not in p1
    # Every bar links back to the index.
    for page in (p1, p2, p3):
        assert 'href="../index.html"' in page
    # Middle: Prev -> first, Next -> last.
    assert 'href="../task_t1_sim_s1/index.html"' in p2
    assert 'href="../task_t3_sim_s3/index.html"' in p2
    assert "packet-nav-disabled" not in p2
    assert "Call 2 of 3" in p2
    # Last: Prev -> middle, Next -> the index ("Done").
    assert 'href="../task_t2_sim_s2/index.html"' in p3
    assert "Done — back to index" in p3
    assert "Call 3 of 3" in p3
    # No page links to itself or skips a neighbor.
    assert 'href="../task_t1_sim_s1/index.html"' not in p1
    assert 'href="../task_t3_sim_s3/index.html"' not in p1
    # Two bars per page (top and bottom).
    for path in ("task_t1_sim_s1", "task_t2_sim_s2", "task_t3_sim_s3"):
        assert len(nav_blocks(out_dir / path / "index.html")) == 2


def test_append_refreshes_prior_pages_nav(built, tmp_path):
    """--append re-renders only new pages, but nav is packet-order derived —
    prior pages' bars must be rewritten to the merged order, byte-identical
    to what a single fresh build over both runs produces."""
    run_dir, out_dir, _ = built
    run2 = make_packet_results_dir(
        tmp_path / "second", name="hi_voice_run2", sim_ids=("s3", "s4")
    )
    # Before the append, s2's page is the last of two.
    s2_page = out_dir / "task_t2_sim_s2" / "index.html"
    assert "Done — back to index" in s2_page.read_text()

    build(run2, out_dir, append=True)

    # Merged dir-name order: t1_s1, t1_s3, t2_s2, t2_s4.
    s1 = (out_dir / "task_t1_sim_s1" / "index.html").read_text()
    assert "Call 1 of 4" in s1
    assert 'href="../task_t1_sim_s3/index.html"' in s1  # next is the NEW page
    s2 = s2_page.read_text()
    assert "Call 3 of 4" in s2
    assert "Done — back to index" not in s2  # no longer last
    assert 'href="../task_t2_sim_s4/index.html"' in s2
    assert (
        "Done — back to index"
        in (out_dir / "task_t2_sim_s4" / "index.html").read_text()
    )

    # Equivalence: nav bars match a fresh single-invocation build exactly.
    fresh_dir = tmp_path / "fresh"
    build_packet(
        PacketBuildOptions(
            form=FormType.VOICE_REVIEW,
            batch_name="round1",
            results=[run_dir, run2],
            out_dir=fresh_dir,
        )
    )
    for dir_name in (
        "task_t1_sim_s1",
        "task_t1_sim_s3",
        "task_t2_sim_s2",
        "task_t2_sim_s4",
    ):
        appended = nav_blocks(out_dir / dir_name / "index.html")
        fresh = nav_blocks(fresh_dir / dir_name / "index.html")
        assert appended == fresh and len(appended) == 2, dir_name


def test_noop_append_keeps_nav_intact(built):
    """An append that adds nothing must leave every page's bars coherent."""
    run_dir, out_dir, _ = built
    before = {
        d: nav_blocks(out_dir / d / "index.html")
        for d in ("task_t1_sim_s1", "task_t2_sim_s2")
    }
    build(run_dir, out_dir, append=True)
    after = {
        d: nav_blocks(out_dir / d / "index.html")
        for d in ("task_t1_sim_s1", "task_t2_sim_s2")
    }
    assert after == before


# ---------------------------------------------------------------------------
# Index page rows come from the manifest entries
# ---------------------------------------------------------------------------


def test_index_rows_match_entries(built):
    _, out_dir, _ = built
    manifest = read_manifest(out_dir)
    index_html = (out_dir / "index.html").read_text()
    sim_ids = re.findall(r'data-sim="([^"]+)"', index_html)
    assert sorted(sim_ids) == sorted(e.sim_id for e in manifest.entries)
    for entry in manifest.entries:
        assert f'href="{entry.dir_name}/index.html"' in index_html
        assert entry.experiment in index_html


# ---------------------------------------------------------------------------
# Tick renderer (ported verbatim — golden-ish behavior checks)
# ---------------------------------------------------------------------------


def test_tick_renderer_groups_and_escapes():
    sim = hi_sim("s1", "t1")
    sim.ticks = packet_ticks()
    html = generate_tick_rows(sim)
    rows = re.findall(
        r"<tr [^>]*data-tick-start=\"(\d+)\" data-tick-end=\"(\d+)\"", html
    )
    # Ticks 0-1 (consecutive agent speech) merge; tick 2 (tool) is isolated;
    # tick 3 (user reply) is its own row.
    assert rows == [("0", "1"), ("2", "2"), ("3", "3")]
    # Merged agent content, HTML-escaped.
    assert "नमस्ते, मैं कैसे &lt;मदद&gt; करूँ?" in html
    # Tool call rendered functionally; tool result collapsible.
    assert "find_reservation" in html
    assert "Result 1" in html
    # Audio sync metadata: 200ms ticks -> tick 3 starts at 0.6s.
    assert 'data-start-time="0.6"' in html


def test_tick_renderer_without_ticks():
    sim = hi_sim("s1", "t1")
    assert "No tick data available" in generate_tick_rows(sim)


def test_tick_renderer_badges_interrupted_agent_rows():
    """A caller barge-in (chunk signal: raw_data.was_truncated) puts ONE
    truncation badge at the cut position — and nowhere else."""
    sim = hi_sim("s1", "t1")
    sim.ticks = packet_ticks()
    assert "truncation-badge" not in generate_tick_rows(sim)

    sim.ticks[1].agent_chunk.raw_data = {"was_truncated": True}
    html = generate_tick_rows(sim)
    assert html.count("truncation-badge") == 1
    assert "[truncated]" in html
    badged_row = html[: html.index("truncation-badge")]
    assert 'data-tick-start="0" data-tick-end="1"' in badged_row


def test_tick_renderer_badge_lands_only_at_the_cut_row(monkeypatch):
    """An utterance the grouping splits across rows (a tool tick isolates its
    own row) gets the badge ONLY in the row holding its last tick — per-row
    overlap marking would smear one cut over every fragment row."""
    from tau2.data_model.message import AssistantMessage, Tick, ToolCall, ToolMessage

    ts = "2026-01-01T00:00:00"
    sim = hi_sim("s1", "t1")
    sim.ticks = [
        Tick(
            tick_id=0,
            timestamp=ts,
            agent_chunk=AssistantMessage(
                role="assistant", content="vou verificar q", utterance_ids=["u0"]
            ),
        ),
        Tick(
            tick_id=1,
            timestamp=ts,
            agent_chunk=AssistantMessage(
                role="assistant", content="ual", utterance_ids=["u0"]
            ),
            agent_tool_calls=[
                ToolCall(id="tc1", name="get_details_by_id", arguments={"id": "L1"})
            ],
            agent_tool_results=[
                ToolMessage(id="tc1", role="tool", content='{"ok": true}')
            ],
        ),
        Tick(
            tick_id=2,
            timestamp=ts,
            agent_chunk=AssistantMessage(
                role="assistant",
                content=" linha",
                utterance_ids=["u0"],
                raw_data={"was_truncated": True},
            ),
        ),
    ]
    html = generate_tick_rows(sim)
    # One utterance, three rows (the tool tick isolates its own row) — but
    # exactly ONE badge, in the final row where the cut actually happened.
    assert html.count("truncation-badge") == 1
    badged_row = html[: html.index("truncation-badge")]
    assert 'data-tick-start="2" data-tick-end="2"' in badged_row


def test_format_time_ms():
    assert format_time_ms(0) == "0:00.000"
    assert format_time_ms(61_234) == "1:01.234"


# ---------------------------------------------------------------------------
# Message renderer (half-duplex / text sims)
# ---------------------------------------------------------------------------


def _text_messages():
    from tau2.data_model.message import (
        AssistantMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    call = ToolCall(
        id="c1", name="apply_for_credit_card", arguments={"card_type": "Gold"}
    )
    return [
        AssistantMessage(role="assistant", content="Hi, how can I <help>?", turn_idx=0),
        UserMessage(role="user", content="I want a card.", turn_idx=1),
        AssistantMessage(role="assistant", content=None, tool_calls=[call], turn_idx=2),
        ToolMessage(
            id="c1",
            role="tool",
            content='{"ok": true}',
            requestor="assistant",
            turn_idx=3,
        ),
        UserMessage(role="user", content="Thanks, bye!", turn_idx=4),
    ]


def test_message_renderer_turns_tools_and_escapes():
    sim = hi_sim("s1", "t1")
    sim.messages = _text_messages()
    html = generate_message_rows(sim)
    rows = re.findall(r"<tr data-tick-start=\"(\d+)\" data-tick-end=\"(\d+)\"", html)
    # One row per participant turn; the tool-call turn absorbs its result, so
    # its range spans message indices 2-3 — the turn_idx span judge findings
    # anchor on.
    assert rows == [("0", "0"), ("1", "1"), ("2", "3"), ("4", "4")]
    assert "Hi, how can I &lt;help&gt;?" in html
    assert "apply_for_credit_card" in html
    assert "Result 1" in html
    # A turn loop has no clock and no audio: no time cells, no seek metadata.
    assert "data-start-time" not in html
    assert "clickable-time" not in html


def test_message_renderer_message_only_skips_tool_turns():
    sim = hi_sim("s1", "t1")
    sim.messages = _text_messages()
    html = generate_message_rows(sim, include_tools=False)
    rows = re.findall(r"<tr data-tick-start=\"(\d+)\"", html)
    assert rows == ["0", "1", "4"]  # the content-less tool turn is dropped
    assert "apply_for_credit_card" not in html


def test_message_renderer_without_messages():
    sim = hi_sim("s1", "t1")
    sim.messages = []
    assert "No messages available" in generate_message_rows(sim)


def test_review_markers_anchor_on_turn_idx_for_text_sims():
    from tau2.annotation.packets.page import build_review_vm
    from tau2.data_model.simulation import Review, ReviewError

    sim = hi_sim("s1", "t1")
    sim.messages = _text_messages()
    sim.review = Review(
        summary="one user slip",
        has_errors=True,
        user_error=True,
        errors=[
            ReviewError(
                source="user",
                error_tags=["premature_hangup"],
                severity="minor",
                turn_idx=4,
                reasoning="hung up early",
            )
        ],
    )
    vm, markers = build_review_vm(sim)
    assert vm is not None and len(markers) == 1
    # turn_idx degrades to a [idx, idx] tick range, which is what the message
    # rows carry in data-tick-start/end — the marker JS needs no text branch.
    assert markers[0].tick_start == markers[0].tick_end == 4
    assert vm.errors[0].clickable


def test_rating_forms_render_na_option_per_dimension(tmp_path):
    from tau2.annotation.models import REALISM_DIMENSION_IDS

    for form in (FormType.USER_REALISM, FormType.VOICE_REVIEW):
        run_dir = make_packet_results_dir(tmp_path / form.value)
        out_dir = tmp_path / f"packet_{form.value}"
        build(run_dir, out_dir, form=form)
        html = (out_dir / "task_t1_sim_s1" / "index.html").read_text()
        # One N/A control per rubric dimension, distinct from the 1-4 scale.
        assert html.count('class="score-option score-option-na"') == len(
            REALISM_DIMENSION_IDS
        )
        assert html.count('value="NA"') == len(REALISM_DIMENSION_IDS)
        assert "Unable to evaluate" in html
        # The form JS treats NA like a low score for the notes requirement.
        assert "checked.value === 'NA'" in html


# ---------------------------------------------------------------------------
# Enriched view: every packet carries the arm label and outcome surfaces
# ---------------------------------------------------------------------------


def test_error_analysis_pages_are_always_enriched(tmp_path):
    # Packets are always enriched — arm label, reward, and judge review are
    # on every annotator-visible surface (the blind option was removed).
    run_dir = make_packet_results_dir(tmp_path)
    out_dir = tmp_path / "packet_labeled"
    build(run_dir, out_dir, form=FormType.ERROR_ANALYSIS)
    html = (out_dir / "task_t1_sim_s1" / "index.html").read_text()
    assert "hi_voice_run" in html
    assert "<th>Reward</th>" in html
    assert "hi_voice_run" in (out_dir / "index.html").read_text()
    assert {e.experiment for e in read_manifest(out_dir).entries} == {"hi_voice_run"}

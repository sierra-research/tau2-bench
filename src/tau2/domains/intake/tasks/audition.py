# Copyright Sierra
"""Pronunciation audition (design doc sec. 6 gate 2): synthesize wav pairs
for owner spot-listening — ``tau2 intake-tasks audition-pronunciations``.

For every bank entry carrying a mispronounced variant (hard person names,
all medications) the verb synthesizes TWO wavs through the exact ElevenLabs
path a run uses (:func:`tau2.voice.synthesis.synthesize.synthesize_voice`)
with a fixed stock voice:

- ``gold`` — the default TTS reading of the gold spelling (what every
  uncomplicated run says: default readings always, owner decision after the
  2026-08-26 audition listen — the correct respellings stay in the banks as
  reference data but are never swapped in);
- ``mispronounced`` — the value with every mispronounced-bearing token
  replaced by its curated bad variant (stored TTS-ready in natural
  orthography, e.g. "iburprofen" — design doc sec. 4), i.e. what a
  ``mispronounced_term`` draw of each such token sends.

File naming is deterministic (``{bank}_{index:03d}_{kind}.wav``; the index
is the entry's position in its bank file, so names survive re-runs), the
job is resumable (existing wavs are skipped, the indexes are rewritten),
and ``index.md`` plus the listen page ``index.html`` (inline players, flag +
copy-report) group entries riskiest-first: hard person names, hard
medications, easy medications.
"""

from pathlib import Path
from typing import Annotated, List, Optional

from loguru import logger
from pydantic import Field

from tau2.utils.pydantic_utils import BaseModelNoExtra

# The fixed stock voice every audition wav uses: comparable across entries,
# and the same ElevenLabs voice family runs use.
AUDITION_PERSONA = "matt_delaney"
# Fixed TTS seed so re-synthesizing a missing file is as reproducible as the
# provider allows.
AUDITION_TTS_SEED = 7

_GOLD = "gold"
_MISPRONOUNCED = "mispronounced"


class AuditionCut(BaseModelNoExtra):
    """One wav to synthesize: its deterministic filename and exact text."""

    filename: Annotated[str, Field(description="Deterministic wav filename.")]
    kind: Annotated[str, Field(description="gold | correct | mispronounced.")]
    text: Annotated[str, Field(description="The exact text sent to TTS.")]


class AuditionItem(BaseModelNoExtra):
    """One audition row: a bank entry (or oddball token) and its cuts."""

    group: Annotated[
        str, Field(description="Riskiest-first section the item renders under.")
    ]
    label: Annotated[str, Field(description="The gold value or token.")]
    cuts: Annotated[List[AuditionCut], Field(description="The wavs to render.")]


class AuditionOutcome(BaseModelNoExtra):
    """What one audition invocation planned and produced."""

    out_dir: Annotated[Path, Field(description="Directory the wavs landed in.")]
    items: Annotated[int, Field(description="Entries/tokens auditioned.")]
    synthesized: Annotated[int, Field(description="Wavs synthesized this run.")]
    skipped: Annotated[int, Field(description="Wavs already present (resumable job).")]
    index_path: Annotated[Path, Field(description="The written index.md.")]


def _entry_cuts(bank: str, index: int, entry) -> Optional[AuditionItem]:
    """The gold/mispronounced texts for one mispronounced-bearing bank entry,
    via the SAME whole-token swap the run applies (the stored variant is
    already TTS-ready natural orthography). Entries without a mispronounced
    variant have nothing to audition (their gold reading is the run's
    reading) and return None."""
    from tau2.voice.utils.pronunciation_swap import apply_pronunciations

    bad_map = {
        p.token: p.mispronounced
        for p in entry.pronunciations or []
        if p.mispronounced is not None
    }
    if not bad_map:
        return None
    cuts = [
        AuditionCut(
            filename=f"{bank}_{index:03d}_{_GOLD}.wav",
            kind=_GOLD,
            text=entry.value,
        ),
        AuditionCut(
            filename=f"{bank}_{index:03d}_{_MISPRONOUNCED}.wav",
            kind=_MISPRONOUNCED,
            text=apply_pronunciations(entry.value, bad_map)[0],
        ),
    ]
    return AuditionItem(
        group=f"{bank} ({entry.difficulty.value})", label=entry.value, cuts=cuts
    )


def build_audition_plan() -> List[AuditionItem]:
    """The deterministic audition plan (pure function of the checked-in data).

    Every mispronounced-bearing bank entry (hard person names, then hard and
    easy medications — riskiest-first). Entries without a mispronounced
    variant are the default TTS reading in runs and are not auditioned.
    """
    from tau2.domains.intake.tasks.banks import Difficulty, load_banks

    banks = load_banks()
    sections: list[tuple[str, Difficulty]] = [
        ("person_names", Difficulty.HARD),
        ("medications", Difficulty.HARD),
        ("medications", Difficulty.EASY),
    ]
    items: List[AuditionItem] = []
    for bank, tier in sections:
        for index, entry in enumerate(getattr(banks, bank)):
            if entry.difficulty is tier:
                item = _entry_cuts(bank, index, entry)
                if item is not None:
                    items.append(item)
    return items


def _render_index(items: List[AuditionItem]) -> str:
    lines: list[str] = []
    lines.append("# Intake pronunciation audition")
    lines.append("")
    lines.append(
        f"Provenance: fixed stock voice persona {AUDITION_PERSONA}, TTS seed "
        f"{AUDITION_TTS_SEED}, gold (default TTS reading) vs mispronounced "
        "pairs for every mispronounced-bearing bank entry. "
        "Regenerable via `tau2 intake-tasks audition-pronunciations`; "
        "existing wavs are kept (resumable), the index is rewritten. "
        "Grouped riskiest-first."
    )
    group = None
    for item in items:
        if item.group != group:
            group = item.group
            lines.append("")
            lines.append(f"## {group}")
            lines.append("")
            lines.append("| value | cut | wav | text sent to TTS |")
            lines.append("|---|---|---|---|")
        for cut in item.cuts:
            lines.append(f"| {item.label} | {cut.kind} | {cut.filename} | {cut.text} |")
    lines.append("")
    return "\n".join(lines)


# Fixed in-code listen page (machine-not-scripts): one row per cut with an
# inline player, a per-item flag, and a copy-report button whose output
# pastes back into chat. Flags persist in localStorage keyed by out-dir mode.
_LISTEN_HTML_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>Intake pronunciation audition</title>
<style>
body{margin:0;background:#faf9f6;color:#20221f;
  font:15px/1.5 system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;background:#faf9f6;border-bottom:1px solid #e5e4dc;
  padding:12px 22px;display:flex;gap:14px;align-items:center}
h1{font-size:16px;margin:0}
.prov{color:#6e7168;font-size:12.5px;flex:1}
button{font:inherit;cursor:pointer;border:1px solid #e5e4dc;border-radius:6px;
  background:#fff;padding:6px 12px}
button.primary{background:#0e6b60;border-color:#0e6b60;color:#fff;font-weight:600}
main{max-width:1100px;margin:0 auto;padding:18px 22px 70px}
h2{font-size:14px;color:#6e7168;text-transform:uppercase;letter-spacing:.05em;
  margin:26px 0 8px}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:#fff;
  border:1px solid #e5e4dc;border-radius:8px}
th{color:#6e7168;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  text-align:left;padding:8px 10px;border-bottom:1px solid #e5e4dc}
td{padding:7px 10px;border-bottom:1px solid #e5e4dc;vertical-align:middle}
tr:last-child td{border-bottom:none}
td.mono{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
tr.flagged td{background:#fbf1e7}
audio{height:30px;width:230px;vertical-align:middle}
</style>
<header><h1>Intake pronunciation audition</h1><span class="prov">PROVENANCE</span>
<span id="n"></span><button class="primary" id="report">Copy flags</button></header>
<main>
"""
_LISTEN_HTML_TAIL = """</main>
<script>
const LS = "intake_audition_flags_MODE";
let flags = {};
try { flags = JSON.parse(localStorage.getItem(LS) || "{}"); } catch(e) {}
const save = () => localStorage.setItem(LS, JSON.stringify(flags));
// Count and report only rows on THIS page: stored flags for labels absent
// from the current render (e.g. after a --limit run) stay in storage but
// never inflate the count or the copied report.
const present = new Set(
  Array.from(document.querySelectorAll("tr[data-k]")).map(tr => tr.dataset.k));
function paint(){
  document.querySelectorAll("tr[data-k]").forEach(tr => {
    const k = tr.dataset.k;
    tr.classList.toggle("flagged", !!flags[k]);
    tr.querySelector("button").textContent = flags[k] ? "Flagged" : "Flag";
  });
  document.getElementById("n").textContent =
    Object.keys(flags).filter(k => present.has(k)).length + " flagged";
}
document.querySelectorAll("tr[data-k] button").forEach(b => {
  b.onclick = () => {
    const k = b.closest("tr").dataset.k;
    if (flags[k]) delete flags[k]; else flags[k] = true;
    save(); paint();
  };
});
document.getElementById("report").onclick = async () => {
  const lines = Object.keys(flags).filter(k => present.has(k)).sort()
    .map(k => "- " + k);
  const text = lines.length
    ? "Audition flags (" + lines.length + "):\\n" + lines.join("\\n")
    : "Audition listen: no flags.";
  try { await navigator.clipboard.writeText(text); alert("Copied."); }
  catch(e) { prompt("Copy:", text); }
};
paint();
</script>
"""


def _render_listen_html(items: List[AuditionItem]) -> str:
    from html import escape

    provenance = (
        f"voice {AUDITION_PERSONA} · TTS seed {AUDITION_TTS_SEED} · "
        "gold (default reading) vs mispronounced pairs · riskiest-first"
    )
    parts: list[str] = [_LISTEN_HTML_HEAD.replace("PROVENANCE", escape(provenance))]
    group = None
    for item in items:
        if item.group != group:
            if group is not None:
                parts.append("</table>")
            group = item.group
            parts.append(f"<h2>{escape(group)}</h2>")
            headers = "".join(
                f"<th>{escape(h)}</th>"
                for h in ["value"] + [cut.kind for cut in item.cuts] + [""]
            )
            parts.append(f"<table><tr>{headers}</tr>")
        cells = [f"<td>{escape(item.label)}</td>"]
        for cut in item.cuts:
            cells.append(
                f'<td class="mono">{escape(cut.text)}<br>'
                f'<audio controls preload="none" src="{escape(cut.filename)}"></audio></td>'
            )
        cells.append("<td><button>Flag</button></td>")
        parts.append(f'<tr data-k="{escape(item.label)}">{"".join(cells)}</tr>')
    if group is not None:
        parts.append("</table>")
    parts.append(_LISTEN_HTML_TAIL.replace("MODE", "banks"))
    return "\n".join(parts)


def run_audition(out_dir: Path, limit: Optional[int] = None) -> AuditionOutcome:
    """Synthesize the audition wavs into ``out_dir``; write ``index.md`` and
    the ``index.html`` listen page.

    Resumable: a wav that already exists is skipped, so an interrupted run
    fills its gaps on re-invocation. ``limit`` caps the number of ITEMS (in
    plan order) — a smoke of the verb, never a different draw.
    """
    from tau2.data_model.voice import ElevenLabsTTSConfig
    from tau2.data_model.voice_personas import get_elevenlabs_voice_id
    from tau2.voice.synthesis.synthesize import synthesize_voice
    from tau2.voice.utils.audio_io import save_wav_file

    items = build_audition_plan()
    if limit is not None:
        items = items[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    provider_config = ElevenLabsTTSConfig(
        voice_id=get_elevenlabs_voice_id(AUDITION_PERSONA),
        language_code="en",
        insert_audio_tags=False,
        seed=AUDITION_TTS_SEED,
    )
    synthesized = 0
    skipped = 0
    for item in items:
        for cut in item.cuts:
            path = out_dir / cut.filename
            if path.exists():
                skipped += 1
                continue
            audio = synthesize_voice(
                text=cut.text,
                provider="elevenlabs",
                provider_config=provider_config,
            )
            save_wav_file(audio, path)
            synthesized += 1
            logger.info(f"Audition wav: {path} ({cut.kind}: {cut.text!r})")

    index_path = out_dir / "index.md"
    index_path.write_text(_render_index(items))
    (out_dir / "index.html").write_text(_render_listen_html(items))
    return AuditionOutcome(
        out_dir=out_dir,
        items=len(items),
        synthesized=synthesized,
        skipped=skipped,
        index_path=index_path,
    )

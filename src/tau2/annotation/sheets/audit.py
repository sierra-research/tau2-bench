# Copyright Sierra
"""Google Sheets batchUpdate bodies for the Nativeness Audit workbook.

Pure functions over ``AuditNuanceRow`` + ``AuditTabConfig`` emitting ready
batchUpdate bodies (formerly two one-off audit-builder scripts). Two rounds:

- **Round 2 (verify)**: each language tab carries the AI-suggested rows
  (Status = 'Auto — needs review'); annotators adjudicate them.
- **Round 1 (blind)**: a COPY of the seeded sheet is stripped back to 2 yellow
  example rows per tab so annotators form their own findings first; the
  Status column is deleted (every row is the annotator's own finding).

Inputs (``<iso>.json``) and outputs (``bodies/`` / ``bodies_blind/``) live with
the audit data under ``data/tau2/multilingual/_factory/nuance_audit/`` —
unchanged from the script era, so the publish workflow (Sheets MCP batchUpdate
per body file) keeps working.
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
from pydantic import BaseModel, Field

from tau2.annotation.artifacts import provenance_stamp, write_json_artifact
from tau2.annotation.models import AuditNuanceRow, AuditStatus
from tau2.multilingual.factory.state import factory_root_dir


class AuditTabConfig(BaseModel):
    """One language tab of the shared Nativeness Audit spreadsheet."""

    iso: Annotated[str, Field(description="ISO 639-1 language code.")]
    sheet_id: Annotated[int, Field(description="Google Sheets tab (sheet) id.")]
    tab_name: Annotated[str, Field(description="English tab name, e.g. 'Spanish'.")]
    conditional_categories: Annotated[
        list[str],
        Field(default_factory=list, description="Language-specific categories."),
    ]


# THE tab registry (sheet ids are stable across copies of the workbook).
AUDIT_TABS: dict[str, AuditTabConfig] = {
    cfg.iso: cfg
    for cfg in [
        AuditTabConfig(
            iso="es",
            sheet_id=2088029326,
            tab_name="Spanish",
            conditional_categories=[
                "Formal vs informal (usted/tú)",
                "Grammatical gender agreement",
                "Regional variety (LatAm vs Spain)",
            ],
        ),
        AuditTabConfig(
            iso="hi",
            sheet_id=523938640,
            tab_name="Hindi",
            conditional_categories=[
                "Code-switching (Hinglish)",
                "Honorific/formality (aap/tum/tu)",
                "Grammatical gender agreement",
            ],
        ),
        AuditTabConfig(
            iso="ko",
            sheet_id=1354151399,
            tab_name="Korean",
            conditional_categories=[
                "Honorifics / speech levels (존댓말)",
                "Counters / number systems",
                "Names (order, titles)",
            ],
        ),
        AuditTabConfig(
            iso="pt",
            sheet_id=1900167037,
            tab_name="Portuguese",
            conditional_categories=[
                "Brazilian vs European (pt-br)",
                "Formal vs informal (você/o senhor)",
                "Grammatical gender agreement",
            ],
        ),
        AuditTabConfig(
            iso="zh",
            sheet_id=923051615,
            tab_name="Mandarin",
            conditional_categories=[
                "Measure words (量词)",
                "二 vs 两 / numbers",
                "Tones (meaning/word choice)",
            ],
        ),
    ]
}

COMMON_CATEGORIES = [
    "Register & formality",
    "Domain terminology",
    "Idiom & word-choice",
    "Names",
    "Numbers (word-forms, gender, dates)",
    "Symbols & code-switching",
    "Entities (booking refs, codes)",
    "Disfluencies & fillers",
    "Turn-taking & backchannels",
    "Politeness / face",
]

SEVERITY_DISPLAY = {1: "1 – subtle", 2: "2 – clearly off", 3: "3 – breaks the illusion"}


class AuditColumn(BaseModel):
    """One column of the audit tab — THE single source the header row,
    positional row writes, pixel widths, dropdown-validation ranges, the
    Status-column delete, and the leaderboard formula letters all derive
    from (they can no longer drift apart)."""

    key: Annotated[str, Field(description="Stable column key (code-facing).")]
    header: Annotated[str, Field(description="Header label; {L} = tab name.")]
    width: Annotated[int, Field(description="Column width in pixels.")]


AUDIT_COLUMNS: list[AuditColumn] = [
    AuditColumn(key="marker", header="#", width=36),
    AuditColumn(key="status", header="Status", width=150),
    AuditColumn(key="category", header="Category", width=190),
    AuditColumn(key="nuance", header="Nuance (short title)", width=210),
    AuditColumn(key="ai_does", header="How the AI caller sounds non-native", width=290),
    AuditColumn(key="native_does", header="What a real {L} speaker does", width=290),
    AuditColumn(key="example", header="Example: caller said → should say", width=300),
    AuditColumn(
        key="conversation_ref",
        header="Conversation ref (task/sim + ~time)",
        width=170,
    ),
    AuditColumn(key="severity", header="Severity", width=120),
    AuditColumn(key="notes", header="Your notes", width=220),
]

HEADERS = [c.header for c in AUDIT_COLUMNS]
N_COLUMNS = len(AUDIT_COLUMNS)


def column_index(key: str) -> int:
    """The 0-based grid index of a column key."""
    for i, c in enumerate(AUDIT_COLUMNS):
        if c.key == key:
            return i
    raise KeyError(f"unknown audit column '{key}'")


def column_letter(key: str, *, status_deleted: bool = False) -> str:
    """The A1 column letter of a key — for leaderboard formulas.

    ``status_deleted=True`` gives the letter AFTER the blind round deletes
    the Status column (everything right of it shifts one left).
    """
    idx = column_index(key)
    if status_deleted:
        status = column_index("status")
        if idx == status:
            raise ValueError("the Status column does not exist once deleted")
        if idx > status:
            idx -= 1
    if idx >= 26:
        raise ValueError(f"column index {idx} needs multi-letter A1 notation")
    return chr(ord("A") + idx)


_HEADER_BG = {"red": 0.81, "green": 0.89, "blue": 0.95}
_YELLOW = {"red": 1.0, "green": 0.97, "blue": 0.80}
_WHITE = {"red": 1, "green": 1, "blue": 1}
_N_BLANK_ROWS = 12  # cleared below the data block (stale-content guard)


def nuance_audit_dir() -> Path:
    """Where the candidate JSONs and built bodies live (script-era location)."""
    return factory_root_dir() / "nuance_audit"


def _sval(s: str) -> dict:
    return {"userEnteredValue": {"stringValue": s}}


def _one_of(values: list[str]) -> dict:
    return {
        "condition": {
            "type": "ONE_OF_LIST",
            "values": [{"userEnteredValue": v} for v in values],
        },
        "showCustomUi": True,
        "strict": False,
    }


def _range(sheet_id: int, r0: int, r1: int, c0: int = 0, c1: int = N_COLUMNS) -> dict:
    return {
        "sheetId": sheet_id,
        "startRowIndex": r0,
        "endRowIndex": r1,
        "startColumnIndex": c0,
        "endColumnIndex": c1,
    }


def tab_categories(cfg: AuditTabConfig) -> list[str]:
    """The tab's category dropdown: common + conditional + tones/other, deduped."""
    cats: list[str] = []
    for c in (
        COMMON_CATEGORIES
        + cfg.conditional_categories
        + ["Tones (meaning/word choice)", "Other"]
    ):
        if c not in cats:
            cats.append(c)
    return cats


def build_tab_body(cfg: AuditTabConfig, rows: list[AuditNuanceRow]) -> dict:
    """The Round-2 batchUpdate body: 10-col schema + AI-suggested rows.

    Example/Conversation-ref columns stay blank (the annotator fills them
    after verifying); Status defaults to 'Auto — needs review'.
    """
    sid, name = cfg.sheet_id, cfg.tab_name
    blurb = (
        'Log every way the AI "customer" sounds non-native in %s — wording, register, '
        "terminology, gender/grammar, how emails/numbers/dates are said, code-switching, and "
        "tone that CHANGES MEANING. NOT emotional tone (polite/angry) or raw audio quality. "
        "👂 Listen to / read a few of your conversations first to calibrate. The rows below are "
        "AI-SUGGESTED: set Status to ✅ Confirmed / ❌ Not true / 🤔 Unsure and add an Example + "
        "Conversation ref when you spot one." % name
    )
    header_cells = [_sval(h.replace("{L}", name)) for h in HEADERS]

    def _cells(row: AuditNuanceRow) -> dict[str, str]:
        # Example/Conversation-ref/notes stay blank (the annotator fills them
        # after verifying); unmapped keys render empty via .get below.
        return {
            "status": row.status.display,
            "category": row.category or "Other",
            "nuance": row.nuance,
            "ai_does": row.ai_does,
            "native_does": row.native_does,
            "severity": SEVERITY_DISPLAY.get(row.severity, SEVERITY_DISPLAY[2]),
        }

    data_rows = [
        {"values": [_sval(_cells(row).get(c.key, "")) for c in AUDIT_COLUMNS]}
        for row in rows
    ]
    # Blank rows to clear any stale content below the data block.
    blank_rows = [
        {"values": [_sval("") for _ in range(N_COLUMNS)]} for _ in range(_N_BLANK_ROWS)
    ]

    reqs: list[dict] = []
    # Unmerge a superset of the title rows (all columns) so it fully contains
    # any existing merge regardless of its width; unmerging a no-merge range is
    # a no-op.
    reqs.append({"unmergeCells": {"range": _range(sid, 0, 3, 0, 26)}})
    for r in range(3):
        reqs.append(
            {"mergeCells": {"range": _range(sid, r, r + 1), "mergeType": "MERGE_ALL"}}
        )
    reqs.append(
        {
            "updateCells": {
                "start": {"sheetId": sid, "rowIndex": 1, "columnIndex": 0},
                "fields": "userEnteredValue",
                "rows": [{"values": [_sval(blurb)]}],
            }
        }
    )
    reqs.append(
        {
            "updateCells": {
                "start": {"sheetId": sid, "rowIndex": 3, "columnIndex": 0},
                "fields": "userEnteredValue",
                "rows": [{"values": header_cells}],
            }
        }
    )
    reqs.append(
        {
            "updateCells": {
                "start": {"sheetId": sid, "rowIndex": 4, "columnIndex": 0},
                "fields": "userEnteredValue",
                "rows": data_rows + blank_rows,
            }
        }
    )
    reqs.append(
        {
            "repeatCell": {
                "range": _range(sid, 3, 4),
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": _HEADER_BG,
                        "textFormat": {"bold": True},
                        "wrapStrategy": "WRAP",
                        "verticalAlignment": "MIDDLE",
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat,wrapStrategy,verticalAlignment)",
            }
        }
    )
    reqs.append(
        {
            "repeatCell": {
                "range": _range(sid, 4, 1000),
                "cell": {
                    "userEnteredFormat": {
                        "wrapStrategy": "WRAP",
                        "verticalAlignment": "TOP",
                        "backgroundColor": _WHITE,
                    }
                },
                "fields": "userEnteredFormat(wrapStrategy,verticalAlignment,backgroundColor)",
            }
        }
    )
    reqs.append(
        {
            "updateSheetProperties": {
                "properties": {"sheetId": sid, "gridProperties": {"frozenRowCount": 4}},
                "fields": "gridProperties.frozenRowCount",
            }
        }
    )
    for key, options in (
        ("status", AuditStatus.options()),
        ("category", tab_categories(cfg)),
        ("severity", list(SEVERITY_DISPLAY.values())),
    ):
        c = column_index(key)
        reqs.append(
            {
                "setDataValidation": {
                    "range": _range(sid, 4, 1000, c, c + 1),
                    "rule": _one_of(options),
                }
            }
        )
    # Clear example/ref validations.
    reqs.append(
        {
            "setDataValidation": {
                "range": _range(
                    sid,
                    4,
                    1000,
                    column_index("example"),
                    column_index("conversation_ref") + 1,
                )
            }
        }
    )
    for i, column in enumerate(AUDIT_COLUMNS):
        reqs.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sid,
                        "dimension": "COLUMNS",
                        "startIndex": i,
                        "endIndex": i + 1,
                    },
                    "properties": {"pixelSize": column.width},
                    "fields": "pixelSize",
                }
            }
        )
    return {"requests": reqs}


def build_leaderboard_body() -> dict:
    """Round-2 Start-here leaderboard: COUNTIF on the Status dropdown.

    GOTCHA — do NOT count contributions with COUNTA: the table template
    pre-fills the blank data rows with empty strings (""), which COUNTA counts
    as non-empty (every untouched tab then shows ~30). Round 2 counts a
    specific dropdown value, so COUNTIF never miscounts blanks.
    """
    rows = [
        {
            "values": [
                {
                    "userEnteredValue": {
                        "formulaValue": "=COUNTIF('%s'!%s5:%s1000,\"%s\")"
                        % (
                            cfg.tab_name,
                            column_letter("status"),
                            column_letter("status"),
                            AuditStatus.CONFIRMED.display,
                        )
                    }
                }
            ]
        }
        for cfg in sorted(AUDIT_TABS.values(), key=lambda c: c.tab_name)
    ]
    return {
        "requests": [
            {
                "updateCells": {
                    "start": {"sheetId": 0, "rowIndex": 7, "columnIndex": 2},
                    "fields": "userEnteredValue",
                    "rows": rows,
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Round 1 (blind): annotators form their own findings first
# ---------------------------------------------------------------------------


def build_blind_tab_body(cfg: AuditTabConfig) -> dict:
    """Strip a copied tab to 2 yellow example rows; delete the Status column."""
    sid, name = cfg.sheet_id, cfg.tab_name
    blurb = (
        "ROUND 1 — log YOUR OWN observations first, before seeing any AI suggestions. "
        "As a native %s speaker, listen to / read a few of your conversations, then add one "
        'row per nuance: how the AI "customer" does NOT sound native — word choice, register, '
        "terminology, gender/grammar, how numbers/dates/emails are said, code-switching, and "
        "tone that CHANGES MEANING (NOT emotional tone or raw audio quality). The 2 YELLOW rows "
        "are EXAMPLES showing the level of detail we want — add your own rows below. "
        "(Afterwards you will get a second sheet with AI-suggested nuances to compare "
        "against.)" % name
    )
    reqs = [
        {
            "updateCells": {
                "start": {"sheetId": sid, "rowIndex": 1, "columnIndex": 0},
                "fields": "userEnteredValue",
                "rows": [{"values": [_sval(blurb)]}],
            }
        },
        # Mark the two kept rows as examples: col A='ex', clear Status (col B);
        # leave C–J content.
        {
            "updateCells": {
                "start": {"sheetId": sid, "rowIndex": 4, "columnIndex": 0},
                "fields": "userEnteredValue",
                "rows": [
                    {"values": [_sval("ex"), _sval("")]},
                    {"values": [_sval("ex"), _sval("")]},
                ],
            }
        },
        # Clear every row below the 2 examples. Round-2 writes AI suggestions
        # starting at row 5 and a tab can hold many, so clear a generous range
        # — a short fixed clear left stale AI text visible (able to bias
        # Round-1 annotators) on any tab with more suggestion rows.
        {
            "updateCells": {
                "start": {"sheetId": sid, "rowIndex": 6, "columnIndex": 0},
                "fields": "userEnteredValue",
                "rows": [
                    {"values": [_sval("") for _ in range(N_COLUMNS)]}
                    for _ in range(994)
                ],
            }
        },
        {
            "repeatCell": {
                "range": _range(sid, 4, 6),
                "cell": {"userEnteredFormat": {"backgroundColor": _YELLOW}},
                "fields": "userEnteredFormat(backgroundColor)",
            }
        },
        {
            "repeatCell": {
                "range": _range(sid, 6, 1000),
                "cell": {"userEnteredFormat": {"backgroundColor": _WHITE}},
                "fields": "userEnteredFormat(backgroundColor)",
            }
        },
        # Delete the Status column — must be LAST so every request above
        # runs against the original Round-2 layout (Status still in place).
        # After this, every column right of Status shifts one left (which is
        # why the blind leaderboard uses column_letter(status_deleted=True)).
        # The 'ex' step above also wrote '' into the Status column; once it
        # is deleted that write is a harmless no-op.
        {
            "deleteDimension": {
                "range": {
                    "sheetId": sid,
                    "dimension": "COLUMNS",
                    "startIndex": column_index("status"),
                    "endIndex": column_index("status") + 1,
                }
            }
        },
    ]
    return {"requests": reqs}


def build_blind_start_body() -> dict:
    title = "tau2 · Nativeness Audit — ROUND 1 (your own findings first)"
    purpose = (
        'ROUND 1 — YOUR OWN findings first. We built AI "callers" (voice user-simulators) that '
        "role-play customers in each language. As a native speaker, log every way they do NOT "
        "sound native — the LANGUAGE layer (word choice, register, terminology, gender/grammar, "
        "how numbers/dates/emails are said, code-switching, meaning-bearing tone). Each language "
        "tab has 2 EXAMPLE rows (yellow) showing the detail we want; add your own rows. "
        "Do this BEFORE the AI-suggested version so your judgement is your own. Voice/audio "
        "realism is rated separately in your conversation packet."
    )
    return {
        "requests": [
            {
                "updateCells": {
                    "start": {"sheetId": 0, "rowIndex": row, "columnIndex": 0},
                    "fields": "userEnteredValue",
                    "rows": [{"values": [_sval(text)]}],
                }
            }
            for row, text in ((0, title), (1, purpose))
        ]
    }


def build_blind_leaderboard_body() -> dict:
    """Round-1 leaderboard: count free-text findings per tab.

    Must use a ``<>""`` test, NOT COUNTA — the template pre-fills blank data
    rows with empty strings that COUNTA would count. The counted column is
    Nuance AFTER the Status delete shifts it left; rows 7+ skip the 2 example
    rows.
    """
    nuance = column_letter("nuance", status_deleted=True)
    rows = [
        {
            "values": [
                {
                    "userEnteredValue": {
                        "formulaValue": "=SUMPRODUCT(--('%s'!%s7:%s1000<>\"\"))"
                        % (cfg.tab_name, nuance, nuance)
                    }
                }
            ]
        }
        for cfg in sorted(AUDIT_TABS.values(), key=lambda c: c.tab_name)
    ]
    return {
        "requests": [
            {
                "updateCells": {
                    "start": {"sheetId": 0, "rowIndex": 7, "columnIndex": 2},
                    "fields": "userEnteredValue",
                    "rows": rows,
                }
            }
        ]
    }


def build_audit_bodies(
    langs: list[str], *, blind: bool = False, audit_dir: Path | None = None
) -> list[Path]:
    """Build batchUpdate bodies for the given languages (+ leaderboard/manifest).

    Round 2 reads each language's ``<iso>.json`` candidates; the blind round
    needs no candidate input. Returns the written body paths.
    """
    from tau2.annotation.nuance_candidates import load_nuance_candidates

    audit_dir = Path(audit_dir) if audit_dir is not None else nuance_audit_dir()
    out_dir = audit_dir / ("bodies_blind" if blind else "bodies")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest: list[str] = []

    for iso in langs:
        cfg = AUDIT_TABS.get(iso)
        if cfg is None:
            raise ValueError(
                f"unknown audit language '{iso}' (known: {sorted(AUDIT_TABS)})"
            )
        if blind:
            body = build_blind_tab_body(cfg)
        else:
            candidates_path = audit_dir / f"{iso}.json"
            if not candidates_path.exists():
                raise FileNotFoundError(
                    f"no nuance candidates at {candidates_path} — run "
                    f"`tau2 annotate audit --generate --lang {iso}` first"
                )
            rows = [c.to_row() for c in load_nuance_candidates(candidates_path)]
            body = build_tab_body(cfg, rows)
            logger.info(f"{iso}: {len(rows)} nuances -> bodies/{iso}.json")
        path = write_json_artifact(out_dir / f"{iso}.json", body)
        written.append(path)
        manifest.append(path.name)

    if blind:
        extras = {
            "start.json": build_blind_start_body(),
            "leaderboard.json": build_blind_leaderboard_body(),
        }
    else:
        extras = {"leaderboard.json": build_leaderboard_body()}
    for name, body in extras.items():
        written.append(write_json_artifact(out_dir / name, body))
        manifest.append(name)
    write_json_artifact(
        out_dir / "manifest.json", {**provenance_stamp(), "files": manifest}
    )
    logger.info(f"wrote {len(written)} bodies + manifest -> {out_dir}")
    return written

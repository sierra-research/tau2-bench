# Copyright Sierra
"""Side-by-side localized-task review sheet: English source vs every language.

One row per source task, one column group per language — the scenario text
exactly as the user simulator will read it, then an approve/deny dropdown and
a notes cell for that language. A native reviewer scans down their language's
column and signs off task by task, with the English source in view.

What is actually under review here is NOT a translation. Every shipped domain
declares ``tasks_translated: false``, so the prose is English in every arm;
what differs per language is the **identity swap** — the caller's name,
address, email and zip, and every place the scenario prose names one of them.
So the reviewable defects are: an identity that does not read as a real person
in that locale, an internally inconsistent scenario (the prose naming a value
the caller's record does not hold), and a swap that ran somewhere it should
not have (a dictated destination, a third party).

The rows come from the SAME task sets a run would execute
(:func:`tau2.multilingual.run_presets.matrix_task_set_name`, so the identity
arm when the pack declares one) — a sheet built off the plain localized set
would review text no run ever sends. Rows are keyed by source task id, which
is what the ``<id>_<lang>[_identity]`` shape strips down to; a language whose
set is missing a source id is a LOUD error, not a blank cell.
"""

from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
from openpyxl import Workbook
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.annotation.artifacts import provenance_stamp, write_json_artifact
from tau2.annotation.models import LocalizationVerdict
from tau2.annotation.sheets.workbook import write_sheet

DEFAULT_LOCALIZATION_SHEET_DIR = Path("data/annotations/localization_review")

TASK_ID_LABEL = "task"
IN_SUBSET_LABEL = "in run set"
_TASK_WIDTH = 78
_VERDICT_WIDTH = 14
_NOTES_WIDTH = 40

# The reviewer-facing tab, authored here rather than mailed separately: a
# verdict column with no stated standard collects opinions, not findings.
_HOW_TO_REVIEW: list[tuple[str, str]] = [
    (
        "What this sheet is",
        "One row per task. The first text column is the English source the "
        "benchmark ships. Each following language column is that same task "
        "as YOUR language's arm will actually run it. Read your language's "
        "column; the English one is there for comparison.",
    ),
    (
        "Why the text is still in English",
        "It is meant to be. The caller SPEAKS your language on the call — "
        "that comes from the language pack, not from this text. This text is "
        "the private scenario the simulated caller reads before dialling, and "
        "it is kept in English on purpose so the instructions are unambiguous.",
    ),
    (
        "So what am I reviewing?",
        "The caller's identity: their name, street address, city, region, "
        "postcode and email. Those ARE localized, and they appear both in this "
        "scenario text and in the shop's records the agent will look up.",
    ),
    (
        "Deny if — the name is not a real name",
        "The first/last name is not a name people in your locale actually "
        "have, the two halves do not go together, or the name reads as a "
        "different country's.",
    ),
    (
        "Deny if — the address is not a real address",
        "The street line, city, region code or postcode is not how addresses "
        "are written where you live, or the pieces contradict each other (a "
        "postcode from one region on a city in another).",
    ),
    (
        "Deny if — the scenario contradicts itself",
        "The task asks the caller to do something with a value that is not "
        "the value they were given (e.g. asks to move an order to an address "
        "that is the one already on file, or names a city the caller does not "
        "live in as if they did).",
    ),
    (
        "Deny if — something was localized that should not have been",
        "A place the caller is shipping TO, or another person's details, is a "
        "destination in the task, not the caller's own identity, and should "
        "stay as the English source has it.",
    ),
    (
        "Do NOT deny for",
        "The English prose itself, order numbers, product names, item ids, or "
        "prices in dollars — those are the shared benchmark and are identical "
        "in every language on purpose.",
    ),
    (
        "Unsure",
        "Use 🤔 unsure and say why in the notes. An unexplained deny costs a "
        "whole extra round to interpret, so always leave a note with a deny.",
    ),
]


class LocalizationSheetOptions(BaseModel):
    """Inputs for one localized-task review workbook."""

    model_config = ConfigDict(extra="forbid")

    domain: Annotated[str, Field(description="Task domain, e.g. 'retail'.")]
    languages: Annotated[
        list[str],
        Field(
            description="Language codes, in column order. The first is the "
            "reference column (conventionally 'en')."
        ),
    ]
    out: Annotated[
        Path,
        Field(
            default=DEFAULT_LOCALIZATION_SHEET_DIR,
            description="Directory the workbook + manifest are written to.",
        ),
    ] = DEFAULT_LOCALIZATION_SHEET_DIR
    only_subset: Annotated[
        bool,
        Field(
            default=False,
            description="Restrict rows to the domain's canonical task subset "
            "(the fixed set a full run scores). Default: every task, with the "
            "subset ones marked.",
        ),
    ] = False
    publish_to: Annotated[
        Optional[Path],
        Field(
            default=None,
            description="Also copy the workbook here (a Drive-mount folder); "
            "the Sheets import API cannot ingest a local .xlsx with its "
            "dropdowns, so publishing is a file copy.",
        ),
    ] = None

    @field_validator("languages")
    @classmethod
    def _at_least_two_distinct(cls, langs: list[str]) -> list[str]:
        if len(langs) < 2:
            raise ValueError(
                "a side-by-side sheet needs at least two languages "
                "(the reference plus one under review)"
            )
        if len(set(langs)) != len(langs):
            raise ValueError(f"languages repeat: {langs}")
        return langs


def scenario_text(task) -> str:
    """The scenario block as the user simulator receives it.

    The WHOLE ``user_scenario`` (persona + instructions), rendered through the
    model's own ``__str__`` — the same string the runtime passes to the user
    simulator (``runner.build``) — so the sheet can never show a prettier,
    staler, or narrower scenario than the one that runs. Retail personas are
    all None today, but any domain that carries them (74 of telecom's 114
    tasks do) must show them: the identity rename lands inside personas too.
    """
    return str(task.user_scenario)


def source_task_id(task_id: str, language: str) -> str:
    """The canonical (unlocalized) id behind ``<id>_<lang>[_identity]``."""
    return task_id.removesuffix("_identity").removesuffix(f"_{language}")


def _tasks_by_source_id(domain: str, language: str) -> tuple[str, dict[str, object]]:
    """(task-set name, {source id -> task}) for the set this language runs.

    Each task goes through :func:`english_user_task_variant` — the SAME seam a
    run's ``user_prompt_task`` builds the user simulator from — rather than
    being read raw off the registry loader. For today's shipped domains
    (``tasks_translated: false``) the two are identical, but the seam is what
    makes the header claim ("as YOUR language's arm will actually run it")
    true by construction: a translated domain, or a stored prose field the
    rename round-trip cannot reproduce, surfaces here as the seam's own loud
    consistency failure instead of a silently wrong sheet.
    """
    from tau2.multilingual.english_prompts import english_user_task_variant
    from tau2.multilingual.run_presets import matrix_task_set_name
    from tau2.registry import registry

    task_set_name = matrix_task_set_name(domain, language)
    tasks = registry.get_tasks_loader(task_set_name)()
    by_id: dict[str, object] = {}
    for task in tasks:
        key = source_task_id(str(task.id), language)
        if key in by_id:
            raise ValueError(
                f"task set '{task_set_name}' has two tasks with source id "
                f"'{key}' — ids must strip to a unique canonical id"
            )
        by_id[key] = english_user_task_variant(
            task, domain=domain, task_set_name=task_set_name
        )
    return task_set_name, by_id


def _language_labels(languages: list[str]) -> dict[str, str]:
    """{code -> column prefix}, e.g. 'hi' -> 'Hindi (hi)'.

    Falls back to the bare code for a language with no registered pack — the
    English baseline arm has no pack and must still get a column.
    """
    from tau2.multilingual.registry import get_language_pack

    labels: dict[str, str] = {}
    for lang in languages:
        pack = get_language_pack(lang)
        display = getattr(pack, "display_name", None) if pack else None
        labels[lang] = f"{display} ({lang})" if display else lang
    return labels


def _subset_ids(domain: str) -> tuple[Optional[str], set[str]]:
    """(subset name, ids) for the domain's canonical subset; ('', set()) if none."""
    from tau2.task_subsets.store import resolve_subset

    subset = resolve_subset(domain=domain, requested="auto")
    if subset is None:
        return None, set()
    return subset.name, set(subset.task_ids)


def _headers(languages: list[str], labels: dict[str, str]) -> list[str]:
    headers = [TASK_ID_LABEL, IN_SUBSET_LABEL]
    for lang in languages:
        headers.append(labels[lang])
        headers.append(f"{labels[lang]} — approve?")
        headers.append(f"{labels[lang]} — notes")
    return headers


def build_localization_sheet(opts: LocalizationSheetOptions) -> tuple[Path, Path]:
    """Build the workbook + its provenance manifest; returns both paths."""
    labels = _language_labels(opts.languages)
    subset_name, subset_ids = _subset_ids(opts.domain)

    sets: dict[str, tuple[str, dict[str, object]]] = {
        lang: _tasks_by_source_id(opts.domain, lang) for lang in opts.languages
    }
    reference = opts.languages[0]
    row_ids = list(sets[reference][1])
    for lang in opts.languages[1:]:
        missing = [i for i in row_ids if i not in sets[lang][1]]
        if missing:
            raise ValueError(
                f"'{lang}' task set '{sets[lang][0]}' is missing "
                f"{len(missing)} task(s) present in '{sets[reference][0]}': "
                f"{missing[:5]} — regenerate that language's arm before "
                "building a side-by-side sheet"
            )
    if opts.only_subset:
        if not subset_ids:
            raise ValueError(
                f"--only-subset: domain '{opts.domain}' declares no canonical "
                "task subset, so there is nothing to restrict to"
            )
        row_ids = [i for i in row_ids if i in subset_ids]

    rows: list[dict[str, str]] = []
    for source_id in row_ids:
        row = {
            TASK_ID_LABEL: source_id,
            IN_SUBSET_LABEL: "✓" if source_id in subset_ids else "",
        }
        for lang in opts.languages:
            row[labels[lang]] = scenario_text(sets[lang][1][source_id])
        rows.append(row)

    headers = _headers(opts.languages, labels)
    verdict_cols = [h for h in headers if h.endswith("— approve?")]
    widths = {TASK_ID_LABEL: 10, IN_SUBSET_LABEL: 9}
    for lang in opts.languages:
        widths[labels[lang]] = _TASK_WIDTH
        widths[f"{labels[lang]} — approve?"] = _VERDICT_WIDTH
        widths[f"{labels[lang]} — notes"] = _NOTES_WIDTH

    wb = Workbook()
    grid = wb.active
    grid.title = "Tasks"
    write_sheet(
        grid,
        headers,
        rows,
        dropdowns={col: LocalizationVerdict.options() for col in verdict_cols},
        hidden=set(),
        widths=widths,
        # Task id + the run-set marker stay on screen while scrolling right
        # across the language columns; without this the reviewer loses track
        # of which task a far-right cell belongs to.
        freeze="C2",
    )

    guide = wb.create_sheet("How to review")
    guide_headers = ["", "What to do"]
    write_sheet(
        guide,
        guide_headers,
        [{"": head, "What to do": body} for head, body in _HOW_TO_REVIEW],
        dropdowns={},
        hidden=set(),
        widths={"": 46, "What to do": 92},
    )

    opts.out.mkdir(parents=True, exist_ok=True)
    stem = f"{opts.domain}_localization_review"
    xlsx = opts.out / f"{stem}.xlsx"
    wb.save(xlsx)

    manifest = write_json_artifact(
        opts.out / f"{stem}_manifest.json",
        {
            **provenance_stamp(),
            "kind": "localization_review",
            "domain": opts.domain,
            "languages": opts.languages,
            "reference_language": reference,
            "task_sets": {lang: sets[lang][0] for lang in opts.languages},
            "canonical_subset": subset_name,
            "only_subset": opts.only_subset,
            "rows": len(rows),
            "rows_in_subset": sum(1 for i in row_ids if i in subset_ids),
            "task_ids": row_ids,
        },
    )
    logger.info(f"wrote {len(rows)} tasks x {len(opts.languages)} languages -> {xlsx}")

    if opts.publish_to is not None:
        import shutil

        opts.publish_to.mkdir(parents=True, exist_ok=True)
        dest = opts.publish_to / xlsx.name
        shutil.copy2(xlsx, dest)
        logger.info(f"published -> {dest}")
    return xlsx, manifest

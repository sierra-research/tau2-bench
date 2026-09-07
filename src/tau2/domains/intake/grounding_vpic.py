# Copyright Sierra
"""Positive grounding of the vehicles bank against NHTSA vPIC.

``tau2 intake-grounding vpic`` validates every vehicles-bank value AND decoy
(make + model) against the NHTSA vPIC vehicle registry
(https://vpic.nhtsa.dot.gov/api/), and writes a provenance-bearing report
(:class:`VpicGroundingReport`) to
``data/tau2/domains/intake/grounding_vpic.json``. It never edits a bank:
findings are surfaced for the owner to act on.

Verdict per entry (design doc ``docs/designs/intake-grounding.md``, roadmap
item 3):

- **PASS** — the make is a vPIC make and a vPIC model for that make matches
  the model portion (exactly, or as a leading word-boundary prefix: vPIC
  lists base models without trim, so ``228i`` grounds
  ``228i xDrive Gran Coupe``).
- **FINDING** — the make is unknown to vPIC or no listed model matches.
  vPIC covers US-market vehicles: European makes sold in the US appear,
  EU-only makes (Skoda, Vauxhall, Cupra, Seat, DS) do not — a FINDING is a
  fact for the owner, not an auto-failure.
- **UNCHECKED** — a lookup failed; never a silent PASS.

Deterministic normalization (documented here, tested in
``test_grounding_positive_intake.py``): names are NFD-decomposed with
combining marks stripped (so a bank ``Skoda`` would match a vPIC ``ŠKODA``),
casefolded, and whitespace-collapsed. The make is the LONGEST word-boundary
prefix of the value that is a vPIC make (so ``Alfa Romeo`` beats ``Alfa``);
the model match prefers the LONGEST listed model (so ``Corolla Cross`` beats
``Corolla`` for ``Toyota Corolla Cross``).

Network access happens only here, at build time — one ``getallmakes``
download (sha256 recorded) plus one cached ``GetModelsForMake`` request per
unique make, each with its request URL recorded in the report.
"""

import hashlib
import time
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, List, Optional
from urllib.parse import quote

import requests
from loguru import logger
from pydantic import Field

from tau2.domains.intake.grounding_screen import SourceRecord, Verdict
from tau2.domains.intake.tasks.banks import Banks, load_banks
from tau2.domains.intake.utils import INTAKE_DATA_DIR
from tau2.utils.pydantic_utils import BaseModelNoExtra

# Report schema version: bump on any change to the report models or to what
# the screen checks, so two reports are comparable only when versions match.
VPIC_SCREEN_VERSION = "1.0.0"

DEFAULT_VPIC_REPORT_PATH = INTAKE_DATA_DIR / "grounding_vpic.json"

VPIC_ALL_MAKES_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/getallmakes?format=json"
VPIC_MODELS_FOR_MAKE_URL = (
    "https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMake/{make}?format=json"
)

# vPIC is a free public API; lookups sleep between requests (politeness lives
# at the network seam — mocked tests never pay it).
_HTTP_USER_AGENT = "tau2-bench (Sierra AI, https://sierra.ai)"
_HTTP_TIMEOUT_S = 30
VPIC_REQUEST_INTERVAL_S = 0.5


def fold_name(text: str) -> str:
    """Deterministic name normalization: strip combining marks (NFD),
    casefold, collapse whitespace."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())


# --- Network seam (monkeypatched by tests) ----------------------------------


class VpicMakesCatalog(BaseModelNoExtra):
    """The vPIC all-makes catalog, downloaded once per screen."""

    makes: Annotated[List[str], Field(description="All make names, as published.")]
    sha256: Annotated[str, Field(description="sha256 of the downloaded bytes.")]


def fetch_vpic_makes() -> VpicMakesCatalog:
    """Download the vPIC ``getallmakes`` catalog and extract make names."""
    response = requests.get(
        VPIC_ALL_MAKES_URL,
        headers={"User-Agent": _HTTP_USER_AGENT},
        timeout=_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    raw = response.content
    payload = response.json()
    makes = [str(row["Make_Name"]) for row in payload["Results"]]
    if not makes:
        raise ValueError("vPIC getallmakes parsed to zero makes")
    return VpicMakesCatalog(makes=makes, sha256=hashlib.sha256(raw).hexdigest())


def models_for_make_url(make: str) -> str:
    """The exact ``GetModelsForMake`` request URL for one make."""
    return VPIC_MODELS_FOR_MAKE_URL.format(make=quote(make, safe=""))


def fetch_vpic_models(make: str) -> List[str]:
    """Fetch the vPIC model list for one make.

    Sleeps :data:`VPIC_REQUEST_INTERVAL_S` before the request (polite rate
    limiting). Raises on transport/protocol failure — the caller records
    UNCHECKED, never PASS.
    """
    time.sleep(VPIC_REQUEST_INTERVAL_S)
    response = requests.get(
        models_for_make_url(make),
        headers={"User-Agent": _HTTP_USER_AGENT},
        timeout=_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    return [str(row["Model_Name"]) for row in payload["Results"]]


# --- Deterministic make/model matching ---------------------------------------


def split_make(value: str, makes_by_fold: dict[str, str]) -> Optional[tuple[str, str]]:
    """Split a bank value into (vPIC make, model remainder).

    The make is the longest word-boundary prefix of the value that is a vPIC
    make after :func:`fold_name` normalization. Returns None when no prefix
    is a known make.
    """
    words = value.split()
    for length in range(len(words), 0, -1):
        candidate = fold_name(" ".join(words[:length]))
        make = makes_by_fold.get(candidate)
        if make is not None:
            return make, " ".join(words[length:])
    return None


def match_model(remainder: str, models: List[str]) -> Optional[tuple[str, str]]:
    """Match the model remainder against a make's vPIC model list.

    Returns (vPIC model as listed, match kind) where kind is ``exact`` when
    the folded remainder equals the folded model and ``prefix`` when the
    model is a leading word-boundary prefix of the remainder (vPIC lists
    base models without trim). Prefers the longest matching model; ties
    break lexicographically for determinism. Returns None when nothing
    matches.
    """
    folded_remainder = fold_name(remainder)
    best: Optional[tuple[str, str]] = None
    best_key: tuple[int, str] = (-1, "")
    for model in models:
        folded = fold_name(model)
        if not folded:
            continue
        if folded == folded_remainder:
            kind = "exact"
        elif folded_remainder.startswith(folded + " "):
            kind = "prefix"
        else:
            continue
        key = (len(folded), folded)
        if best is None or key > best_key:
            best, best_key = (model, kind), key
    return best


# --- Report models -----------------------------------------------------------


class MakeModelsRecord(BaseModelNoExtra):
    """One cached ``GetModelsForMake`` request the screen made."""

    make: Annotated[str, Field(description="The vPIC make queried, as listed.")]
    url: Annotated[str, Field(description="The exact request URL.")]
    model_count: Annotated[
        Optional[int],
        Field(default=None, description="Models returned; None on failure."),
    ]
    error: Annotated[
        Optional[str],
        Field(default=None, description="Transport/protocol error, if any."),
    ]


class VehicleCheck(BaseModelNoExtra):
    """vPIC verdict for one vehicles-bank value or decoy."""

    value: Annotated[str, Field(description="The exact bank string checked.")]
    role: Annotated[str, Field(description="'value' or 'decoy'.")]
    verdict: Annotated[
        Verdict,
        Field(description="pass = make known AND a listed model matches."),
    ]
    make: Annotated[
        Optional[str],
        Field(default=None, description="The vPIC make matched, as listed."),
    ]
    matched_model: Annotated[
        Optional[str],
        Field(default=None, description="The vPIC model matched, as listed."),
    ]
    match_kind: Annotated[
        Optional[str],
        Field(
            default=None,
            description="'exact' or 'prefix' (vPIC lists base models sans trim).",
        ),
    ]
    note: Annotated[
        Optional[str],
        Field(default=None, description="Why the check is a finding/unchecked."),
    ]


class VpicGroundingReport(BaseModelNoExtra):
    """The provenance-bearing output of one vehicles positive screen."""

    screen_date: Annotated[
        str, Field(description="UTC timestamp the screen ran (ISO 8601).")
    ]
    tool_version: Annotated[
        str, Field(description="Installed tau2 package version that ran the screen.")
    ]
    screen_version: Annotated[
        str, Field(description="Report schema / screen-logic version.")
    ]
    bank_shas: Annotated[
        dict[str, str],
        Field(description="sha256 per screened bank file (from the bank loader)."),
    ]
    sources: Annotated[
        List[SourceRecord], Field(description="External sources consulted, with URLs.")
    ]
    model_requests: Annotated[
        List[MakeModelsRecord],
        Field(description="Every GetModelsForMake request URL and outcome."),
    ]
    checks: Annotated[
        List[VehicleCheck], Field(description="Per-value/decoy vPIC verdicts.")
    ]
    findings: Annotated[int, Field(description="Total FINDING verdicts.")]
    unchecked: Annotated[int, Field(description="Total UNCHECKED verdicts.")]
    passes: Annotated[int, Field(description="Total PASS verdicts.")]


# --- The screen ----------------------------------------------------------------


def vehicles_to_check(banks: Banks) -> List[tuple[str, str]]:
    """(role, value) pairs to check, unique and sorted."""
    pairs: set[tuple[str, str]] = set()
    for entry in banks.vehicles:
        pairs.add(("value", entry.value))
        for decoy in entry.decoys:
            pairs.add(("decoy", decoy))
    return sorted(pairs)


def run_vpic_screen(banks: Optional[Banks] = None) -> VpicGroundingReport:
    """Screen every vehicles-bank value and decoy against vPIC."""
    banks = banks or load_banks()
    pairs = vehicles_to_check(banks)

    sources: List[SourceRecord] = []
    makes_by_fold: Optional[dict[str, str]] = None
    try:
        catalog = fetch_vpic_makes()
    except Exception as exc:  # noqa: BLE001 - every check becomes UNCHECKED
        logger.warning(f"vPIC getallmakes fetch failed; all checks UNCHECKED: {exc}")
        sources.append(
            SourceRecord(
                name="NHTSA vPIC getallmakes",
                url=VPIC_ALL_MAKES_URL,
                note=f"fetch failed, all checks UNCHECKED: {exc}",
            )
        )
    else:
        # First-listed make wins a fold collision, deterministically.
        makes_by_fold = {}
        for make in catalog.makes:
            makes_by_fold.setdefault(fold_name(make), make)
        sources.append(
            SourceRecord(
                name="NHTSA vPIC getallmakes",
                url=VPIC_ALL_MAKES_URL,
                sha256=catalog.sha256,
                note=f"{len(catalog.makes)} makes",
            )
        )
    sources.append(
        SourceRecord(
            name="NHTSA vPIC GetModelsForMake",
            url=VPIC_MODELS_FOR_MAKE_URL,
            note="one cached request per unique make; per-request URLs in "
            "model_requests",
        )
    )

    model_lists: dict[str, Optional[List[str]]] = {}
    model_requests: List[MakeModelsRecord] = []

    def models_for(make: str) -> Optional[List[str]]:
        if make in model_lists:
            return model_lists[make]
        record = MakeModelsRecord(make=make, url=models_for_make_url(make))
        try:
            models = fetch_vpic_models(make)
        except Exception as exc:  # noqa: BLE001 - recorded, never masked as PASS
            record.error = str(exc)
            model_lists[make] = None
        else:
            record.model_count = len(models)
            model_lists[make] = models
        model_requests.append(record)
        return model_lists[make]

    checks: List[VehicleCheck] = []
    for role, value in pairs:
        if makes_by_fold is None:
            checks.append(
                VehicleCheck(
                    value=value,
                    role=role,
                    verdict=Verdict.UNCHECKED,
                    note="vPIC make catalog unavailable",
                )
            )
            continue
        split = split_make(value, makes_by_fold)
        if split is None:
            checks.append(
                VehicleCheck(
                    value=value,
                    role=role,
                    verdict=Verdict.FINDING,
                    note="no leading word-boundary prefix is a vPIC make "
                    "(vPIC covers US-market vehicles)",
                )
            )
            continue
        make, remainder = split
        models = models_for(make)
        if models is None:
            checks.append(
                VehicleCheck(
                    value=value,
                    role=role,
                    verdict=Verdict.UNCHECKED,
                    make=make,
                    note="GetModelsForMake lookup failed",
                )
            )
            continue
        matched = match_model(remainder, models) if remainder else None
        if matched is None:
            checks.append(
                VehicleCheck(
                    value=value,
                    role=role,
                    verdict=Verdict.FINDING,
                    make=make,
                    note=f"model {remainder!r} not listed for make {make!r} "
                    "(vPIC covers US-market vehicles)",
                )
            )
            continue
        model, kind = matched
        checks.append(
            VehicleCheck(
                value=value,
                role=role,
                verdict=Verdict.PASS,
                make=make,
                matched_model=model,
                match_kind=kind,
            )
        )

    verdicts = [check.verdict for check in checks]
    return VpicGroundingReport(
        screen_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tool_version=package_version("tau2"),
        screen_version=VPIC_SCREEN_VERSION,
        bank_shas={"vehicles": banks.shas["vehicles"]},
        sources=sources,
        model_requests=sorted(model_requests, key=lambda r: r.make),
        checks=checks,
        findings=sum(v is Verdict.FINDING for v in verdicts),
        unchecked=sum(v is Verdict.UNCHECKED for v in verdicts),
        passes=sum(v is Verdict.PASS for v in verdicts),
    )


def write_vpic_report(
    report: VpicGroundingReport, out_path: Optional[Path] = None
) -> Path:
    """Write the report JSON (indented, trailing newline) and return the path."""
    out_path = out_path or DEFAULT_VPIC_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2) + "\n")
    return out_path

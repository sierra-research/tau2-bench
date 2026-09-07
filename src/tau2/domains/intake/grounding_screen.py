# Copyright Sierra
"""Negative reality screening for the intake entity banks.

``tau2 intake-grounding screen`` checks that the banks' *fictional* content
stays fictional in the real world, and writes a provenance-bearing report
(:class:`GroundingScreenReport`) to
``data/tau2/domains/intake/grounding_screen.json``. It never edits a bank:
findings are surfaced for the owner to act on.

Three screens run (design doc ``docs/designs/intake-grounding.md``):

- **Email domains** — every ``domain`` and ``decoy_domain`` in the emails
  bank must not resolve. A/AAAA/MX are looked up through the Google Public
  DNS DoH JSON API; a domain with any answer record is a FINDING (someone
  owns it), NXDOMAIN across all types is a PASS, and a transport failure is
  UNCHECKED.
- **Registry names** — coined company names, insurance-plan carriers,
  property names, and shop names (values and decoys) are checked, exact and
  case-folded, against (a) the SEC EDGAR ``company_tickers.json`` registrant
  catalog (one small pinned download, sha256 recorded) and (b) the GLEIF LEI
  fulltext search API (one polite request per unique name). A name that
  matches a real registered entity is a FINDING; a failed lookup is
  UNCHECKED, never a silent PASS.

Network access happens only here, at build time — never at bank load. Bank
load runs the pure, offline range validation in
:mod:`tau2.domains.intake.grounding`.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, List, Optional

import requests
from loguru import logger
from pydantic import Field

from tau2.domains.intake.tasks.banks import Banks, load_banks
from tau2.domains.intake.utils import INTAKE_DATA_DIR
from tau2.utils.pydantic_utils import BaseModelNoExtra

# Report schema version: bump on any change to the report models or to what
# a screen checks, so two reports are comparable only when versions match.
SCREEN_VERSION = "1.0.0"

DEFAULT_SCREEN_REPORT_PATH = INTAKE_DATA_DIR / "grounding_screen.json"

# --- The three external sources ---------------------------------------------

DOH_ENDPOINT = "https://dns.google/resolve"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
GLEIF_FULLTEXT_URL = "https://api.gleif.org/api/v1/lei-records"

# SEC asks automated clients to declare who they are; GLEIF's public API is
# rate limited, so lookups sleep between requests (politeness lives at the
# network seam — mocked tests never pay it).
_HTTP_USER_AGENT = "tau2-bench (Sierra AI, https://sierra.ai)"
_HTTP_TIMEOUT_S = 20
GLEIF_REQUEST_INTERVAL_S = 0.6
GLEIF_PAGE_SIZE = 20

_DNS_RRTYPES = ("A", "AAAA", "MX")
_DNS_RCODE_NXDOMAIN = 3


class Verdict(str, Enum):
    """Outcome of one reality check."""

    PASS = "pass"  # checked, nothing real matched
    FINDING = "finding"  # the fictional value collides with reality
    UNCHECKED = "unchecked"  # the check could not run; NOT a pass


# --- Network seam (monkeypatched by tests) ----------------------------------


class DohLookup(BaseModelNoExtra):
    """One DoH lookup result for a (domain, rrtype) pair."""

    rrtype: Annotated[str, Field(description="Record type looked up (A/AAAA/MX).")]
    status: Annotated[
        Optional[int],
        Field(description="DNS rcode (0=NOERROR, 3=NXDOMAIN); None on failure."),
    ]
    answers: Annotated[
        List[str],
        Field(default_factory=list, description="Answer record data, if any."),
    ]
    error: Annotated[
        Optional[str],
        Field(
            default=None, description="Transport/protocol error, if the lookup failed."
        ),
    ]


def doh_resolve(domain: str, rrtype: str) -> DohLookup:
    """Look up one record type for a domain via the Google DoH JSON API."""
    try:
        response = requests.get(
            DOH_ENDPOINT,
            params={"name": domain, "type": rrtype},
            headers={"User-Agent": _HTTP_USER_AGENT},
            timeout=_HTTP_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - mapped to UNCHECKED, never PASS
        return DohLookup(rrtype=rrtype, status=None, error=str(exc))
    answers = [
        str(record.get("data", ""))
        for record in payload.get("Answer", [])
        # DoH follows CNAMEs; only same-type answers prove the type resolves,
        # but any answer at all means the name is alive — keep them all.
    ]
    return DohLookup(rrtype=rrtype, status=payload.get("Status"), answers=answers)


class EdgarCatalog(BaseModelNoExtra):
    """The SEC EDGAR registrant catalog, downloaded once per screen."""

    titles: Annotated[
        List[str], Field(description="All registrant titles, as published.")
    ]
    sha256: Annotated[str, Field(description="sha256 of the downloaded bytes.")]


def fetch_edgar_catalog() -> EdgarCatalog:
    """Download ``company_tickers.json`` and extract registrant titles."""
    response = requests.get(
        EDGAR_TICKERS_URL,
        headers={"User-Agent": _HTTP_USER_AGENT},
        timeout=_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    raw = response.content
    payload = json.loads(raw)
    titles = [str(row["title"]) for row in payload.values()]
    if not titles:
        raise ValueError("EDGAR company_tickers.json parsed to zero titles")
    return EdgarCatalog(titles=titles, sha256=hashlib.sha256(raw).hexdigest())


def gleif_legal_names(name: str) -> List[str]:
    """Fulltext-search the GLEIF LEI API for ``name``; return legal names.

    Sleeps :data:`GLEIF_REQUEST_INTERVAL_S` before the request (polite rate
    limiting). Raises on transport/protocol failure — the caller records
    UNCHECKED, never PASS.
    """
    time.sleep(GLEIF_REQUEST_INTERVAL_S)
    response = requests.get(
        GLEIF_FULLTEXT_URL,
        params={"filter[fulltext]": name, "page[size]": str(GLEIF_PAGE_SIZE)},
        headers={"User-Agent": _HTTP_USER_AGENT, "Accept": "application/vnd.api+json"},
        timeout=_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    names: List[str] = []
    for record in payload.get("data", []):
        entity = record.get("attributes", {}).get("entity", {})
        legal = entity.get("legalName", {}).get("name")
        if legal:
            names.append(str(legal))
        for other in entity.get("otherNames", []) or []:
            if other.get("name"):
                names.append(str(other["name"]))
    return names


# --- Report models -----------------------------------------------------------


class SourceRecord(BaseModelNoExtra):
    """One external source the screen consulted."""

    name: Annotated[str, Field(description="Human name of the source.")]
    url: Annotated[str, Field(description="Endpoint or file URL consulted.")]
    sha256: Annotated[
        Optional[str],
        Field(default=None, description="sha256 of a pinned download, if any."),
    ]
    note: Annotated[
        Optional[str],
        Field(default=None, description="Fetch failure or other caveat."),
    ]


class DomainDnsCheck(BaseModelNoExtra):
    """DNS non-resolution verdict for one email-bank domain."""

    domain: Annotated[str, Field(description="The bank domain checked.")]
    verdict: Annotated[Verdict, Field(description="finding = the domain resolves.")]
    nxdomain: Annotated[
        bool,
        Field(description="True when every looked-up type returned NXDOMAIN."),
    ]
    lookups: Annotated[
        List[DohLookup],
        Field(description="Per-record-type lookup results (A/AAAA/MX)."),
    ]


class RegistryNameCheck(BaseModelNoExtra):
    """Registry verdicts for one screened fictional name."""

    name: Annotated[str, Field(description="The screened name (value or decoy).")]
    bank: Annotated[str, Field(description="Bank the name came from.")]
    edgar: Annotated[
        Verdict, Field(description="Case-folded exact match against EDGAR titles.")
    ]
    edgar_matches: Annotated[
        List[str],
        Field(default_factory=list, description="EDGAR titles that matched."),
    ]
    gleif: Annotated[
        Verdict,
        Field(description="Case-folded exact match against GLEIF legal names."),
    ]
    gleif_matches: Annotated[
        List[str],
        Field(default_factory=list, description="GLEIF entity names that matched."),
    ]
    gleif_error: Annotated[
        Optional[str],
        Field(default=None, description="GLEIF lookup failure, when unchecked."),
    ]

    @property
    def verdict(self) -> Verdict:
        if Verdict.FINDING in (self.edgar, self.gleif):
            return Verdict.FINDING
        if Verdict.UNCHECKED in (self.edgar, self.gleif):
            return Verdict.UNCHECKED
        return Verdict.PASS


class GroundingScreenReport(BaseModelNoExtra):
    """The provenance-bearing output of one negative screen."""

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
    dns_checks: Annotated[
        List[DomainDnsCheck], Field(description="Email-domain DNS verdicts.")
    ]
    registry_checks: Annotated[
        List[RegistryNameCheck], Field(description="Per-name registry verdicts.")
    ]
    findings: Annotated[int, Field(description="Total FINDING verdicts.")]
    unchecked: Annotated[int, Field(description="Total UNCHECKED verdicts.")]
    passes: Annotated[int, Field(description="Total PASS verdicts.")]


# --- What gets screened -------------------------------------------------------

# Banks whose values are freestanding fictional proper nouns: every value and
# every decoy is screened as-is.
REGISTRY_NAME_BANKS = ("coined", "properties", "shops")

# Insurance plans are "<carrier> <plan descriptors>": only the carrier is a
# company-like name worth screening. The descriptor vocabulary is a closed,
# in-code catalog; extraction stops at the first descriptor-like token.
_PLAN_DESCRIPTOR_TOKENS = frozenset(
    {
        "basic",
        "silver",
        "gold",
        "bronze",
        "platinum",
        "metallic",
        "standard",
        "classic",
        "complete",
        "choice",
        "advantage",
        "prime",
        "select",
        "value",
        "family",
        "care",
        "health",
        "direct",
        "flex",
        "narrow",
        "tiered",
        "catastrophic",
        "gap",
        "bridge",
        "rider",
        "duo",
        "lite",
        "step",
        "stepdown",
        "tier",
        "preferred",
        "consumer",
        "short",
        "term",
        "ppo",
        "hmo",
        "epo",
        "pos",
        "hsa",
        "hdhp",
        "hdx",
        "plus",
    }
)


def extract_carrier(plan_name: str) -> str:
    """The carrier prefix of an insurance plan display name.

    Keeps leading tokens until one is a plan descriptor (checked per
    hyphen-separated part, case-folded) or carries a digit. Falls back to
    the first token so a fully-descriptor name still screens something.
    """
    kept: List[str] = []
    for token in plan_name.split():
        parts = token.split("-")
        descriptor = any(part.casefold() in _PLAN_DESCRIPTOR_TOKENS for part in parts)
        if descriptor or any(ch.isdigit() for ch in token):
            break
        kept.append(token)
    return " ".join(kept) if kept else plan_name.split()[0]


def email_domains_to_screen(banks: Banks) -> List[str]:
    """Every domain and decoy_domain in the emails bank, unique and sorted."""
    domains = set()
    for entry in banks.emails:
        domains.add(entry.domain)
        if entry.decoy_domain:
            domains.add(entry.decoy_domain)
    return sorted(domains)


def registry_names_to_screen(banks: Banks) -> List[tuple[str, str]]:
    """(bank, name) pairs to screen, unique per bank and sorted."""
    pairs: set[tuple[str, str]] = set()
    for bank_name in REGISTRY_NAME_BANKS:
        for entry in getattr(banks, bank_name):
            for value in (entry.value, *entry.decoys):
                pairs.add((bank_name, value))
    for entry in banks.insurance_plans:
        for value in (entry.value, *entry.decoys):
            pairs.add(("insurance_plans", extract_carrier(value)))
    return sorted(pairs)


# --- The screen ----------------------------------------------------------------


def _check_domain(domain: str) -> DomainDnsCheck:
    lookups = [doh_resolve(domain, rrtype) for rrtype in _DNS_RRTYPES]
    if any(lookup.answers for lookup in lookups):
        verdict = Verdict.FINDING
    elif any(lookup.error is not None for lookup in lookups):
        verdict = Verdict.UNCHECKED
    else:
        verdict = Verdict.PASS
    nxdomain = all(lookup.status == _DNS_RCODE_NXDOMAIN for lookup in lookups)
    return DomainDnsCheck(
        domain=domain, verdict=verdict, nxdomain=nxdomain, lookups=lookups
    )


def _check_name(
    bank: str, name: str, edgar_titles: Optional[dict[str, List[str]]]
) -> RegistryNameCheck:
    folded = name.casefold()
    if edgar_titles is None:
        edgar_verdict, edgar_matches = Verdict.UNCHECKED, []
    else:
        edgar_matches = edgar_titles.get(folded, [])
        edgar_verdict = Verdict.FINDING if edgar_matches else Verdict.PASS
    try:
        candidates = gleif_legal_names(name)
    except Exception as exc:  # noqa: BLE001 - recorded, never masked as PASS
        return RegistryNameCheck(
            name=name,
            bank=bank,
            edgar=edgar_verdict,
            edgar_matches=edgar_matches,
            gleif=Verdict.UNCHECKED,
            gleif_error=str(exc),
        )
    gleif_matches = sorted({c for c in candidates if c.casefold() == folded})
    return RegistryNameCheck(
        name=name,
        bank=bank,
        edgar=edgar_verdict,
        edgar_matches=edgar_matches,
        gleif=Verdict.FINDING if gleif_matches else Verdict.PASS,
        gleif_matches=gleif_matches,
    )


def run_screen(banks: Optional[Banks] = None) -> GroundingScreenReport:
    """Run every screen over the checked-in banks and build the report."""
    banks = banks or load_banks()

    sources: List[SourceRecord] = [
        SourceRecord(
            name="Google Public DNS (DoH JSON API)",
            url=DOH_ENDPOINT,
            note="A/AAAA/MX non-resolution checks for email-bank domains",
        )
    ]

    edgar_titles: Optional[dict[str, List[str]]] = None
    try:
        catalog = fetch_edgar_catalog()
    except Exception as exc:  # noqa: BLE001 - all EDGAR checks become UNCHECKED
        logger.warning(f"EDGAR catalog fetch failed; EDGAR checks UNCHECKED: {exc}")
        sources.append(
            SourceRecord(
                name="SEC EDGAR company_tickers.json",
                url=EDGAR_TICKERS_URL,
                note=f"fetch failed, all EDGAR checks UNCHECKED: {exc}",
            )
        )
    else:
        edgar_titles = {}
        for title in catalog.titles:
            edgar_titles.setdefault(title.casefold(), []).append(title)
        sources.append(
            SourceRecord(
                name="SEC EDGAR company_tickers.json",
                url=EDGAR_TICKERS_URL,
                sha256=catalog.sha256,
                note=f"{len(catalog.titles)} registrant titles",
            )
        )
    sources.append(
        SourceRecord(
            name="GLEIF LEI records (fulltext search)",
            url=GLEIF_FULLTEXT_URL,
            note=f"one request per unique name, page size {GLEIF_PAGE_SIZE}",
        )
    )

    dns_checks = [_check_domain(d) for d in email_domains_to_screen(banks)]
    registry_checks = [
        _check_name(bank, name, edgar_titles)
        for bank, name in registry_names_to_screen(banks)
    ]

    verdicts = [check.verdict for check in dns_checks] + [
        check.verdict for check in registry_checks
    ]
    screened_banks = (*REGISTRY_NAME_BANKS, "insurance_plans", "emails")
    return GroundingScreenReport(
        screen_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tool_version=package_version("tau2"),
        screen_version=SCREEN_VERSION,
        bank_shas={name: banks.shas[name] for name in sorted(screened_banks)},
        sources=sources,
        dns_checks=dns_checks,
        registry_checks=registry_checks,
        findings=sum(v is Verdict.FINDING for v in verdicts),
        unchecked=sum(v is Verdict.UNCHECKED for v in verdicts),
        passes=sum(v is Verdict.PASS for v in verdicts),
    )


def write_screen_report(
    report: GroundingScreenReport, out_path: Optional[Path] = None
) -> Path:
    """Write the report JSON (sorted keys, trailing newline) and return the path."""
    out_path = out_path or DEFAULT_SCREEN_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2) + "\n")
    return out_path

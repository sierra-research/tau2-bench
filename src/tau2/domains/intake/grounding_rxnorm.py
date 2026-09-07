# Copyright Sierra
"""Positive grounding of the medications bank against RxNorm.

``tau2 intake-grounding rxnorm`` validates every medications-bank value AND
decoy against the NLM RxNorm API (https://rxnav.nlm.nih.gov/), and writes a
provenance-bearing report (:class:`RxnormGroundingReport`) to
``data/tau2/domains/intake/grounding_rxnorm.json``. It never edits a bank:
findings are surfaced for the owner to act on.

Each bank value is ``"<Drug> <strength> <form>"`` (design doc
``docs/designs/intake-grounding.md``, roadmap item 3). Verdict per entry:

- **PASS** — RxNorm knows every ingredient (``/REST/rxcui.json?search=2``,
  exact + normalized) AND a marketed product concept
  (``/REST/drugs.json?name=<first ingredient>``: SCD/SBD/pack names)
  matches the strength AND the form.
- **FINDING** — an ingredient is unknown, or no product concept matches the
  strength+form combination. A FINDING is a fact for the owner, not an
  auto-failure.
- **UNCHECKED** — a lookup failed; never a silent PASS.

Deterministic normalization (the whole of it — tested in
``test_grounding_positive_intake.py``):

- **Parse** — leading non-digit-leading words are the drug name (a single
  hyphenated word is a combination: ``Levodopa-Carbidopa`` -> two
  ingredients; a multi-word name is one ingredient: ``Mycophenolate
  mofetil``); then words containing a digit or in the closed unit vocabulary
  (mg/mcg/g/mL/%) are the strength; the rest is the form.
- **Strength** — rendered into RxNorm's vocabulary: ``N mg`` -> ``N mg``;
  ``N mcg`` -> ``N mcg`` or ``N/1000 mg`` (RxNorm states levothyroxine etc.
  in MG); ``N%`` -> ``N*10 mg/ml`` or ``N/100 mg/mg`` or ``N %``;
  ``A mg/B mL`` -> ``A/B mg/ml``; combination ``A-B mg`` -> both ``A mg``
  and ``B mg`` required (set semantics: ingredient/strength pairing is NOT
  enforced, because label conventions like ``25-100`` do not fix it).
  Numbers match behind digit-boundary lookarounds so ``50 mg`` never
  matches inside ``850 mg``.
- **Form** — a closed map into RxNorm dose-form phrases: ``tablet`` ->
  ``oral tablet`` (a modified-release product also satisfies a plain
  ``tablet`` claim, since the phrase matches inside ``extended release oral
  tablet``); ``ER/XL/XR tablet`` -> ``extended release oral tablet``;
  ``DR capsule`` -> ``delayed release oral capsule``; ``SL tablet`` ->
  ``sublingual tablet``; ``solution`` -> ``oral solution``; ``cream``/
  ``gel`` match any cream/gel dose form.

Network access happens only here, at build time — one cached request per
unique ingredient against each endpoint.
"""

import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, List, Optional, Union

import requests
from pydantic import Field

from tau2.domains.intake.grounding_screen import SourceRecord, Verdict
from tau2.domains.intake.tasks.banks import Banks, load_banks
from tau2.domains.intake.utils import INTAKE_DATA_DIR
from tau2.utils.pydantic_utils import BaseModelNoExtra

# Report schema version: bump on any change to the report models or to what
# the screen checks, so two reports are comparable only when versions match.
RXNORM_SCREEN_VERSION = "1.0.0"

DEFAULT_RXNORM_REPORT_PATH = INTAKE_DATA_DIR / "grounding_rxnorm.json"

RXNORM_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNORM_DRUGS_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"

# NLM asks for no more than 20 requests/second per IP; lookups sleep between
# requests (politeness lives at the network seam — mocked tests never pay it).
_HTTP_USER_AGENT = "tau2-bench (Sierra AI, https://sierra.ai)"
_HTTP_TIMEOUT_S = 30
RXNORM_REQUEST_INTERVAL_S = 0.25


# --- Network seam (monkeypatched by tests) ----------------------------------


def fetch_rxnorm_rxcui(name: str) -> List[str]:
    """RxCUIs for a name via ``rxcui.json?search=2`` (exact + normalized).

    Sleeps :data:`RXNORM_REQUEST_INTERVAL_S` before the request (polite rate
    limiting). Raises on transport/protocol failure — the caller records
    UNCHECKED, never PASS.
    """
    time.sleep(RXNORM_REQUEST_INTERVAL_S)
    response = requests.get(
        RXNORM_RXCUI_URL,
        params={"name": name, "search": "2"},
        headers={"User-Agent": _HTTP_USER_AGENT},
        timeout=_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    return [str(rxcui) for rxcui in payload.get("idGroup", {}).get("rxnormId", [])]


def fetch_rxnorm_drugs(name: str) -> List[str]:
    """All drug-product concept names related to a name via ``drugs.json``.

    Collects every conceptGroup's conceptProperties names (SCD/SBD and pack
    concepts). Sleeps like :func:`fetch_rxnorm_rxcui`; raises on failure.
    """
    time.sleep(RXNORM_REQUEST_INTERVAL_S)
    response = requests.get(
        RXNORM_DRUGS_URL,
        params={"name": name},
        headers={"User-Agent": _HTTP_USER_AGENT},
        timeout=_HTTP_TIMEOUT_S,
    )
    response.raise_for_status()
    payload = response.json()
    names: List[str] = []
    for group in payload.get("drugGroup", {}).get("conceptGroup", []) or []:
        for concept in group.get("conceptProperties", []) or []:
            if concept.get("name"):
                names.append(str(concept["name"]))
    return names


# --- Deterministic parse + normalization -------------------------------------

# The closed unit vocabulary a strength word may carry (besides digits).
_UNIT_WORDS = frozenset({"mg", "mcg", "g", "ml", "%"})

# The closed form map into RxNorm dose-form phrases. A bank form outside this
# map is a grammar violation and fails loud (the bank grammar is ours).
FORM_PHRASES: dict[str, str] = {
    "tablet": "oral tablet",
    "capsule": "oral capsule",
    "er tablet": "extended release oral tablet",
    "xl tablet": "extended release oral tablet",
    "xr tablet": "extended release oral tablet",
    "er capsule": "extended release oral capsule",
    "dr capsule": "delayed release oral capsule",
    "sl tablet": "sublingual tablet",
    "cream": "cream",
    "gel": "gel",
    "solution": "oral solution",
}


def _num(value: Decimal) -> str:
    """A Decimal as RxNorm prints numbers: no exponent, no trailing zeros."""
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        normalized = normalized.quantize(Decimal(1))
    return format(normalized, "f")


def _unit_alternatives(number: str, unit: str) -> List[str]:
    """The RxNorm renderings one ``<number> <unit>`` strength may take."""
    n = Decimal(number)
    if unit == "mg":
        return [f"{_num(n)} mg"]
    if unit == "mcg":
        return [f"{_num(n)} mcg", f"{_num(n / 1000)} mg"]
    if unit == "g":
        return [f"{_num(n)} g", f"{_num(n * 1000)} mg"]
    raise ValueError(f"Unknown strength unit {unit!r}")


def normalize_strength(strength: str) -> List[List[str]]:
    """Alternative RxNorm renderings of a bank strength.

    Returns a list of required strengths (ALL must match a product concept),
    each a list of alternative renderings (ANY may match). Fails loud on a
    strength outside the bank grammar.
    """
    folded = " ".join(strength.casefold().split())
    ratio = re.fullmatch(r"([\d.]+) mg/([\d.]+) ml", folded)
    if ratio:
        per_ml = Decimal(ratio.group(1)) / Decimal(ratio.group(2))
        return [[f"{_num(per_ml)} mg/ml"]]
    percent = re.fullmatch(r"([\d.]+)%", folded)
    if percent:
        n = Decimal(percent.group(1))
        return [[f"{_num(n * 10)} mg/ml", f"{_num(n / 100)} mg/mg", f"{_num(n)} %"]]
    combo = re.fullmatch(r"([\d.]+(?:-[\d.]+)+) (mg|mcg|g)", folded)
    if combo:
        return [
            _unit_alternatives(part, combo.group(2))
            for part in combo.group(1).split("-")
        ]
    simple = re.fullmatch(r"([\d.]+) (mg|mcg|g)", folded)
    if simple:
        return [_unit_alternatives(simple.group(1), simple.group(2))]
    raise ValueError(f"Strength {strength!r} is outside the bank grammar")


class MedicationParse(BaseModelNoExtra):
    """The deterministic decomposition of one ``<Drug> <strength> <form>``."""

    ingredients: Annotated[
        List[str],
        Field(description="Ingredient names (hyphenated combos split)."),
    ]
    strength: Annotated[str, Field(description="The strength words, as written.")]
    form: Annotated[str, Field(description="The form words, as written.")]

    @property
    def strength_alternatives(self) -> List[List[str]]:
        return normalize_strength(self.strength)

    @property
    def form_phrase(self) -> str:
        return FORM_PHRASES[self.form.casefold()]


def parse_medication(value: str) -> MedicationParse:
    """Parse a bank value into ingredients, strength, and form. Fails loud
    on anything outside the bank grammar."""
    words = value.split()
    name_words: List[str] = []
    index = 0
    while index < len(words) and not words[index][0].isdigit():
        name_words.append(words[index])
        index += 1
    if not name_words or index == len(words):
        raise ValueError(f"Medication value {value!r} is outside the bank grammar")
    if len(name_words) == 1 and "-" in name_words[0]:
        ingredients = [part for part in name_words[0].split("-") if part]
    else:
        ingredients = [" ".join(name_words)]
    strength_words: List[str] = []
    while index < len(words):
        word = words[index]
        if any(ch.isdigit() for ch in word) or word.casefold() in _UNIT_WORDS:
            strength_words.append(word)
            index += 1
        else:
            break
    form = " ".join(words[index:])
    if not strength_words or not form:
        raise ValueError(f"Medication value {value!r} is outside the bank grammar")
    if form.casefold() not in FORM_PHRASES:
        raise ValueError(f"Medication form {form!r} is outside the closed form map")
    parse = MedicationParse(
        ingredients=ingredients, strength=" ".join(strength_words), form=form
    )
    parse.strength_alternatives  # noqa: B018 - fail loud on grammar violations
    return parse


def _word_pattern(text: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z])" + re.escape(text) + r"(?![a-z])")


def _strength_pattern(alternative: str) -> re.Pattern[str]:
    # Digit-boundary lookarounds: "50 mg" must not match inside "850 mg",
    # and "1 mg" must not match inside "1 mg/ml".
    return re.compile(r"(?<![\d.])" + re.escape(alternative) + r"(?![\d/])")


def match_product(concept_names: List[str], parse: MedicationParse) -> Optional[str]:
    """The first product concept matching every ingredient, every required
    strength, and the form phrase — shortest name first, so the plain
    clinical drug (SCD) wins over branded/pack concepts, deterministically."""
    ingredient_patterns = [
        _word_pattern(" ".join(ingredient.casefold().split()))
        for ingredient in parse.ingredients
    ]
    strength_patterns = [
        [_strength_pattern(alternative) for alternative in alternatives]
        for alternatives in parse.strength_alternatives
    ]
    form_phrase = parse.form_phrase
    for name in sorted(concept_names, key=lambda name: (len(name), name)):
        folded = " ".join(name.casefold().split())
        if form_phrase not in folded:
            continue
        if not all(pattern.search(folded) for pattern in ingredient_patterns):
            continue
        if not all(
            any(pattern.search(folded) for pattern in alternatives)
            for alternatives in strength_patterns
        ):
            continue
        return name
    return None


# --- Report models -----------------------------------------------------------


class MedicationCheck(BaseModelNoExtra):
    """RxNorm verdict for one medications-bank value or decoy."""

    value: Annotated[str, Field(description="The exact bank string checked.")]
    role: Annotated[str, Field(description="'value' or 'decoy'.")]
    verdict: Annotated[
        Verdict,
        Field(
            description="pass = every ingredient known AND a marketed "
            "product matches strength+form."
        ),
    ]
    ingredients: Annotated[
        List[str], Field(description="Parsed ingredient names checked.")
    ]
    strength: Annotated[str, Field(description="Parsed strength, as written.")]
    form: Annotated[str, Field(description="Parsed form, as written.")]
    matched_concept: Annotated[
        Optional[str],
        Field(default=None, description="The RxNorm product concept that matched."),
    ]
    note: Annotated[
        Optional[str],
        Field(default=None, description="Why the check is a finding/unchecked."),
    ]


class RxnormGroundingReport(BaseModelNoExtra):
    """The provenance-bearing output of one medications positive screen."""

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
    checks: Annotated[
        List[MedicationCheck], Field(description="Per-value/decoy RxNorm verdicts.")
    ]
    findings: Annotated[int, Field(description="Total FINDING verdicts.")]
    unchecked: Annotated[int, Field(description="Total UNCHECKED verdicts.")]
    passes: Annotated[int, Field(description="Total PASS verdicts.")]


# --- The screen ----------------------------------------------------------------


def medications_to_check(banks: Banks) -> List[tuple[str, str]]:
    """(role, value) pairs to check, unique and sorted."""
    pairs: set[tuple[str, str]] = set()
    for entry in banks.medications:
        pairs.add(("value", entry.value))
        for decoy in entry.decoys:
            pairs.add(("decoy", decoy))
    return sorted(pairs)


def run_rxnorm_screen(banks: Optional[Banks] = None) -> RxnormGroundingReport:
    """Screen every medications-bank value and decoy against RxNorm."""
    banks = banks or load_banks()
    pairs = medications_to_check(banks)

    rxcui_cache: dict[str, Union[List[str], Exception]] = {}
    drugs_cache: dict[str, Union[List[str], Exception]] = {}

    def cached(cache: dict, fetch, name: str) -> Union[List[str], Exception]:
        key = name.casefold()
        if key not in cache:
            try:
                cache[key] = fetch(name)
            except Exception as exc:  # noqa: BLE001 - recorded, never PASS
                cache[key] = exc
        return cache[key]

    checks: List[MedicationCheck] = []
    for role, value in pairs:
        parse = parse_medication(value)
        base = dict(
            value=value,
            role=role,
            ingredients=parse.ingredients,
            strength=parse.strength,
            form=parse.form,
        )
        lookup_error: Optional[str] = None
        unknown: List[str] = []
        for ingredient in parse.ingredients:
            result = cached(rxcui_cache, fetch_rxnorm_rxcui, ingredient)
            if isinstance(result, Exception):
                lookup_error = f"rxcui lookup failed for {ingredient!r}: {result}"
                break
            if not result:
                unknown.append(ingredient)
        if lookup_error is not None:
            checks.append(
                MedicationCheck(**base, verdict=Verdict.UNCHECKED, note=lookup_error)
            )
            continue
        if unknown:
            checks.append(
                MedicationCheck(
                    **base,
                    verdict=Verdict.FINDING,
                    note="ingredient unknown to RxNorm: " + ", ".join(unknown),
                )
            )
            continue
        products = cached(drugs_cache, fetch_rxnorm_drugs, parse.ingredients[0])
        if isinstance(products, Exception):
            checks.append(
                MedicationCheck(
                    **base,
                    verdict=Verdict.UNCHECKED,
                    note=f"drugs lookup failed for {parse.ingredients[0]!r}: "
                    f"{products}",
                )
            )
            continue
        matched = match_product(products, parse)
        if matched is None:
            checks.append(
                MedicationCheck(
                    **base,
                    verdict=Verdict.FINDING,
                    note="no marketed RxNorm product matches strength+form",
                )
            )
            continue
        checks.append(
            MedicationCheck(**base, verdict=Verdict.PASS, matched_concept=matched)
        )

    sources = [
        SourceRecord(
            name="NLM RxNorm rxcui (exact + normalized name search)",
            url=RXNORM_RXCUI_URL,
            note="one cached request per unique ingredient (search=2)",
        ),
        SourceRecord(
            name="NLM RxNorm drugs (product concepts by name)",
            url=RXNORM_DRUGS_URL,
            note="one cached request per unique first ingredient",
        ),
    ]
    verdicts = [check.verdict for check in checks]
    return RxnormGroundingReport(
        screen_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tool_version=package_version("tau2"),
        screen_version=RXNORM_SCREEN_VERSION,
        bank_shas={"medications": banks.shas["medications"]},
        sources=sources,
        checks=checks,
        findings=sum(v is Verdict.FINDING for v in verdicts),
        unchecked=sum(v is Verdict.UNCHECKED for v in verdicts),
        passes=sum(v is Verdict.PASS for v in verdicts),
    )


def write_rxnorm_report(
    report: RxnormGroundingReport, out_path: Optional[Path] = None
) -> Path:
    """Write the report JSON (indented, trailing newline) and return the path."""
    out_path = out_path or DEFAULT_RXNORM_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2) + "\n")
    return out_path

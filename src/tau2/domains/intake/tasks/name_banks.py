# Copyright Sierra
"""Census/SSA-based person-name bank: the ``tau2 intake-names`` pipeline.

The intake domain's person-name bank (``person_names.yaml``) is REBUILT
deterministically from real US name statistics — no invented names. (The
``providers.yaml`` bank this pipeline also used to emit was retired
2026-08-24 with the v4 generator rewrite: providers duplicated the same
census pools behind a "Dr." prefix — design doc §11.) Two stages, both
``tau2`` verbs:

1. ``tau2 intake-names extract`` — turn the five raw public-domain datasets
   into small, checked-in extracts under ``banks/../name_sources/`` and record
   full provenance (source URLs, retrieval date, raw-file sha256s, extract
   sha256s, filter parameters):

   - **SSA national baby names** (https://www.ssa.gov/oact/babynames/names.zip,
     ``yob<year>.txt`` per year since 1880): given-name counts per sex are
     aggregated over the **birth-cohort window 1950-1989** — the bank's
     birth dates span 1954-1988, so a drawn caller name is plausible for the
     drawn DOB by construction (no 1958-born "Jayden"). Kept: every name whose
     larger-sex window total is >= 200, with per-decade counts.
   - **US Census Bureau 2010 surnames**
     (https://www2.census.gov/topics/genealogy/2010surnames/names.zip,
     ``Names_2010Census.csv``, all 162,253 surnames with >= 100 bearers):
     kept as the **head** (rank 1-2000, count >= 18,075) plus the **rare
     tail** (count 100-129, ranks ~131,379-160,975 — the bottom fifth of the
     published file and the rarest band the Census publishes at all).
   - **CMUdict** ``cmudict.dict`` (github.com/cmusphinx/cmudict, pinned
     commit) and **WikiPron** ``eng_latn_us_broad.tsv``
     (github.com/CUNY-CL/wikipron, pinned commit): pronunciation rows are
     kept for every extract name (raw and restored-casing forms) and every
     medication drug token — ``pronunciations.csv``
     (word, source, phonemes). Variant CMUdict entries (``word(2)``) are
     dropped; the first row per (word, source) wins.
   - **English wordlist** ``web2`` (FreeBSD ``share/dict/web2``, Webster's
     Second, pinned commit): only the all-lowercase lines count (capitalized
     web2 rows are proper nouns — exactly the leakage the filter exists to
     kill, so they must not shadow it). ``wordlist_hits.csv`` keeps the
     wordlist rows matching an extract name or its singular-stripped form
     (kills "Diets"/"Fowls").

2. ``tau2 intake-names build`` — rebuild the bank from the extracts with
   a fixed seed (byte-identical on re-run). The rules, all in this module:

   - **Difficulty is frequency**: easy = a top-40-per-sex cohort given name
     (window total >= 20,000 and >= 200 in each of the four cohort decades)
     + a top-300 census surname (>= 104,057 bearers, top 0.2% of the file).
     Hard = a rare-tail given name (window total 300-3,000, present with
     >= 10 in at least 3 of 4 cohort decades) + either a single rare-tail
     surname or a **constructed hyphenation of two rare-tail surnames**
     (flagged in the build manifest).
   - **Hard tier is lexicon-grounded** (design doc §2.2): hard given-name
     and tail-surname VALUE pools are restricted to names with a
     dataset-attested pronunciation (CMUdict or WikiPron,
     case-insensitive), that are not an English word (nor their
     singular-stripped form) and are >= 5 characters; the fixed place/brand
     denylist additionally skips candidates at DRAW time (so extending it
     never reshuffles the seeded pool order). Easy pools keep their bands
     (top names are lexicon-covered ~100%; the rare uncovered one is
     dropped). Decoy candidate pools stay unfiltered — decoys are
     text-confusable neighbors, not spoken values.
   - **Pronunciation columns** (design doc §2.4): every person value token
     (first name, each surname token) carries {source, phonemes, respelling,
     mispronounced} — respelling derived by the fixed converters in
     :mod:`tau2.domains.intake.tasks.pronunciation`, hard-tier tokens also a
     seeded distortion-operator ``mispronounced`` variant (design doc §4;
     operators drawn balance-greedily per bank under the fixed per-bank
     applicability table, ties broken by per-token seeded rng streams).
     The build also regenerates the medication bank's pronunciation columns
     in place (drug-name tokens only; the ~9 tokens absent from both
     lexicons are hand-filled from MedlinePlus with per-entry citations).
   - **Gender from SSA sex ratios**: a name is labeled male/female by its
     dominant sex over the cohort window; names below **90% dominance**
     (unisex) are never drawn, so the emitted gender map stays unambiguous.
   - **Casing restoration, ASCII only**: the census file is uppercase with
     apostrophes and diacritics stripped. Restoration is (a) a closed
     apostrophe table (OBRIEN -> O'Brien), (b) the Mc-prefix rule
     (MCDONALD -> McDonald), (c) Title case otherwise. Diacritics are NEVER
     restored (GARCIA -> Garcia): the banks are ASCII-only by decree
     (owner directive 2026-08-23; the repo-wide identity-ASCII-fold
     convention), enforced by a build assert and a bank test.
   - **Decoys are real near-neighbors**: ``decoys[0]`` keeps the given name
     and swaps the surname for its closest real census neighbor (edit
     distance, then longest common prefix); ``decoys[1]`` keeps the surname
     and swaps the given name for its closest real SSA neighbor from the
     same tier pool. Every value/decoy pair is fold-distinct and the whole
     display-string space is containment-free (no name a raw or folded
     substring of another, legacy Phase-1 names included) so the generator's
     capture-absence assert holds by construction.
   - **alt_name** (the record-under-a-different-key former-name path) keeps
     the given name and takes a fresh same-tier census surname; populated on
     every other entry per tier, like the bank it replaces.

The build honors the flat bank contract (design doc §4): exactly
``PERSON_COUNT_PER_TIER`` = 40 easy and 40 hard entries.

The build also emits ``name_sources/name_banks.manifest.json`` carrying the
seed, rule constants, constructed-surname flags, and the given-name -> gender
map used by localized extensions. The generator's caller-gender cross-check
fails loud on drift.
"""

import csv
import hashlib
import io
import json
import random
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Annotated, Callable, Optional

import yaml
from loguru import logger
from pydantic import Field

from tau2.domains.intake.folds import fold_name
from tau2.domains.intake.tasks.banks import (
    Difficulty,
    Gender,
    PronunciationSource,
    TokenPronunciation,
    instantiate_email,
    load_banks,
    medication_drug_tokens,
    person_name_tokens,
)
from tau2.domains.intake.tasks.pronunciation import (
    MispronunciationDrawer,
    Syllable,
    render_respelling,
    syllabify_phonemes,
)
from tau2.domains.intake.utils import INTAKE_DATA_DIR
from tau2.utils.pydantic_utils import BaseModelNoExtra

NAME_SOURCES_DIR = INTAKE_DATA_DIR / "name_sources"

SSA_SOURCE_URL = "https://www.ssa.gov/oact/babynames/names.zip"
CENSUS_SOURCE_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"
# Pronunciation lexicons + wordlist, pinned at fixed commits (design §2.1).
CMUDICT_SOURCE_URL = (
    "https://raw.githubusercontent.com/cmusphinx/cmudict/"
    "0f8072f814306c5ee4fbf992ed853601b12c01f9/cmudict.dict"
)
WIKIPRON_SOURCE_URL = (
    "https://raw.githubusercontent.com/CUNY-CL/wikipron/"
    "d282e848a211ea31cfd730f0ced8bc8cdab9e83d/data/scrape/tsv/"
    "eng_latn_us_broad.tsv"
)
WORDLIST_SOURCE_URL = (
    "https://raw.githubusercontent.com/freebsd/freebsd-src/"
    "72f0bc868bf00586cba1e50057d8f1998b4abe80/share/dict/web2"
)

EXTRACTOR_VERSION = "2"  # v2 (2026-08-25): + lexicons, wordlist hits
# v3 (2026-08-25): lexicon-grounded hard tier + pronunciations
# v4 (2026-08-26): spelling_pronunciation operator + per-bank balance-greedy
#                  operator draw (values/decoys unchanged from v3)
BUILDER_VERSION = "4"
DEFAULT_BUILD_SEED = 20260823

# --- extraction parameters (documented above, pinned here) -----------------
COHORT_START = 1950  # bank birth dates span 1954-1988
COHORT_END = 1989
COHORT_DECADES = (1950, 1960, 1970, 1980)
GIVEN_EXTRACT_FLOOR = 200  # keep a name when its larger sex total >= this
SURNAME_HEAD_MAX_RANK = 2000
SURNAME_TAIL_MIN_COUNT = 100  # the census publication floor
SURNAME_TAIL_MAX_COUNT = 129

# --- build parameters -------------------------------------------------------
GENDER_DOMINANCE_MIN = 0.90  # majority-sex share over the cohort window
EASY_GIVEN_MIN_TOTAL = 20000
EASY_GIVEN_MIN_PER_DECADE = 200  # in every cohort decade
HARD_GIVEN_MIN_TOTAL = 300
HARD_GIVEN_MAX_TOTAL = 3000
HARD_GIVEN_MIN_DECADES = 3  # decades with >= HARD_GIVEN_DECADE_FLOOR
HARD_GIVEN_DECADE_FLOOR = 10
EASY_SURNAME_MAX_RANK = 300  # top 0.2% of the published file
PERSON_COUNT_PER_TIER = 40  # the flat bank contract (design doc §4)

# --- hard-tier lexicon-grounding parameters (design doc §2.2) ---------------
HARD_NAME_MIN_LENGTH = 5
# The filtered pools are 10-20x oversubscribed; a count far below these
# floors means the coverage filter (or an extract) is broken, not the data.
HARD_GIVEN_POOL_FLOOR = 1500
TAIL_SURNAME_POOL_FLOOR = 800

# Place/brand/word leakage the pinned wordlist cannot see (proper nouns and
# post-1934 vocabulary are not lowercase web2 rows). Fixed, reviewed against
# the sampled build output; keys are lowercase. Applied as a DRAW-TIME skip
# over hard given-name and surname VALUE draws (decoys are untouched) so the
# seeded pool order — and therefore every other draw — is stable when an
# entry is added: the draw just advances past the denied candidate.
PRONUNCIATION_DENYLIST = frozenset(
    {
        # places / peoples / demonyms
        "akashi",  # city
        "alicante",  # city
        "amboy",  # Perth Amboy
        "ashanti",  # people/region
        "belfast",  # city
        "bombay",  # city
        "caracas",  # city
        "charlotte",  # city
        "chicano",  # demonym
        "comanche",  # nation/tribe
        "cuban",  # demonym
        "farsi",  # language
        "fribourg",  # city
        "fujisawa",  # city
        "greenville",  # city
        "jammu",  # city/region
        "kanuri",  # people/language
        "louisville",  # city
        "lynwood",  # city
        "macarena",  # La Macarena (and the song)
        "mackinaw",  # place (Mackinaw City) / coat
        "malvern",  # town
        "masai",  # people
        "miquelon",  # Saint-Pierre-et-Miquelon
        "muenchen",  # Munich
        "navajo",  # nation/tribe
        "northridge",  # Los Angeles district
        "osaka",  # city
        "panjabi",  # demonym spelling
        "romania",  # country
        "rusyn",  # people
        "sardinia",  # island
        "shiraz",  # city / grape
        "spanish",  # demonym
        "syria",  # country
        "tecumseh",  # Shawnee leader; namesake towns
        "tilburg",  # city
        "tsushima",  # island
        "zinder",  # city
        # brands / franchises
        "bichon",  # dog breed
        "chivas",  # whisky brand
        "jetson",  # The Jetsons
        "makita",  # tool brand
        "sanka",  # coffee brand
        "vulcan",  # mythology / Star Trek
        # dictionary words web2 (1934, lowercase rows) does not carry
        "balun",
        "boning",
        "celebre",
        "coldwater",
        "favour",
        "fusilli",
        "groundwater",
        "inches",
        "lassi",
        "mardi",
        "mosses",
        "nisan",
        "october",
        "porting",
        "rugger",
        "shorthair",
        "sicko",
        "sideman",
        "stats",
        "touche",
        "trippy",
        "weeding",
        "zeroth",
    }
)

# Pronunciation source preference per bank: CMUdict (built for ASR) carries
# the surnames; WikiPron carries the drugs (design doc §1).
PERSON_SOURCE_PREFERENCE = (PronunciationSource.CMUDICT, PronunciationSource.WIKIPRON)
MEDICATION_SOURCE_PREFERENCE = (
    PronunciationSource.WIKIPRON,
    PronunciationSource.CMUDICT,
)

# Given names never drawn, keyed like name_genders._normalize_given_name:
# reviewed-unisex catalog names whose fixed assignment must not be disturbed.
EXCLUDED_GIVEN_KEYS = frozenset({"chen", "lei", "pat", "sam", "wei"})

# Closed, well-attested apostrophe restorations for the uppercase census file
# (ASCII only — never diacritics). Anything not listed here falls through to
# the Mc rule or plain Title case.
APOSTROPHE_SURNAMES: dict[str, str] = {
    "OBRIEN": "O'Brien",
    "OBRYAN": "O'Bryan",
    "OCONNELL": "O'Connell",
    "OCONNOR": "O'Connor",
    "ODONNELL": "O'Donnell",
    "OGRADY": "O'Grady",
    "OHARA": "O'Hara",
    "OKEEFE": "O'Keefe",
    "OLEARY": "O'Leary",
    "OMALLEY": "O'Malley",
    "ONEAL": "O'Neal",
    "ONEIL": "O'Neil",
    "ONEILL": "O'Neill",
    "OQUINN": "O'Quinn",
    "OREILLY": "O'Reilly",
    "OROURKE": "O'Rourke",
    "OSHEA": "O'Shea",
    "OSULLIVAN": "O'Sullivan",
    "OTOOLE": "O'Toole",
    "DAMICO": "D'Amico",
    "DANGELO": "D'Angelo",
    "DAMBROSIO": "D'Ambrosio",
}

# The Phase-1 hand-authored person-shaped strings that stay in the DB view
# (generator.py LEGACY_*): every generated display string must stay
# containment-free against them (raw and folded).
LEGACY_DISPLAY_NAMES = (
    "Dana Oliveira",
    "Daniel Okafor",
    "Grace Njeri",
    "Dr. Alan Reyes",
    "Dr. Renée Vásquez",
    "Dr. Miriam Osei",
)


class NameBanksError(Exception):
    """A name-bank pipeline invariant failed; outputs must not ship."""


# ---------------------------------------------------------------------------
# Manual medication pronunciations (design doc §2.4)
# ---------------------------------------------------------------------------


class ManualPronunciation(BaseModelNoExtra):
    """A hand-filled pronunciation for a drug token absent from both
    lexicons: the printed reference pronunciation (stored in the bank's
    ``phonemes`` column), its syllable structure in this repo's respelling
    units (the ``respelling`` column renders from it, and the distortion
    operators run on it), and the citation."""

    reference: Annotated[
        str, Field(description="The source pronunciation exactly as printed.")
    ]
    syllables: Annotated[
        list[Syllable], Field(description="The same pronunciation in repo units.")
    ]
    citation: Annotated[str, Field(description="Where the pronunciation is printed.")]


# The combination/uncovered drug tokens (lowercase). Reference pronunciations
# are the MedlinePlus drug-information phonetic respellings (USAN style).
MANUAL_MEDICATION_PRONUNCIATIONS: dict[str, ManualPronunciation] = {
    "amlodipine": ManualPronunciation(
        reference="am LOE di peen",
        syllables=[
            Syllable(onset=[], nucleus="a", coda=["m"]),
            Syllable(onset=["l"], nucleus="oh", coda=[], stressed=True),
            Syllable(onset=["d"], nucleus="ih", coda=[]),
            Syllable(onset=["p"], nucleus="ee", coda=["n"]),
        ],
        citation="MedlinePlus drug information: Amlodipine ('am LOE di peen')",
    ),
    "cephalexin": ManualPronunciation(
        reference="sef a LEX in",
        syllables=[
            Syllable(onset=["s"], nucleus="eh", coda=["f"]),
            Syllable(onset=[], nucleus="uh", coda=[]),
            Syllable(onset=["l"], nucleus="eh", coda=["k"], stressed=True),
            Syllable(onset=["s"], nucleus="ih", coda=["n"]),
        ],
        citation="MedlinePlus drug information: Cephalexin ('sef a LEX in')",
    ),
    "clobazam": ManualPronunciation(
        reference="KLOE ba zam",
        syllables=[
            Syllable(onset=["k", "l"], nucleus="oh", coda=[], stressed=True),
            Syllable(onset=["b"], nucleus="uh", coda=[]),
            Syllable(onset=["z"], nucleus="a", coda=["m"]),
        ],
        citation="MedlinePlus drug information: Clobazam ('KLOE ba zam')",
    ),
    "cyclobenzaprine": ManualPronunciation(
        reference="sye kloe BEN za preen",
        syllables=[
            Syllable(onset=["s"], nucleus="eye", coda=[]),
            Syllable(onset=["k", "l"], nucleus="oh", coda=[]),
            Syllable(onset=["b"], nucleus="eh", coda=["n"], stressed=True),
            Syllable(onset=["z"], nucleus="uh", coda=[]),
            Syllable(onset=["p", "r"], nucleus="ee", coda=["n"]),
        ],
        citation=(
            "MedlinePlus drug information: Cyclobenzaprine ('sye kloe BEN za preen')"
        ),
    ),
    "isosorbide": ManualPronunciation(
        reference="eye soe SOR bide",
        syllables=[
            Syllable(onset=[], nucleus="eye", coda=[]),
            Syllable(onset=["s"], nucleus="oh", coda=[]),
            Syllable(onset=["s"], nucleus="aw", coda=["r"], stressed=True),
            Syllable(onset=["b"], nucleus="eye", coda=["d"]),
        ],
        citation=(
            "MedlinePlus drug information: Isosorbide mononitrate ('eye soe SOR bide')"
        ),
    ),
    "mononitrate": ManualPronunciation(
        reference="mon oh NYE trate",
        syllables=[
            Syllable(onset=["m"], nucleus="ah", coda=["n"]),
            Syllable(onset=[], nucleus="oh", coda=[]),
            Syllable(onset=["n"], nucleus="eye", coda=[], stressed=True),
            Syllable(onset=["t", "r"], nucleus="ay", coda=["t"]),
        ],
        citation=(
            "MedlinePlus drug information: Isosorbide mononitrate ('mon oh NYE trate')"
        ),
    ),
    "nicardipine": ManualPronunciation(
        reference="nye KAR de peen",
        syllables=[
            Syllable(onset=["n"], nucleus="eye", coda=[]),
            Syllable(onset=["k"], nucleus="ah", coda=["r"], stressed=True),
            Syllable(onset=["d"], nucleus="ih", coda=[]),
            Syllable(onset=["p"], nucleus="ee", coda=["n"]),
        ],
        citation="MedlinePlus drug information: Nicardipine ('nye KAR de peen')",
    ),
    "trandolapril": ManualPronunciation(
        reference="tran DOE la pril",
        syllables=[
            Syllable(onset=["t", "r"], nucleus="a", coda=["n"]),
            Syllable(onset=["d"], nucleus="oh", coda=[], stressed=True),
            Syllable(onset=["l"], nucleus="uh", coda=[]),
            Syllable(onset=["p", "r"], nucleus="ih", coda=["l"]),
        ],
        citation="MedlinePlus drug information: Trandolapril ('tran DOE la pril')",
    ),
    "zolmitriptan": ManualPronunciation(
        reference="zohl mi TRIP tan",
        syllables=[
            Syllable(onset=["z"], nucleus="oh", coda=["l"]),
            Syllable(onset=["m"], nucleus="ih", coda=[]),
            Syllable(onset=["t", "r"], nucleus="ih", coda=["p"], stressed=True),
            Syllable(onset=["t"], nucleus="a", coda=["n"]),
        ],
        citation="MedlinePlus drug information: Zolmitriptan ('zohl mi TRIP tan')",
    ),
}


# ---------------------------------------------------------------------------
# Provenance / manifest models
# ---------------------------------------------------------------------------


class RawSource(BaseModelNoExtra):
    """One raw upstream dataset, as retrieved."""

    url: Annotated[str, Field(description="Canonical source URL.")]
    mirror_url: Annotated[
        Optional[str],
        Field(
            default=None,
            description=(
                "URL actually fetched when the canonical host was unreachable "
                "(e.g. a Wayback Machine snapshot of the same file)."
            ),
        ),
    ]
    retrieved: Annotated[str, Field(description="Retrieval date, YYYY-MM-DD.")]
    sha256: Annotated[str, Field(description="sha256 of the raw zip.")]


class ExtractParams(BaseModelNoExtra):
    """The pinned extraction filters (mirrors the module constants)."""

    cohort_start: int
    cohort_end: int
    given_extract_floor: int
    surname_head_max_rank: int
    surname_tail_min_count: int
    surname_tail_max_count: int


class ExtractProvenance(BaseModelNoExtra):
    """Provenance for the checked-in extracts (CODE_DESIGN artifacts rule)."""

    extractor_version: str
    ssa: RawSource
    census: RawSource
    cmudict: RawSource
    wikipron: RawSource
    wordlist: RawSource
    params: ExtractParams
    extract_shas: Annotated[
        dict[str, str], Field(description="sha256 per extract file.")
    ]


class BuildRules(BaseModelNoExtra):
    """The pinned build thresholds (mirrors the module constants)."""

    gender_dominance_min: float
    easy_given_min_total: int
    easy_given_min_per_decade: int
    hard_given_min_total: int
    hard_given_max_total: int
    hard_given_min_decades: int
    hard_given_decade_floor: int
    easy_surname_max_rank: int
    hard_name_min_length: int
    denylist: Annotated[
        list[str],
        Field(description="The fixed place/brand denylist the pools ran under."),
    ]


class PoolStats(BaseModelNoExtra):
    """Hard-tier pool sizes before and after the lexicon-grounding filter."""

    hard_given_band: Annotated[
        int, Field(description="Hard given names in the rarity band (pre-filter).")
    ]
    hard_given_admissible: Annotated[
        int, Field(description="...that are lexicon-covered, non-word, len>=5.")
    ]
    tail_surname_band: Annotated[
        int, Field(description="Tail surnames in the census band (pre-filter).")
    ]
    tail_surname_admissible: Annotated[
        int, Field(description="...that are lexicon-covered, non-word, len>=5.")
    ]


class NameBanksManifest(BaseModelNoExtra):
    """What one ``tau2 intake-names build`` run produced."""

    builder_version: str
    seed: int
    extract_shas: Annotated[
        dict[str, str],
        Field(description="sha256 of the extracts the banks were built from."),
    ]
    rules: BuildRules
    pool_stats: PoolStats
    pronunciation_sources: Annotated[
        dict[str, int],
        Field(
            description=(
                "Token count per pronunciation source over both column-"
                "bearing banks (person_names + medications)."
            )
        ),
    ]
    operator_counts: Annotated[
        dict[str, dict[str, int]],
        Field(
            description=(
                "Per-bank distortion-operator histogram of the balance-"
                "greedy mispronunciation draw (design doc §4)."
            )
        ),
    ]
    person_count: int
    constructed_surnames: Annotated[
        list[str],
        Field(
            description=(
                "Hard-tier surnames that are hyphenated constructions of two "
                "real rare census surnames (not themselves census rows)."
            )
        ),
    ]
    given_name_genders: Annotated[
        dict[str, str],
        Field(
            description=(
                "Normalized given name -> gender for every person VALUE; "
                "localized extensions can consume this map without "
                "re-inferring names."
            )
        ),
    ]


class ExtractOutcome(BaseModelNoExtra):
    """Files one extract run wrote (empty when already up to date)."""

    sources_dir: Path
    written: list[Path]


class BuildOutcome(BaseModelNoExtra):
    """Files one build run wrote (empty when already up to date)."""

    seed: int
    banks_dir: Path
    written: list[Path]
    sample_easy: list[str]
    sample_hard: list[str]
    pool_stats: PoolStats
    pronunciation_sources: dict[str, int]
    operator_counts: dict[str, dict[str, int]]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_if_changed(path: Path, text: str, written: list[Path]) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    written.append(path)


def extract_name_sources(
    ssa_zip: Path,
    census_zip: Path,
    cmudict: Path,
    wikipron: Path,
    wordlist: Path,
    retrieved: str,
    ssa_mirror_url: Optional[str] = None,
    census_mirror_url: Optional[str] = None,
    sources_dir: Optional[Path] = None,
    banks_dir: Optional[Path] = None,
) -> ExtractOutcome:
    """Build the checked-in extracts from the five raw datasets."""
    from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR

    sources_dir = Path(sources_dir) if sources_dir is not None else NAME_SOURCES_DIR
    banks_dir = Path(banks_dir) if banks_dir is not None else INTAKE_BANKS_DIR
    written: list[Path] = []

    given_csv = _extract_given_names(Path(ssa_zip))
    surname_csv = _extract_surnames(Path(census_zip))
    relevant = _relevant_words(given_csv, surname_csv, banks_dir)
    pronunciations_csv = _extract_pronunciations(
        Path(cmudict), Path(wikipron), relevant
    )
    wordlist_csv = _extract_wordlist_hits(Path(wordlist), relevant)
    _write_if_changed(sources_dir / "ssa_given_names.csv", given_csv, written)
    _write_if_changed(sources_dir / "census_surnames.csv", surname_csv, written)
    _write_if_changed(sources_dir / "pronunciations.csv", pronunciations_csv, written)
    _write_if_changed(sources_dir / "wordlist_hits.csv", wordlist_csv, written)

    def _raw(url: str, path: Path, mirror: Optional[str] = None) -> RawSource:
        return RawSource(
            url=url,
            mirror_url=mirror,
            retrieved=retrieved,
            sha256=_sha256_path(path),
        )

    extracts = {
        "ssa_given_names.csv": given_csv,
        "census_surnames.csv": surname_csv,
        "pronunciations.csv": pronunciations_csv,
        "wordlist_hits.csv": wordlist_csv,
    }
    provenance = ExtractProvenance(
        extractor_version=EXTRACTOR_VERSION,
        ssa=_raw(SSA_SOURCE_URL, Path(ssa_zip), ssa_mirror_url),
        census=_raw(CENSUS_SOURCE_URL, Path(census_zip), census_mirror_url),
        cmudict=_raw(CMUDICT_SOURCE_URL, Path(cmudict)),
        wikipron=_raw(WIKIPRON_SOURCE_URL, Path(wikipron)),
        wordlist=_raw(WORDLIST_SOURCE_URL, Path(wordlist)),
        params=ExtractParams(
            cohort_start=COHORT_START,
            cohort_end=COHORT_END,
            given_extract_floor=GIVEN_EXTRACT_FLOOR,
            surname_head_max_rank=SURNAME_HEAD_MAX_RANK,
            surname_tail_min_count=SURNAME_TAIL_MIN_COUNT,
            surname_tail_max_count=SURNAME_TAIL_MAX_COUNT,
        ),
        extract_shas={
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in extracts.items()
        },
    )
    _write_if_changed(
        sources_dir / "provenance.json",
        json.dumps(provenance.model_dump(mode="json"), indent=2) + "\n",
        written,
    )
    logger.info(
        f"intake-names extract: {sources_dir} written={len(written)} "
        f"(given + surname + pronunciation + wordlist rows per the pinned filters)"
    )
    return ExtractOutcome(sources_dir=sources_dir, written=written)


def _relevant_words(given_csv: str, surname_csv: str, banks_dir: Path) -> set[str]:
    """Every lowercase word the builder may look up: extract given names,
    extract surnames (raw and restored casing — apostrophe forms are the
    lexicon keys), and the medication bank's drug tokens."""
    relevant: set[str] = set()
    for row in csv.DictReader(io.StringIO(given_csv)):
        relevant.add(row["name"].lower())
    for row in csv.DictReader(io.StringIO(surname_csv)):
        relevant.add(row["name"].lower())
        relevant.add(restore_surname_casing(row["name"]).lower())
    medications = yaml.safe_load((banks_dir / "medications.yaml").read_text())
    for entry in medications["entries"]:
        for token in medication_drug_tokens(entry["value"]):
            relevant.add(token.lower())
    return relevant


def _extract_pronunciations(cmudict: Path, wikipron: Path, relevant: set[str]) -> str:
    """CMUdict + WikiPron rows for the relevant words: word, source, phonemes.

    CMUdict variant entries (``word(2)``) are dropped; the first row per
    (word, source) wins. Sorted for byte-stable output.
    """
    rows: dict[tuple[str, str], str] = {}
    variant = re.compile(r"\(\d+\)$")
    with cmudict.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            word = parts[0]
            if variant.search(word):
                continue
            key = (word.lower(), PronunciationSource.CMUDICT.value)
            if key[0] in relevant and key not in rows:
                rows[key] = " ".join(parts[1:])
    with wikipron.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            word, phonemes = line.split("\t")
            key = (word.lower(), PronunciationSource.WIKIPRON.value)
            if key[0] in relevant and key not in rows:
                rows[key] = phonemes
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["word", "source", "phonemes"])
    for (word, source), phonemes in sorted(rows.items()):
        writer.writerow([word, source, phonemes])
    return out.getvalue()


def _extract_wordlist_hits(wordlist: Path, relevant: set[str]) -> str:
    """The wordlist rows a relevant name (or its singular-stripped form)
    collides with. Only all-lowercase wordlist lines count: capitalized web2
    rows are proper nouns — the very leakage the filter kills."""
    queries = set(relevant)
    queries.update(word[:-1] for word in relevant if word.endswith("s"))
    hits: set[str] = set()
    with wordlist.open(encoding="utf-8") as handle:
        for line in handle:
            word = line.strip()
            if word and word == word.lower() and word in queries:
                hits.add(word)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["word"])
    for word in sorted(hits):
        writer.writerow([word])
    return out.getvalue()


def _extract_given_names(ssa_zip: Path) -> str:
    """SSA yob files -> per-(name, sex) cohort-window decade counts.

    A name qualifies when its larger-sex window total reaches
    ``GIVEN_EXTRACT_FLOOR``; BOTH sex rows of a qualifying name are kept
    (however small) so the builder computes true sex-dominance ratios.
    """
    counts: dict[tuple[str, str], dict[int, int]] = {}
    with zipfile.ZipFile(ssa_zip) as archive:
        for year in range(COHORT_START, COHORT_END + 1):
            member = f"yob{year}.txt"
            try:
                raw = archive.read(member)
            except KeyError as exc:
                raise NameBanksError(f"SSA zip {ssa_zip} is missing {member}") from exc
            decade = (year // 10) * 10
            for line in io.TextIOWrapper(io.BytesIO(raw), encoding="ascii"):
                line = line.strip()
                if not line:
                    continue
                name, sex, count = line.split(",")
                per_decade = counts.setdefault((name, sex), {})
                per_decade[decade] = per_decade.get(decade, 0) + int(count)

    totals_by_name: dict[str, int] = {}
    for (name, _), per_decade in counts.items():
        totals_by_name[name] = max(
            totals_by_name.get(name, 0), sum(per_decade.values())
        )
    qualifying = {
        name
        for name, largest in totals_by_name.items()
        if largest >= GIVEN_EXTRACT_FLOOR
    }

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        ["name", "sex", "total"] + [f"d{decade}" for decade in COHORT_DECADES]
    )
    for (name, sex), per_decade in sorted(counts.items()):
        if name not in qualifying:
            continue
        writer.writerow(
            [name, sex, sum(per_decade.values())]
            + [per_decade.get(decade, 0) for decade in COHORT_DECADES]
        )
    return out.getvalue()


def _extract_surnames(census_zip: Path) -> str:
    """Census 2010 surnames -> the head (by rank) plus the rare tail."""
    with zipfile.ZipFile(census_zip) as archive:
        try:
            raw = archive.read("Names_2010Census.csv")
        except KeyError as exc:
            raise NameBanksError(
                f"Census zip {census_zip} is missing Names_2010Census.csv"
            ) from exc
    kept: list[tuple[int, str, int]] = []
    for row in csv.DictReader(io.TextIOWrapper(io.BytesIO(raw), encoding="ascii")):
        name = row["name"]
        rank = int(row["rank"])
        count = int(row["count"])
        if rank < 1 or not name.isalpha():  # drops the ALL OTHER NAMES row
            continue
        head = rank <= SURNAME_HEAD_MAX_RANK
        tail = SURNAME_TAIL_MIN_COUNT <= count <= SURNAME_TAIL_MAX_COUNT
        if head or tail:
            kept.append((rank, name, count))
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["name", "rank", "count"])
    for rank, name, count in sorted(kept):
        writer.writerow([name, rank, count])
    return out.getvalue()


# ---------------------------------------------------------------------------
# Casing restoration (ASCII only)
# ---------------------------------------------------------------------------


def restore_surname_casing(upper: str) -> str:
    """Natural ASCII casing for an uppercase census surname.

    Rules, in order: the closed apostrophe table; ``MC`` + >= 3 letters ->
    ``Mc`` + Title case (essentially every MC-prefixed census surname is a
    Mc-name; MAC is NOT auto-cased — Macias/Machado would break); otherwise
    plain Title case. Never emits a non-ASCII character.
    """
    if upper in APOSTROPHE_SURNAMES:
        return APOSTROPHE_SURNAMES[upper]
    if upper.startswith("MC") and len(upper) >= 5:
        return "Mc" + upper[2:].capitalize()
    return upper.capitalize()


# ---------------------------------------------------------------------------
# Build: lexicon index (coverage filter + pronunciation lookups)
# ---------------------------------------------------------------------------


class _LexiconIndex:
    """The pronunciation + wordlist extracts, indexed for the build.

    ``covered``/``hard_admissible`` implement the design §2.2 filter;
    ``require`` returns (source, phonemes) under a per-bank source
    preference, falling back to the manual medication table.
    """

    def __init__(self, sources_dir: Path) -> None:
        self.phonemes: dict[str, dict[PronunciationSource, str]] = {}
        with (sources_dir / "pronunciations.csv").open() as handle:
            for row in csv.DictReader(handle):
                per_source = self.phonemes.setdefault(row["word"], {})
                per_source[PronunciationSource(row["source"])] = row["phonemes"]
        with (sources_dir / "wordlist_hits.csv").open() as handle:
            self.word_hits = {row["word"] for row in csv.DictReader(handle)}

    def covered(self, token: str) -> bool:
        return token.lower() in self.phonemes

    def display_covered(self, display: str) -> bool:
        """Every pronunciation-bearing token of a display string is covered."""
        return all(self.covered(token) for token in person_name_tokens(display))

    def wordlike(self, token: str) -> bool:
        lowered = token.lower()
        if lowered in self.word_hits:
            return True
        return lowered.endswith("s") and lowered[:-1] in self.word_hits

    def hard_admissible(self, token: str) -> bool:
        """The design §2.2 hard-tier value pool filter for one name token.

        The place/brand denylist is deliberately NOT part of the pool filter:
        it applies at draw time (:meth:`denied`), so extending it never
        reshuffles the seeded pool order.
        """
        return (
            len(token) >= HARD_NAME_MIN_LENGTH
            and self.covered(token)
            and not self.wordlike(token)
        )

    @staticmethod
    def denied(token: str) -> bool:
        return token.lower() in PRONUNCIATION_DENYLIST

    def lookup(
        self, token: str, preference: tuple[PronunciationSource, ...]
    ) -> Optional[tuple[PronunciationSource, str]]:
        per_source = self.phonemes.get(token.lower(), {})
        for source in preference:
            if source in per_source:
                return source, per_source[source]
        return None


def _token_pronunciation(
    index: _LexiconIndex,
    token: str,
    preference: tuple[PronunciationSource, ...],
    drawer: Optional[MispronunciationDrawer] = None,
    decoy_tokens: tuple[str, ...] = (),
    mispronounce_rng: Optional[random.Random] = None,
) -> TokenPronunciation:
    """The full pronunciation columns for one value token.

    Lexicon-covered tokens convert through the fixed converters; medication
    tokens absent from both lexicons come from the cited manual table. A
    non-None ``drawer`` draws the distortion operator (per-bank
    balance-greedy, feasibility-gated, ties broken by ``mispronounce_rng``)
    and renders the ``mispronounced`` variant.
    """
    hit = index.lookup(token, preference)
    if hit is not None:
        source, phonemes = hit
        scheme = "arpabet" if source is PronunciationSource.CMUDICT else "ipa"
        syllables = syllabify_phonemes(phonemes, scheme)
        citation = None
    else:
        manual = MANUAL_MEDICATION_PRONUNCIATIONS.get(token.lower())
        if manual is None:
            raise NameBanksError(
                f"Token {token!r} has no lexicon pronunciation and no manual "
                "entry — extend MANUAL_MEDICATION_PRONUNCIATIONS or re-run "
                "`tau2 intake-names extract`."
            )
        source = PronunciationSource.MANUAL
        phonemes = manual.reference
        syllables = manual.syllables
        citation = manual.citation
    mispronounced = None
    operator = None
    if drawer is not None:
        if mispronounce_rng is None:
            raise NameBanksError(f"Token {token!r}: drawer requires its rng stream")
        drawn = drawer.draw(token, syllables, decoy_tokens, mispronounce_rng)
        if drawn is not None:  # None: no non-vacuous natural variant exists
            operator, mispronounced = drawn
    return TokenPronunciation(
        token=token,
        source=source,
        phonemes=phonemes,
        respelling=render_respelling(syllables),
        mispronounced=mispronounced,
        operator=operator,
        citation=citation,
    )


def _named_rng(*parts) -> random.Random:
    """A named rng stream (``intake-pron|...|{seed}``), independent of every
    other draw so pronunciation variants never perturb the name draw."""
    return random.Random("|".join(str(part) for part in parts))


# ---------------------------------------------------------------------------
# Build: pools
# ---------------------------------------------------------------------------


class _GivenName(BaseModelNoExtra):
    name: str
    gender: Gender
    total: int


class _Surname(BaseModelNoExtra):
    display: str  # restored casing
    rank: int
    count: int


def _normalize_key(name: str) -> str:
    """Mirror of name_genders._normalize_given_name for catalog keys."""
    lowered = name.strip().split()[0].lower() if name.strip() else ""
    stripped = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z]", "", stripped)


def _load_given_pools(
    sources_dir: Path, index: _LexiconIndex
) -> tuple[list[_GivenName], list[_GivenName], int]:
    """(easy pool, hard pool, hard band size pre-filter), each
    sex-balanced-eligible and dominance-safe, ordered by (total desc, name)
    for deterministic consumption.

    Every VALUE pool requires lexicon coverage (design §2.2): the hard pool
    additionally runs the full non-word / length / denylist filter; the easy
    pool only skips the rare top-band name without an attested pronunciation.
    """
    rows: dict[tuple[str, str], dict] = {}
    with (sources_dir / "ssa_given_names.csv").open() as handle:
        for row in csv.DictReader(handle):
            rows[(row["name"], row["sex"])] = {
                "total": int(row["total"]),
                "decades": [int(row[f"d{d}"]) for d in COHORT_DECADES],
            }
    by_name: dict[str, dict[str, dict]] = {}
    for (name, sex), data in rows.items():
        by_name.setdefault(name, {})[sex] = data

    easy: list[_GivenName] = []
    hard: list[_GivenName] = []
    hard_band = 0
    for name, per_sex in by_name.items():
        male = per_sex.get("M", {}).get("total", 0)
        female = per_sex.get("F", {}).get("total", 0)
        dominant_sex = "M" if male >= female else "F"
        dominance = max(male, female) / (male + female)
        if dominance < GENDER_DOMINANCE_MIN:
            continue
        if _normalize_key(name) in EXCLUDED_GIVEN_KEYS:
            continue
        data = per_sex[dominant_sex]
        total = data["total"]
        decades = data["decades"]
        gender = Gender.MALE if dominant_sex == "M" else Gender.FEMALE
        entry = _GivenName(name=name, gender=gender, total=total)
        if (
            total >= EASY_GIVEN_MIN_TOTAL
            and all(count >= EASY_GIVEN_MIN_PER_DECADE for count in decades)
            and index.covered(name)
        ):
            easy.append(entry)
        if (
            HARD_GIVEN_MIN_TOTAL <= total <= HARD_GIVEN_MAX_TOTAL
            and sum(1 for count in decades if count >= HARD_GIVEN_DECADE_FLOOR)
            >= HARD_GIVEN_MIN_DECADES
        ):
            hard_band += 1
            if index.hard_admissible(name):
                hard.append(entry)
    key = lambda g: (-g.total, g.name)  # noqa: E731
    return sorted(easy, key=key), sorted(hard, key=key), hard_band


def _load_surname_pools(
    sources_dir: Path, index: _LexiconIndex
) -> tuple[list[_Surname], list[_Surname], list[_Surname], list[_Surname]]:
    """(easy value pool: rank <= EASY_SURNAME_MAX_RANK and covered, easy
    decoy candidates: the whole head, admissible tail value pool, full tail
    decoy candidates), each rank-ordered.

    Value pools are lexicon-covered (design §2.2 for the tail; the easy pool
    only skips the rare uncovered top-band name). Decoy candidate pools stay
    unfiltered — decoys are text-confusable neighbors, not spoken values.
    """
    head: list[_Surname] = []
    tail: list[_Surname] = []
    with (sources_dir / "census_surnames.csv").open() as handle:
        for row in csv.DictReader(handle):
            surname = _Surname(
                display=restore_surname_casing(row["name"]),
                rank=int(row["rank"]),
                count=int(row["count"]),
            )
            if surname.rank <= SURNAME_HEAD_MAX_RANK:
                head.append(surname)
            else:
                tail.append(surname)
    head.sort(key=lambda s: s.rank)
    tail.sort(key=lambda s: (s.rank, s.display))
    easy = [
        s for s in head if s.rank <= EASY_SURNAME_MAX_RANK and index.covered(s.display)
    ]
    tail_admissible = [s for s in tail if index.hard_admissible(s.display)]
    return easy, head, tail_admissible, tail


# ---------------------------------------------------------------------------
# Build: neighbor search (real near-collision decoys)
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ch_a != ch_b),
                )
            )
        previous = current
    return previous[-1]


def _common_prefix_len(a: str, b: str) -> int:
    length = 0
    for ch_a, ch_b in zip(a, b):
        if ch_a != ch_b:
            break
        length += 1
    return length


def _closest_neighbor(
    target: str,
    candidates: list[str],
    reject: Callable[[str], bool],
) -> str:
    """The candidate most confusable with ``target``: smallest edit distance,
    then longest common prefix, then alphabetical — skipping rejects.

    Scores are computed first (cheap) and rejects — which scan the accepted
    display space — are only evaluated walking the ranking, so the usual case
    costs one reject call.
    """
    target_lower = target.lower()
    target_fold = fold_name(target)
    scored = sorted(
        (
            _levenshtein(candidate.lower(), target_lower),
            -_common_prefix_len(candidate.lower(), target_lower),
            candidate,
        )
        for candidate in set(candidates)
        if candidate != target
    )
    for _, _, candidate in scored:
        if fold_name(candidate) == target_fold:
            continue
        if reject(candidate):
            continue
        return candidate
    raise NameBanksError(
        f"No confusable neighbor available for {target!r} — the extract "
        "pools are too small for the pinned rules."
    )


def _prefix_bucket(candidates: list[str], target: str) -> list[str]:
    """Cheap candidate pre-filter for large pools: same first two letters,
    widening to first letter when that leaves nothing."""
    two = [c for c in candidates if c[:2].lower() == target[:2].lower()]
    if len(two) > 1:
        return two
    return [c for c in candidates if c[:1].lower() == target[:1].lower()]


# ---------------------------------------------------------------------------
# Build: display-space guard (fold uniqueness + containment freedom)
# ---------------------------------------------------------------------------


class _DisplayGuard:
    """Every accepted display string must be containment-free against every
    other (raw and folded): the generator's capture-absence assert compares
    raw substrings against the task DB view, and the localization registry
    rejects folded nesting — this guard makes both hold by construction."""

    def __init__(self) -> None:
        self._raw: list[str] = []
        self._folded: list[str] = []

    def conflicts(self, candidate: str) -> bool:
        folded = fold_name(candidate)
        for raw, fold in zip(self._raw, self._folded):
            if candidate in raw or raw in candidate:
                return True
            if folded in fold or fold in folded:
                return True
        return False

    def add(self, candidate: str) -> None:
        if self.conflicts(candidate):
            raise NameBanksError(
                f"Display string {candidate!r} collides with an accepted name"
            )
        self._raw.append(candidate)
        self._folded.append(fold_name(candidate))


E = Difficulty.EASY
H = Difficulty.HARD


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class _PersonRow(BaseModelNoExtra):
    value: str
    difficulty: Difficulty
    gender: Gender
    decoys: list[str]
    alt_name: Optional[str]
    pronunciations: list[TokenPronunciation]


class _Builder:
    def __init__(self, sources_dir: Path, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.guard = _DisplayGuard()
        for legacy in LEGACY_DISPLAY_NAMES:
            self.guard.add(legacy)
        self.index = _LexiconIndex(sources_dir)
        easy_given, hard_given, hard_given_band = _load_given_pools(
            sources_dir, self.index
        )
        self.given_pools = {E: easy_given, H: hard_given}
        easy_sur, head_sur, tail_admissible, tail_sur = _load_surname_pools(
            sources_dir, self.index
        )
        self.head_surnames = head_sur
        self.tail_surnames = tail_sur  # decoy candidates: the full band
        self.pool_stats = PoolStats(
            hard_given_band=hard_given_band,
            hard_given_admissible=len(hard_given),
            tail_surname_band=len(tail_sur),
            tail_surname_admissible=len(tail_admissible),
        )
        if len(hard_given) < HARD_GIVEN_POOL_FLOOR:
            raise NameBanksError(
                f"Admissible hard given-name pool is {len(hard_given)} "
                f"(< {HARD_GIVEN_POOL_FLOOR}) — the coverage filter or the "
                "pronunciation extract is broken."
            )
        if len(tail_admissible) < TAIL_SURNAME_POOL_FLOOR:
            raise NameBanksError(
                f"Admissible tail-surname pool is {len(tail_admissible)} "
                f"(< {TAIL_SURNAME_POOL_FLOOR}) — the coverage filter or the "
                "pronunciation extract is broken."
            )
        # All seeded shuffles happen HERE, in one fixed order, so pool order
        # — and therefore every draw downstream — is a pure function of seed:
        # - hard given names: a draw over the whole admissible rare band
        #   (popularity order would always pick the same near-cap names);
        # - person surname orders: pairing given<->surname is a draw, not an
        #   artifact of census rank order. VALUE surnames come from the
        #   coverage-filtered pools; decoys keep the full band.
        self.rng.shuffle(self.given_pools[H])
        self.person_surname_order = {
            E: self._shuffled([s for s in easy_sur]),
            H: self._shuffled([s for s in tail_admissible]),
        }
        self.used_given_keys: set[str] = set()
        self.used_surnames: set[str] = set()
        self.value_folds: set[str] = set()
        self.constructed_surnames: list[str] = []
        self.given_name_genders: dict[str, str] = {}
        # One drawer for the whole bank: balance-greedy counts run across
        # entries in build order (a pure function of the seed).
        self.mispronounce_drawer = MispronunciationDrawer("person_names")

    def _shuffled(self, pool: list["_Surname"]) -> list["_Surname"]:
        self.rng.shuffle(pool)
        return pool

    # -- pools ----------------------------------------------------------

    def take_given(self, tier: Difficulty, gender: Gender) -> str:
        """The next unused, non-denied given name of the tier/gender, in pool
        order (popularity for easy; the seeded shuffle for hard, see build())."""
        for entry in self.given_pools[tier]:
            key = _normalize_key(entry.name)
            if entry.gender is not gender or key in self.used_given_keys:
                continue
            if self.index.denied(entry.name):
                continue
            self.used_given_keys.add(key)
            self.given_name_genders[key] = entry.gender.value
            return entry.name
        raise NameBanksError(f"Given-name pool exhausted: {tier.value}/{gender.value}")

    def take_surname(self, pool: list[_Surname], reject: Callable[[str], bool]) -> str:
        for surname in pool:
            if surname.display in self.used_surnames:
                continue
            if self.index.denied(surname.display):
                continue
            if reject(surname.display):
                continue
            self.used_surnames.add(surname.display)
            return surname.display
        raise NameBanksError("Surname pool exhausted under the pinned rules")

    def take_constructed_surname(
        self, pool: list[_Surname], reject: Callable[[str], bool]
    ) -> str:
        """A hyphenated construction of two unused rare surnames."""
        for first in pool:
            if first.display in self.used_surnames:
                continue
            if self.index.denied(first.display):
                continue
            for second in pool:
                if second.display in self.used_surnames:
                    continue
                if second.display == first.display:
                    continue
                if self.index.denied(second.display):
                    continue
                combined = f"{first.display}-{second.display}"
                if reject(combined):
                    continue
                self.used_surnames.add(first.display)
                self.used_surnames.add(second.display)
                self.constructed_surnames.append(combined)
                return combined
        raise NameBanksError("Rare surname pool exhausted for constructions")

    # -- decoys ----------------------------------------------------------

    def surname_decoy(self, given: str, surname: str, tier: Difficulty) -> str:
        """decoys[0]: same given name, closest real neighbor surname."""
        if "-" in surname:
            # Constructed surname: keep the first component, swap the second
            # for its closest rare neighbor (a shared-prefix near-collision).
            first, second = surname.split("-", 1)
            candidates = _prefix_bucket([s.display for s in self.tail_surnames], second)
            neighbor = _closest_neighbor(
                second,
                candidates,
                reject=lambda cand: cand == first
                or self.decoy_conflicts(f"{given} {first}-{cand}"),
            )
            return f"{given} {first}-{neighbor}"
        if tier is E:
            candidates = [s.display for s in self.head_surnames]
        else:
            candidates = _prefix_bucket(
                [s.display for s in self.tail_surnames], surname
            )
        neighbor = _closest_neighbor(
            surname,
            candidates,
            reject=lambda cand: self.decoy_conflicts(f"{given} {cand}"),
        )
        return f"{given} {neighbor}"

    def given_decoy(self, given: str, surname: str, tier: Difficulty) -> str:
        """decoys[1]: same surname, closest real neighbor given name."""
        candidates = [g.name for g in self.given_pools[tier]]
        neighbor = _closest_neighbor(
            given,
            candidates,
            reject=lambda cand: self.decoy_conflicts(f"{cand} {surname}"),
        )
        return f"{neighbor} {surname}"

    def decoy_conflicts(self, decoy: str) -> bool:
        """A decoy may not fold onto any value and must stay containment-free
        (it is planted verbatim as a salt record in the task DB view)."""
        return fold_name(decoy) in self.value_folds or self.guard.conflicts(decoy)

    # -- entries ----------------------------------------------------------

    def accept_value(self, value: str) -> None:
        self.guard.add(value)
        self.value_folds.add(fold_name(value))

    def build_person(
        self,
        tier: Difficulty,
        gender: Gender,
        surname_source: Callable[[Callable[[str], bool]], str],
        with_alt: bool,
    ) -> _PersonRow:
        given = self.take_given(tier, gender)
        surname = surname_source(lambda cand: self.guard.conflicts(f"{given} {cand}"))
        value = f"{given} {surname}"
        self.accept_value(value)
        decoy_surname = self.surname_decoy(given, surname, tier)
        self.guard.add(decoy_surname)
        decoy_given = self.given_decoy(given, surname, tier)
        self.guard.add(decoy_given)
        alt_name: Optional[str] = None
        if with_alt:
            alt_surname = self.take_surname(
                self.person_surname_order[tier],
                reject=lambda cand: self.guard.conflicts(f"{given} {cand}"),
            )
            alt_name = f"{given} {alt_surname}"
            self.guard.add(alt_name)
        decoys = [decoy_surname, decoy_given]
        return _PersonRow(
            value=value,
            difficulty=tier,
            gender=gender,
            decoys=decoys,
            alt_name=alt_name,
            pronunciations=self._person_pronunciations(value, tier, decoys),
        )

    def _person_pronunciations(
        self, value: str, tier: Difficulty, decoys: list[str]
    ) -> list[TokenPronunciation]:
        """Design §2.4 columns for every value token; hard-tier tokens draw a
        mispronounced variant via the bank's balance-greedy drawer, ties
        broken by an independent named rng stream per token (the name draw is
        never perturbed by pronunciation work)."""
        decoy_tokens = tuple(
            token for decoy in decoys for token in person_name_tokens(decoy)
        )
        columns: list[TokenPronunciation] = []
        for token in person_name_tokens(value):
            drawer = None
            mispronounce_rng = None
            if tier is H:
                drawer = self.mispronounce_drawer
                mispronounce_rng = _named_rng(
                    "intake-pron", "operator", "person_names", value, token, self.seed
                )
            columns.append(
                _token_pronunciation(
                    self.index,
                    token,
                    PERSON_SOURCE_PREFERENCE,
                    drawer=drawer,
                    decoy_tokens=decoy_tokens,
                    mispronounce_rng=mispronounce_rng,
                )
            )
        return columns


def _assert_ascii(rows: list[str]) -> None:
    offenders = sorted({row for row in rows if not row.isascii()})
    if offenders:
        raise NameBanksError(
            f"Non-ASCII display strings generated: {', '.join(offenders[:5])}"
        )


def build_name_banks(
    seed: int = DEFAULT_BUILD_SEED,
    banks_dir: Optional[Path] = None,
    sources_dir: Optional[Path] = None,
) -> BuildOutcome:
    """Rebuild person_names.yaml from the extracts."""
    from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR

    banks_dir = Path(banks_dir) if banks_dir is not None else INTAKE_BANKS_DIR
    sources_dir = Path(sources_dir) if sources_dir is not None else NAME_SOURCES_DIR

    provenance = ExtractProvenance.model_validate(
        json.loads((sources_dir / "provenance.json").read_text())
    )
    for filename, expected in provenance.extract_shas.items():
        actual = _sha256_path(sources_dir / filename)
        if actual != expected:
            raise NameBanksError(
                f"Extract {filename} sha256 {actual} != provenance {expected} "
                "— re-run `tau2 intake-names extract` or restore the file."
            )

    builder = _Builder(sources_dir, seed)

    persons: list[_PersonRow] = []
    genders = (Gender.FEMALE, Gender.MALE)
    for index in range(PERSON_COUNT_PER_TIER):
        persons.append(
            builder.build_person(
                E,
                genders[index % 2],
                surname_source=lambda reject: builder.take_surname(
                    builder.person_surname_order[E], reject
                ),
                with_alt=index % 2 == 0,
            )
        )
    for index in range(PERSON_COUNT_PER_TIER):
        # Even rows carry a single rare-tail surname; odd rows carry a
        # constructed hyphenation of two rare-tail surnames (flagged in the
        # manifest) so half the hard tier exercises the hyphen fold.
        if index % 2 == 0:
            surname_source = lambda reject: builder.take_surname(  # noqa: E731
                builder.person_surname_order[H], reject
            )
        else:
            surname_source = lambda reject: builder.take_constructed_surname(  # noqa: E731
                builder.person_surname_order[H], reject
            )
        persons.append(
            builder.build_person(
                H,
                genders[index % 2],
                surname_source=surname_source,
                with_alt=index % 2 == 0,
            )
        )

    display_strings = (
        [p.value for p in persons]
        + [d for p in persons for d in p.decoys]
        + [p.alt_name for p in persons if p.alt_name]
    )
    _assert_ascii(display_strings)
    _assert_email_wellformed(persons, banks_dir)

    written: list[Path] = []
    _write_if_changed(banks_dir / "person_names.yaml", _person_yaml(persons), written)
    medication_pronunciations, medication_operators = (
        _rebuild_medication_pronunciations(banks_dir, builder.index, seed, written)
    )
    builder.mispronounce_drawer.assert_pins_consumed()
    operator_counts = {
        "person_names": builder.mispronounce_drawer.histogram(),
        "medications": medication_operators,
    }

    source_counts: dict[str, int] = {source.value: 0 for source in PronunciationSource}
    for columns in [p.pronunciations for p in persons] + medication_pronunciations:
        for column in columns:
            source_counts[column.source.value] += 1

    manifest = NameBanksManifest(
        builder_version=BUILDER_VERSION,
        seed=seed,
        extract_shas=provenance.extract_shas,
        rules=BuildRules(
            gender_dominance_min=GENDER_DOMINANCE_MIN,
            easy_given_min_total=EASY_GIVEN_MIN_TOTAL,
            easy_given_min_per_decade=EASY_GIVEN_MIN_PER_DECADE,
            hard_given_min_total=HARD_GIVEN_MIN_TOTAL,
            hard_given_max_total=HARD_GIVEN_MAX_TOTAL,
            hard_given_min_decades=HARD_GIVEN_MIN_DECADES,
            hard_given_decade_floor=HARD_GIVEN_DECADE_FLOOR,
            easy_surname_max_rank=EASY_SURNAME_MAX_RANK,
            hard_name_min_length=HARD_NAME_MIN_LENGTH,
            denylist=sorted(PRONUNCIATION_DENYLIST),
        ),
        pool_stats=builder.pool_stats,
        pronunciation_sources=source_counts,
        operator_counts=operator_counts,
        person_count=len(persons),
        constructed_surnames=sorted(builder.constructed_surnames),
        given_name_genders=dict(sorted(builder.given_name_genders.items())),
    )
    _write_if_changed(
        sources_dir / "name_banks.manifest.json",
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        written,
    )

    # Loader-level validation of what was just written (fail loud, not ship).
    load_banks(banks_dir)

    sample_easy = [p.value for p in persons if p.difficulty is E][:5]
    sample_hard = [p.value for p in persons if p.difficulty is H][:5]
    logger.info(
        f"intake-names build: seed={seed} persons={len(persons)} written={len(written)}"
    )
    return BuildOutcome(
        seed=seed,
        banks_dir=banks_dir,
        written=written,
        sample_easy=sample_easy,
        sample_hard=sample_hard,
        pool_stats=builder.pool_stats,
        pronunciation_sources=source_counts,
        operator_counts=operator_counts,
    )


def _rebuild_medication_pronunciations(
    banks_dir: Path,
    index: _LexiconIndex,
    seed: int,
    written: list[Path],
) -> tuple[list[list[TokenPronunciation]], dict[str, int]]:
    """Regenerate the medications bank's pronunciation columns in place.

    Values, difficulties, and decoys are preserved verbatim; only the
    ``pronunciations`` columns are recomputed (drug-name tokens only, design
    §2.4; every token draws a mispronounced variant via the bank's
    balance-greedy drawer, design §4). Returns the per-entry columns and the
    bank's operator histogram for manifest accounting.
    """
    path = banks_dir / "medications.yaml"
    raw = yaml.safe_load(path.read_text())
    entries: list[dict] = []
    all_columns: list[list[TokenPronunciation]] = []
    drawer = MispronunciationDrawer("medications")
    for entry in raw["entries"]:
        value = entry["value"]
        decoy_tokens = tuple(
            token
            for decoy in entry.get("decoys") or []
            for token in person_name_tokens(decoy)
        )
        columns = []
        for token in medication_drug_tokens(value):
            rng = _named_rng(
                "intake-pron", "operator", "medications", value, token, seed
            )
            columns.append(
                _token_pronunciation(
                    index,
                    token,
                    MEDICATION_SOURCE_PREFERENCE,
                    drawer=drawer,
                    decoy_tokens=decoy_tokens,
                    mispronounce_rng=rng,
                )
            )
        all_columns.append(columns)
        rebuilt: dict = {"value": value, "difficulty": entry["difficulty"]}
        if entry.get("decoys"):
            rebuilt["decoys"] = entry["decoys"]
        rebuilt["pronunciations"] = [
            column.model_dump(mode="json", exclude_none=True) for column in columns
        ]
        entries.append(rebuilt)
    drawer.assert_pins_consumed()
    text = _yaml_text({"entity": "medications", "entries": entries})
    _write_if_changed(path, text, written)
    return all_columns, drawer.histogram()


def _assert_email_wellformed(persons: list[_PersonRow], banks_dir: Path) -> None:
    """Every value and alt name must instantiate every email pattern (the
    exhaustive property test_folds_intake asserts; cheaper to fail here).
    Reads ``emails.yaml`` from ``banks_dir`` — the same bank tree the build
    writes into and ``load_banks`` validates."""
    from tau2.domains.intake.tasks.banks import BankFile, EmailEntry
    from tau2.utils import load_file

    email_bank = BankFile[EmailEntry].model_validate(
        load_file(banks_dir / "emails.yaml")
    )
    owners = [p.value for p in persons] + [p.alt_name for p in persons if p.alt_name]
    for entry in email_bank.entries:
        for owner in owners:
            instantiate_email(entry, owner)  # raises when malformed


# ---------------------------------------------------------------------------
# YAML emission (matches the repo's bank style: block, insertion-ordered)
# ---------------------------------------------------------------------------


def _yaml_text(payload: dict) -> str:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000,
    )


def _person_yaml(persons: list[_PersonRow]) -> str:
    entries = []
    for person in persons:
        entry: dict = {
            "value": person.value,
            "difficulty": person.difficulty.value,
            "gender": person.gender.value,
            "decoys": person.decoys,
        }
        if person.alt_name:
            entry["alt_name"] = person.alt_name
        entry["pronunciations"] = [
            column.model_dump(mode="json", exclude_none=True)
            for column in person.pronunciations
        ]
        entries.append(entry)
    return _yaml_text({"entity": "person_names", "entries": entries})


# ---------------------------------------------------------------------------
# Bank review packet (tau2 intake-names packet)
# ---------------------------------------------------------------------------


class NameBankPacket(BaseModelNoExtra):
    """The rendered pronunciation review packet plus its summary counts."""

    person_tokens: Annotated[
        int, Field(description="Pronunciation-bearing person tokens.")
    ]
    medication_tokens: Annotated[
        int, Field(description="Pronunciation-bearing medication tokens.")
    ]
    per_source: Annotated[
        dict[str, int], Field(description="Token count per pronunciation source.")
    ]
    per_operator: Annotated[
        dict[str, dict[str, int]],
        Field(
            description=(
                "Mispronounced-variant count per distortion operator, keyed "
                "by bank (applicability is per bank, owner 2026-08-26)."
            )
        ),
    ]
    markdown: Annotated[
        str, Field(description="The full packet body (deterministic per build).")
    ]


def build_name_bank_packet(
    banks_dir: Optional[Path] = None, sources_dir: Optional[Path] = None
) -> NameBankPacket:
    """Render the owner review packet over the checked-in banks + manifest.

    One row per pronunciation-bearing token: (value, tier, token, source,
    phonemes, respelling, mispronounced, operator). Deterministic: pure
    function of the checked-in artifacts (design §6 gate 1).
    """
    from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR

    banks_dir = Path(banks_dir) if banks_dir is not None else INTAKE_BANKS_DIR
    sources_dir = Path(sources_dir) if sources_dir is not None else NAME_SOURCES_DIR
    banks = load_banks(banks_dir)
    manifest = NameBanksManifest.model_validate(
        json.loads((sources_dir / "name_banks.manifest.json").read_text())
    )

    per_source: dict[str, int] = {source.value: 0 for source in PronunciationSource}
    per_operator: dict[str, dict[str, int]] = {}
    person_tokens = 0
    medication_tokens = 0

    def _rows(entries, bank: str) -> list[str]:
        rows = []
        bank_operators = per_operator.setdefault(bank, {})
        for entry in entries:
            for column in entry.pronunciations:
                per_source[column.source.value] += 1
                if column.operator is not None:
                    bank_operators[column.operator.value] = (
                        bank_operators.get(column.operator.value, 0) + 1
                    )
                rows.append(
                    f"| {entry.value} | {entry.difficulty.value} | "
                    f"{column.token} | {column.source.value} | "
                    f"`{column.phonemes}` | {column.respelling} | "
                    f"{column.mispronounced or '-'} | "
                    f"{column.operator.value if column.operator else '-'} |"
                )
        return rows

    header = (
        "| value | tier | token | source | phonemes | respelling | "
        "mispronounced | operator |"
    )
    rule = "|---|---|---|---|---|---|---|---|"
    person_rows = _rows(banks.person_names, "person_names")
    person_tokens = len(person_rows)
    medication_rows = _rows(banks.medications, "medications")
    medication_tokens = len(medication_rows)

    lines: list[str] = []
    lines.append("# Intake name-bank pronunciation review packet")
    lines.append("")
    lines.append(
        f"Provenance: builder {manifest.builder_version}, seed {manifest.seed}. "
        "Regenerable byte-identically via `tau2 intake-names packet`; do not "
        "hand-edit."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- hard given-name pool: {manifest.pool_stats.hard_given_admissible} "
        f"admissible of {manifest.pool_stats.hard_given_band} in band"
    )
    lines.append(
        f"- tail surname pool: {manifest.pool_stats.tail_surname_admissible} "
        f"admissible of {manifest.pool_stats.tail_surname_band} in band"
    )
    lines.append(f"- person tokens: {person_tokens}")
    lines.append(f"- medication tokens: {medication_tokens}")
    for source, count in sorted(per_source.items()):
        lines.append(f"- source {source}: {count}")
    for bank, operators in sorted(per_operator.items()):
        for operator, count in sorted(operators.items()):
            lines.append(f"- {bank} operator {operator}: {count}")
    lines.append("")
    lines.append("## person_names")
    lines.append("")
    lines.append(header)
    lines.append(rule)
    lines.extend(person_rows)
    lines.append("")
    lines.append("## medications")
    lines.append("")
    lines.append(header)
    lines.append(rule)
    lines.extend(medication_rows)
    lines.append("")
    return NameBankPacket(
        person_tokens=person_tokens,
        medication_tokens=medication_tokens,
        per_source=per_source,
        per_operator=per_operator,
        markdown="\n".join(lines),
    )


__all__ = [
    "BUILDER_VERSION",
    "CMUDICT_SOURCE_URL",
    "DEFAULT_BUILD_SEED",
    "EXTRACTOR_VERSION",
    "MANUAL_MEDICATION_PRONUNCIATIONS",
    "WIKIPRON_SOURCE_URL",
    "WORDLIST_SOURCE_URL",
    "BuildOutcome",
    "ExtractOutcome",
    "ExtractProvenance",
    "NameBankPacket",
    "NameBanksError",
    "NameBanksManifest",
    "PoolStats",
    "build_name_bank_packet",
    "build_name_banks",
    "extract_name_sources",
    "restore_surname_casing",
]

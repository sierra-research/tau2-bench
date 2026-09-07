# Copyright Sierra
"""Foreign-token pronunciation columns for properties + vehicles (§8b).

Phase 3 of the intake pronunciation program (design doc
docs/designs/intake-mispronunciation.md §8b). Two ``tau2`` verbs mirror the
person-name pipeline:

1. ``tau2 intake-names extract-foreign`` — scan the properties/vehicles
   banks for FOREIGN-TOKEN CANDIDATES (the fixed rule below), look every
   candidate up in the 16 pinned per-language WikiPron bulk TSVs
   (:data:`tau2.domains.intake.tasks.loanwords.LEXICON_SOURCES`, ASCII-folded
   keys — bank values are ASCII by decree), and write the checked-in
   extracts under ``name_sources/`` with full provenance (URL + retrieval
   date + sha256 per lexicon, like the PR-A extract).

   **Candidate rule** (fixed; reproduces the §8b measurement): a value/decoy
   token is a candidate iff it is letters-only (no digit, no ``&``), at
   least two characters, NOT an oddball-table class (ALL-CAPS or
   mixed-case-inner tokens — GLB, TFSI, xDrive, gCaorach — stay owned by
   the section-8 authored table), and absent from CMUdict and the English
   wordlist (all rows, so English proper nouns count as covered).

2. ``tau2 intake-names build-foreign`` — resolve every candidate through
   the §8b sourcing ladder and rewrite ``properties.yaml`` /
   ``vehicles.yaml`` IN PLACE (values, difficulties, decoys preserved
   verbatim; only ``language`` + ``pronunciations`` columns change):

   - **tier 3**: transparent English compounds misflagged as foreign
     (:data:`TIER3_ENGLISH_TOKENS`) — default reading, no column;
   - **wiktionary**: exact-form IPA printed on en.wiktionary
     (:data:`WIKTIONARY_FOREIGN_PRONUNCIATIONS`, per-token source URL);
   - **authored**: coinages, archaisms, romanized Greek, and inflected
     forms no lexicon or exact-form entry attests
     (:data:`AUTHORED_FOREIGN_PRONUNCIATIONS`, per-token citation);
   - **wikipron**: the entry's curated language pin
     (:data:`LANGUAGE_PINS`) selects the bulk lexicon and adaptation table
     (:func:`tau2.domains.intake.tasks.loanwords.adapt_foreign_ipa`).

   The curated tables take precedence over the bulk lookup: fold collisions
   make blind lexicon-first unsafe (deu ``krümmen`` folds onto the
   declension ``Krummen`` with the wrong vowel; pol ``impreza`` is a false
   friend of the Subaru coinage), and every table row that shadows a bulk
   match is recorded in the manifest and rendered in the packet, so the
   owner reviews exactly what the ladder decided. A candidate no rung
   resolves fails the build.

   Every covered token draws a ``mispronounced`` variant through the SAME
   seeded balance-greedy operator catalog as names/medications
   (:class:`tau2.domains.intake.tasks.pronunciation.MispronunciationDrawer`,
   one drawer per bank, no hard pins initially). Tokens repeated across
   entries (Auberge, Pousada) draw ONCE and share the column — the build
   asserts repeated tokens resolve identically.

3. ``tau2 intake-names foreign-packet`` — render the owner review packet
   (token, language pin, source, phonemes, respelling, mispronounced,
   operator, citation) over the checked-in banks + manifest.

Machine, not scripts: same inputs + same seed -> byte-identical banks.
"""

import csv
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Annotated, Optional

import yaml
from loguru import logger
from pydantic import Field

from tau2.domains.intake.tasks.banks import (
    PronunciationSource,
    TokenPronunciation,
    foreign_value_tokens,
    load_banks,
)
from tau2.domains.intake.tasks.loanwords import (
    LEXICON_SOURCES,
    LoanLanguage,
    adapt_foreign_ipa,
)
from tau2.domains.intake.tasks.name_banks import (
    CMUDICT_SOURCE_URL,
    DEFAULT_BUILD_SEED,
    NAME_SOURCES_DIR,
    WORDLIST_SOURCE_URL,
    NameBanksError,
    RawSource,
    _named_rng,
    _sha256_path,
    _write_if_changed,
    _yaml_text,
)
from tau2.domains.intake.tasks.pronunciation import (
    NO_VARIANT_TOKENS,
    MispronunciationDrawer,
    Syllable,
    render_respelling,
)
from tau2.utils.pydantic_utils import BaseModelNoExtra

FOREIGN_EXTRACTOR_VERSION = "1"
FOREIGN_BUILDER_VERSION = "1"

FOREIGN_BANKS: tuple[str, ...] = ("properties", "vehicles")

# Fixed candidate-rule regexes (mirrors of the section-8 oddball scan's,
# asserted equivalent by the guard test — the two ownership boundaries must
# partition the same token stream).
_HAS_DIGIT = re.compile(r"[0-9]")
_ALL_CAPS = re.compile(r"[A-Z]{2,}")
_MIXED_CASE_INNER = re.compile(r"[A-Za-z]*[a-z][A-Z][A-Za-z]*")

FOREIGN_MIN_TOKEN_LENGTH = 2


def is_foreign_candidate(token: str, english_words: set[str]) -> bool:
    """The fixed §8b candidate rule for one bank token."""
    if len(token) < FOREIGN_MIN_TOKEN_LENGTH:
        return False
    if _HAS_DIGIT.search(token) or "&" in token:
        return False
    if _ALL_CAPS.fullmatch(token) or _MIXED_CASE_INNER.fullmatch(token):
        return False  # owned by the section-8 oddball table
    return token.lower() not in english_words


# ---------------------------------------------------------------------------
# ASCII folding of lexicon keys (#534: bank values are ASCII-folded)
# ---------------------------------------------------------------------------

# Non-combining letters NFD cannot strip.
_FOLD_LETTERS = {
    "ł": "l",
    "ø": "o",
    "đ": "d",
    "ß": "ss",
    "æ": "ae",
    "œ": "oe",
    "ð": "d",
    "þ": "th",
    "ı": "i",
}

# The Germanic transliteration convention (ö -> oe, ...): bank authors used
# it for some tokens (Roessl, Soestre), plain diacritic-stripping for others
# (Krummen), so the extract indexes BOTH fold variants of every lexicon key.
_GERMANIC_LETTERS = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "å": "aa",
    "ø": "oe",
    "æ": "ae",
    "ß": "ss",
}


def _strip_fold(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return "".join(_FOLD_LETTERS.get(ch, ch) for ch in stripped).casefold()


def lexicon_key_folds(word: str) -> set[str]:
    """The ASCII fold variants a lexicon key matches bank tokens under."""
    lowered = word.casefold()
    germanic = "".join(_GERMANIC_LETTERS.get(ch, ch) for ch in lowered)
    return {_strip_fold(lowered), _strip_fold(germanic)}


# ---------------------------------------------------------------------------
# Curated tables (the §8b delegated curation; reviewed via the packet)
# ---------------------------------------------------------------------------

# Per-ENTRY language pins, curated from each entry's linguistic flavor.
# Several tokens are attested in multiple languages (Sete, Fala, Nove, Tri,
# Kuria, Nitti, Rumach, Impreza), so first-hit lookup is NOT acceptable —
# the pin decides which lexicon and adaptation table apply. Entries whose
# foreign tokens are all authored still carry their flavor pin (recorded on
# the entry, reviewed in the packet); entries with no foreign tokens carry
# none. Keys are exact bank values; a stale key fails the build.
LANGUAGE_PINS: dict[str, dict[str, LoanLanguage]] = {
    "properties": {
        "Alpenblick Hotel": LoanLanguage.GERMAN,
        "Hotel du Vieux Port": LoanLanguage.FRENCH,
        "Hotel Mar Azul": LoanLanguage.SPANISH,
        "Pousada Quinta do Alvarinho Velho": LoanLanguage.PORTUGUESE,
        "Gasthof Alte Schmiede": LoanLanguage.GERMAN,
        "Ty Gwyn y Mynydd Country House": LoanLanguage.WELSH,
        "Ostan Chnoc na gCaorach": LoanLanguage.IRISH,
        "Albergo Vecchio Frantoio 1892": LoanLanguage.ITALIAN,
        "Hotel de l'Ecluse aux Herons": LoanLanguage.FRENCH,
        "Hostal El Zaguan de las Animas": LoanLanguage.SPANISH,
        "Pensjonat Pod Zlamana Podkowa": LoanLanguage.POLISH,
        "Pensionat Kvarnviken 7": LoanLanguage.SWEDISH,
        "Landhaus Ravenfeld Q3": LoanLanguage.GERMAN,
        "Quinta das Sete Fontes d'El-Rei": LoanLanguage.PORTUGUESE,
        "Herberg De Negen Zwaluwen": LoanLanguage.DUTCH,
        "Fjellstua Ni-og-Nitti": LoanLanguage.NORWEGIAN,
        "Hotel Zalahazy-Kuria 1848": LoanLanguage.HUNGARIAN,
        "Auberge du Corbeau Bavard": LoanLanguage.FRENCH,
        "Casona del Arriero Cansado": LoanLanguage.SPANISH,
        "Konak Kirik Degirmen 9": LoanLanguage.TURKISH,
        "Rumach Machair House": LoanLanguage.SCOTTISH_GAELIC,
        "Auberge des Trois Chouettes": LoanLanguage.FRENCH,
        "Gasthaus Zum Krummen Ast": LoanLanguage.GERMAN,
        "Pension Weisses Roessl 1897": LoanLanguage.GERMAN,
        "Locanda del Gufo Reale": LoanLanguage.ITALIAN,
        "Hospederia del Peregrino Descalzo": LoanLanguage.SPANISH,
        "Pousada do Moinho Torto": LoanLanguage.PORTUGUESE,
        "Herberge De Scheve Toren": LoanLanguage.DUTCH,
        "Vertshuset Syv Soestre": LoanLanguage.NORWEGIAN,
        "Gjestgiveri Nordlys 66": LoanLanguage.NORWEGIAN,
        "Zajazd Pod Bocianem Bialym": LoanLanguage.POLISH,
        "Fogado A Harom Hollohoz": LoanLanguage.HUNGARIAN,
        "Penzion U Tri Ruzi": LoanLanguage.CZECH,
        "Hanul Lupului Singuratic": LoanLanguage.ROMANIAN,
        "Casale delle Nove Lune": LoanLanguage.ITALIAN,
        "Manoir du Heron Cendre": LoanLanguage.FRENCH,
        "Quinta da Pedra que Fala": LoanLanguage.PORTUGUESE,
    },
    "vehicles": {
        "Renault Megane": LoanLanguage.FRENCH,
        "Renault Captur E-Tech": LoanLanguage.FRENCH,
        "Alfa Romeo Giulia Ti": LoanLanguage.ITALIAN,
        "Alfa Romeo Stelvio Veloce": LoanLanguage.ITALIAN,
        "Seat Ateca": LoanLanguage.SPANISH,
        "Cupra Formentor VZ2": LoanLanguage.SPANISH,
        "Cupra Born V3": LoanLanguage.SPANISH,
        "Volkswagen Tiguan": LoanLanguage.GERMAN,
    },
}

# Transparent English compounds misflagged as foreign by the coverage scan
# (design doc §8b): reclassified tier 3 — reliable default reading, no
# column. Parva is the English village-name convention (Snoring Parva) and
# reads exactly as spelled.
TIER3_ENGLISH_TOKENS: dict[str, frozenset[str]] = {
    "properties": frozenset(
        {
            "Amberfield",
            "Bayfront",
            "Elmsworth",
            "Fernbrook",
            "Gullwing",
            "Harborview",
            "Ivorygate",
            "Parva",
            "Prickwillow",
            "Ravenfeld",
            "Rosewater",
            "Silverpine",
            "Whitecliff",
        }
    ),
    "vehicles": frozenset(
        {
            "Aircross",
            "Crossback",
            "Grandland",
            "Hardtop",
            "Sportline",
        }
    ),
}


def _syl(onset: str, nucleus: str, coda: str = "", stressed: bool = False) -> Syllable:
    """Compact syllable constructor for the curated tables below (onset and
    coda are space-separated respelling units)."""
    return Syllable(
        onset=onset.split(), nucleus=nucleus, coda=coda.split(), stressed=stressed
    )


class CuratedForeignPronunciation(BaseModelNoExtra):
    """One curated foreign-token pronunciation (wiktionary or authored tier).

    ``reference`` is what the bank's ``phonemes`` column shows: the IPA as
    printed on the cited page (wiktionary tier) or the authored reference
    reading (authored tier). ``syllables`` is the same reading in repo
    respelling units — the ``respelling`` column renders from it and the
    distortion operators run on it.
    """

    reference: Annotated[
        str, Field(description="Printed IPA (wiktionary) or authored reading.")
    ]
    syllables: Annotated[
        list[Syllable], Field(description="The reading in repo respelling units.")
    ]
    citation: Annotated[
        str,
        Field(
            description=(
                "Per-token source URL (wiktionary tier) or authoring "
                "rationale/convention (authored tier)."
            )
        ),
    ]
    language: Annotated[
        Optional[LoanLanguage],
        Field(
            default=None,
            description="Source-language flavor (None for coinages/archaisms).",
        ),
    ]


# Exact-form pronunciations printed on en.wiktionary (§8b sourcing ladder
# rung 2): only rows whose EXACT inflected/compound form carries a printed
# IPA qualify — composed or lemma-derived readings are authored instead.
WIKTIONARY_FOREIGN_PRONUNCIATIONS: dict[str, dict[str, CuratedForeignPronunciation]] = {
    "properties": {
        # German spelling in a Dutch-flavored entry: the row's own language
        # pin (deu) selects the adaptation, not the entry's (nld).
        "Herberge": CuratedForeignPronunciation(
            reference="/ˈhɛʁˌbɛʁɡə/",
            syllables=[
                _syl("h", "eh", "r", True),
                _syl("b", "eh", "r"),
                _syl("g", "uh"),
            ],
            citation="https://en.wiktionary.org/wiki/Herberge",
            language=LoanLanguage.GERMAN,
        ),
        "Syv": CuratedForeignPronunciation(
            reference="/syv/ [syːv]",
            syllables=[_syl("s", "ee", "v", True)],
            citation="https://en.wiktionary.org/wiki/syv",
            language=LoanLanguage.NORWEGIAN,
        ),
    },
    "vehicles": {},
}

# Authored respellings (§8b sourcing ladder rung 3): coinages, archaisms,
# romanized Greek, and inflected/compound forms neither the bulk lexicons
# nor an exact-form Wiktionary entry attest. Owner-reviewed via the packet.
AUTHORED_FOREIGN_PRONUNCIATIONS: dict[str, dict[str, CuratedForeignPronunciation]] = {
    "properties": {
        "Alpenblick": CuratedForeignPronunciation(
            reference="AHL pen blik",
            syllables=[
                _syl("", "ah", "l", True),
                _syl("p", "eh", "n"),
                _syl("b l", "ih", "k"),
            ],
            citation="German compound Alpen + Blick (alp view); initial compound stress",
            language=LoanLanguage.GERMAN,
        ),
        "Alvarinho": CuratedForeignPronunciation(
            reference="al vah REE nyoo",
            syllables=[
                _syl("", "a", "l"),
                _syl("v", "uh"),
                _syl("r", "ee", "", True),
                _syl("n y", "oo"),
            ],
            citation=(
                "Portuguese grape/place name; regular -inho reading (cf. "
                "Wiktionary vinho, VEE-nyoo); no attested entry for the name"
            ),
            language=LoanLanguage.PORTUGUESE,
        ),
        "Asimeniou": CuratedForeignPronunciation(
            reference="ah see MEH nee oo",
            syllables=[
                _syl("", "ah"),
                _syl("s", "ee"),
                _syl("m", "eh", "", True),
                _syl("n", "ee"),
                _syl("", "oo"),
            ],
            citation=(
                "Romanized Greek asimeniou (silver, genitive); Modern Greek "
                "stress on -me- (ell lexicon is Greek-script, no ASCII match)"
            ),
        ),
        "Feggariou": CuratedForeignPronunciation(
            reference="fehng gah RYOO",
            syllables=[
                _syl("f", "eh", "ng"),
                _syl("g", "ah"),
                _syl("r y", "oo", "", True),
            ],
            citation=(
                "Romanized Greek feggariou (of the moon); final stress "
                "(ell lexicon is Greek-script, no ASCII match)"
            ),
        ),
        "Bocianem": CuratedForeignPronunciation(
            reference="baw CHAH nem",
            syllables=[
                _syl("b", "aw"),
                _syl("ch", "ah", "", True),
                _syl("n", "eh", "m"),
            ],
            citation=(
                "Polish bocian (WikiPron pol, BAW-chahn) + instrumental "
                "-em; the inflected form has no lexicon or Wiktionary entry"
            ),
            language=LoanLanguage.POLISH,
        ),
        "Brycgstow": CuratedForeignPronunciation(
            reference="BRIJ stoh",
            syllables=[_syl("b r", "ih", "j", True), _syl("s t", "oh")],
            citation=(
                "Old English Brycgstow (bridge place), the early name of "
                "Bristol; brycg reads bridge, stow as in the place suffix"
            ),
        ),
        "Chouettes": CuratedForeignPronunciation(
            reference="SHWEHT",
            syllables=[_syl("sh w", "eh", "t", True)],
            citation=(
                "French chouette (WikiPron fra, SHWEHT); plural -es is "
                "silent; the plural form itself is unattested"
            ),
            language=LoanLanguage.FRENCH,
        ),
        "Fjellstua": CuratedForeignPronunciation(
            reference="FYEL stoo ah",
            syllables=[_syl("f y", "eh", "l", True), _syl("s t", "oo"), _syl("", "ah")],
            citation=(
                "Norwegian fjellstue (mountain lodge) + definite -a; fj- "
                "reads fy-, compound stress initial"
            ),
            language=LoanLanguage.NORWEGIAN,
        ),
        "Frantoio": CuratedForeignPronunciation(
            reference="frahn TOH yoh",
            syllables=[
                _syl("f r", "ah", "n"),
                _syl("t", "oh", "", True),
                _syl("y", "oh"),
            ],
            citation=(
                "Italian frantoio (olive press); regular -oio reading, "
                "penultimate stress; the Wiktionary entry prints no IPA"
            ),
            language=LoanLanguage.ITALIAN,
        ),
        "Gjestgiveri": CuratedForeignPronunciation(
            reference="YEST gee veh ree",
            syllables=[
                _syl("y", "eh", "s t", True),
                _syl("g", "ee"),
                _syl("v", "eh"),
                _syl("r", "ee"),
            ],
            citation=(
                "Norwegian gjestgiveri (coaching inn); gj- reads y-, "
                "compound stress initial"
            ),
            language=LoanLanguage.NORWEGIAN,
        ),
        "Hanul": CuratedForeignPronunciation(
            reference="HAH nool",
            syllables=[_syl("h", "ah", "", True), _syl("n", "oo", "l")],
            citation=(
                "Romanian han (inn, from Turkish) + definite article -ul; "
                "the definite form has no lexicon or Wiktionary entry"
            ),
            language=LoanLanguage.ROMANIAN,
        ),
        "Hollohoz": CuratedForeignPronunciation(
            reference="HOHL loh hohz",
            syllables=[
                _syl("h", "oh", "l", True),
                _syl("l", "oh"),
                _syl("h", "oh", "z"),
            ],
            citation=(
                "Hungarian hollo (raven, Wiktionary hu-IPA HOHL-loh) + "
                "allative -hoz ('at the three ravens'); initial stress"
            ),
            language=LoanLanguage.HUNGARIAN,
        ),
        "Hospederia": CuratedForeignPronunciation(
            reference="ohs peh deh REE ah",
            syllables=[
                _syl("", "oh", "s"),
                _syl("p", "eh"),
                _syl("d", "eh"),
                _syl("r", "ee", "", True),
                _syl("", "ah"),
            ],
            citation=(
                "Spanish hospederia (guesthouse); the native spelling "
                "accent-marks -RI- (the -ia suffix), which the ASCII fold "
                "loses, so authored"
            ),
            language=LoanLanguage.SPANISH,
        ),
        "Krummen": CuratedForeignPronunciation(
            reference="KRUUM uhn",
            syllables=[_syl("k r", "uu", "m", True), _syl("", "uh", "n")],
            citation=(
                "German krumm (https://en.wiktionary.org/wiki/krumm, KRUUM) "
                "+ weak-declension -en; shadows the FALSE bulk fold-match "
                "kruemmen (the verb, wrong vowel)"
            ),
            language=LoanLanguage.GERMAN,
        ),
        "Kuria": CuratedForeignPronunciation(
            reference="KOO ree ah",
            syllables=[_syl("k", "oo", "", True), _syl("r", "ee"), _syl("", "ah")],
            citation=(
                "Hungarian kuria (manor house); the accented u reads long "
                "oo, initial stress; the Polish bulk match is a different "
                "word (curia)"
            ),
            language=LoanLanguage.HUNGARIAN,
        ),
        "Kvarnviken": CuratedForeignPronunciation(
            reference="KVAHRN vee ken",
            syllables=[
                _syl("k v", "ah", "r n", True),
                _syl("v", "ee"),
                _syl("k", "eh", "n"),
            ],
            citation=(
                "Swedish compound kvarn (mill) + viken (the bay); compound "
                "stress initial; the definite compound is unattested"
            ),
            language=LoanLanguage.SWEDISH,
        ),
        "Lupului": CuratedForeignPronunciation(
            reference="LOO poo looy",
            syllables=[
                _syl("l", "oo", "", True),
                _syl("p", "oo"),
                _syl("l", "oo", "y"),
            ],
            citation=(
                "Romanian lup /lup/ (https://en.wiktionary.org/wiki/lup) + "
                "genitive definite -ului (of the wolf); stem stress"
            ),
            language=LoanLanguage.ROMANIAN,
        ),
        "Nitti": CuratedForeignPronunciation(
            reference="NIT tee",
            syllables=[_syl("n", "ih", "t", True), _syl("t", "ee")],
            citation=(
                "Norwegian nitti (ninety, in Ni-og-Nitti); the Italian bulk "
                "match is the surname Nitti, the wrong language here"
            ),
            language=LoanLanguage.NORWEGIAN,
        ),
        "Nordlys": CuratedForeignPronunciation(
            reference="NOOR lees",
            syllables=[_syl("n", "oo", "r", True), _syl("l", "ee", "s")],
            citation=(
                "Norwegian nordlys (northern lights); d assimilates in the "
                "rdl cluster, y reads ee anglicized; initial stress"
            ),
            language=LoanLanguage.NORWEGIAN,
        ),
        "Pensionat": CuratedForeignPronunciation(
            reference="pen shoo NAHT",
            syllables=[
                _syl("p", "eh", "n"),
                _syl("sh", "oo"),
                _syl("n", "ah", "t", True),
            ],
            citation=(
                "Swedish pensionat (guesthouse); -sion- reads sh, final "
                "stress; the German bulk match is the wrong language here"
            ),
            language=LoanLanguage.SWEDISH,
        ),
        "Roessl": CuratedForeignPronunciation(
            reference="REHS ul",
            syllables=[_syl("r", "eh", "s", True), _syl("", "ul")],
            citation=(
                "German Roessl (little horse, oe = o-umlaut ASCII "
                "convention; cf. Wiktionary Roessel and the operetta Im "
                "weissen Roessl); the umlaut vowel anglicizes to eh"
            ),
            language=LoanLanguage.GERMAN,
        ),
        "Rumach": CuratedForeignPronunciation(
            reference="ROO muk",
            syllables=[_syl("r", "oo", "", True), _syl("m", "uh", "k")],
            citation=(
                "Scottish Gaelic rumach (marsh); final ch anglicized k, "
                "initial stress; the Polish bulk match is a different word"
            ),
            language=LoanLanguage.SCOTTISH_GAELIC,
        ),
        "Soestre": CuratedForeignPronunciation(
            reference="SUHS treh",
            syllables=[_syl("s", "uh", "s", True), _syl("t r", "eh")],
            citation=(
                "Norwegian soestre (sisters; oe = o-slash ASCII "
                "convention); the vowel anglicizes to uh, initial stress; "
                "the plural is unattested"
            ),
            language=LoanLanguage.NORWEGIAN,
        ),
        "Vertshuset": CuratedForeignPronunciation(
            reference="VEHRTS hoo set",
            syllables=[
                _syl("v", "eh", "r t s", True),
                _syl("h", "oo"),
                _syl("s", "eh", "t"),
            ],
            citation=(
                "Norwegian vertshus (inn) + definite -et; compound stress "
                "initial; neither form has a lexicon or Wiktionary entry"
            ),
            language=LoanLanguage.NORWEGIAN,
        ),
        "Animas": CuratedForeignPronunciation(
            reference="AH nee mahs",
            syllables=[_syl("", "ah", "", True), _syl("n", "ee"), _syl("m", "ah", "s")],
            citation=(
                "Spanish animas (souls, las Animas): the native spelling "
                "accent-marks ANTEPENULT stress (esdrujula), which the "
                "ASCII fold loses and the Romance stress rule would "
                "misplace (penult); authored, shadows the spa bulk row"
            ),
            language=LoanLanguage.SPANISH,
        ),
        "Zaguan": CuratedForeignPronunciation(
            reference="zah GWAHN",
            syllables=[_syl("z", "ah"), _syl("g w", "ah", "n", True)],
            citation=(
                "Spanish zaguan (entrance hall): the native spelling "
                "accent-marks FINAL stress, which the ASCII fold loses and "
                "the Romance stress rule would misplace (penult); authored, "
                "shadows the spa bulk row"
            ),
            language=LoanLanguage.SPANISH,
        ),
        "Whaligoe": CuratedForeignPronunciation(
            reference="WAY lih goh",
            syllables=[_syl("w", "ay", "", True), _syl("l", "ih"), _syl("g", "oh")],
            citation=(
                "Caithness place name (whale + goe, a rocky inlet); local "
                "reading WAY-li-go"
            ),
        ),
        "Zalahazy": CuratedForeignPronunciation(
            reference="ZAH lah hah zee",
            syllables=[
                _syl("z", "ah", "", True),
                _syl("l", "ah"),
                _syl("h", "ah"),
                _syl("z", "ee"),
            ],
            citation=(
                "Hungarian coined estate name Zala + hazy (of the house of "
                "Zala); regular Hungarian reading, initial stress"
            ),
            language=LoanLanguage.HUNGARIAN,
        ),
        "Zlamana": CuratedForeignPronunciation(
            reference="zwah MAH nah",
            syllables=[_syl("z w", "ah"), _syl("m", "ah", "", True), _syl("n", "ah")],
            citation=(
                "Polish zlamana (broken, feminine; the l-stroke reads w); "
                "WikiPron carries only the masculine zlamany (ZWAH-mah-nih); "
                "penult stress"
            ),
            language=LoanLanguage.POLISH,
        ),
        "Zwaluwen": CuratedForeignPronunciation(
            reference="ZWAH loo wen",
            syllables=[
                _syl("z w", "ah", "", True),
                _syl("l", "oo"),
                _syl("w", "eh", "n"),
            ],
            citation=(
                "Dutch zwaluw (https://en.wiktionary.org/wiki/zwaluw, "
                "ZWAH-loo) + plural -en; the plural form prints no IPA of "
                "its own"
            ),
            language=LoanLanguage.DUTCH,
        ),
        "d'El": CuratedForeignPronunciation(
            reference="DEL",
            syllables=[_syl("d", "eh", "l", True)],
            citation=(
                "Portuguese elision de + El (archaic royal article, "
                "d'El-Rei 'of the King'); reads del"
            ),
            language=LoanLanguage.PORTUGUESE,
        ),
        "l'Ecluse": CuratedForeignPronunciation(
            reference="leh KLOOZ",
            syllables=[_syl("l", "eh"), _syl("k l", "oo", "z", True)],
            citation=(
                "French ecluse (lock; "
                "https://en.wiktionary.org/wiki/%C3%A9cluse, eh-KLOOZ) with "
                "elided article l'; the elided form has no entry of its own"
            ),
            language=LoanLanguage.FRENCH,
        ),
        "Velho": CuratedForeignPronunciation(
            reference="VEL yoo",
            syllables=[_syl("v", "eh", "l", True), _syl("y", "oo")],
            citation=(
                "European Portuguese velho (pt-IPA VEL-yoo, "
                "https://en.wiktionary.org/wiki/velho); shadows the bulk "
                "row, whose first variant carries the dialectal b- onset"
            ),
            language=LoanLanguage.PORTUGUESE,
        ),
    },
    "vehicles": {
        "Ateca": CuratedForeignPronunciation(
            reference="ah TEH kah",
            syllables=[_syl("", "ah"), _syl("t", "eh", "", True), _syl("k", "ah")],
            citation=(
                "Spanish town Ateca (Zaragoza), SEAT's place-name model "
                "convention; regular Spanish reading, no attested entry"
            ),
            language=LoanLanguage.SPANISH,
        ),
        "Captur": CuratedForeignPronunciation(
            reference="kap TOOR",
            syllables=[_syl("k", "a", "p"), _syl("t", "oo", "r", True)],
            citation="Renault coinage (capture clipped); Renault says cap-TOUR",
            language=LoanLanguage.FRENCH,
        ),
        "Cupra": CuratedForeignPronunciation(
            reference="KOO prah",
            syllables=[_syl("k", "oo", "", True), _syl("p r", "ah")],
            citation="SEAT coinage (CUP RAcing); Spanish reading KOO-prah",
            language=LoanLanguage.SPANISH,
        ),
        "Elantra": CuratedForeignPronunciation(
            reference="ih LAN truh",
            syllables=[_syl("", "ih"), _syl("l", "a", "n", True), _syl("t r", "uh")],
            citation="Hyundai coinage; Hyundai USA reading ih-LAN-truh",
        ),
        "Evoque": CuratedForeignPronunciation(
            reference="ih VOHK",
            syllables=[_syl("", "ih"), _syl("v", "oh", "k", True)],
            citation="Land Rover coinage of evoke; reads ih-VOHK",
        ),
        "Formentor": CuratedForeignPronunciation(
            reference="fawr men TAWR",
            syllables=[
                _syl("f", "aw", "r"),
                _syl("m", "eh", "n"),
                _syl("t", "aw", "r", True),
            ],
            citation=(
                "Cap de Formentor, Mallorca (Catalan place name); final "
                "stress; no ASCII-matchable lexicon carries it"
            ),
            language=LoanLanguage.SPANISH,
        ),
        "Impreza": CuratedForeignPronunciation(
            reference="im PREH zuh",
            syllables=[
                _syl("", "ih", "m"),
                _syl("p r", "eh", "", True),
                _syl("z", "uh"),
            ],
            citation=(
                "Subaru coinage (from Italian impresa); Subaru USA reading "
                "im-PREH-za; the Polish bulk match (impreza, a party) is a "
                "false friend"
            ),
        ),
        "Kodiaq": CuratedForeignPronunciation(
            reference="KOH dee ak",
            syllables=[_syl("k", "oh", "", True), _syl("d", "ee"), _syl("", "a", "k")],
            citation=(
                "Skoda coinage from Kodiak (bear), respelled with the "
                "Alutiiq q; reads KOH-dee-ak"
            ),
        ),
        "Sportage": CuratedForeignPronunciation(
            reference="SPAWR tij",
            syllables=[_syl("s p", "aw", "r t", True), _syl("", "ih", "j")],
            citation="Kia coinage (sport + -age); Kia America reading SPOR-tij",
        ),
        "Stelvio": CuratedForeignPronunciation(
            reference="STEL vee oh",
            syllables=[_syl("s t", "eh", "l", True), _syl("v", "ee"), _syl("", "oh")],
            citation=(
                "Passo dello Stelvio (Italian Alpine pass); antepenult "
                "stress; the Wiktionary entry prints no IPA"
            ),
            language=LoanLanguage.ITALIAN,
        ),
        "Tiguan": CuratedForeignPronunciation(
            reference="TEE gwahn",
            syllables=[_syl("t", "ee", "", True), _syl("g w", "ah", "n")],
            citation="VW coinage Tiger + Leguan; VW reading TEE-gwahn",
            language=LoanLanguage.GERMAN,
        ),
    },
}


# ---------------------------------------------------------------------------
# Provenance / manifest models
# ---------------------------------------------------------------------------


class ForeignExtractProvenance(BaseModelNoExtra):
    """Provenance for the checked-in foreign extracts."""

    extractor_version: str
    cmudict: RawSource
    wordlist: RawSource
    lexicons: Annotated[
        dict[str, RawSource],
        Field(description="RawSource per LoanLanguage value (the 16 bulk TSVs)."),
    ]
    min_token_length: int
    extract_shas: Annotated[
        dict[str, str], Field(description="sha256 per extract file.")
    ]


class ForeignBankColumn(BaseModelNoExtra):
    """Manifest record of one resolved token (review-support data)."""

    token: str
    source: PronunciationSource
    language: Optional[LoanLanguage] = None
    native_word: Annotated[
        Optional[str],
        Field(
            default=None,
            description="The lexicon's native-script/diacritic key (tier 1).",
        ),
    ]
    shadowed_bulk_language: Annotated[
        Optional[LoanLanguage],
        Field(
            default=None,
            description=(
                "Set when a curated row took precedence over an existing "
                "pinned-lexicon fold match (Krummen/krümmen; reviewed in the "
                "packet)."
            ),
        ),
    ]


class ForeignBanksManifest(BaseModelNoExtra):
    """What one ``tau2 intake-names build-foreign`` run produced."""

    builder_version: str
    seed: int
    extract_shas: dict[str, str]
    pins: Annotated[
        dict[str, dict[str, str]],
        Field(description="Per bank: entry value -> pinned language."),
    ]
    tier3_tokens: Annotated[
        dict[str, list[str]],
        Field(description="Per bank: transparent-English tokens (no column)."),
    ]
    columns: Annotated[
        dict[str, list[ForeignBankColumn]],
        Field(description="Per bank: every resolved token, in build order."),
    ]
    source_counts: dict[str, int]
    operator_counts: dict[str, dict[str, int]]
    no_variant_tokens: dict[str, list[str]]


class ForeignExtractOutcome(BaseModelNoExtra):
    """Files one extract-foreign run wrote (empty when up to date)."""

    sources_dir: Path
    written: list[Path]
    candidates: dict[str, int]
    lexicon_rows: int


class ForeignBuildOutcome(BaseModelNoExtra):
    """Files one build-foreign run wrote (empty when up to date)."""

    seed: int
    banks_dir: Path
    written: list[Path]
    source_counts: dict[str, int]
    operator_counts: dict[str, dict[str, int]]


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


def _english_words(cmudict: Path, wordlist: Path) -> set[str]:
    """Every lowercase form CMUdict or the wordlist carries (all rows —
    for coverage the question is whether ENGLISH carries the token at all,
    so capitalized wordlist proper nouns count, unlike the PR-A word filter
    where they deliberately did not)."""
    words: set[str] = set()
    variant = re.compile(r"\(\d+\)$")
    with cmudict.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts and not variant.search(parts[0]):
                words.add(parts[0].lower())
    with wordlist.open(encoding="utf-8") as handle:
        for line in handle:
            word = line.strip().lower()
            if word:
                words.add(word)
    return words


def _bank_candidates(banks_dir: Path, english: set[str]) -> dict[str, list[str]]:
    """Per bank: the sorted candidate tokens of values AND decoys."""
    candidates: dict[str, list[str]] = {}
    for bank in FOREIGN_BANKS:
        raw = yaml.safe_load((banks_dir / f"{bank}.yaml").read_text())
        tokens: set[str] = set()
        for entry in raw["entries"]:
            for value in (entry["value"], *(entry.get("decoys") or [])):
                for token in foreign_value_tokens(value):
                    if is_foreign_candidate(token, english):
                        tokens.add(token)
        candidates[bank] = sorted(tokens)
    return candidates


def extract_foreign_sources(
    cmudict: Path,
    wordlist: Path,
    lexicons_dir: Path,
    retrieved: str,
    sources_dir: Optional[Path] = None,
    banks_dir: Optional[Path] = None,
) -> ForeignExtractOutcome:
    """Build the checked-in foreign extracts (design doc §8b).

    ``lexicons_dir`` holds the 16 per-language WikiPron TSVs downloaded at
    the pinned commit, named exactly as in :data:`LEXICON_SOURCES`.
    """
    from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR

    sources_dir = Path(sources_dir) if sources_dir is not None else NAME_SOURCES_DIR
    banks_dir = Path(banks_dir) if banks_dir is not None else INTAKE_BANKS_DIR
    written: list[Path] = []

    english = _english_words(Path(cmudict), Path(wordlist))
    candidates = _bank_candidates(banks_dir, english)
    all_tokens = {token for tokens in candidates.values() for token in tokens}
    fold_to_token = {token.casefold(): token for token in all_tokens}

    rows: dict[tuple[str, str], tuple[str, str]] = {}
    lexicon_sources: dict[str, RawSource] = {}
    for language, source in LEXICON_SOURCES.items():
        path = Path(lexicons_dir) / source.filename
        if not path.exists():
            raise NameBanksError(
                f"Missing lexicon TSV for {language.value}: {path} "
                f"(download {source.url})"
            )
        lexicon_sources[language.value] = RawSource(
            url=source.url, retrieved=retrieved, sha256=_sha256_path(path)
        )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                word, phonemes = line.split("\t")
                for fold in lexicon_key_folds(word):
                    token = fold_to_token.get(fold)
                    if token is None:
                        continue
                    key = (token, language.value)
                    if key not in rows:
                        rows[key] = (word, phonemes)

    tokens_csv = io.StringIO()
    writer = csv.writer(tokens_csv, lineterminator="\n")
    writer.writerow(["bank", "token"])
    for bank in FOREIGN_BANKS:
        for token in candidates[bank]:
            writer.writerow([bank, token])

    pron_csv = io.StringIO()
    writer = csv.writer(pron_csv, lineterminator="\n")
    writer.writerow(["token", "language", "word", "phonemes"])
    for (token, language), (word, phonemes) in sorted(rows.items()):
        writer.writerow([token, language, word, phonemes])

    extracts = {
        "foreign_tokens.csv": tokens_csv.getvalue(),
        "foreign_pronunciations.csv": pron_csv.getvalue(),
    }
    for name, text in extracts.items():
        _write_if_changed(sources_dir / name, text, written)

    provenance = ForeignExtractProvenance(
        extractor_version=FOREIGN_EXTRACTOR_VERSION,
        cmudict=RawSource(
            url=CMUDICT_SOURCE_URL,
            retrieved=retrieved,
            sha256=_sha256_path(Path(cmudict)),
        ),
        wordlist=RawSource(
            url=WORDLIST_SOURCE_URL,
            retrieved=retrieved,
            sha256=_sha256_path(Path(wordlist)),
        ),
        lexicons=lexicon_sources,
        min_token_length=FOREIGN_MIN_TOKEN_LENGTH,
        extract_shas={
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in extracts.items()
        },
    )
    _write_if_changed(
        sources_dir / "foreign_provenance.json",
        json.dumps(provenance.model_dump(mode="json"), indent=2) + "\n",
        written,
    )
    logger.info(
        "intake-names extract-foreign: "
        f"{sources_dir} written={len(written)} candidates="
        + ", ".join(f"{bank} {len(tokens)}" for bank, tokens in candidates.items())
        + f" lexicon rows={len(rows)}"
    )
    return ForeignExtractOutcome(
        sources_dir=sources_dir,
        written=written,
        candidates={bank: len(tokens) for bank, tokens in candidates.items()},
        lexicon_rows=len(rows),
    )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _load_foreign_extracts(
    sources_dir: Path,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], tuple[str, str]]]:
    """(candidates per bank, (token, language) -> (native word, phonemes)),
    after verifying the extract shas against the provenance record."""
    provenance = ForeignExtractProvenance.model_validate(
        json.loads((sources_dir / "foreign_provenance.json").read_text())
    )
    for filename, expected in provenance.extract_shas.items():
        actual = _sha256_path(sources_dir / filename)
        if actual != expected:
            raise NameBanksError(
                f"Extract {filename} sha256 {actual} != provenance {expected} "
                "— re-run `tau2 intake-names extract-foreign` or restore it."
            )
    missing_languages = {lang.value for lang in LoanLanguage} - set(provenance.lexicons)
    if missing_languages:
        raise NameBanksError(
            "foreign_provenance.json lacks lexicons for "
            f"{sorted(missing_languages)} — re-run extract-foreign"
        )
    candidates: dict[str, list[str]] = {bank: [] for bank in FOREIGN_BANKS}
    with (sources_dir / "foreign_tokens.csv").open() as handle:
        for row in csv.DictReader(handle):
            candidates[row["bank"]].append(row["token"])
    lexicon: dict[tuple[str, str], tuple[str, str]] = {}
    with (sources_dir / "foreign_pronunciations.csv").open() as handle:
        for row in csv.DictReader(handle):
            lexicon[(row["token"], row["language"])] = (row["word"], row["phonemes"])
    return candidates, lexicon


class _ResolvedToken(BaseModelNoExtra):
    """One resolved candidate token (build-internal)."""

    column: TokenPronunciation
    manifest: ForeignBankColumn


def _resolve_token(
    bank: str,
    token: str,
    pin: Optional[LoanLanguage],
    lexicon: dict[tuple[str, str], tuple[str, str]],
) -> tuple[
    PronunciationSource,
    str,
    list[Syllable],
    Optional[str],
    Optional[LoanLanguage],
    Optional[str],
    Optional[LoanLanguage],
]:
    """(source, phonemes, syllables, citation, language, native, shadowed)."""
    bulk = lexicon.get((token, pin.value)) if pin is not None else None
    curated = WIKTIONARY_FOREIGN_PRONUNCIATIONS[bank].get(token)
    if curated is not None:
        return (
            PronunciationSource.WIKTIONARY,
            curated.reference,
            [s.model_copy(deep=True) for s in curated.syllables],
            curated.citation,
            curated.language,
            None,
            pin if bulk is not None else None,
        )
    curated = AUTHORED_FOREIGN_PRONUNCIATIONS[bank].get(token)
    if curated is not None:
        return (
            PronunciationSource.AUTHORED,
            curated.reference,
            [s.model_copy(deep=True) for s in curated.syllables],
            curated.citation,
            curated.language,
            None,
            pin if bulk is not None else None,
        )
    if bulk is not None:
        native, phonemes = bulk
        return (
            PronunciationSource.WIKIPRON,
            phonemes,
            adapt_foreign_ipa(phonemes, pin),
            None,
            pin,
            native,
            None,
        )
    raise NameBanksError(
        f"Foreign token {bank}/{token!r} resolves through no §8b rung: not "
        "tier-3, not curated, and "
        + (
            f"no {pin.value} lexicon row"
            if pin is not None
            else "its entry carries no language pin"
        )
    )


def build_foreign_banks(
    seed: int = DEFAULT_BUILD_SEED,
    banks_dir: Optional[Path] = None,
    sources_dir: Optional[Path] = None,
) -> ForeignBuildOutcome:
    """Rebuild the properties/vehicles pronunciation columns in place."""
    from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR

    banks_dir = Path(banks_dir) if banks_dir is not None else INTAKE_BANKS_DIR
    sources_dir = Path(sources_dir) if sources_dir is not None else NAME_SOURCES_DIR
    candidates, lexicon = _load_foreign_extracts(sources_dir)

    written: list[Path] = []
    source_counts = {source.value: 0 for source in PronunciationSource}
    operator_counts: dict[str, dict[str, int]] = {}
    manifest_columns: dict[str, list[ForeignBankColumn]] = {}
    no_variant: dict[str, list[str]] = {}

    for bank in FOREIGN_BANKS:
        path = banks_dir / f"{bank}.yaml"
        raw = yaml.safe_load(path.read_text())
        bank_candidates = set(candidates[bank])
        tier3 = TIER3_ENGLISH_TOKENS[bank]
        stale_tier3 = sorted(tier3 - bank_candidates)
        if stale_tier3:
            raise NameBanksError(
                f"TIER3_ENGLISH_TOKENS[{bank}] lists non-candidates "
                f"(stale after a bank or lexicon change?): {stale_tier3}"
            )
        for table_name, table in (
            ("WIKTIONARY_FOREIGN_PRONUNCIATIONS", WIKTIONARY_FOREIGN_PRONUNCIATIONS),
            ("AUTHORED_FOREIGN_PRONUNCIATIONS", AUTHORED_FOREIGN_PRONUNCIATIONS),
        ):
            stale = sorted(set(table[bank]) - bank_candidates)
            if stale:
                raise NameBanksError(
                    f"{table_name}[{bank}] lists non-candidates (stale?): {stale}"
                )
        stale_pins = sorted(
            set(LANGUAGE_PINS[bank]) - {entry["value"] for entry in raw["entries"]}
        )
        if stale_pins:
            raise NameBanksError(
                f"LANGUAGE_PINS[{bank}] pins unknown entries (stale?): {stale_pins}"
            )

        drawer = MispronunciationDrawer(bank)
        bank_operators: dict[str, int] = {}
        columns_by_token: dict[str, _ResolvedToken] = {}
        covered: set[str] = set(tier3)  # tier 3 = covered, no column
        entries_out: list[dict] = []
        bank_manifest: list[ForeignBankColumn] = []
        bank_no_variant: list[str] = []

        for entry in raw["entries"]:
            value = entry["value"]
            pin = LANGUAGE_PINS[bank].get(value)
            decoy_tokens = tuple(
                token
                for decoy in entry.get("decoys") or []
                for token in foreign_value_tokens(decoy)
            )
            columns: list[TokenPronunciation] = []
            for token in foreign_value_tokens(value):
                if token not in bank_candidates or token in tier3:
                    continue
                cached = columns_by_token.get(token)
                if cached is not None:
                    if (
                        pin is not None
                        and cached.column.source is PronunciationSource.WIKIPRON
                        and cached.column.language is not pin
                    ):
                        raise NameBanksError(
                            f"Token {bank}/{token!r} repeats under conflicting "
                            f"pins ({cached.column.language} vs {pin})"
                        )
                    columns.append(cached.column)
                    continue
                source, phonemes, syllables, citation, language, native, shadowed = (
                    _resolve_token(bank, token, pin, lexicon)
                )
                if citation is not None and not citation.isascii():
                    raise NameBanksError(
                        f"Citation for {bank}/{token!r} is not ASCII — the "
                        "banks are ASCII-only (only the phonemes column is "
                        "exempt); romanize the citation text"
                    )
                mispronounced = None
                operator = None
                rng = _named_rng("intake-pron", "operator", bank, value, token, seed)
                drawn = drawer.draw(token, syllables, decoy_tokens, rng)
                if drawn is not None:
                    operator, mispronounced = drawn
                    if token in NO_VARIANT_TOKENS[bank]:
                        raise NameBanksError(
                            f"Token {bank}/{token!r} is allow-listed in "
                            "NO_VARIANT_TOKENS but drew a variant (stale list)"
                        )
                else:
                    if token not in NO_VARIANT_TOKENS[bank]:
                        raise NameBanksError(
                            f"Token {bank}/{token!r} has no non-vacuous "
                            "variant — add it to NO_VARIANT_TOKENS[{bank}] "
                            "deliberately"
                        )
                    bank_no_variant.append(token)
                column = TokenPronunciation(
                    token=token,
                    source=source,
                    phonemes=phonemes,
                    respelling=render_respelling(syllables),
                    mispronounced=mispronounced,
                    operator=operator,
                    citation=citation,
                    language=language,
                )
                if operator is not None:
                    bank_operators[operator.value] = (
                        bank_operators.get(operator.value, 0) + 1
                    )
                source_counts[source.value] += 1
                record = ForeignBankColumn(
                    token=token,
                    source=source,
                    language=language,
                    native_word=native,
                    shadowed_bulk_language=shadowed,
                )
                columns_by_token[token] = _ResolvedToken(column=column, manifest=record)
                bank_manifest.append(record)
                columns.append(column)
                covered.add(token)

            rebuilt: dict = {"value": value, "difficulty": entry["difficulty"]}
            if entry.get("decoys"):
                rebuilt["decoys"] = entry["decoys"]
            if pin is not None:
                if not columns:
                    raise NameBanksError(
                        f"LANGUAGE_PINS[{bank}] pins {value!r} but the entry "
                        "has no pronunciation columns (stale pin)"
                    )
                rebuilt["language"] = pin.value
            if columns:
                rebuilt["pronunciations"] = [
                    column.model_dump(mode="json", exclude_none=True)
                    for column in columns
                ]
            entries_out.append(rebuilt)

        # Wikipron rows require an entry pin; the load-time model enforces
        # entry.language matches the row. Entries with columns but no pin
        # (romanized Greek, archaisms, coinages) are fine — every row there
        # is curated. Every candidate must be covered.
        uncovered = sorted(bank_candidates - covered)
        if uncovered:
            raise NameBanksError(
                f"Bank {bank}: candidate tokens resolved by no rung and "
                f"present in no value (decoy-only candidates are "
                f"unsupported): {uncovered}"
            )
        drawer.assert_pins_consumed()
        operator_counts[bank] = drawer.histogram()
        # histogram counts only what the drawer drew; keep it (bank_operators
        # matches it by construction).
        manifest_columns[bank] = bank_manifest
        no_variant[bank] = sorted(bank_no_variant)

        header = (
            f"# {bank.capitalize()} bank. Values/difficulties/decoys are the\n"
            "# reviewed originals; the language pins and foreign-token\n"
            "# pronunciation columns (design doc\n"
            "# docs/designs/intake-mispronunciation.md sec. 8b) are\n"
            "# regenerated by `tau2 intake-names build-foreign` — do not\n"
            "# hand-edit.\n"
        )
        text = header + _yaml_text({"entity": bank, "entries": entries_out})
        _write_if_changed(path, text, written)

    manifest = ForeignBanksManifest(
        builder_version=FOREIGN_BUILDER_VERSION,
        seed=seed,
        extract_shas=ForeignExtractProvenance.model_validate(
            json.loads((sources_dir / "foreign_provenance.json").read_text())
        ).extract_shas,
        pins={
            bank: {value: language.value for value, language in pins.items()}
            for bank, pins in LANGUAGE_PINS.items()
        },
        tier3_tokens={
            bank: sorted(tokens) for bank, tokens in TIER3_ENGLISH_TOKENS.items()
        },
        columns=manifest_columns,
        source_counts=source_counts,
        operator_counts=operator_counts,
        no_variant_tokens=no_variant,
    )
    _write_if_changed(
        sources_dir / "foreign_banks.manifest.json",
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        written,
    )

    # Loader-level validation of what was just written (fail loud, not ship).
    load_banks(banks_dir)

    logger.info(
        f"intake-names build-foreign: seed={seed} written={len(written)} "
        + ", ".join(
            f"{bank} {len(cols)} tokens" for bank, cols in manifest_columns.items()
        )
    )
    return ForeignBuildOutcome(
        seed=seed,
        banks_dir=banks_dir,
        written=written,
        source_counts=source_counts,
        operator_counts=operator_counts,
    )


# ---------------------------------------------------------------------------
# Review packet (tau2 intake-names foreign-packet)
# ---------------------------------------------------------------------------


class ForeignBankPacket(BaseModelNoExtra):
    """The rendered foreign-token review packet plus its summary counts."""

    tokens: Annotated[int, Field(description="Column-bearing foreign tokens.")]
    per_source: dict[str, int]
    per_language: dict[str, int]
    per_operator: dict[str, dict[str, int]]
    markdown: str


def build_foreign_packet(
    banks_dir: Optional[Path] = None, sources_dir: Optional[Path] = None
) -> ForeignBankPacket:
    """Render the §8b owner review packet over the checked-in banks +
    manifest: one row per column-bearing token (value, tier, token, language,
    source, phonemes, respelling, mispronounced, operator, citation), plus
    the language-pin table, the tier-3 reclassifications, the oddball-table
    delegations, and the operator-applicability proposal. Deterministic."""
    from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR
    from tau2.domains.intake.tasks.pronunciation import OPERATOR_APPLICABILITY
    from tau2.domains.intake.tasks.token_pronunciations import classify_token

    banks_dir = Path(banks_dir) if banks_dir is not None else INTAKE_BANKS_DIR
    sources_dir = Path(sources_dir) if sources_dir is not None else NAME_SOURCES_DIR
    banks = load_banks(banks_dir)
    manifest = ForeignBanksManifest.model_validate(
        json.loads((sources_dir / "foreign_banks.manifest.json").read_text())
    )
    native_words = {
        (bank, column.token): column.native_word
        for bank, columns in manifest.columns.items()
        for column in columns
    }
    shadowed = {
        (bank, column.token): column.shadowed_bulk_language
        for bank, columns in manifest.columns.items()
        for column in columns
        if column.shadowed_bulk_language is not None
    }

    per_source: dict[str, int] = {}
    per_language: dict[str, int] = {}
    per_operator: dict[str, dict[str, int]] = {}
    total = 0

    lines: list[str] = []
    lines.append("# Intake foreign-token pronunciation review packet (phase 3)")
    lines.append("")
    lines.append(
        f"Provenance: builder {manifest.builder_version}, seed {manifest.seed}, "
        "over the checked-in properties/vehicles banks + "
        "name_sources/foreign_banks.manifest.json. Regenerable "
        "byte-identically via `tau2 intake-names foreign-packet`; do not "
        "hand-edit. Design doc docs/designs/intake-mispronunciation.md §8b."
    )
    lines.append("")
    lines.append("## Operator applicability (owner review requested)")
    lines.append("")
    lines.append(
        "Properties and vehicles are treated LIKE person_names — "
        "metathesis / vowel_swap / spelling_pronunciation, NO syllable_drop "
        "(a place or model name missing a syllable reads as a DIFFERENT "
        "name, not a mispronounced one):"
    )
    lines.append("")
    for bank in FOREIGN_BANKS:
        operators = ", ".join(sorted(op.value for op in OPERATOR_APPLICABILITY[bank]))
        lines.append(f"- {bank}: {operators}")
    lines.append("")

    body_lines: list[str] = []
    for bank in FOREIGN_BANKS:
        body_lines.append(f"## {bank}")
        body_lines.append("")
        body_lines.append("### Language pins")
        body_lines.append("")
        body_lines.append("| entry | pin |")
        body_lines.append("|---|---|")
        for value, language in manifest.pins[bank].items():
            body_lines.append(f"| {value} | {language} |")
        body_lines.append("")
        body_lines.append("### Pronunciation columns")
        body_lines.append("")
        body_lines.append(
            "| value | tier | token | language | source | phonemes | "
            "respelling | mispronounced | operator | citation / native word |"
        )
        body_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        bank_operators = per_operator.setdefault(bank, {})
        seen_tokens: set[str] = set()
        for entry in getattr(banks, bank):
            for column in entry.pronunciations:
                note = column.citation or ""
                native = native_words.get((bank, column.token))
                if native and native.casefold() != column.token.casefold():
                    note = f"native: {native}"
                if (bank, column.token) in shadowed:
                    note = (
                        note
                        + (" — " if note else "")
                        + "shadows a "
                        + shadowed[(bank, column.token)].value
                        + " bulk fold-match"
                    )
                body_lines.append(
                    f"| {entry.value} | {entry.difficulty.value} | "
                    f"{column.token} | "
                    f"{column.language.value if column.language else '-'} | "
                    f"{column.source.value} | `{column.phonemes}` | "
                    f"{column.respelling} | {column.mispronounced or '-'} | "
                    f"{column.operator.value if column.operator else '-'} | "
                    f"{note or '-'} |"
                )
                if column.token in seen_tokens:
                    continue  # repeated tokens share one drawn column
                seen_tokens.add(column.token)
                total += 1
                per_source[column.source.value] = (
                    per_source.get(column.source.value, 0) + 1
                )
                language = column.language.value if column.language else "none"
                per_language[language] = per_language.get(language, 0) + 1
                if column.operator is not None:
                    bank_operators[column.operator.value] = (
                        bank_operators.get(column.operator.value, 0) + 1
                    )
        body_lines.append("")
        body_lines.append("### Tier 3: transparent English compounds (no column)")
        body_lines.append("")
        body_lines.append(", ".join(manifest.tier3_tokens[bank]) or "(none)")
        body_lines.append("")
        oddball = sorted(
            {
                token
                for entry in getattr(banks, bank)
                for value in (entry.value, *entry.decoys)
                for token in foreign_value_tokens(value)
                if classify_token(token) is not None
            }
        )
        body_lines.append("### Owned by the section-8 oddball table (no column here)")
        body_lines.append("")
        body_lines.append(", ".join(oddball) or "(none)")
        body_lines.append("")
        if manifest.no_variant_tokens[bank]:
            body_lines.append("### NO_VARIANT_TOKENS (never drawn)")
            body_lines.append("")
            body_lines.append(", ".join(manifest.no_variant_tokens[bank]))
            body_lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- column-bearing foreign tokens: {total}")
    for source, count in sorted(per_source.items()):
        lines.append(f"- source {source}: {count}")
    for language, count in sorted(per_language.items()):
        lines.append(f"- language {language}: {count}")
    for bank, operators in sorted(per_operator.items()):
        for operator, count in sorted(operators.items()):
            lines.append(f"- {bank} operator {operator}: {count}")
    lines.append("")
    lines.extend(body_lines)
    return ForeignBankPacket(
        tokens=total,
        per_source=per_source,
        per_language=per_language,
        per_operator=per_operator,
        markdown="\n".join(lines),
    )


__all__ = [
    "AUTHORED_FOREIGN_PRONUNCIATIONS",
    "FOREIGN_BANKS",
    "FOREIGN_BUILDER_VERSION",
    "FOREIGN_EXTRACTOR_VERSION",
    "ForeignBankPacket",
    "ForeignBanksManifest",
    "ForeignBuildOutcome",
    "ForeignExtractOutcome",
    "ForeignExtractProvenance",
    "LANGUAGE_PINS",
    "TIER3_ENGLISH_TOKENS",
    "WIKTIONARY_FOREIGN_PRONUNCIATIONS",
    "build_foreign_banks",
    "build_foreign_packet",
    "extract_foreign_sources",
    "is_foreign_candidate",
    "lexicon_key_folds",
]

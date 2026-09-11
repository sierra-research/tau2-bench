# Copyright Sierra
"""Spanish-specific pack content tests: locale orthography.

Generic pack validity is covered for ALL languages by test_pack_invariants.py;
generic corpus/identity-map consistency by test_entity_localization.py
(``TestCommittedMapMatchesCorpus``). This file keeps the one contract those
cannot express, because it is a fact about Spanish and not about the pipeline:
**es locale strings carry their diacritics**.

Why it earns a test. The es locale corpus was once authored ASCII-only, so
every caller identity the factory drew from it was a misspelling — Alvaro
Fernandez, Carmen Gonzalez, Nuria Jimenez, Plaza Espana. es was the only
Latin-script pack with the defect (de/fr/pl/pt/tr/vi corpora all carry
accents). The corpus is the reviewed native raw material and must stay
correctly spelled.

The DERIVED identity artifacts are a different contract, and it points the
other way: since the accents-are-a-tool-call-hazard decision, every value
entering an identity is deliberately ASCII-folded at build time
(``fold_to_ascii`` in entity_localization — DB records and golden-action
arguments are compared byte-exactly, and whether a voice transcription
carries the diacritics is an orthography coin-flip). The all-languages ASCII
guards live in test_entity_localization.py
(``test_committed_identity_map_is_pure_ascii``,
``test_built_identity_values_are_pure_ascii``); this file only guards the
corpus, the accented side of the fold.

The catalog below is closed and keyed by the ASCII spelling: every unaccented
form that once appeared in the es corpus, mapped to its correct Peninsular
Spanish orthography. It is asserted over the corpus only — derived artifacts
are EXPECTED to carry the left-hand forms.
"""

import json
import re

import pytest
import yaml

from tau2.multilingual.factory import entity_localization as ent
from tau2.multilingual.registry import get_language_pack

LANG = "es"

# {misspelling: correct Peninsular Spanish spelling}. Closed catalog: these are
# the exact forms the ASCII-authored es corpus shipped.
MISSPELLINGS = {
    # given names
    "Alvaro": "Álvaro",
    "Adrian": "Adrián",
    "Ivan": "Iván",
    "Lucia": "Lucía",
    "Maria": "María",
    "Raul": "Raúl",
    "Rocio": "Rocío",
    "Ruben": "Rubén",
    "Sofia": "Sofía",
    # surnames
    "Alvarez": "Álvarez",
    "Diaz": "Díaz",
    "Dominguez": "Domínguez",
    "Fernandez": "Fernández",
    "Garcia": "García",
    "Gomez": "Gómez",
    "Gonzalez": "González",
    "Hernandez": "Hernández",
    "Jimenez": "Jiménez",
    "Lopez": "López",
    "Marin": "Marín",
    "Martin": "Martín",
    "Martinez": "Martínez",
    "Perez": "Pérez",
    "Rodriguez": "Rodríguez",
    "Sanchez": "Sánchez",
    "Vazquez": "Vázquez",
    # place / street words
    "Alcala": "Alcalá",
    "Constitucion": "Constitución",
    "Espana": "España",
    "Malaga": "Málaga",
    "Via": "Vía",
}

MISSPELLING_RE = re.compile(r"\b(" + "|".join(sorted(MISSPELLINGS)) + r")\b")


def test_pack_discovered():
    pack = get_language_pack(LANG)
    assert pack.language == LANG
    assert pack.display_name == "Spanish"


class TestCorpusOrthography:
    """The reviewed raw material is spelled correctly."""

    def test_no_misspelled_value_in_the_corpus(self):
        corpus = yaml.safe_load(ent.corpus_path(LANG).read_text())
        values = (
            corpus["female_first_names"]
            + corpus["male_first_names"]
            + corpus["last_names"]
            + corpus["street_patterns"]
            + [c["city"] for c in corpus["cities"]]
        )
        wrong = {
            value: MISSPELLING_RE.findall(value)
            for value in values
            if MISSPELLING_RE.search(value)
        }
        assert not wrong, (
            "unaccented Spanish in the es locale corpus — correct spellings: "
            + repr({k: MISSPELLINGS[v[0]] for k, v in wrong.items()})
        )

    def test_catalog_entries_are_actually_the_accented_corpus_values(self):
        """Coverage guard on the catalog itself: every correct spelling listed
        above must appear somewhere in the corpus, so a name dropped from the
        pools cannot leave a dead catalog entry standing in for coverage."""
        text = ent.corpus_path(LANG).read_text()
        missing = sorted(
            correct
            for correct in set(MISSPELLINGS.values())
            if not re.search(rf"\b{re.escape(correct)}\b", text)
        )
        assert not missing, f"catalog entries absent from the es corpus: {missing}"


@pytest.mark.parametrize("domain", ("airline", "telecom"))
class TestIdentityMapIsTheFoldOfTheCorpus:
    """The derived map really is the ASCII fold of the ACCENTED corpus.

    A pure-ASCII map would also pass the generic guards if someone stripped
    the corpus itself — this pins the intended shape: the corpus stays
    accented (checked above) and the map carries the catalog's ASCII forms,
    proving the fold ran rather than the accents never existing."""

    def test_identity_map_carries_the_folded_catalog_forms(self, domain):
        path = ent.identity_map_path(LANG, domain)
        if not path.exists():
            pytest.skip(f"{LANG}/{domain} identity map not generated")
        identities = json.loads(path.read_text())
        assert identities
        folded_hits = [
            key
            for key, identity in identities.items()
            if MISSPELLING_RE.search(
                f"{identity['first_name']} {identity['last_name']}"
            )
        ]
        assert folded_hits, (
            f"no es {domain} identity carries a folded catalog form "
            "(Ivan, Sanchez, Gomez, ...) — either the corpus lost its accented "
            "names or the map was not rebuilt from it"
        )

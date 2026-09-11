import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_data_dir_environment_variable():
    """Test that DATA_DIR can be set via environment variable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_data_dir = Path(temp_dir) / "custom_data"
        temp_data_dir.mkdir()

        with patch.dict(os.environ, {"TAU2_DATA_DIR": str(temp_data_dir)}):
            # Re-import to get the new DATA_DIR value
            import importlib

            import tau2.utils.utils

            importlib.reload(tau2.utils.utils)

            assert tau2.utils.utils.DATA_DIR == temp_data_dir


def test_data_dir_fallback_to_source():
    """Test that DATA_DIR falls back to source directory when env var is not set."""
    # Clear environment variable
    with patch.dict(os.environ, {}, clear=True):
        # Re-import to get the fallback DATA_DIR value
        import importlib

        import tau2.utils.utils

        importlib.reload(tau2.utils.utils)

        # Check that DATA_DIR points to the source directory
        # Calculate expected path from utils.py location
        utils_file = Path(tau2.utils.utils.__file__)
        expected_source_dir = utils_file.parents[3] / "data"
        assert tau2.utils.utils.DATA_DIR == expected_source_dir


class TestFoldForMatch:
    """``fold_for_match`` is the single normalization the domain lookups
    compare through — it folds case and diacritics, and nothing else."""

    def test_folds_case_and_diacritics(self):
        from tau2.utils.text_match import fold_for_match

        assert fold_for_match("Álvaro Fernández") == "alvaro fernandez"
        assert fold_for_match("ALVARO FERNANDEZ") == "alvaro fernandez"

    def test_strips_surrounding_whitespace_only(self):
        from tau2.utils.text_match import fold_for_match

        # Internal spacing and punctuation are IDENTITY-bearing and survive.
        assert fold_for_match("  Ana  María  ") == "ana  maria"
        assert fold_for_match("O'Brien-Smith") == "o'brien-smith"

    def test_precomposed_and_decomposed_forms_agree(self):
        import unicodedata

        from tau2.utils.text_match import fold_for_match

        precomposed = unicodedata.normalize("NFC", "Núria Jiménez")
        decomposed = unicodedata.normalize("NFD", "Núria Jiménez")
        assert precomposed != decomposed  # different byte strings...
        assert fold_for_match(precomposed) == fold_for_match(decomposed)

    def test_does_not_merge_distinct_names(self):
        from tau2.utils.text_match import fold_for_match

        assert fold_for_match("Álvaro Fernández") != fold_for_match("Alvaro Fernandes")


def _identity_map_languages(domain: str) -> list[str]:
    """Every language pack that has a localized identity map for ``domain``."""
    from tau2.multilingual.factory.entity_localization import identity_map_path
    from tau2.multilingual.localize_lib import multilingual_data_dir

    root = multilingual_data_dir()
    if not root.is_dir():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and identity_map_path(d.name, domain).exists()
    )


def _load_identity_map(lang: str, domain: str) -> dict:
    import json

    from tau2.multilingual.factory.entity_localization import identity_map_path

    return json.loads(identity_map_path(lang, domain).read_text())


def _assert_injective(values, what: str) -> None:
    """No two distinct strings in ``values`` fold to the same key."""
    from tau2.utils.text_match import fold_for_match

    canonical: dict[str, str] = {}
    collisions = []
    for value in values:
        folded = fold_for_match(value)
        clash = canonical.setdefault(folded, value)
        if clash != value:
            collisions.append((clash, value, folded))
    assert not collisions, (
        f"{what}: fold_for_match merges records that are distinct on disk — "
        f"{collisions}. The domain lookups compare through the folded form, so "
        "a lookup would resolve to whichever record it happened to scan first. "
        "Fix the DATA (rename one of the records); do not relax this test."
    )


class TestIdentityKeyspacesAreInjectiveUnderFolding:
    """``fold_for_match``'s documented precondition, checked against the data.

    Its docstring says "the caller is responsible for knowing its key space is
    unique modulo case and accents" — and nothing enforced that. The es
    identity maps have since GAINED accents (a caller now says "Iván Sánchez",
    matched against a record spelled either way), which is exactly the change
    that can fold two distinct records together. A folded collision does not
    fail loudly: the lookup returns whichever record the scan reached first,
    silently attributing a call to the wrong customer.

    Checked for every language pack that has an identity map, over the real
    keyspaces the tools scan:

    - telecom ``get_customer_by_name`` folds ``customer.full_name`` over the
      patched customer list, so the localized names have to be distinct from
      each other AND from the canonical English records;
    - airline ``_resolve_key_folded`` folds over ``db.users``, which during an
      identity run holds the canonical ids PLUS the localized ones (the
      localized record is ADDED under a new id, the canonical one is left in
      place), so the union is the keyspace.
    """

    def test_at_least_one_pack_is_covered(self):
        """A guard on the guard: an empty parametrization would pass silently."""
        assert _identity_map_languages("telecom")
        assert _identity_map_languages("airline")

    def test_english_telecom_customer_names_are_injective(self):
        from tau2.multilingual.localize_lib import load_domain_db

        db = load_domain_db("telecom")
        _assert_injective(
            [c["full_name"] for c in db["customers"]],
            "telecom canonical customer names",
        )

    def test_english_airline_user_ids_are_injective(self):
        from tau2.multilingual.localize_lib import load_domain_db

        db = load_domain_db("airline")
        _assert_injective(list(db["users"]), "airline canonical user ids")

    @pytest.mark.parametrize("lang", _identity_map_languages("telecom"))
    def test_telecom_localized_names_are_injective(self, lang):
        from tau2.multilingual.localize_lib import load_domain_db

        db = load_domain_db("telecom")
        localized = [
            e["full_name"] for e in _load_identity_map(lang, "telecom").values()
        ]
        _assert_injective(localized, f"{lang} telecom identity-map names")
        _assert_injective(
            [c["full_name"] for c in db["customers"]] + localized,
            f"{lang} telecom names against the canonical customer records",
        )

    @pytest.mark.parametrize("lang", _identity_map_languages("airline"))
    def test_airline_localized_user_ids_are_injective(self, lang):
        from tau2.multilingual.localize_lib import load_domain_db

        db = load_domain_db("airline")
        localized = [e["user_id"] for e in _load_identity_map(lang, "airline").values()]
        _assert_injective(localized, f"{lang} airline identity-map user ids")
        _assert_injective(
            list(db["users"]) + localized,
            f"{lang} airline user ids against the canonical DB keys",
        )

    @pytest.mark.parametrize("lang", _identity_map_languages("airline"))
    def test_airline_localized_names_are_injective(self, lang):
        entries = _load_identity_map(lang, "airline").values()
        _assert_injective(
            [f"{e['first_name']} {e['last_name']}" for e in entries],
            f"{lang} airline identity-map names",
        )

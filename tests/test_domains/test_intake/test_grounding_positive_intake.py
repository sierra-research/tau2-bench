# Copyright Sierra
"""Tests for the intake positive grounding screens: vehicles vs NHTSA vPIC
(grounding_vpic.py, ``tau2 intake-grounding vpic``) and medications vs NLM
RxNorm (grounding_rxnorm.py, ``tau2 intake-grounding rxnorm``), with the
network layer mocked."""

import json

import pytest

from tau2.domains.intake.grounding_rxnorm import (
    FORM_PHRASES,
    RxnormGroundingReport,
    match_product,
    medications_to_check,
    normalize_strength,
    parse_medication,
    run_rxnorm_screen,
    write_rxnorm_report,
)
from tau2.domains.intake.grounding_screen import Verdict
from tau2.domains.intake.grounding_vpic import (
    VpicGroundingReport,
    VpicMakesCatalog,
    fold_name,
    match_model,
    run_vpic_screen,
    split_make,
    vehicles_to_check,
    write_vpic_report,
)
from tau2.domains.intake.tasks.banks import load_banks

# --- vPIC: deterministic normalization and matching ---------------------------


@pytest.mark.parametrize(
    ("raw", "folded"),
    [
        ("ŠKODA", "skoda"),
        ("Mercedes-Benz", "mercedes-benz"),
        ("  LAND   ROVER ", "land rover"),
        ("Citroën", "citroen"),
        ("MAZDA3", "mazda3"),
    ],
)
def test_fold_name(raw, folded):
    assert fold_name(raw) == folded


MAKES_BY_FOLD = {
    fold_name(make): make
    for make in ("TOYOTA", "ALFA", "ALFA ROMEO", "LAND ROVER", "MERCEDES-BENZ")
}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Toyota Camry", ("TOYOTA", "Camry")),
        # longest word-boundary prefix wins: "Alfa Romeo" beats "Alfa"
        ("Alfa Romeo Giulia Ti", ("ALFA ROMEO", "Giulia Ti")),
        ("Land Rover Discovery Sport P250", ("LAND ROVER", "Discovery Sport P250")),
        ("Mercedes-Benz GLB 250 4MATIC", ("MERCEDES-BENZ", "GLB 250 4MATIC")),
        ("Vauxhall Grandland X", None),  # unknown make
        ("Toyotaland Cruiser", None),  # prefix must be word-boundary
    ],
)
def test_split_make(value, expected):
    assert split_make(value, MAKES_BY_FOLD) == expected


@pytest.mark.parametrize(
    ("remainder", "models", "expected"),
    [
        ("Camry", ["Camry", "Corolla"], ("Camry", "exact")),
        # vPIC lists base models without trim: prefix match, longest first
        ("228i xDrive Gran Coupe", ["228", "228i"], ("228i", "prefix")),
        ("Corolla Cross", ["Corolla", "Corolla Cross"], ("Corolla Cross", "exact")),
        (
            "Corolla Cross Hybrid",
            ["Corolla", "Corolla Cross"],
            ("Corolla Cross", "prefix"),
        ),
        ("F-PACE P340", ["F-Pace"], ("F-Pace", "prefix")),
        ("Grandland X", ["Corsa"], None),
        # no mid-word matches
        ("Camryland", ["Camry"], None),
    ],
)
def test_match_model(remainder, models, expected):
    assert match_model(remainder, models) == expected


def test_vehicles_to_check_covers_values_and_decoys():
    banks = load_banks()
    pairs = vehicles_to_check(banks)
    assert pairs == sorted(set(pairs))
    assert ("value", "Toyota Camry") in pairs
    assert ("decoy", "Toyota Camry Cross") in pairs
    roles = {role for role, _ in pairs}
    assert roles == {"value", "decoy"}


# --- vPIC: the screen (network mocked) -----------------------------------------

UNKNOWN_MAKE = "Vauxhall"  # dropped from the mocked make catalog
FAILING_MAKE = "Honda"  # GetModelsForMake raises
EMPTY_MAKE = "Kia"  # make known, zero models listed
PREFIX_DECOY = "Toyota Camry Cross"  # matches via base-model prefix only


@pytest.fixture
def vpic_network(monkeypatch):
    """Deterministic vPIC layer built from the checked-in bank: every value
    grounds exactly, except one unknown make, one failing make, one make with
    no models, and one decoy demoted to a base-model prefix match."""
    import tau2.domains.intake.grounding_vpic as vpic_mod

    banks = load_banks()
    pairs = vehicles_to_check(banks)
    make_words = {value.split()[0] for _, value in pairs} - {UNKNOWN_MAKE}
    makes = sorted(make_words | {"Alfa Romeo", "Land Rover"})
    makes_by_fold = {fold_name(make): make for make in makes}
    default_models: dict[str, list[str]] = {}
    for _, value in pairs:
        split = split_make(value, makes_by_fold)
        if split is not None and split[1]:
            default_models.setdefault(split[0], []).append(split[1])
    # Demote the prefix decoy: its full remainder is not listed, its base is.
    default_models["Toyota"] = [
        model for model in default_models["Toyota"] if model != "Camry Cross"
    ]

    def fake_makes():
        return VpicMakesCatalog(makes=makes, sha256="cd" * 32)

    def fake_models(make):
        if make == FAILING_MAKE:
            raise ConnectionError("vpic down")
        if make == EMPTY_MAKE:
            return []
        return default_models.get(make, [])

    monkeypatch.setattr(vpic_mod, "fetch_vpic_makes", fake_makes)
    monkeypatch.setattr(vpic_mod, "fetch_vpic_models", fake_models)
    return vpic_mod


def test_vpic_report_verdicts_and_shape(vpic_network, tmp_path):
    banks = load_banks()
    pairs = vehicles_to_check(banks)
    report = run_vpic_screen(banks)

    by_value = {check.value: check for check in report.checks}
    assert set(by_value) == {value for _, value in pairs}

    unknown = by_value["Vauxhall Grandland X"]
    assert unknown.verdict is Verdict.FINDING
    assert unknown.make is None
    assert "vPIC make" in unknown.note

    failing = by_value["Honda Accord"]
    assert failing.verdict is Verdict.UNCHECKED
    assert failing.make == FAILING_MAKE

    empty = by_value["Kia Sportage"]
    assert empty.verdict is Verdict.FINDING
    assert "not listed" in empty.note

    exact = by_value["Toyota Camry"]
    assert exact.verdict is Verdict.PASS
    assert (exact.make, exact.matched_model, exact.match_kind) == (
        "Toyota",
        "Camry",
        "exact",
    )

    prefix = by_value[PREFIX_DECOY]
    assert prefix.verdict is Verdict.PASS
    assert (prefix.role, prefix.matched_model, prefix.match_kind) == (
        "decoy",
        "Camry",
        "prefix",
    )

    expected_findings = sum(
        value.startswith((f"{UNKNOWN_MAKE} ", f"{EMPTY_MAKE} ")) for _, value in pairs
    )
    expected_unchecked = sum(value.startswith(f"{FAILING_MAKE} ") for _, value in pairs)
    assert report.findings == expected_findings
    assert report.unchecked == expected_unchecked
    assert report.findings + report.unchecked + report.passes == len(pairs)

    # provenance
    assert set(report.bank_shas) == {"vehicles"}
    urls = {source.url for source in report.sources}
    assert any("vpic.nhtsa.dot.gov" in url for url in urls)
    makes_source = next(s for s in report.sources if "getallmakes" in s.url)
    assert makes_source.sha256 == "cd" * 32
    requested = {record.make for record in report.model_requests}
    assert FAILING_MAKE in requested and UNKNOWN_MAKE not in requested
    failed_request = next(r for r in report.model_requests if r.make == FAILING_MAKE)
    assert failed_request.error == "vpic down"
    assert failed_request.model_count is None
    assert all("GetModelsForMake" in r.url for r in report.model_requests)

    # report round-trips through the model and writes where asked
    out = write_vpic_report(report, out_path=tmp_path / "vpic.json")
    reread = VpicGroundingReport.model_validate_json(out.read_text())
    assert reread == report
    assert json.loads(out.read_text())["screen_version"] == report.screen_version


def test_vpic_screen_is_deterministic_up_to_the_date(vpic_network):
    banks = load_banks()
    first = run_vpic_screen(banks).model_dump(exclude={"screen_date"})
    second = run_vpic_screen(banks).model_dump(exclude={"screen_date"})
    assert first == second


def test_vpic_makes_fetch_failure_is_unchecked_not_pass(vpic_network, monkeypatch):
    def broken_makes():
        raise TimeoutError("nhtsa down")

    monkeypatch.setattr(vpic_network, "fetch_vpic_makes", broken_makes)
    report = run_vpic_screen(load_banks())
    assert all(check.verdict is Verdict.UNCHECKED for check in report.checks)
    assert report.model_requests == []
    makes_source = next(s for s in report.sources if "getallmakes" in s.url)
    assert makes_source.sha256 is None
    assert "nhtsa down" in makes_source.note


# --- RxNorm: deterministic parse and normalization -----------------------------


@pytest.mark.parametrize(
    ("value", "ingredients", "strength", "form"),
    [
        ("Metformin 850 mg tablet", ["Metformin"], "850 mg", "tablet"),
        (
            "Sulfamethoxazole-Trimethoprim 800-160 mg tablet",
            ["Sulfamethoxazole", "Trimethoprim"],
            "800-160 mg",
            "tablet",
        ),
        (
            "Mycophenolate mofetil 500 mg tablet",
            ["Mycophenolate mofetil"],
            "500 mg",
            "tablet",
        ),
        ("Tranexamic acid 650 mg tablet", ["Tranexamic acid"], "650 mg", "tablet"),
        ("Prilocaine 2% cream", ["Prilocaine"], "2%", "cream"),
        (
            "Prednisolone 15 mg/5 mL solution",
            ["Prednisolone"],
            "15 mg/5 mL",
            "solution",
        ),
        ("Guaifenesin 600 mg ER tablet", ["Guaifenesin"], "600 mg", "ER tablet"),
        ("Nitroglycerin 0.4 mg SL tablet", ["Nitroglycerin"], "0.4 mg", "SL tablet"),
        ("Levothyroxine 75 mcg tablet", ["Levothyroxine"], "75 mcg", "tablet"),
    ],
)
def test_parse_medication(value, ingredients, strength, form):
    parse = parse_medication(value)
    assert parse.ingredients == ingredients
    assert parse.strength == strength
    assert parse.form == form


@pytest.mark.parametrize(
    "value",
    [
        "500 mg tablet",  # no drug name
        "Metformin",  # no strength or form
        "Metformin 850 mg",  # no form
        "Metformin 850 mg suppository",  # form outside the closed map
        "Metformin 850 stone tablet",  # unit outside the bank grammar
    ],
)
def test_parse_medication_fails_loud(value):
    with pytest.raises(ValueError):
        parse_medication(value)


@pytest.mark.parametrize(
    ("strength", "expected"),
    [
        ("500 mg", [["500 mg"]]),
        ("0.5 mg", [["0.5 mg"]]),
        ("12.5 mg", [["12.5 mg"]]),
        ("1000 mg", [["1000 mg"]]),
        # RxNorm states microgram products in MG (levothyroxine et al.)
        ("75 mcg", [["75 mcg", "0.075 mg"]]),
        ("2%", [["20 mg/ml", "0.02 mg/mg", "2 %"]]),
        ("15 mg/5 mL", [["3 mg/ml"]]),
        ("800-160 mg", [["800 mg"], ["160 mg"]]),
        ("25-100 mg", [["25 mg"], ["100 mg"]]),
    ],
)
def test_normalize_strength(strength, expected):
    assert normalize_strength(strength) == expected


def test_normalize_strength_fails_loud_outside_the_grammar():
    for bad in ("mg", "500", "500 stone", "five mg"):
        with pytest.raises(ValueError):
            normalize_strength(bad)


@pytest.mark.parametrize(
    ("value", "concept", "matches"),
    [
        # salt-form names still ground the ingredient
        ("Metformin 850 mg tablet", "metformin hydrochloride 850 MG Oral Tablet", True),
        # digit boundaries: 50 mg never matches inside 850 mg
        (
            "Sertraline 50 mg tablet",
            "sertraline hydrochloride 850 MG Oral Tablet",
            False,
        ),
        ("Sertraline 50 mg tablet", "sertraline 50 MG Oral Tablet", True),
        # modified-release claims need the release phrase...
        ("Guaifenesin 600 mg ER tablet", "guaifenesin 600 MG Oral Tablet", False),
        (
            "Guaifenesin 600 mg ER tablet",
            "guaifenesin 600 MG Extended Release Oral Tablet",
            True,
        ),
        # ...while a plain tablet claim is satisfied by any oral tablet
        (
            "Metformin 500 mg tablet",
            "24 HR metformin hydrochloride 500 MG Extended Release Oral Tablet",
            True,
        ),
        # combination set semantics: pairing not enforced, both must appear
        (
            "Levodopa-Carbidopa 25-100 mg tablet",
            "carbidopa 25 MG / levodopa 100 MG Oral Tablet",
            True,
        ),
        (
            "Levodopa-Carbidopa 25-100 mg tablet",
            "carbidopa 25 MG / levodopa 250 MG Oral Tablet",
            False,
        ),
        # microgram values ground against RxNorm's MG rendering
        (
            "Levothyroxine 75 mcg tablet",
            "levothyroxine sodium 0.075 MG Oral Tablet",
            True,
        ),
        # percent strengths ground against the MG/ML rendering
        ("Prilocaine 2% cream", "prilocaine 20 MG/ML Topical Cream", True),
        ("Prilocaine 2% cream", "prilocaine 25 MG/ML Topical Cream", False),
        # sublingual and solution forms are specific
        (
            "Nitroglycerin 0.4 mg SL tablet",
            "nitroglycerin 0.4 MG Sublingual Tablet",
            True,
        ),
        ("Nitroglycerin 0.4 mg SL tablet", "nitroglycerin 0.4 MG Oral Tablet", False),
        (
            "Prednisolone 15 mg/5 mL solution",
            "prednisolone 3 MG/ML Oral Solution",
            True,
        ),
        (
            "Prednisolone 15 mg/5 mL solution",
            "prednisolone 3 MG/ML Ophthalmic Solution",
            False,
        ),
    ],
)
def test_match_product(value, concept, matches):
    parse = parse_medication(value)
    got = match_product([concept], parse)
    assert (got == concept) is matches
    if not matches:
        assert got is None


def test_match_product_prefers_the_shortest_concept_name():
    parse = parse_medication("Metformin 850 mg tablet")
    branded = "metformin hydrochloride 850 MG Oral Tablet [Glucophage]"
    plain = "metformin hydrochloride 850 MG Oral Tablet"
    assert match_product([branded, plain], parse) == plain


def test_checked_in_medications_bank_parses_completely():
    banks = load_banks()
    for role, value in medications_to_check(banks):
        parse = parse_medication(value)
        assert parse.ingredients, (role, value)
        assert parse.strength_alternatives, (role, value)
        assert parse.form_phrase in FORM_PHRASES.values(), (role, value)


# --- RxNorm: the screen (network mocked) ----------------------------------------

RXCUI_FAILS = "Clobazam"  # ingredient lookup raises -> UNCHECKED
UNKNOWN_INGREDIENT = "Guanfacine"  # decoy ingredient RxNorm does not know
WRONG_STRENGTH = "Naproxen 250 mg tablet"  # only a 500 mg product exists
DRUGS_FAIL = "Warfarin"  # product lookup raises -> UNCHECKED


@pytest.fixture
def rxnorm_network(monkeypatch):
    """Deterministic RxNorm layer built from the checked-in bank: every value
    grounds, except one failing ingredient lookup, one unknown ingredient,
    one strength mismatch, and one failing product lookup."""
    import tau2.domains.intake.grounding_rxnorm as rx_mod

    banks = load_banks()
    concepts: dict[str, list[str]] = {}
    for _, value in medications_to_check(banks):
        parse = parse_medication(value)
        alternatives = normalize_strength(parse.strength)
        if len(parse.ingredients) == len(alternatives):
            segments = [
                f"{ingredient.casefold()} {alts[0]}"
                for ingredient, alts in zip(parse.ingredients, alternatives)
            ]
        else:
            segments = [
                parse.ingredients[0].casefold()
                + " "
                + " ".join(alts[0] for alts in alternatives)
            ]
        name = " / ".join(segments) + " " + FORM_PHRASES[parse.form.casefold()]
        concepts.setdefault(parse.ingredients[0].casefold(), []).append(name)
    concepts["naproxen"] = ["naproxen 500 mg oral tablet"]

    def fake_rxcui(name):
        if name == RXCUI_FAILS:
            raise TimeoutError("nlm down")
        if name == UNKNOWN_INGREDIENT:
            return []
        return ["12345"]

    def fake_drugs(name):
        if name == DRUGS_FAIL:
            raise ConnectionError("boom")
        return concepts.get(name.casefold(), [])

    monkeypatch.setattr(rx_mod, "fetch_rxnorm_rxcui", fake_rxcui)
    monkeypatch.setattr(rx_mod, "fetch_rxnorm_drugs", fake_drugs)
    return rx_mod


def test_rxnorm_report_verdicts_and_shape(rxnorm_network, tmp_path):
    banks = load_banks()
    pairs = medications_to_check(banks)
    report = run_rxnorm_screen(banks)

    by_value = {check.value: check for check in report.checks}
    assert set(by_value) == {value for _, value in pairs}

    passing = by_value["Metformin 850 mg tablet"]
    assert passing.verdict is Verdict.PASS
    assert passing.matched_concept == "metformin 850 mg oral tablet"

    combo = by_value["Sulfamethoxazole-Trimethoprim 800-160 mg tablet"]
    assert combo.verdict is Verdict.PASS
    assert combo.ingredients == ["Sulfamethoxazole", "Trimethoprim"]

    unchecked_ingredient = by_value["Clobazam 10 mg tablet"]
    assert unchecked_ingredient.verdict is Verdict.UNCHECKED
    assert "nlm down" in unchecked_ingredient.note

    unknown = by_value["Guanfacine 1 mg tablet"]
    assert unknown.verdict is Verdict.FINDING
    assert unknown.role == "decoy"
    assert "ingredient unknown" in unknown.note

    mismatch = by_value[WRONG_STRENGTH]
    assert mismatch.verdict is Verdict.FINDING
    assert "strength+form" in mismatch.note

    unchecked_product = by_value["Warfarin 2 mg tablet"]
    assert unchecked_product.verdict is Verdict.UNCHECKED
    assert "boom" in unchecked_product.note

    assert report.findings == 2
    assert report.unchecked == 2
    assert report.findings + report.unchecked + report.passes == len(pairs)

    # provenance
    assert set(report.bank_shas) == {"medications"}
    urls = {source.url for source in report.sources}
    assert all("rxnav.nlm.nih.gov" in url for url in urls)

    # report round-trips through the model and writes where asked
    out = write_rxnorm_report(report, out_path=tmp_path / "rxnorm.json")
    reread = RxnormGroundingReport.model_validate_json(out.read_text())
    assert reread == report
    assert json.loads(out.read_text())["screen_version"] == report.screen_version


def test_rxnorm_screen_is_deterministic_up_to_the_date(rxnorm_network):
    banks = load_banks()
    first = run_rxnorm_screen(banks).model_dump(exclude={"screen_date"})
    second = run_rxnorm_screen(banks).model_dump(exclude={"screen_date"})
    assert first == second

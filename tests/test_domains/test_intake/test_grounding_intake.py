# Copyright Sierra
"""Tests for the intake grounding program: the load-time fictional phone-range
validator (grounding.py) and the negative reality screen (grounding_screen.py,
``tau2 intake-grounding screen``) with the network layer mocked."""

import json
import shutil

import pytest
import yaml

from tau2.domains.intake.grounding import (
    FICTIONAL_PHONE_RANGES,
    is_fictional_phone,
    validate_fictional_phones,
)
from tau2.domains.intake.grounding_screen import (
    DohLookup,
    EdgarCatalog,
    GroundingScreenReport,
    Verdict,
    email_domains_to_screen,
    extract_carrier,
    registry_names_to_screen,
    run_screen,
    write_screen_report,
)
from tau2.domains.intake.tasks.banks import INTAKE_BANKS_DIR, load_banks

# --- Range table -------------------------------------------------------------

ACCEPT = [
    # NANP fictional 555-0100..0199, any area code (NANPA / ATIS-0300115)
    "(206) 555-0111",
    "(415) 555-0100",
    "(213) 555-0199",
    # ... with a PBX extension suffix
    "(605) 555-0117 ext. 2",
    "(605) 555-0177 ext. 10",
    "(605) 555-0137 ext. 999",
    # UK Ofcom drama range, London
    "+44 20 7946 0000",
    "+44 20 7946 0999",
    "+44 20 7946 0455",
    # AU ACMA fictitious range
    "+61 2 5550 0000",
    "+61 2 5550 9999",
    "+61 2 5550 4796",
    # FR ARCEP fictional mobile range
    "+33 6 39 98 00 00",
    "+33 6 39 98 99 99",
    "+33 6 39 98 06 51",
]

REJECT = [
    # NANP outside the fictional hundred-block
    "(206) 555-0211",
    "(206) 555-1234",
    "(212) 867-5309",
    "206 555-0111",  # not the pinned formatting
    "(206) 555-011",  # short
    "(206) 555-01111",  # long
    # extension violations
    "(605) 555-0117 ext. 0",  # extensions start at 1
    "(605) 555-0117 ext. A",
    "(605) 555-0117 ext 2",  # missing the dot
    "(605) 555-0117 ext. 12345",  # over 4 digits
    # UK outside the drama block
    "+44 20 7946 1000",
    "+44 20 7947 0123",
    "+44 20 7946 045",
    # AU outside the fictitious block
    "+61 2 5551 0000",
    "+61 3 5550 0000",
    "+61 2 5550 123",
    # FR outside the fictional mobile block
    "+33 6 39 99 00 00",
    "+33 7 39 98 00 00",
    "+33 6 39 98 0 00",
    "",
]


@pytest.mark.parametrize("value", ACCEPT)
def test_fictional_ranges_accept(value):
    assert is_fictional_phone(value), value


@pytest.mark.parametrize("value", REJECT)
def test_fictional_ranges_reject(value):
    assert not is_fictional_phone(value), value


def test_range_table_covers_all_four_regulators():
    names = " ".join(name for name, _ in FICTIONAL_PHONE_RANGES)
    for regulator in ("NANP", "Ofcom", "ACMA", "ARCEP"):
        assert regulator in names


def test_validate_fictional_phones_names_the_offender():
    with pytest.raises(ValueError, match=r"\(212\) 867-5309"):
        validate_fictional_phones(["(206) 555-0111", "(212) 867-5309"])
    validate_fictional_phones(ACCEPT)  # no raise


# --- Bank-load wiring ----------------------------------------------------------


def test_checked_in_phones_bank_is_fully_fictional():
    banks = load_banks()
    for entry in banks.phones:
        for value in (entry.value, *entry.decoys):
            assert is_fictional_phone(value), value


@pytest.fixture
def banks_copy(tmp_path):
    target = tmp_path / "banks"
    shutil.copytree(INTAKE_BANKS_DIR, target)
    return target


def corrupt_phones(banks_dir, mutate):
    path = banks_dir / "phones.yaml"
    raw = yaml.safe_load(path.read_text())
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True))


def test_out_of_range_phone_value_fails_load(banks_copy):
    def mutate(raw):
        raw["entries"][0]["value"] = "(212) 867-5309"

    corrupt_phones(banks_copy, mutate)
    with pytest.raises(ValueError, match=r"fictional range.*\(212\) 867-5309"):
        load_banks(banks_copy)


def test_out_of_range_phone_decoy_fails_load(banks_copy):
    def mutate(raw):
        raw["entries"][3]["decoys"].append("+44 20 7946 1000")

    corrupt_phones(banks_copy, mutate)
    with pytest.raises(ValueError, match=r"fictional range.*7946 1000"):
        load_banks(banks_copy)


# --- Carrier extraction ----------------------------------------------------------


@pytest.mark.parametrize(
    ("plan", "carrier"),
    [
        ("Tessara Silver PPO", "Tessara"),
        ("Corvid Shield Classic", "Corvid Shield"),
        ("Kestrel Care Select", "Kestrel"),
        ("Ferrow Health Direct", "Ferrow"),
        ("Tessara K-Bridge 4500 EPO", "Tessara"),
        ("Norvexa X-Tier 6 Catastrophic", "Norvexa"),
        ("Solivar 3Z Metallic Bronze-Select", "Solivar"),
        ("Bellmarsh HDX-21 Consumer Choice", "Bellmarsh"),
        # fully-descriptor fallback: still screens the first token
        ("Standard HMO", "Standard"),
    ],
)
def test_extract_carrier(plan, carrier):
    assert extract_carrier(plan) == carrier


def test_screen_inputs_cover_the_banks():
    banks = load_banks()
    domains = email_domains_to_screen(banks)
    assert domains == sorted(set(domains))
    assert all("." in d for d in domains)
    pairs = registry_names_to_screen(banks)
    assert pairs == sorted(set(pairs))
    screened_banks = {bank for bank, _ in pairs}
    assert screened_banks == {"coined", "properties", "shops", "insurance_plans"}
    names = {name for _, name in pairs}
    assert "Trivexa Labs" in names  # coined value
    assert "Tessara" in names  # extracted carrier


# --- The screen (network mocked) --------------------------------------------------

RESOLVING_DOMAIN = "drovenpost.com"
EDGAR_HIT = "Trivexa Labs"
GLEIF_HIT = "Ironwood Garage"
GLEIF_FAILS = "Silverpine Lodge"


@pytest.fixture
def mocked_network(monkeypatch):
    """Deterministic network layer: one resolving domain, one EDGAR hit, one
    GLEIF hit, one GLEIF transport failure; everything else clean."""
    import tau2.domains.intake.grounding_screen as screen_mod

    def fake_doh(domain, rrtype):
        if domain == RESOLVING_DOMAIN and rrtype in ("A", "MX"):
            return DohLookup(rrtype=rrtype, status=0, answers=["203.0.113.7"])
        return DohLookup(rrtype=rrtype, status=3, answers=[])

    def fake_edgar():
        return EdgarCatalog(titles=["Apple Inc.", EDGAR_HIT.upper()], sha256="ab" * 32)

    def fake_gleif(name):
        if name == GLEIF_FAILS:
            raise ConnectionError("boom")
        if name == GLEIF_HIT:
            return ["IRONWOOD GARAGE", "Ironwood Garage Holdings"]
        return [f"{name} Holdings PLC"]  # near miss, never exact

    monkeypatch.setattr(screen_mod, "doh_resolve", fake_doh)
    monkeypatch.setattr(screen_mod, "fetch_edgar_catalog", fake_edgar)
    monkeypatch.setattr(screen_mod, "gleif_legal_names", fake_gleif)
    return screen_mod


def test_screen_report_verdicts_and_shape(mocked_network, tmp_path):
    banks = load_banks()
    report = run_screen(banks)

    by_domain = {c.domain: c for c in report.dns_checks}
    assert set(by_domain) == set(email_domains_to_screen(banks))
    resolving = by_domain[RESOLVING_DOMAIN]
    assert resolving.verdict is Verdict.FINDING
    assert not resolving.nxdomain
    clean = next(c for d, c in by_domain.items() if d != RESOLVING_DOMAIN)
    assert clean.verdict is Verdict.PASS
    assert clean.nxdomain
    assert {lookup.rrtype for lookup in clean.lookups} == {"A", "AAAA", "MX"}

    by_name = {c.name: c for c in report.registry_checks}
    edgar_hit = by_name[EDGAR_HIT]
    assert edgar_hit.edgar is Verdict.FINDING  # case-folded match
    assert edgar_hit.edgar_matches == [EDGAR_HIT.upper()]
    assert edgar_hit.verdict is Verdict.FINDING
    gleif_hit = by_name[GLEIF_HIT]
    assert gleif_hit.gleif is Verdict.FINDING
    assert gleif_hit.gleif_matches == ["IRONWOOD GARAGE"]
    failed = by_name[GLEIF_FAILS]
    assert failed.gleif is Verdict.UNCHECKED
    assert failed.gleif_error == "boom"
    assert failed.verdict is Verdict.UNCHECKED
    clean_name = by_name["Tessara"]
    assert (clean_name.edgar, clean_name.gleif) == (Verdict.PASS, Verdict.PASS)
    assert clean_name.verdict is Verdict.PASS

    total = len(report.dns_checks) + len(report.registry_checks)
    assert report.findings + report.unchecked + report.passes == total
    assert report.findings == 3  # domain + EDGAR name + GLEIF name
    assert report.unchecked == 1

    # provenance
    assert set(report.bank_shas) == {
        "coined",
        "properties",
        "shops",
        "insurance_plans",
        "emails",
    }
    urls = {source.url for source in report.sources}
    assert any("dns.google" in url for url in urls)
    assert any("sec.gov" in url for url in urls)
    assert any("gleif.org" in url for url in urls)
    edgar_source = next(s for s in report.sources if "sec.gov" in s.url)
    assert edgar_source.sha256 == "ab" * 32

    # report round-trips through the model and writes where asked
    out = write_screen_report(report, out_path=tmp_path / "screen.json")
    reread = GroundingScreenReport.model_validate_json(out.read_text())
    assert reread == report
    assert json.loads(out.read_text())["screen_version"] == report.screen_version


def test_screen_is_deterministic_up_to_the_date(mocked_network):
    banks = load_banks()
    first = run_screen(banks).model_dump(exclude={"screen_date"})
    second = run_screen(banks).model_dump(exclude={"screen_date"})
    assert first == second


def test_edgar_fetch_failure_is_unchecked_not_pass(mocked_network, monkeypatch):
    def broken_edgar():
        raise TimeoutError("sec down")

    monkeypatch.setattr(mocked_network, "fetch_edgar_catalog", broken_edgar)
    report = run_screen(load_banks())
    assert all(c.edgar is Verdict.UNCHECKED for c in report.registry_checks)
    assert all(c.edgar_matches == [] for c in report.registry_checks)
    edgar_source = next(s for s in report.sources if "sec.gov" in s.url)
    assert edgar_source.sha256 is None
    assert "sec down" in edgar_source.note

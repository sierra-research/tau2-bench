# Copyright Sierra
"""Authoritative localized caller-name resolution for nativeness checks."""

import json
from pathlib import Path

import pytest
import yaml

import tau2.judges.nativeness.caller_identity as caller_identity_module
from tau2.data_model.tasks import Task
from tau2.judges.nativeness.caller_identity import (
    KOREAN_CALLER_NAME_PROVENANCE,
    KOREAN_FAMILY_NAME_ALIASES,
    KOREAN_GIVEN_NAME_ALIASES,
    resolve_korean_caller_name,
)
from tau2.multilingual.domain_profiles import DOMAIN_PROFILES

KO_DIR = Path("data/tau2/multilingual/ko")

# Domains with Korean locale-identity task variants. ``identity_swap_supported``
# is the profile flag that owns this resolution guarantee.
KO_IDENTITY_DOMAINS = [
    domain
    for domain, profile in DOMAIN_PROFILES.items()
    if profile.identity_swap_supported
]


def _first_task(domain: str, *, identity: bool) -> Task:
    suffix = "_identity" if identity else ""
    path = KO_DIR / f"{domain}_tasks_ko{suffix}.json"
    return Task.model_validate(json.loads(path.read_text())[0])


def test_reviewed_spoken_aliases_exactly_cover_fixed_korean_name_pools():
    corpus = yaml.safe_load((KO_DIR / "locale_corpus.yaml").read_text())
    assert set(KOREAN_GIVEN_NAME_ALIASES) == set(corpus["female_first_names"]) | set(
        corpus["male_first_names"]
    )
    assert set(KOREAN_FAMILY_NAME_ALIASES) == set(corpus["last_names"])
    assert all(KOREAN_GIVEN_NAME_ALIASES.values())
    assert all(KOREAN_FAMILY_NAME_ALIASES.values())


def test_identity_swap_flag_matches_shipped_ko_identity_data():
    # The flag and the data must agree in both directions: a swap-supported
    # domain without ko identity data is an unshipped promise, and shipped
    # data without the flag would dodge the resolution guarantee below.
    for domain, profile in DOMAIN_PROFILES.items():
        path = KO_DIR / f"{domain}_tasks_ko_identity.json"
        assert path.exists() == profile.identity_swap_supported


def test_identity_task_shapes_resolve_role_preserving_korean_names():
    # Every identity-swap-supported domain, so a new profile entry opting in
    # must ship a working extraction (and ko identity data) before its runs
    # can be judged.
    for domain in KO_IDENTITY_DOMAINS:
        task = _first_task(domain, identity=True)
        name = resolve_korean_caller_name(task, "ko", domain)
        assert name is not None
        assert name.full_romanized == f"{name.family_romanized} {name.given_romanized}"
        assert name.full_spoken == f"{name.family_spoken}{name.given_spoken}"
        assert name.provenance == KOREAN_CALLER_NAME_PROVENANCE


def test_plain_foreign_source_tasks_and_unknown_context_never_resolve():
    for domain in KO_IDENTITY_DOMAINS:
        task = _first_task(domain, identity=False)
        assert resolve_korean_caller_name(task, "ko", domain) is None
    identity = _first_task("airline", identity=True)
    assert resolve_korean_caller_name(identity, "es", "airline") is None
    assert resolve_korean_caller_name(identity, "ko", None) is None


def test_unregistered_domain_raises_instead_of_silently_skipping():
    identity = _first_task("airline", identity=True)
    with pytest.raises(KeyError, match="Unknown multilingual domain"):
        resolve_korean_caller_name(identity, "ko", "banking_knowledge")


def test_unsupported_identity_kind_raises_instead_of_silently_skipping(monkeypatch):
    identity = _first_task("airline", identity=True)
    profile = DOMAIN_PROFILES["airline"].model_copy()
    monkeypatch.setattr(
        caller_identity_module, "get_domain_profile", lambda domain: profile
    )
    # A future CallerIdentityKind member the extraction was never taught.
    monkeypatch.setattr(profile, "caller_identity", "structured_household_account")
    with pytest.raises(NotImplementedError, match="no Korean name-roles extraction"):
        resolve_korean_caller_name(identity, "ko", "airline")

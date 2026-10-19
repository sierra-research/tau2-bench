import re
from collections import Counter

import pytest

from tau2.domains.loan_servicing.environment import get_environment
from tau2.domains.loan_servicing.utils import (
    LOAN_SERVICING_POLICY_FR_PATH,
    LOAN_SERVICING_POLICY_PATH,
)
from tau2.registry import registry

DOMAINS = ["loan_servicing", "loan_servicing_fr"]


def make_env(domain: str):
    return registry.get_env_constructor(domain)()


def read_policy(path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_is_registered_with_its_own_name(domain):
    environment = make_env(domain)
    assert environment.get_domain_name() == domain
    assert "Boréal Finance" in environment.get_policy()


def test_policies_differ_by_language():
    assert make_env("loan_servicing").get_policy() == read_policy(
        LOAN_SERVICING_POLICY_PATH
    )
    assert make_env("loan_servicing_fr").get_policy() == read_policy(
        LOAN_SERVICING_POLICY_FR_PATH
    )
    assert "Politique" in make_env("loan_servicing_fr").get_policy()


def test_domains_share_identical_tools():
    en_tools = make_env("loan_servicing").get_tools()
    fr_tools = make_env("loan_servicing_fr").get_tools()
    assert [t.openai_schema for t in en_tools] == [t.openai_schema for t in fr_tools]


def test_domains_share_identical_db_hashes():
    en = make_env("loan_servicing")
    fr = make_env("loan_servicing_fr")
    assert en.get_db_hash() == fr.get_db_hash()
    before = en.get_db_hash()
    for environment in (en, fr):
        environment.use_tool(
            "transfer_to_human_agents", reason="complaint", summary="Anything"
        )
    assert en.get_db_hash() != before
    assert en.get_db_hash() == fr.get_db_hash()


def test_domains_share_tasks_and_splits():
    en_tasks = registry.get_tasks_loader("loan_servicing")(None)
    fr_tasks = registry.get_tasks_loader("loan_servicing_fr")(None)
    assert en_tasks == fr_tasks
    assert (
        registry.get_task_splits_loader("loan_servicing")()
        == registry.get_task_splits_loader("loan_servicing_fr")()
    )


def test_invalid_policy_language_raises():
    with pytest.raises(ValueError):
        get_environment(policy_language="de")


def test_french_policy_keeps_every_code_and_number():
    en = read_policy(LOAN_SERVICING_POLICY_PATH)
    fr = read_policy(LOAN_SERVICING_POLICY_FR_PATH)
    codes = re.compile(r"`[^`]+`")
    numbers = re.compile(r"\d{4}-\d{2}-\d{2}|\$\d+\.\d{2}|\d+")
    headings = re.compile(r"^#+ ", re.MULTILINE)
    for pattern in (codes, numbers, headings):
        assert Counter(pattern.findall(en)) == Counter(pattern.findall(fr))

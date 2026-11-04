"""Domain-owned regional extensions; frozen existing-domain runtime code stays intact."""

from pathlib import Path

from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.regional import (
    BankingPublishedRegionalBundle,
    BankingScenarioBundle,
)
from tau2.domains.intake.regional import IntakeScenarioBundle, PublishedIntakeBundle
from tau2.multilingual.regional import PublishedRegionalBundle, ScenarioBundle

RegionalBundle = (
    PublishedIntakeBundle | BankingPublishedRegionalBundle | PublishedRegionalBundle
)
Scenario = (
    RegionalBundle | IntakeScenarioBundle | BankingScenarioBundle | ScenarioBundle
)


def bundle_model(domain: str) -> type[PublishedRegionalBundle]:
    """Select the domain's published payload contract before validating its data."""
    if domain == "intake":
        return PublishedIntakeBundle
    if domain == "banking_knowledge":
        return BankingPublishedRegionalBundle
    return PublishedRegionalBundle


def canonical_scenario(domain: str, tasks: list[Task], data_root: Path) -> Scenario:
    """Bind source tasks using their domain-owned control builder."""
    if domain == "intake":
        from tau2.domains.intake.regional import canonical_scenario as intake

        return intake(tasks, data_root)
    if domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.regional import (
            canonical_scenario as banking,
        )

        return banking(tasks, data_root)
    from tau2.multilingual.regional import canonical_scenario as established

    return established(domain, tasks, data_root)


def scenario_environment_kwargs(bundle: Scenario) -> dict:
    """Construct isolated domain state while preserving dynamic retrieval policies."""
    if isinstance(bundle, IntakeScenarioBundle):
        from tau2.domains.intake.regional import scenario_environment_kwargs as intake

        return intake(bundle)
    if isinstance(bundle, BankingScenarioBundle):
        from tau2.domains.banking_knowledge.regional import (
            scenario_environment_kwargs as banking,
        )

        return banking(bundle)
    from tau2.multilingual.regional import scenario_environment_kwargs as established

    return established(bundle)


def runtime_rules_digest(domain: str) -> str:
    """Pin new-domain routing/prompts without changing established-domain hashes."""
    import hashlib

    from tau2.multilingual.regional import runtime_rules_digest as established

    base = established(domain)
    if domain not in {"banking_knowledge", "intake"}:
        return base
    digest = hashlib.sha256(base.encode())
    root = Path(__file__).parent
    for name in ("scenario_dispatch.py", "agent_guidance.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    if domain == "intake":
        for name in (
            "agent/base/voice.py",
            "data_model/voice.py",
            "voice/utils/pronunciation_swap.py",
            "runner/build.py",
            "runner/elicitation.py",
            "runner/helpers.py",
            "multilingual/runtime.py",
        ):
            digest.update(name.encode())
            digest.update((root.parent / name).read_bytes())
    return digest.hexdigest()

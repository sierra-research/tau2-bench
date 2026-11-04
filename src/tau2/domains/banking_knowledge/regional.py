"""Banking regional scenarios extend the shared contract without changing old domains."""

import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.data_model import (
    Document,
    KnowledgeBase,
    TransactionalDB,
)
from tau2.domains.banking_knowledge.localization import knowledge_digest
from tau2.multilingual.regional import PublishedRegionalBundle, ScenarioBundle


def _policies(directory: Path) -> dict[str, str]:
    return {
        p.name: p.read_text()
        for p in sorted((directory / "prompts").glob("*"))
        if p.is_file()
    }


class BankingScenarioBundle(ScenarioBundle):
    """Transactional state plus pinned English KB and localized entity overlays."""

    domain: Literal["banking_knowledge"] = "banking_knowledge"
    knowledge_base_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Exact English KB file-manifest digest."
    )
    knowledge_document_overrides: dict[str, Document] = Field(
        default_factory=dict,
        description="Approved entity-localized versions of changed English KB documents.",
    )

    @model_validator(mode="after")
    def domain_contract(self):
        """Banking has dynamic retrieval policies and no separate caller database."""
        TransactionalDB.model_validate(self.database)
        if self.user_database is not None:
            raise ValueError(
                "Banking uses one transactional database for agent and caller"
            )
        if not self.policies:
            raise ValueError("Banking requires pinned English retrieval templates")
        if not self.tasks or len({t.id for t in self.tasks}) != len(self.tasks):
            raise ValueError("Banking scenario has empty or duplicate task IDs")
        return self


class BankingPublishedRegionalBundle(BankingScenarioBundle, PublishedRegionalBundle):
    """Banking-specific payload with ordinary digest approval and task metadata."""

    @model_validator(mode="after")
    def banking_currency(self):
        """Locale identity never changes the financial experiment."""
        if (
            self.settings.currency_code != "USD"
            or Decimal(self.settings.currency_scale) != 1
        ):
            raise ValueError("Banking requires unchanged USD amounts and scale 1")
        return self


def canonical_scenario(tasks: list[Task], data_root: Path) -> BankingScenarioBundle:
    """Load a control scenario with the same English KB and dynamic policy inputs."""
    directory = data_root / "tau2/domains/banking_knowledge"
    return BankingScenarioBundle(
        database=json.loads((directory / "db.json").read_text()),
        tasks=tasks,
        policies=_policies(directory),
        knowledge_base_sha256=knowledge_digest(directory),
    )


def scenario_environment_kwargs(bundle: BankingScenarioBundle) -> dict:
    """Return fresh DB and selected KB; retrieval owns policy and read logs."""
    from tau2.domains.banking_knowledge.utils import KNOWLEDGE_DOCUMENTS_DIR
    from tau2.environment.regional import REGIONAL_RUNTIME_VERSION

    directory = Path(KNOWLEDGE_DOCUMENTS_DIR).parent
    if knowledge_digest(directory) != bundle.knowledge_base_sha256:
        raise ValueError(
            "English knowledge dependency changed since banking publication"
        )
    if _policies(directory) != bundle.policies:
        raise ValueError(
            "English retrieval templates changed since banking publication"
        )
    if (
        isinstance(bundle, BankingPublishedRegionalBundle)
        and bundle.runtime_rules_version != REGIONAL_RUNTIME_VERSION
    ):
        raise ValueError("Banking bundle requires different runtime rules")
    kwargs = {"db": TransactionalDB.model_validate(bundle.database)}
    if bundle.knowledge_document_overrides:
        from tau2.domains.banking_knowledge.environment import get_knowledge_base

        source = get_knowledge_base()
        documents = dict(source.documents)
        for identifier, replacement in bundle.knowledge_document_overrides.items():
            if identifier not in documents or replacement.id != identifier:
                raise ValueError("Localized knowledge document has an unknown ID")
            documents[identifier] = replacement
        kwargs["knowledge_base"] = KnowledgeBase(documents=documents)
    return kwargs

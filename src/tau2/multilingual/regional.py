"""Published regional scenarios and their domain-owned runtime bindings."""

import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from tau2.data_model.language import PackModel, TaskMetadata
from tau2.data_model.tasks import Task
from tau2.multilingual.caller_identity import TaskCallerIdentity, TelecomCallerRoles
from tau2.multilingual.localization import RegionalSettings


class ScenarioBundle(PackModel):
    """One inseparable database, task and policy selection."""

    domain: Literal["airline", "retail", "telecom"]
    database: dict[str, JsonValue] = Field(
        description="Domain-validated database snapshot."
    )
    user_database: dict[str, JsonValue] | None = Field(
        default=None, description="Separate caller-tool database, when required."
    )
    tasks: list[Task] = Field(
        description="Frozen canonical task identities with selected scenario values."
    )
    policies: dict[str, str] = Field(
        description="Policy documents keyed by their source document names."
    )

    @model_validator(mode="after")
    def domain_contract(self):
        """Reject incomplete or wrong-domain bundles before provider work."""
        from tau2.domains.airline.data_model import FlightDB
        from tau2.domains.retail.data_model import RetailDB
        from tau2.domains.telecom.data_model import TelecomDB
        from tau2.domains.telecom.user_data_model import TelecomUserDB

        {"airline": FlightDB, "retail": RetailDB, "telecom": TelecomDB}[
            self.domain
        ].model_validate(self.database)
        expected = (
            {"main_policy", "tech_support_manual", "tech_support_workflow"}
            if self.domain == "telecom"
            else {"policy"}
        )
        if self.policies.keys() != expected:
            raise ValueError("scenario bundle has incomplete policy documents")
        if self.domain == "telecom":
            TelecomUserDB.model_validate(self.user_database)
        elif self.user_database is not None:
            raise ValueError("domain does not have a separate caller database")
        if not self.tasks or len({t.id for t in self.tasks}) != len(self.tasks):
            raise ValueError("scenario bundle has empty or duplicate task IDs")
        return self


class PublishedRegionalBundle(ScenarioBundle):
    """Native scenario content compiled from one digest-approved map."""

    schema_version: Literal["regional-bundle-v2"] = "regional-bundle-v2"
    compiler_version: Literal["regional-publication-v2"] = "regional-publication-v2"
    policy_clock_rule: Literal["source-wall-clock-v1"] = Field(
        default="source-wall-clock-v1",
        description="Publication preserves the canonical policy wall clock and substitutes only the regional timezone; dates and tool clocks stay aligned.",
    )
    revision: str = Field(min_length=1, description="Reviewed map revision.")
    settings: RegionalSettings
    localization_map_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Exact source map content digest."
    )
    source_document_sha256: dict[str, str] = Field(
        description="Content digests identifying the approved data, policies and "
        "original tool-evidence source documents."
    )
    metadata: dict[str, TaskMetadata]
    airports: dict[str, str] = Field(
        default_factory=dict, description="Regional IATA to regional city catalog."
    )
    runtime_rules_version: str = Field(
        description="Explicit executable regional behavior version."
    )
    caller_roles: TelecomCallerRoles | None = Field(
        default=None,
        description="Explicit canonical gender roles used for telecom caller rotation.",
    )
    caller_identities: list[TaskCallerIdentity] = Field(
        default_factory=list,
        description="Derived per-task identities from approved regional names.",
    )
    airport_catalog_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Exact source airport city catalog content digest.",
    )

    @model_validator(mode="after")
    def assignments(self):
        if set(self.metadata) != {task.id for task in self.tasks}:
            raise ValueError("regional metadata must cover all tasks exactly")
        if self.domain == "airline":
            if self.airport_catalog_sha256 is None:
                raise ValueError(
                    "regional airline publication requires an airport catalog digest"
                )
            codes = {
                flight[key]
                for flight in self.database["flights"].values()
                for key in ("origin", "destination")
            }
            if codes != self.airports.keys():
                raise ValueError(
                    "regional airport directory must cover the flight graph"
                )
        elif self.airports or self.airport_catalog_sha256 is not None:
            raise ValueError("only airline scenarios have an airport directory")
        if self.caller_roles is not None:
            if self.domain != "telecom" or {
                row.task_id for row in self.caller_identities
            } != set(self.metadata):
                raise ValueError(
                    "caller role provenance must cover exactly the telecom tasks"
                )
            if len(self.caller_identities) != len(self.metadata) or any(
                row.gender != self.metadata[row.task_id].caller_gender
                for row in self.caller_identities
            ):
                raise ValueError(
                    "caller provenance conflicts with task gender assignments"
                )
        elif self.caller_identities:
            raise ValueError("caller identities require explicit source roles")
        return self


def canonical_scenario(
    domain: str, tasks: list[Task], data_root: Path
) -> ScenarioBundle:
    """Bind selected control tasks to the canonical database and policy snapshot."""
    directory = data_root / "tau2" / "domains" / domain
    if domain == "telecom":
        database = tomllib.loads((directory / "db.toml").read_text())
        user_database = tomllib.loads((directory / "user_db.toml").read_text())
        names = ("main_policy", "tech_support_manual", "tech_support_workflow")
    else:
        database = json.loads((directory / "db.json").read_text())
        user_database = None
        names = ("policy",)
    return ScenarioBundle(
        domain=domain,
        database=database,
        user_database=user_database,
        tasks=tasks,
        policies={name: (directory / f"{name}.md").read_text() for name in names},
    )


def runtime_rules_digest(domain: str) -> str:
    """Fingerprint executable domain settings as well as their declared version."""
    from tau2.environment.regional import REGIONAL_RUNTIME_VERSION

    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "environment" / "regional.py",
        root / "multilingual" / "regional.py",
        *sorted((root / "domains" / domain).glob("*.py")),
    ]
    digest = hashlib.sha256(REGIONAL_RUNTIME_VERSION.encode())
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def scenario_environment_kwargs(bundle: ScenarioBundle) -> dict:
    """Build fresh domain models for each live environment or evaluator replay."""
    from tau2.domains.airline.data_model import FlightDB
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.telecom.data_model import TelecomDB
    from tau2.domains.telecom.user_data_model import TelecomUserDB

    kwargs = {
        "db": {"airline": FlightDB, "retail": RetailDB, "telecom": TelecomDB}[
            bundle.domain
        ].model_validate(bundle.database)
    }
    if bundle.domain == "telecom":
        kwargs["user_db"] = TelecomUserDB.model_validate(bundle.user_database)
        kwargs["policy"] = (
            "<main_policy>\n"
            + bundle.policies["main_policy"]
            + "\n</main_policy>\n<tech_support_policy>\n"
            + bundle.policies["tech_support_manual"]
            + "\n</tech_support_policy>"
        )
    else:
        kwargs["policy"] = bundle.policies["policy"]
    if isinstance(bundle, PublishedRegionalBundle):
        from tau2.environment.regional import (
            REGIONAL_RUNTIME_VERSION,
            DomainRegionalSettings,
        )

        if bundle.runtime_rules_version != REGIONAL_RUNTIME_VERSION:
            raise ValueError(
                "published regional bundle requires different runtime rules"
            )
        settings = bundle.settings.model_dump(
            include={"country_code", "currency_code", "currency_scale", "time_zone"}
        )
        if bundle.domain == "airline":
            from tau2.domains.airline.region import AirlineRegionalSettings

            kwargs["regional_settings"] = AirlineRegionalSettings(
                **settings,
                current_datetime=datetime(2024, 5, 15, 15),
                airports=[
                    {"iata": code, "city": city}
                    for code, city in bundle.airports.items()
                ],
            )
        elif bundle.domain == "telecom":
            from tau2.domains.telecom.region import TelecomRegionalSettings

            kwargs["regional_settings"] = TelecomRegionalSettings(
                **settings, current_datetime=datetime(2025, 2, 25, 12, 8)
            )
        else:
            kwargs["regional_settings"] = DomainRegionalSettings(**settings)
    return kwargs

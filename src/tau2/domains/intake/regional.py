"""Intake scenario binding without changing existing domains' runtime fingerprints."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tau2.data_model.tasks import Task
from tau2.domains.intake.data_model import IntakeDB
from tau2.domains.intake.localized import IntakeLocalizationData
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.multilingual.regional import PublishedRegionalBundle, ScenarioBundle

INTAKE_POLICY_NAMES = (
    "agent_directed_policy",
    "scaffolded_policy",
    "scaffolded_policy_text",
)


class IntakeScenarioBundle(ScenarioBundle):
    """One intake DB, private DB, frozen task selection and strategy policies."""

    domain: Literal["intake"] = "intake"
    intake_localization: IntakeLocalizationData | None = Field(
        default=None,
        description="Reviewed selected dependencies; absent for the English source control.",
    )

    @model_validator(mode="after")
    def domain_contract(self):
        """Validate the domain-owned models instead of three-domain assumptions."""
        IntakeDB.model_validate(self.database)
        IntakeUserDB.model_validate(self.user_database)
        if self.policies.keys() != set(INTAKE_POLICY_NAMES):
            raise ValueError(
                "Intake scenario requires both strategies and both channel policies"
            )
        if not self.tasks or len({task.id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Intake scenario contains empty or duplicate task IDs")
        return self


class PublishedIntakeBundle(IntakeScenarioBundle, PublishedRegionalBundle):
    """Digest-approved native selection using the ordinary publication contract."""

    @model_validator(mode="after")
    def intake_assignments(self):
        if self.intake_localization is None:
            raise ValueError("Published intake requires reviewed selected stimuli")
        if {row.canonical_task_id for row in self.intake_localization.stimuli} != {
            row.canonical_task_id for row in self.metadata.values()
        }:
            raise ValueError(
                "Intake stimuli must cover exactly the canonical frozen selection"
            )
        by_canonical = {
            self.metadata[task.id].canonical_task_id: task for task in self.tasks
        }
        for row in self.intake_localization.stimuli:
            task = by_canonical[row.canonical_task_id]
            actions = (
                task.evaluation_criteria.actions if task.evaluation_criteria else []
            )
            submissions = [
                action for action in actions or [] if action.name == "submit_fields"
            ]
            if (
                len(submissions) != 1
                or submissions[0].arguments.get("fields", {}).get(row.field_name)
                != row.written
            ):
                raise ValueError(
                    "Published intake task disagrees with its selected written stimulus"
                )
        return self


def canonical_scenario(tasks: list[Task], data_root: Path) -> IntakeScenarioBundle:
    """Bind English tasks without redrawing banks or adapting their strategy."""
    directory = data_root / "tau2" / "domains" / "intake"
    return IntakeScenarioBundle(
        database=tomllib.loads((directory / "db.toml").read_text()),
        user_database=tomllib.loads((directory / "user_db.toml").read_text()),
        tasks=tasks,
        policies={
            name: (directory / f"{name}.md").read_text() for name in INTAKE_POLICY_NAMES
        },
    )


def scenario_environment_kwargs(bundle: IntakeScenarioBundle) -> dict:
    """Fresh agent/caller state; the environment chooses the run's channel/strategy."""
    return {
        "db": IntakeDB.model_validate(bundle.database),
        "user_db": IntakeUserDB.model_validate(bundle.user_database),
        "policies": dict(bundle.policies),
    }

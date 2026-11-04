"""Emit a valid draft request skeleton for a new language, offline.

The skeleton fixes everything the factory requires but a contributor cannot
easily assemble by hand: the canonical frozen-50 task cohort per domain (in
``task_subsets/<domain>_50.json`` order), the reviewed English caller genders,
the task-ID convention, the source inventory for the checkout, and the telecom
caller roles. Inventories are written beside the request and referenced by path
and digest instead of being inlined.
"""

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from tau2.data_model.language import LanguagePack, TaskMetadata
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.request_files import (
    ARTIFACT_KEY,
    artifact_ref,
    json_digest,
    load_request,
)
from tau2.multilingual.factory.schema import (
    AuthorFormArtifact,
    DraftRequest,
    FormModel,
)
from tau2.multilingual.localization import RegionalSettings
from tau2.task_subsets.schema import TaskSubset
from tau2.utils.artifacts import write_artifact

REQUEST_FILE = "draft-request.json"
CALLER_ROLES_FILE = "telecom-caller-roles.json"
INVENTORY_DIR = "inventory"
TASKS_DIR = "tasks"


class InitRequestOptions(FormModel):
    """Human decisions the skeleton needs; everything else comes from the repo."""

    language: str = Field(
        pattern=r"^[a-z]{2,3}(?:-[A-Za-z0-9]+)*$", description="New language code."
    )
    domains: list[Literal["airline", "retail", "telecom"]] = Field(
        min_length=1, description="Supported domains to localize."
    )
    reviewed_by: str = Field(
        min_length=1, description="Reviewer recorded on each domain bundle."
    )
    generation: FactoryModelConfig = Field(
        description="Pinned model settings for pack, entity and airport generation."
    )
    country_code: str = Field(
        pattern=r"^[A-Z]{2}$", description="ISO country of the regional scenario."
    )
    currency_code: str = Field(
        pattern=r"^[A-Z]{3}$", description="ISO currency of localized amounts."
    )
    currency_scale: str = Field(
        description="Fixed decimal benchmark scale from USD, e.g. '1' or '1300'."
    )
    time_zone: str = Field(description="IANA time zone of the scenario clock.")


def localized_task_id(domain: str, canonical_id: str, language: str) -> str:
    """Task-ID convention: airline/retail suffix the language; telecom IDs are kept."""
    return canonical_id if domain == "telecom" else f"{canonical_id}_{language}"


def _english_cohort(domain: str, language: str, data_root: Path):
    """Reviewed English frozen-50 tasks and caller genders, rekeyed for a language."""
    root = data_root / "tau2" / "multilingual" / "en"
    pack = LanguagePack.model_validate(yaml.safe_load((root / "pack.yaml").read_text()))
    binding = pack.domains[domain]
    metadata = {
        key: TaskMetadata.model_validate(value)
        for key, value in json.loads(
            (root / binding.task_metadata_file).read_text()
        ).items()
    }
    if binding.tasks_file is not None:
        tasks = json.loads((root / binding.tasks_file).read_text())
    else:
        # Metadata-only binding: the reviewed tasks are the registry tasks.
        registry = data_root / "tau2" / "domains" / domain / "tasks.json"
        tasks = [
            task for task in json.loads(registry.read_text()) if task["id"] in metadata
        ]
    subset = TaskSubset.model_validate_json(
        (data_root / "tau2" / "task_subsets" / f"{domain}_50.json").read_text()
    )
    canonical = [metadata[task["id"]].canonical_task_id for task in tasks]
    if canonical != list(subset.task_ids):
        raise ValueError(f"English {domain} tasks differ from the frozen-50 subset")
    rows = {}
    for task in tasks:
        row = metadata[task["id"]]
        task_id = localized_task_id(domain, row.canonical_task_id, language)
        task["id"] = task_id
        rows[task_id] = {
            "canonical_task_id": row.canonical_task_id,
            "caller_gender": row.caller_gender,
        }
    return pack, tasks, rows


def _write_json(output: Path, relative: str, value) -> dict:
    """Write one referenced input beside the request and return its reference."""
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return {ARTIFACT_KEY: relative, "sha256": json_digest(value)}


def init_draft_request(
    options: InitRequestOptions, form: Path, output: Path, data_root: Path
) -> Path:
    """Write ``draft-request.json`` plus referenced inventories into a new directory."""
    from tau2.multilingual.factory.localization_inventory import inventory_localization

    artifact = AuthorFormArtifact.model_validate_json(form.read_text())
    identity = artifact.form.identity
    if identity.language != options.language:
        raise ValueError("author form language differs from --language")
    output.mkdir(parents=True, exist_ok=False)
    settings = RegionalSettings(
        language=options.language,
        country_code=options.country_code,
        currency_code=options.currency_code,
        currency_scale=options.currency_scale,
        time_zone=options.time_zone,
        name_order=identity.name_order,
    )
    generation = options.generation.model_dump(mode="json")
    domains = []
    pack = None
    for domain in options.domains:
        pack, tasks, metadata = _english_cohort(domain, options.language, data_root)
        source = inventory_localization(domain, data_root)
        source_path = output / INVENTORY_DIR / f"{domain}.json"
        write_artifact(source, source_path)
        localization = {
            "mode": "generate",
            "request": {
                "source": artifact_ref(source_path, source, relative_to=output),
                "settings": settings.model_dump(mode="json"),
                "revision": f"{options.language}-{domain}-v1",
                "generation": generation,
            },
        }
        if domain == "airline":
            localization["airport_generation"] = {
                "revision": f"{options.language}-airports-v1",
                "generation": generation,
            }
        if domain == "telecom":
            roles = json.loads(
                (data_root / "tau2" / "multilingual" / CALLER_ROLES_FILE).read_text()
            )
            localization["caller_roles"] = _write_json(output, CALLER_ROLES_FILE, roles)
        domains.append(
            {
                "domain": domain,
                "revision": f"{options.language}-{domain}-v1",
                "tasks": _write_json(output, f"{TASKS_DIR}/{domain}.json", tasks),
                "metadata": _write_json(
                    output, f"{TASKS_DIR}/{domain}.metadata.json", metadata
                ),
                "reviewed_by": options.reviewed_by,
                "voice_guidelines": "",
                "localization": localization,
            }
        )
    request = {
        "form": artifact.form.model_dump(mode="json"),
        "revision": f"{options.language}-v1",
        "generation": generation,
        "speech_insert_events_per_minute": pack.voice.speech_insert_events_per_minute,
        "backchannel_level": pack.voice.backchannel_level.value,
        "domains": domains,
    }
    path = output / REQUEST_FILE
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    load_request(DraftRequest, path)
    return path

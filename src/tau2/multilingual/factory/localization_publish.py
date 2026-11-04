"""Publish reviewed regional maps into immutable native runtime bundles."""

import re
from datetime import datetime
from pathlib import Path

from pydantic import TypeAdapter

from tau2.data_model.language import TaskMetadata
from tau2.data_model.tasks import Task
from tau2.multilingual.airport_catalog import AirportCatalog
from tau2.multilingual.caller_identity import TelecomCallerRoles
from tau2.multilingual.factory.localization_artifact import (
    RegionalMapArtifact,
    restore_localization_map,
)
from tau2.multilingual.factory.localization_compile import compile_localization
from tau2.multilingual.factory.localization_inventory import inventory_localization
from tau2.multilingual.localization import LocalizationSource
from tau2.multilingual.regional import PublishedRegionalBundle
from tau2.multilingual.scenario_dispatch import RegionalBundle, bundle_model
from tau2.utils.artifacts import content_digest, write_artifact


def _preserve_policy_wall_clock(source: str, localized: str, time_zone: str) -> str:
    """Keep the source scenario clock; localizing its label never shifts dates.

    The database and task timestamps retain their source wall-clock values. A
    converted policy timestamp would therefore disagree with executable billing
    or booking rules. This publication transform changes only the clock header,
    leaving the approved map and all other policy text untouched.
    """
    pattern = re.compile(
        r"^The current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) [^\n]+\.$",
        re.MULTILINE,
    )
    original, target = list(pattern.finditer(source)), list(pattern.finditer(localized))
    if len(original) != 1 or len(target) != 1:
        raise ValueError("Publication requires exactly one current-time policy header")
    clock = original[0].group(1)
    datetime.strptime(clock, "%Y-%m-%d %H:%M:%S")
    begin, end = target[0].span()
    return (
        localized[:begin]
        + f"The current time is {clock} {time_zone}."
        + localized[end:]
    )


def validate_regional_source(source: LocalizationSource, data_root: Path) -> None:
    """Check canonical data before any localized content generation is allowed."""
    current = inventory_localization(source.domain, data_root)
    before = {d.id: d for d in source.documents if d.kind != "tool_text"}
    after = {d.id: d for d in current.documents if d.kind != "tool_text"}
    if before != after:
        raise ValueError("publication source database, tasks or policies changed")


def build_regional_bundle(
    artifact: RegionalMapArtifact,
    source: LocalizationSource,
    metadata: dict[str, TaskMetadata],
    data_root: Path,
    *,
    airport_catalog: AirportCatalog | None = None,
    caller_roles: TelecomCallerRoles | None = None,
) -> RegionalBundle:
    """Build checked runtime data for the language factory's content-review stage.

    This is not publication or approval. The enclosing draft binds these exact
    bytes to its ordinary content review before voice preparation or publication.
    Tool evidence stays pinned while current non-code source documents must match.
    """
    from tau2.environment.regional import REGIONAL_RUNTIME_VERSION

    proposal = restore_localization_map(artifact, source)
    if source.domain == "intake":
        from tau2.multilingual.factory.intake_gender import (
            validate_intake_gender_coverage,
        )

        validate_intake_gender_coverage(proposal, data_root)
    elif source.domain == "banking_knowledge":
        from tau2.multilingual.factory.localization_gender import (
            require_name_gender_coverage,
        )

        require_name_gender_coverage(proposal)
    validate_regional_source(source, data_root)
    documents = {d.id: d for d in compile_localization(proposal)}
    by_canonical = {
        m.canonical_task_id: (task_id, m) for task_id, m in metadata.items()
    }
    tasks = TypeAdapter(list[Task]).validate_python(documents["tasks"].content)
    if set(by_canonical) != {t.id for t in tasks}:
        raise ValueError("regional publication requires the same frozen task cohort")
    tasks = [task.model_copy(update={"id": by_canonical[task.id][0]}) for task in tasks]
    identities = []
    if source.domain == "telecom" and any(m.caller_gender for m in metadata.values()):
        from tau2.domains.telecom.data_model import TelecomDB
        from tau2.multilingual.factory.localization_callers import (
            localize_telecom_callers,
        )

        if caller_roles is None:
            raise ValueError(
                "telecom caller gender assignments require explicit canonical name roles"
            )
        publication = localize_telecom_callers(
            proposal,
            TelecomDB.model_validate(documents["database"].content),
            tasks,
            metadata,
            caller_roles,
        )
        tasks = publication.tasks
        identities = publication.identities
    elif caller_roles is not None:
        raise ValueError(
            "caller roles require telecom tasks with explicit gender assignments"
        )
    airports = {}
    if source.domain == "airline":
        from tau2.multilingual.factory.airport_catalog import validate_airport_catalog

        if airport_catalog is None:
            raise ValueError("airline build requires an airport city catalog")
        validate_airport_catalog(airport_catalog, artifact)
        airports = {row.iata: row.native_city for row in airport_catalog.airports}
    elif airport_catalog is not None:
        raise ValueError("airport catalogs are only used for airline publication")
    policies = {
        key.split(":", 1)[1]: d.content
        for key, d in documents.items()
        if d.kind == "policy"
    }
    if source.domain == "banking_knowledge":
        policies = {
            key.removeprefix("dependency:policy:"): d.content
            for key, d in documents.items()
            if key.startswith("dependency:policy:")
        }
    if source.domain in {"airline", "telecom"}:
        key = "main_policy" if source.domain == "telecom" else "policy"
        canonical_policy = next(
            d.content for d in source.documents if d.id == f"policy:{key}"
        )
        policies[key] = _preserve_policy_wall_clock(
            canonical_policy, policies[key], proposal.settings.time_zone
        )
    domain_dependencies = (
        {"knowledge_base_sha256": documents["dependency:english_knowledge"].sha256}
        if source.domain == "banking_knowledge"
        else {}
    )
    if source.domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.kb_entities import (
            localize_knowledge_entities,
        )

        domain_dependencies["knowledge_document_overrides"] = (
            localize_knowledge_entities(artifact, data_root)
        )
    if source.domain == "intake":
        domain_dependencies["intake_localization"] = documents[
            "intake_localization"
        ].content
    bundle = bundle_model(source.domain)(
        **domain_dependencies,
        domain=source.domain,
        database=documents["database"].content,
        user_database=documents["user_database"].content
        if "user_database" in documents
        else None,
        tasks=tasks,
        policies=policies,
        revision=proposal.revision,
        settings=proposal.settings,
        localization_map_sha256=content_digest(artifact),
        source_document_sha256={d.id: d.sha256 for d in source.documents},
        metadata=metadata,
        airports=airports,
        runtime_rules_version=REGIONAL_RUNTIME_VERSION,
        airport_catalog_sha256=content_digest(airport_catalog)
        if airport_catalog is not None
        else None,
        caller_roles=caller_roles,
        caller_identities=identities,
    )
    return bundle


_BUNDLE_PROVENANCE_FIELDS = {
    "revision",
    "localization_map_sha256",
    "source_document_sha256",
    "airport_catalog_sha256",
}


def bundle_runtime_content(bundle: PublishedRegionalBundle) -> dict:
    """Everything a call consumes: the bundle without its provenance digests."""
    return bundle.model_dump(mode="json", exclude=_BUNDLE_PROVENANCE_FIELDS)


def refresh_regional_bundle(
    language: str,
    domain: str,
    revision: str,
    output: Path,
    data_root: Path,
) -> PublishedRegionalBundle:
    """Rebind one installed language's map to this checkout and republish its bundle.

    Reads ``localization/<domain>.json``, the airline ``airport-catalog.json`` and
    the pack's published regional bundle from the installed language directory
    under ``data_root``. Writes the same
    relative layout under ``output``. Refuses unless the rebuilt runtime content
    (database, tasks, policies, metadata, airports, caller identities) is
    byte-for-byte the published content; only provenance digests may change.
    """
    import yaml

    from tau2.data_model.language import LanguagePack
    from tau2.multilingual.factory.localization_artifact import (
        refresh_localization_source,
        write_localization_map,
    )

    language_dir = data_root / "tau2" / "multilingual" / language
    pack = LanguagePack.model_validate(
        yaml.safe_load((language_dir / "pack.yaml").read_text())
    )
    binding = pack.domains[domain]
    if binding.regional_bundle_file is None:
        raise ValueError(f"{language}/{domain} has no regional bundle")
    bundle_file = language_dir / binding.regional_bundle_file
    previous = PublishedRegionalBundle.model_validate_json(bundle_file.read_text())
    map_file = language_dir / "localization" / f"{domain}.json"
    artifact = RegionalMapArtifact.model_validate_json(map_file.read_text())
    # Review-only edits (status, entry notes) may change the installed map's
    # digest; the revision ties it to the bundle, and the runtime-content identity
    # check below is the guarantee that nothing a call consumes changed.
    if previous.revision != artifact.revision:
        raise ValueError("published bundle was not built from the installed map")
    source = inventory_localization(domain, data_root)
    proposal = refresh_localization_source(artifact, source, revision)
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new output directory")
    refreshed_file = output / "localization" / f"{domain}.json"
    write_localization_map(proposal, refreshed_file)
    refreshed = RegionalMapArtifact.model_validate_json(refreshed_file.read_text())
    catalog = None
    if domain == "airline":
        catalog_file = language_dir / "localization" / "airport-catalog.json"
        installed = AirportCatalog.model_validate_json(catalog_file.read_text())
        # The published bundle pins the catalog's content; the catalog's own
        # back-pointer is rebound below.
        if content_digest(installed) != previous.airport_catalog_sha256:
            raise ValueError("airport catalog is not the one the bundle was built from")
        catalog = installed.model_copy(update={"map_sha256": content_digest(refreshed)})
        write_artifact(catalog, output / "localization" / "airport-catalog.json")
    bundle = build_regional_bundle(
        refreshed,
        source,
        previous.metadata,
        data_root,
        airport_catalog=catalog,
        caller_roles=previous.caller_roles,
    )
    if bundle_runtime_content(bundle) != bundle_runtime_content(previous):
        raise ValueError("refreshed runtime content differs from the published bundle")
    write_artifact(bundle, output / binding.regional_bundle_file)
    return bundle

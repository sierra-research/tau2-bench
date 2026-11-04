"""Offline coverage for request references, init-request, docs and map refresh."""

import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.init_request import (
    InitRequestOptions,
    init_draft_request,
    localized_task_id,
)
from tau2.multilingual.factory.request_files import json_digest, load_request
from tau2.multilingual.factory.schema import (
    AuthorForm,
    AuthorFormArtifact,
    AutoformRequest,
    DraftRequest,
    LanguageIdentity,
    PersonaSketch,
)
from tau2.utils import DATA_DIR
from tau2.utils.artifacts import ReviewApproval, content_digest, write_artifact

ROOT = Path(__file__).resolve().parents[2]
SHIPPED_MAPS = sorted(
    (path.parents[1].name, path.stem)
    for path in (DATA_DIR / "tau2/multilingual").glob("*/localization/*.json")
    if path.stem in {"airline", "retail", "telecom"}
)


def _author_form(path: Path, language: str = "fr") -> Path:
    identity = LanguageIdentity(
        language=language, display_name="Fixture", script="latn"
    )
    form = AuthorForm(
        identity=identity,
        locale_notes="Fixture locale notes.",
        preferences="Fixture preferences.",
        personas=[
            PersonaSketch(name="Luc", gender="male", sketch="Fixture register A."),
            PersonaSketch(name="Anne", gender="female", sketch="Fixture register B."),
        ],
    )
    request = AutoformRequest(
        identity=identity,
        generation=FactoryModelConfig(model="fixture", reasoning_effort="low"),
        hints="Fixture.",
    )
    write_artifact(
        AuthorFormArtifact(
            form=form,
            request=request,
            prompt_version="fixture",
            input_sha256=content_digest(request),
        ),
        path,
    )
    return path


def _options(domains, language="fr") -> InitRequestOptions:
    return InitRequestOptions(
        language=language,
        domains=domains,
        reviewed_by="Fixture Reviewer",
        generation=FactoryModelConfig(model="fixture", reasoning_effort="medium"),
        country_code="FR",
        currency_code="EUR",
        currency_scale="1",
        time_zone="Europe/Paris",
    )


def test_references_resolve_relative_to_request_and_reject_edits(tmp_path):
    approval = ReviewApproval(content_sha256="a" * 64, reviewed_by="fixture")
    write_artifact(approval, tmp_path / "approvals/content.json")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {"$artifact": "approvals/content.json", "sha256": content_digest(approval)}
        )
    )
    assert load_request(ReviewApproval, request) == approval
    (tmp_path / "approvals/content.json").write_text(
        approval.model_copy(update={"reviewed_by": "someone else"}).model_dump_json()
    )
    with pytest.raises(ValueError, match="does not match its referenced sha256"):
        load_request(ReviewApproval, request)


def test_init_request_emits_valid_frozen50_skeleton(tmp_path):
    form = _author_form(tmp_path / "author-form.json")
    path = init_draft_request(
        _options(["telecom"]), form, tmp_path / "request", DATA_DIR
    )
    request = load_request(DraftRequest, path)
    [bundle] = request.domains
    subset = json.loads((DATA_DIR / "tau2/task_subsets/telecom_50.json").read_text())[
        "task_ids"
    ]
    assert [row.canonical_task_id for row in bundle.metadata.values()] == subset
    assert [task.id for task in bundle.tasks] == list(bundle.metadata)
    english = json.loads(
        (DATA_DIR / "tau2/multilingual/en/telecom_metadata_en.json").read_text()
    )
    assert {
        row.canonical_task_id: row.caller_gender for row in bundle.metadata.values()
    } == {row["canonical_task_id"]: row["caller_gender"] for row in english.values()}
    assert request.generation.reasoning_effort == "medium"
    assert bundle.localization.request.source == _current_inventory("telecom")
    assert bundle.localization.caller_roles is not None
    raw = json.loads(path.read_text())
    assert raw["domains"][0]["localization"]["request"]["source"]["$artifact"] == (
        "inventory/telecom.json"
    )
    assert path.stat().st_size < 20_000
    with pytest.raises(FileExistsError):
        init_draft_request(_options(["telecom"]), form, tmp_path / "request", DATA_DIR)


def test_task_id_convention():
    assert localized_task_id("retail", "6", "fr") == "6_fr"
    assert localized_task_id("airline", "0", "fr") == "0_fr"
    assert localized_task_id("telecom", "[x]y[PERSONA:None]", "fr") == (
        "[x]y[PERSONA:None]"
    )


def test_documented_draft_request_example_validates(tmp_path):
    """The worked example in docs/language-authoring.md is a valid request."""
    text = (ROOT / "docs/language-authoring.md").read_text()
    match = re.search(r"<!-- draft-request-example -->\s*```json\n(.*?)```", text, re.S)
    assert match, "draft-request example block missing from the docs"
    example = json.loads(match.group(1))
    form = _author_form(
        tmp_path / "author-form.json", example["form"]["identity"]["language"]
    )
    generated = init_draft_request(
        _options(
            [bundle["domain"] for bundle in example["domains"]],
            example["form"]["identity"]["language"],
        ),
        form,
        tmp_path / "generated",
        DATA_DIR,
    )

    def rebind(value):
        # The documented digests are illustrative; bind to regenerated files.
        if isinstance(value, dict) and "$artifact" in value:
            data = json.loads((generated.parent / value["$artifact"]).read_text())
            return {"$artifact": value["$artifact"], "sha256": json_digest(data)}
        if isinstance(value, dict):
            return {key: rebind(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebind(item) for item in value]
        return value

    documented = generated.parent / "documented.json"
    documented.write_text(json.dumps(rebind(example)))
    request = load_request(DraftRequest, documented)
    assert request.generation.reasoning_effort in {"low", "medium", "high", "xhigh"}


@lru_cache(maxsize=None)
def _current_inventory(domain: str):
    from tau2.multilingual.factory.localization_inventory import inventory_localization

    return inventory_localization(domain, DATA_DIR)


@pytest.mark.slow
@pytest.mark.parametrize(("language", "domain"), SHIPPED_MAPS)
def test_shipped_maps_reuse_against_current_inventory(language, domain):
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        restore_localization_map,
    )

    artifact = RegionalMapArtifact.model_validate_json(
        (
            DATA_DIR / f"tau2/multilingual/{language}/localization/{domain}.json"
        ).read_text()
    )
    restore_localization_map(artifact, _current_inventory(domain))


@pytest.mark.slow
@pytest.mark.parametrize(("language", "domain"), SHIPPED_MAPS)
def test_shipped_bundles_rebuild_from_their_installed_maps(language, domain):
    """The compiler reproduces every shipped bundle's runtime content exactly."""
    from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
    from tau2.multilingual.factory.localization_publish import (
        AirportCatalog,
        build_regional_bundle,
        bundle_runtime_content,
    )
    from tau2.multilingual.regional import PublishedRegionalBundle

    root = DATA_DIR / "tau2/multilingual" / language
    shipped = PublishedRegionalBundle.model_validate_json(
        (root / f"regional/{domain}.json").read_text()
    )
    catalog = (
        AirportCatalog.model_validate_json(
            (root / "localization/airport-catalog.json").read_text()
        )
        if domain == "airline"
        else None
    )
    rebuilt = build_regional_bundle(
        RegionalMapArtifact.model_validate_json(
            (root / f"localization/{domain}.json").read_text()
        ),
        _current_inventory(domain),
        shipped.metadata,
        DATA_DIR,
        airport_catalog=catalog,
        caller_roles=shipped.caller_roles,
    )
    assert bundle_runtime_content(rebuilt) == bundle_runtime_content(shipped)


@pytest.mark.parametrize(("language", "domain"), SHIPPED_MAPS)
def test_shipped_maps_bundles_and_catalogs_pin_each_other(language, domain):
    from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
    from tau2.multilingual.factory.localization_publish import AirportCatalog
    from tau2.multilingual.regional import PublishedRegionalBundle

    root = DATA_DIR / "tau2/multilingual" / language
    artifact = RegionalMapArtifact.model_validate_json(
        (root / f"localization/{domain}.json").read_text()
    )
    bundle = PublishedRegionalBundle.model_validate_json(
        (root / f"regional/{domain}.json").read_text()
    )
    assert artifact.status == "published"
    assert bundle.revision == artifact.revision
    assert bundle.localization_map_sha256 == content_digest(artifact)
    if domain == "airline":
        catalog = AirportCatalog.model_validate_json(
            (root / "localization/airport-catalog.json").read_text()
        )
        assert bundle.airport_catalog_sha256 == content_digest(catalog)
        assert catalog.map_sha256 == content_digest(artifact)
    assert not any("donor" in entry.notes.lower() for entry in artifact.entries)


def test_refresh_regional_bundle_refuses_an_unchanged_source(tmp_path):
    from tau2.multilingual.factory.localization_publish import refresh_regional_bundle

    with pytest.raises(ValueError, match="changed source"):
        refresh_regional_bundle(
            "es", "telecom", "es-telecom-refresh-test", tmp_path / "out", DATA_DIR
        )


def _data_root_with_writable(tmp_path, language: str) -> Path:
    """A data root linking to shipped data, with one language directory copied."""
    import shutil

    root = tmp_path / "data"
    for child in DATA_DIR.iterdir():
        if child.name != "tau2":
            (root / child.name).parent.mkdir(parents=True, exist_ok=True)
            (root / child.name).symlink_to(child)
    for child in (DATA_DIR / "tau2").iterdir():
        if child.name != "multilingual":
            (root / "tau2" / child.name).parent.mkdir(parents=True, exist_ok=True)
            (root / "tau2" / child.name).symlink_to(child)
    for child in (DATA_DIR / "tau2/multilingual").iterdir():
        target = root / "tau2/multilingual" / child.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if child.name == language:
            shutil.copytree(child, target)
        else:
            target.symlink_to(child)
    return root


@pytest.mark.parametrize("language", ["es", "pt"])
def test_refresh_regional_bundle_rebinds_moved_tool_text(tmp_path, language):
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        read_localization_map,
    )
    from tau2.multilingual.factory.localization_publish import refresh_regional_bundle
    from tau2.multilingual.regional import PublishedRegionalBundle

    root = _data_root_with_writable(tmp_path, language)
    installed = root / "tau2/multilingual" / language
    map_file = installed / "localization/telecom.json"
    stale = json.loads(map_file.read_text())
    tool_text = next(
        doc for doc in stale["source"]["documents"] if doc["kind"] == "tool_text"
    )
    tool_text["sha256"] = "0" * 64  # as if code had moved the inventoried text
    map_file.write_text(json.dumps(stale))
    before = PublishedRegionalBundle.model_validate_json(
        (installed / "regional/telecom.json").read_text()
    )

    bundle = refresh_regional_bundle(
        language, "telecom", f"{language}-telecom-refresh-test", tmp_path / "out", root
    )
    assert bundle.tasks == before.tasks and bundle.metadata == before.metadata
    assert bundle.revision == f"{language}-telecom-refresh-test"
    refreshed_file = tmp_path / "out/localization/telecom.json"
    refreshed = RegionalMapArtifact.model_validate_json(refreshed_file.read_text())
    assert bundle.localization_map_sha256 == content_digest(refreshed)
    assert refreshed.source_refresh.parent_artifact_sha256 == content_digest(
        RegionalMapArtifact.model_validate(stale)
    )
    source = _current_inventory("telecom")
    assert refreshed.source_refresh.source_sha256 == content_digest(source)
    read_localization_map(refreshed_file, DATA_DIR)

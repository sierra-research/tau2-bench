"""Check retained reviewed-correction fixtures against the published maps they produced.

The fixtures record exact before/after forms for each reviewed entry. These
helpers read them as plain JSON and assert that the shipped maps contain exactly
those values, so the published data stays bound to its recorded review.
"""

import json
from pathlib import Path

from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
from tau2.multilingual.localization import (
    IntakeCallerSurface,
    LocalizationMap,
    LocalizedForms,
)
from tau2.utils.artifacts import content_digest


def load_review(path: Path) -> dict:
    """Read a reviewed-corrections fixture as untyped JSON."""
    return json.loads(path.read_text())


def assert_review_reproduces_published_names(
    original: RegionalMapArtifact, published: RegionalMapArtifact, review: dict
) -> None:
    """Every changed name is a reviewed correction, and each one landed exactly."""
    assert review["revision"] == published.revision
    before = {entry.id: entry for entry in original.entries}
    after = {entry.id: entry for entry in published.entries}
    corrected = set()
    for edit in review["corrections"]:
        assert not edit["binding"]
        assert edit["entry_id"] not in corrected, "duplicate reviewed correction"
        corrected.add(edit["entry_id"])
        old, new = before[edit["entry_id"]], after[edit["entry_id"]]
        assert old.source == new.source == edit["source"]
        assert old.forms == LocalizedForms.model_validate(edit["before"])
        assert new.forms == LocalizedForms.model_validate(edit["after"])
        if edit["name_gender"] is not None:
            new.name_gender.require(new.forms)
            assert new.name_gender.model_dump(mode="json") == edit["name_gender"]
    for entry_id, entry in after.items():
        if entry.category == "identity_name" and entry_id not in corrected:
            assert entry.forms == before[entry_id].forms, entry_id


def assert_review_applied(proposal: LocalizationMap, review: dict) -> None:
    """A reviewed correction set is already applied to the current map."""
    assert review["revision"] == proposal.revision
    entries = {entry.id: entry for entry in proposal.entries}
    for edit in review["corrections"]:
        expected = LocalizedForms.model_validate(edit["after"])
        if edit["binding"]:
            matches = [
                binding
                for binding in proposal.prose_bindings
                if binding.entry_id == edit["entry_id"]
                and binding.source == edit["source"]
            ]
            assert len(matches) == 1, edit["entry_id"]
            assert matches[0].forms == expected
        else:
            entry = entries[edit["entry_id"]]
            assert entry.source == edit["source"]
            assert entry.forms == expected, edit["entry_id"]
    surfaces = {s.canonical_task_id: s for s in proposal.intake_caller_surfaces}
    for edit in review.get("caller_surfaces", []):
        after = IntakeCallerSurface.model_validate(edit["after"])
        current = surfaces[after.canonical_task_id]
        assert current.review is not None
        assert current.review.content_sha256 == content_digest(after)
        assert current.model_copy(update={"review": None}) == after

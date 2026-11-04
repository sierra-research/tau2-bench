"""Policy localization cannot move the runtime's fixed scenario wall clock."""

import pytest

from tau2.multilingual.factory.localization_artifact import (
    RegionalMapArtifact,
    write_localization_map,
)
from tau2.multilingual.factory.localization_publish import (
    _preserve_policy_wall_clock,
    build_regional_bundle,
)
from tau2.multilingual.regional import scenario_environment_kwargs
from tau2.utils.artifacts import content_digest
from tests.test_multilingual import test_localization_callers as caller_fixtures
from tests.test_multilingual import test_localization_user_ids as user_id_fixtures

telecom_proposal = user_id_fixtures.telecom_proposal
caller_inputs = caller_fixtures.caller_inputs


@pytest.mark.parametrize(
    "clock,converted,zone",
    [
        ("2025-02-25 12:08:00", "2025-02-26 02:08:00", "Asia/Seoul"),
        ("2025-02-25 12:08:00", "2025-02-25 18:08:00", "Europe/Madrid"),
        ("2024-05-15 15:00:00", "2024-05-16 04:00:00", "Asia/Shanghai"),
    ],
)
def test_policy_preserves_source_clock_and_other_text(clock, converted, zone):
    canonical = f"# Policy\n\nThe current time is {clock} EST.\n\nSource rules."
    localized = (
        f"# Policy\n\nThe current time is {converted} KST.\n\nReviewed regional rules."
    )
    result = _preserve_policy_wall_clock(canonical, localized, zone)
    assert (
        result
        == f"# Policy\n\nThe current time is {clock} {zone}.\n\nReviewed regional rules."
    )
    assert _preserve_policy_wall_clock(canonical, result, zone) == result


@pytest.mark.parametrize("defect", ["missing", "duplicate", "malformed"])
def test_ambiguous_policy_clock_fails_before_publication(defect):
    canonical = "The current time is 2025-02-25 12:08:00 EST."
    bad = {
        "missing": "No current time supplied.",
        "duplicate": canonical + "\n" + canonical,
        "malformed": "The current time is tomorrow EST.",
    }[defect]
    with pytest.raises(ValueError, match="exactly one"):
        _preserve_policy_wall_clock(canonical, bad, "Asia/Shanghai")


def test_publication_records_rule_without_changing_map_or_approval(
    caller_inputs, tmp_path
):
    proposal, _, _, metadata, roles = caller_inputs
    proposal.settings.time_zone = "Asia/Seoul"
    policy = next(
        e
        for e in proposal.entries
        if e.category == "policy" and e.source.startswith("# Telecom Agent Policy")
    )
    for representation in ("native", "romanized"):
        setattr(
            policy.forms,
            representation,
            policy.source.replace("2025-02-25 12:08:00 EST", "2025-02-26 02:08:00 KST"),
        )
    policy.operation = "adapt"
    mapping = tmp_path / "telecom.json"
    write_localization_map(proposal, mapping)
    original_bytes = mapping.read_bytes()
    artifact = RegionalMapArtifact.model_validate_json(original_bytes)
    bundle = build_regional_bundle(
        artifact,
        proposal.source,
        metadata,
        user_id_fixtures.DATA,
        caller_roles=roles,
    )
    assert bundle.policy_clock_rule == "source-wall-clock-v1"
    assert bundle.localization_map_sha256 == content_digest(artifact)
    assert mapping.read_bytes() == original_bytes
    settings = scenario_environment_kwargs(bundle)["regional_settings"]
    assert (
        settings.current_datetime.strftime("%Y-%m-%d %H:%M:%S") == "2025-02-25 12:08:00"
    )
    assert (
        "The current time is 2025-02-25 12:08:00 Asia/Seoul."
        in bundle.policies["main_policy"]
    )
    approved_policy = next(e for e in artifact.entries if e.id == policy.id)
    assert "2025-02-26 02:08:00" in approved_policy.forms.native

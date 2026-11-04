"""Preserve seeded stimuli while tightening their serialization contracts."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tau2.data_model.elicitation import (
    ComplicationKind,
    ComplicationParams,
    ElicitationConfig,
    SampledComplication,
)
from tau2.domains.intake.agent_directed import get_tasks
from tau2.domains.intake.complications import (
    COMPLICATION_CATALOG_VERSION,
    sample_complication,
)
from tau2.domains.intake.prompts import get_user_guidelines


def test_seeded_draws_match_pinned_migration_source():
    fixtures = json.loads(
        (
            Path(__file__).parent / "fixtures/complications_v2_4_0_seed42.json"
        ).read_text()
    )
    tasks = {task.id: task for task in get_tasks()}
    assert {row["sample"]["kind"] for row in fixtures} == set(ComplicationKind)
    for row in fixtures:
        expected = SampledComplication.model_validate(row["sample"])
        # Rendered catalog provenance changed; these non-time seeded stimuli
        # must otherwise remain byte-identical to the original draw contract.
        expected.catalog_version = COMPLICATION_CATALOG_VERSION
        actual = sample_complication(
            42,
            tasks[row["task_id"]],
            rate_override=1,
            channel="voice",
        )
        assert actual == expected
        assert (
            SampledComplication.model_validate_json(actual.model_dump_json()) == actual
        )


def test_unknown_parameters_and_kinds_fail_at_the_result_boundary():
    with pytest.raises(ValidationError):
        ComplicationParams.model_validate({"invented_parameter": "value"})
    with pytest.raises(ValidationError):
        SampledComplication(
            kind="invented_kind",
            line="bad",
            params=ComplicationParams(),
            catalog_version="1",
        )


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_trigger_rate_bounds(rate):
    with pytest.raises(ValidationError):
        ElicitationConfig(rate=rate)


def test_domain_owned_guidelines_are_complete_and_scope_spelling_guidance():
    scaffolded = get_user_guidelines(is_voice=True, agent_directed=False)
    agent_directed = get_user_guidelines(is_voice=True, agent_directed=True)
    assert "ANSWERING a VOICE CALL" in agent_directed
    assert "offer to spell it out letter by letter" not in agent_directed
    assert "offer to spell it out letter by letter" in scaffolded
    assert "<EXAMPLE:" not in agent_directed
    assert "<SPOKEN_VALUES_SPELLOUT>" not in agent_directed
    assert '"J, O, H, N"' in agent_directed
    assert "<PERSONA_GUIDELINES>" in agent_directed

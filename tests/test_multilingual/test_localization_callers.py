"""Caller rotation changes identity while preserving the target account scenario."""

from collections import Counter

import pytest

from tau2.data_model.language import TaskMetadata
from tau2.data_model.tasks import InitializationData, Task
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.environment import get_environment
from tau2.multilingual.caller_identity import CallerIdentitySource, TelecomCallerRoles
from tau2.multilingual.factory.localization_callers import localize_telecom_callers
from tau2.multilingual.factory.localization_compile import compile_localization
from tau2.multilingual.factory.localization_emails import localize_email_entries
from tau2.multilingual.factory.localization_user_ids import (
    finalize_localization_user_ids,
    remap_user_id_references,
)
from tau2.multilingual.localization import UserIdAlias
from tests.test_multilingual import test_localization_user_ids as user_id_fixtures

telecom_proposal = user_id_fixtures.telecom_proposal


@pytest.fixture
def caller_inputs(telecom_proposal):
    # Fresh drafts allocate short deterministic emails before any other stage.
    emails = {
        entry.id: entry
        for entry in localize_email_entries(
            telecom_proposal.source.entries,
            [e for e in telecom_proposal.entries if e.category == "identity_name"],
        )
    }
    telecom_proposal.entries = [
        emails.get(entry.id, entry) for entry in telecom_proposal.entries
    ]
    proposal = finalize_localization_user_ids(telecom_proposal)
    documents = {document.id: document for document in compile_localization(proposal)}
    database = TelecomDB.model_validate(documents["database"].content)
    tasks = [Task.model_validate(row) for row in documents["tasks"].content]
    metadata = {
        task.id: TaskMetadata(
            canonical_task_id=task.id, caller_gender="male" if i % 2 == 0 else "female"
        )
        for i, task in enumerate(tasks)
    }
    roles = TelecomCallerRoles(
        target=CallerIdentitySource(customer_id="C1001", full_name="John Smith"),
        male=CallerIdentitySource(customer_id="C1001", full_name="John Smith"),
        female=CallerIdentitySource(customer_id="C1002", full_name="Sarah Johnson"),
    )
    return proposal, database, tasks, metadata, roles


def test_gender_rotation_preserves_account_relations_and_authentication(caller_inputs):
    proposal, database, tasks, metadata, roles = caller_inputs
    original_db = database.model_dump(mode="json")
    original_tasks = [task.model_dump(mode="json") for task in tasks]
    result = localize_telecom_callers(*caller_inputs)
    assert Counter(row.gender for row in result.identities) == {
        "female": 25,
        "male": 25,
    }
    assert result.tasks[0] == tasks[0]
    identity, task = result.identities[1], result.tasks[1]
    assert identity.customer_id == "sarah_johnson_1001"
    assert identity.canonical_name_source_id == "C1002"
    environment = get_environment(db=database)
    environment.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=[],
    )
    customer = environment.tools.get_customer_by_id(identity.customer_id)
    before = database.customers[0]
    assert customer.model_dump(
        exclude={"customer_id", "full_name", "email"}
    ) == before.model_dump(exclude={"customer_id", "full_name", "email"})
    assert environment.tools.get_customer_by_name(
        identity.native_name, before.date_of_birth
    ) == [customer]
    donor = environment.tools.get_customer_by_id("sarah_johnson_1002")
    assert donor.full_name == customer.full_name
    assert donor.date_of_birth != customer.date_of_birth
    assert environment.user_tools.db.surroundings.name == customer.full_name
    assert environment.user_tools.db.surroundings.phone_number == before.phone_number
    assert customer.line_ids == before.line_ids
    assert environment.tools.db.lines == database.lines
    assert all(
        bill.customer_id == identity.customer_id
        for bill in environment.tools.db.bills
        if bill.bill_id in before.bill_ids
    )
    assert task.evaluation_criteria == tasks[1].evaluation_criteria
    assert task.id == tasks[1].id
    assert (
        "You are Sarah Johnson with phone number"
        in task.user_scenario.instructions.known_info
    )
    assert "Customer name: Sarah Johnson, phone number:" in task.ticket
    assert database.model_dump(mode="json") == original_db
    assert [task.model_dump(mode="json") for task in tasks] == original_tasks
    assert localize_telecom_callers(*caller_inputs) == result


def test_existing_initialization_and_customer_references_are_preserved(caller_inputs):
    proposal, database, tasks, metadata, roles = caller_inputs
    female = tasks[1]
    patched = database.model_copy(deep=True)
    patched.customers[0].goodwill_credit_used_this_year = 12
    female.initial_state.initialization_data = InitializationData(
        agent_data={"customers": patched.model_dump(mode="json")["customers"]},
        user_data={"surroundings": {"phone_number": "unchanged"}},
    )
    female.evaluation_criteria.communicate_info = [
        "Your customer_id is john_smith_1001."
    ]
    result = localize_telecom_callers(*caller_inputs)
    changed = result.tasks[1]
    data = changed.initial_state.initialization_data
    assert data.agent_data["customers"][0]["goodwill_credit_used_this_year"] == 12
    assert data.user_data == female.initial_state.initialization_data.user_data
    assert changed.evaluation_criteria.communicate_info == [
        "Your customer_id is sarah_johnson_1001."
    ]
    assert (
        changed.evaluation_criteria.reward_basis
        == female.evaluation_criteria.reward_basis
    )


def test_all_initialization_effects_match_before_binding(caller_inputs):
    _, database, tasks, _, _ = caller_inputs
    result = localize_telecom_callers(*caller_inputs)
    for before, after, identity in zip(
        tasks, result.tasks, result.identities, strict=True
    ):
        original, localized = get_environment(db=database), get_environment(db=database)
        for task, environment in ((before, original), (after, localized)):
            environment.set_state(
                task.initial_state.initialization_data,
                task.initial_state.initialization_actions,
                [],
            )
        normalized = remap_user_id_references(
            localized.tools.db.model_dump(mode="json"),
            [UserIdAlias(canonical=identity.customer_id, localized="john_smith_1001")],
        )
        target = next(
            row
            for row in normalized["customers"]
            if row["customer_id"] == "john_smith_1001"
        )
        source_target = original.tools.get_customer_by_id("john_smith_1001")
        target.update(full_name=source_target.full_name, email=source_target.email)
        assert normalized == original.tools.db.model_dump(mode="json")
        user_db = localized.user_tools.db.model_copy(deep=True)
        user_db.surroundings.name = original.user_tools.db.surroundings.name
        assert user_db == original.user_tools.db


@pytest.mark.parametrize(
    "defect,error",
    [
        ("role", "canonical source"),
        ("id", "ID collides"),
        ("dob", "birth date are ambiguous"),
        ("prose", "declaration differs"),
        ("gender", "explicit task gender"),
    ],
)
def test_unsupported_or_ambiguous_caller_bindings_fail(caller_inputs, defect, error):
    proposal, database, tasks, metadata, roles = caller_inputs
    if defect == "role":
        roles.female.full_name = "A guessed gender"
    elif defect == "id":
        database.customers[1].customer_id = "sarah_johnson_1001"
    elif defect == "dob":
        database.customers[1].date_of_birth = database.customers[0].date_of_birth
    elif defect == "prose":
        tasks[
            0
        ].user_scenario.instructions.known_info = (
            "You are Someone Else with phone number 1."
        )
    else:
        metadata[tasks[0].id].caller_gender = None
    with pytest.raises(ValueError, match=error):
        localize_telecom_callers(*caller_inputs)

"""Reviewed Spanish banking caller text stays aligned with localized records."""

from tau2.multilingual.runtime import load_language


def test_spanish_banking_scenario_repairs_are_limited_to_four_tasks():
    before = load_language(
        "es", "banking_knowledge", mode="text", pack_extension="banking"
    ).scenario
    after = load_language(
        "es", "banking_knowledge", mode="text", pack_extension="banking-dates-v1"
    ).scenario
    old_tasks = {task.id: task for task in before.tasks}
    new_tasks = {task.id: task for task in after.tasks}
    assert old_tasks.keys() == new_tasks.keys()
    changed = {
        task_id for task_id in old_tasks if old_tasks[task_id] != new_tasks[task_id]
    }
    assert changed == {
        "task_077_es",
        "task_082_es",
        "task_091_es",
        "task_097_es",
    }
    for task_id in changed:
        old = old_tasks[task_id]
        new = new_tasks[task_id]
        assert old.evaluation_criteria == new.evaluation_criteria
        assert old.initial_state == new.initial_state
        assert old.ticket == new.ticket
        assert old.required_documents == new.required_documents
        assert old.user_tools == new.user_tools
        assert old.agent_opener == new.agent_opener

    assert "living in Barcelona" in new_tasks["task_077_es"].user_scenario.instructions
    instructions = new_tasks["task_082_es"].user_scenario.instructions
    assert "Gimnasio Pulso Urbano Premium charged me" in instructions
    assert "FitLife" not in instructions
    assert "I've been in Logroño all week" in (
        new_tasks["task_091_es"].user_scenario.instructions
    )
    assert "customer is in Logroño" in new_tasks["task_091_es"].description.notes
    assert "living in Huelva" in new_tasks["task_097_es"].user_scenario.instructions

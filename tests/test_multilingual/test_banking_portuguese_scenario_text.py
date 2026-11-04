"""Reviewed Portuguese banking caller text stays aligned with localized records."""

from tau2.multilingual.runtime import load_language


def test_portuguese_banking_scenario_repairs_are_limited_to_two_tasks():
    before = load_language(
        "pt", "banking_knowledge", mode="text", pack_extension="banking"
    ).scenario
    after = load_language(
        "pt", "banking_knowledge", mode="text", pack_extension="banking-dates-v1"
    ).scenario
    assert before.database == after.database
    assert before.user_database == after.user_database
    old_tasks = {task.id: task for task in before.tasks}
    new_tasks = {task.id: task for task in after.tasks}
    assert old_tasks.keys() == new_tasks.keys()
    assert {
        task_id for task_id in old_tasks if old_tasks[task_id] != new_tasks[task_id]
    } == {"task_082_pt", "task_097_pt"}
    replacements = {
        "task_082_pt": (
            "FitLife charged me",
            "Academia Corpo Pleno Premium charged me",
        ),
        "task_097_pt": ("living in Seattle", "living in São Paulo"),
    }
    for task_id, (old_text, new_text) in replacements.items():
        old = old_tasks[task_id].model_dump(mode="json")
        new = new_tasks[task_id].model_dump(mode="json")
        instruction = old["user_scenario"]["instructions"]
        assert instruction.count(old_text) == 1
        old["user_scenario"]["instructions"] = instruction.replace(old_text, new_text)
        assert new == old
        assert (
            old_tasks[task_id].evaluation_criteria
            == new_tasks[task_id].evaluation_criteria
        )
        assert old_tasks[task_id].initial_state == new_tasks[task_id].initial_state

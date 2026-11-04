"""Reviewed banking caller prose matches regional merchants and residences."""

import pytest

from tau2.multilingual.runtime import load_language


@pytest.mark.parametrize(
    "language,replacements",
    [
        ("hi", {82: ("FitLife charged me", "गोल्ड्स जिम इंडिया charged me")}),
        (
            "ko",
            {
                82: ("FitLife charged me", "밸런스핏 프리미엄 charged me"),
                91: (
                    "I've been in Seattle all week",
                    "I've been in 서울특별시 all week",
                ),
            },
        ),
        (
            "zh",
            {
                82: ("FitLife charged me", "劲活健身会所 charged me"),
                90: ("my apartment in Boston", "my apartment in 西安市"),
                97: ("living in Seattle", "living in 兰州市"),
            },
        ),
    ],
)
def test_other_scenario_text_repairs_are_exact(language, replacements):
    old = load_language(
        language, "banking_knowledge", mode="text", pack_extension="banking"
    ).scenario
    new = load_language(
        language, "banking_knowledge", mode="text", pack_extension="banking-dates-v1"
    ).scenario
    old_tasks = {task.id: task for task in old.tasks}
    new_tasks = {task.id: task for task in new.tasks}
    expected_ids = {f"task_{number:03}_{language}" for number in replacements}
    changed_ids = {
        task_id for task_id in old_tasks if old_tasks[task_id] != new_tasks[task_id]
    }
    if language in {"ko", "zh"}:
        assert expected_ids <= changed_ids
    else:
        assert changed_ids == expected_ids
        assert old.database == new.database
        assert old.user_database == new.user_database
    for number, (before, after) in replacements.items():
        task_id = f"task_{number:03}_{language}"
        old_task = old_tasks[task_id]
        new_task = new_tasks[task_id]
        assert old_task.user_scenario.instructions.count(before) == 1
        assert after in new_task.user_scenario.instructions
        assert before not in new_task.user_scenario.instructions
        if language not in {"ko", "zh"}:
            assert old_task.evaluation_criteria == new_task.evaluation_criteria
            assert old_task.initial_state == new_task.initial_state

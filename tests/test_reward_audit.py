from reward_audit import find_no_criteria, repair_reward_basis


def test_airline_adds_assertion_without_dropping_state_checks():
    tasks = [{"id": "T0", "reward_basis": ["DB_STATE"], "nl_assertions": ["refuse"]}]
    repaired = repair_reward_basis(tasks, "airline")
    assert repaired[0]["reward_basis"] == ["DB_STATE", "NL_ASSERTION"]
    assert tasks[0]["reward_basis"] == ["DB_STATE"]


def test_airline_removes_empty_communication_gate():
    tasks = [{"id": "T20", "reward_basis": ["DB_STATE", "COMMUNICATE"]}]
    assert repair_reward_basis(tasks, "airline")[0]["reward_basis"] == ["DB_STATE"]


def test_airline_preserves_real_communication_tasks():
    task = {"id": "T3", "reward_basis": [{"type": "COMMUNICATE", "criteria": "quote"}]}
    assert repair_reward_basis([task], "airline") == [task]


def test_retail_no_criteria_is_reported_not_fabricated():
    task = {"id": "T25", "reward_basis": []}
    assert find_no_criteria([task], "retail") == ["T25"]
    assert repair_reward_basis([task], "retail") == [task]


def test_populated_retail_task_is_not_reported():
    task = {"id": "T25", "reward_basis": [{"type": "NL_ASSERTION", "criteria": "tracking"}]}
    assert find_no_criteria([task], "retail") == []


def test_malformed_ids_are_ignored_safely():
    task = {"id": "unknown", "reward_basis": ["COMMUNICATE"]}
    assert repair_reward_basis([task], "airline")[0]["reward_basis"] == []

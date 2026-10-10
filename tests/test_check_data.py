from tau2.scripts.check_data import find_reward_basis_issues


def test_reward_basis_reports_empty_gates():
    errors, warnings = find_reward_basis_issues(
        [
            {
                "id": "empty-criteria",
                "evaluation_criteria": {
                    "communicate_info": [],
                    "nl_assertions": None,
                    "reward_basis": ["DB", "COMMUNICATE", "NL_ASSERTION"],
                },
            }
        ]
    )

    assert errors == [
        "task empty-criteria: COMMUNICATE is in reward_basis but "
        "communicate_info is empty",
        "task empty-criteria: NL_ASSERTION is in reward_basis but "
        "nl_assertions is empty",
    ]
    assert warnings == []


def test_reward_basis_reports_populated_criteria_that_are_not_gates():
    errors, warnings = find_reward_basis_issues(
        [
            {
                "id": "missing-gates",
                "evaluation_criteria": {
                    "communicate_info": ["tracking number"],
                    "nl_assertions": ["The agent follows policy."],
                    "reward_basis": ["DB"],
                },
            }
        ]
    )

    assert errors == []
    assert warnings == [
        "task missing-gates: communicate_info is populated but "
        "COMMUNICATE is not in reward_basis",
    ]


def test_reward_basis_uses_default_basis_when_omitted():
    errors, warnings = find_reward_basis_issues(
        [
            {
                "id": "default-basis",
                "evaluation_criteria": {
                    "communicate_info": ["status updated"],
                },
            }
        ]
    )

    assert errors == []
    assert warnings == []

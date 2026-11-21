from tau2.domains.airline.environment import get_environment, get_tasks


def test_task_21_gift_card_instruction_matches_gold_payment():
    """Task 21's payment instruction must be satisfiable and pick the gold card (#664).

    The user's smallest gift card cannot cover the flight change, so the
    instruction has to name the smallest card that can, which is the gold's.
    """
    task = next(t for t in get_tasks("base") if t.id == "21")
    instructions = task.user_scenario.instructions.task_instructions
    assert "smallest balance that can cover the full amount" in instructions

    env = get_environment()
    user = env.tools.db.users["sofia_kim_7287"]
    balances = {
        pid: pm.amount
        for pid, pm in user.payment_methods.items()
        if pm.source == "gift_card"
    }

    change = next(
        a
        for a in task.evaluation_criteria.actions
        if a.name == "update_reservation_flights"
    )
    env.make_tool_call(change.name, requestor=change.requestor, **change.arguments)
    reservation = env.tools.db.reservations[change.arguments["reservation_id"]]
    charged = reservation.payment_history[-1].amount

    covering = [pid for pid, balance in balances.items() if balance >= charged]
    assert min(balances, key=balances.get) not in covering
    assert change.arguments["payment_id"] == min(covering, key=balances.get)

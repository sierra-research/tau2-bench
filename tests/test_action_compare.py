from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Action

EXCHANGE = "exchange_delivered_order_items"


def _gold(**arguments) -> Action:
    return Action(action_id="a", name=EXCHANGE, arguments=arguments)


def _call(**arguments) -> ToolCall:
    return ToolCall(id="t", name=EXCHANGE, arguments=arguments)


def test_list_argument_order_is_ignored():
    gold = _gold(order_id="#W1", item_ids=["1", "2"], payment_method_id="c1")
    assert gold.compare_with_tool_call(
        _call(order_id="#W1", item_ids=["2", "1"], payment_method_id="c1")
    )


def test_paired_lists_reordered_together_match():
    gold = _gold(order_id="#W1", item_ids=["1", "2"], new_item_ids=["x", "y"])
    assert gold.compare_with_tool_call(
        _call(order_id="#W1", item_ids=["2", "1"], new_item_ids=["y", "x"])
    )


def test_paired_lists_with_swapped_pairs_do_not_match():
    gold = _gold(order_id="#W1", item_ids=["1", "2"], new_item_ids=["x", "y"])
    assert not gold.compare_with_tool_call(
        _call(order_id="#W1", item_ids=["1", "2"], new_item_ids=["y", "x"])
    )


def test_different_list_items_do_not_match():
    gold = _gold(order_id="#W1", item_ids=["1", "2"])
    assert not gold.compare_with_tool_call(_call(order_id="#W1", item_ids=["1", "3"]))
    assert not gold.compare_with_tool_call(_call(order_id="#W1", item_ids=["1"]))
    assert not gold.compare_with_tool_call(
        _call(order_id="#W1", item_ids=["1", "1", "2"])
    )

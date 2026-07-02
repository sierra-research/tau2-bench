"""Tests for the core ErrorInjector engine."""

import json
import pytest

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.tasks import Action, EvaluationCriteria, Task, UserScenario
from tau_robustness.injection_config import (
    FieldMutation,
    InjectionConfig,
    InjectionDef,
    InjectionType,
)
from tau_robustness.injector import ErrorInjector


@pytest.fixture
def retail_config():
    """A minimal retail injection config for testing."""
    return InjectionConfig(
        domain="retail",
        injections=[
            InjectionDef(
                id="stale_order",
                type=InjectionType.STALE_DATA,
                target_tool="get_order_details",
                description="Order shows pending but is delivered",
                difficulty=2,
                mutations=[
                    FieldMutation(field_path="status", action="set", value="pending")
                ],
                precondition={"status": "delivered"},
                detection_signals=["already delivered"],
                recovery_signals=["get_order_details"],
            ),
            InjectionDef(
                id="missing_address",
                type=InjectionType.MISSING_DATA,
                target_tool="get_order_details",
                description="Address field removed",
                difficulty=1,
                mutations=[FieldMutation(field_path="address", action="delete")],
                detection_signals=["missing address"],
                recovery_signals=["ask for address"],
            ),
        ],
    )


def make_tool_call(name: str, call_id: str = "tc_1", **kwargs) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=kwargs)


def make_tool_response(
    content: dict, call_id: str = "tc_1", error: bool = False
) -> ToolMessage:
    return ToolMessage(
        id=call_id,
        role="tool",
        content=json.dumps(content),
        requestor="assistant",
        error=error,
    )


class TestErrorInjector:
    def test_no_injection_on_unknown_tool(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        tc = make_tool_call("find_user_id_by_email", email="test@test.com")
        resp = make_tool_response({"user_id": "u123"})

        result = injector.maybe_inject(tc, resp, turn_idx=5)
        assert result.content == resp.content
        assert injector.num_injections == 0

    def test_no_injection_on_error_response(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"error": "not found"}, error=True)

        result = injector.maybe_inject(tc, resp, turn_idx=5)
        assert result.content == resp.content
        assert injector.num_injections == 0

    def test_no_injection_before_min_turn(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42, min_turn=5)
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        result = injector.maybe_inject(tc, resp, turn_idx=3)
        assert result.content == resp.content
        assert injector.num_injections == 0

    def test_no_injection_after_max_turn(self, retail_config):
        injector = ErrorInjector(
            retail_config, injection_rate=1.0, seed=42, max_turn=10
        )
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        result = injector.maybe_inject(tc, resp, turn_idx=15)
        assert result.content == resp.content
        assert injector.num_injections == 0

    def test_injection_respects_rate_zero(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=0.0, seed=42)
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        result = injector.maybe_inject(tc, resp, turn_idx=5)
        assert result.content == resp.content
        assert injector.num_injections == 0

    def test_injection_applies_with_rate_one(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        result = injector.maybe_inject(tc, resp, turn_idx=5)
        # Should have injected — content should be different
        assert injector.num_injections == 1
        result_data = json.loads(result.content)
        # One of the two injections should have been applied
        # (stale_order flips delivered→pending, or missing_address removes address)
        assert result_data.get("status") == "pending" or "address" not in result_data

    def test_max_injections_per_run(self, retail_config):
        injector = ErrorInjector(
            retail_config,
            injection_rate=1.0,
            seed=42,
            max_injections_per_run=1,
        )
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        # First call should inject
        injector.maybe_inject(tc, resp, turn_idx=5)
        assert injector.num_injections == 1

        # Second call should NOT inject (max reached)
        resp2 = make_tool_response({"status": "delivered", "address": "456 Oak Ave"})
        result2 = injector.maybe_inject(tc, resp2, turn_idx=8)
        assert injector.num_injections == 1
        assert json.loads(result2.content)["status"] == "delivered"

    def test_precondition_filters(self, retail_config):
        # stale_order requires status=="delivered", pending orders should not match
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=100)
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "pending", "address": "123 Main St"})

        result = injector.maybe_inject(tc, resp, turn_idx=5)
        # stale_order precondition fails, but missing_address has no precondition
        if injector.num_injections > 0:
            result_data = json.loads(result.content)
            # If injected, it must be missing_address (not stale_order)
            assert "address" not in result_data

    def test_reset_clears_state(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        injector.maybe_inject(tc, resp, turn_idx=5)
        assert injector.num_injections == 1

        injector.reset()
        assert injector.num_injections == 0
        assert injector.injection_log == []

    def test_injection_log_records_details(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        tc = make_tool_call("get_order_details", order_id="o123", call_id="tc_42")
        resp = make_tool_response(
            {"status": "delivered", "address": "123 Main St"}, call_id="tc_42"
        )

        injector.maybe_inject(tc, resp, turn_idx=7)
        assert len(injector.injection_log) == 1

        event = injector.injection_log[0]
        assert event.turn_idx == 7
        assert event.tool_name == "get_order_details"
        assert event.original_content is not None
        assert event.modified_content is not None
        assert event.original_content != event.modified_content

    def test_deterministic_with_same_seed(self, retail_config):
        """Same seed should produce identical injection patterns."""
        results_a = []
        results_b = []

        for results, seed_val in [(results_a, 42), (results_b, 42)]:
            injector = ErrorInjector(
                retail_config,
                injection_rate=0.5,
                seed=seed_val,
                max_injections_per_run=5,
                min_turn=2,
            )
            for i in range(10):
                tc = make_tool_call("get_order_details", order_id=f"o{i}")
                resp = make_tool_response(
                    {"status": "delivered", "address": f"{i} Main St"}
                )
                result = injector.maybe_inject(tc, resp, turn_idx=i + 3)
                results.append(result.content)

        assert results_a == results_b

    def test_get_injection_summary(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})
        injector.maybe_inject(tc, resp, turn_idx=5)

        summary = injector.get_injection_summary()
        assert summary["num_injections"] == 1
        assert len(summary["injections"]) == 1
        assert "injection_id" in summary["injections"][0]
        assert "injection_type" in summary["injections"][0]

    def test_invalid_injection_rate(self, retail_config):
        with pytest.raises(ValueError, match="injection_rate"):
            ErrorInjector(retail_config, injection_rate=1.5)
        with pytest.raises(ValueError, match="injection_rate"):
            ErrorInjector(retail_config, injection_rate=-0.1)

    def test_non_json_content_skipped(self, retail_config):
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = ToolMessage(
            id="tc_1",
            role="tool",
            content="This is not JSON",
            requestor="assistant",
            error=False,
        )
        result = injector.maybe_inject(tc, resp, turn_idx=5)
        assert result.content == "This is not JSON"
        assert injector.num_injections == 0

    def test_batch_size_skips_non_matching_in_large_batches(self, retail_config):
        """In large batches, skip tool calls that don't match user entities."""
        injector = ErrorInjector(
            retail_config, injection_rate=1.0, seed=42, max_batch_size=2
        )
        # User mentioned a DIFFERENT order
        injector.track_user_message("I need help with order #W9999999")

        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        # batch_size=5 and args don't match user entity → should skip
        result = injector.maybe_inject(tc, resp, turn_idx=5, batch_size=5)
        assert result.content == resp.content
        assert injector.num_injections == 0

    def test_batch_size_allows_small_batches(self, retail_config):
        """Small batches (focused calls) should still allow injection."""
        injector = ErrorInjector(
            retail_config, injection_rate=1.0, seed=42, max_batch_size=2
        )
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        # batch_size=1 is a focused call
        injector.maybe_inject(tc, resp, turn_idx=5, batch_size=1)
        assert injector.num_injections == 1

    def test_batch_size_allows_exact_threshold(self, retail_config):
        """Batch size exactly at threshold should still inject."""
        injector = ErrorInjector(
            retail_config, injection_rate=1.0, seed=42, max_batch_size=2
        )
        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        injector.maybe_inject(tc, resp, turn_idx=5, batch_size=2)
        assert injector.num_injections == 1

    def test_track_user_message_extracts_order_ids(self, retail_config):
        """User messages with order IDs should be tracked."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)

        injector.track_user_message("I want to modify order #W6247578")
        assert "#W6247578" in injector._user_entities

    def test_track_user_message_extracts_reservation_codes(self, retail_config):
        """User messages with reservation codes should be tracked."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)

        injector.track_user_message("My reservation is EHGLP3")
        assert "EHGLP3" in injector._user_entities

    def test_track_user_message_extracts_phone_numbers(self, retail_config):
        """User messages with phone numbers should be tracked."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)

        injector.track_user_message("My phone number is 555-123-2002")
        assert "555-123-2002" in injector._user_entities

    def test_reset_clears_user_entities(self, retail_config):
        """Reset should clear tracked user entities."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)

        injector.track_user_message("Order #W6247578 please")
        assert len(injector._user_entities) > 0

        injector.reset()
        assert len(injector._user_entities) == 0
        assert len(injector._user_keywords) == 0

    def test_default_min_turn_is_4(self, retail_config):
        """Default min_turn should be 4 to skip auth phase."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        assert injector.min_turn == 4

        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        # Turn 3 should be skipped with default min_turn=4
        injector.maybe_inject(tc, resp, turn_idx=3)
        assert injector.num_injections == 0

    def test_batch_allows_injection_when_args_match_user_entity(self, retail_config):
        """In a large batch, inject if tool call args match a user-mentioned entity."""
        injector = ErrorInjector(
            retail_config, injection_rate=1.0, seed=42, max_batch_size=2
        )
        # User mentioned this specific order
        injector.track_user_message("I want to modify order #W6247578")

        tc = make_tool_call(
            "get_order_details", order_id="#W6247578", call_id="tc_match"
        )
        resp = make_tool_response(
            {"status": "delivered", "address": "123 Main St"}, call_id="tc_match"
        )

        # batch_size=5 but args match user entity → should inject
        injector.maybe_inject(tc, resp, turn_idx=8, batch_size=5)
        assert injector.num_injections == 1

    def test_batch_skips_injection_when_args_dont_match(self, retail_config):
        """In a large batch, skip if tool call args don't match user entities."""
        injector = ErrorInjector(
            retail_config, injection_rate=1.0, seed=42, max_batch_size=2
        )
        # User mentioned a different order
        injector.track_user_message("I want to modify order #W6247578")

        tc = make_tool_call(
            "get_order_details", order_id="#W9999999", call_id="tc_other"
        )
        resp = make_tool_response(
            {"status": "delivered", "address": "456 Oak Ave"}, call_id="tc_other"
        )

        # batch_size=5 and args DON'T match → should skip
        result = injector.maybe_inject(tc, resp, turn_idx=8, batch_size=5)
        assert injector.num_injections == 0
        assert result.content == resp.content

    def test_batch_allows_when_no_user_entities_tracked(self, retail_config):
        """In a large batch with no user context, fall back to allowing injection."""
        injector = ErrorInjector(
            retail_config, injection_rate=1.0, seed=42, max_batch_size=2
        )
        # No user messages tracked — can't filter, so allow injection

        tc = make_tool_call("get_order_details", order_id="#W6247578")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        injector.maybe_inject(tc, resp, turn_idx=8, batch_size=5)
        assert injector.num_injections == 1

    def test_default_max_turn_is_30(self, retail_config):
        """Default max_turn should be 30 to avoid late injections."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        assert injector.max_turn == 30

        tc = make_tool_call("get_order_details", order_id="o123")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})

        # Turn 35 should be skipped with default max_turn=30
        injector.maybe_inject(tc, resp, turn_idx=35)
        assert injector.num_injections == 0

    def test_set_task_extracts_action_tools(self, retail_config):
        """set_task() should extract action tool names from evaluation criteria."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        task = Task(
            id="test_task",
            user_scenario=UserScenario(
                instructions="I want to exchange an item",
            ),
            evaluation_criteria=EvaluationCriteria(
                actions=[
                    Action(
                        action_id="a1",
                        name="exchange_delivered_order_items",
                        arguments={"order_id": "o1"},
                    ),
                    Action(
                        action_id="a2",
                        name="get_order_details",
                        arguments={"order_id": "o1"},
                    ),
                ]
            ),
        )
        injector.set_task(task)
        assert injector._task_action_tools == {
            "exchange_delivered_order_items",
            "get_order_details",
        }
        # Also sets task description for keyword matching
        assert injector._task_description is not None

    def test_set_task_handles_no_actions(self, retail_config):
        """set_task() with no evaluation actions should set empty tools."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        task = Task(
            id="test_task",
            user_scenario=UserScenario(
                instructions="Just a question",
            ),
            evaluation_criteria=EvaluationCriteria(actions=None),
        )
        injector.set_task(task)
        assert injector._task_action_tools == set()

    def test_reset_clears_task_action_tools(self, retail_config):
        """reset() should clear _task_action_tools."""
        injector = ErrorInjector(retail_config, injection_rate=1.0, seed=42)
        injector._task_action_tools = {"some_tool"}
        injector.reset()
        assert injector._task_action_tools == set()

    def test_blocking_injections_tried_before_cosmetic(self):
        """Blocking injections (blocks_actions overlaps task) tried before cosmetic ones."""
        config = InjectionConfig(
            domain="retail",
            injections=[
                # Cosmetic injection (no blocks_actions)
                InjectionDef(
                    id="cosmetic",
                    type=InjectionType.MISSING_DATA,
                    target_tool="get_order_details",
                    description="Cosmetic: removes address",
                    difficulty=1,
                    mutations=[FieldMutation(field_path="address", action="delete")],
                    blocks_actions=[],
                ),
                # Blocking injection
                InjectionDef(
                    id="blocking",
                    type=InjectionType.STATUS_FLIP,
                    target_tool="get_order_details",
                    description="Blocking: flips delivered→cancelled",
                    difficulty=2,
                    mutations=[
                        FieldMutation(
                            field_path="status",
                            action="flip",
                            flip_map={"delivered": "cancelled"},
                        )
                    ],
                    precondition={"status": "delivered"},
                    blocks_actions=["exchange_delivered_order_items"],
                ),
            ],
        )
        # Try multiple seeds — when both match, the blocking one should always fire
        for seed in range(20):
            injector = ErrorInjector(config, injection_rate=1.0, seed=seed)
            injector._task_action_tools = {"exchange_delivered_order_items"}

            tc = make_tool_call("get_order_details", order_id="o1")
            resp = make_tool_response(
                {"status": "delivered", "address": "123 Main St"}
            )
            injector.maybe_inject(tc, resp, turn_idx=5)
            assert injector.num_injections == 1
            # The blocking injection should always be chosen since it's tried first
            assert injector.injection_log[0].injection_id == "blocking"

    def test_no_blocks_actions_falls_back_to_shuffle(self):
        """When no injection has blocks_actions, all go to cosmetic (old behavior)."""
        config = InjectionConfig(
            domain="retail",
            injections=[
                InjectionDef(
                    id="inj_a",
                    type=InjectionType.MISSING_DATA,
                    target_tool="get_order_details",
                    description="Injection A",
                    difficulty=1,
                    mutations=[FieldMutation(field_path="address", action="delete")],
                    # No blocks_actions → cosmetic
                ),
                InjectionDef(
                    id="inj_b",
                    type=InjectionType.STALE_DATA,
                    target_tool="get_order_details",
                    description="Injection B",
                    difficulty=2,
                    mutations=[
                        FieldMutation(
                            field_path="status", action="set", value="pending"
                        )
                    ],
                    precondition={"status": "delivered"},
                    # No blocks_actions → cosmetic
                ),
            ],
        )
        injector = ErrorInjector(config, injection_rate=1.0, seed=42)
        # Even with task_action_tools set, no injections have blocks_actions
        injector._task_action_tools = {"exchange_delivered_order_items"}

        tc = make_tool_call("get_order_details", order_id="o1")
        resp = make_tool_response({"status": "delivered", "address": "123 Main St"})
        injector.maybe_inject(tc, resp, turn_idx=5)
        # Should still inject (one of the cosmetic ones)
        assert injector.num_injections == 1

    def test_blocking_falls_to_cosmetic_when_precondition_fails(self):
        """If blocking injection's precondition fails, fall through to cosmetic."""
        config = InjectionConfig(
            domain="retail",
            injections=[
                InjectionDef(
                    id="blocking",
                    type=InjectionType.STATUS_FLIP,
                    target_tool="get_order_details",
                    description="Blocks exchange",
                    difficulty=2,
                    mutations=[
                        FieldMutation(
                            field_path="status",
                            action="flip",
                            flip_map={"delivered": "cancelled"},
                        )
                    ],
                    precondition={"status": "delivered"},
                    blocks_actions=["exchange_delivered_order_items"],
                ),
                InjectionDef(
                    id="cosmetic",
                    type=InjectionType.MISSING_DATA,
                    target_tool="get_order_details",
                    description="Cosmetic fallback",
                    difficulty=1,
                    mutations=[FieldMutation(field_path="address", action="delete")],
                ),
            ],
        )
        injector = ErrorInjector(config, injection_rate=1.0, seed=42)
        injector._task_action_tools = {"exchange_delivered_order_items"}

        tc = make_tool_call("get_order_details", order_id="o1")
        # Status is pending, so blocking precondition (status=delivered) fails
        resp = make_tool_response({"status": "pending", "address": "123 Main St"})
        injector.maybe_inject(tc, resp, turn_idx=5)
        assert injector.num_injections == 1
        assert injector.injection_log[0].injection_id == "cosmetic"

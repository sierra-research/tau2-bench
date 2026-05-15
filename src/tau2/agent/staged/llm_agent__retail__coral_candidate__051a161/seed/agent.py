from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from typing import Optional

from pydantic import BaseModel

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import (
    HalfDuplexAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

AGENT_INSTRUCTION = """
You are a disciplined customer service agent that helps the user according to the <policy> provided below.
In each turn you must do exactly one of the following:
- Send a message to the user.
- Make exactly one tool call.
You cannot do both at the same time, and you cannot make multiple tool calls in one turn.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

AUTH_TOOL_NAMES = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
}

ORDER_DISCOVERY_CUES = (
    "don't have the order",
    "do not have the order",
    "don't have the order id",
    "do not have the order id",
    "don't know the order",
    "do not know the order",
    "don't know my order number",
    "do not know my order number",
    "don't remember the order",
    "do not remember the order",
)

ORDER_BULK_DISCOVERY_CUES = (
    "all possible orders",
    "cancel or return all possible orders",
    "return everything",
    "don't remember exactly which items",
    "do not remember exactly which items",
    "five items",
)

ORDER_CROSS_ORDER_DISCOVERY_CUES = (
    "other order",
    "another order",
    "contains both",
    "same one as in my other order",
    "same as my other",
    "match my other",
    "other bottle",
)

ORDER_ACTION_CUES = (
    "arriving",
    "order status",
    "tracking",
    "exchange",
    "return",
    "cancel",
    "modify",
    "change",
    "swap",
    "replace",
    "ordered",
    "bought",
    "purchased",
)

ADDRESS_SEMANTIC_CUES = (
    "address",
    "zip",
    "moved",
    "default address",
    "current address",
    "shipping address",
    "ship to",
    "new york address",
    "la order",
    "nyc address",
    "other order",
    "orders profile",
)

ORDER_ID_PATTERN = re.compile(r"#w\d+", re.IGNORECASE)

RECENT_ORDER_RESOLUTION_CUES = (
    "most recent order",
    "latest order",
    "recent order",
    "recent purchase",
    "most recently placed",
    "most recently ordered",
    "most recently bought",
)

UNCERTAIN_ORDER_ID_CUES = (
    "not sure",
    "not 100% sure",
    "not completely sure",
    "i think",
    "i guess",
    "maybe",
    "might be",
)

TRANSFER_TOOL_NAME = "transfer_to_human_agents"

WRITE_TOOL_NAMES = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}

WRITE_CONFIRMATION_GUARDRAIL = """
You attempted to call a write tool before the required confirmation step.
Do not call a write tool in this turn.
Instead, send a message that:
- summarizes the exact action details that would be taken;
- includes any price difference or refund destination/timing that follows from policy;
- asks for explicit yes/no confirmation.
If the derived retail reasoning note identified a single best available variant from the user's own stated preferences, treat that variant as the selected option unless the user explicitly overrides it.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def _record_guardrail_metadata(message: AssistantMessage, **metadata: object) -> None:
    if message.raw_data is None:
        message.raw_data = {}
    if not isinstance(message.raw_data, dict):
        return
    guardrail_info = message.raw_data.setdefault("policy_guardrails", {})
    if isinstance(guardrail_info, dict):
        guardrail_info.update(metadata)


def _sanitize_assistant_message(message: AssistantMessage) -> AssistantMessage:
    sanitized = message.model_copy(deep=True)
    if sanitized.tool_calls:
        if len(sanitized.tool_calls) > 1:
            sanitized.tool_calls = [sanitized.tool_calls[0]]
            _record_guardrail_metadata(
                sanitized,
                dropped_tool_calls=len(message.tool_calls or []) - 1,
            )
        if sanitized.content and sanitized.content.strip():
            sanitized.content = None
            _record_guardrail_metadata(sanitized, stripped_content_for_tool_call=True)
    sanitized.validate()
    return sanitized


def _parse_tool_payload(tool_message: ToolMessage) -> dict | None:
    if tool_message.error or not tool_message.content:
        return None
    try:
        payload = json.loads(tool_message.content)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_guardrail_tool_call(
    tool_name: str,
    arguments: dict,
    *,
    reason: str,
) -> AssistantMessage:
    message = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id=f"guardrail_{tool_name}_{uuid.uuid4().hex[:12]}",
                name=tool_name,
                arguments=arguments,
            )
        ],
    )
    _record_guardrail_metadata(message, forced_tool_call=reason)
    message.validate()
    return message


def _extract_auth_user_id(
    tool_call: ToolCall,
    tool_message: ToolMessage,
) -> str | None:
    if tool_call.name not in AUTH_TOOL_NAMES or tool_message.error or not tool_message.content:
        return None
    user_id = tool_message.content.strip()
    if not user_id or user_id.lower().startswith("error"):
        return None
    return user_id


def _authenticated_user_id(messages: list[APICompatibleMessage]) -> str | None:
    for tool_call, tool_message in reversed(_iter_tool_results(messages)):
        user_id = _extract_auth_user_id(tool_call, tool_message)
        if user_id is not None:
            return user_id
    return None


def _has_failed_auth_attempt(messages: list[APICompatibleMessage]) -> bool:
    return any(
        tool_call.name in AUTH_TOOL_NAMES
        and _extract_auth_user_id(tool_call, tool_message) is None
        for tool_call, tool_message in _iter_tool_results(messages)
    )


def _has_user_details(
    messages: list[APICompatibleMessage],
    *,
    user_id: str | None = None,
) -> bool:
    for tool_call, tool_message in _iter_tool_results(messages):
        if tool_call.name != "get_user_details":
            continue
        payload = _parse_tool_payload(tool_message)
        if payload is None:
            continue
        if user_id is None or str(payload.get("user_id")) == user_id:
            return True
    return False


def _known_order_ids(
    messages: list[APICompatibleMessage],
    *,
    user_id: str | None = None,
) -> list[str]:
    for tool_call, tool_message in reversed(_iter_tool_results(messages)):
        if tool_call.name != "get_user_details":
            continue
        payload = _parse_tool_payload(tool_message)
        if payload is None:
            continue
        if user_id is not None and str(payload.get("user_id")) != user_id:
            continue
        orders = payload.get("orders")
        if not isinstance(orders, list):
            return []
        return [str(order_id) for order_id in orders if order_id]
    return []


def _inspected_order_ids(messages: list[APICompatibleMessage]) -> set[str]:
    inspected: set[str] = set()
    for tool_call, tool_message in _iter_tool_results(messages):
        if tool_call.name != "get_order_details":
            continue
        payload = _parse_tool_payload(tool_message)
        if payload is None:
            continue
        order_id = payload.get("order_id")
        if order_id:
            inspected.add(str(order_id))
    return inspected


def _all_user_text(messages: list[APICompatibleMessage]) -> str:
    return " ".join(
        message.content
        for message in messages
        if isinstance(message, UserMessage) and message.content
    ).lower()


def _latest_user_text(messages: list[APICompatibleMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, UserMessage) and message.content:
            return message.content.lower()
    return ""


def _relevant_user_text(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool,
) -> str:
    return _latest_user_text(messages) if latest_only else _all_user_text(messages)


def _text_has_any_cue(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _user_has_explicit_order_id(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool = False,
) -> bool:
    return bool(ORDER_ID_PATTERN.search(_relevant_user_text(messages, latest_only=latest_only)))


def _mentioned_order_ids(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool = False,
) -> list[str]:
    user_text = _relevant_user_text(messages, latest_only=latest_only)
    order_ids: list[str] = []
    for match in ORDER_ID_PATTERN.finditer(user_text):
        order_id = match.group(0).upper()
        if order_id not in order_ids:
            order_ids.append(order_id)
    return order_ids


def _user_has_uncertain_order_id(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool = False,
) -> bool:
    user_text = _relevant_user_text(messages, latest_only=latest_only)
    return bool(ORDER_ID_PATTERN.search(user_text)) and _text_has_any_cue(
        user_text,
        UNCERTAIN_ORDER_ID_CUES,
    )


def _user_needs_order_discovery(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool = False,
) -> bool:
    user_text = _relevant_user_text(messages, latest_only=latest_only)
    return _text_has_any_cue(user_text, ORDER_DISCOVERY_CUES)


def _user_requests_bulk_order_discovery(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool = False,
) -> bool:
    user_text = _relevant_user_text(messages, latest_only=latest_only)
    return _text_has_any_cue(user_text, ORDER_BULK_DISCOVERY_CUES)


def _user_requests_cross_order_discovery(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool = False,
) -> bool:
    user_text = _relevant_user_text(messages, latest_only=latest_only)
    return _text_has_any_cue(user_text, ORDER_CROSS_ORDER_DISCOVERY_CUES)


def _user_requests_recent_order_resolution(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool = False,
) -> bool:
    user_text = _relevant_user_text(messages, latest_only=latest_only)
    return _text_has_any_cue(user_text, RECENT_ORDER_RESOLUTION_CUES)


def _user_requests_existing_purchase_lookup(
    messages: list[APICompatibleMessage],
    *,
    latest_only: bool = False,
) -> bool:
    if _user_has_explicit_order_id(messages, latest_only=latest_only) and not _user_has_uncertain_order_id(
        messages,
        latest_only=latest_only,
    ):
        return False
    user_text = _relevant_user_text(messages, latest_only=latest_only)
    return _text_has_any_cue(user_text, ORDER_ACTION_CUES)


def _user_mentions_address_semantics(messages: list[APICompatibleMessage]) -> bool:
    user_text = _all_user_text(messages)
    return any(cue in user_text for cue in ADDRESS_SEMANTIC_CUES)


def _user_likely_moved_or_used_new_zip(messages: list[APICompatibleMessage]) -> bool:
    user_text = _all_user_text(messages)
    cues = (
        "moved",
        "new zip",
        "new address",
        "old zip",
        "previous zip",
        "current address should be in my recent order",
    )
    return any(cue in user_text for cue in cues)


def _build_auth_recovery_message(
    messages: list[APICompatibleMessage],
) -> AssistantMessage:
    if _user_likely_moved_or_used_new_zip(messages):
        return AssistantMessage.text(
            "I couldn't verify the account with that ZIP yet. Since you mentioned you moved, "
            "please share the previous ZIP code on the account or another email you may have used "
            "so I can authenticate you and continue."
        )
    return AssistantMessage.text(
        "I couldn't verify the account with that information yet. Please share another email you may "
        "have used or the ZIP code currently on the account so I can authenticate you and continue."
    )


def _build_post_auth_user_details_call(
    messages: list[APICompatibleMessage],
) -> AssistantMessage | None:
    tool_results = _iter_tool_results(messages)
    if not tool_results:
        return None
    tool_call, tool_message = tool_results[-1]
    user_id = _extract_auth_user_id(tool_call, tool_message)
    if user_id is None or _has_user_details(messages, user_id=user_id):
        return None
    return _build_guardrail_tool_call(
        "get_user_details",
        {"user_id": user_id},
        reason="eager_post_auth_user_details",
    )


def _latest_user_details_tool_index(
    messages: list[APICompatibleMessage],
    *,
    user_id: str | None = None,
) -> int | None:
    pending_calls: dict[str, ToolCall] = {}
    latest_index: int | None = None
    for index, message in enumerate(messages):
        if isinstance(message, AssistantMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                pending_calls[tool_call.id] = tool_call
            continue
        if not isinstance(message, ToolMessage):
            continue
        tool_call = pending_calls.get(message.id)
        if tool_call is None or tool_call.name != "get_user_details":
            continue
        payload = _parse_tool_payload(message)
        if payload is None:
            continue
        if user_id is not None and str(payload.get("user_id")) != user_id:
            continue
        latest_index = index
    return latest_index


def _assistant_replied_after_index(messages: list[APICompatibleMessage], index: int) -> bool:
    return any(
        isinstance(message, AssistantMessage)
        and bool(message.content and message.content.strip())
        and not message.tool_calls
        for message in messages[index + 1 :]
    )


def _in_pre_response_order_discovery_window(
    messages: list[APICompatibleMessage],
    *,
    user_id: str,
) -> bool:
    latest_user_details_index = _latest_user_details_tool_index(messages, user_id=user_id)
    if latest_user_details_index is None:
        return False
    return not _assistant_replied_after_index(messages, latest_user_details_index)


def _latest_order_discovery_user_index(messages: list[APICompatibleMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, UserMessage) or not message.content:
            continue
        user_text = message.content.lower()
        if (
            _text_has_any_cue(user_text, ORDER_DISCOVERY_CUES)
            or _text_has_any_cue(user_text, ORDER_BULK_DISCOVERY_CUES)
            or _text_has_any_cue(user_text, ORDER_CROSS_ORDER_DISCOVERY_CUES)
            or _text_has_any_cue(user_text, RECENT_ORDER_RESOLUTION_CUES)
            or (
                ORDER_ID_PATTERN.search(user_text)
                and _text_has_any_cue(user_text, UNCERTAIN_ORDER_ID_CUES)
            )
        ):
            return index
    return None


def _in_latest_order_discovery_turn_window(
    messages: list[APICompatibleMessage],
) -> bool:
    latest_discovery_user_index = _latest_order_discovery_user_index(messages)
    if latest_discovery_user_index is None:
        return False
    return not _assistant_replied_after_index(messages, latest_discovery_user_index)


def _order_discovery_candidates(
    messages: list[APICompatibleMessage],
    known_orders: list[str],
    *,
    latest_only: bool,
) -> list[str]:
    candidates: list[str] = []
    for order_id in _mentioned_order_ids(messages, latest_only=latest_only):
        if order_id in known_orders and order_id not in candidates:
            candidates.append(order_id)
    for order_id in reversed(known_orders):
        if order_id not in candidates:
            candidates.append(order_id)
    return candidates


def _normalized_order_status(status: object) -> str:
    normalized = str(status or "").strip().lower()
    if normalized.startswith("pending"):
        return "pending"
    if normalized.startswith("delivered"):
        return "delivered"
    if normalized.startswith("cancelled"):
        return "cancelled"
    return normalized


def _requested_order_statuses(messages: list[APICompatibleMessage]) -> set[str]:
    user_text = _all_user_text(messages)
    requested_statuses: set[str] = set()
    if "return" in user_text or "exchange" in user_text:
        requested_statuses.add("delivered")
    pending_cues = (
        "cancel",
        "modify",
        "update the shipping address",
        "update my shipping address",
        "change the shipping address",
        "change my shipping address",
        "update the address",
        "update my address",
        "change the address",
        "change my address",
        "payment method",
        "split the payment",
    )
    if _text_has_any_cue(user_text, pending_cues):
        requested_statuses.add("pending")
    return requested_statuses


def _inspected_order_statuses(
    messages: list[APICompatibleMessage],
    *,
    candidate_orders: list[str],
) -> set[str]:
    candidate_set = set(candidate_orders)
    statuses: set[str] = set()
    for tool_call, tool_message in _iter_tool_results(messages):
        if tool_call.name != "get_order_details":
            continue
        payload = _parse_tool_payload(tool_message)
        if payload is None:
            continue
        order_id = str(payload.get("order_id", ""))
        if order_id not in candidate_set:
            continue
        statuses.add(_normalized_order_status(payload.get("status")))
    return statuses


def _target_order_discovery_depth(
    messages: list[APICompatibleMessage],
    *,
    known_orders: list[str],
    latest_only: bool,
) -> int:
    if not known_orders:
        return 0
    if _user_requests_bulk_order_discovery(messages, latest_only=latest_only):
        return len(known_orders)
    if _user_requests_cross_order_discovery(messages, latest_only=latest_only):
        return min(len(known_orders), 3)
    if _user_has_uncertain_order_id(messages, latest_only=latest_only):
        return min(len(known_orders), 3)
    if _user_requests_recent_order_resolution(messages, latest_only=latest_only):
        return min(len(known_orders), 3)
    if len(known_orders) == 1 and _user_requests_existing_purchase_lookup(
        messages,
        latest_only=latest_only,
    ):
        return 1
    if _user_needs_order_discovery(messages, latest_only=latest_only):
        return 1
    return 0


def _build_order_discovery_call(
    messages: list[APICompatibleMessage],
    *,
    reason: str,
    require_latest_user_details: bool,
    latest_only: bool = False,
) -> AssistantMessage | None:
    user_id = _authenticated_user_id(messages)
    if user_id is None:
        return None
    if require_latest_user_details and not _in_pre_response_order_discovery_window(
        messages,
        user_id=user_id,
    ):
        return None
    known_orders = _known_order_ids(messages, user_id=user_id)
    if not known_orders:
        return None
    target_depth = _target_order_discovery_depth(
        messages,
        known_orders=known_orders,
        latest_only=latest_only,
    )
    if target_depth <= 0:
        return None
    candidate_orders = _order_discovery_candidates(
        messages,
        known_orders,
        latest_only=latest_only,
    )
    inspected_orders = _inspected_order_ids(messages)
    inspected_candidate_count = sum(1 for order_id in candidate_orders if order_id in inspected_orders)
    requested_statuses = _requested_order_statuses(messages)
    inspected_statuses = _inspected_order_statuses(messages, candidate_orders=candidate_orders)
    if inspected_candidate_count >= target_depth and requested_statuses.issubset(inspected_statuses):
        return None
    for order_id in candidate_orders:
        if order_id not in inspected_orders:
            return _build_guardrail_tool_call(
                "get_order_details",
                {"order_id": order_id},
                reason=reason,
            )
    return None


def _build_transfer_guardrail_response(
    assistant_message: AssistantMessage,
    messages: list[APICompatibleMessage],
) -> AssistantMessage | None:
    if not assistant_message.tool_calls:
        return None
    if assistant_message.tool_calls[0].name != TRANSFER_TOOL_NAME:
        return None
    user_id = _authenticated_user_id(messages)
    if user_id is None:
        if _has_failed_auth_attempt(messages):
            return _build_auth_recovery_message(messages)
        return None
    if not _has_user_details(messages, user_id=user_id):
        return _build_guardrail_tool_call(
            "get_user_details",
            {"user_id": user_id},
            reason="pre_transfer_user_details",
        )
    return _build_order_discovery_call(
        messages,
        reason="pre_transfer_order_discovery",
        require_latest_user_details=False,
    )


def _split_primary_and_fallback_text(user_text: str) -> tuple[str, str]:
    lower_text = user_text.lower()
    markers = [
        " if there is no ",
        " if there are no ",
        " if it is not available ",
        " if it's not available ",
        " if unavailable ",
        " otherwise ",
        " if and only if ",
        " only if ",
    ]
    indices = [lower_text.find(marker) for marker in markers if marker in lower_text]
    if not indices:
        return lower_text, ""
    split_idx = min(idx for idx in indices if idx >= 0)
    return lower_text[:split_idx], lower_text[split_idx:]


def _normalize_free_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("’", "'").replace("`", "'").replace("´", "'")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9#' ]+", " ", normalized)
    normalized = re.sub(r"\binches\b", "inch", normalized)
    normalized = re.sub(r"\bhours\b", "hour", normalized)
    return " ".join(normalized.split())


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {text} "


def _value_requires_boundary_match(value: str) -> bool:
    tokens = value.split()
    if len(tokens) != 1:
        return False
    token = tokens[0]
    return len(token) <= 3 or token in {"yes", "no"}


def _normalized_value_in_text(text: str, value: str) -> bool:
    if not value:
        return False
    if _value_requires_boundary_match(value):
        return _contains_normalized_phrase(text, value)
    return value in text


def _option_matches_text(key: str, value: str, text: str) -> bool:
    lowered_text = _normalize_free_text(text)
    lowered_key = _normalize_free_text(key)
    lowered_value = _normalize_free_text(value)
    if lowered_value and _normalized_value_in_text(lowered_text, lowered_value):
        return True
    if lowered_key in {"water resistance", "waterproof"} and lowered_value in {
        "not resistant",
        "no",
    }:
        negative_cues = (
            "without water resistance",
            "no water resistance",
            "not water resistant",
            "not waterproof",
        )
        return any(
            _contains_normalized_phrase(lowered_text, _normalize_free_text(cue))
            for cue in negative_cues
        )
    return False


def _constrained_option_keys(variants: dict, user_text: str) -> set[str]:
    values_by_key: dict[str, set[str]] = {}
    for variant in variants.values():
        if not isinstance(variant, dict):
            continue
        options = variant.get("options")
        if not isinstance(options, dict):
            continue
        for key, value in options.items():
            values_by_key.setdefault(str(key), set()).add(str(value))
    constrained: set[str] = set()
    lowered_text = _normalize_free_text(user_text)
    for key, values in values_by_key.items():
        if _normalize_free_text(key) in lowered_text:
            constrained.add(key)
            continue
        if any(_option_matches_text(key, value, lowered_text) for value in values):
            constrained.add(key)
    return constrained


def _normalize_option_map(options: dict) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in options.items():
        normalized[_normalize_free_text(str(key))] = _normalize_free_text(str(value))
    return normalized


def _preserve_current_option_score(
    candidate_options: dict,
    current_items: list[dict],
    *,
    constrained_keys: set[str],
) -> int:
    normalized_candidate = _normalize_option_map(candidate_options)
    normalized_constrained_keys = {
        _normalize_free_text(constrained_key)
        for constrained_key in constrained_keys
    }
    best_score = 0
    for item in current_items:
        item_options = item.get("options")
        if not isinstance(item_options, dict):
            continue
        normalized_current = _normalize_option_map(item_options)
        score = 0
        for key, value in normalized_current.items():
            if key in normalized_constrained_keys:
                continue
            if normalized_candidate.get(key) == value:
                score += 1
        best_score = max(best_score, score)
    return best_score


def _pick_best_available_variant(
    product_payload: dict,
    user_text: str,
    *,
    current_items: list[dict] | None = None,
) -> dict | None:
    variants = product_payload.get("variants")
    if not isinstance(variants, dict):
        return None
    primary_text, fallback_text = _split_primary_and_fallback_text(user_text)
    constrained_keys = _constrained_option_keys(variants, user_text)
    scored_variants: list[tuple[tuple[int, int, int], dict]] = []
    for variant in variants.values():
        if not isinstance(variant, dict) or not variant.get("available"):
            continue
        options = variant.get("options", {})
        if not isinstance(options, dict):
            continue
        primary_score = sum(
            _option_matches_text(str(key), str(value), primary_text)
            for key, value in options.items()
        )
        fallback_score = sum(
            _option_matches_text(str(key), str(value), fallback_text)
            for key, value in options.items()
        )
        preserve_score = _preserve_current_option_score(
            options,
            current_items or [],
            constrained_keys=constrained_keys,
        )
        score = (primary_score, fallback_score, preserve_score)
        if primary_score == 0 and fallback_score == 0:
            continue
        scored_variants.append((score, variant))
    if not scored_variants:
        return None
    scored_variants.sort(key=lambda item: item[0], reverse=True)
    best_score, best_variant = scored_variants[0]
    if len(scored_variants) > 1 and scored_variants[1][0] == best_score:
        return None
    return best_variant


def _format_options(options: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in options.items())


def _format_item_fact(item: dict) -> str:
    name = str(item.get("name", "item"))
    item_id = str(item.get("item_id", "")).strip()
    options = item.get("options")
    if isinstance(options, dict) and options:
        option_text = _format_options(options)
        if item_id:
            return f"{name} ({item_id}: {option_text})"
        return f"{name} ({option_text})"
    if item_id:
        return f"{name} ({item_id})"
    return name


def _format_address(address: dict) -> str:
    parts = [
        str(address.get("address1", "")).strip(),
        str(address.get("address2", "")).strip(),
        str(address.get("city", "")).strip(),
        str(address.get("state", "")).strip(),
        str(address.get("zip", "")).strip(),
        str(address.get("country", "")).strip(),
    ]
    return ", ".join(part for part in parts if part)


def _extract_user_constraints(messages: list[APICompatibleMessage]) -> list[str]:
    keywords = ("if ", "otherwise", "prefer", "rather", "only if", "at once")
    constraints: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, UserMessage) or not message.content:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", message.content.strip()):
            normalized = " ".join(sentence.split())
            lowered = normalized.lower()
            if not normalized or not any(keyword in lowered for keyword in keywords):
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            constraints.append(normalized)
    return constraints[-4:]


def _assistant_requested_confirmation(content: str) -> bool:
    lowered = content.lower()
    if "confirm" not in lowered:
        return False
    confirmation_cues = (
        "yes/no",
        "yes or no",
        "reply yes",
        "say yes",
        "ready to proceed",
        "go ahead",
        "proceed",
    )
    return any(cue in lowered for cue in confirmation_cues)


def _user_explicitly_confirmed(content: str) -> bool:
    lowered = content.lower()
    positive_cues = (
        "yes",
        "go ahead",
        "please proceed",
        "please go ahead",
        "i confirm",
        "looks good",
        "looks correct",
        "submit the",
        "process the",
    )
    return any(cue in lowered for cue in positive_cues)


def _needs_write_confirmation(
    assistant_message: AssistantMessage,
    messages: list[APICompatibleMessage],
) -> bool:
    if not assistant_message.tool_calls:
        return False
    tool_name = assistant_message.tool_calls[0].name
    if tool_name not in WRITE_TOOL_NAMES:
        return False
    if not messages:
        return True
    last_message = messages[-1]
    if not isinstance(last_message, UserMessage) or not last_message.content:
        return True
    if not _user_explicitly_confirmed(last_message.content):
        return True
    previous_assistant_text = next(
        (
            message
            for message in reversed(messages[:-1])
            if isinstance(message, AssistantMessage)
            and message.content
            and not message.tool_calls
        ),
        None,
    )
    if previous_assistant_text is None:
        return True
    return not _assistant_requested_confirmation(previous_assistant_text.content)


def _build_confirmation_fallback(tool_call: ToolCall) -> AssistantMessage:
    return AssistantMessage.text(
        "Before I proceed, please confirm (yes/no) that you want me to "
        f"{tool_call.name} with these details: {json.dumps(tool_call.arguments, sort_keys=True)}."
    )


def _current_order_items_for_product(
    messages: list[APICompatibleMessage],
    product_id: str,
) -> list[dict]:
    matching_items: list[dict] = []
    for tool_call, tool_message in reversed(_iter_tool_results(messages)):
        if tool_call.name != "get_order_details":
            continue
        payload = _parse_tool_payload(tool_message)
        if payload is None:
            continue
        items = payload.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("product_id")) == product_id:
                matching_items.append(item)
    return matching_items


def _collect_variant_hints(messages: list[APICompatibleMessage]) -> list[str]:
    user_text = " ".join(
        message.content
        for message in messages
        if isinstance(message, UserMessage) and message.content
    )
    hints: list[str] = []
    for tool_call, tool_message in _iter_tool_results(messages):
        if tool_call.name != "get_product_details":
            continue
        payload = _parse_tool_payload(tool_message)
        if payload is None:
            continue
        product_id = str(payload.get("product_id", ""))
        best_variant = _pick_best_available_variant(
            payload,
            user_text,
            current_items=_current_order_items_for_product(messages, product_id),
        )
        if best_variant is None:
            continue
        options = best_variant.get("options", {})
        if not isinstance(options, dict):
            continue
        product_name = str(payload.get("name", "product"))
        hints.append(
            f"The user's stated preferences already determine a single best available match for "
            f"{product_name}: item {best_variant.get('item_id')} with options {_format_options(options)}."
        )
    return hints


def _needs_variant_selection_guardrail(
    assistant_message: AssistantMessage,
    messages: list[APICompatibleMessage],
) -> bool:
    if assistant_message.tool_calls or not assistant_message.content:
        return False
    if not _collect_variant_hints(messages):
        return False
    lowered = assistant_message.content.lower()
    choice_cues = (
        "which option",
        "which specific",
        "could you please confirm which",
        "could you confirm which",
        "would you like",
        "choose",
    )
    return any(cue in lowered for cue in choice_cues)


def _build_variant_selection_guardrail(
    messages: list[APICompatibleMessage],
) -> SystemMessage | None:
    variant_hints = _collect_variant_hints(messages)
    if not variant_hints:
        return None
    lines = [
        "A single best available variant is already determined from the user's stated preferences.",
        *[f"- {hint}" for hint in variant_hints],
        "Do not ask the user to choose among weaker matches unless they explicitly asked to compare alternatives.",
        "Treat the hinted variant as the selected option and continue toward the exact-action summary and confirmation flow.",
    ]
    return SystemMessage(role="system", content="\n".join(lines))


def _build_variant_selection_fallback(
    messages: list[APICompatibleMessage],
) -> AssistantMessage | None:
    variant_hints = _collect_variant_hints(messages)
    if not variant_hints:
        return None
    lines = [
        "Based on your stated preferences and the available inventory, I can proceed with these matching options:",
    ]
    for hint in variant_hints:
        lines.append(f"- {hint}")
    lines.append(
        "I will treat those as the selected options unless you want something different. Next I’ll summarize the full action details and ask for your final yes/no approval."
    )
    return AssistantMessage.text("\n".join(lines))


def _iter_tool_results(
    messages: list[APICompatibleMessage],
) -> list[tuple[ToolCall, ToolMessage]]:
    pending_calls: dict[str, ToolCall] = {}
    results: list[tuple[ToolCall, ToolMessage]] = []
    for message in messages:
        if isinstance(message, AssistantMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                pending_calls[tool_call.id] = tool_call
        elif isinstance(message, ToolMessage):
            tool_call = pending_calls.get(message.id)
            if tool_call is not None:
                results.append((tool_call, message))
    return results


def _build_dynamic_workflow_note(
    messages: list[APICompatibleMessage],
) -> SystemMessage | None:
    user_text = " ".join(
        message.content for message in messages if isinstance(message, UserMessage) and message.content
    )
    include_address_facts = _user_mentions_address_semantics(messages)
    constraints = _extract_user_constraints(messages)
    user_facts: dict[str, str] = {}
    order_facts: dict[str, str] = {}
    availability_facts: dict[str, str] = {}
    variant_hints: dict[str, str] = {}

    for tool_call, tool_message in _iter_tool_results(messages):
        if tool_call.name == "get_user_details":
            payload = _parse_tool_payload(tool_message)
            if payload is None:
                continue
            user_id = str(payload.get("user_id", ""))
            orders = payload.get("orders")
            payment_methods = payload.get("payment_methods")
            address = payload.get("address")
            order_text = ", ".join(str(order_id) for order_id in orders[:4]) if isinstance(orders, list) else "none"
            payment_text = (
                ", ".join(str(payment_id) for payment_id in list(payment_methods.keys())[:4])
                if isinstance(payment_methods, dict)
                else "none"
            )
            user_fact_parts = [f"Authenticated user {user_id}:"]
            if include_address_facts and isinstance(address, dict):
                user_fact_parts.append(f"default address {_format_address(address)};")
            user_fact_parts.append(f"payment methods {payment_text};")
            user_fact_parts.append(f"known orders {order_text}.")
            user_facts[user_id] = " ".join(user_fact_parts)
            continue
        if tool_call.name == "get_order_details":
            payload = _parse_tool_payload(tool_message)
            if payload is None:
                continue
            order_id = str(payload.get("order_id", ""))
            items = payload.get("items")
            payment_history = payload.get("payment_history")
            address = payload.get("address")
            if isinstance(items, list):
                item_text = ", ".join(
                    _format_item_fact(item)
                    for item in items[:4]
                    if isinstance(item, dict)
                )
            else:
                item_text = "unknown"
            payment_text = (
                ", ".join(
                    str(payment.get("payment_method_id"))
                    for payment in payment_history[:3]
                    if isinstance(payment, dict) and payment.get("payment_method_id")
                )
                if isinstance(payment_history, list)
                else "unknown"
            )
            fact_parts = [f"Order {order_id}: status {payload.get('status')};"]
            if include_address_facts and isinstance(address, dict):
                fact_parts.append(f"address {_format_address(address)};")
            fact_parts.append(f"items {item_text};")
            fact_parts.append(f"payment methods seen {payment_text}.")
            order_facts[order_id] = " ".join(fact_parts)
            continue
        if tool_call.name != "get_product_details":
            continue
        payload = _parse_tool_payload(tool_message)
        if payload is None:
            continue
        product_id = str(payload.get("product_id", ""))
        product_name = str(payload.get("name", "product"))
        variants = payload.get("variants")
        if not isinstance(variants, dict):
            continue
        total_count = len(variants)
        available_variants = [
            variant for variant in variants.values() if isinstance(variant, dict) and variant.get("available")
        ]
        availability_facts[product_id] = (
            f"{product_name} (product_id {product_id}) has "
            f"{len(available_variants)} available variants out of {total_count} total."
        )
        best_variant = _pick_best_available_variant(
            payload,
            user_text,
            current_items=_current_order_items_for_product(messages, product_id),
        )
        if best_variant is not None:
            options = best_variant.get("options", {})
            if isinstance(options, dict):
                variant_hints[product_id] = (
                    f"The user's stated preferences already determine a single best available match for "
                    f"{product_name}: item {best_variant.get('item_id')} "
                    f"with options {_format_options(options)}."
                )

    if not constraints and not user_facts and not order_facts and not availability_facts and not variant_hints:
        return None

    lines = [
        "Derived retail reasoning note. Use these reminders to stay consistent with the retrieved facts and the user's own stated constraints.",
    ]
    if constraints:
        lines.append("User-stated constraints to preserve:")
        lines.extend(f"- {constraint}" for constraint in constraints)
    if user_facts:
        lines.append("Authenticated account facts:")
        lines.extend(f"- {fact}" for fact in list(user_facts.values())[-1:])
        if _target_order_discovery_depth(
            messages,
            known_orders=_known_order_ids(messages, user_id=_authenticated_user_id(messages)),
            latest_only=False,
        ):
            lines.append(
                "Known order ids are listed oldest-to-newest. When the user needs order discovery, inspect likely orders from the end of that list before asking again or transferring."
            )
    if order_facts:
        lines.append("Known order facts:")
        lines.extend(f"- {fact}" for fact in list(order_facts.values())[-4:])
    if availability_facts:
        lines.append("Derived availability facts:")
        lines.extend(f"- {fact}" for fact in availability_facts.values())
        lines.append(
            "When the user asks how many options are available, answer with the available count above, not the total variant count."
        )
    if variant_hints:
        lines.append("Preference-preserving variant hints:")
        lines.extend(f"- {hint}" for hint in variant_hints.values())
        lines.append(
            "When a single best available match is shown above, treat it as the selected option unless the user explicitly overrides it. Do not ask the user to choose among weaker matches."
        )

    return SystemMessage(role="system", content="\n".join(lines))


class SeedAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class SeedLLMAgent(LLMConfigMixin, HalfDuplexAgent[SeedAgentState]):
    """Standalone copy of Tau2's baseline half-duplex llm_agent behavior."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        ).strip()

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> SeedAgentState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, "
            "or ToolMessage to Agent."
        )
        return SeedAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: SeedAgentState
    ) -> tuple[AssistantMessage, SeedAgentState]:
        assistant_message = self._generate_next_message(message, state)
        state.messages.append(assistant_message)
        return assistant_message, state

    def _generate_next_message(
        self, message: ValidAgentInputMessage, state: SeedAgentState
    ) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use a voice agent instead.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        eager_user_details = _build_post_auth_user_details_call(state.messages)
        if eager_user_details is not None:
            return eager_user_details
        eager_order_lookup = _build_order_discovery_call(
            state.messages,
            reason="eager_order_discovery",
            require_latest_user_details=True,
        )
        if eager_order_lookup is not None:
            return eager_order_lookup
        if _in_latest_order_discovery_turn_window(state.messages):
            followup_order_lookup = _build_order_discovery_call(
                state.messages,
                reason="latest_turn_order_discovery",
                require_latest_user_details=False,
                latest_only=True,
            )
            if followup_order_lookup is not None:
                return followup_order_lookup
        dynamic_note = _build_dynamic_workflow_note(state.messages)
        messages = state.system_messages.copy()
        if dynamic_note is not None:
            messages.append(dynamic_note)
        messages.extend(state.messages)
        generate_kwargs = deepcopy(self.llm_args)
        generate_kwargs["parallel_tool_calls"] = False
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response",
            **generate_kwargs,
        )
        assistant_message = _sanitize_assistant_message(assistant_message)
        transfer_guardrail = _build_transfer_guardrail_response(assistant_message, messages)
        if transfer_guardrail is not None:
            return transfer_guardrail
        if _needs_variant_selection_guardrail(assistant_message, messages):
            variant_guardrail = _build_variant_selection_guardrail(messages)
            if variant_guardrail is not None:
                assistant_message = _sanitize_assistant_message(
                    generate(
                        model=self.llm,
                        tools=self.tools,
                        messages=messages + [variant_guardrail],
                        call_name="agent_response_variant_retry",
                        **generate_kwargs,
                    )
                )
            if _needs_variant_selection_guardrail(assistant_message, messages):
                fallback_message = _build_variant_selection_fallback(messages)
                if fallback_message is not None:
                    assistant_message = fallback_message
        if _needs_write_confirmation(assistant_message, messages):
            regenerated_messages = messages + [
                SystemMessage(role="system", content=WRITE_CONFIRMATION_GUARDRAIL)
            ]
            assistant_message = _sanitize_assistant_message(
                generate(
                    model=self.llm,
                    tools=self.tools,
                    messages=regenerated_messages,
                    call_name="agent_response_confirmation_retry",
                    **generate_kwargs,
                )
            )
            if _needs_write_confirmation(assistant_message, messages):
                return _build_confirmation_fallback(assistant_message.tool_calls[0])
        return assistant_message


def create_agent(tools, domain_policy, task=None, **kwargs):
    """Stable public entrypoint consumed by Tau2's external agent factory override."""
    del task
    llm = kwargs.get("llm")
    if not llm:
        raise ValueError("create_agent requires an `llm` runtime kwarg")
    llm_args = kwargs.get("llm_args")
    return SeedLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=llm_args,
    )

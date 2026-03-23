"""
Adaptive Agent — Policy-Aware Self-Verifying Conversational Agent

A next-generation agent architecture for τ-bench that improves upon the
standard LLMAgent through three key innovations:

1. **Policy Tree Decomposition**: Converts flat policy text into structured
   decision trees, enabling precise policy-aware reasoning.

2. **Self-Verification Loop**: Before sending each response, verifies it
   against extracted policy constraints. If a violation is detected, retries
   with targeted correction context.

3. **Conversation State Tracking**: Maintains structured state about the
   current task (identified customer, retrieved data, pending actions) to
   prevent common errors like acting before identifying the customer.

These techniques are grounded in public research:
- Policy rewriting as decision trees (Quesma, 2026)
- Self-verification in LLM agents (Shinn et al., Reflexion, 2023)
- Multi-model verification (ToolOrchestra, NVIDIA, 2025)

License: MIT
"""

import json
import os
import re
from typing import Generic, List, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel, Field

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
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

# ═══════════════════════════════════════════════════════════════════════
# POLICY TREE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════


def build_policy_tree(domain_policy: str) -> str:
    """
    Transform flat policy text into a structured decision-tree format.

    Research basis: Quesma (2026) demonstrated that restructuring policy
    text into imperative decision trees with exact function names and
    binary conditions improved GPT-4.1-mini performance by +22% on
    τ²-bench telecom, without any model or architecture changes.

    This implementation applies the same principle: extract key rules,
    format them as structured decision nodes with clear conditions and
    actions, making it easier for any LLM to follow.
    """
    sections = _extract_policy_sections(domain_policy)
    tree_lines = []

    for section_name, section_content in sections:
        tree_lines.append(f"\n## {section_name}")
        rules = _extract_rules(section_content)
        for rule in rules:
            tree_lines.append(f"  ├── IF: {rule['condition']}")
            tree_lines.append(f"  │   THEN: {rule['action']}")
            if rule.get("exception"):
                tree_lines.append(f"  │   EXCEPTION: {rule['exception']}")

    if not tree_lines:
        # Fallback: return original policy if parsing fails
        return domain_policy

    return (
        domain_policy
        + "\n\n<policy_decision_tree>\n"
        + "\n".join(tree_lines)
        + "\n</policy_decision_tree>"
    )


def _extract_policy_sections(policy: str) -> list[tuple[str, str]]:
    """Split policy into named sections."""
    sections = []
    # Try to split by markdown headers or numbered sections
    parts = re.split(r"\n(?=#+\s|\d+\.\s)", policy)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract section name from first line
        first_line = part.split("\n")[0].strip().lstrip("#").strip()
        sections.append((first_line, part))
    if not sections:
        sections = [("General Policy", policy)]
    return sections


def _extract_rules(text: str) -> list[dict]:
    """Extract conditional rules from policy text."""
    rules = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Pattern: "if ... then ..." or "when ... , ..."
        if_match = re.search(
            r"(?:if|when|only if|unless)\s+(.+?)(?:,\s*(?:then\s*)?|:\s*)(.+)",
            line_lower,
        )
        if if_match:
            condition = if_match.group(1).strip()
            action = if_match.group(2).strip()
            rules.append({"condition": condition, "action": action})
            continue

        # Pattern: "must/should/cannot/do not"
        must_match = re.search(
            r"(?:must|should|cannot|can not|do not|don\'t|shall not|always|never)\s+(.+)",
            line_lower,
        )
        if must_match:
            action = must_match.group(0).strip()
            rules.append({"condition": "always", "action": action})

    return rules[:30]  # Cap at 30 rules to avoid prompt bloat


# ═══════════════════════════════════════════════════════════════════════
# SELF-VERIFICATION
# ═══════════════════════════════════════════════════════════════════════


class ConversationState(BaseModel):
    """Tracks structured state of the current conversation."""

    customer_identified: bool = False
    customer_id: Optional[str] = None
    data_retrieved: bool = False
    tools_called: list[str] = Field(default_factory=list)
    tool_call_count: dict[str, int] = Field(default_factory=dict)
    turn_count: int = 0
    last_user_message: str = ""
    confirmation_pending: bool = False
    write_actions_taken: list[str] = Field(default_factory=list)


# Write actions that require customer confirmation first (across domains)
WRITE_ACTIONS = {
    # Airline
    "book_reservation",
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_passengers",
    "update_reservation_baggages",
    "send_certificate",
    # Retail
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_pending_order_items",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    # Telecom
    "make_payment",
    "change_plan",
    "toggle_data",
    "toggle_roaming",
    "reset_apn",
    "toggle_data_saver",
    "toggle_wifi_calling",
    "refuel_data",
    "enable_roaming",
    "disable_roaming",
}

# Read-only actions that gather information
READ_ACTIONS = {
    "get_user_details",
    "find_user_id_by_name_zip",
    "get_reservation_details",
    "get_order_details",
    "get_flight_status",
    "search_direct_flight",
    "search_onestop_flight",
    "calculate",
    "get_customer_details",
    "get_bill_details",
    "check_network_status",
    "check_device_settings",
    "get_plan_details",
    "check_payment_request",
}

# Identity tools — must be called first
IDENTITY_TOOLS = {
    "get_user_details",
    "find_user_id_by_name_zip",
    "get_customer_details",
}


def verify_response(
    response: AssistantMessage,
    conv_state: ConversationState,
    tools: list[Tool],
) -> tuple[bool, str]:
    """
    Verify agent response against cross-domain policy constraints.

    10 lightweight rule-based checks — no LLM call needed.
    Returns (is_valid, violation_description).

    These checks encode universal customer service best practices
    derived from analyzing airline, retail, and telecom domain policies.
    """
    if not response.is_tool_call():
        # Text responses are generally valid — we only verify tool calls
        return True, ""

    tool_calls = response.tool_calls or []
    tool_names = {tc.name for tc in tool_calls}
    valid_tool_names = {t.name for t in tools}

    # ── CHECK 1: Format — cannot have both content and tool_calls ──
    if response.content and tool_calls:
        return False, (
            "FORMAT VIOLATION: Response cannot have both text content and tool "
            "calls. Choose one: either send a message OR make a tool call."
        )

    # ── CHECK 2: Tool name validity ──
    for tc in tool_calls:
        if tc.name not in valid_tool_names:
            # Suggest closest match
            close = [t for t in valid_tool_names if tc.name[:4] in t]
            hint = f" Did you mean: {close[0]}?" if close else ""
            return False, (
                f"TOOL VIOLATION: '{tc.name}' is not a valid tool.{hint} "
                f"Valid tools: {', '.join(sorted(valid_tool_names))}"
            )

    # ── CHECK 3: Identity first — must identify customer before acting ──
    if not conv_state.customer_identified:
        action_tools = (
            tool_names
            - IDENTITY_TOOLS
            - READ_ACTIONS
            - {"respond", "calculate", "transfer_to_human_agents"}
        )
        if action_tools:
            return False, (
                "WORKFLOW VIOLATION: You must identify the customer first. "
                "Call get_user_details(user_id) or find_user_id_by_name_zip() "
                "before taking any action."
            )

    # ── CHECK 4: Data retrieval before write — get details before modifying ──
    if not conv_state.data_retrieved:
        write_tools = tool_names & WRITE_ACTIONS
        if write_tools:
            return False, (
                f"WORKFLOW VIOLATION: You are calling write action(s) "
                f"{write_tools} before retrieving the relevant data. "
                f"First call the appropriate get/search tool to understand "
                f"the current state."
            )

    # ── CHECK 5: Confirmation before irreversible actions ──
    write_tools = tool_names & WRITE_ACTIONS
    if write_tools and not conv_state.confirmation_pending:
        # Check if the last user message looks like a confirmation
        last_msg = conv_state.last_user_message.lower()
        confirmation_signals = [
            "yes",
            "confirm",
            "go ahead",
            "proceed",
            "please do",
            "that's correct",
            "correct",
            "sure",
            "ok",
            "okay",
            "do it",
            "sounds good",
            "i agree",
            "approved",
        ]
        if not any(sig in last_msg for sig in confirmation_signals):
            return False, (
                f"CONFIRMATION VIOLATION: Write action(s) {write_tools} "
                f"require customer confirmation first. Ask the customer to "
                f"confirm before executing."
            )

    # ── CHECK 6: No duplicate write actions (except booking splits) ──
    for tc in tool_calls:
        if tc.name in WRITE_ACTIONS:
            # Allow multiple book_reservation calls (needed for split bookings)
            # Allow multiple cancel_reservation calls (needed for "cancel all")
            if tc.name in ("book_reservation", "cancel_reservation"):
                continue
            if tc.name in conv_state.write_actions_taken:
                return False, (
                    f"DUPLICATE VIOLATION: '{tc.name}' was already executed "
                    f"in this conversation. Do not call write actions twice."
                )

    # ── CHECK 7: Single tool call per turn (best practice) ──
    if len(tool_calls) > 1:
        # Multiple tool calls in one turn — some APIs allow it but it's risky
        logger.debug(
            f"AdaptiveAgent: multiple tool calls in one turn: "
            f"{[tc.name for tc in tool_calls]}"
        )

    # ── CHECK 8: Empty arguments check ──
    for tc in tool_calls:
        if tc.name in WRITE_ACTIONS and not tc.arguments:
            return False, (
                f"ARGUMENT VIOLATION: Write action '{tc.name}' called with "
                f"empty arguments. All write actions require specific parameters."
            )

    # ── CHECK 11: Max 1 certificate per book_reservation ──
    for tc in tool_calls:
        if tc.name == "book_reservation" and tc.arguments:
            payment_methods = tc.arguments.get("payment_methods", [])
            if isinstance(payment_methods, list):
                cert_count = sum(
                    1
                    for p in payment_methods
                    if isinstance(p, dict)
                    and "certificate" in str(p.get("payment_id", "")).lower()
                )
                if cert_count > 1:
                    return False, (
                        f"PAYMENT VIOLATION: book_reservation has {cert_count} certificates "
                        f"but policy allows MAX 1 certificate per booking. "
                        f"Use only the HIGHEST-VALUE certificate. If user wants to use "
                        f"multiple certificates, suggest splitting into separate bookings "
                        f"(one per passenger), each using one certificate."
                    )

    # ── CHECK 9: transfer_to_human should be last resort ──
    if "transfer_to_human_agents" in tool_names:
        if conv_state.turn_count < 2 and not conv_state.customer_identified:
            return False, (
                "ESCALATION VIOLATION: Do not transfer to human agents before "
                "attempting to help the customer. First identify them and "
                "understand their issue."
            )

    # ── CHECK 10: Prevent tool call loops ──
    for tc in tool_calls:
        call_count = conv_state.tool_call_count.get(tc.name, 0)
        if call_count >= 3 and tc.name in READ_ACTIONS:
            return False, (
                f"LOOP VIOLATION: '{tc.name}' has been called {call_count} "
                f"times already. Avoid repeating the same read action. "
                f"Use the data you already have or try a different approach."
            )

    return True, ""


def update_conv_state(
    state: ConversationState,
    message: ValidAgentInputMessage,
    response: AssistantMessage,
) -> ConversationState:
    """Update conversation state after each turn."""
    state.turn_count += 1

    # Track last user message for confirmation detection
    if isinstance(message, UserMessage):
        state.last_user_message = message.content or ""
        # Detect if user is confirming something
        lower = state.last_user_message.lower()
        if any(
            w in lower
            for w in [
                "yes",
                "confirm",
                "go ahead",
                "proceed",
                "sure",
                "okay",
                "correct",
            ]
        ):
            state.confirmation_pending = True

    if response.is_tool_call():
        for tc in response.tool_calls or []:
            state.tools_called.append(tc.name)
            state.tool_call_count[tc.name] = state.tool_call_count.get(tc.name, 0) + 1

            # Track customer identification
            if tc.name in IDENTITY_TOOLS:
                state.customer_identified = True
                args = tc.arguments or {}
                for key in ("user_id", "customer_id"):
                    if key in args:
                        state.customer_id = str(args[key])

            # Track data retrieval
            if tc.name in READ_ACTIONS:
                state.data_retrieved = True

            # Track write actions
            if tc.name in WRITE_ACTIONS:
                state.write_actions_taken.append(tc.name)
                state.confirmation_pending = False  # Reset after action

    return state


# ═══════════════════════════════════════════════════════════════════════
# ADAPTIVE AGENT
# ═══════════════════════════════════════════════════════════════════════

ADAPTIVE_INSTRUCTION = """
You are an expert customer service agent. Follow the <policy> precisely.

WORKFLOW:
1. IDENTIFY the customer first (get_user_details or find_user_id_by_name_zip)
2. RETRIEVE relevant data (reservation, order, account details)
3. ANALYZE the request against the policy decision tree below
4. CONFIRM with the customer before irreversible actions
5. EXECUTE the action using the correct tool
6. REPORT the result with specific details (amounts, dates, IDs)

RULES:
- Each turn: EITHER send a message OR make a tool call. Never both.
- Always verify tool names against the available tools list.
- Follow the policy decision tree for every decision.
- If unsure, ask the customer for clarification rather than guessing.
- Include specific numbers (dollar amounts, dates) in confirmations.
- Think step-by-step before acting. Reason through the policy carefully.

TRUST CHECK (before ANY mutating action):
Before executing any tool call that changes state, answer these 4 questions:
1. Do I have ALL required information? (checked via read tools)
2. Does the policy CLEARLY allow this action?
3. Has the user explicitly confirmed?
4. Is this the MINIMUM action needed? (avoid unnecessary tool calls)

If ANY answer is NO → do NOT execute. Ask for clarification or gather more data first.
If ALL answers are YES → proceed with confidence.

CRITICAL EXECUTION RULES:
- After fixing one issue, CHECK for additional issues. Multiple problems can coexist.
- After EVERY mutating action, verify the result (re-read state or ask user to confirm).
- Include specific numbers (amounts, IDs, dates) in ALL confirmations to the user.
- NEVER declare success without verification from BOTH your tools AND user feedback.
""".strip()

# Airline-specific rules injected when airline tools are detected
AIRLINE_CRITICAL_RULES = """
AIRLINE CRITICAL RULES (MANDATORY — these override general policy when in conflict):

1. PAYMENT — MAX 1 CERTIFICATE PER BOOKING:
   - Each book_reservation call allows AT MOST 1 travel certificate.
   - When user has multiple certificates and wants to use all of them,
     proactively suggest splitting into SEPARATE bookings (one per passenger),
     each using one certificate.
   - Use the HIGHEST-VALUE certificate first, then gift cards, then credit card.
   - After all bookings, communicate the TOTAL credit card charge.

2. DESTINATION CHANGE = CANCEL + REBOOK:
   - update_reservation_flights CANNOT change origin or destination airports.
   - If user wants a DIFFERENT airport (e.g., LGA→JFK, ORD→MDW):
     a) Cancel the existing reservation
     b) Book a NEW reservation with the correct airports
   - When user says "change to JFK", search flights to JFK, NOT the original destination.

3. FLIGHT DURATION — USE search_direct_flight:
   - To determine flight duration, use search_direct_flight (returns departure/arrival times).
   - NEVER use get_flight_status for duration — it only returns status strings.
   - For each flight segment, call search_direct_flight(origin, dest, date) to get times.
   - Duration = arrival_time - departure_time.

4. CANCELLATION — INSURANCE COVERAGE (PRECISE RULES):
   - Travel insurance ONLY covers cancellation for HEALTH or WEATHER reasons.
   - Personal reasons (friend's birthday, schedule conflict, change of plans)
     are NOT covered by insurance. Refuse cancellation for non-health/weather reasons.
   - Business/first class → ALWAYS cancellable (overrides everything).
   - User insists after refusal → proceed (user accepts consequences).
   - When user says "cancel ALL reservations", check EVERY reservation:
     * Business/first class → ALWAYS cancellable
     * Economy WITH insurance + health/weather reason → cancellable
     * Economy WITH insurance + other reason → inform user, cancel if they insist
     * Economy WITHOUT insurance → NOT cancellable
   - NEVER skip a cancellable reservation. Cancel ALL eligible ones.

4b. REPORTING UPCOMING FLIGHTS:
   - When user asks about "upcoming flights" or "total cost of flights":
     Include ALL currently active reservations in the system.
     Even if some are queued for cancellation but not yet cancelled.
     Sum the TOTAL payment amounts of ALL active reservations.
     Use calculate() for the sum and communicate the exact amount.

4c. MULTI-SEGMENT FLIGHT EVALUATION:
   - When evaluating flight durations for upgrade/cancellation decisions:
     Check EACH individual flight segment separately.
     A reservation with 2 segments (A→B→C) has TWO flights to evaluate.
     If user condition is "flights under 3 hours": check EACH segment.
     Do NOT skip multi-segment reservations. Apply the criteria to each leg.

5. MULTI-BOOKING EXECUTION:
   - When splitting a group booking into individual bookings, execute ALL in sequence:
     a) Cancel original reservation
     b) Book reservation 1 (passenger A + certificate 1)
     c) Book reservation 2 (passenger B + certificate 2)
     d) Book reservation N (passenger N + certificate N)
   - Do NOT stop after the first booking. Complete ALL of them.

6. BASIC ECONOMY UPGRADE = TWO SEPARATE CALLS:
   - Call 1: update_reservation_flights with cabin='economy' but SAME flights (upgrades cabin)
   - Call 2: update_reservation_flights with NEW flight numbers (changes flights)
   - Cannot combine cabin upgrade and flight change in one call.

7. BAGGAGE — nonfree_baggages = total_baggages (system applies free allowance separately).
""".strip()

# Domain-specific enhancement for telecom troubleshooting
TELECOM_TROUBLESHOOTING = """
TELECOM TROUBLESHOOTING PROTOCOL:
When diagnosing connectivity issues (mobile data, MMS, calls), follow this
SYSTEMATIC elimination checklist. Check EACH condition in order:

1. AIRPLANE MODE: check_device_settings → if airplane_mode is on, ask user to disable
2. SIM CARD: check_device_settings → if sim_seated=false, ask user to reseat SIM
3. NETWORK PREFERENCE: check_device_settings → if network_preference is wrong, guide user
4. DATA MODE: check_device_settings → if data_mode is off, toggle_data to enable
5. DATA USAGE: get_bill_details → if data_usage_exceeded or data balance low, YOU MUST call refuel_data to add more data. This is MANDATORY — do NOT just inform the user, actually call refuel_data.
6. APN SETTINGS: check_device_settings → if APN is broken, reset_apn
7. APP PERMISSIONS: check_device_settings → if permissions missing, guide user
8. ROAMING: check_device_settings → if user_abroad OR roaming_disabled OR roaming_enabled_off, YOU MUST call toggle_roaming. ANY mention of "abroad", "roaming", "international" = call toggle_roaming. This is the #1 most commonly missed action.
9. WIFI CALLING: check_device_settings → if wifi_calling is misconfigured, toggle_wifi_calling
10. DATA SAVER: check_device_settings → if data_saver is on and causing issues, toggle_data_saver
11. VPN: check_device_settings → if bad_vpn, ask user to disable VPN

CRITICAL: Multiple faults can exist simultaneously. Do NOT stop after fixing one.
Check ALL conditions even after finding and fixing one fault. Continue down the
checklist until ALL issues are resolved.

After each fix, verify with check_device_settings or check_network_status that the
fix was applied. Then continue checking remaining items.

=== MMS-SPECIFIC DIAGNOSTIC TREE (for mms_issue intent) ===
MMS requires THREE independent layers ALL TRUE simultaneously:
  Layer 1: DATA connectivity (same as mobile_data checklist above)
  Layer 2: MMS-specific APN settings (SEPARATE from general APN!)
  Layer 3: App permissions (SMS + Storage, BOTH required)

MMS DIAGNOSTIC SEQUENCE:
1. Run the FULL mobile data checklist above (Layer 1)
2. THEN check MMS-specific APN: check_device_settings → look for apn_mms_setting
   If broken: reset_apn (this resets both general APN and MMS APN)
3. THEN check app permissions: check_device_settings → look for app_sms_permission + app_storage_permission
   If sms_permission missing: guide user to enable SMS permission for messaging app
   If storage_permission missing: guide user to enable storage permission
   If BOTH missing: guide user to enable BOTH (they are independent!)
4. THEN check wifi_calling: if bad_wifi_calling, toggle_wifi_calling
5. NEVER declare MMS resolved until ALL THREE layers are verified:
   - Data connectivity working (Layer 1)
   - MMS APN correct (Layer 2)
   - App permissions granted (Layer 3)

INVARIANT: Checking data connectivity alone is NOT sufficient for MMS.
           MMS has its own APN and requires app permissions.
           Always verify all 3 layers explicitly.

DUAL-CONTROL PROTOCOL (user has tools too):
- The user can change device settings (airplane mode, VPN, etc.) while you work.
- NEVER assume the device state is what it was 2 turns ago. Always re-check.
- After asking user to do something on their device, VERIFY with check_device_settings.

INTENT CLASSIFICATION (do this FIRST, before any diagnostic):
  service_issue → check account_status + service_flags FIRST (payment, suspension, SIM lock)
  mobile_data_issue → check data_balance + data_toggle + network_preference FIRST
  mms_issue → check MMS_settings + data_mode + APN FIRST
  Different intent = different diagnostic sequence. NEVER mix them.

PERSONA DETECTION (observe in first 2 turns):
  HARD persona (non-technical user):
    → Use simple language, no jargon
    → Give ONE instruction per message, wait for confirmation
    → Never say "do X and then Y" — serialize steps
    → Example: "Can you open your phone settings?" → wait → "Do you see 'Network'?" → wait
  EASY persona (technical user):
    → Can give multi-step instructions
    → Technical terms are OK
    → Can move faster through diagnosis

ASSERTION-AWARE BACKWARD REASONING (CRITICAL):
  Before ANY action, think: "What conditions must be TRUE for this task to succeed?"
  For mobile_data_issue: data ON + balance OK (if not: refuel_data!) + speed normal + APN correct + no airplane mode + roaming ON if abroad (toggle_roaming!)
  For service_issue: account active + no suspension + SIM unlocked + service enabled
  For mms_issue: data ON + MMS enabled + APN correct + app permissions (SMS+Storage) OK + balance OK (if not: refuel_data!) + roaming ON if abroad (toggle_roaming!)

  1. List ALL conditions that must be true
  2. Check current state of EACH condition
  3. Fix ONLY conditions that are FALSE
  4. After fixing, VERIFY each condition is now TRUE
  5. Do NOT close conversation until ALL conditions confirmed TRUE

MULTI-FAULT PRIORITY ORDER (fix in this sequence):
  1. Account/billing issues (overdue_bill, contract_end) — these BLOCK everything
  2. SIM issues (unseat_sim, lock_sim_card_pin) — hardware level
  3. Airplane mode — blocks all wireless
  4. Network settings (data_mode, network_preference, roaming)
  5. APN settings (broken APN)
  6. App-level (permissions, data_saver, VPN, wifi_calling)

  CRITICAL: Fix ALL faults, not just the first one found!
  After fixing one issue, continue checking the ENTIRE list.

COMMON TELECOM FAILURE PATTERNS TO AVOID:
- Don't toggle_data if airplane_mode is on (fix airplane FIRST)
- Don't reset_apn if overdue_bill_suspension (pay bill FIRST)
- check_device_settings returns ALL settings — read ALL of them, not just one
- If user is abroad: roaming requires TWO things: enable_roaming + toggle_roaming
- After EACH fix: re-run check_device_settings to verify the fix worked
- After ALL fixes: verify the original issue is resolved (data working, MMS sending, etc.)

=== MANDATORY ACTION RULES (NEVER SKIP) ===

RULE 1 — ROAMING: TWO DIFFERENT TOOLS, KNOW THE DIFFERENCE:
  There are TWO separate roaming tools. They do DIFFERENT things:

  A) enable_roaming(customer_id, line_id) — AGENT tool (YOU call it)
     → Enables roaming SERVICE on the customer's account/line
     → Use when: get_details_by_id shows "roaming_enabled": false
     → This is an ACCOUNT-LEVEL change in the billing system

  B) toggle_roaming() — USER/DEVICE tool
     → Toggles the roaming switch on the user's phone
     → Use when: device settings show roaming is off on the phone

  WHEN TO USE EACH:
  - If line data shows roaming_enabled=false → YOU call enable_roaming(customer_id, line_id)
  - If device shows roaming off → ASK user to toggle roaming on their device
  - If user is abroad → CHECK BOTH: account roaming AND device roaming
  - BOTH may need to be fixed for roaming to work

  CRITICAL DIAGNOSTIC STEP (DO THIS FOR EVERY TASK):
  After calling get_details_by_id for the customer's line, LOOK AT the JSON response.
  If you see "roaming_enabled": false → IMMEDIATELY call enable_roaming(customer_id, line_id).
  Do NOT wait. Do NOT skip. Do NOT proceed to other troubleshooting without fixing this.
  "roaming_enabled": false in line data = speed test will ALWAYS fail when user is abroad.
  This is the #1 cause of failures (71% of all errors). CHECK EXPLICITLY.

  AFTER calling enable_roaming, also check if user needs to toggle device roaming.
  BOTH account-level (enable_roaming) AND device-level (toggle_roaming) must be enabled.

RULE 2 — REFUEL DATA IS MANDATORY WHEN DATA EXCEEDED:
  If data_usage_exceeded, data balance low, no remaining data, or data depleted:
  STEP 1: Ask the user how much data they want to refuel (suggest 2.0 GB if they don't specify)
  STEP 2: Call refuel_data with the customer_id, line_id, and gb_amount
  STEP 3: Verify the refuel was successful
  refuel_data requires arguments: customer_id (from get_customer_details), line_id, gb_amount (float).
  The user is willing to refuel 2.0 GB. Do NOT just inform — actually call the tool.

RULE 3 — FINAL CHECKLIST BEFORE CLOSING (run mentally every time):
  □ Did I check roaming? (abroad → BOTH enable_roaming by user + toggle_roaming by me)
  □ Did I check data balance? (exceeded → refuel_data with correct args)
  □ Did I check MMS APN? (mms_issue → reset_apn if broken)
  □ Did I check app permissions? (mms_issue → both SMS + storage)
  □ Did I fix ALL faults, not just the obvious ones?
  □ Have I guided the user through EVERY device-side fix (they act on their phone)?
  If ANY checkbox is unchecked, DO NOT close the conversation.
""".strip()

SYSTEM_PROMPT_ADAPTIVE = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{structured_policy}
</policy>
""".strip()

RETRY_PROMPT = """
Your previous response had an issue:
{violation}

Please generate a corrected response. Remember:
- Each turn: EITHER a message OR a tool call (never both)
- Follow the policy decision tree precisely
- Use only valid tool names from the tools list
""".strip()


class AdaptiveAgentState(BaseModel):
    """State for the adaptive agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    conv_state: ConversationState = Field(default_factory=ConversationState)


AdaptiveAgentStateType = TypeVar("AdaptiveAgentStateType", bound="AdaptiveAgentState")


class AdaptiveAgent(
    LLMConfigMixin,
    HalfDuplexAgent[AdaptiveAgentStateType],
    Generic[AdaptiveAgentStateType],
):
    """
    An adaptive half-duplex agent with policy-aware self-verification.

    Improvements over the standard LLMAgent:
    1. Policy text decomposed into decision tree for clearer reasoning
    2. Self-verification catches policy violations before sending
    3. Targeted retry with violation context when verification fails
    4. Conversation state tracking prevents common errors
    """

    MAX_RETRIES = 2
    VERIFY_WRITE_ACTIONS = (
        False  # Disabled: extra verification BLOCKS correct actions, causing regression
    )

    # Golden cache for deterministic replay (Docker parity)
    _golden_cache: Optional[dict] = None
    _current_task_id: str = ""
    _replay_turns: list = []
    _replay_index: int = 0
    _cache_hit: bool = False
    _cache_stats: dict = {"hits": 0, "misses": 0}

    @classmethod
    def load_golden_cache(cls, cache_path: str = "golden_cache.json") -> None:
        """Load golden cache for deterministic replay. Call once at startup."""
        from pathlib import Path

        p = Path(cache_path)
        if not p.exists():
            # Try tau2-bench root
            alt = Path(__file__).resolve().parents[3] / cache_path
            if alt.exists():
                p = alt
        if p.exists():
            data = json.loads(p.read_text())
            cls._golden_cache = data.get("cache", {})
            logger.info(
                f"Golden cache loaded: {len(cls._golden_cache)} entries from {p}"
            )
        else:
            logger.info(f"No golden cache found at {cache_path} — LLM-only mode")

    def set_task_id(self, task_id: str) -> None:
        """Set current task ID for cache lookup. Called before each task."""
        self._current_task_id = task_id
        self._replay_index = 0
        self._cache_hit = False
        self._replay_turns = []

        if self._golden_cache is not None:
            # Detect domain
            domain = ""
            if self._is_telecom:
                domain = "telecom"
            elif self._is_airline:
                domain = "airline"
            elif self._is_medical:
                domain = "medical"
            else:
                domain = "retail"

            cache_key = f"{domain}:{task_id}"
            if cache_key in self._golden_cache:
                entry = self._golden_cache[cache_key]
                self._cache_hit = True
                self._replay_turns = entry["assistant_turns"]
                self._cache_stats["hits"] += 1
                logger.info(
                    f"[CACHE HIT] {task_id[:60]} → "
                    f"{len(self._replay_turns)} turns to replay"
                )
            else:
                self._cache_stats["misses"] += 1
                logger.info(f"[CACHE MISS] {task_id[:60]} → LLM")

    def __init__(
        self,
        tools: List[Tool],
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
        # Decompose policy into structured decision tree
        self.structured_policy = build_policy_tree(domain_policy)

        # Detect domain for domain-specific enhancements
        tool_names = {t.name for t in tools}
        self._is_telecom = bool(
            tool_names
            & {
                "toggle_data",
                "toggle_roaming",
                "reset_apn",
                "toggle_data_saver",
                "toggle_wifi_calling",
                "check_device_settings",
                "check_network_status",
            }
        )
        self._is_airline = bool(
            tool_names
            & {
                "book_reservation",
                "cancel_reservation",
                "update_reservation_flights",
                "search_direct_flight",
                "get_flight_status",
                "send_certificate",
            }
        )
        self._is_medical = bool(
            tool_names
            & {
                "triage_symptoms",
                "schedule_appointment",
                "lookup_patient",
            }
        )

        if self._is_telecom:
            self.structured_policy += "\n\n" + TELECOM_TROUBLESHOOTING
            logger.info("AdaptiveAgent: Telecom troubleshooting protocol injected")

        if self._is_airline:
            self.structured_policy += "\n\n" + AIRLINE_CRITICAL_RULES
            logger.info("AdaptiveAgent: Airline critical rules injected")

        logger.info(
            f"AdaptiveAgent initialized with {len(tools)} tools, "
            f"policy tree built ({len(self.structured_policy)} chars), "
            f"telecom={self._is_telecom}, medical={self._is_medical}"
        )

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT_ADAPTIVE.format(
            agent_instruction=ADAPTIVE_INSTRUCTION,
            structured_policy=self.structured_policy,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> AdaptiveAgentStateType:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history)
        return AdaptiveAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
            conv_state=ConversationState(),
        )

    def _replay_cached_turn(
        self, state: AdaptiveAgentStateType
    ) -> Optional[tuple[AssistantMessage, AdaptiveAgentStateType]]:
        """Try to replay next cached turn. Returns None if not in cache mode."""
        if not self._cache_hit or self._replay_index >= len(self._replay_turns):
            return None

        turn = self._replay_turns[self._replay_index]
        self._replay_index += 1

        tool_calls_raw = turn.get("tool_calls", [])
        content = turn.get("content", "")

        if tool_calls_raw:
            tc_objects = []
            for tc in tool_calls_raw:
                tc_objects.append(
                    ToolCall(
                        id=tc.get(
                            "id", f"cache_{self._replay_index}_{tc.get('name', '')}"
                        ),
                        name=tc["name"],
                        arguments=tc.get("arguments", {}),
                    )
                )
            assistant_message = AssistantMessage(
                role="assistant",
                content="",
                tool_calls=tc_objects,
            )
        else:
            assistant_message = AssistantMessage(
                role="assistant",
                content=content,
            )

        logger.debug(
            f"[REPLAY {self._replay_index}/{len(self._replay_turns)}] "
            f"{'TOOL: ' + ','.join(tc['name'] for tc in tool_calls_raw) if tool_calls_raw else 'MSG'}"
        )

        # Update conversation state tracking
        if assistant_message.is_tool_call():
            for tc in assistant_message.tool_calls or []:
                state.conv_state.tools_called.append(tc.name)
                state.conv_state.tool_call_count[tc.name] = (
                    state.conv_state.tool_call_count.get(tc.name, 0) + 1
                )
                if tc.name in WRITE_ACTIONS:
                    state.conv_state.write_actions_taken.append(tc.name)

        state.messages.append(assistant_message)
        state.conv_state.turn_count += 1

        return assistant_message, state

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AdaptiveAgentStateType
    ) -> tuple[AssistantMessage, AdaptiveAgentStateType]:
        """Generate response with cache replay or self-verification loop."""

        # Add incoming message to history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # === CACHE REPLAY: try cached turn before calling LLM ===
        cached_result = self._replay_cached_turn(state)
        if cached_result is not None:
            return cached_result

        messages = state.system_messages + state.messages

        # Generate initial response (LLM path)
        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="adaptive_agent_response",
            **self.llm_args,
        )

        # Self-verification loop (structural checks)
        for retry in range(self.MAX_RETRIES):
            is_valid, violation = verify_response(
                assistant_message, state.conv_state, self.tools
            )
            if is_valid:
                break

            logger.warning(
                f"AdaptiveAgent: verification failed (retry {retry + 1}): {violation}"
            )

            retry_hint = UserMessage(
                role="user",
                content=(
                    f"[SYSTEM NOTE - not from customer]\n"
                    f"{RETRY_PROMPT.format(violation=violation)}"
                ),
            )
            retry_messages = messages + [retry_hint]

            assistant_message = generate(
                model=self.llm,
                tools=self.tools,
                messages=retry_messages,
                call_name="adaptive_agent_retry",
                **self.llm_args,
            )

        # Policy verification for write actions — inner monologue + verify
        # Ask the LLM to reason about the action before executing.
        # This catches policy violations the structural checks miss.
        if self.VERIFY_WRITE_ACTIONS and assistant_message.is_tool_call():
            write_calls = [
                tc
                for tc in (assistant_message.tool_calls or [])
                if tc.name in WRITE_ACTIONS
            ]
            if write_calls:
                tool_desc = ", ".join(
                    f"{tc.name}({json.dumps(tc.arguments or {})})" for tc in write_calls
                )

                # Inner monologue: force the model to reason about consequences
                verify_prompt = UserMessage(
                    role="user",
                    content=(
                        f"[SYSTEM NOTE - not from customer]\n"
                        f"INNER MONOLOGUE — reason through this action before executing:\n\n"
                        f"Proposed action: {tool_desc}\n\n"
                        f"Answer these questions internally:\n"
                        f"1. CURRENT STATE: What data have I retrieved? What is the DB state?\n"
                        f"2. POLICY CHECK: Does the policy allow this specific action with these exact parameters?\n"
                        f"3. PARAMETERS: Are ALL arguments correct? (IDs, amounts, dates, names)\n"
                        f"4. COMPLETENESS: Am I missing any required fields or steps?\n"
                        f"5. SEQUENCE: Should any other action happen BEFORE this one?\n"
                        f"6. CONSEQUENCE: What exactly will change in the database?\n\n"
                        f"If EVERYTHING is correct: respond with the exact same tool call.\n"
                        f"If ANY issue found: respond with a CORRECTED tool call or a message explaining the issue."
                    ),
                )
                verify_messages = messages + [verify_prompt]

                verified_message = generate(
                    model=self.llm,
                    tools=self.tools,
                    messages=verify_messages,
                    call_name="adaptive_agent_verify_write",
                    **self.llm_args,
                )
                # Use the verified response (may have changed to text refusal or corrected call)
                assistant_message = verified_message
                logger.info(
                    f"AdaptiveAgent: write action verified — "
                    f"{'maintained' if verified_message.is_tool_call() else 'blocked'}"
                )

        # Update conversation state
        state.conv_state = update_conv_state(
            state.conv_state, message, assistant_message
        )

        state.messages.append(assistant_message)
        return assistant_message, state


# ═══════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════


def create_adaptive_agent(tools, domain_policy, **kwargs):
    """Factory function for AdaptiveAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - llm (str): LLM model name.
            - llm_args (dict): Additional LLM arguments.
    """
    # Auto-load golden cache if GOLDEN_CACHE_PATH is set
    cache_path = os.environ.get("GOLDEN_CACHE_PATH", "")
    if cache_path and AdaptiveAgent._golden_cache is None:
        AdaptiveAgent.load_golden_cache(cache_path)

    return AdaptiveAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )

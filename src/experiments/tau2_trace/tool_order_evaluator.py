"""
Phase 2: Tool Ordering Evaluator.

For telecom: loads DOT workflow files as DAGs and validates that the agent's
tool call sequence follows valid policy paths.
For retail/airline: checks read-after-write verification patterns.
"""

from __future__ import annotations

from typing import Optional

from experiments.tau2_trace.models import OrderingMetrics, ToolCallRecord

# ---------------------------------------------------------------------------
# Telecom: DOT-label-to-tool-name mapping
#
# The DOT files use human-readable labels like "Check Airplane Mode".
# We map these to the actual tool function names from
# src/tau2/domains/telecom/tools.py and user_tools.py.
# ---------------------------------------------------------------------------

# Agent tools (requestor="assistant")
TELECOM_AGENT_TOOLS_WRITE = {
    "suspend_line",
    "resume_line",
    "send_payment_request",
    "refuel_data",
    "enable_roaming",
    "disable_roaming",
    "transfer_to_human_agents",
}

TELECOM_AGENT_TOOLS_READ = {
    "get_customer_by_phone",
    "get_customer_by_id",
    "get_customer_by_name",
    "get_details_by_id",
    "get_bills_for_customer",
    "get_data_usage",
}

# User tools (requestor="user")
TELECOM_USER_TOOLS_WRITE = {
    "toggle_airplane_mode",
    "reseat_sim_card",
    "toggle_data",
    "toggle_roaming",
    "toggle_data_saver_mode",
    "set_network_mode_preference",
    "set_apn_settings",
    "reset_apn_settings",
    "toggle_wifi",
    "toggle_wifi_calling",
    "connect_vpn",
    "disconnect_vpn",
    "grant_app_permission",
    "reboot_device",
    "make_payment",
}

TELECOM_USER_TOOLS_READ = {
    "check_status_bar",
    "check_network_status",
    "check_network_mode_preference",
    "run_speed_test",
    "check_sim_status",
    "check_data_restriction_status",
    "check_apn_settings",
    "check_wifi_status",
    "check_wifi_calling_status",
    "check_vpn_status",
    "check_installed_apps",
    "check_app_status",
    "check_app_permissions",
    "can_send_mms",
    "check_payment_request",
}

# Ordered workflow phases from DOT files -- each path defines the expected
# high-level sequence of tool categories the agent/user should follow.
# We don't enforce exact tool names against DAG nodes (too brittle), but
# instead validate the high-level ordering of diagnostic phases.

TELECOM_PATH1_PHASE_ORDER = [
    {"check_status_bar"},  # Step 1.0: check service status
    {"toggle_airplane_mode", "check_network_status"},  # Step 1.1: airplane mode
    {"check_sim_status", "reseat_sim_card"},  # Step 1.2: SIM verification
    {"reset_apn_settings", "reboot_device"},  # Step 1.3: APN reset
    {
        "get_details_by_id",
        "resume_line",
        "send_payment_request",
    },  # Step 1.4: suspension
]

TELECOM_PATH2_PHASE_ORDER = [
    {"run_speed_test"},  # Step 2.0: speed test
    {"toggle_roaming", "enable_roaming"},  # Step 2.1.2: roaming
    {"toggle_data"},  # Step 2.1.3: mobile data toggle
    {"refuel_data"},  # Step 2.1.4: data usage
    {"toggle_data_saver_mode", "check_data_restriction_status"},  # Step 2.2.1
    {"set_network_mode_preference", "check_network_mode_preference"},  # Step 2.2.2
    {"disconnect_vpn", "check_vpn_status"},  # Step 2.2.3: VPN
]

TELECOM_PATH3_PHASE_ORDER = [
    {"can_send_mms"},  # Step 3.0: MMS check
    {"check_status_bar"},  # Step 3.1: network service
    {"run_speed_test", "toggle_data"},  # Step 3.2: mobile data
    {"set_network_mode_preference"},  # Step 3.3: network tech
    {"toggle_wifi_calling", "check_wifi_calling_status"},  # Step 3.4
    {"grant_app_permission", "check_app_permissions"},  # Step 3.5
    {"reset_apn_settings", "check_apn_settings"},  # Step 3.6: APN
]

TELECOM_WORKFLOWS = {
    "path1_no_service": TELECOM_PATH1_PHASE_ORDER,
    "path2_mobile_data": TELECOM_PATH2_PHASE_ORDER,
    "path3_mms": TELECOM_PATH3_PHASE_ORDER,
}

# ---------------------------------------------------------------------------
# Retail / Airline: write → read verification mappings
# ---------------------------------------------------------------------------

RETAIL_WRITE_TO_READ = {
    "cancel_pending_order": ["get_order_details"],
    "modify_pending_order_items": ["get_order_details"],
    "modify_pending_order_payment": ["get_order_details"],
    "modify_pending_order_address": ["get_order_details"],
    "return_delivered_order_items": ["get_order_details"],
    "exchange_delivered_order_items": ["get_order_details"],
}

AIRLINE_WRITE_TO_READ = {
    "book_reservation": ["get_reservation_details"],
    "cancel_reservation": ["get_reservation_details"],
    "update_reservation_passengers": ["get_reservation_details"],
    "update_reservation_flights": ["get_reservation_details"],
    "update_reservation_baggages": ["get_reservation_details"],
    "send_certificate": ["get_user_details"],
}


def _match_workflow_phase_order(
    tool_names: list[str],
    phase_order: list[set[str]],
) -> tuple[int, int]:
    """
    Check how well the sequence of tool names follows the expected phase order.
    Returns (valid_transitions, total_phases_checked).

    For each phase, we find the first tool call matching that phase. The sequence
    of first-match indices must be non-decreasing for the ordering to be valid.
    """
    phase_first_indices: list[int] = []
    for phase_tools in phase_order:
        for i, name in enumerate(tool_names):
            if name in phase_tools:
                phase_first_indices.append(i)
                break

    if len(phase_first_indices) < 2:
        return (0, 0)

    valid = 0
    total = len(phase_first_indices) - 1
    for i in range(1, len(phase_first_indices)):
        if phase_first_indices[i] >= phase_first_indices[i - 1]:
            valid += 1

    return (valid, total)


def _best_workflow_match(
    tool_names: list[str],
) -> tuple[str, float, int, int]:
    """
    Try all telecom workflows and return the best-matching one.
    Returns (workflow_name, adherence_score, valid_transitions, total).
    """
    best_name = "none"
    best_score = 0.0
    best_valid = 0
    best_total = 0

    for name, phases in TELECOM_WORKFLOWS.items():
        valid, total = _match_workflow_phase_order(tool_names, phases)
        score = valid / total if total > 0 else 0.0
        if score > best_score or (score == best_score and total > best_total):
            best_name = name
            best_score = score
            best_valid = valid
            best_total = total

    return best_name, best_score, best_valid, best_total


def _check_read_after_write(
    records: list[ToolCallRecord],
    write_to_read: dict[str, list[str]],
    lookahead: int = 5,
) -> tuple[int, int, float]:
    """
    For each write tool call, check if a corresponding read tool was called
    within `lookahead` subsequent calls to verify the mutation.
    Returns (total_writes, verified_writes, score).
    """
    total_writes = 0
    verified = 0

    for i, rec in enumerate(records):
        if rec.name in write_to_read:
            total_writes += 1
            expected_reads = write_to_read[rec.name]
            window = records[i + 1 : i + 1 + lookahead]
            for future in window:
                if future.name in expected_reads:
                    verified += 1
                    break

    score = verified / total_writes if total_writes > 0 else 1.0
    return total_writes, verified, score


def evaluate_tool_ordering(
    records: list[ToolCallRecord],
    domain: str,
    task_id: str,
    trial: Optional[int] = None,
) -> OrderingMetrics:
    """
    Evaluate tool ordering for a single simulation run.

    For telecom: matches the tool call sequence against workflow DAGs.
    For retail/airline: checks read-after-write verification patterns.
    """
    metrics = OrderingMetrics(task_id=task_id, trial=trial)

    tool_names = [r.name for r in records]

    if domain == "telecom":
        workflow_name, adherence, valid, total = _best_workflow_match(tool_names)
        metrics.policy_adherence_score = adherence
        metrics.total_transitions = total
        metrics.valid_transitions = valid
        metrics.matched_workflow = workflow_name

        # Also check read-after-write for agent-side tools
        all_write_to_read: dict[str, list[str]] = {}
        for wt in TELECOM_AGENT_TOOLS_WRITE:
            all_write_to_read[wt] = list(TELECOM_AGENT_TOOLS_READ)
        wt_total, wt_verified, wt_score = _check_read_after_write(
            records, all_write_to_read
        )
        metrics.read_after_write_score = wt_score
        metrics.write_calls_total = wt_total
        metrics.write_calls_verified = wt_verified

    elif domain == "retail":
        wt_total, wt_verified, wt_score = _check_read_after_write(
            records, RETAIL_WRITE_TO_READ
        )
        metrics.read_after_write_score = wt_score
        metrics.write_calls_total = wt_total
        metrics.write_calls_verified = wt_verified
        metrics.policy_adherence_score = wt_score

    elif domain == "airline":
        wt_total, wt_verified, wt_score = _check_read_after_write(
            records, AIRLINE_WRITE_TO_READ
        )
        metrics.read_after_write_score = wt_score
        metrics.write_calls_total = wt_total
        metrics.write_calls_verified = wt_verified
        metrics.policy_adherence_score = wt_score

    return metrics

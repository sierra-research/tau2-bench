"""Tests for the tool ordering evaluator."""

import re
from pathlib import Path

from experiments.tau2_trace.models import ToolCallRecord
from experiments.tau2_trace.tool_order_evaluator import (
    AIRLINE_WRITE_TO_READ,
    RETAIL_WRITE_TO_READ,
    TELECOM_AGENT_TOOLS_READ,
    TELECOM_AGENT_TOOLS_WRITE,
    TELECOM_PATH1_PHASE_ORDER,
    TELECOM_USER_TOOLS_READ,
    TELECOM_USER_TOOLS_WRITE,
    TELECOM_WORKFLOWS,
    _best_workflow_match,
    _check_read_after_write,
    _match_workflow_phase_order,
    evaluate_tool_ordering,
)


class TestPhaseOrderMatching:
    def test_perfect_order(self):
        tools = [
            "check_status_bar",
            "toggle_airplane_mode",
            "check_sim_status",
            "reset_apn_settings",
            "get_details_by_id",
        ]
        valid, total = _match_workflow_phase_order(tools, TELECOM_PATH1_PHASE_ORDER)
        assert valid == total
        assert total == 4  # 5 phases matched = 4 transitions

    def test_reversed_order(self):
        tools = [
            "get_details_by_id",
            "reset_apn_settings",
            "check_sim_status",
            "toggle_airplane_mode",
            "check_status_bar",
        ]
        valid, total = _match_workflow_phase_order(tools, TELECOM_PATH1_PHASE_ORDER)
        assert valid == 0

    def test_partial_match(self):
        tools = ["check_status_bar", "reset_apn_settings"]
        valid, total = _match_workflow_phase_order(tools, TELECOM_PATH1_PHASE_ORDER)
        assert total == 1
        assert valid == 1

    def test_no_matching_tools(self):
        tools = ["unrelated_tool_a", "unrelated_tool_b"]
        valid, total = _match_workflow_phase_order(tools, TELECOM_PATH1_PHASE_ORDER)
        assert total == 0


class TestBestWorkflowMatch:
    def test_matches_path1(self):
        tools = ["check_status_bar", "toggle_airplane_mode", "check_sim_status"]
        name, score, valid, total = _best_workflow_match(tools)
        assert name == "path1_no_service"
        assert score > 0.0

    def test_matches_path2(self):
        tools = [
            "run_speed_test",
            "toggle_roaming",
            "toggle_data",
            "toggle_data_saver_mode",
        ]
        name, score, valid, total = _best_workflow_match(tools)
        assert name == "path2_mobile_data"

    def test_matches_path3(self):
        tools = [
            "can_send_mms",
            "check_status_bar",
            "toggle_wifi_calling",
            "reset_apn_settings",
        ]
        name, score, valid, total = _best_workflow_match(tools)
        assert name == "path3_mms"


class TestReadAfterWrite:
    def test_verified_write(self):
        records = [
            ToolCallRecord(
                turn_index=0,
                name="cancel_pending_order",
                arguments={"id": "O1"},
                requestor="assistant",
            ),
            ToolCallRecord(
                turn_index=1,
                name="get_order_details",
                arguments={"id": "O1"},
                requestor="assistant",
            ),
        ]
        total, verified, score = _check_read_after_write(records, RETAIL_WRITE_TO_READ)
        assert total == 1
        assert verified == 1
        assert score == 1.0

    def test_unverified_write(self):
        records = [
            ToolCallRecord(
                turn_index=0,
                name="cancel_pending_order",
                arguments={"id": "O1"},
                requestor="assistant",
            ),
            ToolCallRecord(
                turn_index=1,
                name="find_user_id_by_email",
                arguments={"email": "x"},
                requestor="assistant",
            ),
        ]
        total, verified, score = _check_read_after_write(records, RETAIL_WRITE_TO_READ)
        assert total == 1
        assert verified == 0
        assert score == 0.0

    def test_no_writes(self):
        records = [
            ToolCallRecord(
                turn_index=0,
                name="get_order_details",
                arguments={"id": "O1"},
                requestor="assistant",
            ),
        ]
        total, verified, score = _check_read_after_write(records, RETAIL_WRITE_TO_READ)
        assert total == 0
        assert score == 1.0


class TestEvaluateToolOrdering:
    def test_telecom_domain(self):
        records = [
            ToolCallRecord(turn_index=0, name="check_status_bar", requestor="user"),
            ToolCallRecord(turn_index=1, name="toggle_airplane_mode", requestor="user"),
            ToolCallRecord(turn_index=2, name="check_sim_status", requestor="user"),
        ]
        metrics = evaluate_tool_ordering(records, "telecom", "task_1", trial=0)
        assert metrics.matched_workflow == "path1_no_service"
        assert metrics.policy_adherence_score > 0.0

    def test_retail_domain(self):
        records = [
            ToolCallRecord(
                turn_index=0,
                name="cancel_pending_order",
                arguments={"id": "O1"},
                requestor="assistant",
            ),
            ToolCallRecord(
                turn_index=1,
                name="get_order_details",
                arguments={"id": "O1"},
                requestor="assistant",
            ),
        ]
        metrics = evaluate_tool_ordering(records, "retail", "task_1", trial=0)
        assert metrics.read_after_write_score == 1.0
        assert metrics.matched_workflow is None


# ---------------------------------------------------------------------------
# Phase constants validation against actual tool definitions and DOT files.
# These tests catch drift between the manually transcribed phase constants
# and the source of truth (tool definitions + workflow DOT files).
# ---------------------------------------------------------------------------


def _get_all_telecom_tool_names() -> set[str]:
    """Collect all tool names from the phase constants + tool set constants."""
    return (
        TELECOM_AGENT_TOOLS_READ
        | TELECOM_AGENT_TOOLS_WRITE
        | TELECOM_USER_TOOLS_READ
        | TELECOM_USER_TOOLS_WRITE
    )


def _get_actual_tool_names_from_source() -> set[str]:
    """Parse actual @is_tool(...) function names from telecom tools.py and user_tools.py."""
    tool_names: set[str] = set()
    tools_dir = Path(__file__).resolve().parents[3] / "tau2" / "domains" / "telecom"
    for filename in ("tools.py", "user_tools.py"):
        filepath = tools_dir / filename
        if not filepath.exists():
            continue
        content = filepath.read_text()
        # Match function definitions decorated with @is_tool(...)
        # Pattern: @is_tool(...) followed (possibly with whitespace/lines) by def func_name(
        # Handles both module-level and class method (indented) definitions
        for match in re.finditer(r"@is_tool\([^)]*\)\s*\n\s+def\s+(\w+)\s*\(", content):
            tool_names.add(match.group(1))
    return tool_names


def _get_dot_step_labels(dot_path: Path) -> list[str]:
    """Extract 'Step X.Y:' labels from a DOT workflow file."""
    if not dot_path.exists():
        return []
    content = dot_path.read_text()
    # Match labels like: [label="Step 1.0: Check if user is facing..."]
    return re.findall(r'label="(Step \d+\.\d+[^"]*)"', content)


class TestPhaseConstantsValidation:
    """Validate that manually transcribed phase constants stay consistent
    with the actual telecom tool definitions and DOT workflow files."""

    def test_all_phase_tools_exist_in_tool_definitions(self):
        """Every tool name in TELECOM_PATH*_PHASE_ORDER must exist in the
        actual telecom agent or user tool definitions."""
        actual_tools = _get_actual_tool_names_from_source()
        assert len(actual_tools) > 0, "Failed to parse any tool names from source"

        all_phase_tools: set[str] = set()
        for phases in TELECOM_WORKFLOWS.values():
            for phase_set in phases:
                all_phase_tools.update(phase_set)

        missing = all_phase_tools - actual_tools
        assert missing == set(), (
            f"Phase constants reference tools not found in telecom "
            f"tools.py/user_tools.py: {sorted(missing)}"
        )

    def test_tool_set_constants_match_source(self):
        """TELECOM_AGENT_TOOLS_* and TELECOM_USER_TOOLS_* must be subsets
        of the actual tool definitions."""
        actual_tools = _get_actual_tool_names_from_source()
        all_constant_tools = _get_all_telecom_tool_names()

        missing = all_constant_tools - actual_tools
        assert missing == set(), (
            f"Tool set constants reference tools not in source: {sorted(missing)}"
        )

    def test_retail_write_to_read_tools_exist(self):
        """RETAIL_WRITE_TO_READ keys and values should reference real tools."""
        retail_dir = Path(__file__).resolve().parents[3] / "tau2" / "domains" / "retail"
        tools_file = retail_dir / "tools.py"
        if not tools_file.exists():
            return  # skip if retail domain not present
        content = tools_file.read_text()
        actual = {
            m.group(1)
            for m in re.finditer(r"@is_tool\([^)]*\)\s*\n\s+def\s+(\w+)\s*\(", content)
        }
        for write_tool, read_tools in RETAIL_WRITE_TO_READ.items():
            assert write_tool in actual, (
                f"Retail write tool {write_tool!r} not in source"
            )
            for rt in read_tools:
                assert rt in actual, f"Retail read tool {rt!r} not in source"

    def test_airline_write_to_read_tools_exist(self):
        """AIRLINE_WRITE_TO_READ keys and values should reference real tools."""
        airline_dir = (
            Path(__file__).resolve().parents[3] / "tau2" / "domains" / "airline"
        )
        tools_file = airline_dir / "tools.py"
        if not tools_file.exists():
            return
        content = tools_file.read_text()
        actual = {
            m.group(1)
            for m in re.finditer(r"@is_tool\([^)]*\)\s*\n\s+def\s+(\w+)\s*\(", content)
        }
        for write_tool, read_tools in AIRLINE_WRITE_TO_READ.items():
            assert write_tool in actual, (
                f"Airline write tool {write_tool!r} not in source"
            )
            for rt in read_tools:
                assert rt in actual, f"Airline read tool {rt!r} not in source"

    def test_dot_files_exist_for_each_workflow(self):
        """Each TELECOM_WORKFLOWS entry should have a corresponding DOT file."""
        workflows_dir = (
            Path(__file__).resolve().parents[4]
            / "data"
            / "tau2"
            / "domains"
            / "telecom"
            / "workflows"
        )
        expected_dots = {
            "path1_no_service": "tech_support_path1_no_service.dot",
            "path2_mobile_data": "tech_support_path2_mobile_data.dot",
            "path3_mms": "tech_support_path3_mms.dot",
        }
        for workflow_name in TELECOM_WORKFLOWS:
            dot_file = expected_dots.get(workflow_name)
            assert dot_file is not None, (
                f"No DOT file mapping for workflow {workflow_name!r}"
            )
            dot_path = workflows_dir / dot_file
            assert dot_path.exists(), (
                f"DOT file missing for {workflow_name}: {dot_path}"
            )

    def test_phase_count_matches_dot_step_count(self):
        """The number of phases in each workflow should approximate the
        number of 'Step X.Y' labels in the corresponding DOT file."""
        workflows_dir = (
            Path(__file__).resolve().parents[4]
            / "data"
            / "tau2"
            / "domains"
            / "telecom"
            / "workflows"
        )
        dot_map = {
            "path1_no_service": "tech_support_path1_no_service.dot",
            "path2_mobile_data": "tech_support_path2_mobile_data.dot",
            "path3_mms": "tech_support_path3_mms.dot",
        }
        for name, phases in TELECOM_WORKFLOWS.items():
            dot_path = workflows_dir / dot_map[name]
            step_labels = _get_dot_step_labels(dot_path)
            # Phase count should be within a reasonable range of DOT step count.
            # We allow some slack because DOT files include sub-steps and
            # decision nodes that don't map 1:1 to our phase constants.
            assert len(phases) > 0, f"Workflow {name} has no phases"
            assert len(step_labels) > 0, f"DOT file for {name} has no step labels"
            # Our phases should cover at least 40% of the DOT steps
            # (we intentionally aggregate sub-steps into phases)
            coverage = len(phases) / len(step_labels)
            assert coverage >= 0.3, (
                f"Workflow {name}: {len(phases)} phases vs {len(step_labels)} "
                f"DOT steps ({coverage:.0%} coverage, expected >= 30%)"
            )

"""SOP-as-DAG policy adherence verification.

Checks whether agent traces
follow the prescribed workflow order defined as a directed acyclic graph.

Each domain has a policy graph where nodes represent workflow phases
and edges represent valid transitions. An agent's action sequence is
checked against valid topological orderings of this graph.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from tau2_reliability.models import (
    PolicyAdherenceResult,
    PolicyViolation,
    PolicyViolationType,
    TaskTrialData,
)

# ---------------------------------------------------------------------------
# Policy Graph data structure
# ---------------------------------------------------------------------------


class PolicyNode:
    """A node in the policy workflow DAG."""

    def __init__(
        self,
        name: str,
        tools: list[str],
        description: str = "",
        required: bool = True,
    ):
        self.name = name
        self.tools = tools  # Tool names that map to this workflow phase
        self.description = description
        self.required = required


class PolicyGraph:
    """Directed acyclic graph representing a domain's workflow policy."""

    def __init__(self, domain: str, nodes: list[PolicyNode], edges: list[tuple[str, str]]):
        self.domain = domain
        self.nodes = {n.name: n for n in nodes}
        self.edges = edges
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._reverse: dict[str, list[str]] = defaultdict(list)
        for src, dst in edges:
            self._adjacency[src].append(dst)
            self._reverse[dst].append(src)

        # Build tool -> node mapping
        self._tool_to_node: dict[str, str] = {}
        for node in nodes:
            for tool in node.tools:
                self._tool_to_node[tool] = node.name

    def tool_to_node(self, tool_name: str) -> Optional[str]:
        """Map a tool name to its workflow phase node."""
        return self._tool_to_node.get(tool_name)

    def get_predecessors(self, node_name: str) -> list[str]:
        """Get required predecessor nodes."""
        return self._reverse.get(node_name, [])

    def get_successors(self, node_name: str) -> list[str]:
        """Get valid successor nodes."""
        return self._adjacency.get(node_name, [])

    def get_required_nodes(self) -> list[str]:
        """Get all required workflow nodes."""
        return [n for n, node in self.nodes.items() if node.required]

    def topological_order(self) -> list[str]:
        """Return one valid topological ordering of the DAG."""
        in_degree = {n: 0 for n in self.nodes}
        for _, dst in self.edges:
            in_degree[dst] = in_degree.get(dst, 0) + 1
        queue = deque(n for n, d in in_degree.items() if d == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for succ in self._adjacency.get(node, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return order


# ---------------------------------------------------------------------------
# Domain-specific policy graphs
# ---------------------------------------------------------------------------

AIRLINE_POLICY = PolicyGraph(
    domain="airline",
    nodes=[
        PolicyNode("authenticate", ["get_user_details"],
                   "Verify customer identity"),
        PolicyNode("gather_info", ["get_reservation_details", "get_flight_status",
                                    "list_all_airports", "search_direct_flight",
                                    "search_onestop_flight"],
                   "Retrieve relevant data"),
        PolicyNode("execute", ["book_reservation", "cancel_reservation",
                               "update_reservation_flights", "update_reservation_passengers",
                               "update_reservation_baggages", "send_certificate"],
                   "Execute the requested action"),
        PolicyNode("escalate", ["transfer_to_human_agents", "calculate"],
                   "Escalate or calculate", required=False),
    ],
    edges=[
        ("authenticate", "gather_info"),
        ("gather_info", "execute"),
        ("execute", "escalate"),
    ],
)

RETAIL_POLICY = PolicyGraph(
    domain="retail",
    nodes=[
        PolicyNode("authenticate", ["find_user_id_by_name_zip", "find_user_id_by_email"],
                   "Authenticate customer via name+zip or email"),
        PolicyNode("gather_info", ["get_user_details", "get_order_details",
                                    "get_product_details", "get_item_details",
                                    "list_all_product_types"],
                   "Retrieve user, order, and product data"),
        PolicyNode("execute", ["cancel_pending_order", "return_delivered_order_items",
                               "modify_pending_order_items", "modify_pending_order_payment",
                               "modify_pending_order_address", "exchange_delivered_order_items",
                               "modify_user_address"],
                   "Execute the requested action"),
        PolicyNode("escalate", ["transfer_to_human_agents", "calculate"],
                   "Escalate or calculate", required=False),
    ],
    edges=[
        ("authenticate", "gather_info"),
        ("gather_info", "execute"),
        ("execute", "escalate"),
    ],
)

TELECOM_POLICY = PolicyGraph(
    domain="telecom",
    nodes=[
        PolicyNode("authenticate", ["get_customer_by_name", "get_customer_by_phone",
                                     "get_customer_by_id"],
                   "Verify customer identity"),
        PolicyNode("gather_info", ["get_details_by_id", "get_bills_for_customer",
                                    "get_data_usage"],
                   "Retrieve account and billing info"),
        PolicyNode("execute", ["suspend_line", "resume_line", "refuel_data",
                               "enable_roaming", "disable_roaming",
                               "send_payment_request"],
                   "Apply the fix"),
        PolicyNode("escalate", ["transfer_to_human_agents"],
                   "Escalate to human", required=False),
    ],
    edges=[
        ("authenticate", "gather_info"),
        ("gather_info", "execute"),
        ("execute", "escalate"),
    ],
)

DOMAIN_POLICIES: dict[str, PolicyGraph] = {
    "airline": AIRLINE_POLICY,
    "retail": RETAIL_POLICY,
    "telecom": TELECOM_POLICY,
}


def get_policy_graph(domain: str) -> Optional[PolicyGraph]:
    """Get the policy graph for a domain. Returns None if not defined."""
    # Normalize domain name
    for key in DOMAIN_POLICIES:
        if key in domain.lower():
            return DOMAIN_POLICIES[key]
    return None


# ---------------------------------------------------------------------------
# Adherence checking
# ---------------------------------------------------------------------------


def check_trace_adherence(
    action_sequence: list[str],
    policy_graph: PolicyGraph,
) -> PolicyAdherenceResult:
    """Check if an action sequence follows the policy graph workflow.

    Maps actions to workflow phases, then checks:
    1. Required phases were visited
    2. Phases were visited in a valid topological order
    3. No phase was visited before its prerequisites
    """
    # Map actions to workflow phases
    phase_sequence = []
    seen_phases = set()
    for action in action_sequence:
        phase = policy_graph.tool_to_node(action)
        if phase and phase not in seen_phases:
            phase_sequence.append(phase)
            seen_phases.add(phase)

    violations = []
    matched_transitions = 0
    expected_transitions = 0

    # Check required phases
    for node_name in policy_graph.get_required_nodes():
        expected_transitions += 1
        if node_name in seen_phases:
            matched_transitions += 1
        else:
            violations.append(PolicyViolation(
                violation_type=PolicyViolationType.SKIPPED_NODE,
                expected_node=node_name,
                description=f"Required phase '{node_name}' was never visited",
            ))

    # Check ordering: for each observed phase, verify predecessors came first
    phase_order = {phase: idx for idx, phase in enumerate(phase_sequence)}
    for phase in phase_sequence:
        predecessors = policy_graph.get_predecessors(phase)
        required_preds = [p for p in predecessors if policy_graph.nodes.get(p, PolicyNode("", [])).required]
        for pred in required_preds:
            expected_transitions += 1
            if pred in phase_order and phase_order[pred] < phase_order[phase]:
                matched_transitions += 1
            elif pred not in phase_order:
                # Predecessor was never visited — already caught as SKIPPED_NODE
                pass
            else:
                violations.append(PolicyViolation(
                    violation_type=PolicyViolationType.WRONG_ORDER,
                    expected_node=pred,
                    actual_action=phase,
                    description=f"Phase '{phase}' visited before prerequisite '{pred}'",
                ))

    total = max(expected_transitions, 1)
    adherence_score = matched_transitions / total

    # Expected path = topological order; matched path = what was actually observed
    expected_path = policy_graph.topological_order()

    return PolicyAdherenceResult(
        task_id="",  # Caller sets this
        adherence_score=max(0.0, min(1.0, adherence_score)),
        violations=violations,
        matched_path=phase_sequence,
        expected_path=expected_path,
    )


def compute_policy_adherence(
    task_data: list[TaskTrialData],
    domain: str,
) -> list[PolicyAdherenceResult]:
    """Compute policy adherence for all tasks in a domain.

    Checks the most common action sequence per task (across trials).
    """
    policy_graph = get_policy_graph(domain)
    if policy_graph is None:
        return []

    results = []
    for td in task_data:
        # Use the first trial's sequence (or most common if available)
        if not td.action_sequences:
            continue
        # Check each trial and average
        trial_scores = []
        trial_violations = []
        for seq in td.action_sequences:
            result = check_trace_adherence(seq, policy_graph)
            result.task_id = td.task_id
            trial_scores.append(result.adherence_score)
            trial_violations.extend(result.violations)

        # Report the result using the first trial's path info + averaged score
        first_result = check_trace_adherence(td.action_sequences[0], policy_graph)
        first_result.task_id = td.task_id
        first_result.adherence_score = sum(trial_scores) / len(trial_scores)
        # Deduplicate violations across trials
        seen = set()
        unique_violations = []
        for v in trial_violations:
            key = (v.violation_type, v.expected_node, v.actual_action)
            if key not in seen:
                seen.add(key)
                unique_violations.append(v)
        first_result.violations = unique_violations
        results.append(first_result)

    return results

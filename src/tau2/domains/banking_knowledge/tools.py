"""Tools for the knowledge domain."""

import inspect
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, get_args

from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.banking_knowledge.db_query import (
    add_to_db,
    query_database_tool,
    update_record_in_db,
)
from tau2.domains.banking_knowledge.utils import (
    _deterministic_id,
    generate_account_flag_id,
    generate_agent_discoverable_tool_id,
    generate_application_id,
    generate_closure_reason_id,
    generate_credit_card_order_id,
    generate_credit_limit_increase_request_id,
    generate_debit_card_id,
    generate_debit_card_order_id,
    generate_dispute_id,
    generate_referral_id,
    generate_referral_link_id,
    generate_transaction_id,
    generate_user_discoverable_tool_call_id,
    generate_user_discoverable_tool_id,
    generate_verification_id,
    get_now,
    get_today_str,
)
from tau2.environment.toolkit import (
    DISCOVERABLE_ATTR,
    ToolKitBase,
    ToolType,
    is_discoverable_tool,
    is_tool,
)

if TYPE_CHECKING:
    pass


TransferReasonLiteral = Literal[
    "fraud_or_security_concern",
    "account_closure_request",
    "deceased_account_holder",
    "legal_or_regulatory_matter",
    "account_ownership_dispute",
    "complex_billing_dispute",
    "abusive_customer_behavior",
    "third_party_inquiry",
    "technical_system_error",
    "unconfirmed_external_communication",
    "customer_demands_after_unavailable_offer_refusal",
    "kb_search_unsuccessful_customer_requests_transfer",
    "specialized_department_required",
    "accessibility_or_special_needs",
    "customer_frustrated_demands_human",
    "supervisor_request_service_complaint",
    "customer_requests_human_no_specific_reason",
    "request_completed_customer_wants_human_followup",
    # Tier 4
    "other",
]


# =============================================================================
# Helper functions for discoverable tools
# =============================================================================


def _parse_balance(val: Any) -> float:
    """Parse a balance value that may be a number or a string like '$2,850.00'."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val.replace("$", "").replace(",", ""))
    return 0.0


def _get_account_balance(account: Dict[str, Any]) -> float:
    """Get account balance from either 'current_holdings' or 'balance' field."""
    return _parse_balance(account.get("current_holdings", account.get("balance", 0)))


def parse_discoverable_tool_docstring(func) -> Dict[str, Any]:
    """Parse a discoverable tool's docstring into description and parameters.

    Expected docstring format:
    ```
    Short description of the tool.

    Args:
        param_name (type): Description of the parameter
        another_param (type, optional): Optional parameter description

    Returns:
        Description of return value (used as success_message hint)
    ```

    Args:
        func: The decorated function to parse

    Returns:
        Dictionary with 'description', 'parameters', and optionally 'success_message'
    """
    docstring = inspect.getdoc(func) or ""

    # Split into sections
    parts = re.split(r"\n\s*Args:\s*\n", docstring, maxsplit=1)
    description = parts[0].strip()

    parameters = {}
    success_message = "Action completed successfully."

    if len(parts) > 1:
        remaining = parts[1]

        # Check if there's a Returns section
        returns_split = re.split(r"\n\s*Returns:\s*\n", remaining, maxsplit=1)
        args_section = returns_split[0]

        if len(returns_split) > 1:
            # Use the Returns section as success message hint
            returns_text = returns_split[1].strip()
            # Take first line/paragraph as success message
            success_message = returns_text.split("\n")[0].strip()

        # Parse each arg line: "param_name (type): description" or "param_name (type, optional): description"
        # Also handle simple format: "param_name: description"
        arg_pattern = re.compile(
            r"^\s*(\w+)\s*"  # param name
            r"(?:\(([^)]+)\))?\s*"  # optional (type) or (type, optional)
            r":\s*(.+?)$",  # description
            re.MULTILINE,
        )

        for match in arg_pattern.finditer(args_section):
            param_name = match.group(1)
            type_info = match.group(2) or "string"
            param_desc = match.group(3).strip()

            # Check if optional
            is_optional = "optional" in type_info.lower()
            # Extract just the type (remove 'optional' if present)
            param_type = re.sub(
                r",?\s*optional", "", type_info, flags=re.IGNORECASE
            ).strip()
            if not param_type:
                param_type = "string"

            parameters[param_name] = {
                "type": param_type,
                "description": param_desc,
                "required": not is_optional,
            }

    return {
        "name": func.__name__,
        "description": description,
        "parameters": parameters,
        "success_message": success_message,
    }


def format_discoverable_tool_for_agent(tool_info: Dict[str, Any]) -> str:
    """Format a discoverable tool's info for display to the agent.

    Produces output matching the original AgentDiscoverableToolDefinition.format_for_agent().

    Args:
        tool_info: Dictionary from parse_discoverable_tool_docstring()

    Returns:
        Formatted string for agent display
    """
    param_strs = []
    for param_name, param_def in tool_info.get("parameters", {}).items():
        required = param_def.get("required", True)
        param_type = param_def.get("type", "string")
        desc = param_def.get("description", "")
        req_str = " (required)" if required else " (optional)"
        param_strs.append(f"  - {param_name}: {param_type}{req_str} - {desc}")

    params_section = "\n".join(param_strs) if param_strs else "  (no parameters)"

    return (
        f"Tool: {tool_info['name']}\n"
        f"Description: {tool_info['description']}\n"
        f"Parameters:\n{params_section}"
    )


def _validate_pin(pin: str) -> Optional[str]:
    """Validate a PIN meets security requirements.

    Returns None if valid, or an error message if invalid.
    """
    if not pin or not pin.isdigit() or len(pin) != 4:
        return "PIN must be exactly 4 digits."

    # Check for sequential PIN (e.g., 1234, 4321)
    sequential_pins = [
        "0123",
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "6789",
        "9876",
        "8765",
        "7654",
        "6543",
        "5432",
        "4321",
        "3210",
    ]
    if pin in sequential_pins:
        return "PIN cannot be sequential (e.g., 1234). Please choose a more secure PIN."

    # Check for repeating PIN (e.g., 1111, 2222)
    if len(set(pin)) == 1:
        return "PIN cannot be all the same digit (e.g., 1111). Please choose a more secure PIN."

    return None


def _validate_activation_common(
    args: Dict[str, Any],
    db: "TransactionalDB",
    allowed_issue_reasons: List[str],
    tool_name: str,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Common validation for all debit card activation tools.

    Returns (error_message, card) - if error_message is not None, activation should fail.
    """
    card_id = args.get("card_id")
    last_4_digits = args.get("last_4_digits")
    expiration_date = args.get("expiration_date")
    cvv = args.get("cvv")
    pin = args.get("pin")

    if not all([card_id, last_4_digits, expiration_date, cvv, pin]):
        return (
            "Error: Missing required parameters. Required: card_id, last_4_digits, expiration_date, cvv, pin.",
            None,
        )

    # Validate PIN
    pin_error = _validate_pin(pin)
    if pin_error:
        return f"Error: {pin_error}", None

    # Validate CVV format
    if not cvv.isdigit() or len(cvv) != 3:
        return "Error: CVV must be exactly 3 digits.", None

    # Validate last 4 digits format
    if not last_4_digits.isdigit() or len(last_4_digits) != 4:
        return "Error: Last 4 digits must be exactly 4 digits.", None

    # Verify the card exists
    if card_id not in db.debit_cards.data:
        return f"Error: Debit card '{card_id}' not found.", None

    card = db.debit_cards.data[card_id]

    # Check issue_reason matches allowed reasons for this tool
    issue_reason = card.get("issue_reason", "new_account")
    if issue_reason not in allowed_issue_reasons:
        reason_map = {
            "new_account": "activate_debit_card_8291",
            "first_card": "activate_debit_card_8291",
            "lost": "activate_debit_card_8292",
            "stolen": "activate_debit_card_8292",
            "fraud": "activate_debit_card_8292",
            "expired": "activate_debit_card_8293",
            "damaged": "activate_debit_card_8293",
            "upgrade": "activate_debit_card_8293",
            "bank_reissue": "activate_debit_card_8293",
        }
        correct_tool = reason_map.get(issue_reason, "unknown")
        return (
            f"Error: Wrong activation tool. This card has issue_reason='{issue_reason}'. Please use {correct_tool} instead of {tool_name}.",
            None,
        )

    # Check card status
    if card.get("status") == "ACTIVE":
        return f"Error: Debit card '{card_id}' is already active.", None

    if card.get("status") != "PENDING":
        return (
            f"Error: Debit card '{card_id}' cannot be activated. Current status: {card.get('status')}. Only PENDING cards can be activated.",
            None,
        )

    # Verify last 4 digits match
    if card.get("last_4_digits") != last_4_digits:
        return (
            "Error: Card verification failed. The last 4 digits do not match our records.",
            None,
        )

    # Verify the linked account is still open
    account_id = card.get("account_id")
    if account_id and account_id in db.accounts.data:
        account = db.accounts.data[account_id]
        if account.get("status") != "OPEN":
            return (
                f"Error: The linked checking account '{account_id}' is no longer open. Card cannot be activated.",
                None,
            )

    return None, card


class KnowledgeTools(ToolKitBase):
    """Tools for the knowledge domain (Agent tools).

    The `db` attribute is the TransactionalDB (users, accounts, applications, referrals)
    which is used for DB state hashing during evaluation.

    Note: KB_search and other knowledge-retrieval tools are provided by retrieval configs,
    not by this base toolkit.

    Agent discoverable tools are defined as methods decorated with @is_discoverable_tool.
    Their docstrings serve as the source of truth for description and parameters.
    """

    db: TransactionalDB

    def __init__(
        self,
        db: TransactionalDB,
    ) -> None:
        super().__init__(db)
        self._user_discoverable_tools_state: Dict[str, Dict[str, Any]] = {}
        self._agent_discoverable_tools_state: Dict[str, Dict[str, Any]] = {}

    def get_user_discoverable_tools_state(self) -> Dict[str, Dict[str, Any]]:
        """Get the current state of user discoverable tools (for sharing with user tools)."""
        return self._user_discoverable_tools_state

    def get_agent_discoverable_tools_state(self) -> Dict[str, Dict[str, Any]]:
        """Get the current state of agent discoverable tools."""
        return self._agent_discoverable_tools_state

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(
        self,
        summary: str,
        reason: TransferReasonLiteral = "other",
    ) -> str:
        """Transfer the user to a human agent.

        The proper transfer reason enum can be found in the knowledge base: search it before calling this tool to select the proper applicable reason.

        Args:
            summary: A summary of the user's issue and what was attempted before transfer.
            reason: The specific reason code for the transfer.
        """
        valid_reasons = list(get_args(TransferReasonLiteral))
        if reason not in valid_reasons:
            return f"Error: Invalid transfer reason '{reason}'. Must be one of: {', '.join(valid_reasons)}"

        return f"Transfer successful (reason: {reason}). A human agent will assist you shortly."

    @is_tool(ToolType.READ)
    def get_current_time(self) -> str:
        """Get the current time. Use this to get the current timestamp for logging verification records.

        Returns:
            The current time in the format "YYYY-MM-DD HH:MM:SS TZ"
        """
        return "The current time is 2025-11-14 03:40:00 EST."

    @is_tool(ToolType.READ)
    def get_user_information_by_id(self, user_id: str) -> str:
        """Get the information (date of birth, email, phone number, address) for a user by their user id.

        Args:
            user_id: The ID of the user
        """
        return query_database_tool("users", f'{{"user_id": "{user_id}"}}', db=self.db)

    @is_tool(ToolType.READ)
    def get_user_information_by_name(self, customer_name: str) -> str:
        """Get the information (date of birth, email, phone number, address) for a user by their name. Case Sensitive.

        Args:
            customer_name: The name of the user
        """
        return query_database_tool(
            "users", f'{{"name": "{customer_name}"}}', db=self.db
        )

    @is_tool(ToolType.READ)
    def get_user_information_by_email(self, email: str) -> str:
        """Get the information (date of birth, email, phone number, address) for a user by their email.

        Args:
            email: The email of the user
        """
        return query_database_tool("users", f'{{"email": "{email}"}}', db=self.db)

    @is_tool(ToolType.WRITE)
    def change_user_email(self, user_id: str, new_email: str) -> str:
        """Change the email address for a user.

        Args:
            user_id: The ID of the user whose email should be changed
            new_email: The new email address to set for the user
        """
        success, updated_record = update_record_in_db(
            "users", db=self.db, record_id=user_id, updates={"email": new_email}
        )

        if not success:
            return f"Error: User with ID '{user_id}' not found."

        return (
            f"Email updated successfully.\n"
            f"  - User ID: {user_id}\n"
            f"  - New Email: {new_email}"
        )

    @is_tool(ToolType.READ)
    def get_referrals_by_user(self, user_id: str) -> str:
        """Get all referrals made by a user.

        Args:
            user_id: The ID of the user (referrer) to look up referrals for
        """
        return query_database_tool(
            "referrals", f'{{"referrer_id": "{user_id}"}}', db=self.db
        )

    @is_tool(ToolType.READ)
    def get_credit_card_transactions_by_user(self, user_id: str) -> str:
        """Get all credit card transactions for a user.

        Args:
            user_id: The ID of the user to look up transactions for
        """
        return query_database_tool(
            "credit_card_transaction_history", f'{{"user_id": "{user_id}"}}', db=self.db
        )

    @is_tool(ToolType.READ)
    def get_credit_card_accounts_by_user(self, user_id: str) -> str:
        """Get all credit card accounts for a user.

        Returns information about each credit card account including card type,
        date opened, current balance, and reward points.

        Args:
            user_id: The ID of the user to look up credit card accounts for
        """
        return query_database_tool(
            "credit_card_accounts", f'{{"user_id": "{user_id}"}}', db=self.db
        )

    @is_tool(ToolType.WRITE)
    def log_verification(
        self,
        name: str,
        user_id: str,
        address: str,
        email: str,
        phone_number: str,
        date_of_birth: str,
        time_verified: str,
    ) -> str:
        """Log a verification record after successfully verifying a user's identity.

        Call this tool after you have verified a user by confirming 2 out of 4 identity fields
        (date of birth, email, phone number, address). This creates an audit record of the verification.

        Args:
            name: The verified user's full name
            user_id: The verified user's ID
            address: The verified user's address
            email: The verified user's email
            phone_number: The verified user's phone number
            date_of_birth: The verified user's date of birth (MM/DD/YYYY format)
            time_verified: The timestamp of the verification (e.g., "2025-11-14 03:40:00 EST")
        """
        # Generate a deterministic ID for this verification record
        record_id = generate_verification_id(user_id, time_verified)

        # Create the verification record
        record = {
            "name": name,
            "user_id": user_id,
            "address": address,
            "email": email,
            "phone_number": phone_number,
            "date_of_birth": date_of_birth,
            "time_verified": time_verified,
        }

        # Add to the verification_history table
        success = add_to_db("verification_history", record_id, record, db=self.db)

        if not success:
            return f"Failed to log verification: Record may already exist."

        return (
            f"Verification logged successfully.\n"
            f"  - User: {name} (ID: {user_id})\n"
            f"  - Verified at: {time_verified}"
        )

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def give_discoverable_user_tool(
        self, discoverable_tool_name: str, arguments: str = "{}"
    ) -> str:
        """Pass a tool to the user so they can execute it themselves.

        Use this when the knowledge base indicates that the user should perform
        an action themselves (e.g., "to do X, have the user call tool_name(args)").

        The user will then be able to call `call_discoverable_tool` with the same
        tool name and arguments to simulate executing the action.

        Args:
            discoverable_tool_name: The name of the discoverable tool (e.g., "open_webpage", "navigate_to_section")
            arguments: JSON string of arguments for the tool (e.g., '{"url": "https://example.com"}')

        Returns:
            A confirmation message with instructions for the user
        """
        # Check if the tool exists as a user discoverable method on KnowledgeUserTools
        if not hasattr(KnowledgeUserTools, discoverable_tool_name):
            return f"Error: Unknown discoverable tool '{discoverable_tool_name}'."

        method = getattr(KnowledgeUserTools, discoverable_tool_name)
        if not getattr(method, DISCOVERABLE_ATTR, False):
            return f"Error: Unknown discoverable tool '{discoverable_tool_name}'."

        # Parse arguments
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in arguments: {e}"

        # Validate arguments against method signature (don't require all - user fills in)
        sig = inspect.signature(method)
        for arg_name in args_dict:
            if arg_name not in sig.parameters and arg_name != "self":
                return f"Error: Unexpected parameter: {arg_name}"

        # Parse the docstring for description
        tool_info = parse_discoverable_tool_docstring(method)

        # Store the user discoverable tool state
        self._user_discoverable_tools_state[discoverable_tool_name] = {
            "arguments": args_dict,
            "given_at": get_now().isoformat(),
            "tool_info": tool_info,
        }

        # Store in the database for persistence and evaluation
        discoverable_tool_record = {
            "tool_name": discoverable_tool_name,
            "status": "GIVEN",
        }

        record_id = generate_user_discoverable_tool_id(discoverable_tool_name)
        add_to_db(
            "user_discoverable_tools", record_id, discoverable_tool_record, db=self.db
        )

        # Format instructions for the user
        args_str = json.dumps(args_dict, indent=2) if args_dict else "(no arguments)"
        return (
            f"Tool given to user: {discoverable_tool_name}\n"
            f"Description: {tool_info['description']}\n"
            f"Arguments: {args_str}\n\n"
            f"The user can now execute this action by calling `call_discoverable_user_tool` "
            f"with discoverable_tool_name='{discoverable_tool_name}' and the same arguments."
        )

    @is_tool(ToolType.GENERIC, mutates_state=True)
    def unlock_discoverable_agent_tool(self, agent_tool_name: str) -> str:
        """Unlock an agent discoverable tool that was found in the knowledge base.

        Use this when the knowledge base indicates that you have access to a specialized
        internal tool. The knowledge base will tell you the tool name to unlock.

        After unlocking, you can use the tool by calling `call_discoverable_agent_tool` with
        the tool name and required arguments.

        Args:
            agent_tool_name: The name of the agent discoverable tool to unlock
                            (e.g., "calculate_apr_adjustment_7842")

        Returns:
            A confirmation message with the tool's description and parameters
        """
        # Check if the tool exists as a discoverable method on this class
        if not self.has_discoverable_tool(agent_tool_name):
            return f"Error: Unknown agent tool '{agent_tool_name}'. This tool is not available."

        # Get the method and parse its docstring for tool info
        method = self.get_discoverable_tools()[agent_tool_name]
        tool_info = parse_discoverable_tool_docstring(method)

        # Store the agent discoverable tool state (in-memory only, DB write happens on call)
        self._agent_discoverable_tools_state[agent_tool_name] = {
            "unlocked_at": get_now().isoformat(),
            "tool_info": tool_info,
        }

        # Format tool info for the agent (same format as before)
        formatted_tool = format_discoverable_tool_for_agent(tool_info)
        return (
            f"Tool unlocked: {agent_tool_name}\n"
            f"Description: {tool_info['description']}\n\n"
            f"{formatted_tool}\n\n"
            f"You can now use this tool by calling `call_discoverable_agent_tool` with "
            f"agent_tool_name='{agent_tool_name}' and the required arguments."
        )

    @is_tool(ToolType.WRITE)
    def call_discoverable_agent_tool(
        self, agent_tool_name: str, arguments: str = "{}"
    ) -> str:
        """Call an agent discoverable tool that you have previously unlocked.

        Use this after unlocking a tool with `unlock_discoverable_agent_tool`. The knowledge base
        will tell you which tool to use and what arguments to provide.

        Args:
            agent_tool_name: The name of the agent discoverable tool to call
            arguments: JSON string of arguments for the tool (e.g., '{"user_id": "abc123"}')

        Returns:
            The result of executing the agent tool
        """
        # Check if the tool exists as a discoverable method on this class
        if not self.has_discoverable_tool(agent_tool_name):
            return f"Error: Unknown agent tool '{agent_tool_name}'. This tool is not available."

        # Check if the tool was unlocked (in-memory state)
        if agent_tool_name not in self._agent_discoverable_tools_state:
            return (
                f"Error: Tool '{agent_tool_name}' has not been unlocked. "
                f"You must first use `unlock_discoverable_agent_tool` to unlock this tool before calling it."
            )

        # Parse arguments
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in arguments: {e}"

        # Get the method and call it directly with the parsed arguments
        method = self.get_discoverable_tools()[agent_tool_name]
        try:
            result = method(**args_dict)
        except TypeError as e:
            return f"Error: Invalid arguments: {e}"

        # Record the call in the database for evaluation (only unique tool names)
        agent_tool_record = {"tool_name": agent_tool_name, "status": "CALLED"}
        record_id = generate_agent_discoverable_tool_id(agent_tool_name)
        add_to_db("agent_discoverable_tools", record_id, agent_tool_record, db=self.db)

        return result

    @is_tool(ToolType.READ)
    def list_discoverable_agent_tools(self) -> str:
        """List all agent discoverable tools that you have called.

        Use this to see what specialized tools you have used.

        Returns:
            A list of tools that you have called
        """
        result = query_database_tool("agent_discoverable_tools", "{}", db=self.db)

        if "No results found" in result:
            return "No agent tools have been called yet. Search the knowledge base to discover available tools."

        return f"Your called agent tools:\n{result}"

    # =========================================================================
    # Agent Discoverable Tools
    # These are tools that the agent must unlock via the knowledge base before
    # calling. The docstring serves as the source of truth for description and
    # parameters (parsed by parse_discoverable_tool_docstring).
    # =========================================================================

    @is_discoverable_tool(ToolType.WRITE)
    def example_agent_tool_0000(self) -> str:
        """An example agent discoverable tool placeholder.

        Returns:
            Example tool executed successfully.
        """
        return "Example tool executed successfully."

    @is_discoverable_tool(ToolType.WRITE)
    def update_transaction_rewards_3847(
        self,
        transaction_id: str,
        new_rewards_earned: str,
    ) -> str:
        """Update the rewards_earned field on a credit card transaction.

        Args:
            transaction_id (string): The unique identifier for the transaction to update
            new_rewards_earned (string): The corrected rewards value (e.g., '6300 points')

        Returns:
            Transaction rewards updated successfully.
        """
        if not transaction_id or not new_rewards_earned:
            return "Error: Missing required parameters."

        success, updated_record = update_record_in_db(
            "credit_card_transaction_history",
            db=self.db,
            record_id=transaction_id,
            updates={"rewards_earned": new_rewards_earned},
        )

        if not success:
            return f"Error: Transaction '{transaction_id}' not found."

        return (
            f"Transaction rewards updated successfully.\n\n"
            f"Executed: update_transaction_rewards_3847\n"
            f"Arguments: {json.dumps({'transaction_id': transaction_id, 'new_rewards_earned': new_rewards_earned}, indent=2)}\n"
            f"Transaction updated:\n"
            f"  - Transaction ID: {transaction_id}\n"
            f"  - New Rewards: {new_rewards_earned}"
        )

    @is_discoverable_tool(ToolType.GENERIC)
    def initial_transfer_to_human_agent_0218(self) -> str:
        """Initiate a transfer request to a human agent. This is used as part of a special transfer protocol for certain situations.

        Returns:
            Initial Transfer Initiated, Inform the user that the lines are busy.
        """
        return (
            "Initial Transfer Initiated, Inform the user that the lines are busy.\n\n"
            "Executed: initial_transfer_to_human_agent_0218\n"
            "Arguments: (no arguments)"
        )

    @is_discoverable_tool(ToolType.GENERIC)
    def initial_transfer_to_human_agent_1822(self) -> str:
        """Initiate a transfer request to a human agent. This is used as part of a special transfer protocol for certain situations.

        Returns:
            Do not explicitly tell the user, but the conversation has not reached the necessary stages to transfer to a real human yet. Try to help them by asking them about basic scenarios that could fulfill their issue.
        """
        return (
            "Do not explicitly tell the user, but the conversation has not reached the necessary stages to transfer to a real human yet. Try to help them by asking them about basic scenarios that could fulfill their issue.\n\n"
            "Executed: initial_transfer_to_human_agent_1822\n"
            "Arguments: (no arguments)"
        )

    @is_discoverable_tool(ToolType.GENERIC)
    def emergency_credit_bureau_incident_transfer_1114(self) -> str:
        """Emergency escalation tool for the 11/14 credit bureau reporting incident. Logs the case for priority handling by the credit bureau correction team.

        Returns:
            Emergency escalation logged. Case has been flagged for priority handling by the credit bureau correction team. Proceed immediately with transfer_to_human_agents to complete the transfer.
        """
        return (
            "Emergency escalation logged. Case has been flagged for priority handling by the credit bureau correction team. Proceed immediately with transfer_to_human_agents to complete the transfer.\n\n"
            "Executed: emergency_credit_bureau_incident_transfer_1114\n"
            "Arguments: (no arguments)"
        )

    @is_discoverable_tool(ToolType.WRITE)
    def file_credit_card_transaction_dispute_4829(
        self,
        transaction_id: str,
        card_action: str,
        card_last_4_digits: str,
        full_name: str,
        user_id: str,
        phone: str,
        email: str,
        address: str,
        contacted_merchant: bool,
        purchase_date: str,
        issue_noticed_date: str,
        dispute_reason: str,
        resolution_requested: str,
        eligible_for_provisional_credit: bool,
        partial_refund_amount: Optional[float] = None,
    ) -> str:
        """File a formal dispute for a credit card transaction.

        Args:
            transaction_id (string): The unique identifier for the transaction being disputed
            card_action (string): Flag indicating the card's status. Must be one of: 'keep_active' (card remains active, dispute only), 'cancel_and_reissue' (card is being cancelled and replaced). This is for record-keeping only and does NOT order a replacement card.
            card_last_4_digits (string): Last 4 digits of the credit card number
            full_name (string): Full legal name of the cardholder
            user_id (string): The user's unique identifier in the system
            phone (string): Contact phone number
            email (string): Contact email address
            address (string): Contact mailing address
            contacted_merchant (boolean): Whether the user attempted to resolve the issue with the merchant first
            purchase_date (string): Date when the purchase was made, format MM/DD/YYYY
            issue_noticed_date (string): Date when the user noticed the issue, format MM/DD/YYYY
            dispute_reason (string): Reason for the dispute. Must be one of: 'unauthorized_fraudulent_charge', 'duplicate_charge', 'incorrect_amount', 'goods_services_not_received', 'goods_services_not_as_described', 'canceled_subscription_still_charging', 'refund_never_processed'
            resolution_requested (string): Resolution being requested. Must be one of: 'full_refund', 'partial_refund'
            eligible_for_provisional_credit (boolean): Whether the user is eligible for provisional credit
            partial_refund_amount (number, optional): Amount requested for partial refund (required only if resolution_requested is 'partial_refund')

        Returns:
            Credit card transaction dispute filed successfully. A case has been opened and will be reviewed within 10 business days.
        """
        if not transaction_id or not user_id:
            return "Error: Missing required parameters."

        # Validate card_action
        valid_card_actions = ["keep_active", "cancel_and_reissue"]
        if card_action not in valid_card_actions:
            return f"Error: Invalid card_action. Must be one of: {valid_card_actions}"

        # Validate dispute_reason
        valid_reasons = [
            "unauthorized_fraudulent_charge",
            "duplicate_charge",
            "incorrect_amount",
            "goods_services_not_received",
            "goods_services_not_as_described",
            "canceled_subscription_still_charging",
            "refund_never_processed",
        ]
        if dispute_reason not in valid_reasons:
            return f"Error: Invalid dispute_reason. Must be one of: {valid_reasons}"

        # Validate resolution_requested
        valid_resolutions = ["full_refund", "partial_refund"]
        if resolution_requested not in valid_resolutions:
            return f"Error: Invalid resolution_requested. Must be one of: {valid_resolutions}"

        # If partial refund, check for amount
        if resolution_requested == "partial_refund":
            if partial_refund_amount is None:
                return "Error: partial_refund_amount is required when resolution_requested is 'partial_refund'."

        # Generate a deterministic dispute ID
        dispute_id = generate_dispute_id(user_id, transaction_id)

        # Create the dispute record
        dispute_record = {
            "dispute_id": dispute_id,
            "transaction_id": transaction_id,
            "user_id": user_id,
            "card_action": card_action,
            "card_last_4_digits": card_last_4_digits,
            "full_name": full_name,
            "phone": phone,
            "email": email,
            "address": address,
            "contacted_merchant": contacted_merchant,
            "purchase_date": purchase_date,
            "issue_noticed_date": issue_noticed_date,
            "dispute_reason": dispute_reason,
            "resolution_requested": resolution_requested,
            "partial_refund_amount": partial_refund_amount,
            "eligible_for_provisional_credit": eligible_for_provisional_credit,
            "provisional_credit_given": eligible_for_provisional_credit,
            "submitted_at": get_today_str(),
            "status": "SUBMITTED",
        }

        # Add to the transaction_disputes table
        success = add_to_db(
            "transaction_disputes", dispute_id, dispute_record, db=self.db
        )

        if not success:
            return "Error: Dispute may have already been filed for this transaction."

        # Build response
        result_parts = [
            "Credit card transaction dispute filed successfully. A case has been opened and will be reviewed within 10 business days.",
            "",
            f"Executed: file_credit_card_transaction_dispute_4829",
            f"Dispute ID: {dispute_id}",
            f"Transaction: {transaction_id}",
            f"Reason: {dispute_reason.replace('_', ' ').title()}",
            f"Resolution Requested: {resolution_requested.replace('_', ' ').title()}",
        ]

        if partial_refund_amount:
            result_parts.append(f"Partial Refund Amount: ${partial_refund_amount:.2f}")

        if eligible_for_provisional_credit:
            result_parts.append(
                "Provisional Credit: ELIGIBLE - Credit will be applied within 2 business days."
            )
        else:
            result_parts.append("Provisional Credit: Not eligible at this time.")

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def file_debit_card_transaction_dispute_6281(
        self,
        transaction_id: str,
        account_id: str,
        card_id: str,
        user_id: str,
        dispute_category: str,
        transaction_date: str,
        discovery_date: str,
        disputed_amount: float,
        transaction_type: str,
        card_in_possession: bool,
        pin_compromised: str,
        contacted_merchant: bool,
        police_report_filed: bool,
        written_statement_provided: bool,
        provisional_credit_eligible: bool,
        customer_max_liability_amount: float,
        card_action: str,
    ) -> str:
        """File a formal dispute for a debit card transaction under Regulation E. Debit card disputes affect actual bank funds and have different liability rules based on reporting timing.

        Args:
            transaction_id (string): The unique identifier for the transaction being disputed
            account_id (string): The checking account ID linked to the debit card
            card_id (string): The debit card ID
            user_id (string): The user's unique identifier in the system
            dispute_category (string): Category of the dispute. Must be one of: 'unauthorized_transaction', 'atm_cash_discrepancy', 'atm_deposit_not_credited', 'duplicate_charge', 'incorrect_amount', 'goods_services_not_received', 'recurring_charge_after_cancellation', 'card_present_fraud', 'card_not_present_fraud'
            transaction_date (string): Date when the transaction occurred, format MM/DD/YYYY
            discovery_date (string): Date when the user first noticed the issue, format MM/DD/YYYY
            disputed_amount (number): The dollar amount being disputed
            transaction_type (string): Type of transaction. Must be one of: 'pin_purchase', 'signature_purchase', 'online_purchase', 'atm_withdrawal', 'atm_deposit', 'recurring_payment', 'person_to_person'
            card_in_possession (boolean): Whether the customer still has their physical debit card in their possession
            pin_compromised (string): Whether the customer's PIN may have been compromised. Must be one of: 'yes_shared', 'yes_observed', 'no', 'unknown'
            contacted_merchant (boolean): Whether the user attempted to resolve the issue with the merchant first
            police_report_filed (boolean): Whether a police report has been filed (recommended for fraud over $500)
            written_statement_provided (boolean): Whether the customer has provided a written statement describing what happened (required for Reg E provisional credit)
            provisional_credit_eligible (boolean): Whether the user is eligible for provisional credit based on Debit Card Provisional Credit Guidelines
            customer_max_liability_amount (number): The maximum dollar amount the customer could be liable for based on Regulation E reporting timing rules and the disputed amount. Use -1 for unlimited liability.
            card_action (string): Action to take on the card. Must be one of: 'keep_active', 'freeze_pending_investigation', 'close_and_reissue'

        Returns:
            Debit card transaction dispute filed successfully. A case has been opened and provisional credit determination has been recorded.
        """
        if not all(
            [
                transaction_id,
                account_id,
                card_id,
                user_id,
                dispute_category,
                transaction_date,
                discovery_date,
                disputed_amount,
                transaction_type,
                pin_compromised,
                card_action,
            ]
        ):
            return "Error: Missing required parameters."

        if customer_max_liability_amount is None:
            return "Error: customer_max_liability_amount is required."

        if (
            card_in_possession is None
            or contacted_merchant is None
            or police_report_filed is None
            or written_statement_provided is None
        ):
            return "Error: card_in_possession, contacted_merchant, police_report_filed, and written_statement_provided are required boolean fields."

        # Validate dispute_category
        valid_categories = [
            "unauthorized_transaction",
            "atm_cash_discrepancy",
            "atm_deposit_not_credited",
            "duplicate_charge",
            "incorrect_amount",
            "goods_services_not_received",
            "recurring_charge_after_cancellation",
            "card_present_fraud",
            "card_not_present_fraud",
        ]
        if dispute_category not in valid_categories:
            return (
                f"Error: Invalid dispute_category. Must be one of: {valid_categories}"
            )

        # Validate transaction_type
        valid_transaction_types = [
            "pin_purchase",
            "signature_purchase",
            "online_purchase",
            "atm_withdrawal",
            "atm_deposit",
            "recurring_payment",
            "person_to_person",
        ]
        if transaction_type not in valid_transaction_types:
            return f"Error: Invalid transaction_type. Must be one of: {valid_transaction_types}"

        # Validate pin_compromised
        valid_pin_statuses = ["yes_shared", "yes_observed", "no", "unknown"]
        if pin_compromised not in valid_pin_statuses:
            return (
                f"Error: Invalid pin_compromised. Must be one of: {valid_pin_statuses}"
            )

        # Validate card_action
        valid_card_actions = [
            "keep_active",
            "freeze_pending_investigation",
            "close_and_reissue",
        ]
        if card_action not in valid_card_actions:
            return f"Error: Invalid card_action. Must be one of: {valid_card_actions}"

        # Validate disputed_amount is positive
        if disputed_amount <= 0:
            return "Error: disputed_amount must be a positive number."

        # Generate a deterministic dispute ID
        dispute_id = generate_dispute_id(user_id, transaction_id)

        # Determine fraud-related fields
        is_fraud_category = dispute_category in [
            "unauthorized_transaction",
            "card_present_fraud",
            "card_not_present_fraud",
        ]
        pin_shared_voluntarily = pin_compromised == "yes_shared"

        # Create the debit card dispute record
        dispute_record = {
            "dispute_id": dispute_id,
            "transaction_id": transaction_id,
            "account_id": account_id,
            "card_id": card_id,
            "user_id": user_id,
            "dispute_category": dispute_category,
            "transaction_date": transaction_date,
            "discovery_date": discovery_date,
            "disputed_amount": disputed_amount,
            "transaction_type": transaction_type,
            "card_in_possession": card_in_possession,
            "pin_compromised": pin_compromised,
            "contacted_merchant": contacted_merchant,
            "police_report_filed": police_report_filed,
            "written_statement_provided": written_statement_provided,
            "provisional_credit_eligible": provisional_credit_eligible,
            "provisional_credit_issued": provisional_credit_eligible,
            "provisional_credit_amount": disputed_amount
            if provisional_credit_eligible
            else None,
            "customer_max_liability_amount": customer_max_liability_amount,
            "card_action": card_action,
            "is_fraud_category": is_fraud_category,
            "pin_shared_voluntarily": pin_shared_voluntarily,
            "submitted_at": get_today_str(),
            "status": "OPEN",
        }

        # Add to the debit_card_disputes table
        success = add_to_db(
            "debit_card_disputes", dispute_id, dispute_record, db=self.db
        )

        if not success:
            return "Error: Dispute may have already been filed for this transaction."

        # Build response
        result_parts = [
            f"Dispute ID: {dispute_id}",
            f"Transaction: {transaction_id}",
            f"Account: {account_id}",
            f"Category: {dispute_category.replace('_', ' ').title()}",
            f"Disputed Amount: ${disputed_amount:.2f}",
            f"Card Action: {card_action.replace('_', ' ').title()}",
        ]

        if provisional_credit_eligible:
            result_parts.append(
                f"Provisional Credit: ISSUED - ${disputed_amount:.2f} credited within 10 business days per Regulation E."
            )
        else:
            result_parts.append(
                "Provisional Credit: Not eligible - see Debit Card Provisional Credit Guidelines for details."
            )

        if pin_shared_voluntarily:
            result_parts.append(
                "WARNING: Customer indicated PIN was shared voluntarily. This may affect liability determination."
            )

        if is_fraud_category and not police_report_filed and disputed_amount > 500:
            result_parts.append(
                "RECOMMENDATION: For fraud disputes over $500, filing a police report is recommended."
            )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def set_debit_card_recurring_block_7382(
        self,
        card_id: str,
        block_recurring: bool,
    ) -> str:
        """Block or unblock all recurring payments on a debit card. When blocked, all recurring/subscription charges will be declined. One-time purchases are not affected.

        Args:
            card_id (string): The debit card ID to update
            block_recurring (boolean): True to block all recurring payments, False to unblock/allow recurring payments

        Returns:
            Debit card recurring payment settings updated successfully.
        """
        if not card_id:
            return "Error: Missing required parameter: card_id"

        # Check if card exists
        debit_card = self.db.debit_cards.data.get(card_id)
        if not debit_card:
            return f"Error: Debit card '{card_id}' not found."

        # Check card status - can only update ACTIVE cards
        card_status = debit_card.get("status", "").upper()
        if card_status != "ACTIVE":
            return f"Error: Cannot update recurring block settings for a card with status '{card_status}'. Card must be ACTIVE."

        # Update the recurring_blocked field
        debit_card["recurring_blocked"] = block_recurring

        if block_recurring:
            return (
                f"Recurring payments BLOCKED for debit card {card_id}.\n"
                f"All recurring/subscription charges will be declined.\n"
                f"One-time purchases are not affected.\n"
                f"This change takes effect within 24 hours."
            )
        else:
            return (
                f"Recurring payments UNBLOCKED for debit card {card_id}.\n"
                f"Recurring/subscription charges will now be allowed.\n"
                f"This change takes effect within 24 hours."
            )

    @is_discoverable_tool(ToolType.READ)
    def get_debit_dispute_status_7483(self, user_id: str) -> str:
        """Retrieve a user's debit card dispute history from the debit_card_disputes table. Returns all debit card disputes filed by the user, including dispute IDs, categories, amounts, statuses, and provisional credit information.

        Args:
            user_id (string): The user's unique identifier in the system

        Returns:
            Debit card dispute history retrieved successfully.
        """
        if not user_id:
            return "Error: Missing required parameter: user_id"

        debit_disputes_result = query_database_tool(
            "debit_card_disputes", f'{{"user_id": "{user_id}"}}', db=self.db
        )

        result_parts = [
            "Debit card dispute history retrieved successfully.",
            "",
            f"Executed: get_debit_dispute_status_7483",
            f"Debit card dispute history for user {user_id}:",
        ]

        has_disputes = (
            "No records found" not in debit_disputes_result
            and "No results found" not in debit_disputes_result
        )

        if has_disputes:
            result_parts.append(debit_disputes_result)
        else:
            result_parts.append("\nNo debit card disputes found for this user.")

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.READ)
    def get_atm_deposit_images_8473(self, transaction_id: str) -> str:
        """Retrieve ATM deposit envelope/check images for a specific ATM deposit transaction.

        Args:
            transaction_id (string): The transaction ID of the ATM deposit to retrieve images for

        Returns:
            ATM deposit images retrieved successfully.
        """
        if not transaction_id:
            return "Error: Missing required parameter: transaction_id"

        # Verify the transaction exists
        if transaction_id not in self.db.bank_account_transaction_history.data:
            return f"Error: Transaction '{transaction_id}' not found."

        txn = self.db.bank_account_transaction_history.data[transaction_id]

        # Verify it's an ATM deposit transaction
        if txn.get("type") != "atm_deposit":
            return f"Error: Transaction '{transaction_id}' is not an ATM deposit. This tool only works for ATM deposit transactions."

        # Check if it's a Rho-Bank ATM
        description = txn.get("description", "")
        if (
            "RHO-BANK" not in description.upper()
            and "RHOBANK" not in description.upper()
        ):
            return f"Error: Transaction '{transaction_id}' is from a third-party ATM. Deposit images are only available for Rho-Bank ATM deposits. For third-party ATM disputes, a chargeback request must be submitted to the ATM network."

        # Hardcoded deposit image descriptions for specific test transaction IDs
        deposit_image_data = {
            "btxn_834027370c20": {
                "atm_id": "ATM #3921",
                "deposit_date": txn.get("date", "Unknown"),
                "envelope_contents": """
=== ATM DEPOSIT ENVELOPE SCAN ===
Envelope ID: ENV-2025-3921-00923
Deposit Time: 3:47 PM CST
ATM Location: Rho-Bank ATM #3921, Cedar Lane Branch, Austin, TX

--- ENVELOPE CONTENTS ---
Item 1: Personal Check
  - Check Number: 7284
  - Drawn On: First Texas Bank
  - Payee: Derek Yamamoto
  - Amount: $875.00
  - Memo: "October rent refund"
  - Signature: Present and legible
  - Date on Check: 11/03/2025

Item 2: Cash
  - Denomination breakdown:
    * 2 x $100 bills = $200.00
    * 3 x $20 bills = $60.00
  - Total Cash: $260.00

--- DEPOSIT SUMMARY ---
Check Total: $875.00
Cash Total: $260.00
GRAND TOTAL: $1,135.00

--- MACHINE RECORD ---
Amount Recorded by ATM: $385.00
DISCREPANCY DETECTED: $750.00 difference

--- IMAGE QUALITY ---
Envelope scan: CLEAR
Check front: CLEAR
Check back (endorsement): CLEAR - endorsed "For Deposit Only - Derek Yamamoto"
Cash image: CLEAR - bills visible and countable
""",
                "verification_notes": "Images clearly show deposit contents totaling $1,135.00. ATM machine recorded only $385.00. Discrepancy of $750.00 confirmed via image review.",
            },
            "btxn_test_deposit_001": {
                "atm_id": "ATM #3921",
                "deposit_date": txn.get("date", "Unknown"),
                "envelope_contents": """
=== ATM DEPOSIT ENVELOPE SCAN ===
Envelope ID: ENV-2025-3921-00847
Deposit Time: 2:34 PM EST
ATM Location: Rho-Bank ATM #3921, 742 Oak Avenue, Portland, OR

--- ENVELOPE CONTENTS ---
Item 1: Personal Check
  - Check Number: 4821
  - Drawn On: First National Bank
  - Payee: Linda Patterson
  - Amount: $875.00
  - Memo: "October rent refund"
  - Signature: Present and legible
  - Date on Check: 11/05/2025

Item 2: Cash
  - Denomination breakdown:
    * 2 x $100 bills = $200.00
    * 3 x $20 bills = $60.00
  - Total Cash: $260.00

--- DEPOSIT SUMMARY ---
Check Total: $875.00
Cash Total: $260.00
GRAND TOTAL: $1,135.00

--- MACHINE RECORD ---
Amount Recorded by ATM: $385.00
DISCREPANCY DETECTED: $750.00 difference

--- IMAGE QUALITY ---
Envelope scan: CLEAR
Check front: CLEAR
Check back (endorsement): CLEAR - endorsed "For Deposit Only - Linda Patterson"
Cash image: CLEAR - bills visible and countable
""",
                "verification_notes": "Images clearly show deposit contents totaling $1,135.00. ATM machine recorded only $385.00. Discrepancy of $750.00 confirmed via image review.",
            },
            "btxn_test_atm_dep_partial": {
                "atm_id": "ATM #5847",
                "deposit_date": txn.get("date", "Unknown"),
                "envelope_contents": """
=== ATM DEPOSIT ENVELOPE SCAN ===
Envelope ID: ENV-2025-5847-00293
Deposit Time: 10:15 AM EST

--- ENVELOPE CONTENTS ---
Item 1: Personal Check
  - Check Number: 7392
  - Amount: $500.00

--- DEPOSIT SUMMARY ---
Check Total: $500.00
Cash Total: $0.00
GRAND TOTAL: $500.00

--- MACHINE RECORD ---
Amount Recorded by ATM: $500.00
No discrepancy detected.
""",
                "verification_notes": "Images confirm deposit matches recorded amount. No discrepancy found.",
            },
        }

        if transaction_id in deposit_image_data:
            data = deposit_image_data[transaction_id]
            result = f"""ATM Deposit Image Retrieval Results
=====================================
Transaction ID: {transaction_id}
ATM: {data["atm_id"]}
Deposit Date: {data["deposit_date"]}

{data["envelope_contents"]}

--- VERIFICATION NOTES ---
{data["verification_notes"]}
"""
            return result
        else:
            return f"""ATM Deposit Image Retrieval Results
=====================================
Transaction ID: {transaction_id}
ATM: {description}
Deposit Date: {txn.get("date", "Unknown")}
Amount Recorded: ${abs(txn.get("amount", 0)):.2f}

--- IMAGE STATUS ---
Status: IMAGES NOT AVAILABLE
Reason: Deposit images for this transaction have either expired (older than 90 days) or were not captured by the ATM system.

For deposits without available images, the dispute will proceed based on customer statement and ATM journal records only. Investigation timeline may be extended.
"""

    @is_discoverable_tool(ToolType.WRITE)
    def order_replacement_credit_card_7291(
        self,
        credit_card_account_id: str,
        user_id: str,
        shipping_address: str,
        reason: str,
        expedited_shipping: bool = False,
    ) -> str:
        """Order a replacement credit card for a customer. The old card will be automatically cancelled when the replacement is ordered.

        Args:
            credit_card_account_id (string): The credit card account ID for the card being replaced
            user_id (string): The user's unique identifier in the system
            shipping_address (string): Full shipping address where the new card should be sent
            reason (string): Reason for replacement. Must be one of: 'fraud_suspected', 'lost', 'stolen', 'damaged', 'expired', 'other'
            expedited_shipping (boolean, optional): Whether to use expedited shipping (2-3 business days instead of 7-10)

        Returns:
            Replacement credit card order placed successfully. The old card has been cancelled for security.
        """
        if (
            not credit_card_account_id
            or not user_id
            or not shipping_address
            or not reason
        ):
            return "Error: Missing required parameters (credit_card_account_id, user_id, shipping_address, reason)."

        valid_reasons = [
            "fraud_suspected",
            "lost",
            "stolen",
            "damaged",
            "expired",
            "other",
        ]
        if reason not in valid_reasons:
            return f"Error: Invalid reason. Must be one of: {valid_reasons}"

        # Verify credit card account exists
        result = query_database_tool(
            "credit_card_accounts",
            f'{{"account_id": "{credit_card_account_id}"}}',
            db=self.db,
        )

        if "No results found" in result or "No records found" in result:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        # Generate order ID
        order_id = generate_credit_card_order_id(
            credit_card_account_id, user_id, reason
        )

        # Create the replacement order record
        today = get_today_str()
        order_record = {
            "order_id": order_id,
            "credit_card_account_id": credit_card_account_id,
            "user_id": user_id,
            "shipping_address": shipping_address,
            "reason": reason,
            "expedited_shipping": expedited_shipping,
            "order_date": today,
            "status": "ORDERED",
            "old_card_cancelled": True,
        }

        success = add_to_db("credit_card_orders", order_id, order_record, db=self.db)

        if not success:
            return (
                "Error: Order may have already been placed for this card replacement."
            )

        # Cancel the old credit card for security
        if credit_card_account_id in self.db.credit_card_accounts.data:
            self.db.credit_card_accounts.data[credit_card_account_id]["status"] = (
                "CLOSED"
            )
            self.db.credit_card_accounts.data[credit_card_account_id]["closed_date"] = (
                get_today_str()
            )

        shipping_method = "Expedited" if expedited_shipping else "Standard"
        expected_delivery = (
            "2-3 business days" if expedited_shipping else "7-10 business days"
        )

        result_parts = [
            f"Order ID: {order_id}",
            f"Card Account: {credit_card_account_id}",
            f"Reason: {reason.replace('_', ' ').title()}",
            f"Shipping Address: {shipping_address}",
            f"Shipping Method: {shipping_method}",
            f"Expected Delivery: {expected_delivery}",
            "",
            "The old card has been cancelled for security. The new card will have the same account number but a new card number and CVV.",
        ]

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.READ)
    def get_user_dispute_history_7291(self, user_id: str) -> str:
        """Retrieve a user's credit card transaction dispute history from the transaction_disputes table. Returns all credit card transaction disputes filed by the user, including dispute IDs, transaction IDs, dispute reasons, statuses, and submission dates.

        Args:
            user_id (string): The user's unique identifier in the system

        Returns:
            User transaction dispute history retrieved successfully.
        """
        if not user_id:
            return "Error: Missing required parameter: user_id"

        transaction_disputes_result = query_database_tool(
            "transaction_disputes", f'{{"user_id": "{user_id}"}}', db=self.db
        )

        result_parts = [
            "User transaction dispute history retrieved successfully.",
            "",
            f"Executed: get_user_dispute_history_7291",
            f"Transaction dispute history for user {user_id}:",
        ]

        has_disputes = (
            "No records found" not in transaction_disputes_result
            and "No results found" not in transaction_disputes_result
        )

        if has_disputes:
            result_parts.append(transaction_disputes_result)
        else:
            result_parts.append("\nNo transaction disputes found for this user.")

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.READ)
    def get_pending_replacement_orders_5765(self, credit_card_account_id: str) -> str:
        """Check if a credit card account has any pending replacement card orders.

        Args:
            credit_card_account_id (string): The credit card account ID to check for pending replacement orders

        Returns:
            Pending replacement orders check completed.
        """
        if not credit_card_account_id:
            return "Error: Missing required parameter: credit_card_account_id"

        orders_result = query_database_tool(
            "credit_card_orders",
            f'{{"credit_card_account_id": "{credit_card_account_id}"}}',
            db=self.db,
        )

        result_parts = [
            "Pending replacement orders check completed.",
            "",
            f"Executed: get_pending_replacement_orders_5765",
            f"Replacement orders for credit card account {credit_card_account_id}:",
        ]

        has_orders = (
            "No records found" not in orders_result
            and "No results found" not in orders_result
        )

        if has_orders:
            result_parts.append(orders_result)
        else:
            result_parts.append(
                "\nNo pending replacement orders found for this credit card account."
            )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def log_credit_card_closure_reason_4521(
        self,
        credit_card_account_id: str,
        user_id: str,
        closure_reason: str,
    ) -> str:
        """Log the reason why a customer wants to close their credit card account.

        Args:
            credit_card_account_id (string): The credit card account ID the customer wants to close
            user_id (string): The user's unique identifier in the system
            closure_reason (string): Reason for closure. Must be one of: 'annual_fee', 'not_using_card', 'found_better_card', 'unhappy_with_rewards', 'simplifying_finances', 'negative_experience', 'other'

        Returns:
            Closure reason logged successfully.
        """
        if not credit_card_account_id or not user_id or not closure_reason:
            return "Error: Missing required parameters."

        valid_reasons = [
            "annual_fee",
            "not_using_card",
            "found_better_card",
            "unhappy_with_rewards",
            "simplifying_finances",
            "negative_experience",
            "other",
        ]
        if closure_reason not in valid_reasons:
            return f"Error: Invalid closure_reason. Must be one of: {valid_reasons}"

        record_id = generate_closure_reason_id(credit_card_account_id, user_id)

        closure_record = {
            "record_id": record_id,
            "credit_card_account_id": credit_card_account_id,
            "user_id": user_id,
            "closure_reason": closure_reason,
            "logged_at": get_today_str(),
            "status": "LOGGED",
        }

        add_to_db("credit_card_closure_reasons", record_id, closure_record, db=self.db)

        return (
            f"Closure reason logged successfully.\n\n"
            f"Executed: log_credit_card_closure_reason_4521\n"
            f"Arguments: {json.dumps({'credit_card_account_id': credit_card_account_id, 'user_id': user_id, 'closure_reason': closure_reason}, indent=2)}\n"
            f"Closure reason '{closure_reason}' logged for account {credit_card_account_id}."
        )

    @is_discoverable_tool(ToolType.READ)
    def get_closure_reason_history_8293(self, credit_card_account_id: str) -> str:
        """Retrieve the closure reason history for a specific credit card account.

        Args:
            credit_card_account_id (string): The credit card account ID to check for previous closure attempts

        Returns:
            Closure reason history retrieved successfully.
        """
        if not credit_card_account_id:
            return "Error: Missing required parameter: credit_card_account_id"

        closure_reasons_result = query_database_tool(
            "credit_card_closure_reasons",
            f'{{"credit_card_account_id": "{credit_card_account_id}"}}',
            db=self.db,
        )

        result_parts = [
            "Closure reason history retrieved successfully.",
            "",
            f"Executed: get_closure_reason_history_8293",
            f"Closure reason history for credit card account {credit_card_account_id}:",
        ]

        has_records = (
            "No records found" not in closure_reasons_result
            and "No results found" not in closure_reasons_result
        )

        if has_records:
            result_parts.append(closure_reasons_result)
        else:
            result_parts.append(
                "\nNo closure reason records found for this credit card account."
            )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def apply_statement_credit_8472(
        self,
        user_id: str,
        credit_card_account_id: str,
        amount: float,
        reason: str,
    ) -> str:
        """Apply a statement credit to a customer's credit card account.

        Args:
            user_id (string): The user's unique identifier in the system
            credit_card_account_id (string): The credit card account ID to apply the credit to
            amount (number): The credit amount in dollars (e.g., 25.00 for a $25 credit)
            reason (string): Reason for the statement credit. Must be one of: 'goodwill_adjustment', 'promotional_credit', 'annual_fee_reversal', 'late_fee_reversal', 'interest_charge_reversal', 'dispute_resolution', 'price_match', 'retention_offer', 'error_correction', 'other'

        Returns:
            Statement credit applied successfully.
        """
        if not user_id or not credit_card_account_id or amount is None or not reason:
            return "Error: Missing required parameters (user_id, credit_card_account_id, amount, reason)."

        if amount <= 0:
            return "Error: Credit amount must be positive."

        valid_reasons = [
            "goodwill_adjustment",
            "promotional_credit",
            "annual_fee_reversal",
            "late_fee_reversal",
            "interest_charge_reversal",
            "dispute_resolution",
            "price_match",
            "retention_offer",
            "error_correction",
            "other",
        ]
        if reason not in valid_reasons:
            return f"Error: Invalid reason. Must be one of: {valid_reasons}"

        result = query_database_tool(
            "credit_card_accounts",
            f'{{"account_id": "{credit_card_account_id}"}}',
            db=self.db,
        )

        if "No results found" in result or "No records found" in result:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        transaction_id = generate_transaction_id(
            user_id, "STATEMENT_CREDIT", reason, amount, "Statement Credit"
        )

        today = get_today_str()

        credit_record = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "credit_card_account_id": credit_card_account_id,
            "credit_card_type": "N/A",
            "merchant_name": "Rho-Bank Statement Credit",
            "transaction_amount": f"-${amount:.2f}",
            "transaction_date": today,
            "category": "Statement Credit",
            "status": "COMPLETED",
            "rewards_earned": "0 points",
            "credit_reason": reason,
        }

        success = add_to_db(
            "credit_card_transaction_history", transaction_id, credit_record, db=self.db
        )

        if not success:
            return f"Error: Failed to apply statement credit. Transaction ID '{transaction_id}' may already exist."

        return (
            f"Statement credit applied successfully.\n\n"
            f"Executed: apply_statement_credit_8472\n"
            f"  - Transaction ID: {transaction_id}\n"
            f"  - User ID: {user_id}\n"
            f"  - Account: {credit_card_account_id}\n"
            f"  - Credit Amount: ${amount:.2f}\n"
            f"  - Reason: {reason.replace('_', ' ').title()}\n"
            f"  - Date: {today}"
        )

    @is_discoverable_tool(ToolType.WRITE)
    def apply_credit_card_account_flag_6147(
        self,
        credit_card_account_id: str,
        user_id: str,
        flag_type: str,
        expiration_date: str,
        reason: str,
    ) -> str:
        """Apply a flag to a customer's credit card account. Flags can include annual fee waivers, promotional APR rates, rewards bonuses, or other account-level modifiers. Each flag has an effective date and expiration date.

        Args:
            credit_card_account_id (string): The credit card account ID to apply the flag to
            user_id (string): The user's unique identifier in the system
            flag_type (string): Type of flag to apply. Must be one of: 'annual_fee_waived', 'promotional_apr', 'rewards_bonus', 'other'
            expiration_date (string): Date when the flag expires (MM/DD/YYYY format)
            reason (string): Reason for applying this flag. Must be one of: 'retention_offer', 'loyalty_benefit', 'promotional', 'error_correction', 'other'

        Returns:
            Account flag applied successfully.
        """
        if (
            not credit_card_account_id
            or not user_id
            or not flag_type
            or not expiration_date
            or not reason
        ):
            return "Error: Missing required parameters (credit_card_account_id, user_id, flag_type, expiration_date, reason)."

        valid_flag_types = [
            "annual_fee_waived",
            "promotional_apr",
            "rewards_bonus",
            "other",
        ]
        if flag_type not in valid_flag_types:
            return f"Error: Invalid flag_type. Must be one of: {valid_flag_types}"

        valid_reasons = [
            "retention_offer",
            "loyalty_benefit",
            "promotional",
            "error_correction",
            "other",
        ]
        if reason not in valid_reasons:
            return f"Error: Invalid reason. Must be one of: {valid_reasons}"

        # Verify the credit card account exists
        result = query_database_tool(
            "credit_card_accounts",
            f'{{"account_id": "{credit_card_account_id}"}}',
            db=self.db,
        )

        if "No results found" in result or "No records found" in result:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        flag_id = generate_account_flag_id(
            credit_card_account_id, flag_type, expiration_date
        )

        today = get_today_str()

        flag_record = {
            "flag_id": flag_id,
            "credit_card_account_id": credit_card_account_id,
            "user_id": user_id,
            "flag_type": flag_type,
            "effective_date": today,
            "expiration_date": expiration_date,
            "reason": reason,
            "applied_at": today,
            "status": "ACTIVE",
        }

        success = add_to_db(
            "credit_card_account_flags", flag_id, flag_record, db=self.db
        )

        if not success:
            return f"Error: Failed to apply account flag. Flag ID '{flag_id}' may already exist."

        flag_type_display = flag_type.replace("_", " ").title()
        reason_display = reason.replace("_", " ").title()

        return (
            f"Account flag applied successfully!\n"
            f"  - Flag ID: {flag_id}\n"
            f"  - Account: {credit_card_account_id}\n"
            f"  - User ID: {user_id}\n"
            f"  - Flag Type: {flag_type_display}\n"
            f"  - Effective Date: {today}\n"
            f"  - Expiration Date: {expiration_date}\n"
            f"  - Reason: {reason_display}"
        )

    @is_discoverable_tool(ToolType.WRITE)
    def close_credit_card_account_7834(
        self,
        credit_card_account_id: str,
        user_id: str,
    ) -> str:
        """Close a customer's credit card account permanently.

        Args:
            credit_card_account_id (string): The credit card account ID to close
            user_id (string): The user's unique identifier in the system

        Returns:
            Credit card account closed successfully.
        """
        if not credit_card_account_id or not user_id:
            return (
                "Error: Missing required parameters (credit_card_account_id, user_id)."
            )

        result = query_database_tool(
            "credit_card_accounts",
            f'{{"account_id": "{credit_card_account_id}"}}',
            db=self.db,
        )

        if "No results found" in result or "No records found" in result:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        success, updated_record = update_record_in_db(
            "credit_card_accounts",
            db=self.db,
            record_id=credit_card_account_id,
            updates={
                "status": "CLOSED",
                "closed_date": get_today_str(),
                "closed_by": user_id,
            },
        )

        if not success:
            return f"Error: Failed to close credit card account '{credit_card_account_id}'."

        return (
            f"Credit card account closed successfully.\n\n"
            f"Executed: close_credit_card_account_7834\n"
            f"Arguments: {json.dumps({'credit_card_account_id': credit_card_account_id, 'user_id': user_id}, indent=2)}\n"
            f"Account {credit_card_account_id} has been closed."
        )

    @is_discoverable_tool(ToolType.WRITE)
    def pay_credit_card_from_checking_9182(
        self,
        user_id: str,
        checking_account_id: str,
        credit_card_account_id: str,
        amount: float,
    ) -> str:
        """Pay off a credit card balance using funds from the customer's Rho-Bank checking account. This deducts the specified amount from the checking account and reduces the credit card balance by the same amount.

        Args:
            user_id (string): The customer's unique identifier in the system
            checking_account_id (string): The ID of the Rho-Bank checking account to debit funds from
            credit_card_account_id (string): The ID of the credit card account to apply the payment to
            amount (number): The payment amount in dollars. Must be a positive number.

        Returns:
            Credit card payment processed successfully.
        """
        if (
            not user_id
            or not checking_account_id
            or not credit_card_account_id
            or amount is None
        ):
            return "Error: Missing required parameters (user_id, checking_account_id, credit_card_account_id, amount)."

        # Validate amount is positive
        try:
            amount = float(amount)
            if amount <= 0:
                return "Error: Payment amount must be a positive number."
        except (ValueError, TypeError):
            return "Error: Invalid payment amount. Must be a positive number."

        # Verify checking account exists and belongs to the user
        if checking_account_id not in self.db.accounts.data:
            return f"Error: Checking account '{checking_account_id}' not found."

        checking_account = self.db.accounts.data[checking_account_id]
        if checking_account.get("user_id") != user_id:
            return f"Error: Checking account '{checking_account_id}' does not belong to user '{user_id}'."

        if checking_account.get("class") != "checking":
            return f"Error: Account '{checking_account_id}' is not a checking account."

        # Get checking account balance
        current_balance = _get_account_balance(checking_account)

        if amount > current_balance:
            return f"Error: Insufficient funds in checking account. Available balance: ${current_balance:.2f}, requested payment: ${amount:.2f}."

        # Verify credit card account exists and belongs to the user
        if credit_card_account_id not in self.db.credit_card_accounts.data:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        cc_account = self.db.credit_card_accounts.data[credit_card_account_id]
        if cc_account.get("user_id") != user_id:
            return f"Error: Credit card account '{credit_card_account_id}' does not belong to user '{user_id}'."

        # Get credit card balance
        cc_balance_str = cc_account.get("current_balance", "$0.00")
        try:
            cc_balance = float(str(cc_balance_str).replace("$", "").replace(",", ""))
        except (ValueError, TypeError):
            cc_balance = 0.0

        if amount > cc_balance:
            return f"Error: Payment amount (${amount:.2f}) exceeds credit card balance (${cc_balance:.2f}). Please specify an amount up to the outstanding balance."

        # Process the payment - deduct from checking, reduce CC balance
        new_checking_balance = current_balance - amount
        new_cc_balance = cc_balance - amount

        # Update the database
        self.db.accounts.data[checking_account_id]["current_holdings"] = (
            f"{new_checking_balance:.2f}"
        )
        self.db.credit_card_accounts.data[credit_card_account_id]["current_balance"] = (
            f"${new_cc_balance:.2f}"
        )

        return (
            f"Payment processed successfully!\n"
            f"  - Payment Amount: ${amount:.2f}\n"
            f"  - From Checking Account: {checking_account_id}\n"
            f"  - To Credit Card Account: {credit_card_account_id}\n"
            f"  - New Checking Balance: ${new_checking_balance:.2f}\n"
            f"  - New Credit Card Balance: ${new_cc_balance:.2f}\n"
            f"The payment has been applied immediately."
        )

    @is_discoverable_tool(ToolType.WRITE)
    def submit_credit_limit_increase_request_7392(
        self,
        credit_card_account_id: str,
        user_id: str,
        requested_increase_amount: int,
    ) -> str:
        """Submit a credit limit increase request for a customer's credit card.

        Args:
            credit_card_account_id (string): The credit card account ID to request increase for
            user_id (string): The customer's unique identifier in the system
            requested_increase_amount (integer): The dollar amount by which to increase the credit limit (e.g., 2500 for $2,500)

        Returns:
            Credit limit increase request submitted successfully.
        """
        if (
            not credit_card_account_id
            or not user_id
            or requested_increase_amount is None
        ):
            return "Error: Missing required parameters."

        if requested_increase_amount <= 0:
            return "Error: Requested increase amount must be positive."

        # Verify the credit card account exists
        if credit_card_account_id not in self.db.credit_card_accounts.data:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        cc_account = self.db.credit_card_accounts.data[credit_card_account_id]
        if cc_account.get("user_id") != user_id:
            return f"Error: Credit card account '{credit_card_account_id}' does not belong to user '{user_id}'."

        request_id = generate_credit_limit_increase_request_id(
            credit_card_account_id, user_id, requested_increase_amount
        )

        today = get_today_str()

        request_record = {
            "request_id": request_id,
            "credit_card_account_id": credit_card_account_id,
            "user_id": user_id,
            "requested_increase_amount": requested_increase_amount,
            "submitted_at": today,
            "status": "PENDING",
        }

        success = add_to_db(
            "credit_limit_increase_requests", request_id, request_record, db=self.db
        )

        if not success:
            return "Error: A similar request may already exist."

        return (
            f"Credit limit increase request submitted successfully.\n\n"
            f"Executed: submit_credit_limit_increase_request_7392\n"
            f"  - Request ID: {request_id}\n"
            f"  - Account: {credit_card_account_id}\n"
            f"  - Requested Increase: ${requested_increase_amount:,}\n"
            f"  - Status: PENDING"
        )

    @is_discoverable_tool(ToolType.READ)
    def get_credit_limit_increase_history_4829(
        self, credit_card_account_id: str
    ) -> str:
        """Retrieve the credit limit increase request history for a specific credit card account. Returns all previous CLI requests including dates, amounts, and statuses.

        Args:
            credit_card_account_id (string): The credit card account ID to check for CLI history

        Returns:
            Credit limit increase history retrieved.
        """
        if not credit_card_account_id:
            return "Error: Missing required parameter: credit_card_account_id"

        cli_result = query_database_tool(
            "credit_limit_increase_requests",
            f'{{"credit_card_account_id": "{credit_card_account_id}"}}',
            db=self.db,
        )

        result_parts = [
            "Credit limit increase history retrieved.",
            "",
            f"Executed: get_credit_limit_increase_history_4829",
            f"Credit limit increase history for account {credit_card_account_id}:",
        ]

        has_records = (
            "No records found" not in cli_result
            and "No results found" not in cli_result
        )

        if has_records:
            result_parts.append(cli_result)
        else:
            result_parts.append(
                "\nNo credit limit increase requests found for this account."
            )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.READ)
    def get_payment_history_6183(self, credit_card_account_id: str, months: int) -> str:
        """Retrieve payment history for a credit card account.

        Args:
            credit_card_account_id (string): The credit card account ID to check payment history for
            months (integer): Number of months of payment history to retrieve

        Returns:
            Payment history retrieved.
        """
        if not credit_card_account_id or months is None:
            return (
                "Error: Missing required parameters (credit_card_account_id, months)."
            )

        try:
            months = int(months)
            if months <= 0:
                return "Error: months must be a positive integer."
        except (ValueError, TypeError):
            return "Error: Invalid months value. Must be a positive integer."

        # Query the payment_history table
        payments = []
        for payment_id, payment_data in self.db.payment_history.data.items():
            if payment_data.get("credit_card_account_id") == credit_card_account_id:
                payments.append(payment_data)

        if not payments:
            return f"No payment history found for account '{credit_card_account_id}'."

        # Sort by date (most recent first)
        payments.sort(key=lambda x: x.get("payment_date") or "", reverse=True)

        # Limit to requested months
        payments = payments[:months]

        # Count consecutive on-time payments
        consecutive_on_time = 0
        for payment in payments:
            if payment.get("status") == "ON_TIME":
                consecutive_on_time += 1
            else:
                break

        result_parts = [
            f"Payment history for account '{credit_card_account_id}' (last {months} months):",
            f"Consecutive on-time payments: {consecutive_on_time}",
        ]

        for payment in payments:
            result_parts.append(
                f"\n  - Payment Date: {payment.get('payment_date')}\n"
                f"    Amount: {payment.get('amount')}\n"
                f"    Status: {payment.get('status')}"
            )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def approve_credit_limit_increase_5847(
        self,
        credit_card_account_id: str,
        user_id: str,
        new_credit_limit: int,
    ) -> str:
        """Approve and apply a credit limit increase for a customer's credit card.

        Args:
            credit_card_account_id (string): The credit card account ID
            user_id (string): The customer's unique identifier in the system
            new_credit_limit (integer): The new total credit limit in dollars (e.g., 7500 for $7,500)

        Returns:
            Credit limit increase approved and applied successfully.
        """
        if not credit_card_account_id or not user_id or new_credit_limit is None:
            return "Error: Missing required parameters."

        # Verify the credit card account exists
        if credit_card_account_id not in self.db.credit_card_accounts.data:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        cc_account = self.db.credit_card_accounts.data[credit_card_account_id]
        if cc_account.get("user_id") != user_id:
            return f"Error: Credit card account '{credit_card_account_id}' does not belong to user '{user_id}'."

        # ===========================================
        # ELIGIBILITY CHECKS - All must pass
        # Generic error message if ANY check fails
        # ===========================================
        ineligible = False

        # Check 1: No pending disputes
        if (
            hasattr(self.db, "transaction_disputes")
            and self.db.transaction_disputes.data
        ):
            for dispute_id, dispute in self.db.transaction_disputes.data.items():
                if dispute.get("user_id") == user_id and dispute.get("status") in [
                    "SUBMITTED",
                    "UNDER_REVIEW",
                    "PENDING",
                ]:
                    ineligible = True
                    break

        # Check 2: No pending replacement cards
        if (
            not ineligible
            and hasattr(self.db, "credit_card_orders")
            and self.db.credit_card_orders.data
        ):
            for order_id, order in self.db.credit_card_orders.data.items():
                if order.get(
                    "credit_card_account_id"
                ) == credit_card_account_id and order.get("status") in [
                    "PENDING",
                    "PROCESSING",
                    "SHIPPED",
                ]:
                    ineligible = True
                    break

        # Check 3: Account must be in good standing
        if not ineligible:
            account_status = cc_account.get("account_status", "").upper()
            if account_status in [
                "PAST_DUE",
                "DELINQUENT",
                "COLLECTIONS",
                "CLOSED",
            ]:
                ineligible = True

        if ineligible:
            return (
                "Error: Credit limit increase request cannot be approved at this time."
            )

        # Get current credit limit
        current_limit_str = cc_account.get("credit_limit", "$0.00")
        try:
            current_limit = float(
                str(current_limit_str).replace("$", "").replace(",", "")
            )
        except (ValueError, TypeError):
            current_limit = 0.0

        # Update the credit limit
        new_limit = float(new_credit_limit)
        self.db.credit_card_accounts.data[credit_card_account_id]["credit_limit"] = (
            f"${new_limit:.2f}"
        )

        today = get_today_str()

        # Create an approval record
        request_id = generate_credit_limit_increase_request_id(
            credit_card_account_id, user_id, new_limit - current_limit
        )

        approval_record = {
            "request_id": request_id,
            "credit_card_account_id": credit_card_account_id,
            "user_id": user_id,
            "previous_limit": f"${current_limit:.2f}",
            "new_limit": f"${new_limit:.2f}",
            "increase_amount": f"${new_limit - current_limit:.2f}",
            "decision_date": today,
            "status": "APPROVED",
        }

        add_to_db(
            "credit_limit_increase_requests",
            request_id,
            approval_record,
            db=self.db,
        )

        return (
            f"Credit limit increase approved!\n"
            f"  - Account: {credit_card_account_id}\n"
            f"  - Previous Limit: ${current_limit:.2f}\n"
            f"  - New Limit: ${new_limit:.2f}\n"
            f"  - Increase: ${new_limit - current_limit:.2f}\n"
            f"  - Effective Date: {today}\n"
            f"The customer will receive a confirmation email."
        )

    @is_discoverable_tool(ToolType.WRITE)
    def deny_credit_limit_increase_5848(
        self,
        credit_card_account_id: str,
        user_id: str,
        denial_reason: str,
    ) -> str:
        """Deny a credit limit increase request for a customer's credit card.

        Args:
            credit_card_account_id (string): The credit card account ID
            user_id (string): The customer's unique identifier in the system
            denial_reason (string): The reason for denying the request. Must be one of: 'insufficient_account_age', 'cooldown_period_active', 'pending_disputes', 'pending_replacement_card', 'past_due_balance', 'high_utilization', 'insufficient_payment_history', 'requested_amount_exceeds_limit', 'other'

        Returns:
            Credit limit increase request denied.
        """
        if not credit_card_account_id or not user_id or not denial_reason:
            return "Error: Missing required parameters."

        valid_reasons = [
            "insufficient_account_age",
            "cooldown_period_active",
            "pending_disputes",
            "pending_replacement_card",
            "past_due_balance",
            "high_utilization",
            "insufficient_payment_history",
            "requested_amount_exceeds_limit",
            "other",
        ]
        if denial_reason not in valid_reasons:
            return f"Error: Invalid denial_reason. Must be one of: {', '.join(valid_reasons)}"

        # Verify the credit card account exists
        if credit_card_account_id not in self.db.credit_card_accounts.data:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        today = get_today_str()

        # Create a denial record
        request_id = generate_credit_limit_increase_request_id(
            credit_card_account_id, user_id, 0.0
        )

        denial_record = {
            "request_id": request_id,
            "credit_card_account_id": credit_card_account_id,
            "user_id": user_id,
            "denial_reason": denial_reason,
            "decision_date": today,
            "status": "DENIED",
        }

        add_to_db(
            "credit_limit_increase_requests",
            request_id,
            denial_record,
            db=self.db,
        )

        return (
            f"Credit limit increase request denied.\n"
            f"  - Account: {credit_card_account_id}\n"
            f"  - Denial Reason: {denial_reason}\n"
            f"  - Date: {today}\n"
            f"The customer will receive a notification explaining the denial."
        )

    @is_discoverable_tool(ToolType.WRITE)
    def open_bank_account_4821(
        self,
        user_id: str,
        account_type: str,
        account_class: str,
    ) -> str:
        """Open a new bank account for a customer.

        Args:
            user_id (string): The customer's unique identifier in the system
            account_type (string): Type of account to open. Must be one of: 'checking' (personal checking), 'savings' (personal savings), 'business_checking', 'business_savings'
            account_class (string): The full official account class name

        Returns:
            Bank account opened successfully.
        """
        from datetime import datetime

        if not user_id or not account_type or not account_class:
            return "Error: Missing required parameters."

        valid_types = ["checking", "savings", "business_checking", "business_savings"]
        if account_type not in valid_types:
            return f"Error: Invalid account_type. Must be one of: {valid_types}"

        today_date = get_now()

        def get_account_age_days(acc: Dict[str, Any]) -> int:
            date_opened_str = acc.get("date_opened", "")
            try:
                date_opened = datetime.strptime(date_opened_str, "%m/%d/%Y")
                return (today_date - date_opened).days
            except ValueError:
                return 0

        # Get all user's accounts for eligibility checks
        user_accounts = [
            acc
            for acc_id, acc in self.db.accounts.data.items()
            if acc.get("user_id") == user_id
        ]

        # ========== PERSONAL SAVINGS ELIGIBILITY ==========
        # Requires: personal checking account open for at least 14 days
        if account_type == "savings":
            has_eligible_checking = False
            for acc in user_accounts:
                acc_type = acc.get("account_type", acc.get("class", ""))
                status = acc.get("status", "")
                if acc_type in ["checking", "personal_checking"] and status in [
                    "OPEN",
                    "ACTIVE",
                ]:
                    if get_account_age_days(acc) >= 14:
                        has_eligible_checking = True
                        break

            if not has_eligible_checking:
                return "Error: Account eligibility requirements not met."

        # ========== BUSINESS CHECKING ELIGIBILITY ==========
        # Requires: 1) No accounts with CLOSED status, 2) Personal checking with OPEN status
        if account_type == "business_checking":
            has_closed_account = any(
                acc.get("status") == "CLOSED" for acc in user_accounts
            )
            if has_closed_account:
                return "Error: Account eligibility requirements not met."

            has_personal_checking = any(
                acc.get("account_type", acc.get("class", ""))
                in ["checking", "personal_checking"]
                and acc.get("status", "") in ["OPEN", "ACTIVE"]
                for acc in user_accounts
            )
            if not has_personal_checking:
                return "Error: Account eligibility requirements not met."

        # ========== BUSINESS SAVINGS ELIGIBILITY ==========
        # Requires: 1) Business checking with OPEN status for 30+ days, 2) No negative balances
        if account_type == "business_savings":
            for acc in user_accounts:
                balance = _parse_balance(
                    acc.get("current_holdings", acc.get("balance", 0))
                )
                if balance < 0:
                    return "Error: Account eligibility requirements not met."

            has_eligible_business_checking = False
            for acc in user_accounts:
                acc_type = acc.get("account_type", acc.get("class", ""))
                status = acc.get("status", "")
                if acc_type == "business_checking" and status in ["OPEN", "ACTIVE"]:
                    if get_account_age_days(acc) >= 30:
                        has_eligible_business_checking = True
                        break

            if not has_eligible_business_checking:
                return "Error: Account eligibility requirements not met."

        # Generate account ID
        seed = f"account:{user_id}:{account_type}:{account_class}"
        account_id = _deterministic_id(seed)

        today = get_today_str()

        account_record = {
            "account_id": account_id,
            "user_id": user_id,
            "account_type": account_type,
            "account_class": account_class,
            "current_holdings": "0.00",
            "status": "OPEN",
            "date_opened": today,
        }

        success = add_to_db("accounts", account_id, account_record, db=self.db)

        if not success:
            return (
                f"Failed to open account: Account ID '{account_id}' may already exist."
            )

        return (
            f"Bank account opened successfully!\n"
            f"  - Account ID: {account_id}\n"
            f"  - User ID: {user_id}\n"
            f"  - Account Type: {account_type}\n"
            f"  - Account Class: {account_class}\n"
            f"  - Status: OPEN\n"
            f"  - Initial Balance: $0.00\n"
            f"  - Date Opened: {today}"
        )

    @is_discoverable_tool(ToolType.WRITE)
    def close_bank_account_7392(
        self,
        account_id: str,
        reason: str = "Customer requested closure",
        waive_early_closure_fee: bool = False,
    ) -> str:
        """Close a customer's bank account (checking or savings).

        Args:
            account_id (string): The ID of the bank account to close
            reason (string, optional): The reason for closing the account
            waive_early_closure_fee (boolean, optional): Whether to waive early closure fees

        Returns:
            Bank account closed successfully.
        """
        from datetime import datetime

        # Early closure fee configuration by account tier
        PERSONAL_CHECKING_EARLY_CLOSURE = {
            "Light Blue Account": {"fee": 15, "window_days": 30},
            "Light Green Account": {"fee": 15, "window_days": 30},
            "Green Fee-Free Account": {"fee": 15, "window_days": 30},
            "Blue Account": {"fee": 25, "window_days": 60},
            "Green Account": {"fee": 25, "window_days": 60},
            "Evergreen Account": {"fee": 50, "window_days": 90},
            "Bluest Account": {"fee": 100, "window_days": 180},
        }
        PERSONAL_SAVINGS_EARLY_CLOSURE = {
            "Bronze Account": {"fee": 20, "window_days": 60},
            "Silver Account": {"fee": 35, "window_days": 90},
            "Silver Plus Account": {"fee": 35, "window_days": 90},
            "Gold Account": {"fee": 75, "window_days": 180},
            "Gold Plus Account": {"fee": 75, "window_days": 180},
            "Gold Years Account": {"fee": 75, "window_days": 180},
            "Platinum Account": {"fee": 150, "window_days": 270},
            "Platinum Plus Account": {"fee": 150, "window_days": 270},
            "Diamond Elite Account": {"fee": 150, "window_days": 270},
        }

        if not account_id:
            return "Error: Missing required parameter (account_id)."

        if account_id not in self.db.accounts.data:
            return f"Error: Account '{account_id}' not found."

        account = self.db.accounts.data[account_id]

        if account.get("status") == "CLOSED":
            return f"Error: Account '{account_id}' is already closed."

        balance = _get_account_balance(account)

        # Check for early closure fee requirements
        early_closure_fee_applied = 0.0
        if not waive_early_closure_fee:
            account_level = account.get("level", "")
            account_class = account.get("class", "")
            date_opened_str = account.get("date_opened", "")

            early_closure_config = None
            if account_class == "checking":
                early_closure_config = PERSONAL_CHECKING_EARLY_CLOSURE.get(
                    account_level
                )
            elif account_class == "savings":
                early_closure_config = PERSONAL_SAVINGS_EARLY_CLOSURE.get(account_level)

            if early_closure_config and date_opened_str:
                try:
                    date_opened = datetime.strptime(date_opened_str, "%m/%d/%Y")
                    today_date = get_now()
                    account_age_days = (today_date - date_opened).days

                    if account_age_days < early_closure_config["window_days"]:
                        required_fee = early_closure_config["fee"]
                        if balance < required_fee:
                            return "Error: Account unable to be closed."
                        else:
                            early_closure_fee_applied = required_fee
                except ValueError:
                    pass

        # After deducting early closure fee (if any), remaining balance must be $0
        remaining_balance = balance - early_closure_fee_applied
        if remaining_balance != 0:
            return f"Error: Account balance must be $0.00 before closing. Current balance: ${balance:.2f}"

        today = get_today_str()

        account["status"] = "CLOSED"
        account["date_closed"] = today
        account["closure_reason"] = reason
        account["early_closure_fee_waived"] = waive_early_closure_fee

        return (
            f"Bank account closed successfully!\n"
            f"  - Account ID: {account_id}\n"
            f"  - Account Type: {account.get('account_type', 'N/A')}\n"
            f"  - Account Class: {account.get('account_class', 'N/A')}\n"
            f"  - Status: CLOSED\n"
            f"  - Date Closed: {today}\n"
            f"  - Reason: {reason}\n"
            f"  - Early Closure Fee Waived: {'Yes' if waive_early_closure_fee else 'No'}"
        )

    @is_discoverable_tool(ToolType.READ)
    def get_all_user_accounts_by_user_id_3847(self, user_id: str) -> str:
        """Retrieve all accounts (checking, savings, credit cards) for a customer.

        Args:
            user_id (string): The customer's unique identifier in the system

        Returns:
            User accounts retrieved successfully.
        """
        if not user_id:
            return "Error: Missing required parameter: user_id"

        accounts_result = query_database_tool(
            "accounts", f'{{"user_id": "{user_id}"}}', db=self.db
        )

        cc_result = query_database_tool(
            "credit_card_accounts", f'{{"user_id": "{user_id}"}}', db=self.db
        )

        result_parts = [
            "User accounts retrieved successfully.",
            "",
            f"Executed: get_all_user_accounts_by_user_id_3847",
            f"Accounts for user {user_id}:",
            "",
            "Bank Accounts:",
        ]

        if (
            "No records found" not in accounts_result
            and "No results found" not in accounts_result
        ):
            result_parts.append(accounts_result)
        else:
            result_parts.append("  No bank accounts found.")

        result_parts.append("\nCredit Card Accounts:")
        if "No records found" not in cc_result and "No results found" not in cc_result:
            result_parts.append(cc_result)
        else:
            result_parts.append("  No credit card accounts found.")

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def transfer_funds_between_bank_accounts_7291(
        self,
        source_account_id: str,
        destination_account_id: str,
        amount: float,
    ) -> str:
        """Transfer funds from one bank account to another.

        Args:
            source_account_id (string): The account ID to transfer funds from
            destination_account_id (string): The account ID to transfer funds to
            amount (number): The amount to transfer in USD

        Returns:
            Funds transferred successfully.
        """
        if not source_account_id or not destination_account_id or amount is None:
            return "Error: Missing required parameters (source_account_id, destination_account_id, amount)."

        try:
            amount = float(amount)
        except (ValueError, TypeError):
            return f"Error: Invalid amount '{amount}'. Must be a number."

        if amount <= 0:
            return "Error: Transfer amount must be positive."

        if source_account_id == destination_account_id:
            return "Error: Source and destination accounts cannot be the same."

        if source_account_id not in self.db.accounts.data:
            return f"Error: Source account '{source_account_id}' not found."

        if destination_account_id not in self.db.accounts.data:
            return f"Error: Destination account '{destination_account_id}' not found."

        source = self.db.accounts.data[source_account_id]
        if source.get("status") not in ("ACTIVE", "OPEN"):
            return f"Error: Source account '{source_account_id}' is not active."

        dest = self.db.accounts.data[destination_account_id]
        if dest.get("status") not in ("ACTIVE", "OPEN"):
            return (
                f"Error: Destination account '{destination_account_id}' is not active."
            )

        source_balance = _get_account_balance(source)
        if source_balance < amount:
            return (
                f"Error: Insufficient funds. Source account balance is ${source_balance:.2f}, "
                f"but transfer amount is ${amount:.2f}."
            )

        dest_balance = _get_account_balance(dest)

        # Update balances
        new_source = source_balance - amount
        new_dest = dest_balance + amount
        source["current_holdings"] = f"${new_source:.2f}"
        dest["current_holdings"] = f"${new_dest:.2f}"

        return (
            f"Transfer completed successfully!\n"
            f"  - Amount: ${amount:.2f}\n"
            f"  - From: {source_account_id} (new balance: ${new_source:.2f})\n"
            f"  - To: {destination_account_id} (new balance: ${new_dest:.2f})"
        )

    @is_discoverable_tool(ToolType.WRITE)
    def apply_checking_account_credit_5829(
        self,
        account_id: str,
        amount: float,
        credit_type: str,
    ) -> str:
        """Apply a credit to a customer's checking account.

        Args:
            account_id (string): The checking account ID to credit
            amount (number): The positive dollar amount to credit (must be greater than 0)
            credit_type (string): The type of credit: 'rebate_credit' for missing rebates, 'fee_refund' for incorrect fee charges

        Returns:
            Credit applied to checking account successfully.
        """
        if not account_id or amount is None or not credit_type:
            return "Error: Missing required parameters."

        if amount <= 0:
            return "Error: Credit amount must be positive."

        valid_types = ["rebate_credit", "fee_refund"]
        if credit_type not in valid_types:
            return f"Error: Invalid credit_type. Must be one of: {valid_types}"

        if account_id not in self.db.accounts.data:
            return f"Error: Account '{account_id}' not found."

        account = self.db.accounts.data[account_id]

        # Verify it's a checking account
        account_class = account.get("class", "").lower()
        if account_class != "checking":
            return f"Error: Account '{account_id}' is not a checking account. Credits can only be applied to checking accounts."

        # Check account status
        if account.get("status") not in ("ACTIVE", "OPEN"):
            return f"Error: Account '{account_id}' is not active."

        current_balance = _get_account_balance(account)
        new_balance = current_balance + amount

        # Update the account balance
        account["current_holdings"] = f"${new_balance:.2f}"

        # Create transaction record
        transaction_id = f"txn_{_deterministic_id(f'checking_credit:{account_id}:{credit_type}:{amount}:{get_today_str()}')}"

        if credit_type == "rebate_credit":
            description = "REBATE CREDIT - CUSTOMER SERVICE"
        else:
            description = "FEE REFUND - CUSTOMER SERVICE"

        transaction_record = {
            "transaction_id": transaction_id,
            "account_id": account_id,
            "date": get_today_str(),
            "description": description,
            "amount": amount,
            "type": credit_type,
            "status": "posted",
        }

        self.db.bank_account_transaction_history.data[transaction_id] = (
            transaction_record
        )

        return (
            f"\nCredit applied successfully!\n"
            f"  - Transaction ID: {transaction_id}\n"
            f"  - Account: {account_id}\n"
            f"  - Credit Type: {credit_type}\n"
            f"  - Amount: ${amount:.2f}\n"
            f"  - Previous Balance: ${current_balance:.2f}\n"
            f"  - New Balance: ${new_balance:.2f}"
        )

    @is_discoverable_tool(ToolType.WRITE)
    def apply_savings_account_credit_6831(
        self,
        account_id: str,
        amount: float,
        credit_type: str,
    ) -> str:
        """Apply a credit to a customer's savings account for interest corrections, fee refunds, or goodwill adjustments.

        Args:
            account_id (string): The savings account ID to credit
            amount (number): The positive dollar amount to credit (must be greater than 0)
            credit_type (string): The type of credit: 'interest_correction' for APY/interest calculation errors, 'fee_refund' for incorrect fee charges, 'goodwill_credit' for customer service gestures

        Returns:
            Credit applied to savings account successfully.
        """
        if not account_id or amount is None or not credit_type:
            return "Error: Missing required parameters."

        if amount <= 0:
            return "Error: Credit amount must be positive."

        valid_types = ["interest_correction", "fee_refund", "goodwill_credit"]
        if credit_type not in valid_types:
            return f"Error: Invalid credit_type. Must be one of: {valid_types}"

        if account_id not in self.db.accounts.data:
            return f"Error: Account '{account_id}' not found."

        account = self.db.accounts.data[account_id]

        # Verify it's a savings account
        account_class = account.get("class", "").lower()
        if account_class not in ("saving", "savings"):
            return f"Error: Account '{account_id}' is not a savings account. This tool only applies to savings accounts."

        # Check account status
        if account.get("status") not in ("ACTIVE", "OPEN"):
            return f"Error: Account '{account_id}' is not active."

        current_balance = _get_account_balance(account)
        new_balance = current_balance + amount

        # Update the account balance
        account["current_holdings"] = f"{new_balance:.2f}"

        # Create transaction record
        transaction_id = f"txn_{_deterministic_id(f'savings_credit:{account_id}:{credit_type}:{amount}:{get_today_str()}')}"

        if credit_type == "interest_correction":
            description = "INTEREST CORRECTION - CUSTOMER SERVICE"
        elif credit_type == "fee_refund":
            description = "FEE REFUND - CUSTOMER SERVICE"
        else:
            description = "GOODWILL CREDIT - CUSTOMER SERVICE"

        transaction_record = {
            "transaction_id": transaction_id,
            "account_id": account_id,
            "date": get_today_str(),
            "description": description,
            "amount": amount,
            "type": credit_type,
            "status": "posted",
        }

        self.db.bank_account_transaction_history.data[transaction_id] = (
            transaction_record
        )

        return (
            f"\nCredit applied successfully!\n"
            f"  - Transaction ID: {transaction_id}\n"
            f"  - Account: {account_id}\n"
            f"  - Credit Type: {credit_type}\n"
            f"  - Amount: ${amount:.2f}\n"
            f"  - Previous Balance: ${current_balance:.2f}\n"
            f"  - New Balance: ${new_balance:.2f}"
        )

    @is_discoverable_tool(ToolType.WRITE)
    def submit_interest_discrepancy_report_7294(
        self,
        account_id: str,
        user_id: str,
        expected_apy: float,
        actual_apy: float,
        amount_difference: float,
    ) -> str:
        """Submit a report for interest calculation discrepancies to the backend team for investigation. Use this when the interest credited to a customer's account does not match expected APY calculations.

        Args:
            account_id (string): The savings account ID with the discrepancy
            user_id (string): The customer's unique identifier
            expected_apy (number): The APY percentage the customer should have received (e.g., 2.775 for 2.775%)
            actual_apy (number): The APY percentage that was actually applied (e.g., 2.5 for 2.5%)
            amount_difference (number): The dollar amount difference between expected and actual interest credited

        Returns:
            Interest discrepancy report submitted successfully. Backend team will investigate.
        """
        if (
            not account_id
            or not user_id
            or expected_apy is None
            or actual_apy is None
            or amount_difference is None
        ):
            return "Error: Missing required parameters."

        # Validate numeric values
        try:
            expected_apy = float(expected_apy)
            actual_apy = float(actual_apy)
            amount_difference = float(amount_difference)
        except (ValueError, TypeError):
            return "Error: expected_apy, actual_apy, and amount_difference must be numbers."

        # Verify account exists
        account = self.db.accounts.data.get(account_id)
        if account is None:
            return f"Error: Account '{account_id}' not found."

        # Verify user exists
        user = self.db.users.data.get(user_id)
        if user is None:
            return f"Error: User '{user_id}' not found."

        report_id = f"IDR_{_deterministic_id(f'interest_report:{account_id}:{user_id}:{expected_apy}:{actual_apy}:{get_today_str()}')}"

        report_record = {
            "report_id": report_id,
            "account_id": account_id,
            "user_id": user_id,
            "account_level": account.get("level", "Unknown"),
            "expected_apy": expected_apy,
            "actual_apy": actual_apy,
            "apy_difference": round(expected_apy - actual_apy, 4),
            "amount_difference": amount_difference,
            "submitted_date": get_today_str(),
            "status": "PENDING_REVIEW",
        }

        add_to_db("interest_discrepancy_reports", report_id, report_record, db=self.db)

        return (
            f"\nInterest Discrepancy Report Submitted Successfully!\n"
            f"  - Report ID: {report_id}\n"
            f"  - Account: {account_id} ({account.get('level', 'Unknown')})\n"
            f"  - Customer: {user.get('name', 'Unknown')}\n"
            f"  - Expected APY: {expected_apy}%\n"
            f"  - Actual APY: {actual_apy}%\n"
            f"  - APY Difference: {round(expected_apy - actual_apy, 4)}%\n"
            f"  - Amount Difference: ${amount_difference:.2f}\n"
            f"  - Status: PENDING_REVIEW\n"
            f"\nThe backend team will investigate this discrepancy and ensure "
            f"correct APY calculations are applied going forward."
        )

    @is_discoverable_tool(ToolType.READ)
    def get_bank_account_transactions_9173(self, account_id: str) -> str:
        """Retrieve the transaction history for a bank account.

        Args:
            account_id (string): The bank account ID to retrieve transactions for

        Returns:
            Bank account transactions retrieved successfully.
        """
        if not account_id:
            return "Error: Missing required parameter: account_id"

        # Verify the account exists
        if account_id not in self.db.accounts.data:
            return f"Error: Account '{account_id}' not found."

        txn_result = query_database_tool(
            "bank_account_transaction_history",
            f'{{"account_id": "{account_id}"}}',
            db=self.db,
        )

        result_parts = [
            "Bank account transactions retrieved successfully.",
            "",
            f"Executed: get_bank_account_transactions_9173",
            f"Transactions for account {account_id}:",
        ]

        if (
            "No records found" not in txn_result
            and "No results found" not in txn_result
        ):
            result_parts.append(txn_result)
        else:
            result_parts.append("\nNo transactions found for this account.")

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def order_debit_card_5739(
        self,
        account_id: str,
        user_id: str,
        delivery_option: str,
        delivery_fee: float,
        card_design: str,
        design_fee: float,
        shipping_address: str,
        excess_replacement_fee: Optional[float] = None,
    ) -> str:
        """Order a new debit card for a customer's checking account.

        Args:
            account_id (string): The checking account ID to link the debit card to
            user_id (string): The customer's unique identifier
            delivery_option (string): Shipping speed: STANDARD, EXPEDITED, or RUSH
            delivery_fee (number): Fee to charge for delivery in dollars
            card_design (string): Card design: CLASSIC, PREMIUM, or CUSTOM
            design_fee (number): Fee to charge for card design in dollars
            shipping_address (string): Full shipping address for card delivery
            excess_replacement_fee (number, optional): Fee for exceeding replacement limit, if applicable

        Returns:
            Debit card order placed successfully.
        """
        excess_replacement_fee = excess_replacement_fee or 0

        # Ensure fees are numbers
        try:
            delivery_fee = float(delivery_fee) if delivery_fee is not None else None
        except (TypeError, ValueError):
            return "Error: delivery_fee must be a number."

        try:
            design_fee = float(design_fee) if design_fee is not None else None
        except (TypeError, ValueError):
            return "Error: design_fee must be a number."

        try:
            excess_replacement_fee = float(excess_replacement_fee)
        except (TypeError, ValueError):
            excess_replacement_fee = 0

        if not all(
            [
                account_id,
                user_id,
                delivery_option,
                delivery_fee is not None,
                card_design,
                design_fee is not None,
                shipping_address,
            ]
        ):
            return "Error: Missing required parameters. Required: account_id, user_id, delivery_option, delivery_fee, card_design, design_fee, shipping_address."

        valid_delivery = ["STANDARD", "EXPEDITED", "RUSH"]
        if delivery_option.upper() not in valid_delivery:
            return f"Error: Invalid delivery_option. Must be one of: {valid_delivery}"
        delivery_option = delivery_option.upper()

        valid_design = ["CLASSIC", "PREMIUM", "CUSTOM"]
        if card_design.upper() not in valid_design:
            return f"Error: Invalid card_design. Must be one of: {valid_design}"
        card_design = card_design.upper()

        # Verify the account exists and is a checking account
        if account_id not in self.db.accounts.data:
            return f"Error: Account '{account_id}' not found."

        account = self.db.accounts.data[account_id]

        if account.get("class") != "checking":
            return f"Error: Debit cards can only be ordered for checking accounts. Account '{account_id}' is a {account.get('class')} account."

        if account.get("status") != "OPEN":
            return f"Error: Account must be OPEN. Account '{account_id}' has status: {account.get('status')}"

        if account.get("user_id") != user_id:
            return f"Error: Account '{account_id}' does not belong to user '{user_id}'."

        # Check minimum balance ($25)
        try:
            current_holdings = float(
                str(account.get("current_holdings", "0")).replace(",", "")
            )
        except ValueError:
            current_holdings = 0.0

        if current_holdings < 25.0:
            return f"Error: Account must have a minimum balance of $25. Current balance: ${current_holdings:.2f}"

        # Check for pending debit card orders for this account
        for order in self.db.debit_card_orders.data.values():
            if (
                order.get("account_id") == account_id
                and order.get("status") == "PENDING"
            ):
                return f"Error: There is already a pending debit card order for account '{account_id}'."

        # Check if customer already has an active debit card for this account
        active_cards_count = sum(
            1
            for card in self.db.debit_cards.data.values()
            if card.get("account_id") == account_id and card.get("status") == "ACTIVE"
        )
        if active_cards_count >= 1:
            return f"Error: Account '{account_id}' already has an active debit card. Maximum 1 active card per checking account."

        # Calculate total fees
        total_fee = delivery_fee + design_fee + excess_replacement_fee

        # Check if account has sufficient funds for fees
        if total_fee > 0 and current_holdings < total_fee:
            return f"Error: Insufficient funds for fees. Total fees: ${total_fee:.2f}. Current balance: ${current_holdings:.2f}"

        # Calculate expected delivery
        delivery_times = {
            "STANDARD": "7-10 business days",
            "EXPEDITED": "3-5 business days",
            "RUSH": "1-2 business days",
        }
        expected_delivery = delivery_times[delivery_option]

        # Generate order ID
        order_date = get_today_str()
        order_id = generate_debit_card_order_id(account_id, user_id, delivery_option)

        # Create the order record
        order_record = {
            "order_id": order_id,
            "account_id": account_id,
            "user_id": user_id,
            "delivery_option": delivery_option,
            "card_design": card_design,
            "shipping_address": shipping_address,
            "delivery_fee": delivery_fee,
            "design_fee": design_fee,
            "excess_replacement_fee": excess_replacement_fee,
            "total_fee": total_fee,
            "order_date": order_date,
            "expected_delivery": expected_delivery,
            "status": "PENDING",
        }

        success = add_to_db("debit_card_orders", order_id, order_record, db=self.db)
        if not success:
            return "Error: Failed to create debit card order. Order may already exist."

        # Deduct fees from the checking account if applicable
        if total_fee > 0:
            new_balance = current_holdings - total_fee
            self.db.accounts.data[account_id]["current_holdings"] = f"{new_balance:.2f}"

            # Create a transaction record for the fee
            fee_description_parts = []
            if delivery_fee > 0:
                fee_description_parts.append(f"Delivery ${delivery_fee}")
            if design_fee > 0:
                fee_description_parts.append(f"Design ${design_fee}")
            if excess_replacement_fee > 0:
                fee_description_parts.append(
                    f"Excess Replacement ${excess_replacement_fee:.0f}"
                )
            fee_description = (
                f"DEBIT CARD ORDER FEE - {', '.join(fee_description_parts)}"
            )

            fee_txn_id = f"btxn_dcfee_{order_id[-8:]}"
            fee_transaction = {
                "transaction_id": fee_txn_id,
                "account_id": account_id,
                "date": order_date,
                "description": fee_description,
                "amount": -total_fee,
                "type": "debit_card_fee",
                "status": "posted",
            }
            add_to_db(
                "bank_account_transaction_history",
                fee_txn_id,
                fee_transaction,
                db=self.db,
            )

        # Create the debit card entry in debit_cards table with PENDING status
        card_id = generate_debit_card_id(account_id, user_id, order_date)

        # Generate last 4 digits and CVV deterministically
        card_details_seed = f"card_details:{card_id}"
        last_4_hash = _deterministic_id(card_details_seed + ":last4", length=8)
        cvv_hash = _deterministic_id(card_details_seed + ":cvv", length=6)

        last_4_digits = "".join(c for c in last_4_hash if c.isdigit())[:4].zfill(4)
        cvv = "".join(c for c in cvv_hash if c.isdigit())[:3].zfill(3)

        # Get cardholder name from users table
        cardholder_name = "CARDHOLDER"
        if user_id in self.db.users.data:
            user = self.db.users.data[user_id]
            cardholder_name = user.get("name", "CARDHOLDER").upper()

        # Calculate expiration date (4 years from now)
        try:
            parts = order_date.split("/")
            exp_month = parts[0]
            exp_year = str(int(parts[2]) + 4)
            if exp_month in ["01", "03", "05", "07", "08", "10", "12"]:
                exp_day = "31"
            elif exp_month == "02":
                exp_day = "28"
            else:
                exp_day = "30"
            expiration_date = f"{exp_month}/{exp_day}/{exp_year}"
        except (ValueError, IndexError):
            expiration_date = "12/31/2029"

        # Determine issue_reason
        existing_cards = [
            c
            for c in self.db.debit_cards.data.values()
            if c.get("account_id") == account_id
        ]
        closed_card = next(
            (c for c in existing_cards if c.get("status") == "CLOSED"), None
        )
        if closed_card:
            closure_reason = closed_card.get("closure_reason", "first_card")
            if closure_reason in ["lost", "stolen", "fraud", "fraud_suspected"]:
                issue_reason = (
                    closure_reason if closure_reason != "fraud_suspected" else "fraud"
                )
            else:
                issue_reason = "first_card"
        elif existing_cards:
            issue_reason = "first_card"
        else:
            issue_reason = "new_account"

        card_record = {
            "card_id": card_id,
            "account_id": account_id,
            "user_id": user_id,
            "cardholder_name": cardholder_name,
            "last_4_digits": last_4_digits,
            "cvv": cvv,
            "status": "PENDING",
            "issue_date": order_date,
            "expiration_date": expiration_date,
            "card_design": card_design,
            "issue_reason": issue_reason,
        }

        add_to_db("debit_cards", card_id, card_record, db=self.db)

        # Build response
        result_parts = [
            "Debit Card Order Confirmed",
            f"Order ID: {order_id}",
            f"Card ID: {card_id}",
            f"Linked Account: {account_id}",
            f"Delivery Option: {delivery_option}",
            f"Card Design: {card_design}",
            f"Shipping Address: {shipping_address}",
            f"Expected Delivery: {expected_delivery}",
            "",
            "Note: Card will arrive with status PENDING. Customer must call to activate after receiving the card.",
        ]

        if total_fee > 0:
            fee_details = []
            if delivery_fee > 0:
                fee_details.append(f"Delivery: ${delivery_fee}")
            if design_fee > 0:
                fee_details.append(f"Design: ${design_fee}")
            if excess_replacement_fee > 0:
                fee_details.append(f"Excess Replacement: ${excess_replacement_fee:.0f}")
            result_parts.append(
                f"Total Fees: ${total_fee:.2f} ({', '.join(fee_details)}) - CHARGED to account {account_id}"
            )
            result_parts.append(
                f"New Account Balance: ${current_holdings - total_fee:.2f}"
            )
        else:
            result_parts.append("Total Fees: $0 (No additional charges)")

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def activate_debit_card_8291(
        self,
        card_id: str,
        last_4_digits: str,
        expiration_date: str,
        cvv: str,
        pin: str,
    ) -> str:
        """Activate a NEW debit card for a customer. Use ONLY for first-time cards on a checking account (issue_reason = 'new_account' or 'first_card'). For replacement or reissued cards, use the appropriate variant.

        Args:
            card_id (string): The debit card ID to activate
            last_4_digits (string): Last 4 digits of the card number (for verification)
            expiration_date (string): Card expiration date in MM/YY format
            cvv (string): 3-digit CVV from the back of the card
            pin (string): 4-digit PIN chosen by the customer

        Returns:
            New debit card activated successfully.
        """
        args = {
            "card_id": card_id,
            "last_4_digits": last_4_digits,
            "expiration_date": expiration_date,
            "cvv": cvv,
            "pin": pin,
        }

        error, card = _validate_activation_common(
            args, self.db, ["new_account", "first_card"], "activate_debit_card_8291"
        )
        if error:
            return error

        account_id = card.get("account_id")

        # Activate the card
        card["status"] = "ACTIVE"
        card["activated_date"] = get_today_str()

        # Deactivate any other active cards for the same account
        deactivated_cards = []
        for other_card_id, other_card in self.db.debit_cards.data.items():
            if (
                other_card_id != card_id
                and other_card.get("account_id") == account_id
                and other_card.get("status") == "ACTIVE"
            ):
                other_card["status"] = "DEACTIVATED"
                other_card["deactivated_date"] = get_today_str()
                other_card["deactivation_reason"] = "New card activated"
                deactivated_cards.append(other_card_id)

        result_parts = [
            "New Debit Card Activation Successful",
            f"Card ID: {card_id}",
            "Status: ACTIVE",
            f"Activation Date: {get_today_str()}",
            "",
            "Your card is now ready to use at any ATM or point of sale terminal.",
            "For security, please sign the back of your card.",
        ]

        if deactivated_cards:
            result_parts.append(
                f"\nNote: Previous card(s) have been deactivated: {', '.join(deactivated_cards)}"
            )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def activate_debit_card_8292(
        self,
        card_id: str,
        last_4_digits: str,
        expiration_date: str,
        cvv: str,
        pin: str,
    ) -> str:
        """Activate a REPLACEMENT debit card. Use ONLY for cards replacing lost, stolen, or fraud-suspected cards (issue_reason = 'lost', 'stolen', or 'fraud'). For new or reissued cards, use the appropriate variant.

        Args:
            card_id (string): The debit card ID to activate
            last_4_digits (string): Last 4 digits of the card number (for verification)
            expiration_date (string): Card expiration date in MM/YY format
            cvv (string): 3-digit CVV from the back of the card
            pin (string): 4-digit PIN chosen by the customer

        Returns:
            Replacement debit card activated successfully.
        """
        args = {
            "card_id": card_id,
            "last_4_digits": last_4_digits,
            "expiration_date": expiration_date,
            "cvv": cvv,
            "pin": pin,
        }

        error, card = _validate_activation_common(
            args, self.db, ["lost", "stolen", "fraud"], "activate_debit_card_8292"
        )
        if error:
            return error

        account_id = card.get("account_id")
        issue_reason = card.get("issue_reason", "lost")

        # Activate the card
        card["status"] = "ACTIVE"
        card["activated_date"] = get_today_str()

        # Deactivate any other active cards for the same account (immediately for security)
        deactivated_cards = []
        for other_card_id, other_card in self.db.debit_cards.data.items():
            if (
                other_card_id != card_id
                and other_card.get("account_id") == account_id
                and other_card.get("status") == "ACTIVE"
            ):
                other_card["status"] = "DEACTIVATED"
                other_card["deactivated_date"] = get_today_str()
                other_card["deactivation_reason"] = (
                    f"Replacement card activated ({issue_reason})"
                )
                deactivated_cards.append(other_card_id)

        result_parts = [
            "Replacement Debit Card Activation Successful",
            f"Card ID: {card_id}",
            f"Replacement Reason: {issue_reason.replace('_', ' ').title()}",
            "Status: ACTIVE",
            f"Activation Date: {get_today_str()}",
            "",
            "Your replacement card is now ready to use.",
            "",
            "IMPORTANT SECURITY REMINDERS:",
            "- Please review your recent transactions for any unauthorized charges",
            "- Report any suspicious activity immediately",
        ]

        if issue_reason == "fraud":
            result_parts.append(
                "- Since fraud was suspected, we recommend changing your online banking password"
            )

        if deactivated_cards:
            result_parts.append(
                f"\nPrevious card(s) have been deactivated for security: {', '.join(deactivated_cards)}"
            )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def activate_debit_card_8293(
        self,
        card_id: str,
        last_4_digits: str,
        expiration_date: str,
        cvv: str,
        pin: str,
    ) -> str:
        """Activate a REISSUED debit card. Use ONLY for cards reissued due to expiration, damage, design upgrade, or bank-initiated replacement (issue_reason = 'expired', 'damaged', 'upgrade', or 'bank_reissue'). For new or replacement cards, use the appropriate variant.

        Args:
            card_id (string): The debit card ID to activate
            last_4_digits (string): Last 4 digits of the card number (for verification)
            expiration_date (string): Card expiration date in MM/YY format
            cvv (string): 3-digit CVV from the back of the card
            pin (string): 4-digit PIN chosen by the customer

        Returns:
            Reissued debit card activated successfully.
        """
        args = {
            "card_id": card_id,
            "last_4_digits": last_4_digits,
            "expiration_date": expiration_date,
            "cvv": cvv,
            "pin": pin,
        }

        error, card = _validate_activation_common(
            args,
            self.db,
            ["expired", "damaged", "upgrade", "bank_reissue"],
            "activate_debit_card_8293",
        )
        if error:
            return error

        account_id = card.get("account_id")
        issue_reason = card.get("issue_reason", "expired")

        # Activate the card
        card["status"] = "ACTIVE"
        card["activated_date"] = get_today_str()

        # For reissued cards, old card has 24-hour grace period
        old_cards_with_grace = []
        for other_card_id, other_card in self.db.debit_cards.data.items():
            if (
                other_card_id != card_id
                and other_card.get("account_id") == account_id
                and other_card.get("status") == "ACTIVE"
            ):
                other_card["status"] = "GRACE_PERIOD"
                other_card["grace_period_ends"] = get_today_str()
                other_card["deactivation_reason"] = (
                    f"Reissued card activated ({issue_reason})"
                )
                old_cards_with_grace.append(other_card_id)

        result_parts = [
            "Reissued Debit Card Activation Successful",
            f"Card ID: {card_id}",
            f"Reissue Reason: {issue_reason.replace('_', ' ').title()}",
            "Status: ACTIVE",
            f"Activation Date: {get_today_str()}",
            "",
            "Your reissued card is now ready to use.",
        ]

        if old_cards_with_grace:
            result_parts.append(
                f"\nNote: Your previous card(s) ({', '.join(old_cards_with_grace)}) will remain active for 24 hours as a grace period."
            )
            result_parts.append(
                "After 24 hours, the old card(s) will be automatically deactivated."
            )

        if issue_reason in ["expired", "bank_reissue"]:
            result_parts.append(
                "\nReminder: If your card number changed, please update any recurring payments with your new card details."
            )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def close_debit_card_4721(self, card_id: str, reason: str) -> str:
        """Close or cancel a debit card permanently.

        Args:
            card_id (string): The debit card ID to close
            reason (string): Reason for closing: lost, stolen, fraud_suspected, damaged, no_longer_needed, or account_closing

        Returns:
            Debit card closed successfully.
        """
        if not card_id or not reason:
            return "Error: Missing required parameters. Required: card_id, reason."

        valid_reasons = [
            "lost",
            "stolen",
            "fraud_suspected",
            "damaged",
            "no_longer_needed",
            "account_closing",
        ]
        if reason.lower() not in valid_reasons:
            return f"Error: Invalid reason. Must be one of: {valid_reasons}"
        reason = reason.lower()

        if card_id not in self.db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."

        card = self.db.debit_cards.data[card_id]

        # Check card status - can only close ACTIVE or PENDING cards
        if card.get("status") not in ["ACTIVE", "PENDING"]:
            return f"Error: Debit card '{card_id}' cannot be closed. Current status: {card.get('status')}. Only ACTIVE or PENDING cards can be closed."

        previous_status = card.get("status")

        # Close the card
        card["status"] = "CLOSED"
        card["closed_date"] = get_today_str()
        card["closure_reason"] = reason

        result_parts = [
            "Debit Card Closed Successfully",
            f"Card ID: {card_id}",
            f"Previous Status: {previous_status}",
            "New Status: CLOSED",
            f"Closure Reason: {reason.replace('_', ' ').title()}",
            f"Closure Date: {get_today_str()}",
            "",
        ]

        if reason in ["lost", "stolen", "fraud_suspected"]:
            result_parts.append(
                "IMPORTANT: This card has been immediately deactivated for security."
            )
            result_parts.append("Any pending transactions may still be processed.")
            if reason == "fraud_suspected":
                result_parts.append(
                    "Please advise the customer to review recent transactions and file disputes for any unauthorized charges."
                )
                result_parts.append(
                    "Also recommend changing their online banking password."
                )

        result_parts.append("")
        result_parts.append(
            "Note: This card cannot be reactivated. If the customer needs a new card, they can order one through the standard ordering process."
        )
        result_parts.append(
            "Any recurring payments linked to this card will need to be updated with new payment information."
        )

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def freeze_debit_card_3892(self, card_id: str) -> str:
        """Temporarily freeze a debit card. The card can be unfrozen later.

        Args:
            card_id (string): The debit card ID to freeze

        Returns:
            Debit card frozen successfully.
        """
        if not card_id:
            return "Error: Missing required parameter: card_id."

        if card_id not in self.db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."

        card = self.db.debit_cards.data[card_id]

        # Check card status - can only freeze ACTIVE cards
        if card.get("status") == "FROZEN":
            return f"Error: Debit card '{card_id}' is already frozen."

        if card.get("status") != "ACTIVE":
            return f"Error: Debit card '{card_id}' cannot be frozen. Current status: {card.get('status')}. Only ACTIVE cards can be frozen."

        # Freeze the card
        card["status"] = "FROZEN"
        card["frozen_date"] = get_today_str()

        result_parts = [
            "Debit Card Frozen Successfully",
            f"Card ID: {card_id}",
            "Status: FROZEN",
            f"Frozen Date: {get_today_str()}",
            "",
            "While frozen:",
            "- All new purchase transactions will be declined",
            "- Recurring payments and subscriptions will be declined",
            "- Pending transactions already authorized may still process",
            "",
            "To unfreeze, the customer can call customer service or use the mobile app.",
            "If the card is confirmed lost or stolen, recommend closing the card permanently instead.",
        ]

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def unfreeze_debit_card_3893(self, card_id: str) -> str:
        """Unfreeze a previously frozen debit card.

        Args:
            card_id (string): The debit card ID to unfreeze

        Returns:
            Debit card unfrozen successfully.
        """
        if not card_id:
            return "Error: Missing required parameter: card_id."

        if card_id not in self.db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."

        card = self.db.debit_cards.data[card_id]

        if card.get("status") == "ACTIVE":
            return f"Error: Debit card '{card_id}' is already active."

        if card.get("status") != "FROZEN":
            return f"Error: Debit card '{card_id}' cannot be unfrozen. Current status: {card.get('status')}. Only FROZEN cards can be unfrozen."

        # Verify the linked account is still open
        account_id = card.get("account_id")
        if account_id and account_id in self.db.accounts.data:
            account = self.db.accounts.data[account_id]
            if account.get("status") != "OPEN":
                return f"Error: The linked checking account '{account_id}' is no longer open. Card cannot be unfrozen."

        # Unfreeze the card
        card["status"] = "ACTIVE"
        card["unfrozen_date"] = get_today_str()

        result_parts = [
            "Debit Card Unfrozen Successfully",
            f"Card ID: {card_id}",
            "Status: ACTIVE",
            f"Unfrozen Date: {get_today_str()}",
            "",
            "The card is now active and ready to use immediately.",
            "All transactions will process normally.",
        ]

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def clear_debit_card_fraud_alert_4892(self, card_id: str, reason: str) -> str:
        """Clear a fraud alert or velocity block on a debit card.

        Args:
            card_id (string): The debit card ID to clear the alert/block for
            reason (string): Reason for clearing: 'customer_verified' (for fraud alerts after customer verification) or 'velocity_clear' (for velocity blocks after identity verification)

        Returns:
            Fraud alert/velocity block cleared successfully.
        """
        if not card_id:
            return "Error: Missing required parameter: card_id."

        if not reason:
            return "Error: Missing required parameter: reason."

        valid_reasons = ["customer_verified", "velocity_clear"]
        if reason not in valid_reasons:
            return f"Error: Invalid reason '{reason}'. Must be one of: {', '.join(valid_reasons)}"

        if card_id not in self.db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."

        card = self.db.debit_cards.data[card_id]

        # Handle velocity block clear
        if reason == "velocity_clear":
            if not card.get("velocity_blocked", False):
                return f"Error: Debit card '{card_id}' does not have an active velocity block."

            card["velocity_blocked"] = False
            card["velocity_cleared_date"] = get_today_str()

            result_parts = [
                "Velocity Block Cleared Successfully",
                f"Card ID: {card_id}",
                f"Cleared Date: {get_today_str()}",
                "",
                "The card is now unblocked and ready for normal use.",
                "The velocity monitoring will continue - if the same unusual patterns recur,",
                "the card may be blocked again automatically.",
            ]
            return "\n".join(result_parts)

        # Handle fraud alert clear (customer_verified)
        if reason == "customer_verified":
            if not card.get("fraud_alert_active", False):
                return f"Error: Debit card '{card_id}' does not have an active fraud alert."

            # Check if bank-initiated - cannot clear those
            alert_source = card.get("alert_source")
            if alert_source == "bank_initiated":
                return "Error: BANK_INITIATED_ALERT - This fraud alert was initiated by the bank's fraud detection system and cannot be cleared by customer service agents. Please transfer the customer to the security team using transfer_to_human_agents."

            # Clear the fraud alert
            card["fraud_alert_active"] = False
            card["alert_source"] = None
            card["fraud_alert_cleared_date"] = get_today_str()

            result_parts = [
                "Fraud Alert Cleared Successfully",
                f"Card ID: {card_id}",
                f"Cleared Date: {get_today_str()}",
                "",
                "The fraud alert has been removed from the card.",
                "All transactions will process normally.",
                "",
                "Remind the customer to review recent transactions and report any unauthorized charges.",
            ]
            return "\n".join(result_parts)

        return "Error: Unexpected error processing the request."

    @is_discoverable_tool(ToolType.WRITE)
    def reset_debit_card_pin_6284(
        self,
        card_id: str,
        last_4_digits: str,
        new_pin: str,
    ) -> str:
        """Reset a debit card PIN when the customer has forgotten it.

        Args:
            card_id (string): The debit card ID to reset PIN for
            last_4_digits (string): Last 4 digits of the card number (for verification)
            new_pin (string): The new 4-digit PIN chosen by the customer

        Returns:
            Debit card PIN reset successfully.
        """
        if not all([card_id, last_4_digits, new_pin]):
            return "Error: Missing required parameters. Required: card_id, last_4_digits, new_pin."

        # Validate last 4 digits format
        if not last_4_digits.isdigit() or len(last_4_digits) != 4:
            return "Error: Last 4 digits must be exactly 4 digits."

        pin_error = _validate_pin(new_pin)
        if pin_error:
            return f"Error: {pin_error}"

        if card_id not in self.db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."

        card = self.db.debit_cards.data[card_id]

        # Verify last 4 digits match
        if card.get("last_4_digits") != last_4_digits:
            return "Error: Card verification failed. The last 4 digits do not match our records."

        # Check card status - can only reset PIN on ACTIVE cards
        if card.get("status") != "ACTIVE":
            return f"Error: Cannot reset PIN. Card status is {card.get('status')}. Only ACTIVE cards can have their PIN reset."

        # Reset the PIN and unlock the card
        card["pin_last_changed"] = get_today_str()
        card["pin_locked"] = False
        card["pin_attempts_remaining"] = 3

        result_parts = [
            "Debit Card PIN Reset Successfully",
            f"Card ID: {card_id}",
            f"PIN Changed: {get_today_str()}",
            "",
            "The new PIN is effective immediately.",
            "Your card has been unlocked and is ready to use.",
            "Customer can use the new PIN for ATM withdrawals and point-of-sale transactions.",
            "",
            "Security reminder: Never share your PIN with anyone.",
        ]

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.WRITE)
    def change_debit_card_pin_6285(
        self,
        card_id: str,
        current_pin: str,
        new_pin: str,
    ) -> str:
        """Change a debit card PIN when the customer knows their current PIN.

        Args:
            card_id (string): The debit card ID to change PIN for
            current_pin (string): The customer's current 4-digit PIN
            new_pin (string): The new 4-digit PIN chosen by the customer

        Returns:
            Debit card PIN changed successfully.
        """
        if not all([card_id, current_pin, new_pin]):
            return "Error: Missing required parameters. Required: card_id, current_pin, new_pin."

        # Validate current PIN format
        if not current_pin.isdigit() or len(current_pin) != 4:
            return "Error: Current PIN must be exactly 4 digits."

        pin_error = _validate_pin(new_pin)
        if pin_error:
            return f"Error: {pin_error}"

        # Check that new PIN is different from current
        if current_pin == new_pin:
            return "Error: New PIN must be different from current PIN."

        if card_id not in self.db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."

        card = self.db.debit_cards.data[card_id]

        # Check card status - can only change PIN on ACTIVE cards
        if card.get("status") != "ACTIVE":
            return f"Error: Cannot change PIN. Card status is {card.get('status')}. Only ACTIVE cards can have their PIN changed."

        # Change the PIN
        card["pin_last_changed"] = get_today_str()

        result_parts = [
            "Debit Card PIN Changed Successfully",
            f"Card ID: {card_id}",
            f"PIN Changed: {get_today_str()}",
            "",
            "The new PIN is effective immediately.",
            "Customer can use the new PIN for ATM withdrawals and point-of-sale transactions.",
            "",
            "Security reminder: Never share your PIN with anyone.",
        ]

        return "\n".join(result_parts)

    @is_discoverable_tool(ToolType.READ)
    def get_debit_cards_by_account_id_7823(self, account_id: str) -> str:
        """Retrieve all debit cards associated with a checking account. Returns card details including status, issue reason, and expiration date.

        Args:
            account_id (string): The checking account ID to retrieve debit cards for

        Returns:
            Debit cards retrieved successfully.
        """
        if not account_id:
            return "Error: Missing required parameter 'account_id'."

        # Verify the account exists and is a checking account
        if account_id not in self.db.accounts.data:
            return f"Error: Account '{account_id}' not found."

        account = self.db.accounts.data[account_id]
        account_class = account.get("class", "").lower()
        if account_class not in ["checking", "business_checking"]:
            return f"Error: Account '{account_id}' is not a checking account. Debit cards are only available for checking accounts."

        # Find all debit cards for this account
        account_cards = []
        for card_id, card in self.db.debit_cards.data.items():
            if card.get("account_id") == account_id:
                card_info = {"card_id": card_id}
                card_info.update(card)

                # Normalize field names for consistency
                if (
                    "last_4_digits" in card_info
                    and "card_number_last_4" not in card_info
                ):
                    card_info["card_number_last_4"] = card_info.pop("last_4_digits")
                if "issue_date" in card_info and "date_issued" not in card_info:
                    card_info["date_issued"] = card_info.pop("issue_date")
                elif "created_date" in card_info and "date_issued" not in card_info:
                    card_info["date_issued"] = card_info.pop("created_date")

                account_cards.append(card_info)

        if not account_cards:
            return f"No debit cards found for account '{account_id}'."

        # Sort by date issued (most recent first)
        account_cards.sort(key=lambda x: x.get("date_issued") or "", reverse=True)

        return json.dumps(account_cards, indent=2)

    @is_discoverable_tool(ToolType.WRITE)
    def request_temporary_debit_card_limit_increase_8374(
        self,
        card_id: str,
        limit_type: str,
        new_limit: int,
    ) -> str:
        """Request a temporary 24-hour increase to a debit card's daily ATM or purchase limit.

        Args:
            card_id (string): The debit card ID to increase limits for
            limit_type (string): Type of limit to increase: 'atm' for daily ATM withdrawal limit, 'purchase' for daily purchase limit
            new_limit (integer): The requested new temporary limit amount in dollars

        Returns:
            Temporary limit increase granted successfully.
        """
        from datetime import datetime

        if not card_id:
            return "Error: Missing required parameter: card_id."

        if not limit_type:
            return "Error: Missing required parameter: limit_type."

        if limit_type not in ["atm", "purchase"]:
            return f"Error: Invalid limit_type '{limit_type}'. Must be 'atm' or 'purchase'."

        if new_limit is None:
            return "Error: Missing required parameter: new_limit."

        try:
            new_limit = int(new_limit)
        except (ValueError, TypeError):
            return f"Error: new_limit must be an integer, got '{new_limit}'."

        if new_limit <= 0:
            return "Error: new_limit must be a positive amount."

        # Verify the card exists
        if card_id not in self.db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."

        card = self.db.debit_cards.data[card_id]

        # Check card status - must be ACTIVE
        if card.get("status") != "ACTIVE":
            return f"Error: Debit card '{card_id}' is not active. Current status: {card.get('status')}. Only ACTIVE cards can have limit increases."

        # Get the linked account
        account_id = card.get("account_id")
        if not account_id or account_id not in self.db.accounts.data:
            return f"Error: Could not find linked account for debit card '{card_id}'."

        account = self.db.accounts.data[account_id]

        # Check account status
        if account.get("status") != "OPEN":
            return f"Error: The linked account '{account_id}' is not in good standing. Account status: {account.get('status')}."

        # Check account age (must be at least 60 days old)
        date_opened_str = account.get("date_opened")
        if date_opened_str:
            try:
                date_opened = datetime.strptime(date_opened_str, "%m/%d/%Y")
                today = datetime.strptime(get_today_str(), "%m/%d/%Y")
                account_age_days = (today - date_opened).days
                if account_age_days < 60:
                    return f"Error: Account must be at least 60 days old for a temporary limit increase. Account age: {account_age_days} days."
            except ValueError:
                pass

        # Get current limit based on limit_type
        if limit_type == "atm":
            current_limit = card.get("daily_atm_limit")
            limit_field = "daily_atm_limit"
            limit_name = "Daily ATM Withdrawal Limit"

            if current_limit is None:
                account_level = account.get("level", "").lower()

                default_atm_limits = {
                    "blue account": 500,
                    "green account": 600,
                    "light blue account": 400,
                    "green fee-free account": 500,
                    "evergreen account": 750,
                    "bluest account": 1500,
                    "dark green account": 300,
                    "gold years account": 600,
                    "purple account": 1000,
                }

                # Teen account (Light Green) limits cannot be modified
                if "light green" in account_level:
                    return f"Error: Light Green Account (teen checking) cards have policy-based limits that cannot be modified. The daily ATM withdrawal limit of $150 is fixed by account policy for safety reasons. The customer may request the parent/guardian to withdraw cash from their own account if needed."

                current_limit = default_atm_limits.get(account_level)
                if current_limit is None:
                    return f"Error: Could not determine the default ATM limit for account type '{account.get('level')}'. Please verify the account type."
        else:  # purchase
            current_limit = card.get("daily_purchase_limit")
            limit_field = "daily_purchase_limit"
            limit_name = "Daily Purchase Limit"

        if current_limit is None:
            return f"Error: The card does not have a {limit_name.lower()} configured. This may be a restricted account type where limits are set by account policy and cannot be modified."

        # Check that new limit doesn't exceed 150% of current limit
        max_allowed = int(current_limit * 1.5)
        if new_limit > max_allowed:
            return f"Error: Requested limit ${new_limit} exceeds the maximum allowed temporary increase. Maximum temporary limit is ${max_allowed} (150% of current ${current_limit} limit)."

        # Check that new limit is actually higher than current
        if new_limit <= current_limit:
            return f"Error: Requested limit ${new_limit} is not higher than the current limit of ${current_limit}."

        # Grant the temporary increase
        original_limit = current_limit
        card[limit_field] = new_limit
        card[f"temporary_{limit_type}_limit_increase"] = True
        card[f"original_{limit_type}_limit"] = original_limit
        card[f"temporary_{limit_type}_limit_expires"] = get_today_str()

        result_parts = [
            f"Temporary {limit_name} Increase Granted Successfully",
            f"Card ID: {card_id}",
            f"Previous Limit: ${original_limit}",
            f"New Temporary Limit: ${new_limit}",
            f"Increase Amount: ${new_limit - original_limit}",
            "",
            "Important Information:",
            "- This temporary increase expires in 24 hours",
            f"- After expiration, the limit will revert to ${original_limit}",
            "- Only one temporary increase is allowed per 24-hour period",
            "",
            "Note: If the customer is at a third-party (non-Rho-Bank) ATM, that ATM may have its own",
            "per-transaction or daily limits that Rho-Bank cannot override. The customer may need to",
            "use a Rho-Bank ATM or make multiple smaller withdrawals at third-party ATMs.",
        ]

        return "\n".join(result_parts)


class KnowledgeUserTools(ToolKitBase):
    """Tools available to the user (customer) in the knowledge domain.

    The `db` attribute is the TransactionalDB which is used for DB state
    hashing during evaluation.
    """

    db: TransactionalDB

    def __init__(
        self,
        db: TransactionalDB,
    ) -> None:
        super().__init__(db)

    def _check_tool_given(self, tool_name: str) -> Optional[str]:
        """Check if a user discoverable tool was given by the agent.

        Returns None if tool was given, or an error message if not.
        """
        result = query_database_tool(
            "user_discoverable_tools", f'{{"tool_name": "{tool_name}"}}', db=self.db
        )
        if "No records found" in result:
            return (
                f"Error: Tool '{tool_name}' has not been given to you by the agent. "
                f"The agent must first use `give_discoverable_user_tool` to give this tool to you."
            )
        return None

    def _log_user_tool_call(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Log a user discoverable tool call to the database."""
        call_record = {
            "tool_name": tool_name,
            "arguments": args,
            "called_at": get_today_str(),
            "status": "CALLED",
        }
        call_record_id = generate_user_discoverable_tool_call_id(tool_name, args)
        add_to_db(
            "user_discoverable_tool_calls", call_record_id, call_record, db=self.db
        )

    # =========================================================================
    # User Discoverable Tools
    # These tools represent actions users take in the real world. The agent
    # gives them to the user via give_discoverable_user_tool, and the user
    # calls them directly. They are NOT included in the default tool list.
    # =========================================================================

    @is_discoverable_tool(ToolType.WRITE)
    def submit_cash_back_dispute_0589(self, user_id: str, transaction_id: str) -> str:
        """Submit a cash back dispute for a specific transaction.

        Args:
            user_id (string): The user's unique identifier in the system
            transaction_id (string): The unique identifier for the transaction with incorrect cash back

        Returns:
            Cash back dispute submitted successfully. Your case has been queued for review.
        """
        error = self._check_tool_given("submit_cash_back_dispute_0589")
        if error:
            return error

        args = {"user_id": user_id, "transaction_id": transaction_id}
        self._log_user_tool_call("submit_cash_back_dispute_0589", args)

        # Business logic from _handle_submit_cash_back_dispute
        dispute_id = generate_dispute_id(user_id, transaction_id)

        auto_resolve = False
        if hasattr(self.db, "task_config") and self.db.task_config.data:
            config = self.db.task_config.data.get("dispute_settings", {})
            auto_resolve = config.get("auto_resolve_disputes", False)

        if auto_resolve:
            dispute_record = {
                "dispute_id": dispute_id,
                "user_id": user_id,
                "transaction_id": transaction_id,
                "submitted_at": get_today_str(),
                "status": "RESOLVED",
                "resolution": "APPROVED",
            }
            status_msg = "Status: RESOLVED - The dispute has been reviewed and approved. The transaction rewards need to be updated."
        else:
            dispute_record = {
                "dispute_id": dispute_id,
                "user_id": user_id,
                "transaction_id": transaction_id,
                "submitted_at": get_today_str(),
                "status": "SUBMITTED",
            }
            status_msg = "Status: SUBMITTED - Your dispute has been queued for review."

        success = add_to_db(
            "cash_back_disputes", dispute_id, dispute_record, db=self.db
        )

        result = f"Cash back dispute submitted successfully. Your case has been queued for review.\n\nExecuted: submit_cash_back_dispute_0589\nArguments: {json.dumps(args, indent=2)}\n"
        if success:
            result += f"Dispute ID: {dispute_id}\n{status_msg}"
        else:
            result += (
                "Note: Dispute may have already been submitted for this transaction."
            )

        return result

    @is_discoverable_tool(ToolType.WRITE)
    def get_referral_link(self, user_id: str, card_name: str) -> str:
        """Generate a referral link for a specific credit card to share with friends or family.

        Args:
            user_id (string): The user's unique identifier in the system (the referrer)
            card_name (string): The name of the credit card to create a referral for (e.g., 'Gold Rewards Card')

        Returns:
            Referral link generated successfully. Share this link with the person you want to refer.
        """
        error = self._check_tool_given("get_referral_link")
        if error:
            return error

        args = {"user_id": user_id, "card_name": card_name}
        self._log_user_tool_call("get_referral_link", args)

        # Business logic from _handle_get_referral_link
        referral_id = generate_referral_link_id(user_id, card_name)

        referral_record = {
            "referral_id": referral_id,
            "referrer_id": user_id,
            "referred_account_type": card_name,
            "referral_status": "NO_PROGRESS",
            "date": get_today_str(),
        }

        success = add_to_db("referrals", referral_id, referral_record, db=self.db)

        result = f"Referral link generated successfully. Share this link with the person you want to refer.\n\nExecuted: get_referral_link\nArguments: {json.dumps(args, indent=2)}\n"
        if success:
            result += f"Referral ID: {referral_id}\nReferral link: https://rhobank.com/refer/{referral_id}"
        else:
            result += (
                "Note: A referral link for this card may have already been generated."
            )

        return result

    @is_discoverable_tool(ToolType.READ)
    def get_card_last_4_digits(self, credit_card_account_id: str) -> str:
        """Look up the last 4 digits of a credit card number.

        Args:
            credit_card_account_id (string): The credit card account ID to look up (e.g., 'cc_76ad9cc60e_gold')

        Returns:
            Card information retrieved successfully.
        """
        error = self._check_tool_given("get_card_last_4_digits")
        if error:
            return error

        args = {"credit_card_account_id": credit_card_account_id}
        self._log_user_tool_call("get_card_last_4_digits", args)

        # Business logic from _handle_get_card_last_4_digits
        result = query_database_tool(
            "credit_card_accounts",
            f'{{"account_id": "{credit_card_account_id}"}}',
            db=self.db,
        )

        if "No results found" in result or "No records found" in result:
            return f"Error: Credit card account '{credit_card_account_id}' not found."

        import hashlib

        hash_input = f"card_last4:{credit_card_account_id}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()
        last_4 = ""
        for char in hash_digest:
            if char.isdigit():
                last_4 += char
                if len(last_4) == 4:
                    break
        last_4 = last_4.ljust(4, "0")

        return f"Card information retrieved successfully.\n\nExecuted: get_card_last_4_digits\nArguments: {json.dumps(args, indent=2)}\nLast 4 digits of card: {last_4}"

    @is_discoverable_tool(ToolType.WRITE)
    def deposit_check_3847(self, account_id: str, check_amount: float) -> str:
        """Deposit a check into a checking or savings account. The user takes a photo of the check and submits it through their mobile banking app.

        Args:
            account_id (string): The bank account ID to deposit the check into
            check_amount (number): The amount of the check in USD

        Returns:
            Check deposited successfully. Funds will be available according to your account's deposit policy.
        """
        error = self._check_tool_given("deposit_check_3847")
        if error:
            return error

        args = {"account_id": account_id, "check_amount": check_amount}
        self._log_user_tool_call("deposit_check_3847", args)

        # Business logic from _handle_deposit_check
        try:
            check_amount = float(check_amount)
        except (ValueError, TypeError):
            return f"Error: Invalid check amount '{check_amount}'. Must be a number."

        if check_amount <= 0:
            return "Error: Check amount must be positive."

        if account_id not in self.db.accounts.data:
            return f"Error: Account '{account_id}' not found."

        account = self.db.accounts.data[account_id]

        if account.get("status") not in ("ACTIVE", "OPEN"):
            return f"Error: Account '{account_id}' is not active."

        def parse_balance(val: Any) -> float:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                return float(val.replace("$", "").replace(",", ""))
            return 0.0

        current_balance = parse_balance(
            account.get("current_holdings", account.get("balance", 0))
        )
        new_balance = current_balance + check_amount

        self.db.accounts.data[account_id]["current_holdings"] = f"${new_balance:.2f}"

        return (
            f"Check deposited successfully. Funds will be available according to your account's deposit policy.\n\n"
            f"Executed: deposit_check_3847\n"
            f"Arguments: {json.dumps(args, indent=2)}\n"
            f"Check deposit processed!\n"
            f"  - Account: {account_id}\n"
            f"  - Check Amount: ${check_amount:.2f}\n"
            f"  - Previous Balance: ${current_balance:.2f}\n"
            f"  - New Balance: ${new_balance:.2f}"
        )

    # Valid credit card types for applications
    VALID_CREDIT_CARD_TYPES = [
        "Bronze Rewards Card",
        "Business Bronze Rewards Card",
        "Business Gold Rewards Card",
        "Business Platinum Rewards Card",
        "Business Silver Rewards Card",
        "Crypto-Cash Back",
        "Diamond Elite Card",
        "EcoCard",
        "Gold Rewards Card",
        "Green Rewards Card",
        "Platinum Rewards Card",
        "Silver Rewards Card",
        "Silver Zoom Card",
    ]

    @is_tool(ToolType.WRITE)
    def apply_for_credit_card(
        self,
        card_type: str,
        customer_name: str,
        annual_income: float,
        rho_bank_subscription: bool = False,
    ) -> str:
        """Apply for a credit card.

        Args:
            card_type: Type of credit card
            customer_name: Full legal name
            annual_income: Annual income in USD
            rho_bank_subscription: Whether user has Rho-Bank+ subscription
        """
        # Validate card_type against known credit card types
        if card_type not in self.VALID_CREDIT_CARD_TYPES:
            return (
                f"Error: Invalid card_type '{card_type}'. "
                f"Must be one of: {self.VALID_CREDIT_CARD_TYPES}"
            )

        # Generate a deterministic application ID from the input parameters
        # This ensures the same inputs produce the same ID for environment evaluation
        application_id = generate_application_id(
            card_type, customer_name, annual_income, rho_bank_subscription
        )

        # Get today's date in MM/DD/YYYY format
        today = get_today_str()

        # Create the application record
        record = {
            "application_id": application_id,
            "card_type": card_type,
            "customer_name": customer_name,
            "annual_income": annual_income,
            "rho_bank_subscription": rho_bank_subscription,
            "status": "PENDING",
            "date": today,
        }

        # Add to the credit_card_applications table (in-memory via db_query)
        success = add_to_db(
            "credit_card_applications", application_id, record, db=self.db
        )

        if not success:
            return f"Failed to submit application: Record ID '{application_id}' may already exist."

        return (
            f"Credit card application submitted:\n"
            f"Your application has been successfully submitted. "
            f"You will receive a decision within 5-7 business days via email."
        )

    @is_tool(ToolType.WRITE)
    def submit_referral(self, user_id: str, account_type: str) -> str:
        """Submit a referral request to refer someone to open an account.

        Args:
            user_id: Your user ID (the referrer)
            account_type: The type of account you are referring someone to open
        """
        # Generate a deterministic 16-character hex referral ID from the input parameters
        # This ensures the same inputs produce the same ID for environment evaluation
        referral_id = generate_referral_id(user_id, account_type)

        # Get today's date in MM/DD/YYYY format
        today = get_today_str()

        # Create the referral record
        record = {
            "referral_id": referral_id,
            "referrer_id": user_id,
            "referred_account_type": account_type,
            "referral_status": "NO_PROGRESS",
            "date": today,
        }

        # Add to the referrals table (in-memory via db_query)
        success = add_to_db("referrals", referral_id, record, db=self.db)

        if not success:
            return f"Failed to submit referral: Record ID '{referral_id}' may already exist."

        return (
            f"Referral request submitted successfully!\n"
            f"  - Referral ID: {referral_id}\n"
            f"  - Referrer ID: {user_id}\n"
            f"  - Account Type: {account_type}\n"
            f"  - Status: NO_PROGRESS\n"
            f"  - Date: {today}\n\n"
            f"Share your referral ID with the person you're referring. "
            f"They will need to use this when applying for their account."
        )

    def query_database(self, database_name: str, constraints: str = "{}") -> str:
        """Query a database with constraints.

        Args:
            database_name: Name of the database to query
            constraints: JSON string of field constraints
        """
        return query_database_tool(database_name, constraints, db=self.db)

    @is_tool(ToolType.WRITE)
    def call_discoverable_user_tool(
        self, discoverable_tool_name: str, arguments: str = "{}"
    ) -> str:
        """Call a tool that was given to you by the agent.

        Use this when the agent has instructed you to perform an action using
        a discoverable tool. The agent will have told you the tool name and arguments.

        This simulates you performing the action in the real world (e.g., opening
        a webpage, navigating to a section, clicking a button).

        Args:
            discoverable_tool_name: The name of the discoverable tool to call (e.g., "open_webpage")
            arguments: JSON string of arguments for the tool (e.g., '{"url": "https://example.com"}')

        Returns:
            The result of executing the discoverable tool
        """
        # Check if the tool exists as a discoverable method
        if not self.has_discoverable_tool(discoverable_tool_name):
            return f"Error: Unknown discoverable tool '{discoverable_tool_name}'."

        # Parse arguments
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in arguments: {e}"

        # Get the method and validate arguments against method signature
        method = self.get_discoverable_tools()[discoverable_tool_name]
        sig = inspect.signature(method)

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.default is inspect.Parameter.empty and param_name not in args_dict:
                return f"Error: Missing required parameter: {param_name}"

        for arg_name in args_dict:
            if arg_name not in sig.parameters:
                return f"Error: Unexpected parameter: {arg_name}"

        # Call the method directly - it handles checking if tool was given,
        # logging the call, and executing the business logic
        try:
            return method(**args_dict)
        except TypeError as e:
            return f"Error: Invalid arguments for tool '{discoverable_tool_name}': {e}"

    @is_tool(ToolType.READ)
    def list_discoverable_user_tools(self) -> str:
        """List all tools that have been given to you by the agent.

        Use this to see what actions the agent has instructed you to perform.

        Returns:
            A list of tools that have been given to you
        """
        result = query_database_tool("user_discoverable_tools", "{}", db=self.db)

        if "No results found" in result:
            return "No tools have been given to you yet by the agent."

        return f"Tools given to you by the agent:\n{result}"

    @is_tool(ToolType.WRITE)
    def request_human_agent_transfer(self) -> str:
        """Request to be transferred to a human agent for assistance.

        Use this when you want to speak with a real human agent instead of
        the automated system. Each request will be logged and processed.

        Returns:
            Confirmation that your transfer request has been submitted
        """
        # Record this transfer request in the database so the agent can track the count
        today = get_today_str()

        # Query existing requests to get count
        existing_requests = query_database_tool(
            "human_transfer_requests", "{}", db=self.db
        )

        # Count existing requests (simple count based on results)
        if (
            "No records found" in existing_requests
            or "No results found" in existing_requests
        ):
            request_count = 1
        else:
            # Count the number of request entries
            request_count = existing_requests.count("request_id") + 1

        # Create a new request record
        request_id = f"transfer_request_{request_count}"
        record = {
            "request_id": request_id,
            "request_number": request_count,
            "requested_at": today,
            "status": "PENDING",
        }

        add_to_db("human_transfer_requests", request_id, record, db=self.db)

        return (
            f"Transfer request #{request_count} submitted.\n"
            f"The agent will process your request."
        )

    # Reward rates dictionary based on profile.json
    # Maps credit card type -> {category -> reward_percentage}
    # "default" key is used when category doesn't have a special rate
    CREDIT_CARD_REWARDS = {
        # Personal Credit Cards
        "Bronze Rewards Card": {
            "default": 1.0  # 1% on all purchases
        },
        "Silver Rewards Card": {
            "Travel": 4.0,  # 4% on travel
            "Software": 4.0,  # 4% on software
            "default": 1.0,  # 1% on other purchases
        },
        "Gold Rewards Card": {
            "default": 2.5  # 2.5% on all purchases
        },
        "Platinum Rewards Card": {
            "default": 10.0  # 10% on all purchases
        },
        # Business Credit Cards
        "Business Bronze Rewards Card": {
            "default": 1.0  # 1% on all purchases
        },
        "Business Silver Rewards Card": {
            "Travel": 10.0,  # 10% on travel
            "Software": 10.0,  # 10% on software
            "default": 1.0,  # 1% on other purchases
        },
        "Green Rewards Card": {
            "Sustainable": 3.0,  # 3% on sustainable/eco-friendly merchants
            "default": 1.0,  # 1% on other purchases
        },
        "Business Gold Rewards Card": {
            "Operations": 2.5,  # 2.5% on operations spending
            "default": 1.0,  # 1% on other purchases
        },
        "Business Platinum Rewards Card": {
            "Travel": 4.0,  # 4% on travel
            "Software": 4.0,  # 4% on software
            "Media": 4.0,  # 4% on media advertising
            "default": 1.5,  # 1.5% on other purchases
        },
        "Silver Zoom Card": {
            "Transportation": 3.0,  # 3% on transportation/logistics
            "default": 1.0,  # 1% on other purchases
        },
        "Diamond Elite Card": {
            "default": 5.0  # 5% on all purchases (invitation-only)
        },
        # EcoCard uses points (5 pts/$ green, 1 pt/$ other) at $0.01/pt conversion
        # Effective rates: 5% green, 1% other
        "EcoCard": {
            "Green": 5.0,  # 5 pts × $0.01 = 5% equivalent on green purchases
            "default": 1.0,  # 1 pt × $0.01 = 1% equivalent on other purchases
        },
        "Crypto-Cash Back": {
            "default": 2.0  # 2% on all purchases (redeemable to crypto wallet)
        },
    }

    @is_tool(ToolType.WRITE)
    def submit_transaction(
        self,
        user_id: str,
        credit_card_type: str,
        merchant_name: str,
        amount: float,
        category: str,
    ) -> str:
        """Submit a credit card transaction.

        Args:
            user_id: Your user ID
            credit_card_type: Type of credit card used (e.g., "Bronze Rewards Card", "Gold Rewards Card")
            merchant_name: Name of the merchant where the purchase was made
            amount: Transaction amount in USD (e.g., 127.43)
            category: Transaction category (e.g., "Groceries", "Dining", "Travel", "Software", "Entertainment", "Utilities", "Shopping")
        """
        # Validate credit card type
        if credit_card_type not in self.CREDIT_CARD_REWARDS:
            available_cards = list(self.CREDIT_CARD_REWARDS.keys())
            return f"Error: Unknown credit card type '{credit_card_type}'. Available types: {available_cards}"

        # Generate a deterministic transaction ID
        transaction_id = generate_transaction_id(
            user_id, credit_card_type, merchant_name, amount, category
        )

        # Get today's date in MM/DD/YYYY format
        today = get_today_str()

        # Calculate rewards based on credit card type and category
        card_rewards = self.CREDIT_CARD_REWARDS[credit_card_type]
        reward_rate = card_rewards.get(category, card_rewards["default"])

        # Calculate points earned (1 point = 1 cent of cashback)
        # reward_rate is percentage, so 1% means 1 point per dollar
        points_earned = int(amount * reward_rate)

        # Create the transaction record
        record = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "credit_card_type": credit_card_type,
            "merchant_name": merchant_name,
            "transaction_amount": f"${amount:.2f}",
            "transaction_date": today,
            "category": category,
            "status": "COMPLETED",
            "rewards_earned": f"{points_earned} points",
        }

        # Add to the credit_card_transaction_history table
        success = add_to_db(
            "credit_card_transaction_history", transaction_id, record, db=self.db
        )

        if not success:
            return f"Failed to submit transaction: Record ID '{transaction_id}' may already exist."

        return (
            f"Transaction submitted successfully!\n"
            f"  - Transaction ID: {transaction_id}\n"
            f"  - User ID: {user_id}\n"
            f"  - Card Type: {credit_card_type}\n"
            f"  - Merchant: {merchant_name}\n"
            f"  - Amount: ${amount:.2f}\n"
            f"  - Category: {category}\n"
            f"  - Date: {today}\n"
            f"  - Rewards Earned: {points_earned} points ({reward_rate}% cashback rate)\n"
        )

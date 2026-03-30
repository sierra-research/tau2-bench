# FastWorkflow Agent Adapter for Tau2 Bench
import contextlib
import logging
import os
import queue
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import dotenv_values

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

logger = logging.getLogger(__name__)

# Default paths for FastWorkflow config files, overridable via environment variables.
FW_ENV_PATH = os.environ.get("FASTWORKFLOW_ENV", "examples/fastworkflow.env")
FW_PASSWORDS_PATH = os.environ.get(
    "FASTWORKFLOW_PASSWORDS_ENV", "examples/fastworkflow.passwords.env"
)


class FastWorkflowAgentAdapter(HalfDuplexAgent):
    """
    FastWorkflow agent adapter that integrates with Tau2 Bench.

    This adapter bridges FastWorkflow's command trace queue architecture with
    Tau2 Bench's message-based orchestration system. It converts between:
    - FastWorkflow's command traces → Tau2 AssistantMessage with tool calls
    - Tau2 UserMessage/ToolMessage → FastWorkflow's user_message_queue

    The adapter operates in a stateless manner, processing FastWorkflow events
    during each generate_next_message() call and converting them to Tau2 messages.
    """

    # Maps tau2 domain names to their workflow folder and base domain.
    # The base_domain is used for DB loading; the workflow_folder is the
    # directory name under examples/.
    DOMAIN_CONFIG = {
        "retail": {"workflow_folder": "retail_workflow", "base_domain": "retail"},
        "airline": {"workflow_folder": "airline_workflow", "base_domain": "airline"},
        "telecom": {"workflow_folder": "telecom_workflow", "base_domain": "telecom"},
        "telecom-workflow": {
            "workflow_folder": "telecom_workflow",
            "base_domain": "telecom",
        },
    }

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        workflow_type: str = "retail",
        env_db: Any = None,
    ):
        """
        Initialize the FastWorkflow adapter.

        Args:
            tools: List of Tau2 Tool objects available to the agent
            domain_policy: Domain-specific policy text
            workflow_type: Type of workflow to use (retail/airline/telecom)
            env_db: Reference to the Tau2 environment's DB object.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.workflow_type = workflow_type

        # Reference to the environment's live DB
        self._env_db = env_db

        # Find the workflow path
        self.workflow_path = self._find_workflow_path(workflow_type)

        # FastWorkflow session (initialized per task)
        self.fastworkflow = None
        self.chat_session = None
        self.is_initialized = False

    def _find_workflow_path(self, workflow_type: str) -> str:
        """Find the path to the specified workflow."""
        config = self.DOMAIN_CONFIG.get(workflow_type)
        if config is None:
            raise ValueError(
                f"Unsupported domain '{workflow_type}'. "
                f"Supported domains: {list(self.DOMAIN_CONFIG.keys())}"
            )

        current_dir = os.getcwd()
        workflow_folder = config["workflow_folder"]
        workflow_path = os.path.join(current_dir, "examples", workflow_folder)

        if os.path.exists(workflow_path):
            return workflow_path

        raise FileNotFoundError(
            f"Could not find {workflow_type} workflow. Expected at: {workflow_path}. "
            f"Run 'fastworkflow examples fetch {workflow_folder}' to install it."
        )

    def _load_domain_db(self):
        """Load a database copy for FastWorkflow's internal use.

        Each domain has its own DB class and path. The returned object is an
        independent in-memory copy that FW commands can mutate without
        affecting Tau2's environment database.
        """
        if self._env_db is not None:
            return self._env_db.model_copy(deep=True)

        base_domain = self.DOMAIN_CONFIG[self.workflow_type]["base_domain"]

        # Fallback: load fresh from disk (no initialization state)
        if base_domain == "retail":
            from tau2.domains.retail.data_model import RetailDB
            from tau2.domains.retail.utils import RETAIL_DB_PATH

            return RetailDB.load(RETAIL_DB_PATH)
        elif base_domain == "airline":
            from tau2.domains.airline.data_model import FlightDB
            from tau2.domains.airline.utils import AIRLINE_DB_PATH

            return FlightDB.load(AIRLINE_DB_PATH)
        elif base_domain == "telecom":
            from tau2.domains.telecom.data_model import TelecomDB
            from tau2.domains.telecom.utils import TELECOM_DB_PATH

            return TelecomDB.load(TELECOM_DB_PATH)
        else:
            raise ValueError(f"Unsupported base domain: {base_domain}")

    def _sync_fw_db(self):
        """Refresh FastWorkflow's internal DB from the environment's live DB."""
        if self._env_db is None or not self.is_initialized or not self.chat_session:
            return
        try:
            workflow = self.chat_session.get_active_workflow()
            if workflow is not None:
                workflow.context["db"] = self._env_db.model_copy(deep=True)
        except Exception as e:
            logger.debug(f"Could not sync FW DB: {e}")

    def _initialize_fastworkflow(self, initial_message: Optional[str] = None):
        """Initialize FastWorkflow session if not already initialized."""
        if self.is_initialized:
            return

        try:
            # Load environment variables
            env_vars = {
                **dotenv_values(FW_ENV_PATH),
                **dotenv_values(FW_PASSWORDS_PATH),
            }

            # Import and initialize FastWorkflow
            import fastworkflow

            self.fastworkflow = fastworkflow
            fastworkflow.init(env_vars=env_vars)

            # Clear any lingering workflow stack
            with contextlib.suppress(Exception):
                fastworkflow.ChatSession.clear_workflow_stack()

            # Create chat session and run in agent mode
            run_as_agent = True
            self.chat_session = fastworkflow.ChatSession(run_as_agent=run_as_agent)

            # Load a fresh, isolated DB copy for FastWorkflow's internal use.
            # This is separate from Tau2's environment DB — FW commands mutate
            # this copy for their own reasoning, while Tau2's orchestrator
            # executes the same tool calls against its own DB for evaluation.
            fw_db = self._load_domain_db()

            # Start workflow if we have an initial message.
            # Domain policy is passed via workflow_context so that
            # workflow_agent.py reads it from the active workflow's context.
            if initial_message:
                self.chat_session.start_workflow(
                    self.workflow_path,
                    workflow_context={
                        "db": fw_db,
                        "domain_policy": self.domain_policy,
                    },
                    startup_command=initial_message,
                    startup_action=None,
                    keep_alive=True,
                    project_folderpath=None,
                )

            self.is_initialized = True

        except Exception as e:
            logger.error(f"Error initializing FastWorkflow: {e}")
            raise

    def _to_plain_kwargs(self, params: Any) -> Dict[str, Any]:
        """Convert parameters to plain dict."""
        if params is None:
            return {}
        if isinstance(params, dict):
            return params
        # Handle pydantic models
        with contextlib.suppress(Exception):
            return params.model_dump()
        with contextlib.suppress(Exception):
            return params.dict()
        # Generic objects
        with contextlib.suppress(Exception):
            return dict(params)
        return {}

    def _drain_command_trace(
        self,
        max_drain: int = 200,
    ) -> List[Tuple[str, Dict[str, Any], str, bool]]:
        """
        Drain the command_trace_queue and return executed commands.

        Returns:
            List of tuples (command_name, parameters, response_text, success)
        """
        if not self.is_initialized or not self.chat_session:
            return []

        executed_commands = []
        processed = 0

        while processed < max_drain:
            try:
                evt = self.chat_session.command_trace_queue.get_nowait()
            except queue.Empty:
                break

            processed += 1

            # Check direction (skip AGENT_TO_WORKFLOW)
            is_agent_to_workflow = (
                getattr(evt, "direction", None)
                == getattr(
                    self.fastworkflow, "CommandTraceEventDirection", None
                ).AGENT_TO_WORKFLOW
                if hasattr(self.fastworkflow, "CommandTraceEventDirection")
                else False
            )

            if not is_agent_to_workflow:
                cmd_name = getattr(evt, "command_name", None)
                params = self._to_plain_kwargs(getattr(evt, "parameters", None))
                response_text = getattr(evt, "response_text", None)
                success = getattr(evt, "success", True)

                if isinstance(cmd_name, str) and len(cmd_name) > 0:
                    executed_commands.append((cmd_name, params, response_text, success))

        return executed_commands

    def _drain_agent_outputs(
        self,
        max_drain: int = 200,
    ) -> List[str]:
        """
        Drain FastWorkflow's command_output_queue to get agent questions/responses.

        Returns:
            List of agent text outputs
        """
        if not self.is_initialized or not self.chat_session:
            return []

        agent_texts = []
        processed = 0

        while processed < max_drain:
            try:
                out = self.chat_session.command_output_queue.get_nowait()
            except queue.Empty:
                break

            processed += 1

            # Extract text from CommandOutput objects
            texts = []
            if hasattr(out, "command_responses") and isinstance(
                out.command_responses, list
            ):
                for cr in out.command_responses:
                    txt = getattr(cr, "response", None)
                    if isinstance(txt, str) and txt.strip():
                        texts.append(txt.strip())
            elif isinstance(out, str) and out.strip():
                texts.append(out.strip())

            if texts:
                agent_texts.append("\n".join(texts))

        return agent_texts

    def _push_user_message(self, user_text: str):
        """Push user message to FastWorkflow's user_message_queue."""
        if self.is_initialized and self.chat_session:
            self.chat_session.user_message_queue.put(user_text)

    def _drain_queues_until_idle(
        self,
        idle_limit: int = 200,
        max_num_steps: int = 5000,
        sleep_interval: float = 0.15,
    ) -> Tuple[List[Tuple[str, Dict[str, Any], str, bool]], List[str]]:
        """
        Drain FastWorkflow's command_trace and command_output queues until idle.

        Args:
            idle_limit: Number of consecutive empty cycles before stopping.
            max_num_steps: Safety limit to prevent infinite loops.
            sleep_interval: Seconds to sleep between drain attempts.

        Returns:
            Tuple of (all_commands, all_texts) collected from the queues.
        """
        idle_cycles = 0
        all_commands = []
        all_texts = []

        for _ in range(max_num_steps):
            commands = self._drain_command_trace(max_drain=200)
            texts = self._drain_agent_outputs(max_drain=200)

            if commands or texts:
                all_commands.extend(commands)
                all_texts.extend(texts)
                idle_cycles = 0
            else:
                idle_cycles += 1

            if idle_cycles >= idle_limit:
                break

            time.sleep(sleep_interval)

        return all_commands, all_texts

    def _convert_commands_to_tool_calls(
        self, commands: List[Tuple[str, Dict[str, Any], str, bool]]
    ) -> List[ToolCall]:
        """Convert FastWorkflow commands to Tau2 ToolCall objects.

        Skips commands that:
        - Don't match any Tau2 tool (e.g. FW-internal commands like
          ``ErrorCorrection/abort``)
        - Have ``success=False`` (e.g. parameter extraction failures that
          would produce empty/invalid arguments)
        """
        valid_names = {tool.name for tool in self.tools}
        tool_calls = []
        for cmd_name, params, response_text, success in commands:
            if cmd_name not in valid_names:
                continue
            if not success:
                continue
            tool_call = ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=cmd_name,
                arguments=params,
                requestor="assistant",
            )
            tool_calls.append(tool_call)
        return tool_calls

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: Any
    ) -> Tuple[AssistantMessage, Any]:
        """
        Generate the next message from FastWorkflow agent.

        This method:
        1. Initializes FastWorkflow on first call with UserMessage
        2. Feeds user/tool messages to FastWorkflow
        3. Drains FastWorkflow's queues to get agent responses/tool calls
        4. Converts to Tau2 AssistantMessage

        Flow:
        - On UserMessage: start/continue workflow, drain queues for commands + texts.
          If commands found, return them as tool_calls and stash texts in state.
          If only texts found, return as content.
        - On ToolMessage/MultiToolMessage: FastWorkflow already executed the tools
          internally, so queues are empty. Return the stashed text response from
          the prior drain instead of uselessly waiting on empty queues.

        Args:
            message: UserMessage, ToolMessage, or MultiToolMessage from orchestrator
            state: Agent state (maintained as dict with message history)

        Returns:
            Tuple of (AssistantMessage, updated_state)
        """
        try:
            # Initialize state if needed
            if state is None:
                state = {
                    "message_history": [],
                    "last_tool_calls": [],
                    "pending_texts": [],
                }
            # Ensure pending_texts exists for backward compat
            if "pending_texts" not in state:
                state["pending_texts"] = []

            # Sync FW's DB with the environment's current state.
            self._sync_fw_db()

            # Handle UserMessage (initial or follow-up)
            if isinstance(message, UserMessage):
                user_content = message.content

                # First message - initialize workflow
                if not self.is_initialized:
                    self._initialize_fastworkflow(initial_message=user_content)
                    state["message_history"].append(message)
                else:
                    self._push_user_message(user_content)
                    state["message_history"].append(message)

                # Drain FastWorkflow queues after user message
                all_commands, all_texts = self._drain_queues_until_idle()

                # Determine what to return
                if all_commands:
                    response_tool_calls = self._convert_commands_to_tool_calls(
                        all_commands
                    )
                    if response_tool_calls:
                        state["last_tool_calls"] = response_tool_calls
                        # Stash text outputs for when ToolMessages come back
                        state["pending_texts"] = all_texts
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content=None,
                            tool_calls=response_tool_calls,
                            cost=0.0,
                        )
                    elif all_texts:
                        # Commands were emitted but none mapped to Tau2 tools.
                        # Fall back to text to avoid emitting empty tool_calls.
                        response_content = "\n".join(all_texts)
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content=response_content,
                            tool_calls=None,
                            cost=0.0,
                        )
                    else:
                        print(
                            "FastWorkflow produced commands but no valid tool calls or texts"
                        )
                        logger.warning(
                            "FastWorkflow produced commands but no valid tool calls or texts"
                        )
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content="I'm still processing your request. Could you please repeat or clarify?",
                            tool_calls=None,
                            cost=0.0,
                        )
                elif all_texts:
                    response_content = "\n".join(all_texts)
                    assistant_msg = AssistantMessage(
                        role="assistant",
                        content=response_content,
                        tool_calls=None,
                        cost=0.0,
                    )
                else:
                    print("No response from FastWorkflow after draining queues")
                    logger.warning(
                        "No response from FastWorkflow after draining queues"
                    )
                    assistant_msg = AssistantMessage(
                        role="assistant",
                        content="I'm still processing your request. Could you please repeat or clarify?",
                        tool_calls=None,
                        cost=0.0,
                    )

            # Handle ToolMessage / MultiToolMessage (results from environment)
            elif isinstance(message, ToolMessage) or hasattr(message, "tool_messages"):
                if isinstance(message, ToolMessage):
                    state["message_history"].append(message)
                else:
                    for tm in message.tool_messages:
                        state["message_history"].append(tm)

                # FastWorkflow already executed these tools internally.
                # Return the stashed text responses from the prior drain.
                pending_texts = state.get("pending_texts", [])

                # Do a quick non-blocking drain in case FastWorkflow produced
                # additional output after the commands completed
                extra_commands, extra_texts = self._drain_queues_until_idle(
                    idle_limit=20
                )
                if extra_texts:
                    pending_texts = extra_texts  # prefer fresher output

                if extra_commands:
                    # FastWorkflow generated more tool calls after the previous batch
                    response_tool_calls = self._convert_commands_to_tool_calls(
                        extra_commands
                    )
                    if response_tool_calls:
                        state["last_tool_calls"] = response_tool_calls
                        state["pending_texts"] = pending_texts
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content=None,
                            tool_calls=response_tool_calls,
                            cost=0.0,
                        )
                    elif pending_texts:
                        response_content = "\n".join(pending_texts)
                        state["pending_texts"] = []
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content=response_content,
                            tool_calls=None,
                            cost=0.0,
                        )
                    else:
                        logger.warning(
                            "FastWorkflow produced extra commands but no valid tool calls or texts"
                        )
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content="I've processed the information. How can I help you further?",
                            tool_calls=None,
                            cost=0.0,
                        )
                elif pending_texts:
                    response_content = "\n".join(pending_texts)
                    state["pending_texts"] = []  # clear after use
                    assistant_msg = AssistantMessage(
                        role="assistant",
                        content=response_content,
                        tool_calls=None,
                        cost=0.0,
                    )
                else:
                    # No pending texts and no new output - generate a summary
                    # from the tool results we received
                    logger.warning("No pending text after tool results, summarizing")
                    tool_summary_parts = []
                    if isinstance(message, ToolMessage):
                        tool_summary_parts.append(str(message.content))
                    elif hasattr(message, "tool_messages"):
                        for tm in message.tool_messages:
                            tool_summary_parts.append(str(tm.content))

                    # Push tool results to FastWorkflow so it can generate a response
                    summary_text = "\n".join(tool_summary_parts)
                    self._push_user_message(f"Tool results: {summary_text}")

                    # Drain again with standard timeout
                    retry_commands, retry_texts = self._drain_queues_until_idle()
                    if retry_commands:
                        response_tool_calls = self._convert_commands_to_tool_calls(
                            retry_commands
                        )
                        if response_tool_calls:
                            state["last_tool_calls"] = response_tool_calls
                            state["pending_texts"] = retry_texts
                            assistant_msg = AssistantMessage(
                                role="assistant",
                                content=None,
                                tool_calls=response_tool_calls,
                                cost=0.0,
                            )
                        elif retry_texts:
                            assistant_msg = AssistantMessage(
                                role="assistant",
                                content="\n".join(retry_texts),
                                tool_calls=None,
                                cost=0.0,
                            )
                        else:
                            assistant_msg = AssistantMessage(
                                role="assistant",
                                content="I've processed the information. How can I help you further?",
                                tool_calls=None,
                                cost=0.0,
                            )
                    elif retry_texts:
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content="\n".join(retry_texts),
                            tool_calls=None,
                            cost=0.0,
                        )
                    else:
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content="I've processed the information. How can I help you further?",
                            tool_calls=None,
                            cost=0.0,
                        )
            else:
                logger.warning(f"Unexpected message type: {type(message)}")
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content="I'm sorry, I encountered an unexpected situation. Could you please try again?",
                    tool_calls=None,
                    cost=0.0,
                )

            state["message_history"].append(assistant_msg)
            return assistant_msg, state

        except Exception as e:
            logger.error(f"Error in generate_next_message: {e}")
            traceback.print_exc()
            error_msg = AssistantMessage(
                role="assistant",
                content=f"I encountered an error: {str(e)}",
                tool_calls=None,
                cost=0.0,
            )
            return error_msg, state

    def reset(self) -> None:
        """
        Reset the FastWorkflow session for a new task.

        This ensures task isolation by clearing the workflow stack and
        resetting the initialized flag so the next task starts fresh.
        """
        if self.is_initialized and self.chat_session:
            with contextlib.suppress(Exception):
                # Clear workflow stack
                if self.fastworkflow:
                    self.fastworkflow.ChatSession.clear_workflow_stack()
                # Cleanup session
                self.chat_session = None
        self.is_initialized = False

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[Any] = None,
    ) -> None:
        """
        Stop the agent and cleanup resources.

        Args:
            message: Optional last message
            state: Optional agent state
        """
        self.reset()

    def get_init_state(
        self,
        message_history: Optional[List[Message]] = None,
    ) -> Any:
        """
        Get the initial state of the agent.

        Args:
            message_history: Optional message history to initialize from

        Returns:
            Initial agent state (dict)
        """
        return {
            "message_history": list(message_history) if message_history else [],
            "last_tool_calls": [],
            "pending_texts": [],
        }

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """
        Check if the message is a stop signal.

        Args:
            message: Assistant message to check

        Returns:
            True if message contains stop signal
        """
        if message.content and "###STOP###" in message.content:
            return True
        return False

    def set_seed(self, seed: int):
        """
        Set the seed for the agent.

        Args:
            seed: Random seed
        """
        pass


def create_fastworkflow_agent(tools, domain_policy, **kwargs):
    """Factory function for FastWorkflowAgentAdapter.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments. Supports:
            - domain (str): Domain name passed by build_agent (e.g. "airline", "retail", "telecom").
            - workflow_type (str): Explicit override for workflow type.
            - env_db: Reference to the environment's DB object. When provided,
              FastWorkflow will deep-copy this (already-initialized) DB instead
              of loading a fresh one from disk.
    """
    # Prefer explicit workflow_type, fall back to domain from build system
    workflow_type = kwargs.get("workflow_type") or kwargs.get("domain", "retail")
    env_db = kwargs.get("env_db")
    return FastWorkflowAgentAdapter(
        tools=tools,
        domain_policy=domain_policy,
        workflow_type=workflow_type,
        env_db=env_db,
    )

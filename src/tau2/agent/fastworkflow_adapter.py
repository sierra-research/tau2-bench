# FastWorkflow Agent Adapter for Tau2 Bench
import contextlib
import json
import logging
import os
import time
import queue
import copy
from typing import List, Dict, Any, Optional, Tuple
from dotenv import dotenv_values

from tau2.agent.base import BaseAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import get_cost

logger = logging.getLogger(__name__)


def _json_deepcopy(obj: Any) -> Any:
    """Safe deep copy using JSON serialization with fallback."""
    with contextlib.suppress(Exception):
        return json.loads(json.dumps(obj))
    return copy.deepcopy(obj)


class FastWorkflowAgentAdapter(BaseAgent):
    """
    FastWorkflow agent adapter that integrates with Tau2 Bench.
    
    This adapter bridges FastWorkflow's command trace queue architecture with
    Tau2 Bench's message-based orchestration system. It converts between:
    - FastWorkflow's command traces → Tau2 AssistantMessage with tool calls
    - Tau2 UserMessage/ToolMessage → FastWorkflow's user_message_queue
    
    The adapter operates in a stateless manner, processing FastWorkflow events
    during each generate_next_message() call and converting them to Tau2 messages.
    """
    
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        model: str = "mistral-small-latest",
        provider: str = "mistral",
        temperature: float = 0.0,
        use_reasoning: bool = True,
        workflow_type: str = "retail",
        **kwargs
    ):
        """
        Initialize the FastWorkflow adapter.
        
        Args:
            tools: List of Tau2 Tool objects available to the agent
            domain_policy: Domain-specific policy text
            model: LLM model name (default: mistral-small-latest)
            provider: LLM provider (default: mistral)
            temperature: Model temperature (default: 0.0)
            use_reasoning: Enable reasoning mode (default: True)
            workflow_type: Type of workflow to use (retail/airline/telecom)
            **kwargs: Additional configuration
        """
        self.tools = tools
        self.domain_policy = domain_policy
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.workflow_type = workflow_type
        
        # Find the workflow path
        self.workflow_path = self._find_workflow_path(workflow_type)
        
        # FastWorkflow session (initialized per task)
        self.fastworkflow = None
        self.chat_session = None
        self.is_initialized = False
        
        logger.info(f"FastWorkflow adapter initialized for {workflow_type} domain")
        logger.info(f"Model: {model} from {provider}")
        logger.info(f"Workflow path: {self.workflow_path}")
    
    def _find_workflow_path(self, workflow_type: str) -> str:
        """Find the path to the specified workflow."""
        current_dir = os.getcwd()
        workflow_path = os.path.join(current_dir, "examples", f"{workflow_type}_workflow")
        
        if os.path.exists(workflow_path):
            return workflow_path
        
        raise FileNotFoundError(
            f"Could not find {workflow_type} workflow. Expected at: {workflow_path}. "
            f"Run 'fastworkflow examples fetch {workflow_type}_workflow' to install it."
        )
    
    def _initialize_fastworkflow(self, initial_message: Optional[str] = None):
        """Initialize FastWorkflow session if not already initialized."""
        if self.is_initialized:
            return
        
        try:
            # Load environment variables
            env_vars = {
                **dotenv_values('examples/fastworkflow.env'),
                **dotenv_values('examples/fastworkflow.passwords.env')
            }
            
            # Import and initialize FastWorkflow
            import fastworkflow
            self.fastworkflow = fastworkflow
            fastworkflow.init(env_vars=env_vars)
            logger.info("✅ FastWorkflow initialized")
            
            # Clear any lingering workflow stack
            with contextlib.suppress(Exception):
                fastworkflow.ChatSession.clear_workflow_stack()
            
            # Create chat session
            run_as_agent = True
            self.chat_session = fastworkflow.ChatSession(run_as_agent=run_as_agent)
            logger.info("✅ Chat session created")
            
            # Start workflow if we have an initial message
            if initial_message:
                self.chat_session.start_workflow(
                    self.workflow_path,
                    workflow_context=None,
                    startup_command=initial_message,
                    startup_action=None,
                    keep_alive=True,
                    project_folderpath=None
                )
                logger.info(f"✅ Workflow started with message: {initial_message}")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Error initializing FastWorkflow: {e}")
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
                == getattr(self.fastworkflow, "CommandTraceEventDirection", None).AGENT_TO_WORKFLOW
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
            if hasattr(out, "command_responses") and isinstance(out.command_responses, list):
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
            logger.debug(f"Pushed user message to FastWorkflow: {user_text[:100]}...")
    
    def _convert_commands_to_tool_calls(
        self,
        commands: List[Tuple[str, Dict[str, Any], str, bool]]
    ) -> List[ToolCall]:
        """Convert FastWorkflow commands to Tau2 ToolCall objects."""
        tool_calls = []
        for cmd_name, params, response_text, success in commands:
            tool_call = ToolCall(
                id=f"call_{len(tool_calls)}",
                name=cmd_name,
                arguments=params,
                requestor="assistant"
            )
            tool_calls.append(tool_call)
        return tool_calls
    
    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: Any
    ) -> Tuple[AssistantMessage, Any]:
        """
        Generate the next message from FastWorkflow agent.
        
        This method:
        1. Initializes FastWorkflow on first call with UserMessage
        2. Feeds user/tool messages to FastWorkflow
        3. Drains FastWorkflow's queues to get agent responses/tool calls
        4. Converts to Tau2 AssistantMessage
        
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
                    "awaiting_response": False,
                    "last_tool_calls": []
                }
            
            # Handle UserMessage (initial or follow-up)
            if isinstance(message, UserMessage):
                user_content = message.content
                
                # First message - initialize workflow
                if not self.is_initialized:
                    logger.info(f"🎯 Starting FastWorkflow with: {user_content}")
                    self._initialize_fastworkflow(initial_message=user_content)
                    state["message_history"].append(message)
                else:
                    # Subsequent user message - push to FastWorkflow
                    logger.info(f"👤 User says: {user_content}")
                    self._push_user_message(user_content)
                    state["message_history"].append(message)
            
            # Handle ToolMessage (results from environment)
            elif isinstance(message, ToolMessage):
                logger.info(f"🔧 Tool result: {str(message.content)[:100]}...")
                state["message_history"].append(message)
                # FastWorkflow already executed the tool, just tracking for state
            
            # Handle MultiToolMessage
            elif hasattr(message, "tool_messages"):
                logger.info(f"🔧 Multiple tool results received")
                for tm in message.tool_messages:
                    state["message_history"].append(tm)
                # FastWorkflow already executed the tools, just tracking for state
            
            # Drain FastWorkflow queues
            max_num_steps = 5000  # Safety limit to prevent infinite loops
            idle_limit = 150
            idle_cycles = 0
            steps_taken = 0
            all_commands = []
            all_texts = []
            
            # Interaction loop
            for _ in range(max_num_steps):
                commands = self._drain_command_trace(max_drain=200)
                texts = self._drain_agent_outputs(max_drain=200)
                
                progressed = 0
                if commands or texts:
                    progressed = 1
                    all_commands.extend(commands)
                    all_texts.extend(texts)
                    idle_cycles = 0  # Reset on activity
                else:
                    idle_cycles += 1
                
                steps_taken += progressed
                
                # Exit when idle too long
                if idle_cycles >= idle_limit:
                    break
                
                time.sleep(0.15)
            
            # Determine what to return
            response_content = None
            response_tool_calls = None
            
            # If we have text outputs, use the last one
            if all_texts:
                response_content = all_texts[-1]
                logger.info(f"🤖 Agent says: {response_content[:100]}...")
            
            # If we have commands, convert to tool calls
            if all_commands:
                response_tool_calls = self._convert_commands_to_tool_calls(all_commands)
                logger.info(f"🔧 Agent making {len(response_tool_calls)} tool calls")
                state["last_tool_calls"] = response_tool_calls
            
            # Create AssistantMessage
            # Messages must have EITHER content OR tool_calls, not both
            if response_tool_calls:
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=response_tool_calls,
                    cost=0.0
                )
            elif response_content:
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content=response_content,
                    tool_calls=None,
                    cost=0.0
                )
            else:
                logger.warning(f"⚠️ No response after {idle_limit} idle cycles, returning default message")
                assistant_msg = AssistantMessage(
                    role="assistant",
                    content="I apologize, but I am unable to assist with this request at this time.",
                    tool_calls=None,
                    cost=0.0
                )
            
            state["message_history"].append(assistant_msg)
            return assistant_msg, state
            
        except Exception as e:
            logger.error(f"❌ Error in generate_next_message: {e}")
            import traceback
            traceback.print_exc()
            error_msg = AssistantMessage(
                role="assistant",
                content=f"I encountered an error: {str(e)}",
                tool_calls=None,
                cost=0.0
            )
            return error_msg, state
    
    def reset(self) -> None:
        """
        Reset the FastWorkflow session for a new task.
        
        This ensures task isolation by clearing the workflow stack and
        resetting the initialized flag so the next task starts fresh.
        """
        logger.info("🔄 Resetting FastWorkflow agent for new task")
        if self.is_initialized and self.chat_session:
            with contextlib.suppress(Exception):
                # Clear workflow stack
                if self.fastworkflow:
                    self.fastworkflow.ChatSession.clear_workflow_stack()
                # Cleanup session
                self.chat_session = None
        self.is_initialized = False
        logger.info("✅ FastWorkflow agent reset complete")
    
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
        logger.info("🛑 Stopping FastWorkflow agent")
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
            "awaiting_response": False,
            "last_tool_calls": []
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
        logger.info(f"Setting seed {seed} for FastWorkflow adapter")

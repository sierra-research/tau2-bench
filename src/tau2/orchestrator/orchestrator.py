import hashlib
import json
import random
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from loguru import logger

from tau2.agent.base import BaseAgent, is_valid_agent_history_message
from tau2.agent.llm_agent import LLMSoloAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
    OmissionMetadata,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import EnvFunctionCall, InitializationData, Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.environment.toolkit import ToolType
from tau2.user.base import BaseUser, is_valid_user_history_message
from tau2.user.user_simulator import DummyUser, UserSimulator, UserState
from tau2.utils.llm_utils import get_cost
from tau2.utils.utils import format_time, get_now


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(
    role="assistant", content="Hi! How can I help you today?", cost=0.0
)


class Orchestrator:
    """
    Orchestrator for the simulation given a task.
    Passes messages between the Agent, User, and Environment.
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgent,
        user: BaseUser,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        omission_config: Optional[dict] = None,
    ):
        self.domain = domain
        self.agent = agent
        self.user = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.solo_mode = solo_mode
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[UserState] = None
        self.trajectory: list[Message] = []
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors = 0
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None
        
        # Omission feature setup
        self.omission_config = omission_config
        self.omission_events: list[OmissionMetadata] = []
        self.omission_attempt_counters: dict[tuple[str, str], int] = {}
        
        # Simple RNG setup if omission is enabled
        if omission_config and omission_config.get("enabled"):
            seed = omission_config.get("seed", self.seed) or self.seed
            self.omission_rng = random.Random(seed)

    def initialize(self):
        """
        Initialize the orchestrator.
        - If the tasks specifies an initial state, use it to initialize the environment.
        - Initialize the agent and user states.
        - Send the first message (default message from the agent to the user).
        """
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        for msg in message_history:
            msg.turn_idx = None

        # Add timestamps to the message history
        message_history = self._add_timestamps(message_history)

        if self.solo_mode:
            assert self.environment.solo_mode, "Environment should be in solo mode"
            assert isinstance(self.agent, LLMSoloAgent), (
                "Agent must be a LLMSoloAgent in solo mode"
            )
            assert isinstance(self.user, DummyUser), (
                "User must be a DummyUser in solo mode"
            )

        # Initialize Environment state
        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Set seeds for the agent, user
        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        # Initialize the agent and user states
        if len(message_history) > 0:
            self.validate_message_history(message_history)

            last_message = message_history[-1]
            # Last message is an assistant message
            if isinstance(last_message, AssistantMessage):
                self.from_role = Role.AGENT
                if not last_message.is_tool_call():  # Last message is for the user
                    self.to_role = Role.USER
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.message = last_message
                if self.agent.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            # Last message is a user message
            elif isinstance(last_message, UserMessage):
                self.from_role = Role.USER
                if not last_message.is_tool_call():  # Last message is for the agent
                    self.to_role = Role.AGENT
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.message = last_message
                self.done = UserSimulator.is_stop(last_message)
                if self.done:
                    self.termination_reason = TerminationReason.USER_STOP
            # Last message is a tool message
            elif isinstance(last_message, ToolMessage):
                self.from_role = Role.ENV
                if last_message.requestor == "assistant":
                    self.to_role = Role.AGENT
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_user_history_message(msg)
                        ]
                    )
                else:
                    self.to_role = Role.USER
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_user_history_message(msg)
                        ]
                    )
                self.message = last_message
            else:
                raise ValueError(
                    f"Last message should be of type AssistantMessage, UserMessage, or ToolMessage, got {type(last_message)}"
                )
            self.trajectory = message_history

        else:
            self.agent_state = self.agent.get_init_state()
            self.user_state = self.user.get_init_state()
            if not self.solo_mode:
                first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
                first_message.timestamp = get_now()
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.USER
            else:
                first_message, agent_state = self.agent.generate_next_message(
                    None, self.agent_state
                )
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.ENV
                self.done = self.agent.is_stop(first_message)
                if self.done:
                    self.termination_reason = TerminationReason.AGENT_STOP

        self.environment.sync_tools()

    def run(self) -> SimulationRun:
        """
        Run the simulation.

        Returns:
            SimulationRun: The simulation run.
        """
        start_time = get_now()
        start = time.perf_counter()
        self.initialize()
        while not self.done:
            self.step()
            if self.step_count >= self.max_steps:
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS
            if self.num_errors >= self.max_errors:
                self.done = True
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
        duration = time.perf_counter() - start
        messages = self.get_trajectory()
        res = get_cost(messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res
        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.seed,
        )
        return simulation_run

    def step(self):
        """
        Perform one step of the simulation.
        Sends self.message from self.from_role to self.to_role
        This can either be a message from agent to user/environment, environment to agent, or user to agent
        Updates self.trajectory
        """
        if self.done:
            raise ValueError("Simulation is done")
        logger.debug(
            f"Step {self.step_count}. Sending message from {self.from_role} to {self.to_role}"
        )
        logger.debug(
            f"Step {self.step_count}.\nFrom role: {self.from_role}\nTo role: {self.to_role}\nMessage: {self.message}"
        )
        # AGENT/ENV -> USER
        if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
            user_msg, self.user_state = self.user.generate_next_message(
                self.message, self.user_state
            )
            
            # Modify message if omission injection is enabled
            user_msg = self._apply_omission_injection(user_msg)
            
            user_msg.validate()
            if UserSimulator.is_stop(user_msg):
                self.done = True
                self.termination_reason = TerminationReason.USER_STOP
            self.trajectory.append(user_msg)
            self.message = user_msg
            self.from_role = Role.USER
            if user_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.AGENT
        # USER/ENV -> AGENT
        elif (
            self.from_role == Role.USER or self.from_role == Role.ENV
        ) and self.to_role == Role.AGENT:
            agent_msg, self.agent_state = self.agent.generate_next_message(
                self.message, self.agent_state
            )
            agent_msg.validate()
            if self.agent.is_stop(agent_msg):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            self.trajectory.append(agent_msg)
            self.message = agent_msg
            self.from_role = Role.AGENT
            if agent_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.USER
        # AGENT/USER -> ENV
        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            if not self.message.is_tool_call():
                raise ValueError("Agent or User should send tool call to environment")
            tool_msgs = []
            for tool_call in self.message.tool_calls:
                tool_msg = self.environment.get_response(tool_call)
                tool_msgs.append(tool_msg)
            assert len(self.message.tool_calls) == len(tool_msgs), (
                "Number of tool calls and tool messages should be the same"
            )
            self.trajectory.extend(tool_msgs)
            if (
                len(tool_msgs) > 1
            ):  # Packaging multiple tool messages into a MultiToolMessage
                self.message = MultiToolMessage(
                    role="tool",
                    tool_messages=tool_msgs,
                )
            else:
                self.message = tool_msgs[0]
            self.to_role = self.from_role
            self.from_role = Role.ENV
        else:
            raise ValueError(
                f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}"
            )
        self.step_count += 1
        self.environment.sync_tools()

    def get_trajectory(self) -> list[Message]:
        """
        Get the trajectory of the simulation.
        The trajectory is sorted by timestamp, turn_idx are added to messages, trajectory is returned.
        """
        messages: list[Message] = sorted(
            deepcopy(self.trajectory),
            key=lambda x: x.timestamp,
        )
        trajectory = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            trajectory.append(msg)
        return trajectory

    @classmethod
    def validate_message_history(cls, message_history: list[Message]):
        """
        Validate a message history.
            - Should only contain AssistantMessage, UserMessage, ToolMessage
            - All assistant/user messages should be either to user or tool call, not both.
            - If n tool calls are made by a participant, exactly n tool messages should follow with requestor matching the participant.
        """
        num_expected_tool_messages = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                msg.validate()
                if msg.is_tool_call():
                    if num_expected_tool_messages > 0:
                        raise ValueError(
                            f"{num_expected_tool_messages} tool messages are missing. Got {msg.role} message."
                        )
                    num_expected_tool_messages = len(msg.tool_calls)
                    requestor = msg.role
                else:
                    num_expected_tool_messages == 0
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if num_expected_tool_messages == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(
                        f"Got tool message from {msg.requestor}, expected {requestor}."
                    )
                num_expected_tool_messages -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

    def _initialize_environment(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ):
        """
        Initialize the environment.
        """
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

    def _get_environment_info(self) -> EnvironmentInfo:
        """
        Get the environment info.
        """
        return self.environment.get_info()

    def _count_errors(self, message_history: list[Message]) -> int:
        """
        Count the number of errors in the message history.
        """
        return sum(
            1 for msg in message_history if isinstance(msg, ToolMessage) and msg.error
        )

    def _add_timestamps(
        self, message_history: list[Message]
    ) -> list[tuple[str, Message]]:
        """
        Add timestamps to the message history.
        This is used to sort the messages by timestamp.
        """
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history


    def _get_args_hash(self, tool_args: dict) -> str:
        """Get normalized hash of tool arguments for operation key generation.
        
        Operation key is (tool_name, args_hash) so retries of the same instruction
        converge; distinct operations are independent.
        """
        try:
            # Normalize arguments for consistent hashing
            normalized_args = json.dumps(tool_args, sort_keys=True, default=str)
            return hashlib.md5(normalized_args.encode()).hexdigest()[:8]
        except Exception:
            # Fallback if JSON serialization fails
            return hashlib.md5(str(tool_args).encode()).hexdigest()[:8]

    def _should_omit_tool_call(self, tool_name: str, tool_args: dict) -> bool:
        """Check if a tool call should be omitted."""
        if not hasattr(self, 'omission_rng') or not self.omission_rng:
            return False
            
        # Check if it's a WRITE tool we can omit
        try:
            if (not hasattr(self.environment, 'user_tools') or
                not self.environment.user_tools.has_tool(tool_name) or
                self.environment.user_tools.tool_type(tool_name) != ToolType.WRITE):
                return False
        except Exception:
            return False
            
        # Apply filters
        tool_filter = self.omission_config.get("tool_name_filter")
        if tool_filter and tool_name not in tool_filter:
            return False
            
        # Simple probability calculation
        args_hash = self._get_args_hash(tool_args)
        operation_key = (tool_name, args_hash)
        attempt_index = self.omission_attempt_counters.get(operation_key, 0)
        
        max_failures = self.omission_config.get("max_failures", 1)
        if attempt_index >= max_failures:
            return False
            
        p0 = self.omission_config.get("p0", 0.05)
        probability = p0 / (2 ** attempt_index)
        should_omit = self.omission_rng.random() < probability
        
        # Update counter and record event if omitting
        self.omission_attempt_counters[operation_key] = attempt_index + 1
        if should_omit:
            p_effective = p0 / (2 ** attempt_index)
            self.omission_events.append(OmissionMetadata(
                domain=self.domain,
                tool_name=tool_name,
                args_hash=args_hash,
                attempt_index=attempt_index,
                p0=p0,
                p_effective=p_effective,
                max_failures=max_failures,
                seed=self.omission_config.get("seed", self.seed) or 0,
            ))
            
        return should_omit


    def _get_user_claim_text(self, tool_name: str, tool_args: dict) -> str:
        """Get user claim text for an omitted tool call.
        
        Delegates to the domain-specific user_tools implementation when available
        (e.g., TelecomUserTools._get_user_claim_text) to ensure claims align with
        current state and use appropriate domain phrasing.
        """
        # Try user_tools first
        try:
            if (hasattr(self.environment, 'user_tools') and 
                hasattr(self.environment.user_tools, '_get_user_claim_text')):
                return self.environment.user_tools._get_user_claim_text(tool_name, **tool_args)
        except Exception:
            pass
                
        # Simple fallback
        return f"I used {tool_name.replace('_', ' ')}."

    def _apply_omission_injection(self, user_msg: Message) -> Message:
        """Apply omission injection logic to user messages. Modifies the user_msg if omission conditions are met or simply returns it."""
        # We skip omission for multi-tool messages to avoid ambiguity about which operation
        # the user "claims" to have executed and to prevent partial state inconsistency.
        if (not self.omission_config or 
            not self.omission_config.get("enabled") or
            user_msg.role != 'user' or 
            not user_msg.is_tool_call() or 
            len(user_msg.tool_calls) != 1):
            return user_msg
            
        # Quick domain check
        domains = self.omission_config.get("domains", ["telecom", "telecom-workflow"])
        if self.domain not in domains:
            return user_msg
            
        tool_call = user_msg.tool_calls[0]
        tool_name = tool_call.name
        tool_args = tool_call.arguments or {}
        
        # Check if this should be omitted
        if self._should_omit_tool_call(tool_name, tool_args):
            # Mutate the returned user_msg so user_state.messages stays consistent
            claim_text = self._get_user_claim_text(tool_name, tool_args)
            user_msg.content = claim_text
            user_msg.tool_calls = None
            # Attach omission metadata for viewers/debugging
            p0 = self.omission_config.get("p0", 0.05)
            args_hash = self._get_args_hash(tool_args)
            operation_key = (tool_name, args_hash)
            attempt_index = max(self.omission_attempt_counters.get(operation_key, 1) - 1, 0)
            max_failures = self.omission_config.get("max_failures", 1)
            p_effective = p0 / (2 ** attempt_index)
            user_msg.omission_metadata = OmissionMetadata(
                domain=self.domain,
                tool_name=tool_name,
                args_hash=args_hash,
                attempt_index=attempt_index,
                p0=p0,
                p_effective=p_effective,
                max_failures=max_failures,
                seed=self.omission_config.get("seed", self.seed),
            )
            return user_msg
            
        return user_msg

    def get_omission_events(self) -> list[OmissionMetadata]:
        """Get the collected omission events."""
        return self.omission_events.copy()

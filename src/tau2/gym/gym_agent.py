import json
import threading
from copy import deepcopy
from typing import Any, List, Optional

import gymnasium as gym
from gymnasium.envs.registration import register
from loguru import logger
from pydantic import BaseModel

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent
from tau2.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_USER,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    UserMessage,
)
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.data_model.tasks import Task
from tau2.environment.tool import Tool, as_tool
from tau2.multilingual.runtime import load_run_language
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.runner.build import build_text_orchestrator
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator_base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    HalfDuplexUser,
    ValidUserInputMessage,
)
from tau2.utils.tools import parse_action_string, to_functional_format

TAU_BENCH_ENV_NAME = "tau-bench"
TAU_BENCH_ENV_VERSION = "v0"
TAU_BENCH_ENV_ID = f"{TAU_BENCH_ENV_NAME}-{TAU_BENCH_ENV_VERSION}"

TAU_BENCH_USER_ENV_NAME = "tau-bench-user"
TAU_BENCH_USER_ENV_VERSION = "v0"
TAU_BENCH_USER_ENV_ID = f"{TAU_BENCH_USER_ENV_NAME}-{TAU_BENCH_USER_ENV_VERSION}"

# Registry names of the externally controlled participants, so gym simulations
# are built by the same text-run builder (and language resolution) as `tau2 run`.
GYM_AGENT = "gym_agent"
GYM_SOLO_AGENT = "gym_agent_solo"
GYM_USER = "gym_user"


class TauSpace(gym.spaces.Space):
    """
    A space for the tau-bench gym environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, *args, **kwargs) -> str:
        """
        Sample a string from the space.
        """
        raise NotImplementedError(
            "Sampling not supported for tau-bench gym environment"
        )

    def contains(self, x: Any) -> bool:
        """
        Check if a string is in the space.
        """
        return isinstance(x, str)


def register_gym_agent() -> None:
    """
    Register the tau-bench gym environments with gymnasium.

    Registers:
    - AgentGymEnv: Play as the agent against a user simulator
    - UserGymEnv: Play as the user against an automated agent
    """
    register(
        id=TAU_BENCH_ENV_ID,
        entry_point="tau2.gym.gym_agent:AgentGymEnv",
    )
    register(
        id=TAU_BENCH_USER_ENV_ID,
        entry_point="tau2.gym.gym_agent:UserGymEnv",
    )


class GymAgentState(BaseModel):
    """The state of the gym agent containing the conversation history."""

    messages: list[APICompatibleMessage]


def done() -> str:
    """Call this function when you are done with the task."""
    return GymAgent.STOP_TOKEN


class GymAgent(HalfDuplexAgent):
    """
    A gym-style agent that provides a step-based interface for task execution.

    This agent implements a gym-like interface where external code can control
    the agent's actions step-by-step. It uses threading events to synchronize
    between the external set_action() calls and internal message generation.

    The agent maintains an observation-action cycle:
    1. External code calls set_action(action_msg) to provide the next action
    2. The agent processes the action and generates a response
    3. The agent waits for the next set_action() call
    """

    TRANSFER_TOOL_NAME = "transfer_to_human_agents"
    STOP_FUNCTION_NAME = "done"
    STOP_TOKEN = "###STOP###"

    def __init__(self, tools: List[Tool], domain_policy: str):
        """
        Initialize the gym agent with tools and domain policy.

        Args:
            tools: List of tools available to the agent
            domain_policy: Policy string defining the agent's behavior in the domain
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self._observation: Optional[list[Message]] = None
        self._next_action: Optional[AssistantMessage] = None
        self._agent_turn_finished = threading.Event()
        self._lock = threading.Lock()
        self._agent_turn_finished.set()
        self.add_stop_tool()
        self.validate_tools()

    def add_stop_tool(self) -> None:
        """Add the stop tool to the tools."""
        self.tools.append(as_tool(done))

    def validate_tools(self) -> None:
        """Check if the tools are valid."""
        tool_names = {tool.name for tool in self.tools}
        if self.TRANSFER_TOOL_NAME not in tool_names:
            logger.warning(
                f"Tool {self.TRANSFER_TOOL_NAME} not found in tools. This tool is required for the agent to transfer the user to a human agent."
            )
        if self.STOP_FUNCTION_NAME not in tool_names:
            raise ValueError(f"Tool {self.STOP_FUNCTION_NAME} not found in tools.")

    @property
    def observation(self) -> list[Message]:
        """
        Get the current observation.

        Returns:
            List of messages representing the current conversation state,
            or empty list if no observation has been set yet.
        """
        return self._observation if self._observation else []

    def stop(
        self,
        message: Optional[AssistantMessage] = None,
        state: Optional[GymAgentState] = None,
    ) -> None:
        """
        Stops the agent and finalizes the observation.

        This method is called when the simulation ends. It updates the final
        observation with the complete message history and signals that the
        agent's turn is finished.

        Args:
            message: The last message to the agent (optional)
            state: The current agent state containing message history (optional)
        """
        super().stop(message, state)
        history = deepcopy(state.messages) if state else []
        with self._lock:
            self._observation = history + [message] if message else []
            self._agent_turn_finished.set()

    def set_action(self, action_msg: AssistantMessage) -> None:
        """
        Provide the next action to the agent.

        This method is called by external code to provide the next action
        that the agent should take. It sets the action and signals that
        the agent can continue processing.

        The method uses threading events to synchronize with generate_next_message():
        - Sets the next action and signals that an action is available
        - Raises an error if called when it's not the agent's turn to act

        Args:
            action_msg: The AssistantMessage containing the action to be executed

        Raises:
            RuntimeError: If called when it's not the agent's turn to act
        """
        with self._lock:
            if self._agent_turn_finished.is_set():
                raise RuntimeError("It is not the agent's turn to act.")
            logger.info(f"Stepping with action: {str(action_msg)}")
            self._next_action = action_msg
            self._agent_turn_finished.set()

    def _check_if_stop_toolcall(self, message: AssistantMessage) -> AssistantMessage:
        """Check if the message is a stop message.
        If the message contains a tool call with the name STOP_FUNCTION_NAME, then the message is a stop message.
        Replace the tool call with a content message containing the STOP_TOKEN.
        """
        if message.tool_calls is None:
            return message
        is_stop = False
        for tool_call in message.tool_calls:
            if tool_call.name == self.STOP_FUNCTION_NAME:
                is_stop = True
                break
        if is_stop:
            message.content = self.STOP_TOKEN
            message.tool_calls = None
        return message

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        """Check if the message is a stop message."""
        if message.tool_calls is not None:
            for tool_call in message.tool_calls:
                if tool_call.name == cls.STOP_FUNCTION_NAME:
                    return True
        if message.content is not None:
            return cls.STOP_TOKEN in message.content
        return False

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GymAgentState
    ) -> tuple[AssistantMessage, GymAgentState]:
        """
        Generate the next message in the conversation.

        This method is called by the orchestrator to process incoming messages
        and generate responses. It implements a two-phase synchronization:

        1. **Observation Phase**: Updates the agent's observation with the current
           message history and signals that an observation is ready
        2. **Action Phase**: Waits for an external action to be provided via set_action(),
           then generates and returns the response message

        The method handles both regular messages and MultiToolMessages by
        appropriately updating the state's message history.

        Args:
            message: The incoming message to process (can be a regular message
                    or MultiToolMessage containing tool call results)
            state: The current agent state containing message history

        Returns:
            A tuple containing:
            - The generated AssistantMessage response
            - The updated GymAgentState with the new message history

        Note:
            This method blocks during the action phase until set_action() is called
            to provide the next action. The synchronization ensures that the
            agent's responses are controlled externally through the gym interface.
        """
        with self._lock:
            self._agent_turn_finished.clear()
            logger.info(f"Got message: {message}")
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif (
                message is not None
            ):  # TODO: Review. Added this to handle solo mode. But there might be a better way to do this.
                state.messages.append(message)
            # If message is None, we don't add it to the messages list
            self._observation = deepcopy(state.messages)
            logger.info(f"Setting observation: {self._observation}")

        # Wait for set_action() to provide the next action
        logger.info("Waiting for action")
        self._agent_turn_finished.wait()

        logger.info(f"Continuing with action: {str(self._next_action)}")

        with self._lock:
            response_message = self._next_action
            # Reset for next iteration
            self._next_action = None

        response_message = self._check_if_stop_toolcall(response_message)
        state.messages.append(response_message)
        return response_message, state

    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> GymAgentState:
        """
        Create and return the initial state for the agent.

        Args:
            message_history: Optional list of existing messages to initialize
                           the state with. If None, starts with an empty list.

        Returns:
            A new GymAgentState instance with the provided or empty message history
        """
        messages = message_history.copy() if message_history else []
        return GymAgentState(messages=messages)

    @property
    def is_agent_turn(self) -> bool:
        """
        Check if the agent is currently waiting for input via set_action().

        This property indicates whether the agent has set an observation
        and is waiting for an external action to be provided through
        the set_action() method.

        Returns:
            True if the agent is waiting for input, False otherwise
        """
        return not self._agent_turn_finished.is_set()


def create_gym_agent(tools, domain_policy, **kwargs):
    """Factory function for GymAgent.

    Args:
        tools: Environment tools the agent can call.
        domain_policy: Policy text the agent must follow.
        **kwargs: Additional arguments (unused by GymAgent).
    """
    return GymAgent(tools=tools, domain_policy=domain_policy)


class GymUserState(BaseModel):
    """The state of the gym user containing the conversation history."""

    messages: list[APICompatibleMessage]


class GymUser(HalfDuplexUser):
    """
    A gym-style user that provides a step-based interface for user actions.

    This user implements a gym-like interface where external code can control
    the user's actions step-by-step. It uses threading events to synchronize
    between the external set_action() calls and internal message generation.

    The user maintains an observation-action cycle:
    1. External code calls set_action(action_msg) to provide the next user action
    2. The user processes the action and the agent generates a response
    3. The user waits for the next set_action() call

    This is the inverse of GymAgent - here the user is controlled externally
    while the agent is automated (typically an LLMAgent).
    """

    def __init__(
        self,
        tools: Optional[List[Tool]] = None,
        instructions: Optional[str] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ):
        """
        Initialize the gym user with optional tools and instructions.

        Args:
            tools: List of tools available to the user (optional)
            instructions: Instructions for the user scenario (optional)
            llm: Ignored; the gym caller supplies every user action.
            llm_args: Ignored; accepted so the shared text builder can construct it.
        """
        super().__init__(instructions=instructions, tools=tools)
        self.tools = tools
        self._observation: Optional[list[Message]] = None
        self._next_action: Optional[UserMessage] = None
        self._user_turn_finished = threading.Event()
        self._lock = threading.Lock()
        self._user_turn_finished.set()

    @property
    def observation(self) -> list[Message]:
        """
        Get the current observation.

        Returns:
            List of messages representing the current conversation state,
            or empty list if no observation has been set yet.
        """
        return self._observation if self._observation else []

    def stop(
        self,
        message: Optional[UserMessage] = None,
        state: Optional[GymUserState] = None,
    ) -> None:
        """
        Stops the user and finalizes the observation.

        This method is called when the simulation ends. It updates the final
        observation with the complete message history and signals that the
        user's turn is finished.

        Args:
            message: The last message from the agent (optional)
            state: The current user state containing message history (optional)
        """
        super().stop(message, state)
        history = deepcopy(state.messages) if state else []
        with self._lock:
            self._observation = history + [message] if message else []
            self._user_turn_finished.set()

    def set_action(self, action_msg: UserMessage) -> None:
        """
        Provide the next action to the user.

        This method is called by external code to provide the next action
        that the user should take. It sets the action and signals that
        the user can continue processing.

        The method uses threading events to synchronize with generate_next_message():
        - Sets the next action and signals that an action is available
        - Raises an error if called when it's not the user's turn to act

        Args:
            action_msg: The UserMessage containing the action to be executed

        Raises:
            RuntimeError: If called when it's not the user's turn to act
        """
        with self._lock:
            if self._user_turn_finished.is_set():
                raise RuntimeError("It is not the user's turn to act.")
            logger.info(f"Stepping with user action: {str(action_msg)}")
            self._next_action = action_msg
            self._user_turn_finished.set()

    def generate_next_message(
        self, message: ValidUserInputMessage, state: GymUserState
    ) -> tuple[UserMessage, GymUserState]:
        """
        Generate the next message in the conversation.

        This method is called by the orchestrator to process incoming messages
        and generate responses. It implements a two-phase synchronization:

        1. **Observation Phase**: Updates the user's observation with the current
           message history and signals that an observation is ready
        2. **Action Phase**: Waits for an external action to be provided via set_action(),
           then generates and returns the response message

        The method handles both regular messages and MultiToolMessages by
        appropriately updating the state's message history.

        Args:
            message: The incoming message to process (can be a regular message
                    or MultiToolMessage containing tool call results)
            state: The current user state containing message history

        Returns:
            A tuple containing:
            - The generated UserMessage response
            - The updated GymUserState with the new message history

        Note:
            This method blocks during the action phase until set_action() is called
            to provide the next action. The synchronization ensures that the
            user's responses are controlled externally through the gym interface.
        """
        with self._lock:
            self._user_turn_finished.clear()
            logger.info(f"Got message: {message}")
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            elif message is not None:
                state.messages.append(message)
            # If message is None, we don't add it to the messages list
            self._observation = deepcopy(state.messages)
            logger.info(f"Setting user observation: {self._observation}")

        # Wait for set_action() to provide the next action
        logger.info("Waiting for user action")
        self._user_turn_finished.wait()

        logger.info(f"Continuing with user action: {str(self._next_action)}")

        with self._lock:
            response_message = self._next_action
            # Reset for next iteration
            self._next_action = None

        state.messages.append(response_message)
        return response_message, state

    def get_init_state(
        self,
        message_history: Optional[list[Message]] = None,
    ) -> GymUserState:
        """
        Create and return the initial state for the user.

        Args:
            message_history: Optional list of existing messages to initialize
                           the state with. If None, starts with an empty list.

        Returns:
            A new GymUserState instance with the provided or empty message history
        """
        messages = message_history.copy() if message_history else []
        return GymUserState(messages=messages)

    @property
    def is_user_turn(self) -> bool:
        """
        Check if the user is currently waiting for input via set_action().

        This property indicates whether the user has set an observation
        and is waiting for an external action to be provided through
        the set_action() method.

        Returns:
            True if the user is waiting for input, False otherwise
        """
        return not self._user_turn_finished.is_set()

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """Check if the message is a stop message.

        A user message is a stop message if it contains any of:
        - STOP token (###STOP###): User is satisfied/done
        - TRANSFER token (###TRANSFER###): User wants to transfer to human agent
        - OUT_OF_SCOPE token (###OUT-OF-SCOPE###): Request is out of scope
        """
        if message.is_tool_call():
            return False
        if message.content is None:
            return False
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )


def _register_gym_participants() -> None:
    """Register the gym participants so the shared text builder can construct them."""
    if registry.get_agent_factory(GYM_AGENT) is None:
        registry.register_agent_factory(create_gym_agent, GYM_AGENT)
    if registry.get_agent_factory(GYM_SOLO_AGENT) is None:
        registry.register_agent_factory(
            create_gym_agent, GYM_SOLO_AGENT, metadata={"solo_mode": True}
        )
    if GYM_USER not in registry.get_users():
        registry.register_user(GymUser, GYM_USER)


_register_gym_participants()


def _find_task(config: TextRunConfig, task_id: str) -> Task:
    """One task from the run's language binding (English: the registry tasks)."""
    for task in load_run_language(config).tasks:
        if task.id == task_id:
            return task
    raise ValueError(
        f"No task found with id {task_id} for domain {config.domain} "
        f"in language {config.language}"
    )


class AgentGymEnv(gym.Env):
    """
    A Gymnasium environment that wraps the Tau2 simulation system.

    This environment provides a standard gym interface for interacting with
    Tau2 simulations. It manages the lifecycle of orchestrators, agents,
    and user simulators in a thread-safe manner.

    The environment coordinates between:
    - The external gym interface (reset/step calls)
    - The internal Tau2 orchestrator running in a separate thread
    - The GymAgent that provides step-by-step control

    Key Features:
    - Thread-safe operation with proper synchronization
    - Automatic orchestrator lifecycle management
    - Standard gym observation/action spaces
    - Graceful handling of simulation termination

    Action Input Format:
    The step() method accepts action strings in multiple formats:
    1. JSON-formatted tool calls: Valid ToolCall JSON objects
       Example: '{"name": "search", "arguments": {"query": "flights"}}'

    2. Functional tool calls: Function-style syntax with keyword arguments
       Example: "search_flights(origin='NYC', destination='LAX')"
       Example: "book_ticket(flight_id=123, passenger_name='John Doe')"

    3. Plain text content: Regular text messages for communication
       Example: "Hello, how can I help you?"
       Example: "I need to book a flight from New York to Los Angeles"

    The environment automatically detects the format and converts it to the appropriate
    message type (AssistantMessage with tool calls or content). Plain text messages
    are sent to the user simulator, while tool calls are executed against the environment
    to perform actions like searching databases, making bookings, or retrieving information.
    """

    def __init__(
        self,
        domain: str,
        task_id: str,
        max_steps: int = 100,
        solo_mode: bool = False,
        user_llm: Optional[str] = None,
        user_llm_args: Optional[dict] = None,
        all_messages_as_observation: bool = False,
        language: str = DEFAULT_LANGUAGE,
        communicate_scoring: Optional[CommunicateScoring] = None,
    ):
        """
        Initialize the Tau2 gym environment.

        Args:
            domain: The domain name (e.g., 'retail', 'telecom', 'airline')
            task_id: The specific task ID to run within the domain
            language: Published language pack (canonical English by default);
                resolved exactly as for `tau2 run --language`.
            communicate_scoring: Communicate-info scorer; the run default
                (pinned semantic judge) when omitted.
        """
        self.domain = domain
        self.task_id = task_id
        self.max_steps = max_steps
        self.solo_mode = solo_mode
        self.user_llm = user_llm if user_llm else DEFAULT_LLM_USER
        self.user_llm_args = (
            user_llm_args if user_llm_args else deepcopy(DEFAULT_LLM_ARGS_USER)
        )
        self.all_messages_as_observation = all_messages_as_observation
        self.language = language
        self.communicate_scoring = communicate_scoring
        self._task: Optional[Task] = None

        self._lock = threading.Lock()
        self._agent: Optional[GymAgent] = None
        self._user: Optional[HalfDuplexUser] = None
        self._orchestrator: Optional[Orchestrator] = None
        self._orchestrator_thread: Optional[threading.Thread] = None
        self._simulation_done = threading.Event()
        self._simulation_run: Optional[SimulationRun] = None
        self.observation_space = TauSpace()
        self.action_space = TauSpace()

    def _get_tools(self) -> List[Tool]:
        """
        Get the tools for the environment.
        """
        if self._agent is None:
            raise ValueError("Agent not initialized. Call reset() first.")
        return self._agent.tools

    def _get_policy(self) -> str:
        """
        Get the policy for the environment.
        """
        if self._orchestrator is None:
            raise ValueError("Orchestrator not initialized. Call reset() first.")
        return self._orchestrator.environment.get_policy()

    def _log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message with the task ID.
        """
        logger.log(level, f"[{self.task_id}] {message}")

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[str, dict]:
        """
        Reset the environment and start a new simulation.

        This method creates a fresh simulation by:
        1. Creating a new orchestrator with the specified domain and task
        2. Starting the orchestrator in a separate thread
        3. Waiting for the agent to be ready for input
        4. Returning the initial observation

        The method ensures proper cleanup of any existing simulation
        and thread-safe initialization of the new one.

        Args:
            seed: Optional random seed for reproducibility (passed to gym.Env.reset)
            options: Optional configuration options (passed to gym.Env.reset)

        Returns:
            A tuple containing:
            - observation: String representation of the initial message history
            - info: Dictionary with additional information (currently empty)

        Note:
            This method blocks until the orchestrator has started and the agent
            is waiting for the first action. If the simulation ends immediately
            (e.g., due to an error), an empty observation is returned.
        """
        super().reset(seed=seed)

        with self._lock:
            # Reset state
            self._simulation_run = None
            self._simulation_done.clear()

            # Wait for any existing thread to finish
            if self._orchestrator_thread and self._orchestrator_thread.is_alive():
                self._orchestrator_thread.join(timeout=1.0)
                if self._orchestrator_thread.is_alive():
                    self._log(
                        "Previous orchestrator thread did not terminate within timeout. "
                        "Continuing anyway (daemon thread will be cleaned up).",
                        "WARNING",
                    )

            # Create new orchestrator
            self._orchestrator = self._get_orchestrator(seed)
            self._agent = self._orchestrator.agent
            self._user = self._orchestrator.user

            # Start orchestrator in a separate thread
            self._orchestrator_thread = threading.Thread(target=self._run_orchestrator)
            self._orchestrator_thread.daemon = True
            self._orchestrator_thread.start()

            # Wait for orchestrator to send the initial observation
            # Use a timeout to periodically check if simulation is done
            while not self._simulation_done.is_set() and not self._agent.is_agent_turn:
                self._simulation_done.wait(timeout=0.01)

            if self._simulation_done.is_set():
                # Simulation ended immediately, return empty observation
                self._log("Simulation ended immediately", "WARNING")
                return "", self._get_info()

            # Get the initial observation from the agent
            initial_observation = self._agent.observation.copy()

            # Convert observation to string format
            observation_str = self._format_observation(initial_observation)

            return observation_str, self._get_info()

    def _get_info(self) -> dict:
        """
        Get the current info dictionary for the gym environment.

        Returns:
            A dictionary containing the current simulation run information
            in JSON format, or an empty dictionary if no simulation has run yet.
        """
        return {
            "task": self._get_task(),
            "simulation_run": self._get_simulation_run(),
            "tools": self._get_tools(),
            "policy": self._get_policy(),
        }

    def _get_simulation_run(self) -> str:
        """
        Get the current simulation run as a JSON string.

        Returns:
            A JSON string representation of the current simulation run,
            or an empty dictionary if no simulation has run yet.
        """
        if self._simulation_run is None:
            return json.dumps({}, indent=2)
        return self._simulation_run.model_dump_json(indent=2)

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """
        Execute an action and advance the simulation.

        This method provides the standard gym step interface. It:
        1. Passes the action to the GymAgent via its set_action() method
        2. Waits for the agent to process the action and receive a response
        3. Checks if the simulation has terminated
        4. Returns the updated observation and termination status

        The method handles the coordination between the external gym interface
        and the internal Tau2 simulation running in a separate thread.

        Args:
            action: The action string to be executed by the agent. Supports multiple formats:
                - JSON-formatted tool calls: '{"name": "search", "arguments": {"query": "flights"}}'
                - Functional tool calls: "search_flights(origin='NYC', destination='LAX')"
                - Plain text content: "Hello, how can I help you?"
                See the class docstring for detailed format examples.
                Note: Plain text messages are sent to the user simulator, while tool calls
                are executed against the environment to perform actions.

        Returns:
            A tuple containing:
            - observation: String representation of the current message
            - reward: Based on the evaluation result of the simulation run
            - terminated: True if the simulation has ended, False otherwise
            - truncated: Always False (not used in current implementation)
            - info: Dictionary with additional information (currently empty)

        Raises:
            RuntimeError: If reset() has not been called before step()

        Note:
            This method blocks until the agent has processed the action and
            is ready for the next step. The simulation may terminate during
            this process, in which case terminated will be True.
        """
        if self._orchestrator is None:
            raise RuntimeError("Orchestrator not initialized. Call reset() first.")

        with self._lock:
            if self._simulation_done.is_set():
                self._log("Simulation already terminated.", "WARNING")
                return "", 0.0, True, False, self._get_info()
                # raise ValueError("Simulation already terminated.")

            # Parse the action string into a message
            try:
                action_msg = parse_action_string(action)
            except Exception as e:
                self._log(f"Error parsing action: {e}", "ERROR")
                return (
                    f"Invalid action with error: {e}",
                    0.0,
                    False,
                    False,
                    self._get_info(),
                )

            # Provide the action to the agent
            self._agent.set_action(action_msg)

            # Wait for the orchestrator to send the next observation
            # Use a timeout to periodically check if simulation is done
            while not self._simulation_done.is_set() and not self._agent.is_agent_turn:
                self._simulation_done.wait(timeout=0.01)

            # Check if simulation is done
            terminated = self._simulation_done.is_set()
            self._log(f"Simulation done: {terminated}", "INFO")
            # Convert observation to string format
            observation_str = self._format_observation(self._agent.observation)

            reward, reward_info = self._get_reward()
            info = self._get_info()
            info["reward_info"] = reward_info
            return (
                observation_str,
                reward,
                terminated,
                False,
                info,
            )

    def _get_reward(self) -> tuple[float, str]:
        """
        The reward of the finished simulation run.

        The run is evaluated by the shared text-run path (``run_simulation``)
        with the call context's communicate scorer and language environment.

        Returns:
            A tuple containing:
            - reward: The computed reward value, or 0.0 before a run finishes.
            - reward_info: A JSON string containing the reward information.
        """
        if self._simulation_run is None or self._simulation_run.reward_info is None:
            return 0.0, json.dumps({}, indent=2)
        reward_info = self._simulation_run.reward_info
        self._log(f"Evaluation result: {reward_info}", "INFO")
        return reward_info.reward, reward_info.model_dump_json(indent=2)

    def _run_orchestrator(self):
        """
        Run the orchestrator in a separate thread.

        This private method is the target for the orchestrator thread.
        It runs the orchestrator's main simulation loop and handles
        any exceptions that occur during execution.

        The method sets the _simulation_done flag when the orchestrator
        finishes (either normally or due to an error), which signals
        to the main thread that the simulation has ended.
        It also sets the simulation run to the orchestrator's simulation run.

        Thread Safety:
            This method is designed to be run in a separate thread and
            uses the _simulation_done event to communicate with the main thread.
            Any exceptions are logged but do not propagate to avoid thread crashes.

        Error Handling:
            If the orchestrator raises an exception, it is logged as an error
            but the thread continues to set the simulation as done to prevent
            the main thread from hanging indefinitely.
        """
        simulation_run = None
        try:
            if self._orchestrator:
                self._log("Starting orchestrator", "INFO")
                simulation_run = run_simulation(self._orchestrator)
                self._log("Orchestrator finished", "INFO")
        except Exception as e:
            self._log(f"Orchestrator error: {e}", "ERROR")
        finally:
            self._simulation_run = simulation_run
            self._simulation_done.set()

    def _format_observation(self, messages: list[Message]) -> str:
        """
        Convert a list of messages to a string observation.

        This method formats the message history into a readable string
        format for the gym observation space. Each message is formatted
        as "role: content" and messages are separated by newlines.

        The method handles different message types:
        - UserMessage: Formatted as "user: content" or "user: tool_calls"
        - AssistantMessage: Formatted as "assistant: content" or "assistant: tool_calls"
        - Other messages: Formatted as "role: content"

        Tool calls are converted to functional format for readability.

        Args:
            messages: List of Message objects representing the conversation history

        Returns:
            A string representation of the message history, or empty string
            if no messages are provided. Each message is on a separate line
            in the format "role: content".
        """
        if not messages:
            return ""
        turns = []
        for m in messages:
            if isinstance(m, UserMessage):
                if not m.is_tool_call():
                    turns.append(f"user: {m.content}")
                else:
                    tool_calls = ", ".join(
                        [to_functional_format(t) for t in m.tool_calls]
                    )
                    turns.append(f"user: {tool_calls}")
            elif isinstance(m, AssistantMessage):
                if not m.is_tool_call():
                    turns.append(f"assistant: {m.content}")
                else:
                    tool_calls = ", ".join(
                        [to_functional_format(t) for t in m.tool_calls]
                    )
                    turns.append(f"assistant: {tool_calls}")
                if not self.all_messages_as_observation:
                    # reset the turns contents, only keep the response to the assistant messages.
                    turns = []
            else:
                turns.append(f"{m.role}: {m.content}")
        return "\n".join(turns)

    def _run_config(self, seed: Optional[int] = None) -> TextRunConfig:
        """The text run this environment executes, as `tau2 run` would."""
        return TextRunConfig(
            domain=self.domain,
            language=self.language,
            agent=GYM_SOLO_AGENT if self.solo_mode else GYM_AGENT,
            user="dummy_user" if self.solo_mode else "user_simulator",
            llm_user=self.user_llm,
            llm_args_user=self.user_llm_args,
            max_steps=self.max_steps,
            **({} if seed is None else {"seed": seed}),
            **(
                {}
                if self.communicate_scoring is None
                else {"communicate_scoring": self.communicate_scoring}
            ),
        )

    def _get_task(self) -> Task:
        """
        The configured task from the run's resolved language binding.

        Raises:
            ValueError: If no task with the configured ID exists for the domain
                in this language.
        """
        if self._task is None:
            self._task = _find_task(self._run_config(), self.task_id)
        return self._task

    def _get_orchestrator(self, seed: Optional[int] = None) -> Orchestrator:
        """
        Build the orchestrator with the shared text-run builder.

        Language resolution, persona, agent language guidance, environment
        bindings and the communicate scorer are identical to a `tau2 run`
        text call; only the agent (or, in solo mode, the dummy user) is the
        externally controlled gym participant.
        """
        return build_text_orchestrator(self._run_config(seed), self._get_task())


class UserGymEnv(gym.Env):
    """
    A Gymnasium environment for playing as the USER against an automated agent.

    This environment is the inverse of AgentGymEnv - here you control the user's
    actions while an LLMAgent (automated) responds to your messages. This is useful for:
    - Testing agent behavior from a user's perspective
    - Human evaluation of agent performance
    - Debugging conversational flows
    - Training user simulators via RL

    The environment coordinates between:
    - The external gym interface (reset/step calls for user actions)
    - The internal Tau2 orchestrator running in a separate thread
    - The GymUser that provides step-by-step control of user actions
    - The LLMAgent that responds automatically

    Action Input Format:
    The step() method accepts user action strings in multiple formats:
    1. JSON-formatted tool calls: Valid ToolCall JSON objects (if user tools available)
       Example: '{"name": "check_balance", "arguments": {}}'

    2. Functional tool calls: Function-style syntax with keyword arguments
       Example: "check_balance()"
       Example: "update_address(street='123 Main St', city='New York')"

    3. Plain text content: Regular text messages for communication
       Example: "Hello, I need help booking a flight"
       Example: "I want to fly from New York to Los Angeles next week"

    Note: Solo mode is not compatible with UserGymEnv since solo mode has no user interaction.
    """

    def __init__(
        self,
        domain: str,
        task_id: str,
        max_steps: int = 100,
        agent_llm: Optional[str] = None,
        agent_llm_args: Optional[dict] = None,
        all_messages_as_observation: bool = False,
        language: str = DEFAULT_LANGUAGE,
        communicate_scoring: Optional[CommunicateScoring] = None,
    ):
        """
        Initialize the Tau2 user gym environment.

        Args:
            domain: The domain name (e.g., 'retail', 'telecom', 'airline')
            task_id: The specific task ID to run within the domain
            max_steps: Maximum number of steps before truncation
            agent_llm: LLM to use for the automated agent (default: from config)
            agent_llm_args: Arguments to pass to the agent LLM
            all_messages_as_observation: If True, show all messages; if False, only show agent responses
            language: Published language pack (canonical English by default);
                resolved exactly as for `tau2 run --language`.
            communicate_scoring: Communicate-info scorer; the run default
                (pinned semantic judge) when omitted.
        """
        self.domain = domain
        self.task_id = task_id
        self.max_steps = max_steps
        self.agent_llm = agent_llm if agent_llm else DEFAULT_LLM_AGENT
        self.agent_llm_args = (
            agent_llm_args if agent_llm_args else deepcopy(DEFAULT_LLM_ARGS_AGENT)
        )
        self.all_messages_as_observation = all_messages_as_observation
        self.language = language
        self.communicate_scoring = communicate_scoring
        self._task: Optional[Task] = None

        self._lock = threading.Lock()
        self._user: Optional[GymUser] = None
        self._agent: Optional[LLMAgent] = None
        self._orchestrator: Optional[Orchestrator] = None
        self._orchestrator_thread: Optional[threading.Thread] = None
        self._simulation_done = threading.Event()
        self._simulation_run: Optional[SimulationRun] = None
        self.observation_space = TauSpace()
        self.action_space = TauSpace()

    def _get_tools(self) -> List[Tool]:
        """
        Get the tools for the environment (agent tools).
        """
        if self._agent is None:
            raise ValueError("Agent not initialized. Call reset() first.")
        return self._agent.tools

    def _get_user_tools(self) -> List[Tool]:
        """
        Get the user tools for the environment.
        """
        if self._orchestrator is None:
            raise ValueError("Orchestrator not initialized. Call reset() first.")
        try:
            return self._orchestrator.environment.get_user_tools() or []
        except ValueError:
            return []

    def _get_policy(self) -> str:
        """
        Get the policy for the environment.
        """
        if self._orchestrator is None:
            raise ValueError("Orchestrator not initialized. Call reset() first.")
        return self._orchestrator.environment.get_policy()

    def _log(self, message: str, level: str = "INFO") -> None:
        """
        Log a message with the task ID.
        """
        logger.log(level, f"[UserGym:{self.task_id}] {message}")

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[str, dict]:
        """
        Reset the environment and start a new simulation.

        This method creates a fresh simulation by:
        1. Creating a new orchestrator with automated agent and controlled user
        2. Starting the orchestrator in a separate thread
        3. Waiting for the user to be ready for input (agent sends first message)
        4. Returning the initial observation (agent's greeting)

        Args:
            seed: Optional random seed for reproducibility
            options: Optional configuration options

        Returns:
            A tuple containing:
            - observation: String representation of the agent's initial message
            - info: Dictionary with additional information (task, tools, policy, etc.)
        """
        super().reset(seed=seed)

        with self._lock:
            # Reset state
            self._simulation_run = None
            self._simulation_done.clear()

            # Wait for any existing thread to finish
            if self._orchestrator_thread and self._orchestrator_thread.is_alive():
                self._orchestrator_thread.join(timeout=1.0)
                if self._orchestrator_thread.is_alive():
                    self._log(
                        "Previous orchestrator thread did not terminate within timeout. "
                        "Continuing anyway (daemon thread will be cleaned up).",
                        "WARNING",
                    )

            # Create new orchestrator
            self._orchestrator = self._get_orchestrator(seed)
            self._user = self._orchestrator.user
            self._agent = self._orchestrator.agent

            # Start orchestrator in a separate thread
            self._orchestrator_thread = threading.Thread(target=self._run_orchestrator)
            self._orchestrator_thread.daemon = True
            self._orchestrator_thread.start()

            # Wait for orchestrator to send the initial observation (agent's first message)
            # Use a timeout to periodically check if simulation is done
            while not self._simulation_done.is_set() and not self._user.is_user_turn:
                self._simulation_done.wait(timeout=0.01)

            if self._simulation_done.is_set():
                # Simulation ended immediately, return empty observation
                self._log("Simulation ended immediately", "WARNING")
                return "", self._get_info()

            # Get the initial observation from the user (shows agent's greeting)
            initial_observation = self._user.observation.copy()

            # Convert observation to string format
            observation_str = self._format_observation(initial_observation)

            return observation_str, self._get_info()

    def _get_info(self) -> dict:
        """
        Get the current info dictionary for the gym environment.

        Returns:
            A dictionary containing the current simulation run information.
        """
        return {
            "task": self._get_task(),
            "simulation_run": self._get_simulation_run(),
            "agent_tools": self._get_tools(),
            "user_tools": self._get_user_tools(),
            "policy": self._get_policy(),
        }

    def _get_simulation_run(self) -> str:
        """
        Get the current simulation run as a JSON string.

        Returns:
            A JSON string representation of the current simulation run,
            or an empty dictionary if no simulation has run yet.
        """
        if self._simulation_run is None:
            return json.dumps({}, indent=2)
        return self._simulation_run.model_dump_json(indent=2)

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """
        Execute a user action and advance the simulation.

        This method provides the standard gym step interface for user actions. It:
        1. Passes the action to the GymUser via its set_action() method
        2. Waits for the agent to process the action and respond
        3. Checks if the simulation has terminated
        4. Returns the updated observation (agent's response) and termination status

        Args:
            action: The user action string to be executed. Supports multiple formats:
                - JSON-formatted tool calls: '{"name": "check_balance", "arguments": {}}'
                - Functional tool calls: "check_balance()"
                - Plain text content: "Hello, I need help with my account"
                See the class docstring for detailed format examples.

        Returns:
            A tuple containing:
            - observation: String representation of the agent's response
            - reward: Based on the evaluation result of the simulation run
            - terminated: True if the simulation has ended, False otherwise
            - truncated: Always False (not used in current implementation)
            - info: Dictionary with additional information

        Raises:
            RuntimeError: If reset() has not been called before step()
        """
        if self._orchestrator is None:
            raise RuntimeError("Orchestrator not initialized. Call reset() first.")

        with self._lock:
            if self._simulation_done.is_set():
                self._log("Simulation already terminated.", "WARNING")
                return "", 0.0, True, False, self._get_info()

            # Parse the action string into a UserMessage
            try:
                # Parse as a user message (not agent message)
                action_msg = parse_action_string(action, requestor="user")
            except Exception as e:
                self._log(f"Error parsing user action: {e}", "ERROR")
                return (
                    f"Invalid action with error: {e}",
                    0.0,
                    False,
                    False,
                    self._get_info(),
                )

            # Provide the action to the user
            self._user.set_action(action_msg)

            # Wait for the orchestrator to send the next observation (agent's response)
            # Use a timeout to periodically check if simulation is done
            while not self._simulation_done.is_set() and not self._user.is_user_turn:
                self._simulation_done.wait(timeout=0.01)

            # Check if simulation is done
            terminated = self._simulation_done.is_set()
            self._log(f"Simulation done: {terminated}", "INFO")

            # Convert observation to string format
            observation_str = self._format_observation(self._user.observation)
            reward, reward_info = self._get_reward()
            info = self._get_info()
            info["reward_info"] = reward_info
            return (
                observation_str,
                reward,
                terminated,
                False,
                info,
            )

    def _get_reward(self) -> tuple[float, str]:
        """
        The reward of the finished simulation run.

        The run is evaluated by the shared text-run path (``run_simulation``)
        with the call context's communicate scorer and language environment.

        Returns:
            A tuple containing:
            - reward: The computed reward value, or 0.0 before a run finishes.
            - reward_info: A JSON string containing the reward information.
        """
        if self._simulation_run is None or self._simulation_run.reward_info is None:
            return 0.0, json.dumps({}, indent=2)
        reward_info = self._simulation_run.reward_info
        self._log(f"Evaluation result: {reward_info}", "INFO")
        return reward_info.reward, reward_info.model_dump_json(indent=2)

    def _run_orchestrator(self):
        """
        Run the orchestrator in a separate thread.

        This private method is the target for the orchestrator thread.
        It runs the orchestrator's main simulation loop and handles
        any exceptions that occur during execution.
        """
        simulation_run = None
        try:
            if self._orchestrator:
                self._log("Starting orchestrator", "INFO")
                simulation_run = run_simulation(self._orchestrator)
                self._log("Orchestrator finished", "INFO")
        except Exception as e:
            self._log(f"Orchestrator error: {e}", "ERROR")
        finally:
            self._simulation_run = simulation_run
            self._simulation_done.set()

    def _format_observation(self, messages: list[Message]) -> str:
        """
        Convert a list of messages to a string observation from the user's perspective.

        This method formats the message history into a readable string showing what
        the user sees (primarily agent messages and tool results).

        Args:
            messages: List of Message objects representing the conversation history

        Returns:
            A string representation of the message history from the user's perspective.
        """
        if not messages:
            return ""
        turns = []
        for m in messages:
            if isinstance(m, UserMessage):
                if not m.is_tool_call():
                    turns.append(f"user: {m.content}")
                else:
                    tool_calls = ", ".join(
                        [to_functional_format(t) for t in m.tool_calls]
                    )
                    turns.append(f"user: {tool_calls}")
                if not self.all_messages_as_observation:
                    # When playing as user, clear after your (user) messages
                    # to only show the agent's response
                    turns = []
            elif isinstance(m, AssistantMessage):
                if not m.is_tool_call():
                    turns.append(f"assistant: {m.content}")
                else:
                    tool_calls = ", ".join(
                        [to_functional_format(t) for t in m.tool_calls]
                    )
                    turns.append(f"assistant: {tool_calls}")
                # Don't clear after assistant messages - we want to see the agent's response!
            else:
                turns.append(f"{m.role}: {m.content}")
        return "\n".join(turns)

    def _run_config(self, seed: Optional[int] = None) -> TextRunConfig:
        """The text run this environment executes, as `tau2 run` would."""
        return TextRunConfig(
            domain=self.domain,
            language=self.language,
            agent="llm_agent",
            llm_agent=self.agent_llm,
            llm_args_agent=self.agent_llm_args,
            user=GYM_USER,
            max_steps=self.max_steps,
            **({} if seed is None else {"seed": seed}),
            **(
                {}
                if self.communicate_scoring is None
                else {"communicate_scoring": self.communicate_scoring}
            ),
        )

    def _get_task(self) -> Task:
        """
        The configured task from the run's resolved language binding.

        Raises:
            ValueError: If no task with the configured ID exists for the domain
                in this language.
        """
        if self._task is None:
            self._task = _find_task(self._run_config(), self.task_id)
        return self._task

    def _get_orchestrator(self, seed: Optional[int] = None) -> Orchestrator:
        """
        Build the orchestrator with the shared text-run builder.

        The automated LLMAgent receives the same language guidance as a
        `tau2 run` text call; the user is the externally controlled GymUser.
        """
        return build_text_orchestrator(self._run_config(seed), self._get_task())

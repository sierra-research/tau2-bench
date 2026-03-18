"""
Full-duplex orchestrator for streaming/voice communication.

This extends the BaseOrchestrator to support simultaneous bidirectional
communication between agent and user.
"""

import time
from copy import deepcopy
from typing import Any, Optional, TypeVar, Union

from loguru import logger

from tau2.agent.base.streaming import compute_responsiveness_info
from tau2.agent.base_agent import FullDuplexAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE, BaseOrchestrator
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import FullDuplexUser
from tau2.utils.llm_utils import get_cost
from tau2.utils.utils import get_now
from tau2.voice.utils.transcript_utils import compute_proportional_user_transcripts

# Type variables for generic full-duplex orchestrator
StreamingAgentT = TypeVar("StreamingAgentT", bound=FullDuplexAgent)
StreamingUserT = TypeVar("StreamingUserT", bound=FullDuplexUser)


class FullDuplexOrchestrator(BaseOrchestrator[StreamingAgentT, StreamingUserT, Tick]):
    """
    Orchestrator for full-duplex (streaming) communication.

    Extends the BaseOrchestrator to support:
    - Simultaneous agent and user communication
    - Chunk-based message passing
    - Tick-based trajectory that groups concurrent events

    Key differences from half-duplex (Orchestrator):
    - Uses get_next_chunk() instead of generate_next_message()
    - Both agent and user can "speak" at the same time
    - Messages are chunked rather than complete
    - Trajectory is organized by ticks, not individual messages

    Trajectory Structure:
    - get_trajectory() returns a list of Tick objects
    - Each Tick contains all events from a single simulation step
    - Tool calls are queued and executed within the tick
    - Participants receive incoming chunks per tick
    - Use get_messages() to get a flat message list
    """

    def __init__(
        self,
        domain: str,
        agent: StreamingAgentT,
        user: StreamingUserT,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        simulation_id: Optional[str] = None,
        tick_duration_seconds: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize FullDuplexOrchestrator.

        Args:
            domain: The domain name of the simulation.
            agent: The streaming agent instance (must have get_next_chunk method).
            user: The streaming user instance (must have get_next_chunk method).
            environment: The environment instance.
            task: The task specification.
            max_steps: Maximum number of simulation steps.
            max_errors: Maximum number of tool execution errors.
            seed: Optional random seed for reproducibility.
            simulation_id: Optional simulation ID.
            tick_duration_seconds: Duration of each simulation tick in seconds (for timing metadata).
            timeout: Maximum wallclock time in seconds. None means no timeout.
        """
        super().__init__(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            simulation_id=simulation_id,
            timeout=timeout,
        )

        # Set mode to FULL_DUPLEX
        self.mode = CommunicationMode.FULL_DUPLEX

        # Full-duplex specific attributes
        self.current_user_chunk: Optional[UserMessage] = None
        self.current_agent_chunk: Optional[AssistantMessage] = None
        self.tick_duration_seconds = tick_duration_seconds

        # Pending tool results: executed this tick, delivered to participant next tick.
        # This decouples tool execution from the audio/tick loop so that:
        # - No silence is injected during tool result delivery
        # - Agent audio from the tool-call tick is preserved
        # - User audio continuity is maintained
        self.pending_agent_tool_results: Optional[Message] = None
        self.pending_user_tool_results: Optional[Message] = None

        # Tick-based trajectory structure
        self.ticks: list[Tick] = []

        # Validate mode compatibility
        self._validate_mode_compatibility()

    def _validate_mode_compatibility(self):
        """Validate that agent and user support full-duplex streaming."""
        # Check if agent has get_next_chunk method
        if not hasattr(self.agent, "get_next_chunk"):
            raise ValueError(
                f"FULL_DUPLEX mode requires agent to have 'get_next_chunk' method. "
                f"Agent {self.agent.__class__.__name__} does not support streaming."
            )

        # Check if user has get_next_chunk method
        if not hasattr(self.user, "get_next_chunk"):
            raise ValueError(
                f"FULL_DUPLEX mode requires user to have 'get_next_chunk' method. "
                f"User {self.user.__class__.__name__} does not support streaming."
            )

        logger.info(
            f"FullDuplexOrchestrator initialized with "
            f"agent={self.agent.__class__.__name__}, "
            f"user={self.user.__class__.__name__}"
        )

    def initialize(self):
        """Initialize the orchestrator for full-duplex communication."""
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

        # Full-duplex doesn't support message history initialization yet
        if len(message_history) > 0:
            raise ValueError(
                "Full duplex mode does not yet support message history initialization"
            )

        # Initialize Environment state
        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Set seeds
        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        # Full-duplex specific initialization
        self._initialize_full_duplex()

        self.environment.sync_tools()

    def _initialize_full_duplex(self):
        """Initialize full-duplex specific state."""
        # Initialize user and agent states
        self.user_state = self.user.get_init_state()

        # Create first agent message
        # If agent has create_initial_message (audio-native agents), use it to get
        # a message with proper audio content. Otherwise, use the text-only default.
        if hasattr(self.agent, "create_initial_message"):
            first_agent_message = self.agent.create_initial_message()
        else:
            first_agent_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
            first_agent_message.chunk_id = 0
            first_agent_message.is_final_chunk = True
            first_agent_message.timestamp = get_now()

        self.agent_state = self.agent.get_init_state(
            message_history=[first_agent_message]
        )

        # Initialize current message trackers with dummy message
        dummy_agent_message = AssistantMessage(
            role="assistant",
            content=None,
            timestamp=get_now(),
            chunk_id=0,
            is_final_chunk=True,
            contains_speech=False,
        )

        self.current_agent_chunk = first_agent_message

        # Get first user chunk
        self.user_state = self.user.get_init_state()
        self.current_user_chunk, self.user_state = self.user.get_next_chunk(
            self.user_state, participant_chunk=dummy_agent_message
        )

        # Initialize trajectory with first tick
        init_start_time = time.perf_counter()
        first_tick = Tick(
            tick_id=0,
            timestamp=get_now(),
            agent_chunk=deepcopy(self.current_agent_chunk),
            user_chunk=deepcopy(self.current_user_chunk),
            tick_duration_seconds=self.tick_duration_seconds,
        )
        first_tick.wall_clock_duration_seconds = time.perf_counter() - init_start_time
        self.ticks = [first_tick]

    def _format_chunk_summary(self, chunk: Optional[Message], role: str) -> str:
        """Format a chunk for logging with key information."""
        if chunk is None:
            return f"{role}: (none)"

        parts = [f"{role}:"]

        # Content summary
        if chunk.content:
            content_preview = (
                chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
            )
            parts.append(f'  content="{content_preview}"')
        else:
            parts.append("  content=(empty)")

        # Speech indicator
        if hasattr(chunk, "contains_speech"):
            parts.append(f"  speech={chunk.contains_speech}")

        # Chunk metadata
        if hasattr(chunk, "chunk_id") and chunk.chunk_id is not None:
            parts.append(f"  chunk_id={chunk.chunk_id}")
        if hasattr(chunk, "is_final_chunk") and chunk.is_final_chunk is not None:
            parts.append(f"  is_final={chunk.is_final_chunk}")

        # Tool calls
        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
            tool_names = [tc.name for tc in chunk.tool_calls]
            parts.append(f"  tool_calls={tool_names}")

        return " | ".join(parts)

    def step(self):
        """
        Perform one tick of full-duplex streaming communication.

        Both agent and user generate chunks simultaneously.
        Tool calls are executed within the tick but results are delivered
        on the next tick, keeping audio flow uninterrupted.
        All events are grouped into a single Tick object.
        """
        if self.done:
            raise ValueError("Simulation is done")

        tick_id = len(self.ticks)
        logger.debug(f"Tick {tick_id}. step_count={self.step_count}")

        # Sanity check: stored chunks should never carry tool_calls
        # (tool_calls are stripped and tracked separately)
        if self.current_agent_chunk and self.current_agent_chunk.is_tool_call():
            raise ValueError(
                f"Agent chunk should not have tool_calls at step start: "
                f"{self.current_agent_chunk}"
            )
        if self.current_user_chunk and self.current_user_chunk.is_tool_call():
            raise ValueError(
                f"User chunk should not have tool_calls at step start: "
                f"{self.current_user_chunk}"
            )

        # Create new tick
        tick_start_time = time.perf_counter()
        tick = Tick(
            tick_id=tick_id,
            timestamp=get_now(),
            tick_duration_seconds=self.tick_duration_seconds,
        )

        # Both participants receive the other's previous chunk
        incoming_for_user = self.current_agent_chunk
        incoming_for_agent = self.current_user_chunk

        # Log incoming chunks from previous tick
        logger.debug(f"[Tick {tick_id}] Incoming chunks from previous tick:")
        logger.debug(
            f"  → User receives: {self._format_chunk_summary(incoming_for_user, 'agent_chunk')}"
        )
        logger.debug(
            f"  → Agent receives: {self._format_chunk_summary(incoming_for_agent, 'user_chunk')}"
        )

        # --- 1. Process user turn ---
        (
            user_chunk,
            self.user_state,
            user_tool_calls,
            user_tool_results,
        ) = self._process_participant_turn(
            participant=self.user,
            state=self.user_state,
            incoming_chunk=incoming_for_user,
            is_agent=False,
            pending_tool_results=self.pending_user_tool_results,
        )
        tick.user_chunk = deepcopy(user_chunk)
        tick.user_tool_calls = [deepcopy(tc) for tc in user_tool_calls]
        tick.user_tool_results = [deepcopy(r) for r in user_tool_results]
        self.pending_user_tool_results = (
            self._wrap_tool_results(user_tool_results) if user_tool_results else None
        )

        # --- 2. Process agent turn ---
        (
            agent_chunk,
            self.agent_state,
            agent_tool_calls,
            agent_tool_results,
        ) = self._process_participant_turn(
            participant=self.agent,
            state=self.agent_state,
            incoming_chunk=incoming_for_agent,
            is_agent=True,
            pending_tool_results=self.pending_agent_tool_results,
        )
        tick.agent_chunk = deepcopy(agent_chunk)
        tick.agent_tool_calls = [deepcopy(tc) for tc in agent_tool_calls]
        tick.agent_tool_results = [deepcopy(r) for r in agent_tool_results]
        self.pending_agent_tool_results = (
            self._wrap_tool_results(agent_tool_results) if agent_tool_results else None
        )

        # --- 3. Update state and bookkeeping ---
        self.current_user_chunk = user_chunk
        self.current_agent_chunk = agent_chunk

        # Record wall clock duration for this tick
        tick.wall_clock_duration_seconds = time.perf_counter() - tick_start_time

        logger.debug(
            f"[Tick {tick_id}] Wall-clock duration: "
            f"{tick.wall_clock_duration_seconds:.3f}s"
        )
        if (
            self.tick_duration_seconds is not None
            and tick.wall_clock_duration_seconds < self.tick_duration_seconds
        ):
            logger.warning(
                f"[Tick {tick_id}] Completed in "
                f"{tick.wall_clock_duration_seconds:.3f}s, which is less than "
                f"the expected tick duration of {self.tick_duration_seconds:.3f}s. "
                f"Wall-clock pacing may not be enforced by the adapter."
            )

        self.ticks.append(tick)

        self.step_count += 1
        self.environment.sync_tools()

    def _process_participant_turn(
        self,
        participant: Union[StreamingAgentT, StreamingUserT],
        state: Any,
        incoming_chunk: Optional[Message],
        is_agent: bool,
        pending_tool_results: Optional[Message] = None,
    ) -> tuple[Message, Any, list[ToolCall], list[ToolMessage]]:
        """
        Process a participant's turn with dual-channel input.

        The participant receives both channels in a single get_next_chunk call:
        - participant_chunk: speech/audio from the other participant
        - tool_results: results from previously executed tool calls (if any)

        If the participant's response contains tool calls, they are executed
        immediately. The caller is responsible for wrapping them via
        _wrap_tool_results and storing as pending for the next tick.

        Args:
            participant: The agent or user instance.
            state: The current state of the participant.
            incoming_chunk: The incoming message chunk from the other participant.
            is_agent: True if processing agent, False if processing user.
            pending_tool_results: Tool results from previous tick to deliver.

        Returns:
            Tuple of (chunk, new_state, tool_calls, tool_results).
        """
        participant_name = "AGENT" if is_agent else "USER"

        is_stop = self.agent.is_stop if is_agent else UserSimulator.is_stop
        termination_reason = (
            TerminationReason.AGENT_STOP if is_agent else TerminationReason.USER_STOP
        )

        new_state = state

        if pending_tool_results is not None:
            logger.debug(
                f"  [{participant_name}] Delivering tool results alongside "
                f"participant chunk"
            )

        # Single get_next_chunk call with both channels
        logger.debug(f"  [{participant_name}] Calling get_next_chunk...")
        new_chunk, new_state = participant.get_next_chunk(
            state=new_state,
            participant_chunk=incoming_chunk,
            tool_results=pending_tool_results,
        )
        logger.debug(
            f"  [{participant_name}] Returned chunk: "
            f"{self._format_chunk_summary(new_chunk, 'chunk')}"
        )

        # Validate and check stop condition
        if new_chunk.contains_speech:
            new_chunk.validate()
        if is_stop(new_chunk):
            logger.info(f"  [{participant_name}] *** STOP signal detected ***")
            self.done = True
            self.termination_reason = termination_reason

        # Handle tool calls: execute now, deliver results next tick
        tool_calls: list[ToolCall] = []
        tool_results: list[ToolMessage] = []

        if new_chunk and new_chunk.is_tool_call():
            tool_calls = list(new_chunk.tool_calls)
            tool_names = [tc.name for tc in tool_calls]
            logger.info(f"  [{participant_name}] Tool calls: {tool_names}")

            results = self._execute_tool_calls(tool_calls)
            tool_results = list(results)

            for tc, result in zip(tool_calls, results):
                result_preview = (
                    result.content[:100] + "..."
                    if len(result.content) > 100
                    else result.content
                )
                logger.debug(f"  [{participant_name}]   {tc.name}() → {result_preview}")

            # Return a copy with tool_calls stripped so the chunk only carries
            # speech/audio for the other participant. The original chunk (stored
            # in the participant's tick history via record_tick) must keep its
            # tool_calls intact for linearization / tool-result pairing.
            new_chunk = deepcopy(new_chunk)
            new_chunk.tool_calls = None

            logger.info(
                f"  [{participant_name}] Executed {len(tool_calls)} tool(s), "
                f"results pending for next tick"
            )

        return new_chunk, new_state, tool_calls, tool_results

    def get_trajectory(self) -> list[Tick]:
        """
        Get the tick-based trajectory of the simulation.

        Returns:
            List of Tick objects, each containing all events from a simulation tick.
        """
        return deepcopy(self.ticks)

    def get_ticks(self) -> list[Tick]:
        """
        Get the tick-based trajectory.

        Alias for get_trajectory() for backward compatibility.

        Returns:
            List of Tick objects, each containing all events from a simulation tick.
        """
        return self.get_trajectory()

    def get_messages(self) -> list[Message]:
        """
        Get all messages from the simulation as a flat list.

        Converts the tick-based trajectory to a flat message list,
        sorted by timestamp with turn_idx assigned.

        Returns:
            List of all messages sorted by timestamp with turn_idx assigned.
        """
        messages: list[Message] = []
        for tick in self.ticks:
            messages.extend(tick.get_all_messages())
        messages = sorted(messages, key=lambda m: m.timestamp)
        result = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            result.append(msg)
        return result

    def _check_termination(self) -> None:
        """
        Check for full-duplex specific termination conditions.

        Checks max_steps, max_errors, and timeout after each tick.
        """
        if self.step_count >= self.max_steps:
            self.done = True
            self.termination_reason = TerminationReason.MAX_STEPS
        if self.num_errors >= self.max_errors:
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS
        self._check_timeout()

    def _finalize(self) -> SimulationRun:
        """
        Finalize the full-duplex simulation and create the SimulationRun result.

        Sends stop signals to agent and user, calculates costs, and builds the result.

        Returns:
            SimulationRun with all simulation data.
        """
        # Send stop signals to agent and user, forwarding any pending tool results
        try:
            self.agent.stop(
                None, self.agent_state, tool_results=self.pending_agent_tool_results
            )
        except Exception as e:
            logger.warning(f"Error stopping agent during finalization: {e}")
        try:
            self.user.stop(
                None, self.user_state, tool_results=self.pending_user_tool_results
            )
        except Exception as e:
            logger.warning(f"Error stopping user during finalization: {e}")

        # Calculate duration
        duration = time.perf_counter() - self._run_start_perf

        # Derive messages for cost computation (not stored in SimulationRun)
        ticks = self.get_trajectory()
        res = get_cost(self.get_messages())
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res

        # Get speech_environment from user's voice_settings if available
        speech_environment = None
        if (
            hasattr(self.user, "voice_settings")
            and self.user.voice_settings is not None
        ):
            speech_environment = self.user.voice_settings.speech_environment

        # Compute responsiveness metrics
        info = compute_responsiveness_info(ticks)

        # Extract provider session ID if available (e.g., OpenAI session ID)
        provider_session_id = None
        if hasattr(self.agent, "adapter") and hasattr(self.agent.adapter, "provider"):
            provider = self.agent.adapter.provider
            provider_session_id = getattr(provider, "session_id", None)

        # Collect effect timeline from user simulator if available
        effect_timeline = None
        if hasattr(self.user, "get_effect_timeline"):
            effect_timeline = self.user.get_effect_timeline()
            if effect_timeline and effect_timeline.events:
                logger.info(
                    f"Effect timeline: {len(effect_timeline.events)} events recorded"
                )

        simulation_run = SimulationRun(
            id=self.simulation_id,
            task_id=self.task.id,
            start_time=self._run_start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            ticks=ticks,
            seed=self.seed,
            mode=self.mode.value,
            speech_environment=speech_environment,
            info=info,
            provider_session_id=provider_session_id,
            effect_timeline=effect_timeline,
        )
        return simulation_run

    def run(self) -> SimulationRun:
        """
        Run the simulation with post-processing for user transcripts.

        Overrides the base class run() to track timing and add transcript processing.

        Returns:
            SimulationRun: The simulation run.
        """
        result = super().run()
        compute_proportional_user_transcripts(self.ticks)
        return result

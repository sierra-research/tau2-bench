"""
Core error injection engine.

Intercepts tool responses during simulation and optionally injects
controlled errors based on the injection configuration.
"""

import json
import re
import random
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.data_model.tasks import Task
from tau_robustness.injection_config import (
    InjectionConfig,
    InjectionDef,
    InjectionType,
    apply_mutation,
    check_precondition,
)

# Pattern to extract entity-like IDs from text (order IDs, reservation codes, etc.)
_ENTITY_PATTERN = re.compile(
    r"#W\d+"  # retail order IDs like #W6247578
    r"|[A-Z0-9]{6}"  # airline reservation codes like EHGLP3
    r"|[A-Z]\d{4}"  # telecom line/customer IDs like L1001, C1001
    r"|555-\d{3}-\d{4}"  # phone numbers
)


class InjectionEvent(BaseModel):
    """Record of a single error injection that occurred during simulation."""

    turn_idx: int
    tool_call_id: str
    tool_name: str
    injection_id: str
    injection_type: InjectionType
    original_content: str
    modified_content: str
    detection_signals: list[str]
    recovery_signals: list[str]
    blocks_actions: list[str] = Field(default_factory=list)
    detected: bool = False
    recovered: bool = False
    is_replay: bool = False


class ErrorInjector:
    """Intercepts tool responses and optionally injects errors.

    Maintains injection state for tracking detection/recovery across
    the full simulation trajectory.

    Targeting strategy:
        - Skip batch tool calls (agent fetching many items at once) since
          injecting on a random item in a bulk lookup rarely affects the task.
        - Track user messages to identify which entities (order IDs, reservation
          codes, etc.) the user has mentioned, and prefer injecting on tool calls
          that reference those entities.
        - Respect a turn window to avoid injecting during auth or too late.

    Args:
        injection_config: Domain-specific injection definitions.
        injection_rate: Probability of injecting an error on eligible tool calls (0.0-1.0).
        seed: Random seed for reproducible injection patterns.
        max_injections_per_run: Maximum number of errors to inject per simulation.
        min_turn: Earliest turn at which injection can happen (skip initial auth turns).
        max_turn: Latest turn for injection (avoid injecting near the end).
        max_batch_size: Maximum number of parallel tool calls to still allow injection.
            Batch calls larger than this are skipped (they're bulk lookups).
        persistent: If True, injected errors replay on subsequent calls to the same
            tool (simulates persistent DB corruption). If False (default), only the
            first call is corrupted (simulates transient cache miss).
    """

    def __init__(
        self,
        injection_config: InjectionConfig,
        injection_rate: float = 0.3,
        seed: int = 42,
        max_injections_per_run: int = 1,
        min_turn: int = 4,
        max_turn: int = 30,
        max_batch_size: int = 2,
        persistent: bool = False,
    ):
        if not 0.0 <= injection_rate <= 1.0:
            raise ValueError(
                f"injection_rate must be between 0.0 and 1.0, got {injection_rate}"
            )

        self.config = injection_config
        self.injection_rate = injection_rate
        self.rng = random.Random(seed)
        self.max_injections_per_run = max_injections_per_run
        self.min_turn = min_turn
        self.max_turn = max_turn
        self.max_batch_size = max_batch_size
        self.persistent = persistent
        self.injection_log: list[InjectionEvent] = []
        self._injection_count = 0
        self._persistent_injections: dict[str, InjectionDef] = {}
        self._task_description: Optional[str] = None
        self._task_action_tools: set[str] = set()
        self._user_entities: set[str] = set()
        self._user_keywords: list[str] = []

    def set_task_context(self, task_description: str) -> None:
        """Set the current task's user scenario for relevance filtering.

        When set, only injections whose task_keywords match the description
        will be eligible. Call this before running each task.
        """
        self._task_description = task_description.lower() if task_description else None

    def set_task(self, task: Task) -> None:
        """Set the current task for relevance filtering and blocking priority.

        Extracts action tool names from evaluation criteria and also sets
        the task description context for backward compatibility.
        """
        self._task_action_tools = set()
        if task.evaluation_criteria and task.evaluation_criteria.actions:
            self._task_action_tools = {
                action.name for action in task.evaluation_criteria.actions
            }
        if task.user_scenario:
            self.set_task_context(str(task.user_scenario))

    def track_user_message(self, content: str) -> None:
        """Track a user message to extract entity references.

        Extracts entity-like IDs (order IDs, reservation codes, phone numbers)
        from user messages. These are used to prioritize injecting on tool calls
        that reference entities the user has actually mentioned.
        """
        if not content:
            return
        entities = _ENTITY_PATTERN.findall(content)
        self._user_entities.update(e.upper() for e in entities)
        self._user_keywords.extend(content.lower().split())

    def reset(self) -> None:
        """Reset injector state for a new simulation run."""
        self.injection_log.clear()
        self._injection_count = 0
        self._persistent_injections.clear()
        self._task_description = None
        self._task_action_tools = set()
        self._user_entities.clear()
        self._user_keywords.clear()

    @property
    def num_injections(self) -> int:
        return len(self.injection_log)

    def maybe_inject(
        self,
        tool_call: ToolCall,
        tool_response: ToolMessage,
        turn_idx: int,
        batch_size: int = 1,
    ) -> ToolMessage:
        """Possibly inject an error into a tool response.

        Returns the original ToolMessage if no injection occurs,
        or a modified ToolMessage with the injected error.

        Args:
            tool_call: The tool call being responded to.
            tool_response: The real tool response from the environment.
            turn_idx: Current simulation step count.
            batch_size: Number of parallel tool calls in this turn.
                If batch_size > max_batch_size, injection is skipped
                (bulk lookups are not good injection targets).
        """
        # Guard: don't inject on error responses
        if tool_response.error:
            return tool_response

        # Persistent replay: if this tool was previously injected in persistent
        # mode, re-apply the same injection (no count increment, no dice roll).
        if self.persistent and tool_call.name in self._persistent_injections:
            return self._replay_persistent(
                tool_call, tool_response, turn_idx
            )

        # Guard: respect turn window
        if turn_idx < self.min_turn or turn_idx > self.max_turn:
            return tool_response

        # Guard: respect max injections per run
        if self._injection_count >= self.max_injections_per_run:
            return tool_response

        # Guard: skip large batch calls unless this specific call targets
        # an entity the user has mentioned (conversation-aware targeting).
        if batch_size > self.max_batch_size:
            if not self._tool_call_matches_user_entities(tool_call):
                logger.debug(
                    f"Skipping injection: batch_size={batch_size} > max={self.max_batch_size} "
                    f"and no user-entity match (tool={tool_call.name}, turn={turn_idx})"
                )
                return tool_response
            logger.info(
                f"Allowing injection in batch (size={batch_size}): tool call "
                f"arguments match user-mentioned entities (tool={tool_call.name})"
            )

        # Check if this tool is a target for any injection
        applicable = self.config.get_injections_for_tool(tool_call.name)
        if not applicable:
            return tool_response

        # Filter by task relevance if task context is set
        if self._task_description is not None:
            applicable = [inj for inj in applicable if self._is_injection_relevant(inj)]
            if not applicable:
                return tool_response

        # Roll dice for injection
        if self.rng.random() > self.injection_rate:
            return tool_response

        # Partition into blocking (overlaps task actions) and cosmetic (other)
        blocking = []
        cosmetic = []
        for inj in applicable:
            if (
                inj.blocks_actions
                and self._task_action_tools
                and set(inj.blocks_actions) & self._task_action_tools
            ):
                blocking.append(inj)
            else:
                cosmetic.append(inj)

        # If this tool only has cosmetic injections but blocking injections exist
        # for OTHER tools in this config, skip cosmetic to reserve the injection
        # slot for the blocking one (which fires on a later tool call).
        if not blocking and cosmetic and self._has_blocking_injections_for_task():
            logger.debug(
                f"Skipping cosmetic injection on '{tool_call.name}' — "
                f"reserving injection slot for blocking injection on another tool"
            )
            return tool_response

        # Try blocking injections first, then cosmetic
        self.rng.shuffle(blocking)
        self.rng.shuffle(cosmetic)
        for injection_def in blocking + cosmetic:
            injected, modified_response = self._try_inject(
                tool_call, tool_response, turn_idx, injection_def
            )
            if injected:
                return modified_response
        # None of the injections' preconditions matched
        return tool_response

    def _try_inject(
        self,
        tool_call: ToolCall,
        tool_response: ToolMessage,
        turn_idx: int,
        injection_def: InjectionDef,
    ) -> tuple[bool, Optional[ToolMessage]]:
        """Attempt to apply an injection definition to a tool response.

        Returns (True, modified_response) if injection was applied,
        (False, None) if preconditions weren't met.
        """
        original_content = tool_response.content
        if original_content is None:
            return False, None

        try:
            data = json.loads(original_content)
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                f"Skipping injection {injection_def.id}: content is not valid JSON"
            )
            return False, None

        # Check preconditions
        if injection_def.precondition:
            if not check_precondition(data, injection_def.precondition):
                return False, None

        # Apply mutations
        try:
            for mutation in injection_def.mutations:
                apply_mutation(data, mutation)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.debug(f"Skipping injection {injection_def.id}: mutation failed: {e}")
            return False, None

        modified_content = json.dumps(data)

        # Create modified ToolMessage
        modified_response = ToolMessage(
            id=tool_response.id,
            content=modified_content,
            requestor=tool_response.requestor,
            role=tool_response.role,
            error=tool_response.error,
            turn_idx=tool_response.turn_idx,
            timestamp=tool_response.timestamp,
        )

        # Store for persistent replay before logging
        if self.persistent:
            self._persistent_injections[tool_call.name] = injection_def

        # Log the injection
        event = InjectionEvent(
            turn_idx=turn_idx,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            injection_id=injection_def.id,
            injection_type=injection_def.type,
            original_content=original_content,
            modified_content=modified_content,
            detection_signals=injection_def.detection_signals,
            recovery_signals=injection_def.recovery_signals,
            blocks_actions=injection_def.blocks_actions,
        )
        self.injection_log.append(event)
        self._injection_count += 1

        logger.info(
            f"INJECTED ERROR: {injection_def.id} ({injection_def.type.value}) "
            f"on tool '{tool_call.name}' at turn {turn_idx}"
        )

        return True, modified_response

    def _replay_persistent(
        self,
        tool_call: ToolCall,
        tool_response: ToolMessage,
        turn_idx: int,
    ) -> ToolMessage:
        """Re-apply a previously injected error in persistent mode.

        Does NOT increment _injection_count (replays are free).
        Logs the replay as a separate InjectionEvent with is_replay=True.
        """
        injection_def = self._persistent_injections[tool_call.name]
        original_content = tool_response.content
        if original_content is None:
            return tool_response

        try:
            data = json.loads(original_content)
        except (json.JSONDecodeError, TypeError):
            return tool_response

        try:
            for mutation in injection_def.mutations:
                apply_mutation(data, mutation)
        except (KeyError, IndexError, TypeError, ValueError):
            return tool_response

        modified_content = json.dumps(data)
        modified_response = ToolMessage(
            id=tool_response.id,
            content=modified_content,
            requestor=tool_response.requestor,
            role=tool_response.role,
            error=tool_response.error,
            turn_idx=tool_response.turn_idx,
            timestamp=tool_response.timestamp,
        )

        event = InjectionEvent(
            turn_idx=turn_idx,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            injection_id=injection_def.id,
            injection_type=injection_def.type,
            original_content=original_content,
            modified_content=modified_content,
            detection_signals=injection_def.detection_signals,
            recovery_signals=injection_def.recovery_signals,
            blocks_actions=injection_def.blocks_actions,
            is_replay=True,
        )
        self.injection_log.append(event)

        logger.info(
            f"REPLAYED ERROR (persistent): {injection_def.id} "
            f"on tool '{tool_call.name}' at turn {turn_idx}"
        )

        return modified_response

    def _is_injection_relevant(self, injection: InjectionDef) -> bool:
        """Check if an injection is relevant to the current task.

        An injection is relevant if it has no task_keywords (always eligible)
        or if any of its keywords appear in the task description.
        """
        if not injection.task_keywords:
            return True
        return any(
            kw.lower() in self._task_description for kw in injection.task_keywords
        )

    def _has_blocking_injections_for_task(self) -> bool:
        """Check if any injection in the config blocks a task-required action.

        Used to decide whether to skip cosmetic injections on early tools
        and reserve the injection slot for a blocking injection on a later tool.
        """
        if not self._task_action_tools:
            return False
        for inj in self.config.injections:
            if inj.blocks_actions and set(inj.blocks_actions) & self._task_action_tools:
                # Also check task relevance
                if self._task_description is not None:
                    if not self._is_injection_relevant(inj):
                        continue
                return True
        return False

    def _tool_call_matches_user_entities(self, tool_call: ToolCall) -> bool:
        """Check if a tool call's arguments reference a user-mentioned entity.

        Used for conversation-aware targeting: in a large batch of tool calls
        (e.g., fetching all 5 orders at once), only inject on the call whose
        arguments match something the user actually mentioned.

        Returns True if:
        - Any argument value matches a tracked user entity, OR
        - No entities have been tracked yet (fall back to allowing injection
          since we can't determine relevance without context)
        """
        if not self._user_entities:
            # No user context yet — allow injection (fall back to random targeting)
            return True

        for value in tool_call.arguments.values():
            value_str = str(value).upper()
            for entity in self._user_entities:
                if entity in value_str:
                    return True
        return False

    def get_injection_summary(self) -> dict:
        """Get a summary of all injections for this simulation run."""
        return {
            "num_injections": len(self.injection_log),
            "persistent": self.persistent,
            "injections": [
                {
                    "injection_id": event.injection_id,
                    "injection_type": event.injection_type.value,
                    "tool_name": event.tool_name,
                    "turn_idx": event.turn_idx,
                    "detected": event.detected,
                    "recovered": event.recovered,
                    "is_replay": event.is_replay,
                }
                for event in self.injection_log
            ],
        }

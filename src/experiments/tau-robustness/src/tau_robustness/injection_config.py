"""
Data models for error injection definitions and configuration.

Each domain has a YAML file defining which tools can be targeted,
what kinds of errors to inject, and how to modify the JSON response.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class InjectionType(str, Enum):
    """Types of errors that can be injected into tool responses."""

    STALE_DATA = "stale_data"
    CORRUPT_FIELD = "corrupt_field"
    MISSING_DATA = "missing_data"
    CONTRADICTORY = "contradictory"
    PHANTOM = "phantom"
    STATUS_FLIP = "status_flip"


class FieldMutation(BaseModel):
    """Describes how to mutate a specific field in a JSON response."""

    field_path: str = Field(
        description="Dot-separated path to the target field (e.g., 'status', 'items.0.item_id'). "
        "Supports integer indices for lists."
    )
    action: str = Field(
        description="Mutation action: 'set', 'delete', 'append', 'flip'.",
        default="set",
    )
    value: Optional[Any] = Field(
        description="The value to set or append. Required for 'set' and 'append' actions.",
        default=None,
    )
    flip_map: Optional[dict[str, str]] = Field(
        description="For 'flip' action: mapping of original values to injected values.",
        default=None,
    )


class InjectionDef(BaseModel):
    """A single error injection definition."""

    id: str = Field(description="Unique identifier for this injection.")
    type: InjectionType = Field(description="The type of error to inject.")
    target_tool: str = Field(description="Name of the tool whose response to modify.")
    description: str = Field(
        description="Human-readable description of the error scenario."
    )
    difficulty: int = Field(
        description="How hard the error is to detect (1=easy, 3=hard).",
        default=2,
        ge=1,
        le=3,
    )
    mutations: list[FieldMutation] = Field(
        description="List of field mutations to apply to the tool response."
    )
    detection_signals: list[str] = Field(
        description="Expected agent behaviors indicating error detection.",
        default_factory=list,
    )
    recovery_signals: list[str] = Field(
        description="Expected agent behaviors indicating error recovery.",
        default_factory=list,
    )
    precondition: Optional[dict[str, Any]] = Field(
        description="Optional conditions on tool response content that must be true "
        "for this injection to apply (e.g., {'status': 'delivered'}).",
        default=None,
    )
    task_keywords: list[str] = Field(
        description="Keywords that must appear in the task's user scenario for this "
        "injection to be relevant. If empty, the injection is always eligible.",
        default_factory=list,
    )
    blocks_actions: list[str] = Field(
        description="Write-action tool names this injection is designed to block. "
        "When non-empty and overlapping with a task's required actions, "
        "this injection is prioritized over cosmetic ones.",
        default_factory=list,
    )


class InjectionConfig(BaseModel):
    """Configuration for all injections available for a domain."""

    domain: str = Field(
        description="Domain name (e.g., 'retail', 'airline', 'telecom')."
    )
    injections: list[InjectionDef] = Field(
        description="List of injection definitions for this domain."
    )

    def get_injections_for_tool(self, tool_name: str) -> list[InjectionDef]:
        """Get all injection definitions targeting a specific tool."""
        return [inj for inj in self.injections if inj.target_tool == tool_name]

    def get_injections_by_type(
        self, injection_type: InjectionType
    ) -> list[InjectionDef]:
        """Get all injection definitions of a specific type."""
        return [inj for inj in self.injections if inj.type == injection_type]

    def get_injections_by_difficulty(self, difficulty: int) -> list[InjectionDef]:
        """Get all injection definitions at a specific difficulty level."""
        return [inj for inj in self.injections if inj.difficulty == difficulty]

    @classmethod
    def from_yaml(cls, path: Path) -> "InjectionConfig":
        """Load an InjectionConfig from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_domain(
        cls, domain: str, data_dir: Optional[Path] = None
    ) -> "InjectionConfig":
        """Load the injection config for a given domain from the default data directory.

        Default data_dir resolves to <repo_root>/data/tau2/injections/.
        """
        if data_dir is None:
            # Navigate from src/experiments/tau-robustness/src/tau_robustness/
            # up to repo root, then into data/tau2/injections/
            data_dir = (
                Path(__file__).parent.parent.parent.parent.parent.parent
                / "data"
                / "tau2"
                / "injections"
            )
        path = data_dir / f"{domain}_injections.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"No injection config found for domain '{domain}' at {path}. "
                f"Available configs: {list(data_dir.glob('*_injections.yaml'))}"
            )
        return cls.from_yaml(path)


def get_nested(data: Any, field_path: str) -> Any:
    """Navigate a nested dict/list by dot-separated path.

    Examples:
        get_nested({"a": {"b": 1}}, "a.b") -> 1
        get_nested({"items": [{"id": 1}]}, "items.0.id") -> 1
    """
    keys = field_path.split(".")
    current = data
    for key in keys:
        if isinstance(current, list):
            current = current[int(key)]
        elif isinstance(current, dict):
            current = current[key]
        else:
            raise KeyError(f"Cannot navigate into {type(current)} with key '{key}'")
    return current


def set_nested(data: Any, field_path: str, value: Any) -> None:
    """Set a value in a nested dict/list by dot-separated path."""
    keys = field_path.split(".")
    current = data
    for key in keys[:-1]:
        if isinstance(current, list):
            current = current[int(key)]
        elif isinstance(current, dict):
            current = current[key]
        else:
            raise KeyError(f"Cannot navigate into {type(current)} with key '{key}'")
    last_key = keys[-1]
    if isinstance(current, list):
        current[int(last_key)] = value
    elif isinstance(current, dict):
        current[last_key] = value
    else:
        raise KeyError(f"Cannot set on {type(current)} with key '{last_key}'")


def delete_nested(data: Any, field_path: str) -> Any:
    """Delete a field from a nested dict/list by dot-separated path. Returns deleted value."""
    keys = field_path.split(".")
    current = data
    for key in keys[:-1]:
        if isinstance(current, list):
            current = current[int(key)]
        elif isinstance(current, dict):
            current = current[key]
        else:
            raise KeyError(f"Cannot navigate into {type(current)} with key '{key}'")
    last_key = keys[-1]
    if isinstance(current, list):
        return current.pop(int(last_key))
    elif isinstance(current, dict):
        return current.pop(last_key)
    else:
        raise KeyError(f"Cannot delete from {type(current)} with key '{last_key}'")


def append_nested(data: Any, field_path: str, value: Any) -> None:
    """Append a value to a list at a nested path."""
    target = get_nested(data, field_path)
    if not isinstance(target, list):
        raise TypeError(f"Cannot append to {type(target)} at path '{field_path}'")
    target.append(value)


def apply_mutation(data: dict, mutation: FieldMutation) -> dict:
    """Apply a single FieldMutation to a parsed JSON dict.

    Returns the modified dict (mutated in place).
    """
    if mutation.action == "set":
        set_nested(data, mutation.field_path, mutation.value)
    elif mutation.action == "delete":
        delete_nested(data, mutation.field_path)
    elif mutation.action == "append":
        append_nested(data, mutation.field_path, mutation.value)
    elif mutation.action == "flip":
        current = get_nested(data, mutation.field_path)
        current_str = str(current)
        if mutation.flip_map and current_str in mutation.flip_map:
            flipped = mutation.flip_map[current_str]
            # Try to preserve original type
            if isinstance(current, bool):
                flipped = flipped.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                flipped = int(flipped)
            elif isinstance(current, float):
                flipped = float(flipped)
            set_nested(data, mutation.field_path, flipped)
        else:
            raise ValueError(
                f"Flip map does not contain current value '{current_str}'. "
                f"Available: {mutation.flip_map}"
            )
    else:
        raise ValueError(f"Unknown mutation action: {mutation.action}")
    return data


def check_precondition(data: dict, precondition: dict[str, Any]) -> bool:
    """Check if preconditions are met on the parsed JSON response."""
    for field_path, expected_value in precondition.items():
        try:
            actual = get_nested(data, field_path)
            if actual != expected_value:
                return False
        except (KeyError, IndexError, TypeError):
            return False
    return True

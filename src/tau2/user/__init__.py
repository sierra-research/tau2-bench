"""
User module exports.
"""

import warnings

from tau2.user.user_simulator import DummyUser, UserSimulator, VoiceUserSimulator
from tau2.user.user_simulator_base import (
    FullDuplexUser,
    FullDuplexVoiceUser,
    HalfDuplexUser,
    HalfDuplexVoiceUser,
    UserState,
    ValidUserInputMessage,
)
from tau2.user.user_simulator_streaming import VoiceStreamingUserSimulator

# =============================================================================
# DEPRECATION ALIASES
# =============================================================================
# These aliases maintain backward compatibility with code using old names.
# They will emit DeprecationWarning when used.


def __getattr__(name: str):
    """Module-level __getattr__ for deprecation warnings."""
    deprecated_aliases = {
        "BaseUser": ("HalfDuplexUser", HalfDuplexUser),
        "BaseStreamingUser": ("FullDuplexUser", FullDuplexUser),
        "BaseVoiceUser": ("HalfDuplexVoiceUser", HalfDuplexVoiceUser),
    }

    if name in deprecated_aliases:
        new_name, new_class = deprecated_aliases[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_class

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Direct aliases for static analysis tools (these don't trigger warnings on import)
BaseUser = HalfDuplexUser
BaseStreamingUser = FullDuplexUser
BaseVoiceUser = HalfDuplexVoiceUser


__all__ = [
    # Base classes
    "HalfDuplexUser",
    "FullDuplexUser",
    "HalfDuplexVoiceUser",
    "FullDuplexVoiceUser",
    "UserState",
    "ValidUserInputMessage",
    # User simulators
    "UserSimulator",
    "DummyUser",
    # Voice users
    "VoiceUserSimulator",
    "VoiceStreamingUserSimulator",
    # Deprecated aliases (kept for backward compatibility)
    "BaseUser",
    "BaseStreamingUser",
    "BaseVoiceUser",
]

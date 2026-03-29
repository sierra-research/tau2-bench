"""
User module exports.
"""

import warnings

from tau2.user.user_simulator import DummyUser, UserSimulator
from tau2.user.user_simulator_base import (
    FullDuplexUser,
    FullDuplexVoiceUser,
    HalfDuplexUser,
    HalfDuplexVoiceUser,
    UserState,
    ValidUserInputMessage,
)

# =============================================================================
# DEPRECATION ALIASES
# =============================================================================
# These aliases maintain backward compatibility with code using old names.
# They will emit DeprecationWarning when used.

_VOICE_LAZY_IMPORTS = {
    "VoiceUserSimulator": "tau2.user.user_simulator_voice",
    "VoiceStreamingUserSimulator": "tau2.user.user_simulator_streaming",
}


def __getattr__(name: str):
    """Module-level __getattr__ for deprecation warnings and lazy voice imports."""
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

    if name in _VOICE_LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_VOICE_LAZY_IMPORTS[name])
        return getattr(module, name)

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
    # Voice users (lazy imports)
    "VoiceUserSimulator",
    "VoiceStreamingUserSimulator",
    # Deprecated aliases (kept for backward compatibility)
    "BaseUser",
    "BaseStreamingUser",
    "BaseVoiceUser",
]

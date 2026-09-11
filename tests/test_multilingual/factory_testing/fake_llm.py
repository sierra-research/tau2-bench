# Copyright Sierra
"""A scriptable fake for the Factory LLM contract.

The real implementation lands in a parallel PR at
``src/tau2/multilingual/factory/llm.py`` with exactly this interface::

    class FactoryLLM:
        def chat(self, model: str, system: str, user: str,
                 *, call_name: str,
                 reasoning_effort: str | None = None) -> str: ...
        def json_call(self, model: str, system: str, user: str,
                      *, call_name: str,
                      reasoning_effort: str | None = None) -> dict | list: ...

``FakeFactoryLLM`` is coded against that contract only — it imports nothing
from ``tau2.multilingual.factory`` — so factory tests can be written (and this
fake validated) before the real class exists.

Scripts are keyed by ``call_name``. Each scripted response is one of:

- ``str``: returned by ``chat``; rejected as malformed JSON by ``json_call``;
- ``dict`` / ``list``: returned by ``json_call``; a contract violation for
  ``chat`` (raises);
- ``Exception`` instance: raised when popped (simulates provider failure).

Every call is recorded in ``calls``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

Response = Union[str, dict, list, Exception]


class FakeLLMScriptError(AssertionError):
    """A test script problem: no script for a call_name, or one ran dry."""


class FakeLLMMalformedJSONError(ValueError):
    """json_call received a malformed (str) response."""


@dataclass
class FakeCall:
    """One recorded LLM call."""

    call_name: str
    model: str
    system: str
    user: str
    reasoning_effort: Optional[str] = None


@dataclass
class FakeFactoryLLM:
    """Scripted FactoryLLM stand-in. See module docstring for the contract."""

    scripts: dict[str, list[Response]]
    calls: list[FakeCall] = field(default_factory=list)

    def _pop(
        self,
        call_name: str,
        model: str,
        system: str,
        user: str,
        reasoning_effort: Optional[str] = None,
    ) -> Response:
        self.calls.append(
            FakeCall(
                call_name=call_name,
                model=model,
                system=system,
                user=user,
                reasoning_effort=reasoning_effort,
            )
        )
        if call_name not in self.scripts:
            raise FakeLLMScriptError(
                f"FakeFactoryLLM has no script for call_name '{call_name}' "
                f"(scripted: {sorted(self.scripts)})"
            )
        script = self.scripts[call_name]
        if not script:
            n = sum(1 for c in self.calls if c.call_name == call_name)
            raise FakeLLMScriptError(
                f"FakeFactoryLLM script for call_name '{call_name}' is exhausted "
                f"(attempt #{n} but the script had {n - 1} response(s))"
            )
        response = script.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    # ---- The FactoryLLM contract ----

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        *,
        call_name: str,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        response = self._pop(call_name, model, system, user, reasoning_effort)
        if not isinstance(response, str):
            raise FakeLLMScriptError(
                f"chat('{call_name}') popped a {type(response).__name__} response; "
                "chat scripts must contain str (or Exception) responses"
            )
        return response

    def json_call(
        self,
        model: str,
        system: str,
        user: str,
        *,
        call_name: str,
        reasoning_effort: Optional[str] = None,
    ) -> dict | list:
        """Return a structured response or fail on malformed output."""
        response = self._pop(call_name, model, system, user, reasoning_effort)
        if isinstance(response, (dict, list)):
            return response
        raise FakeLLMMalformedJSONError(
            f"json_call('{call_name}') response is not valid JSON: {response[:200]!r}"
        )

    # ---- Test assertion helpers ----

    def assert_called(self, call_name: str, times: int | None = None) -> None:
        """Assert call_name was attempted (exactly ``times`` if given)."""
        count = sum(1 for call in self.calls if call.call_name == call_name)
        if times is None:
            assert count > 0, (
                f"expected at least one '{call_name}' call; recorded calls: "
                f"{[c.call_name for c in self.calls]}"
            )
        else:
            assert count == times, (
                f"expected exactly {times} '{call_name}' call(s), got {count}; "
                f"recorded calls: {[c.call_name for c in self.calls]}"
            )

    def prompts_for(self, call_name: str) -> list[str]:
        """The user prompts of every recorded attempt for call_name, in order."""
        return [call.user for call in self.calls if call.call_name == call_name]

    def reasonings_for(self, call_name: str) -> list[Optional[str]]:
        """The reasoning_effort seen on every recorded attempt for call_name."""
        return [
            call.reasoning_effort for call in self.calls if call.call_name == call_name
        ]

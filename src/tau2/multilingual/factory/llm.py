# Copyright Sierra
"""The factory's single LLM seam.

Every factory stage (drafting, translation, verification, calibration) calls
the model exclusively through :class:`FactoryLLM`, and every factory entry
point takes an optional ``llm: FactoryLLM`` parameter, so tests and dry runs
inject a fake with the same contract instead of monkeypatching transports.

The implementation rides on ``tau2.utils.llm_utils.generate``, the same retry,
cost-accounting, and call-name logging path as the simulators.
"""

import json
from typing import Optional, TypeVar, Union

from pydantic import BaseModel, ValidationError

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

M = TypeVar("M", bound=BaseModel)


class FactoryLLM:
    """Thin chat/JSON wrapper over ``llm_utils.generate``.

    The contract (``chat`` and ``json_call`` signatures) is frozen: fakes used
    in tests implement exactly these two methods. The optional keyword-only
    ``reasoning_effort`` is forwarded to ``generate()`` only when set; when
    omitted, ``generate()`` applies its own per-model default.
    """

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        *,
        call_name: str,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """One system+user completion; returns the assistant text.

        ``reasoning_effort`` (e.g. ``"medium"``/``"high"``) is forwarded to the
        underlying ``generate()`` only when not None; otherwise generate's own
        per-model default applies.
        """
        extra: dict = {}
        if reasoning_effort is not None:
            extra["reasoning_effort"] = reasoning_effort
        message = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=system),
                UserMessage(role="user", content=user),
            ],
            call_name=call_name,
            **extra,
        )
        return message.content or ""

    def json_call(
        self,
        model: str,
        system: str,
        user: str,
        *,
        call_name: str,
        reasoning_effort: Optional[str] = None,
    ) -> Union[dict, list]:
        """A completion that must parse to a JSON object or array.

        Parses with ``extract_json_from_llm_response`` (markdown-fence
        tolerant). Invalid output fails loudly; retries belong to
        ``generate()``, not to a hidden second prompt in this wrapper.

        ``reasoning_effort`` is forwarded to the inner ``self.chat`` (and thus
        to ``generate()``) only when not None.

        Raises:
            ValueError: If no attempt produced valid JSON.
        """
        response = self.chat(
            model,
            system,
            user,
            call_name=call_name,
            reasoning_effort=reasoning_effort,
        )
        try:
            parsed = json.loads(extract_json_from_llm_response(response))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM call '{call_name}' returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, (dict, list)):
            raise ValueError(
                f"LLM call '{call_name}' returned {type(parsed).__name__}; "
                "expected a JSON object or array"
            )
        return parsed


def json_model(
    llm: FactoryLLM,
    response_model: type[M],
    model: str,
    system: str,
    user: str,
    *,
    call_name: str,
    reasoning_effort: Optional[str] = None,
) -> M:
    """A ``json_call`` validated into a pydantic response model.

    The typed counterpart of ``judges.base.judge_structured`` for the factory
    seam: the reply must parse to a JSON OBJECT and ``model_validate`` into
    ``response_model``. A shape failure raises loudly without re-prompting.

    A free function (not a ``FactoryLLM`` method) so the frozen two-method
    fake contract (``chat`` + ``json_call``) stays exactly as tests implement
    it.
    """
    data = llm.json_call(
        model,
        system,
        user,
        call_name=call_name,
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(data, dict):
        raise ValueError(
            f"LLM call '{call_name}' returned a JSON array; expected a JSON "
            f"object with {response_model.__name__} fields"
        )
    try:
        return response_model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"LLM call '{call_name}' failed {response_model.__name__} validation: {exc}"
        ) from exc


_DEFAULT_LLM: Optional[FactoryLLM] = None


def get_default_llm() -> FactoryLLM:
    """The process-wide default FactoryLLM (lazily created singleton)."""
    global _DEFAULT_LLM
    if _DEFAULT_LLM is None:
        _DEFAULT_LLM = FactoryLLM()
    return _DEFAULT_LLM

"""Common execution contract for independent diagnostic judges."""

from abc import ABC, abstractmethod

from tau2.data_model.judges import (
    DiagnosticId,
    JudgeContract,
    MeasureVerdict,
    UnitError,
)
from tau2.judges.run_evidence import RunEvidence


class BaseJudge(ABC):
    """A versioned definition evaluated against typed recorded evidence."""

    def __init__(self, contract: JudgeContract):
        self.contract = contract

    def resume(
        self, evidence: RunEvidence, measure: DiagnosticId, previous: MeasureVerdict
    ) -> MeasureVerdict:
        """Resume adapters that support preserving successful utterance verdicts."""
        raise NotImplementedError("This judge does not support partial retries")

    @abstractmethod
    def evaluate(self, evidence: RunEvidence, measure: DiagnosticId) -> MeasureVerdict:
        """Return measured/no-opportunity units, or raise a recorded unit error."""


def response_finish_reason(response) -> str | None:
    """Read the provider completion status retained by generate."""
    raw = getattr(response, "raw_data", None) or {}
    choices = raw.get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def require_complete_response(response) -> None:
    """Even syntactically valid JSON is incomplete when the provider hit its limit."""
    if response_finish_reason(response) == "length":
        raise ValueError("Judge response reached its output token limit")


def response_error(unit_id: str, exc: Exception, response=None) -> UnitError:
    """Retain the failed response and provider diagnostics without request audio."""
    raw = getattr(response, "raw_data", None) or {}
    return UnitError(
        unit_id=unit_id,
        kind="response",
        error=f"{type(exc).__name__}: {exc}",
        response_text=getattr(response, "content", None),
        finish_reason=response_finish_reason(response),
        usage=raw.get("usage"),
    )

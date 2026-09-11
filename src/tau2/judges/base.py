# Copyright Sierra
"""Shared judge plumbing: structured LLM calls, verdict shapes, scoring helpers."""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Callable, Iterable, Optional, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from tau2.data_model.message import Message
from tau2.data_model.simulation import (
    DeliveryFactorCheck,
    JudgeOutcome,
    NativenessFactorCheck,
)
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

R = TypeVar("R", bound=BaseModel)

#: Marker appended to a barged-in agent turn's text in LLM judge prompts — and
#: only there: deterministic checkers, lexical stats / corpus builders, and
#: preference inputs always see the raw text (design rule from PR #763). Each
#: judge prompt explains the marker; without it a judge reads caller-caused
#: truncation (an incomplete final word, a question cut off mid-sentence) as
#: an agent defect.
INTERRUPTED_TURN_MARKER = "[cut off by caller]"


def interruption_marked_text(text: str, interrupted: bool) -> str:
    """LLM-judge-facing rendering of one delivered utterance."""
    return f"{text} {INTERRUPTED_TURN_MARKER}" if interrupted else text


def judge_structured(
    model: str,
    messages: list[Message],
    response_model: type[R],
    call_name: str,
    model_args: Optional[dict] = None,
) -> R:
    """One LLM judge call through the ``generate()`` seam, validated into a model.

    The reply is JSON-extracted (fence/prose tolerant), must be a JSON object,
    and is ``model_validate``d into ``response_model``. A reply that fails
    parsing or validation raises ``ValueError``; harnesses catch it and record
    an ERROR outcome. API failures inside ``generate()`` propagate untouched.
    """
    reply = generate(
        model=model,
        messages=messages,
        call_name=call_name,
        **(model_args or {}),
    )
    try:
        payload = json.loads(extract_json_from_llm_response(reply.content or ""))
        if not isinstance(payload, dict):
            raise ValueError(
                f"judge reply is JSON but not an object: {type(payload).__name__}"
            )
        return response_model.model_validate(payload)
    except (ValueError, ValidationError) as exc:
        raise ValueError(
            f"{call_name}: judge reply failed {response_model.__name__} "
            f"validation: {exc}"
        ) from exc


class VerdictReplyBase(BaseModel):
    """The opportunity/violated verdict shape both judges ask the model for.

    Field validators absorb known model quirks: stringly booleans ("false" /
    "no" / "0") must not read as truthy, and null text fields coerce to empty
    strings. Extra keys are ignored.
    """

    model_config = ConfigDict(extra="ignore")

    opportunity: Annotated[
        bool,
        Field(
            description="Whether the conversation/clip gave this factor a "
            "chance to surface."
        ),
    ]
    violated: Annotated[
        bool,
        Field(
            description="Whether the agent behaved non-natively / the problem "
            "audibly occurred (only meaningful when opportunity is true)."
        ),
    ] = False
    reasoning: Annotated[
        str, Field(description="Brief explanation of the verdict.")
    ] = ""
    quote: Annotated[
        str,
        Field(
            description="Exact offending span when the violation localizes to "
            "one identifiable phrase; empty when diffuse or when there is no "
            "violation."
        ),
    ] = ""

    @field_validator("opportunity", "violated", mode="before")
    @classmethod
    def _lenient_bool(cls, value: object) -> bool:
        """Coerce a JSON flag to bool: some models emit the *string* "false"
        (or "no"/"0"), which plain truthiness would misread as True."""
        if isinstance(value, str):
            return value.strip().lower() not in {
                "",
                "false",
                "0",
                "no",
                "none",
                "null",
            }
        return bool(value)

    @field_validator("reasoning", "quote", mode="before")
    @classmethod
    def _none_to_empty(cls, value: object) -> object:
        return "" if value is None else value

    @model_validator(mode="after")
    def _violation_requires_reasoning(self) -> "VerdictReplyBase":
        """A violation verdict must explain itself.

        Deliberately on the shared base, not per-suite: every consumer of
        this verdict shape (quality factors, nativeness v14, delivery pack
        factors) prompts the model for ``reasoning``, and a FAIL without an
        explanation is unactionable evidence — on the 480-call es+pt
        judge-calibration corpus, quality-factor FAILs were ~100%
        evidence-free before this rule because nothing required the field. A
        violated reply with empty reasoning now fails validation and is
        recorded as a loud ERROR outcome by the harnesses, never a silently
        evidence-free FAIL.
        """
        if self.opportunity and self.violated and not self.reasoning.strip():
            raise ValueError("a violation verdict requires non-empty reasoning")
        return self


def verdict_outcome(
    verdict: VerdictReplyBase,
) -> tuple[JudgeOutcome, Optional[str], Optional[str]]:
    """Map an opportunity/violated verdict to ``(outcome, evidence, quote)``.

    NO_OPPORTUNITY when the factor never had a chance to surface; FAIL with the
    judge's reasoning as evidence (and the offending quote, when it localizes
    to one span); PASS otherwise. Evidence/quote are None on non-FAIL outcomes.
    """
    if not verdict.opportunity:
        return JudgeOutcome.NO_OPPORTUNITY, None, None
    if verdict.violated:
        return (
            JudgeOutcome.FAIL,
            verdict.reasoning.strip() or None,
            verdict.quote.strip() or None,
        )
    return JudgeOutcome.PASS, None, None


def severity_weighted_score(
    checks: Iterable[NativenessFactorCheck | DeliveryFactorCheck],
) -> Optional[float]:
    """Severity-weighted pass fraction over the factor checks that FIRED.

    ``score = Σ(severity over PASS) / Σ(severity over PASS+FAIL)``, or None
    when nothing fired (NO_OPPORTUNITY / DEFERRED / ERROR are excluded from
    both sums).
    """
    fired = [c for c in checks if c.outcome in (JudgeOutcome.PASS, JudgeOutcome.FAIL)]
    if not fired:
        return None
    denom = sum(c.severity for c in fired)
    passed = sum(c.severity for c in fired if c.outcome == JudgeOutcome.PASS)
    return passed / denom


def binary_pass_fraction(
    checks: Iterable[NativenessFactorCheck | DeliveryFactorCheck],
) -> Optional[float]:
    """Equal-weight pass fraction over binary factors that had an opportunity."""
    fired = [
        check
        for check in checks
        if check.outcome in (JudgeOutcome.PASS, JudgeOutcome.FAIL)
    ]
    if not fired:
        return None
    return sum(check.outcome == JudgeOutcome.PASS for check in fired) / len(fired)


T = TypeVar("T")
U = TypeVar("U")


def ordered_map(fn: Callable[[T], U], items: Iterable[T], concurrency: int) -> list[U]:
    """Map ``fn`` over ``items`` preserving input order.

    Uses a ``ThreadPoolExecutor`` (``pool.map`` keeps order) when
    ``concurrency > 1`` and there is more than one item; otherwise runs
    serially. Exceptions from ``fn`` propagate — per-unit error policy belongs
    in ``fn`` itself.
    """
    items = list(items)
    if concurrency <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(fn, items))

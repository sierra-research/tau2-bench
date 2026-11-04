"""Portable frozen quality-v7 tool-use measure, independent of annotation code."""

import json
import re
from typing import Annotated

from pydantic import ConfigDict, Field, field_validator, model_validator

from tau2.data_model.judges import (
    ContractModel,
    DiagnosticId,
    ImplementationFamily,
    InputExclusions,
    JudgeContract,
    Measure,
    MeasureVerdict,
    ToolCriterion,
    UnitVerdict,
)
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task, UserInstructions
from tau2.judges.base import BaseJudge
from tau2.judges.run_evidence import RunEvidence
from tau2.utils.llm_utils import extract_json_from_llm_response, generate
from tau2.utils.transcript.display import marked_text
from tau2.utils.transcript.views import extract_spoken_turns

TOOL_USE_ADAPTER_VERSION = "quality-v7-context-v1"
AUTH_TOOLS = {
    "telecom": {"get_customer_by_phone", "get_customer_by_name", "get_customer_by_id"},
    "airline": {"get_user_details"},
    "retail": {"find_user_id_by_name_zip", "find_user_id_by_email"},
}


class QualityTaskContract(ContractModel):
    """Exact frozen caller-visible context, excluding evaluator goldens."""

    task_id: Annotated[str, Field(description="Recorded task ID.")]
    user_instructions: Annotated[
        UserInstructions, Field(description="Recorded caller-visible instructions.")
    ]
    ticket: Annotated[
        str | None,
        Field(default=None, description="Caller-visible ticket when present."),
    ]


class ToolUseContext(ContractModel):
    """Typed evidence consumed by the frozen quality-v7 prompt."""

    domain: Annotated[str, Field(description="Recorded environment domain.")]
    task_contract: Annotated[
        QualityTaskContract, Field(description="Caller-visible task contract.")
    ]
    domain_policy: Annotated[
        str, Field(description="Actual saved policy, not today's domain policy.")
    ]
    transcript: Annotated[
        str, Field(description="Delivered turns with frozen interruption markers.")
    ]
    tool_trace: Annotated[
        str, Field(description="Frozen deduplicated agent tool trace.")
    ]
    tool_calls: Annotated[
        int, Field(description="Agent-requested calls in the stored messages.")
    ]
    tool_errors: Annotated[int, Field(description="Agent-requested error results.")]
    auth_calls: Annotated[int, Field(description="Registered authentication lookups.")]
    auth_no_match: Annotated[
        int,
        Field(
            description="Authentication no-match responses, requiring cause attribution."
        ),
    ]


class QualityReply(ContractModel):
    """Validated calibrated reply; string false is false, unknown values error."""

    model_config = ConfigDict(extra="ignore")

    opportunity: Annotated[
        bool, Field(description="Whether this criterion had an opportunity.")
    ]
    violated: Annotated[bool, Field(description="Whether the criterion was violated.")]
    reasoning: Annotated[
        str, Field(default="", description="Concrete judge explanation.")
    ]
    quote: Annotated[
        str, Field(default="", description="Exact offending transcript span.")
    ]

    @field_validator("reasoning", "quote", mode="before")
    @classmethod
    def nullable_text(cls, value):
        """Preserve the frozen parser's known null-text normalization."""
        return "" if value is None else value

    @model_validator(mode="after")
    def explained_violation(self):
        """Reject evidence-free failures and contradictory opportunity flags."""
        if self.violated and not self.opportunity:
            raise ValueError("Violation requires an opportunity")
        if self.violated and not self.reasoning.strip():
            raise ValueError("Violation requires concrete reasoning")
        return self


def build_context(simulation: SimulationRun, task: Task, domain: str) -> ToolUseContext:
    """Preserve frozen transcript, trace and selected preclassification semantics."""
    if domain not in AUTH_TOOLS:
        raise ValueError("Tool-use adapter supports only the released domains")
    transcript = (
        "\n".join(
            f"{turn.speaker}: "
            + marked_text(
                turn.text,
                turn.speaker == "agent" and turn.interrupted,
                marker="[cut off by caller]",
            )
            for turn in extract_spoken_turns(simulation)
        )
        or "(no spoken transcript)"
    )
    lines = []
    seen_calls = set()
    seen_results = set()
    calls = {}
    n_calls = n_errors = n_auth = n_no_match = 0
    for message in simulation.get_messages():
        for call in getattr(message, "tool_calls", None) or []:
            if call.requestor != "assistant":
                continue
            n_calls += 1
            if call.id:
                calls[call.id] = call
            if call.name in AUTH_TOOLS[domain]:
                n_auth += 1
            identity = call.id or f"{call.name}:{len(lines)}"
            if identity not in seen_calls:
                seen_calls.add(identity)
                lines.append(
                    f"tool_call id={call.id} name={call.name} arguments={json.dumps(call.arguments, ensure_ascii=False, default=str)}"
                )
        if getattr(message, "role", None) != "tool" or message.requestor != "assistant":
            continue
        n_errors += int(message.error)
        call = calls.get(message.id)
        text = (message.content or "").strip()
        if call and call.name in AUTH_TOOLS[domain]:
            no_match = (
                not message.error
                and (
                    text in {"", "[]", "null", "None"}
                    or re.match(r"^no records found", text, re.I)
                )
            ) or (message.error and re.search(r"not\s+found", text, re.I))
            n_no_match += int(bool(no_match))
        if message.id not in seen_results:
            seen_results.add(message.id)
            content = (message.content or "").replace("\n", " ")[:1200]
            lines.append(f"tool_result id={message.id} error={message.error} {content}")
    return ToolUseContext(
        domain=domain,
        task_contract=QualityTaskContract(
            task_id=task.id,
            user_instructions=task.user_scenario.instructions,
            ticket=task.ticket,
        ),
        domain_policy=simulation.policy or "(not recorded)",
        transcript=transcript,
        tool_trace="\n".join(lines) or "(no tool calls)",
        tool_calls=n_calls,
        tool_errors=n_errors,
        auth_calls=n_auth,
        auth_no_match=n_no_match,
    )


def render_prompt(
    context: ToolUseContext, criterion: ToolCriterion, contract: JudgeContract
) -> str:
    """Format the exact archived template without retuning its instructions."""
    return contract.prompt.user_prompt_template.format(
        criterion_json=json.dumps(criterion.model_dump(), ensure_ascii=False, indent=2),
        domain=context.domain,
        task_contract_json=context.task_contract.model_dump_json(exclude_none=True),
        domain_policy=context.domain_policy,
        transcript=context.transcript,
        tool_trace=context.tool_trace,
    )


class ToolUseJudge(BaseJudge):
    """Evaluate the three calibrated tool-use criteria as one call measure."""

    def evaluate(self, evidence: RunEvidence, measure: DiagnosticId) -> MeasureVerdict:
        """Evaluate recorded evidence without changing the calibrated request or parser."""
        contract = self.contract
        simulation, task, domain = evidence.simulation, evidence.task, evidence.domain
        if (
            contract.implementation != ImplementationFamily.TOOL_USE
            or measure != Measure.TOOL_USE
        ):
            raise ValueError("Unsupported calibrated tool-use contract")
        # The full spoken transcript (opener and transfer line included) is
        # context for judging tool calls; the transfer line is evidence of
        # policy-conformant transfers. No speech is excluded.
        if contract.input_exclusions != InputExclusions.NONE:
            raise ValueError("Tool-use v7 judges the unfiltered transcript")
        criterion_ids = [
            criterion.criterion_id for criterion in contract.quality_criteria
        ]
        if len(criterion_ids) != 3 or set(criterion_ids) != {
            "agent_caused_tool_error",
            "auth_arg_mismatch",
            "incorrect_tool_parameters",
        }:
            raise ValueError("Tool-use contract requires the three calibrated criteria")
        context = build_context(simulation, task, domain)
        outcomes = []
        for criterion in contract.quality_criteria:
            identifier = criterion.criterion_id
            if identifier == "agent_caused_tool_error" and not context.tool_errors:
                reply = QualityReply(opportunity=False, violated=False)
            elif identifier == "auth_arg_mismatch" and not context.auth_calls:
                reply = QualityReply(opportunity=False, violated=False)
            elif identifier == "auth_arg_mismatch" and not context.auth_no_match:
                reply = QualityReply(opportunity=True, violated=False)
            elif identifier == "incorrect_tool_parameters" and not context.tool_calls:
                reply = QualityReply(opportunity=False, violated=False)
            else:
                response = generate(
                    model=contract.model,
                    messages=[
                        SystemMessage(
                            role="system", content=contract.prompt.system_prompt
                        ),
                        UserMessage(
                            role="user",
                            content=render_prompt(context, criterion, contract),
                        ),
                    ],
                    call_name=f"quality_judge_{identifier}",
                    **contract.model_args,
                )
                reply = QualityReply.model_validate_json(
                    extract_json_from_llm_response(response.content or "")
                )
            outcomes.append((identifier, reply))
        return MeasureVerdict(
            units=[
                UnitVerdict(
                    unit_id="call",
                    opportunity=any(r.opportunity for _, r in outcomes),
                    violation=any(r.opportunity and r.violated for _, r in outcomes),
                    evidence=json.dumps(
                        {key: reply.model_dump() for key, reply in outcomes},
                        ensure_ascii=False,
                    ),
                )
            ]
        )

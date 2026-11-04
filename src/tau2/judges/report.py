"""Compact terminal reports over diagnostic exports; the JSON export stays canonical."""

from collections import Counter
from pathlib import Path

from tau2.data_model.judges import (
    ImplementationFamily,
    JudgeCatalog,
    PlanAction,
    SuitePlan,
)
from tau2.judges.summary import DiagnosticExport, MatrixDiagnosticExport
from tau2.utils.table import format_fraction, format_table

JUDGE_ACTIONS = frozenset(
    {
        PlanAction.MISSING,
        PlanAction.STALE,
        PlanAction.FORCE,
        PlanAction.RETRY_ERROR,
    }
)


def _cells(
    export: DiagnosticExport | MatrixDiagnosticExport,
) -> list[tuple[str, DiagnosticExport | None]]:
    if isinstance(export, DiagnosticExport):
        return [("", export)]
    return [
        (f"{entry.cell.language}/{entry.cell.domain}", entry.diagnostics)
        for entry in export.cells
    ]


def _table(
    export: DiagnosticExport | MatrixDiagnosticExport,
    headers: list[str],
    rows: list[list[object]],
) -> str:
    """Matrix rows lead with their cell; an ordinary run has no cell column."""
    if isinstance(export, DiagnosticExport):
        return format_table(headers, [row[1:] for row in rows])
    return format_table(["cell", *headers], rows)


def render_summary(export: DiagnosticExport | MatrixDiagnosticExport) -> str:
    """One row per cell × measure: call coverage, unit outcomes and completeness."""
    rows = []
    for cell, diagnostics in _cells(export):
        if diagnostics is None:
            rows.append([cell, "(no results)", "", "", "", "", "", "", "", ""])
            continue
        for summary in diagnostics.summary:
            rows.append(
                [
                    cell,
                    str(summary.measure),
                    summary.applicable_calls,
                    summary.judged_calls,
                    summary.error_calls,
                    summary.not_judged_calls,
                    summary.pass_units,
                    summary.fail_units,
                    format_fraction(summary.observed_pass_fraction),
                    "yes" if summary.complete else "no",
                ]
            )
    if not rows:
        return "No diagnostics selected."
    return _table(
        export,
        [
            "measure",
            "applicable",
            "judged",
            "errors",
            "not_judged",
            "pass",
            "fail",
            "pass_frac",
            "complete",
        ],
        rows,
    )


def render_plan(
    export: DiagnosticExport | MatrixDiagnosticExport,
    model_calls: int | None = None,
) -> str:
    """One row per cell × measure with planned judge/reuse/not-applicable counts."""
    rows = []
    totals: Counter[str] = Counter()
    for cell, diagnostics in _cells(export):
        if diagnostics is None:
            rows.append([cell, "(no results)", "", "", "", ""])
            continue
        actions: dict[str, Counter[PlanAction]] = {}
        for item in diagnostics.plan.items:
            actions.setdefault(str(item.measure), Counter())[item.action] += 1
        for measure, counts in actions.items():
            row = {
                "judge": sum(counts[action] for action in JUDGE_ACTIONS),
                "reuse": counts[PlanAction.REUSE],
                "not_applicable": counts[PlanAction.NOT_APPLICABLE],
                "missing_input": counts[PlanAction.MISSING_INPUT],
            }
            totals.update(row)
            rows.append([cell, measure, *row.values()])
    if not rows:
        return "No diagnostics selected."
    lines = [
        _table(
            export,
            ["measure", "judge", "reuse", "not_applicable", "missing_input"],
            rows,
        ),
        "",
        f"Verdicts to judge: {totals['judge']}; reused: {totals['reuse']}.",
    ]
    if model_calls is not None:
        lines.append(f"Estimated model calls (upper bound): {model_calls}.")
    return "\n".join(lines)


def estimate_model_calls(results: Path, catalog: JudgeCatalog, plan: SuitePlan) -> int:
    """Upper-bound model requests for planned work, from the recorded calls.

    Utterance-level judges make one request per delivered agent turn after the
    contract's exclusions; call-level judges make one; tool use makes at most
    one per criterion. Batched nativeness factors share a request per unit.
    """
    from tau2.judges.criteria import _select_criteria
    from tau2.judges.exclusions import judged_agent_turns
    from tau2.judges.run_evidence import SavedRun

    work = [item for item in plan.items if item.action in JUDGE_ACTIONS]
    if not work:
        return 0
    transcripts = dict(
        (simulation.id, (simulation, opener))
        for simulation, opener in SavedRun(results).iter_transcripts(
            {item.input.simulation_id for item in work}
        )
    )
    contracts = {contract.contract_id: contract for contract in catalog.contracts}
    requests = set()
    for item in work:
        assert item.identity is not None
        contract = contracts[item.identity.contract_id]
        call = (item.input.run_id, item.input.simulation_id, item.input.trial)
        if contract.implementation == ImplementationFamily.TOOL_USE:
            requests.update(
                (*call, contract.contract_id, criterion.criterion_id)
                for criterion in contract.quality_criteria
            )
            continue
        group, per_call = str(item.measure), contract.evaluation_level == "call"
        if contract.implementation != ImplementationFamily.FIDELITY:
            try:
                selection = _select_criteria(
                    contract,
                    item.input.language,
                    item.measure,
                    item.input.agent_gender,
                    item.input.caller_gender,
                )
            except ValueError:
                continue  # Recorded as a judge error without a model request.
            per_call = selection.is_call
            if selection.batched:
                group = "call" if per_call else "utterance"
        if per_call:
            requests.add((*call, contract.contract_id, group, "call"))
            continue
        simulation, opener = transcripts[item.input.simulation_id]
        requests.update(
            (*call, contract.contract_id, group, turn.segment_id or str(turn.index))
            for turn in judged_agent_turns(
                simulation, opener, contract.input_exclusions
            )
        )
    return len(requests)


def estimate_export_calls(
    results: Path,
    catalog: JudgeCatalog,
    export: DiagnosticExport | MatrixDiagnosticExport,
) -> int:
    """Sum the planned-request bound over an ordinary run or each saved matrix cell."""
    if isinstance(export, DiagnosticExport):
        return estimate_model_calls(results, catalog, export.plan)
    return sum(
        estimate_model_calls(
            results / f"{entry.cell.language}-{entry.cell.domain}",
            catalog,
            entry.diagnostics.plan,
        )
        for entry in export.cells
        if entry.diagnostics is not None
    )

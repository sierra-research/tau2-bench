"""Explicit candidate execution isolated from ordinary production sidecars."""

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import Field

from tau2.data_model.judges import ContractModel, JudgeCatalog, SuitePlan, UnitVerdict
from tau2.judges.authoring_models import CandidateJudge
from tau2.judges.run_evidence import progress_path
from tau2.judges.suite import judge_saved_run
from tau2.judges.validation import Prediction, PredictionBatch
from tau2.utils.artifacts import content_digest as digest
from tau2.utils.artifacts import write_artifact


class CandidateRun(ContractModel):
    candidate: Annotated[
        CandidateJudge,
        Field(
            description="Exact unvalidated definition used only in this candidate workspace."
        ),
    ]
    plan: Annotated[
        SuitePlan, Field(description="Explicit pending/error/applicability coverage.")
    ]
    predictions: Annotated[
        PredictionBatch,
        Field(description="Usable units and failures for human alignment."),
    ]


def candidate_catalog(candidate: CandidateJudge) -> JudgeCatalog:
    """Create an in-memory selection for explicit candidate runs only."""
    return JudgeCatalog(
        revision="candidate-" + digest(candidate),
        donor_revision=candidate.revision,
        source_prompt_sha256=candidate.review.content_sha256,
        contracts=[candidate.contract],
        pairs=[candidate.scope],
    )


def run_candidate(
    results: Path, candidate: CandidateJudge, output: Path, *, dry_run: bool = False
) -> CandidateRun:
    """Reuse saved units in a separate workspace; never write production diagnostics."""
    source = results.resolve()
    destination = output.resolve()
    if (
        destination == source
        or destination == progress_path(source)
        or (source.is_dir() and destination.is_relative_to(source))
        or source.is_relative_to(destination)
    ):
        raise ValueError(
            "Candidate output must be separate from saved results and production diagnostics"
        )
    catalog = candidate_catalog(candidate)
    progress = destination / "verdicts.json"
    if (destination / "candidate.json").exists() and CandidateJudge.model_validate_json(
        (destination / "candidate.json").read_text()
    ) != candidate:
        raise ValueError("Candidate output belongs to a different definition")
    if not dry_run:
        write_artifact(candidate, destination / "candidate.json")
    result = judge_saved_run(
        source,
        catalog,
        only=[candidate.scope.measure],
        execute=not dry_run,
        output=progress,
    )
    plan, stored = result.plan, result.verdicts
    desired = [item.identity for item in plan.items if item.identity is not None]
    predictions = []
    for row in stored:
        item = row.identity.input
        if row.identity not in desired:
            continue
        units = (
            [*row.verdict.units, *row.verdict.unit_errors]
            if row.verdict is not None
            else [None]
        )
        for unit in units:
            unit_id = unit.unit_id if unit else "__error__"
            judged = isinstance(unit, UnitVerdict)
            key = json.dumps(
                [item.run_id, item.simulation_id, item.trial, unit_id],
                separators=(",", ":"),
            )
            predictions.append(
                Prediction(
                    row_id=key,
                    input_sha256=item.input_fingerprint,
                    language=item.language,
                    domain=item.domain or "",
                    agent_gender=item.agent_gender,
                    status="judged" if judged else "error",
                    opportunity=unit.opportunity if judged else None,
                    violation=unit.violation if judged else None,
                    error=None if judged else (unit.error if unit else row.error),
                )
            )
    artifact = CandidateRun(
        candidate=candidate,
        plan=plan,
        predictions=PredictionBatch(
            candidate_sha256=digest(candidate), rows=predictions
        ),
    )
    if not dry_run:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=destination, delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(artifact.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.replace(temporary, destination / "candidate-results.json")
            finally:
                temporary.unlink(missing_ok=True)
    return artifact

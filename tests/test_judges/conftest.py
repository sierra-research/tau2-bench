"""Reviewed judge fixture shared across authoring and registry workflows."""

import pytest

from tau2.judges.authoring import compile_judge
from tau2.judges.authoring_models import CompileJudgeRequest, DraftJudgeRequest


@pytest.fixture
def candidate(tmp_path):
    from tau2.judges.authoring_models import CriteriaDraft, JudgeDraft
    from tau2.utils.artifacts import ReviewApproval, content_digest

    request = DraftJudgeRequest(
        mode="new_diagnostic",
        diagnostic_id="promise_accuracy",
        language="ro",
        variety="Romanian in Romania",
        agent_requirements="Only supported dates.",
        domains=["retail"],
        evaluation_level="call",
        agent_gender=None,
        author_model="fixture-author",
        author_model_args={},
        judge_model="fixture-judge",
        judge_model_args={},
    )
    draft = JudgeDraft(
        request=request,
        reference=None,
        criteria=CriteriaDraft(
            question="Unsupported date?",
            opportunity="Promise made.",
            positive_examples="Supported date.",
            negative_examples="Invented date.",
        ),
    )
    return compile_judge(
        CompileJudgeRequest(
            draft=draft,
            review=ReviewApproval(
                reviewed_by="human", content_sha256=content_digest(draft)
            ),
            revision="v1",
        ),
        tmp_path / "compiled.json",
    )

"""Speech-fidelity v6 judge over tick-aligned agent audio."""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from tau2.data_model.judges import (
    DiagnosticId,
    InputExclusions,
    MeasureVerdict,
    UnitVerdict,
)
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.judges.base import BaseJudge, require_complete_response, response_error
from tau2.judges.exclusions import injected_opener
from tau2.judges.run_evidence import (
    RunEvidence,
    inspect_audio,
    live_input_errors,
    load_audio_clips,
)
from tau2.utils.llm_utils import extract_json_from_llm_response, generate


class Finding(BaseModel):
    """One audible fidelity issue and its clip-truncation classification."""

    model_config = ConfigDict(extra="forbid")
    category: Annotated[
        Literal[
            "mispronunciation",
            "word_substitution",
            "missing_word",
            "extra_or_hallucinated_word",
            "number_date_currency",
            "email_url_code",
            "punctuation_or_formatting",
            "acronym_brand_name",
            "clipped_or_garbled",
            "clip_truncation",
            "other",
        ],
        Field(description="Fidelity error type or clip-boundary truncation."),
    ]
    issue: Annotated[str, Field(min_length=1, description="Audible discrepancy.")]
    time_range: Annotated[
        str | None,
        Field(description="Approximate start-end seconds within the supplied clip."),
    ]
    severity: Annotated[int, Field(ge=1, le=3, description="Finding severity.")]
    confidence: Annotated[float, Field(ge=0, le=1, description="Finding confidence.")]
    is_truncation: Annotated[
        StrictBool,
        Field(description="Whether the discrepancy may be a clip-cutoff artifact."),
    ]

    @model_validator(mode="after")
    def truncation_category_requires_flag(self):
        if self.category == "clip_truncation" and not self.is_truncation:
            raise ValueError("clip_truncation findings must set is_truncation=true")
        return self


class FidelityReply(BaseModel):
    """Require an explicit findings list; malformed replies never masquerade as clean."""

    model_config = ConfigDict(extra="forbid")
    findings: Annotated[
        list[Finding], Field(description="Audible speech-fidelity findings.")
    ]

    @model_validator(mode="before")
    @classmethod
    def drop_no_issue_findings(cls, data):
        """The rubric's severity 0 means no clear issue, to be returned as an
        empty findings list. A finding the judge still marks severity 0 is that
        same no-issue answer, so it is dropped rather than failing the reply."""
        if isinstance(data, dict) and isinstance(data.get("findings"), list):
            data = {
                **data,
                "findings": [
                    finding
                    for finding in data["findings"]
                    if not (
                        isinstance(finding, dict)
                        and type(finding.get("severity")) in (int, float)
                        and finding["severity"] == 0
                    )
                ],
            }
        return data


class FidelityJudge(BaseJudge):
    """Execute the calibrated fidelity protocol."""

    def resume(
        self, evidence: RunEvidence, measure: DiagnosticId, previous: MeasureVerdict
    ) -> MeasureVerdict:
        """Retry failed units while retaining successful units from matching inputs."""
        return self.evaluate(evidence, measure, previous=previous)

    def evaluate(
        self,
        evidence: RunEvidence,
        measure: DiagnosticId,
        *,
        previous: MeasureVerdict | None = None,
    ) -> MeasureVerdict:
        """Evaluate recorded speech against the versioned fidelity prompt."""
        contract = self.contract
        simulation, language = evidence.simulation, evidence.language
        material = contract.speech_material
        if contract.implementation != "speech-fidelity-v6" or material is None:
            raise ValueError("Missing v6 speech material")
        if contract.input_exclusions != InputExclusions.OPENER_AND_CONTROL_TURNS:
            raise ValueError("Speech fidelity v6 requires the control-turn rule")
        audio = evidence.audio
        if audio is None and evidence.results_path is not None:
            audio = inspect_audio(evidence.results_path, simulation)
        if audio is None or audio.path is None:
            raise ValueError("Missing aligned agent audio")
        errors = live_input_errors(simulation)
        clips = load_audio_clips(audio.path, simulation)
        # Control markers identify whole utterances excluded from both judges.
        clips = [
            clip
            for clip in clips
            if not clip.has_control_marker and not clip.terminal_cutoff
        ]
        # The injected opener is delivery group 0 (tick 0). Audio-native runs
        # record it as text over one tick of silence.
        if injected_opener(simulation, evidence.opener):
            clips = [
                clip
                for clip in clips
                if not (
                    clip.index == 0
                    and clip.reference.strip() == (evidence.opener or "").strip()
                )
            ]
        if not clips and errors:
            return MeasureVerdict(units=[], unit_errors=errors)
        if not clips:
            return MeasureVerdict(
                units=[
                    UnitVerdict(
                        unit_id="call",
                        opportunity=False,
                        violation=False,
                        evidence="No delivered agent speech.",
                    )
                ]
            )
        locale = getattr(simulation.speech_environment, "locale", None)
        guidance = material.language_guidance.get(language)
        if guidance is None:
            raise ValueError("No speech-fidelity guidance for language")
        outcomes = []
        saved = {u.unit_id: u for u in previous.units} if previous else {}
        for clip in clips:
            unit_id = clip.segment_id or str(clip.index)
            if unit_id in saved:
                outcomes.append(saved[unit_id])
                continue
            prompt = contract.prompt.user_prompt_template.format(
                ref=clip.reference,
                was_interrupted=str(clip.interrupted).lower(),
                locale=locale or language,
                language_guidance=guidance,
                output_shape=material.output_shape,
            )
            response = None
            try:
                response = generate(
                    model=contract.model,
                    messages=[
                        SystemMessage(
                            role="system", content=contract.prompt.system_prompt
                        ),
                        UserMessage(
                            role="user", content=prompt, audio_content=clip.wav_base64
                        ),
                    ],
                    call_name="speech_fidelity_judge",
                    **contract.model_args,
                )
                require_complete_response(response)
                reply = FidelityReply.model_validate_json(
                    extract_json_from_llm_response(response.content or "")
                )
            except Exception as exc:
                errors.append(response_error(unit_id, exc, response))
                continue
            retained = [f for f in reply.findings if not f.is_truncation]
            evidence = json.dumps(
                {
                    "filter_version": "explicit-truncation-v1",
                    "clip_duration_seconds": clip.duration_seconds,
                    "was_interrupted": clip.interrupted,
                    "findings": [
                        dict(
                            f.model_dump(),
                            excluded_as_truncation=f.is_truncation,
                        )
                        for f in reply.findings
                    ],
                },
                ensure_ascii=False,
            )
            outcomes.append(
                UnitVerdict(
                    unit_id=clip.segment_id or str(clip.index),
                    opportunity=True,
                    violation=bool(retained),
                    evidence=evidence,
                )
            )
        return MeasureVerdict(units=outcomes, unit_errors=errors)

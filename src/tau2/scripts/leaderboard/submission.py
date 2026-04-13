"""Pydantic models for tau-bench leaderboard submissions.

Single source of truth for all submission-related data models.
The JSON schema at web/leaderboard/public/submissions/schema.json
is auto-generated from these models.
"""

from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.simulation import Results as TrajectoryResults
from tau2.utils.pydantic_utils import BaseModelNoExtra


class ContactInfo(BaseModelNoExtra):
    """Contact information for the submission."""

    email: Optional[str] = Field(
        None, description="Contact email for questions about this submission"
    )
    name: Optional[str] = Field(None, description="Name of the submitter")
    github: Optional[str] = Field(None, description="GitHub username (optional)")


class DomainResults(BaseModelNoExtra):
    """Results for a specific domain."""

    pass_1: Optional[float] = Field(
        None, ge=0, le=100, description="Pass^1 success rate percentage"
    )
    pass_2: Optional[float] = Field(
        None, ge=0, le=100, description="Pass^2 success rate percentage"
    )
    pass_3: Optional[float] = Field(
        None, ge=0, le=100, description="Pass^3 success rate percentage"
    )
    pass_4: Optional[float] = Field(
        None, ge=0, le=100, description="Pass^4 success rate percentage"
    )
    cost: Optional[float] = Field(
        None,
        ge=0,
        description="Average cost in USD to run one trajectory in this domain (optional)",
    )
    retrieval_config: Optional[str] = Field(
        None,
        description="Retrieval method used for knowledge base access (banking_knowledge domain only)",
    )

    def get_pass_k(self, k: int) -> Optional[float]:
        """Get pass^k score for a given k."""
        if k < 1 or k > 4:
            raise ValueError(f"k must be between 1 and 4, got {k}")
        return getattr(self, f"pass_{k}")


class Results(BaseModelNoExtra):
    """Performance results for each domain."""

    retail: Optional[DomainResults] = None
    airline: Optional[DomainResults] = None
    telecom: Optional[DomainResults] = None
    banking_knowledge: Optional[DomainResults] = None

    @model_validator(mode="after")
    def _validate_banking_knowledge_retrieval_config(self) -> "Results":
        if (
            self.banking_knowledge is not None
            and self.banking_knowledge.retrieval_config is None
        ):
            raise ValueError(
                "banking_knowledge results require retrieval_config "
                "(e.g. 'bm25', 'terminal', 'text-emb-3-large')"
            )
        return self

    def get_domain(self, domain: str) -> Optional[DomainResults]:
        """Get results for a specific domain."""
        domain_lower = domain.lower()
        if domain_lower == "retail":
            return self.retail
        elif domain_lower == "airline":
            return self.airline
        elif domain_lower == "telecom":
            return self.telecom
        elif domain_lower == "banking_knowledge":
            return self.banking_knowledge
        else:
            raise ValueError(
                f"Invalid domain: {domain}. "
                f"Must be retail, airline, telecom, or banking_knowledge."
            )

    # Backward-compat alias
    get_domain_results = get_domain

    @property
    def available_domains(self) -> list[str]:
        """Get list of domains that have results."""
        domains = []
        if self.retail is not None:
            domains.append("retail")
        if self.airline is not None:
            domains.append("airline")
        if self.telecom is not None:
            domains.append("telecom")
        if self.banking_knowledge is not None:
            domains.append("banking_knowledge")
        return domains


class Reference(BaseModelNoExtra):
    """A reference link (paper, blog post, github repo, etc.)."""

    title: str = Field(..., description="Title or description of the reference")
    url: str = Field(..., description="URL to the reference")
    type: Optional[str] = Field(
        None,
        description="Type of reference: paper, blog_post, documentation, model_card, github, huggingface, other",
    )


class Verification(BaseModelNoExtra):
    """Verification details for result authenticity."""

    modified_prompts: Optional[bool] = Field(
        None,
        description="Whether any modifications were made to user simulator or agent prompts",
    )
    omitted_questions: Optional[bool] = Field(
        None,
        description="Whether any questions/tasks were omitted from the evaluation",
    )
    details: Optional[str] = Field(
        None, description="Additional verification details or explanations"
    )


class Methodology(BaseModelNoExtra):
    """Information about how the evaluation was conducted."""

    evaluation_date: Optional[date] = Field(
        None, description="Date when evaluation was conducted"
    )
    tau2_bench_version: Optional[str] = Field(
        None, description="Version of tau-bench used for evaluation"
    )
    user_simulator: Optional[str] = Field(
        None,
        description="For text: model name (e.g. 'gpt-4.1-2025-04-14'). "
        "For voice: version identifier (e.g. 'v1.0') anchored to git tag voice-user-sim-<version>.",
    )
    notes: Optional[str] = Field(
        None, description="Additional notes about the evaluation methodology"
    )
    verification: Optional[Verification] = Field(
        None, description="Verification details for result authenticity"
    )


class VoiceConfig(BaseModelNoExtra):
    """Voice-specific configuration for reproducing audio-native evaluations."""

    provider: str = Field(
        ...,
        description="Audio-native provider (e.g. 'openai', 'gemini', 'xai')",
    )
    model: str = Field(
        ...,
        description="Audio-native model identifier (e.g. 'gpt-realtime-1.5')",
    )
    tick_duration_seconds: Optional[float] = Field(
        None,
        description="Duration of each simulation tick in seconds",
    )
    max_steps_seconds: Optional[float] = Field(
        None,
        description="Maximum simulation duration in seconds",
    )
    user_tts_provider: Optional[str] = Field(
        None,
        description="User simulator TTS provider and model (e.g. 'elevenlabs/eleven_v3')",
    )


class Submission(BaseModelNoExtra):
    """Tau2-Bench Leaderboard Submission model."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "model_name": "GPT-4.1",
                    "model_organization": "OpenAI",
                    "submitting_organization": "OpenAI",
                    "submission_date": "2024-01-15",
                    "submission_type": "standard",
                    "contact_info": {
                        "email": "researcher@openai.com",
                        "name": "Jane Doe",
                        "github": "janedoe",
                    },
                    "is_new": True,
                    "trajectories_available": True,
                    "trajectory_files": {
                        "retail": "gpt-4.1_retail_default_gpt-4o_4trials.json",
                        "airline": "gpt-4.1_airline_default_gpt-4o_4trials.json",
                        "telecom": "gpt-4.1_telecom_default_gpt-4o_4trials.json",
                    },
                    "results": {
                        "retail": {
                            "pass_1": 85.5,
                            "pass_2": 92.3,
                            "pass_3": 96.1,
                            "pass_4": 98.2,
                        },
                        "airline": {
                            "pass_1": 78.9,
                            "pass_2": 89.4,
                            "pass_3": 94.7,
                            "pass_4": 97.1,
                        },
                        "telecom": {
                            "pass_1": 82.1,
                            "pass_2": 90.8,
                            "pass_3": 95.3,
                            "pass_4": 98.5,
                            "cost": 10.0,
                        },
                    },
                    "methodology": {
                        "evaluation_date": "2024-01-10",
                        "tau2_bench_version": "1.0.0",
                        "user_simulator": "gpt-4.1",
                        "notes": "Evaluated using default configuration with 4 trials per task",
                        "verification": {
                            "modified_prompts": False,
                            "omitted_questions": False,
                            "details": "Standard evaluation with unmodified prompts",
                        },
                    },
                }
            ]
        },
    )

    model_name: str = Field(..., description="Name of the model being evaluated")
    model_organization: str = Field(
        ..., description="Organization or company that developed the model"
    )
    submitting_organization: str = Field(
        ...,
        description="Organization that actually ran the evaluation and submitted the results",
    )
    submission_date: date = Field(..., description="Date of submission")
    submission_type: Literal["standard", "custom"] = Field(
        "standard",
        description="Type of submission: 'standard' uses the default tau2-bench scaffold, "
        "'custom' uses modified scaffolds (multi-model routers, additional tools, custom prompting, etc.)",
    )
    modality: Literal["text", "voice"] = Field(
        "text",
        description="Evaluation modality: 'text' for standard text-based, 'voice' for audio-native",
    )
    contact_info: ContactInfo = Field(..., description="Contact information")
    results: Results = Field(..., description="Performance results for each domain")
    is_new: bool = Field(
        False,
        description="Whether this model should be highlighted as new on the leaderboard",
    )
    trajectories_available: bool = Field(
        False,
        description="Whether trajectory files are available for this submission",
    )
    trajectory_files: Optional[dict[str, str]] = Field(
        None,
        description="Mapping of domain name to trajectory filename (e.g. {'retail': 'my-model_retail_...json'})",
    )
    references: Optional[list[Reference]] = Field(
        None,
        description="Links to papers, blog posts, documentation, or other resources",
    )
    methodology: Optional[Methodology] = Field(
        None, description="Information about how the evaluation was conducted"
    )
    voice_config: Optional[VoiceConfig] = Field(
        None,
        description="Voice-specific configuration for audio-native evaluations (only for voice submissions)",
    )
    reasoning_effort: Optional[str] = Field(
        None,
        description="Reasoning/thinking effort level used during evaluation "
        "(e.g. 'high', 'low', 'none', 'enabled')",
    )

    _submission_id: Optional[str] = None

    @property
    def submission_id(self) -> Optional[str]:
        """Get the submission ID (folder name)."""
        return self._submission_id

    def set_submission_id(self, submission_id: str) -> None:
        """Set the submission ID."""
        self._submission_id = submission_id

    @classmethod
    def load(cls, path: Path | str) -> "Submission":
        """Load a submission from a JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            submission = cls.model_validate_json(f.read())
        submission.set_submission_id(path.parent.name)
        return submission

    def get_pass_1_average(self) -> Optional[float]:
        """Get the average pass^1 score across all available domains."""
        scores = []
        for domain in self.results.available_domains:
            domain_results = self.results.get_domain(domain)
            if domain_results and domain_results.pass_1 is not None:
                scores.append(domain_results.pass_1)
        if not scores:
            return None
        return sum(scores) / len(scores)


# Constants
SUBMISSION_FILE_NAME = "submission.json"
TRAJECTORY_FILES_DIR_NAME = "trajectories"
MANIFEST_FILE_NAME = "manifest.json"
DOMAINS = ["retail", "airline", "telecom", "banking_knowledge"]
METRICS = ["pass_1", "pass_2", "pass_3", "pass_4", "cost"]


class LeaderboardManifest(BaseModelNoExtra):
    """Manifest file listing all submissions."""

    submissions: list[str] = Field(
        default_factory=list, description="List of text submission folder names"
    )
    voice_submissions: list[str] = Field(
        default_factory=list, description="List of voice submission folder names"
    )
    legacy_submissions: list[str] = Field(
        default_factory=list,
        description="List of legacy submission folder names (previous benchmark versions)",
    )
    last_updated: Optional[str] = Field(
        None, description="ISO timestamp of last update"
    )


class LeaderboardEntry(BaseModel):
    """A leaderboard entry with computed ranking information."""

    submission: Submission
    rank: Optional[int] = None
    score: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SubmissionData(BaseModelNoExtra):
    """Submission data with associated trajectory results."""

    submission_dir: str
    submission_file: str
    trajectory_files: list[str]
    submission: Submission
    results: list[TrajectoryResults]

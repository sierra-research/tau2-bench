"""Pydantic models for tau2-bench leaderboard submissions."""

from datetime import date
from typing import Optional

from pydantic import Field

from tau2.data_model.simulation import Results as TrajectoryResults
from tau2.utils.pydantic_utils import BaseModelNoExtra


class ContactInfo(BaseModelNoExtra):
    """Contact information for the submission."""

    email: str = Field(
        ..., description="Contact email for questions about this submission"
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


class Results(BaseModelNoExtra):
    """Performance results for each domain."""

    retail: Optional[DomainResults] = None
    airline: Optional[DomainResults] = None
    telecom: Optional[DomainResults] = None

    def get_domain_results(self, domain: str) -> DomainResults:
        """Get the domain results for a given domain."""
        if domain == "retail":
            return self.retail
        elif domain == "airline":
            return self.airline
        elif domain == "telecom":
            return self.telecom
        else:
            raise ValueError(f"Invalid domain: {domain}")


class Methodology(BaseModelNoExtra):
    """Information about how the evaluation was conducted."""

    evaluation_date: Optional[date] = Field(
        None, description="Date when evaluation was conducted"
    )
    tau2_bench_version: Optional[str] = Field(
        None, description="Version of tau2-bench used for evaluation"
    )
    user_simulator: Optional[str] = Field(
        None,
        description="Model used for user simulation during evaluation, or null if unknown",
    )
    notes: Optional[str] = Field(
        None, description="Additional notes about the evaluation methodology"
    )


class Submission(BaseModelNoExtra):
    """Tau2-Bench Leaderboard Submission model."""

    model_name: str = Field(..., description="Name of the model being evaluated")
    organization: str = Field(
        ..., description="Organization or company that developed the model"
    )
    submission_date: date = Field(..., description="Date of submission")
    contact_info: ContactInfo = Field(..., description="Contact information")
    results: Results = Field(..., description="Performance results for each domain")
    is_new: bool = Field(
        False,
        description="Whether this model should be highlighted as new on the leaderboard",
    )
    methodology: Optional[Methodology] = Field(
        None, description="Information about how the evaluation was conducted"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "examples": [
                {
                    "model_name": "GPT-4.1",
                    "organization": "OpenAI",
                    "submission_date": "2024-01-15",
                    "contact_info": {
                        "email": "researcher@openai.com",
                        "name": "Jane Doe",
                        "github": "janedoe",
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
                    "is_new": True,
                    "methodology": {
                        "evaluation_date": "2024-01-10",
                        "tau2_bench_version": "1.0.0",
                        "user_simulator": "gpt-4.1",
                        "notes": "Evaluated using default configuration with 4 trials per task",
                    },
                }
            ]
        }


SUBMISSION_FILE_NAME = "submission.json"
TRAJECTORY_FILES_DIR_NAME = "trajectories"


class SubmissionData(BaseModelNoExtra):
    """Submission data."""

    submission_dir: str
    submission_file: str
    trajectory_files: list[str]
    submission: Submission
    results: list[TrajectoryResults]

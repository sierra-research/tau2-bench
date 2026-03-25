"""
Tau2-Bench Leaderboard Access Module.

This module provides easy programmatic access to the tau-bench leaderboard data.

There are two main use cases:

1. **Reading leaderboard data** (use `Leaderboard` class):
   Access and query the published leaderboard results from the web.

2. **Preparing submissions** (use `prepare_submission` module):
   Create new submission files from trajectory data.

Example usage for reading leaderboard data:

    from tau2.scripts.leaderboard import Leaderboard

    # Load leaderboard from default location
    lb = Leaderboard.load()

    # Get a summary of the leaderboard
    print(lb.summary())

    # Get top 5 models by pass^1 in retail domain
    top_retail = lb.get_top_models(domain="retail", metric="pass_1", limit=5)
    for entry in top_retail:
        print(f"{entry.rank}. {entry.submission.model_name}: {entry.score:.1f}%")

    # Get top models overall (average across all domains)
    top_overall = lb.get_top_models_overall(metric="pass_1", limit=5)

    # Get all submissions as a pandas DataFrame
    df = lb.to_dataframe()

    # Get a specific submission by ID
    gpt5 = lb.get_submission("gpt-5_sierra_2025-08-09")
    print(gpt5.model_name, gpt5.results.retail.pass_1)

    # Filter by organization
    anthropic_models = lb.filter_by_organization("Anthropic")

    # Get submissions with trajectory data available
    with_trajectories = lb.filter_by_trajectories_available(available=True)

Example usage for preparing submissions:

    from tau2.scripts.leaderboard.prepare_submission import prepare_submission

    # Prepare a submission from trajectory files
    prepare_submission(
        input_paths=["./my_trajectories/"],
        output_dir="./my_submission/"
    )
"""

from .data_models import (
    DOMAINS,
    METRICS,
    SUBMISSION_FILE_NAME,
    TRAJECTORY_FILES_DIR_NAME,
    ContactInfo,
    DomainResults,
    LeaderboardEntry,
    LeaderboardManifest,
    Methodology,
    ReferenceInfo,
    Results,
    Submission,
    VerificationInfo,
    VoiceConfig,
)
from .leaderboard import Leaderboard

# Also export SubmissionData for trajectory verification workflows
from .submission import SubmissionData

__all__ = [
    # Main Leaderboard class
    "Leaderboard",
    # Data models for reading leaderboard
    "Submission",
    "ContactInfo",
    "DomainResults",
    "Results",
    "Methodology",
    "VerificationInfo",
    "ReferenceInfo",
    "VoiceConfig",
    "LeaderboardManifest",
    "LeaderboardEntry",
    # For submission preparation
    "SubmissionData",
    # Constants
    "DOMAINS",
    "METRICS",
    "SUBMISSION_FILE_NAME",
    "TRAJECTORY_FILES_DIR_NAME",
]

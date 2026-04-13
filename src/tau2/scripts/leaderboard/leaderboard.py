"""Leaderboard access module for tau-bench.

Provides easy programmatic access to the tau-bench leaderboard data.

Usage:
    from tau2.scripts.leaderboard import Leaderboard

    # Load leaderboard from default location
    lb = Leaderboard.load()

    # Get top models by domain
    top_retail = lb.get_top_models(domain="retail", metric="pass_1", limit=5)

    # Get all submissions
    all_subs = lb.submissions

    # Get a specific submission by ID
    gpt5 = lb.get_submission("gpt-5_sierra_2025-08-09")
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .submission import (
    DOMAINS,
    MANIFEST_FILE_NAME,
    METRICS,
    SUBMISSION_FILE_NAME,
    LeaderboardEntry,
    LeaderboardManifest,
    Submission,
)

# Default path to submissions relative to the repo root
DEFAULT_SUBMISSIONS_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "web"
    / "leaderboard"
    / "public"
    / "submissions"
)


MetricType = Literal["pass_1", "pass_2", "pass_3", "pass_4", "cost"]
DomainType = Literal["retail", "airline", "telecom", "banking_knowledge"]


class Leaderboard(BaseModel):
    """Main class for accessing and querying leaderboard data.

    Attributes:
        submissions: Dictionary mapping submission IDs to Submission objects
        submissions_path: Path to the submissions directory
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    submissions: dict[str, Submission] = Field(default_factory=dict)
    submissions_path: Path = Field(default=DEFAULT_SUBMISSIONS_PATH)

    @classmethod
    def load(cls, submissions_path: Optional[Path | str] = None) -> "Leaderboard":
        """Load the leaderboard from the submissions directory.

        Args:
            submissions_path: Path to the submissions directory. If None, uses
                the default path in the tau-bench repository.

        Returns:
            Leaderboard instance with all submissions loaded.
        """
        if submissions_path is None:
            submissions_path = DEFAULT_SUBMISSIONS_PATH
        submissions_path = Path(submissions_path)

        if not submissions_path.exists():
            raise FileNotFoundError(
                f"Submissions directory not found: {submissions_path}"
            )

        submissions = {}

        # Try to load from manifest first
        manifest_path = submissions_path / MANIFEST_FILE_NAME
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest = LeaderboardManifest.model_validate_json(f.read())
            submission_ids = manifest.submissions + manifest.voice_submissions
        else:
            # Fall back to scanning directories
            submission_ids = [
                d.name
                for d in submissions_path.iterdir()
                if d.is_dir() and not d.name.startswith((".", "A_EXAMPLE"))
            ]

        # Load each submission
        for submission_id in submission_ids:
            submission_file = submissions_path / submission_id / SUBMISSION_FILE_NAME
            if submission_file.exists():
                try:
                    submission = Submission.load(submission_file)
                    submissions[submission_id] = submission
                except Exception as e:
                    print(f"Warning: Failed to load submission {submission_id}: {e}")

        return cls(submissions=submissions, submissions_path=submissions_path)

    def get_submission(self, submission_id: str) -> Optional[Submission]:
        """Get a submission by its ID.

        Args:
            submission_id: The submission folder name (e.g., "gpt-5_sierra_2025-08-09")

        Returns:
            The Submission object, or None if not found.
        """
        return self.submissions.get(submission_id)

    def list_submissions(self) -> list[str]:
        """Get a list of all submission IDs.

        Returns:
            List of submission IDs (folder names).
        """
        return list(self.submissions.keys())

    def list_models(self) -> list[str]:
        """Get a list of all model names.

        Returns:
            List of unique model names.
        """
        return list(set(s.model_name for s in self.submissions.values()))

    def list_organizations(self) -> list[str]:
        """Get a list of all model organizations.

        Returns:
            List of unique organization names.
        """
        return list(set(s.model_organization for s in self.submissions.values()))

    def get_top_models(
        self,
        domain: DomainType,
        metric: MetricType = "pass_1",
        limit: Optional[int] = None,
        ascending: bool = False,
    ) -> list[LeaderboardEntry]:
        """Get the top models for a specific domain and metric.

        Args:
            domain: The domain to rank by ("retail", "airline", "telecom", or "banking_knowledge")
            metric: The metric to rank by ("pass_1", "pass_2", "pass_3", "pass_4", or "cost")
            limit: Maximum number of results to return. If None, returns all.
            ascending: If True, sort in ascending order (useful for cost).
                       If False (default), sort in descending order (for pass_k).

        Returns:
            List of LeaderboardEntry objects sorted by the specified metric.
        """
        if domain not in DOMAINS:
            raise ValueError(f"Invalid domain: {domain}. Must be one of {DOMAINS}")
        if metric not in METRICS:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {METRICS}")

        entries = []
        for submission_id, submission in self.submissions.items():
            domain_results = submission.results.get_domain(domain)
            if domain_results is None:
                continue

            score = getattr(domain_results, metric)
            if score is None:
                continue

            entries.append(LeaderboardEntry(submission=submission, score=score))

        # Sort by score
        entries.sort(key=lambda e: e.score or 0, reverse=not ascending)

        # Assign ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        if limit is not None:
            entries = entries[:limit]

        return entries

    def get_top_models_overall(
        self,
        metric: MetricType = "pass_1",
        limit: Optional[int] = None,
        ascending: bool = False,
    ) -> list[LeaderboardEntry]:
        """Get the top models by average score across all domains.

        Args:
            metric: The metric to rank by ("pass_1", "pass_2", "pass_3", "pass_4", or "cost")
            limit: Maximum number of results to return. If None, returns all.
            ascending: If True, sort in ascending order (useful for cost).

        Returns:
            List of LeaderboardEntry objects sorted by average score across domains.
        """
        if metric not in METRICS:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {METRICS}")

        entries = []
        for submission_id, submission in self.submissions.items():
            scores = []
            for domain in DOMAINS:
                domain_results = submission.results.get_domain(domain)
                if domain_results is not None:
                    score = getattr(domain_results, metric)
                    if score is not None:
                        scores.append(score)

            if not scores:
                continue

            avg_score = sum(scores) / len(scores)
            entries.append(LeaderboardEntry(submission=submission, score=avg_score))

        # Sort by score
        entries.sort(key=lambda e: e.score or 0, reverse=not ascending)

        # Assign ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        if limit is not None:
            entries = entries[:limit]

        return entries

    def filter_by_organization(self, organization: str) -> list[Submission]:
        """Get all submissions from a specific organization.

        Args:
            organization: The organization name to filter by (case-insensitive).

        Returns:
            List of Submission objects from that organization.
        """
        org_lower = organization.lower()
        return [
            s
            for s in self.submissions.values()
            if s.model_organization.lower() == org_lower
        ]

    def filter_by_trajectories_available(
        self, available: bool = True
    ) -> list[Submission]:
        """Get submissions that have trajectories available (or not).

        Args:
            available: If True, return submissions with trajectories.
                      If False, return submissions without trajectories.

        Returns:
            List of Submission objects.
        """
        return [
            s
            for s in self.submissions.values()
            if s.trajectories_available == available
        ]

    def get_domain_leaderboard(
        self,
        domain: DomainType,
    ) -> dict[str, dict]:
        """Get a summary leaderboard for a specific domain.

        Args:
            domain: The domain ("retail", "airline", "telecom", or "banking_knowledge")

        Returns:
            Dictionary with model names as keys and their metrics as values.
        """
        if domain not in DOMAINS:
            raise ValueError(f"Invalid domain: {domain}. Must be one of {DOMAINS}")

        leaderboard = {}
        for submission_id, submission in self.submissions.items():
            domain_results = submission.results.get_domain(domain)
            if domain_results is None:
                continue

            leaderboard[submission.model_name] = {
                "submission_id": submission_id,
                "organization": submission.model_organization,
                "pass_1": domain_results.pass_1,
                "pass_2": domain_results.pass_2,
                "pass_3": domain_results.pass_3,
                "pass_4": domain_results.pass_4,
                "cost": domain_results.cost,
            }

        return leaderboard

    def to_dataframe(self, domain: Optional[DomainType] = None):
        """Convert leaderboard data to a pandas DataFrame.

        Args:
            domain: If specified, only include results for that domain.
                   If None, creates a multi-domain view.

        Returns:
            pandas DataFrame with leaderboard data.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). Install it with: pip install pandas"
            )

        rows = []
        for submission_id, submission in self.submissions.items():
            if domain is not None:
                # Single domain view
                domain_results = submission.results.get_domain(domain)
                if domain_results is None:
                    continue
                rows.append(
                    {
                        "submission_id": submission_id,
                        "model_name": submission.model_name,
                        "organization": submission.model_organization,
                        "submitting_org": submission.submitting_organization,
                        "submission_date": submission.submission_date,
                        "domain": domain,
                        "pass_1": domain_results.pass_1,
                        "pass_2": domain_results.pass_2,
                        "pass_3": domain_results.pass_3,
                        "pass_4": domain_results.pass_4,
                        "cost": domain_results.cost,
                        "trajectories_available": submission.trajectories_available,
                    }
                )
            else:
                # Multi-domain view: one row per domain
                for d in DOMAINS:
                    domain_results = submission.results.get_domain(d)
                    if domain_results is None:
                        continue
                    rows.append(
                        {
                            "submission_id": submission_id,
                            "model_name": submission.model_name,
                            "organization": submission.model_organization,
                            "submitting_org": submission.submitting_organization,
                            "submission_date": submission.submission_date,
                            "domain": d,
                            "pass_1": domain_results.pass_1,
                            "pass_2": domain_results.pass_2,
                            "pass_3": domain_results.pass_3,
                            "pass_4": domain_results.pass_4,
                            "cost": domain_results.cost,
                            "trajectories_available": submission.trajectories_available,
                        }
                    )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["domain", "pass_1"], ascending=[True, False])
        return df

    def summary(self) -> str:
        """Get a human-readable summary of the leaderboard.

        Returns:
            String with summary statistics.
        """
        lines = [
            "=== Tau2-Bench Leaderboard Summary ===",
            f"Total submissions: {len(self.submissions)}",
            f"Unique models: {len(self.list_models())}",
            f"Organizations: {len(self.list_organizations())}",
            "",
        ]

        for domain in DOMAINS:
            top_entries = self.get_top_models(domain=domain, metric="pass_1", limit=3)
            if top_entries:
                lines.append(f"Top 3 in {domain.capitalize()} (Pass^1):")
                for entry in top_entries:
                    lines.append(
                        f"  {entry.rank}. {entry.submission.model_name} "
                        f"({entry.submission.model_organization}): {entry.score:.1f}%"
                    )
                lines.append("")

        return "\n".join(lines)

    def display(
        self,
        domain: Optional[DomainType] = None,
        metric: MetricType = "pass_1",
        limit: Optional[int] = None,
    ) -> None:
        """Display the leaderboard as a rich table in the console.

        Args:
            domain: If specified, show leaderboard for that domain only.
                   If None, shows overall leaderboard (average across domains).
            metric: The metric to rank and display by. Default is "pass_1".
            limit: Maximum number of entries to show. If None, shows all.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()

        if domain is not None:
            # Single domain leaderboard
            entries = self.get_top_models(domain=domain, metric=metric, limit=limit)
            title = f"Tau2-Bench Leaderboard - {domain.capitalize()} ({metric})"
        else:
            # Overall leaderboard (average across domains)
            entries = self.get_top_models_overall(metric=metric, limit=limit)
            title = f"Tau2-Bench Leaderboard - Overall ({metric})"

        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Model", style="bold")
        table.add_column("Organization")
        table.add_column("Modality", justify="center")
        table.add_column(metric.replace("_", "^"), justify="right", style="green")

        if domain is None:
            for d in DOMAINS:
                table.add_column(d.replace("_", " ").title(), justify="right")

        for entry in entries:
            sub = entry.submission
            score_str = f"{entry.score:.1f}%" if entry.score is not None else "-"
            modality = getattr(sub, "modality", "text")

            if domain is None:
                domain_scores = []
                for d in DOMAINS:
                    d_results = sub.results.get_domain(d)
                    d_score = (
                        f"{getattr(d_results, metric):.1f}%"
                        if d_results and getattr(d_results, metric) is not None
                        else "-"
                    )
                    domain_scores.append(d_score)

                table.add_row(
                    str(entry.rank),
                    sub.model_name,
                    sub.model_organization,
                    modality,
                    score_str,
                    *domain_scores,
                )
            else:
                table.add_row(
                    str(entry.rank),
                    sub.model_name,
                    sub.model_organization,
                    modality,
                    score_str,
                )

        console.print()
        console.print(table)
        console.print()
        console.print(f"Total submissions: {len(self.submissions)}", style="dim")

    def __len__(self) -> int:
        """Return the number of submissions."""
        return len(self.submissions)

    def __repr__(self) -> str:
        return f"Leaderboard(submissions={len(self.submissions)}, path={self.submissions_path})"


def show_leaderboard(
    domain: Optional[str] = None,
    metric: str = "pass_1",
    limit: Optional[int] = None,
) -> None:
    """Show the leaderboard in the console.

    This is a convenience function for CLI usage.

    Args:
        domain: If specified, show leaderboard for that domain only.
               If None, shows overall leaderboard.
        metric: The metric to rank by. Default is "pass_1".
        limit: Maximum number of entries to show.
    """
    lb = Leaderboard.load()
    lb.display(domain=domain, metric=metric, limit=limit)

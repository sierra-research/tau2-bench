from rich.console import Console

from tau2.data_model.simulation import Results


def private_checks(results: Results, console: Console) -> tuple[bool, str]:
    """Those are not available in the public version of the leaderboard."""
    ...

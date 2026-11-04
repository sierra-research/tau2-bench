"""Load the packaged judge selection independently of execution."""

from pathlib import Path

from tau2.data_model.judges import JudgeCatalog


def load_catalog(path: Path | None = None) -> JudgeCatalog:
    """Read only the compact packaged selection, never annotation archives."""
    selected = path or Path(__file__).with_name("production_judges.json")
    return JudgeCatalog.model_validate_json(selected.read_text())

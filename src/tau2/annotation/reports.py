# Copyright Sierra
"""Human-readable calibration reports: per-factor gates + per-language agreement.

- ``format_precision_report`` / ``format_cold_report`` — the per-(language,
  factor) tables with the LIVE/shadow gate on the precision bar.
- ``agreement_report`` — one language's persisted agreement summaries
  (translation, communicate judge) plus the parity-probe verdict sink written
  by ``factory.parity``.
"""

import csv
import json
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import ValidationError

from tau2.annotation.communicate_judge import JUDGE_AGREEMENT_SUFFIX
from tau2.annotation.evaluation import gate
from tau2.annotation.metrics import (
    ColdFactorMetrics,
    PrecisionFactorMetrics,
    summarize_agreement,
)
from tau2.annotation.translation import TRANSLATION_AGREEMENT_SUFFIX, AgreementFile
from tau2.multilingual.factory.paths import PARITY_PROBE_FILENAME, calibration_dir


def _fmt(x: Optional[float]) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def _write_flat_csv(
    path: Path, rows: list[dict], fieldnames: Optional[list[str]] = None
) -> None:
    """Write flat dict rows to a CSV (header from the first row unless given)."""
    if not rows:
        return
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {len(rows)} metric rows -> {path}")


def format_precision_report(
    metrics: dict[tuple[str, str], PrecisionFactorMetrics], precision_bar: float
) -> str:
    lines = [
        f"{'language':10} {'factor':22} {'precision':>9} {'n':>4} {'unsure':>6}  gate"
    ]
    for (lang, factor), m in sorted(metrics.items()):
        lines.append(
            f"{lang:10} {factor:22} {_fmt(m.precision):>9} "
            f"{m.n_adjudicated:>4} {m.unsure:>6}  {gate(m.precision, precision_bar)}"
        )
    return "\n".join(lines)


def format_cold_report(
    metrics: dict[tuple[str, str], ColdFactorMetrics],
    precision_bar: float,
    human_kappa: Optional[dict[tuple[str, str], Optional[float]]] = None,
) -> str:
    hdr = (
        f"{'language':10} {'factor':22} {'prec':>6} {'recall':>6} "
        f"{'f1':>6} {'kappa':>6} {'n':>4}"
    )
    if human_kappa is not None:
        hdr += f" {'hkappa':>6}"
    hdr += "  gate"
    lines = [hdr]
    for (lang, factor), m in sorted(metrics.items()):
        row = (
            f"{lang:10} {factor:22} {_fmt(m.precision):>6} "
            f"{_fmt(m.recall):>6} {_fmt(m.f1):>6} {_fmt(m.kappa):>6} "
            f"{m.n_opportunity:>4}"
        )
        if human_kappa is not None:
            row += f" {_fmt(human_kappa.get((lang, factor))):>6}"
        row += f"  {gate(m.precision, precision_bar)}"
        lines.append(row)
        # Surface the off-the-2x2 buckets so they aren't silently dropped.
        diag = [
            (m.counts.judge_unscored, "judge error/missing"),
            (m.counts.noopp_opp, "judge no-opp vs human opp"),
            (m.counts.unsure, "human unsure"),
            (m.counts.unlabeled, "unlabeled"),
        ]
        note = "; ".join(f"{n} {label}" for n, label in diag if n)
        if note:
            lines.append(f"{'':33} ({note})")
    return "\n".join(lines)


def write_metrics_csv(
    metrics: dict[tuple[str, str], ColdFactorMetrics | PrecisionFactorMetrics],
    path: Path,
) -> None:
    """Flatten per-factor metrics to a CSV (counts inlined for cold metrics)."""
    rows = []
    for (lang, factor), m in sorted(metrics.items()):
        dump = m.model_dump()
        counts = dump.pop("counts", None)
        if counts:
            dump.update(counts)
        rows.append({"language": lang, "factor_id": factor, **dump})
    _write_flat_csv(path, rows)


# ---------------------------------------------------------------------------
# Per-language agreement report
# ---------------------------------------------------------------------------


def summarize_parity_probe(records: list[dict]) -> dict:
    """Summary of the parity-probe verdict sink (written by ``factory.parity``).

    Each record is one ``analyze`` run. Surfaces how often the parity probe
    flagged a behavioral gap — the "parity caught K bugs the bilingual judge
    passed" comparison the probe exists to enable.
    """
    if not records:
        return {"n_runs": 0, "latest": None}
    latest = records[-1]
    flagged = latest.get("flagged", []) or []
    return {
        "n_runs": len(records),
        "latest": {
            "domain": latest.get("domain"),
            "aggregate_gap": latest.get("aggregate_gap"),
            "total_tasks": latest.get("total_tasks"),
            "n_flagged": len(flagged),
            "flagged": flagged,
        },
    }


def agreement_report(lang: str) -> dict:
    """Summaries of all persisted agreement files for one language.

    Agreement files are per-(language, domain); each report section maps
    ``domain -> summary`` over every domain found (None when no domain has
    been ingested yet).
    """
    report: dict = {"language": lang}
    for key, suffix in (
        ("translation", TRANSLATION_AGREEMENT_SUFFIX),
        ("communicate_judge", JUDGE_AGREEMENT_SUFFIX),
    ):
        per_domain: dict[str, dict] = {}
        for path in sorted(calibration_dir(lang).glob(f"*{suffix}")):
            try:
                payload = AgreementFile.model_validate_json(path.read_text())
            except ValidationError as exc:
                raise ValueError(
                    f"corrupt agreement file {path} — re-run the ingest that "
                    f"wrote it: {exc}"
                ) from exc
            domain = payload.domain or path.name.removesuffix(suffix)
            per_domain[domain] = summarize_agreement(payload.records).model_dump(
                mode="json"
            )
        report[key] = per_domain or None

    parity_path = calibration_dir(lang) / PARITY_PROBE_FILENAME
    report["parity_probe"] = None
    if parity_path.exists():
        try:
            records = json.loads(parity_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt parity-probe sink {parity_path}: {exc}") from exc
        if not isinstance(records, list):
            raise ValueError(
                f"parity-probe sink {parity_path} must be a JSON list of "
                f"analyze-run records, got {type(records).__name__}"
            )
        report["parity_probe"] = summarize_parity_probe(records)
    return report

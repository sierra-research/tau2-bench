# Copyright Sierra
"""Communicate-judge calibration: native verdicts vs the communicate_info judge.

Export samples judged criteria from a run's PERSISTED results
(``reward_info.communicate_checks``) — read from disk — into a
``CommunicateJudgeRow`` sheet family with the judge verdict prefilled and the
native verdict blank. Ingest reuses the same agreement machinery as the
translation calibration (``pipeline_ok`` carries the judge verdict; ``field``
carries the criterion text).
"""

import random
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.annotation.artifacts import (
    LoadedArtifact,
    payload_from_rows,
    write_sheet_family,
)
from tau2.annotation.loading import iter_loaded_sims
from tau2.annotation.metrics import AgreementRecord
from tau2.annotation.models import CommunicateJudgeRow, YesNo
from tau2.annotation.translation import persist_agreement
from tau2.config import (
    DEFAULT_ANNOTATION_AGENT_EXCERPT_MAX_CHARS,
    DEFAULT_ANNOTATION_JUDGE_SAMPLE_N,
)
from tau2.data_model.message import AssistantMessage
from tau2.evaluator.evaluator_communicate import LLM_JUDGE_JUSTIFICATION_PREFIX
from tau2.multilingual.factory.paths import calibration_dir

TRUNCATION_NOTE = "\n[... transcript truncated for this sheet]"

# Persisted agreement files are keyed by domain: <domain> + this suffix.
JUDGE_AGREEMENT_SUFFIX = "_communicate_judge_agreement.json"


def _agent_excerpt(
    sim, max_chars: int = DEFAULT_ANNOTATION_AGENT_EXCERPT_MAX_CHARS
) -> str:
    """The full agent-side transcript, capped; truncation is noted IN the cell
    so the annotator knows the criterion-relevant turn may be missing."""
    full = " | ".join(
        message.content
        for message in sim.messages or []
        if isinstance(message, AssistantMessage) and message.has_text_content()
    )
    if len(full) <= max_chars:
        return full
    return full[:max_chars] + TRUNCATION_NOTE


def export_communicate_judge_sample(
    run_dir: Path,
    n: int = DEFAULT_ANNOTATION_JUDGE_SAMPLE_N,
    seed: int = 0,
    lang: Optional[str] = None,
    out_stem: Optional[Path] = None,
) -> Path:
    """Sample judged communicate_info verdicts from persisted results.

    One row per sampled (simulation, criterion), judge verdict prefilled,
    ``native_verdict`` blank for the annotator. Returns the manifest path.
    """
    candidates: list[CommunicateJudgeRow] = []
    languages: set[str] = set()
    domains: set[str] = set()
    for loaded in iter_loaded_sims([run_dir]):
        sim = loaded.sim
        domains.add(loaded.domain)
        env = sim.speech_environment
        sim_lang = env.language if env is not None else None
        if sim_lang:
            languages.add(sim_lang)
        checks = sim.reward_info.communicate_checks if sim.reward_info else None
        if not checks:
            continue
        # When a language is resolved/passed, restrict candidates to sims whose
        # speech_environment language matches — a mixed-language run must not
        # pollute one language's calibration sheet with another's rows (they
        # are judged with different criteria and context).
        if lang is not None and sim_lang != lang:
            continue
        excerpt = _agent_excerpt(sim)
        for check in checks:
            justification = check.justification or ""
            candidates.append(
                CommunicateJudgeRow(
                    criterion=check.info,
                    agent_turn_excerpt=excerpt,
                    judge_verdict=YesNo.YES if check.met else YesNo.NO,
                    comments=justification,
                    llm_judged=(
                        YesNo.YES
                        if justification.startswith(LLM_JUDGE_JUSTIFICATION_PREFIX)
                        else YesNo.NO
                    ),
                    task_id=str(sim.task_id),
                    sim_id=sim.id,
                )
            )

    if lang is None:
        if len(languages) != 1:
            raise ValueError(
                f"Cannot infer the run language from {run_dir} "
                f"(found {sorted(languages)}) — pass lang explicitly."
            )
        lang = next(iter(languages))

    rng = random.Random(seed)
    sample = candidates if len(candidates) <= n else rng.sample(candidates, n)
    logger.info(f"sampled {len(sample)}/{len(candidates)} judged criteria")

    out_stem = out_stem or calibration_dir(lang) / "communicate_judge_sample"
    return write_sheet_family(
        out_stem,
        kind="communicate_judge",
        payloads={"sheet": payload_from_rows(CommunicateJudgeRow, sample)},
        provenance={
            # The persisted run schema (CommunicateCheck / Info) records no
            # communicate-judge model, so the honest provenance is "unknown" —
            # stamping the CURRENT config default would misattribute verdicts
            # produced by an older run to today's model.
            "judge_model": "unknown",
            "run_dir": str(run_dir),
            "sample_n": len(sample),
            "sample_seed": seed,
        },
        language=lang,
        domain=next(iter(domains)) if len(domains) == 1 else None,
    )


def ingest_communicate_judge(artifact: LoadedArtifact, source: Path) -> Path:
    """Compare native verdicts vs the communicate judge; persist agreement."""
    lang = artifact.manifest.language
    if not lang:
        raise ValueError(f"manifest {artifact.manifest.batch_name} carries no language")
    domain = artifact.manifest.domain
    if not domain:
        raise ValueError(f"manifest {artifact.manifest.batch_name} carries no domain")
    records: list[AgreementRecord] = []
    for row in artifact.rows("sheet"):
        if row.native_verdict is None:
            continue  # not annotated
        records.append(
            AgreementRecord(
                task_id=row.task_id,
                sim_id=row.sim_id,
                field=row.criterion,
                pipeline_ok=(
                    None
                    if row.judge_verdict is None
                    else row.judge_verdict is YesNo.YES
                ),
                native_ok=row.native_verdict is YesNo.YES,
                comments=row.comments,
            )
        )
    return persist_agreement(
        lang, domain, JUDGE_AGREEMENT_SUFFIX, "communicate_judge", source, records
    )

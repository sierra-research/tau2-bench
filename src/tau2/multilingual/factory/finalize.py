# Copyright Sierra
"""Finalize: promote a guardrail-clean draft into ``data/tau2/multilingual/``."""

from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.drafting import (
    GUIDELINES_DRAFT_FILENAME,
    GUIDELINES_TEXT_DRAFT_FILENAME,
    delivery_scaffold_comment,
)
from tau2.multilingual.factory.guardrails import (
    GuardrailReport,
    validate_final_pack,
    validate_pack_draft,
)
from tau2.multilingual.factory.paths import (
    default_text_guidelines_filename,
    default_voice_guidelines_filename,
)
from tau2.multilingual.factory.state import (
    PACK_DRAFT_FILENAME,
    FactoryProject,
    FactoryStage,
    StageStatus,
    factory_project_dir,
)
from tau2.multilingual.loader import normalize_pack_data
from tau2.multilingual.localize_lib import multilingual_data_dir

# =============================================================================
# Finalize: promote the draft into data/tau2/multilingual/<lang>/
# =============================================================================


class FinalizeOutcome(BaseModel):
    """What ``finalize`` produced."""

    language: str
    pack_path: Path
    guidelines_path: Path
    text_guidelines_path: Path
    report: GuardrailReport = Field(
        description="The validate_final_pack report on the promoted pack"
    )
    checklist_lines: list[str] = Field(
        default_factory=list,
        description="The human-bottleneck checklist, rendered for printing",
    )


def render_checklist(project: FactoryProject) -> list[str]:
    """The human-bottleneck checklist as printable lines."""
    lines = ["Human-bottleneck checklist (automation stops here):"]
    for name, field in type(project.checklist).model_fields.items():
        done = getattr(project.checklist, name)
        box = "[x]" if done else "[ ]"
        lines.append(f"  {box} {name}: {field.description}")
    return lines


def _carry_voice_pins_forward(
    lang: str,
    project: FactoryProject,
    dest_pack_path: Path,
    draft_data: dict,
) -> None:
    """Preserve ``generate-assets`` voice pins across a ``--force`` re-finalize.

    ``tau2 factory generate-assets`` pins designed ElevenLabs ``voice_id``s into
    the FINAL pack.yaml — the draft never has them. A re-finalize would silently
    destroy those paid, human-checked pins, so they are merged forward whenever
    the persona ids still match. Pins that no longer map to a draft persona are
    dropped LOUDLY and the ``voice_ids`` checklist item is reset so the human
    bottleneck re-engages.
    """
    if not dest_pack_path.exists():
        return
    existing = yaml.safe_load(dest_pack_path.read_text()) or {}
    # The loader's ONE normalization keys list-form personas by persona_id, so
    # the merge below sees the same shape the runtime loads.
    existing = normalize_pack_data(existing, dest_pack_path.parent)
    existing_personas = existing.get("personas") or {}
    pinned = {
        persona_id: persona["voice_id"]
        for persona_id, persona in existing_personas.items()
        if isinstance(persona, dict) and persona.get("voice_id")
    }
    if not pinned:
        return

    draft_personas = draft_data.get("personas") or {}
    carried: list[str] = []
    unmapped: list[str] = []
    for persona_id, voice_id in pinned.items():
        draft_persona = draft_personas.get(persona_id)
        if isinstance(draft_persona, dict):
            draft_persona.setdefault("voice_id", voice_id)
            carried.append(persona_id)
        else:
            unmapped.append(persona_id)

    if carried:
        logger.info(
            f"finalize: carried pinned voice_id(s) forward from the existing "
            f"'{lang}' pack for persona(s) {carried}"
        )
    if unmapped:
        logger.warning(
            f"finalize --force: existing '{lang}' pack pins voice_id(s) for "
            f"persona(s) {unmapped} that do not exist in the new draft "
            f"(draft personas: {sorted(draft_personas)}). Those pins are LOST — "
            f"re-run `tau2 factory generate-assets --lang {lang}` and re-verify "
            "voices. Resetting checklist item 'voice_ids'."
        )
        project.checklist.voice_ids = False


def finalize(lang: str, *, force: bool = False) -> FinalizeOutcome:
    """Promote a guardrail-clean draft to ``data/tau2/multilingual/<lang>/``.

    Refuses when the draft fails guardrails, and when a final ``pack.yaml``
    already exists unless ``force``. The guidelines file is copied FIRST and
    ``pack.yaml`` last, so the loader glob can never discover a pack whose
    declared guidelines file is missing.
    """
    workspace = factory_project_dir(lang)
    # A generated rubric is never promoted before the explicit human-review
    # apply step. Languages that have not started the nativeness stage retain
    # the existing finalize behavior; once a packet exists, it is a real pause.
    from tau2.multilingual.factory.nativeness_drafting import (
        ReviewStatus,
        load_nativeness_review,
    )

    nativeness_review = load_nativeness_review(lang)
    if nativeness_review is not None:
        draft_path = workspace / PACK_DRAFT_FILENAME
        if draft_path.exists():
            draft_preview = yaml.safe_load(draft_path.read_text())
            project = FactoryProject.load_or_create(lang)
            review_applied = (
                nativeness_review.review_status is ReviewStatus.APPROVED
                and project.stages[FactoryStage.NATIVENESS] == StageStatus.DONE
            )
            draft_nativeness = (draft_preview or {}).get("nativeness")
            if review_applied and draft_nativeness is not None:
                from tau2.multilingual.schema import NativenessPackConfig

                try:
                    review_applied = (
                        NativenessPackConfig.model_validate(draft_nativeness)
                        == nativeness_review.to_pack_config()
                    )
                except ValueError:
                    review_applied = False
            if not review_applied:
                raise FactoryDraftError(
                    f"refusing to finalize '{lang}': nativeness review is "
                    f"'{nativeness_review.review_status}' and the approved packet "
                    "has not been applied unchanged through the factory stage. "
                    "Review the packet, set review_status: approved, then run "
                    f"`tau2 factory apply-nativeness --lang {lang}`."
                )
    report = validate_pack_draft(workspace)
    if not report.ok:
        problems = "\n".join(f"  - {problem}" for problem in report.problems)
        raise FactoryDraftError(
            f"refusing to finalize '{lang}': the draft fails "
            f"{len(report.problems)} guardrail(s):\n{problems}\n"
            f"Fix {workspace / PACK_DRAFT_FILENAME} and re-run "
            f"`tau2 factory validate --draft --lang {lang}`."
        )

    dest_dir = multilingual_data_dir() / lang
    dest_pack_path = dest_dir / "pack.yaml"
    if dest_pack_path.exists() and not force:
        raise FactoryDraftError(
            f"refusing to overwrite existing pack {dest_pack_path} "
            "(pass --force to replace it)"
        )

    project = FactoryProject.load_or_create(lang)
    guidelines_filename = (
        project.guidelines_final_filename or default_voice_guidelines_filename(lang)
    )
    text_guidelines_filename = (
        project.guidelines_text_final_filename or default_text_guidelines_filename(lang)
    )

    draft_data = yaml.safe_load((workspace / PACK_DRAFT_FILENAME).read_text())
    guidelines_text = (workspace / GUIDELINES_DRAFT_FILENAME).read_text()
    draft_data["guidelines_voice_path"] = guidelines_filename

    # Finalize-path invariant: a draft must carry BOTH guidelines files (Call C
    # authors the text guidelines in every draft). The pack-schema field stays
    # Optional until every shipped pack is regenerated, but no NEW pack may be
    # promoted without text guidelines.
    text_draft_path = workspace / GUIDELINES_TEXT_DRAFT_FILENAME
    if not text_draft_path.exists():
        raise FactoryDraftError(
            f"refusing to finalize '{lang}': the draft has no text guidelines "
            f"({text_draft_path} is missing). Re-run `tau2 factory draft --lang "
            f"{lang} --form ...` (Call C authors them in every draft)."
        )
    text_guidelines_body = text_draft_path.read_text()
    draft_data["guidelines_text_path"] = text_guidelines_filename

    _carry_voice_pins_forward(lang, project, dest_pack_path, draft_data)

    # Guidelines files first, pack.yaml last (discoverability ordering: the
    # loader glob must never find a pack.yaml whose declared guidelines are
    # missing).
    dest_dir.mkdir(parents=True, exist_ok=True)
    # The declared filenames are pack-RELATIVE and namespaced under
    # ``guidelines/``, so create the parent.
    guidelines_path = dest_dir / guidelines_filename
    guidelines_path.parent.mkdir(parents=True, exist_ok=True)
    guidelines_path.write_text(guidelines_text)
    text_guidelines_path = dest_dir / text_guidelines_filename
    text_guidelines_path.parent.mkdir(parents=True, exist_ok=True)
    text_guidelines_path.write_text(text_guidelines_body)
    pack_text = yaml.safe_dump(draft_data, sort_keys=False, allow_unicode=True)
    if "delivery" not in draft_data:
        # Same for the delivery scaffold (audio-layer judge factors).
        pack_text += "\n" + delivery_scaffold_comment()
    dest_pack_path.write_text(pack_text)
    logger.info(f"Finalized pack for '{lang}' at {dest_pack_path}")

    final_report = validate_final_pack(lang)
    project.mark_stage(FactoryStage.FINALIZE, success=final_report.ok)

    return FinalizeOutcome(
        language=lang,
        pack_path=dest_pack_path,
        guidelines_path=guidelines_path,
        text_guidelines_path=text_guidelines_path,
        report=final_report,
        checklist_lines=render_checklist(project),
    )

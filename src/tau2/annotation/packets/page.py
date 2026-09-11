# Copyright Sierra
"""Packet page assembly: jinja2 templates over typed view models.

Replaces the old ~1,000 lines of f-string HTML and the placeholder
string-replace mechanism. Autoescaping is ON; the only ``| safe`` content is
(a) the pre-escaped transcript rows from ``transcript.generate_tick_rows``,
(b) the static CSS/JS assets inlined at render time, (c) the embedded
guidelines documents, and (d) the ``PACKET_CONFIG`` JSON block.

Every page carries ONE injected ``<script>window.PACKET_CONFIG = {...}</script>``
block (batch identity, form, sim/task/trial ids, model-generated
``csv_headers``, rubric dims, judge error markers) — the page JS reads all of
its configuration from there.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

from tau2.annotation.models import BreakingPoint, ErrorSource, ErrorType
from tau2.annotation.packets.forms import (
    CALLER_EXPERIENCE_ANCHORS,
    EXPERIENCE_ELABORATE_FACTORS,
    FORM_LABELS,
    FORM_SPECS,
    ExperienceFactor,
    FormType,
    csv_headers,
    dims_config,
    experience_config,
    guidelines_anchors,
    guidelines_html,
    load_rubric,
)
from tau2.annotation.packets.transcript import (
    generate_message_rows,
    generate_tick_rows,
)
from tau2.data_model.message import ToolCall
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import StructuredUserInstructions, Task
from tau2.utils.tools import to_functional_format

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"

FORM_JS_FILES: dict[FormType, str] = {
    FormType.ERROR_ANALYSIS: "error_form.js",
    FormType.USER_REALISM: "realism_form.js",
    FormType.VOICE_REVIEW: "combined_form.js",
}

INDEX_JS_FILES: dict[FormType, str] = {
    FormType.ERROR_ANALYSIS: "index.js",
    FormType.USER_REALISM: "index.js",
    FormType.VOICE_REVIEW: "index.js",
}


@lru_cache(maxsize=None)
def _static(name: str) -> str:
    return (STATIC_DIR / name).read_text()


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        keep_trailing_newline=True,
        # A template referencing a context var the renderer forgot to pass is
        # a loud error, never a silently-empty page section.
        undefined=StrictUndefined,
    )


def _page_css(form: FormType) -> str:
    css = _static("annotation.css")
    spec = FORM_SPECS[form]
    # The experience section reuses the realism score-option card styles.
    if spec.show_audio or spec.show_experience:
        css += "\n" + _static("realism_form.css")
    return css


def _page_js(form: FormType) -> str:
    # form_core.js (rater/storage/CSV plumbing) loads before every form
    # script; the form files only carry their form-specific logic.
    parts = [_static("shared_ui.js"), _static("form_core.js")]
    parts.append(_static(FORM_JS_FILES[form]))
    parts.append(_static("guidelines.js"))
    return "\n".join(parts)


def _config_json(config: dict[str, Any]) -> str:
    """PACKET_CONFIG as a script-safe JSON literal (no '</' breakouts)."""
    return json.dumps(config, ensure_ascii=False).replace("</", "<\\/")


# ---------------------------------------------------------------------------
# View models (typed template context)
# ---------------------------------------------------------------------------


class BatchIdentity(BaseModel):
    """The batch identity every annotation packet carries."""

    batch_id: Annotated[str, Field(description="Content-derived batch id.")]
    batch_name: Annotated[str, Field(description="Human batch name.")]


class PacketIdentity(BatchIdentity):
    """A batch identity for one of the standard per-simulation forms."""

    form: FormType


class PageNavVM(BaseModel):
    """Static prev/next navigation, injected at build time from the packet's
    canonical page order (sim packets: the index's dir-name sort; preference
    packets: the pair manifest's pre-randomized order).

    Blind-safe by construction: hrefs are built from page dir names only,
    which carry nothing beyond the identities already shown on every page.
    """

    prev_href: Annotated[
        Optional[str],
        Field(description="Previous page (None on the first page)."),
    ] = None
    next_href: Annotated[
        str,
        Field(description="Next page; the index on the last page."),
    ]
    is_last: Annotated[
        bool, Field(description="True when next_href points at the index.")
    ] = False
    position: Annotated[int, Field(description="1-based packet position.")]
    total: Annotated[int, Field(description="Number of pages in the packet.")]


def build_nav_vms(dir_names: list[str]) -> list[PageNavVM]:
    """One PageNavVM per page, in packet order: the first page gets a
    disabled Previous, the last page's Next is 'Done — back to index'."""
    total = len(dir_names)
    return [
        PageNavVM(
            prev_href=(f"../{dir_names[i - 1]}/index.html" if i > 0 else None),
            next_href=(
                f"../{dir_names[i + 1]}/index.html"
                if i + 1 < total
                else "../index.html"
            ),
            is_last=(i + 1 == total),
            position=i + 1,
            total=total,
        )
        for i in range(total)
    ]


# The page unit sim-packet nav labels use ("Call 3 of 12"); sim.html.j2 passes
# the same literal to the macro — render_nav_fragment must stay in lockstep
# with it (guarded by the append-refresh equivalence test).
SIM_NAV_UNIT = "call"


def render_nav_fragment(nav: PageNavVM, unit: str = SIM_NAV_UNIT) -> str:
    """One nav bar alone, rendered through the SAME macro the page templates
    import — the builder's --append refresh of prior pages' bars uses this."""
    module = _env().get_template("_nav.html.j2").module
    return str(module.packet_nav(nav, unit)).strip()


class TaskInfoVM(BaseModel):
    """The task's user-scenario instructions for the collapsible section."""

    structured: bool
    reason_for_call: str = ""
    known_info: str = ""
    unknown_info: str = ""
    task_instructions: str = ""
    raw: str = ""


class MetaVM(BaseModel):
    """Simulation Info table + page title fields."""

    experiment: str
    domain: str
    task_id: str
    sim_id: str
    trial: int
    duration: str
    termination: str
    reward: str
    reward_class: str


class EvalVM(BaseModel):
    """The task's EXPECTED evaluation criteria (rendered even when unscored)."""

    reward_basis: list[str] = []
    actions: list[str] = []
    communicate: list[str] = []
    nl_assertions: list[str] = []
    env_assertions: list[str] = []


class CheckVM(BaseModel):
    """One met/unmet check line in the reward details."""

    ok: bool
    text: str
    justification: str = ""


class BreakdownVM(BaseModel):
    label: str
    score: Any
    cls: str


class RewardVM(BaseModel):
    """The collapsible Reward Details section."""

    breakdown: list[BreakdownVM] = []
    note: str = ""
    db_check: Optional[bool] = None
    action_checks: list[CheckVM] = []
    communicate_checks: list[CheckVM] = []
    nl_checks: list[CheckVM] = []
    env_checks: list[CheckVM] = []


class ReviewErrorVM(BaseModel):
    """One LLM-judge error item (with optional clickable tick range)."""

    id: int
    source: str
    source_class: str
    severity: str
    severity_class: str
    tags: list[str] = []
    tick_start: Optional[int] = None
    tick_end: Optional[int] = None
    reasoning: str = ""
    correct_behavior: str = ""

    @property
    def clickable(self) -> bool:
        return self.tick_start is not None and self.tick_end is not None


class ReviewVM(BaseModel):
    """The LLM Judge Review section."""

    summary: str
    agent_error: bool
    user_error: bool
    errors: list[ReviewErrorVM] = []


class JudgeErrorMarker(BaseModel):
    """One transcript error marker injected into PACKET_CONFIG.judge_errors."""

    id: int
    source: str
    tags: list[str]
    severity: str
    tick_start: int
    tick_end: int


class IndexRowVM(BaseModel):
    """One row of the packet index table."""

    href: str
    task_id: str
    display_task_id: str
    domain_prefix: str
    sim_id: str
    experiment: str
    double_annotate: Annotated[
        bool,
        Field(
            description="Badged on the index: this sim is annotated by TWO "
            "raters (assignment is operational; the rater field in each "
            "export distinguishes them at ingest)."
        ),
    ] = False


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def build_task_vm(task: Optional[Task]) -> Optional[TaskInfoVM]:
    if task is None:
        return None
    instructions = task.user_scenario.instructions
    if isinstance(instructions, StructuredUserInstructions):
        return TaskInfoVM(
            structured=True,
            reason_for_call=instructions.reason_for_call or "N/A",
            known_info=instructions.known_info or "N/A",
            unknown_info=instructions.unknown_info or "N/A",
            task_instructions=instructions.task_instructions or "N/A",
        )
    return TaskInfoVM(structured=False, raw=str(instructions))


def build_eval_vm(task: Optional[Task]) -> Optional[EvalVM]:
    if task is None or task.evaluation_criteria is None:
        return None
    ec = task.evaluation_criteria

    def functional(name: str, arguments: dict) -> str:
        return to_functional_format(ToolCall(name=name, arguments=arguments))

    return EvalVM(
        reward_basis=[b.value for b in ec.reward_basis],
        actions=[functional(a.name, a.arguments) for a in (ec.actions or [])],
        communicate=list(ec.communicate_info or []),
        nl_assertions=list(ec.nl_assertions or []),
        env_assertions=[
            functional(a.func_name, a.arguments) for a in (ec.env_assertions or [])
        ],
    )


def build_meta_vm(sim: SimulationRun, experiment_label: str, domain: str) -> MetaVM:
    reward = sim.reward_info.reward if sim.reward_info else None
    reward_class = (
        "success" if sim.reward_info and sim.reward_info.reward > 0 else "failure"
    )
    return MetaVM(
        experiment=experiment_label,
        domain=domain,
        task_id=str(sim.task_id),
        sim_id=sim.id,
        trial=sim.trial if sim.trial is not None else 0,
        duration=f"{sim.duration:.2f}s",
        termination=str(
            getattr(sim.termination_reason, "value", sim.termination_reason)
        ),
        reward="N/A" if reward is None else str(reward),
        reward_class=reward_class,
    )


def build_reward_vm(sim: SimulationRun) -> Optional[RewardVM]:
    ri = sim.reward_info
    if not ri:
        return None
    return RewardVM(
        breakdown=[
            BreakdownVM(
                label=comp.value,
                score=score,
                cls="success" if (score or 0) > 0 else "failure",
            )
            for comp, score in (ri.reward_breakdown or {}).items()
        ],
        note=str(ri.info["note"]) if ri.info and ri.info.get("note") else "",
        db_check=ri.db_check.db_match if ri.db_check else None,
        action_checks=[
            CheckVM(
                ok=ac.action_match,
                text=to_functional_format(
                    ToolCall(name=ac.action.name, arguments=ac.action.arguments)
                ),
            )
            for ac in ri.action_checks or []
        ],
        communicate_checks=[
            CheckVM(ok=c.met, text=c.info, justification=c.justification)
            for c in ri.communicate_checks or []
        ],
        nl_checks=[
            CheckVM(ok=c.met, text=c.nl_assertion, justification=c.justification)
            for c in ri.nl_assertions or []
        ],
        env_checks=[
            CheckVM(ok=ea.met, text=ea.env_assertion.func_name)
            for ea in ri.env_assertions or []
        ],
    )


def build_review_vm(
    sim: SimulationRun,
) -> tuple[Optional[ReviewVM], list[JudgeErrorMarker]]:
    review = sim.review
    if not review:
        return None, []

    errors: list[ReviewErrorVM] = []
    markers: list[JudgeErrorMarker] = []
    for i, err in enumerate(review.errors):
        severity = err.severity or ""
        # Half-duplex findings carry turn_idx instead of a tick range. The
        # message rows key their data-tick-start/end on the same turn_idx
        # (SimulationRun.messages[].turn_idx, the index the judge numbers),
        # so a degenerate [turn_idx, turn_idx] range anchors them with the
        # marker JS untouched.
        tick_start, tick_end = err.tick_start, err.tick_end
        if tick_start is None and tick_end is None and err.turn_idx is not None:
            tick_start = tick_end = err.turn_idx
        errors.append(
            ReviewErrorVM(
                id=i,
                source=err.source,
                source_class="agent" if err.source == "agent" else "user",
                severity=severity,
                severity_class="critical" if "critical" in severity else "minor",
                tags=err.error_tags,
                tick_start=tick_start,
                tick_end=tick_end,
                reasoning=err.reasoning,
                correct_behavior=err.correct_behavior or "",
            )
        )
        if tick_start is not None and tick_end is not None:
            markers.append(
                JudgeErrorMarker(
                    id=i,
                    source=err.source,
                    tags=err.error_tags,
                    severity=severity,
                    tick_start=tick_start,
                    tick_end=tick_end,
                )
            )

    vm = ReviewVM(
        summary=review.summary,
        agent_error=review.agent_error,
        user_error=review.user_error,
        errors=errors,
    )
    return vm, markers


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_sim_page(
    *,
    sim: SimulationRun,
    task: Optional[Task],
    form: FormType,
    identity: PacketIdentity,
    experiment_label: str,
    domain: str,
    policy: Optional[str],
    audio_filename: Optional[str],
    nav: PageNavVM,
) -> str:
    """Render one standalone annotation page for a simulation.

    Blind safety is STRUCTURAL: when the form spec's ``show_judge_review`` is
    off, the judge-review and reward view models are never built, so no
    outcome/judge trace can reach the page regardless of what the templates do.
    """
    spec = FORM_SPECS[form]
    meta = build_meta_vm(sim, experiment_label, domain)
    if spec.show_judge_review:
        review_vm, markers = build_review_vm(sim)
        reward_vm = build_reward_vm(sim)
    else:
        review_vm, markers, reward_vm = None, [], None

    packet_config = {
        "batch_id": identity.batch_id,
        "batch_name": identity.batch_name,
        "form": form.value,
        "sim_id": sim.id,
        "task_id": str(sim.task_id),
        "trial": sim.trial if sim.trial is not None else 0,
        "csv_headers": csv_headers(form),
        "dims": dims_config(form),
        "judge_errors": [m.model_dump() for m in markers],
        "guidelines_anchors": guidelines_anchors(form),
    }
    if spec.show_experience:
        packet_config["experience"] = experience_config(form)

    return (
        _env()
        .get_template("sim.html.j2")
        .render(
            packet_config=_config_json(packet_config),
            css=_page_css(form),
            js=_page_js(form),
            meta=meta,
            task=build_task_vm(task),
            policy=policy,
            eval=build_eval_vm(task),
            reward=reward_vm,
            review=review_vm,
            ticks_html=(
                generate_tick_rows(sim) if sim.ticks else generate_message_rows(sim)
            ),
            text_mode=not sim.ticks,
            collapse_transcript=form is FormType.USER_REALISM,
            audio_filename=audio_filename,
            nav=nav,
            guidelines=guidelines_html(form),
            form=spec,
            dims=load_rubric().dimensions if spec.show_audio else [],
            error_sources=list(ErrorSource),
            error_types=list(ErrorType),
            experience_anchors=CALLER_EXPERIENCE_ANCHORS,
            experience_factors=list(ExperienceFactor),
            experience_elaborate=EXPERIENCE_ELABORATE_FACTORS,
            breaking_points=list(BreakingPoint),
        )
    )


def render_index_page(*, identity: PacketIdentity, rows: list[IndexRowVM]) -> str:
    """Render the packet index (progress tracking + CSV export/import)."""
    form = identity.form
    packet_config = {
        "batch_id": identity.batch_id,
        "batch_name": identity.batch_name,
        "form": form.value,
        "csv_headers": csv_headers(form),
        "dims": dims_config(form),
        "guidelines_anchors": guidelines_anchors(form),
    }
    css = _static("index.css")
    js_parts = [_static("form_core.js")]
    js_parts.append(_static(INDEX_JS_FILES[form]))
    js = "\n".join(js_parts)
    return (
        _env()
        .get_template("index.html.j2")
        .render(
            packet_config=_config_json(packet_config),
            css=css,
            js=js,
            batch_name=identity.batch_name,
            task_label=FORM_LABELS[form],
            rows=rows,
        )
    )

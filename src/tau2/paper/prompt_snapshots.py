# Copyright Sierra
"""Exact prompts and prompt-input snapshots for the frozen τ-Multilingual cohort.

The paper cohort spans multiple repository revisions.  A current language pack
therefore cannot stand in for every run.  This module builds a portable,
content-addressed archive from two authoritative sources:

* rendered user and agent system prompts rebuilt by the run's recorded code;
* direct comparisons with the retained request logs wherever they exist;
* prompt inputs recorded directly in each ``results.json``; and
* runtime language-pack values read from the run's recorded git revision.

Comments and post-hoc judge configuration are deliberately excluded from the
language-pack object.  They were not benchmark prompt inputs.  The source pack
blob hash is retained so the extracted values remain auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from tau2.paper.multilingual import (
    DOMAINS,
    LANGUAGES,
    REPEATED_SYSTEMS,
    TEXT_SYSTEMS,
    _retail_ablation_path,
    _text_result_path,
    _voice_result_path,
)

PROMPT_SNAPSHOT_VERSION = "tau-multilingual-prompt-snapshots-v2"
RUNTIME_PROMPT_PACK_FIELDS = (
    "language",
    "display_name",
    "personas",
    "acoustic_presets",
    "backchannel_level",
    "agent_language_clause",
    "agent_native_script_db_clause",
    "agent_greeting",
    "default_out_of_turn_events_per_minute",
    "localization",
    "text_input",
)
PROMPT_RUN_INFO_FIELDS = (
    "agent_info",
    "audio_native_config",
    "channel_effects_mode",
    "complication_profile",
    "complication_rate",
    "environment_info",
    "max_steps",
    "seed",
    "speech_complexity",
    "speech_effects_mode",
    "task_set_name",
    "task_split_name",
    "task_subset",
    "text_input_style",
    "text_noise",
    "text_streaming_config",
    "timeout",
    "user_info",
    "user_persona_id",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )


class PromptObject(BaseModel):
    """One immutable object in the portable prompt archive."""

    model_config = ConfigDict(frozen=True)

    kind: Annotated[
        Literal[
            "agent_policy",
            "agent_system_prompt",
            "run_config",
            "runtime_language_pack",
            "tasks",
            "user_guidelines",
            "user_system_prompt",
        ],
        Field(description="Prompt-input object category."),
    ]
    sha256: Annotated[str, Field(description="SHA-256 of the exported bytes.")]
    file: Annotated[str, Field(description="Path relative to the archive root.")]
    bytes: Annotated[int, Field(description="Exported object size in bytes.")]


class PromptCell(BaseModel):
    """Prompt provenance for one frozen results file."""

    model_config = ConfigDict(frozen=True)

    cohort: Annotated[
        Literal["voice", "text", "voice_ablation"],
        Field(description="Frozen result cohort."),
    ]
    language: Annotated[str, Field(description="Language code.")]
    domain: Annotated[str, Field(description="Benchmark domain.")]
    system: Annotated[str, Field(description="System or ablation arm.")]
    results_path: Annotated[
        str, Field(description="Results path relative to the evidence root.")
    ]
    results_sha256: Annotated[
        str, Field(description="SHA-256 of the authoritative results file.")
    ]
    git_commit: Annotated[
        str, Field(description="Repository revision recorded by the run.")
    ]
    source_pack_path: Annotated[
        str, Field(description="Language-pack path at the recorded revision.")
    ]
    source_pack_sha256: Annotated[
        str, Field(description="SHA-256 of the complete historical pack bytes.")
    ]
    objects: Annotated[
        dict[str, str],
        Field(description="Object kind to content SHA-256 for this cell."),
    ]
    simulations: Annotated[
        list["SimulationPrompts"],
        Field(description="Exact rendered prompts for every scored simulation."),
    ]
    prompt_profile_sha256: Annotated[
        str,
        Field(description="Digest of the ordered prompt-object identities."),
    ]


class PromptSnapshotManifest(BaseModel):
    """Complete prompt archive for every frozen paper result cell."""

    model_config = ConfigDict(frozen=True)

    schema_version: Annotated[
        int, Field(description="Serialized manifest schema version.")
    ] = 2
    export_version: Annotated[
        str, Field(description="Versioned archive construction contract.")
    ] = PROMPT_SNAPSHOT_VERSION
    artifact_id: Annotated[
        str, Field(description="Content-derived identity of the complete archive.")
    ]
    cells: Annotated[list[PromptCell], Field(description="Frozen cell mappings.")]
    objects: Annotated[
        list[PromptObject], Field(description="Deduplicated prompt objects.")
    ]


class PromptSnapshotVerification(BaseModel):
    """Integrity result for an existing prompt archive."""

    model_config = ConfigDict(frozen=True)

    ok: Annotated[bool, Field(description="Whether every object passed validation.")]
    checked_objects: Annotated[int, Field(description="Number of objects checked.")]
    problems: Annotated[list[str], Field(description="Integrity failures.")]


class PromptSourceVerification(BaseModel):
    """Comparison of an archive with frozen results and historical packs."""

    model_config = ConfigDict(frozen=True)

    ok: Annotated[bool, Field(description="Whether every source comparison passed.")]
    checked_cells: Annotated[int, Field(description="Number of cells checked.")]
    checked_simulations: Annotated[
        int, Field(description="Number of scored simulations checked.")
    ]
    logged_user_prompts_compared: Annotated[
        int, Field(description="Retained user-request prompts compared byte for byte.")
    ]
    logged_agent_prompts_compared: Annotated[
        int, Field(description="Retained agent-request prompts compared byte for byte.")
    ]
    problems: Annotated[list[str], Field(description="Source mismatches.")]


class SimulationPrompts(BaseModel):
    """Exact system prompts used by one scored simulation."""

    model_config = ConfigDict(frozen=True)

    simulation_id: Annotated[str, Field(description="Recorded simulation UUID.")]
    task_id: Annotated[str, Field(description="Recorded task identifier.")]
    trial: Annotated[int, Field(description="Recorded trial number.")]
    persona_id: Annotated[str, Field(description="Resolved language-pack persona.")]
    agent_system_prompt_sha256: Annotated[
        str, Field(description="Exact rendered agent system-prompt object.")
    ]
    user_system_prompt_sha256: Annotated[
        str, Field(description="Exact rendered user system-prompt object.")
    ]


class _HistoricalPromptRequest(BaseModel):
    """One unique prompt-rendering context passed to frozen source code."""

    key: str
    git_commit: str
    cohort: Literal["voice", "text", "voice_ablation"]
    language: str
    domain: str
    provider: str | None
    use_xml_prompt: bool
    policy: str
    user_guidelines: str
    task: dict[str, object]
    persona_override: str
    run_seed: int


class _HistoricalPromptPair(BaseModel):
    """Rendered prompt pair returned by frozen source code."""

    agent: str
    user: str
    persona_id: str


class _ResultSpec(BaseModel):
    """Internal identity of one paper results file."""

    cohort: Literal["voice", "text", "voice_ablation"]
    language: str
    domain: str
    system: str
    path: Path


_HISTORICAL_RENDERER = r"""
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
sys.path.insert(0, str(source_root / "src"))

from tau2.agent.discrete_time_audio_native_agent import (  # noqa: E402
    DiscreteTimeAudioNativeAgent,
)
from tau2.agent.llm_agent import LLMAgent  # noqa: E402
from tau2.data_model.tasks import Task  # noqa: E402
from tau2.multilingual.registry import (  # noqa: E402
    get_multilingual_persona,
    resolve_task_persona,
)
from tau2.user import user_simulator as user_simulator_module  # noqa: E402
from tau2.user import user_simulator_streaming as streaming_module  # noqa: E402
from tau2.user.user_simulator import UserSimulator  # noqa: E402
try:  # Present only in the later frozen revisions.
    from tau2.user.user_simulator import CallDirection  # noqa: E402
except ImportError:  # pragma: no cover - executed only by historical source
    CallDirection = None
from tau2.user.user_simulator_streaming import (  # noqa: E402
    VoiceStreamingUserSimulator,
)

requests = json.loads(input_path.read_text())
rendered = {}
for request in requests:
    task = Task.model_validate(request["task"])
    persona_id = resolve_task_persona(
        request["persona_override"],
        task_id=task.id,
        run_seed=request["run_seed"],
        domain=request["domain"],
    )
    hit = get_multilingual_persona(persona_id)
    if hit is None:
        raise ValueError(f"unknown historical persona: {persona_id}")
    _pack, persona = hit

    if request["cohort"] == "text":
        user = UserSimulator.__new__(UserSimulator)
        user.input_style_directive = None
        user.entity_noise = None
        agent = LLMAgent.__new__(LLMAgent)
    else:
        user = VoiceStreamingUserSimulator.__new__(VoiceStreamingUserSimulator)
        agent = DiscreteTimeAudioNativeAgent.__new__(DiscreteTimeAudioNativeAgent)
        agent.provider = request["provider"]
        agent.use_xml_prompt = request["use_xml_prompt"]

    user.instructions = str(task.user_scenario)
    user.persona_config = persona
    user.tools = None
    user.domain = request["domain"]
    # ``results.json`` records the actual guideline bytes. A few frozen runs
    # came from a working tree whose guideline file differed from HEAD, so the
    # recorded snapshot is more authoritative than ``git show`` for this one
    # input. The request-log comparison below independently checks the choice.
    user_simulator_module.get_global_user_sim_guidelines = (
        lambda *args, **kwargs: request["user_guidelines"]
    )
    user_simulator_module.get_global_user_sim_guidelines_voice = (
        lambda *args, **kwargs: request["user_guidelines"]
    )
    streaming_module.get_global_user_sim_guidelines_voice = (
        lambda *args, **kwargs: request["user_guidelines"]
    )
    # The paper cohort used the English prompt scaffold throughout; this
    # attribute existed only in the revisions where the retired alternative
    # scaffold was still implemented.
    user.prompt_language = "english"
    if CallDirection is not None:
        user.call_direction = CallDirection.INBOUND

    agent.domain_policy = request["policy"]
    agent.language = request["language"]
    agent.locale = persona.locale
    # Harmless on revisions predating the native-script arm; required on the
    # two frozen retail-ablation revisions that read this attribute.
    agent.native_script_db = request["task"]["id"].endswith("_identity_native")
    agent_prompt = (
        agent.system_prompt
        if request["cohort"] == "text"
        else agent._build_system_prompt()
    )
    rendered[request["key"]] = {
        "agent": agent_prompt,
        "user": user.system_prompt,
        "persona_id": persona.persona_id,
    }

output_path.write_text(json.dumps(rendered, ensure_ascii=False))
"""


def _artifact_dir(spec: _ResultSpec, task_id: str, simulation_id: str) -> Path:
    return spec.path.parent / "artifacts" / f"task_{task_id}" / f"sim_{simulation_id}"


def _historical_request_key(request: _HistoricalPromptRequest) -> str:
    payload = request.model_dump(exclude={"key"})
    return _sha256_bytes(_canonical_json_bytes(payload))


def _render_historical_prompts(
    repo: Path,
    commit: str,
    requests: list[_HistoricalPromptRequest],
) -> dict[str, _HistoricalPromptPair]:
    """Render prompt pairs by executing the exact recorded repository revision."""
    if not requests:
        return {}
    with tempfile.TemporaryDirectory(prefix="tau2-frozen-prompts-") as temporary:
        root = Path(temporary)
        archive = root / "source.tar"
        source = root / "source"
        source.mkdir()
        completed = subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={archive}",
                commit,
                "src/tau2",
                "data/tau2/multilingual",
                "data/tau2/user_simulator",
            ],
            cwd=repo,
            check=False,
            capture_output=True,
        )
        if completed.returncode:
            detail = completed.stderr.decode(errors="replace").strip()
            raise ValueError(f"cannot archive prompt source at {commit}: {detail}")
        subprocess.run(
            ["tar", "-xf", str(archive), "-C", str(source)],
            check=True,
            capture_output=True,
        )
        request_path = root / "requests.json"
        response_path = root / "responses.json"
        request_path.write_text(
            json.dumps(
                [request.model_dump(mode="json") for request in requests],
                ensure_ascii=False,
            )
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _HISTORICAL_RENDERER,
                str(source),
                str(request_path),
                str(response_path),
            ],
            cwd=source,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(
                f"historical prompt renderer failed at {commit}: {detail[-4000:]}"
            )
        payload = json.loads(response_path.read_text())
    return {
        key: _HistoricalPromptPair.model_validate(value)
        for key, value in payload.items()
    }


def _prompt_requests_for_cell(
    spec: _ResultSpec, payload: dict[str, object]
) -> tuple[list[_HistoricalPromptRequest], list[tuple[dict[str, object], str]]]:
    """Build unique render requests and map each simulation to its request key."""
    info = payload["info"]
    assert isinstance(info, dict)
    task_by_id = {
        task["id"]: task for task in payload["tasks"] if isinstance(task, dict)
    }
    provider = None
    use_xml_prompt = False
    audio_config = info.get("audio_native_config")
    if isinstance(audio_config, dict):
        provider = audio_config.get("provider")
        use_xml_prompt = bool(audio_config.get("use_xml_prompt", False))

    by_key: dict[str, _HistoricalPromptRequest] = {}
    mapping: list[tuple[dict[str, object], str]] = []
    for index in payload["simulation_index"]:
        if not isinstance(index, dict):
            raise ValueError(f"invalid simulation index in {spec.path}")
        task_id = str(index["task_id"])
        request = _HistoricalPromptRequest(
            key="pending",
            git_commit=str(info["git_commit"]),
            cohort=spec.cohort,
            language=spec.language,
            domain=spec.domain,
            provider=str(provider) if provider is not None else None,
            use_xml_prompt=use_xml_prompt,
            policy=str(info["environment_info"]["policy"]),
            user_guidelines=str(info["user_info"]["global_simulation_guidelines"]),
            task=task_by_id[task_id],
            # Early result schemas did not record this field even though the
            # task set and task logs show the language override. Every paper
            # cell is a registered language-pack run, so its cell language is
            # the exact runtime fallback for those files.
            persona_override=str(info.get("user_persona_id") or spec.language),
            run_seed=int(info["seed"]),
        )
        key = _historical_request_key(request)
        request = request.model_copy(update={"key": key})
        by_key.setdefault(key, request)
        mapping.append((index, key))
    return list(by_key.values()), mapping


def _restore_logged_content(content: object) -> str:
    """Undo the debug logger's newline splitting without changing prompt bytes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(line, str) for line in content):
        return "\n".join(content)
    raise ValueError("logged system prompt has unsupported content")


def _logged_request_prompts(directory: Path, patterns: tuple[str, ...]) -> list[str]:
    prompts: list[str] = []
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            payload = json.loads(path.read_text())
            messages = payload.get("request", {}).get("messages", [])
            system = next(
                (message for message in messages if message.get("role") == "system"),
                None,
            )
            if system is not None:
                prompts.append(_restore_logged_content(system.get("content")))
    return prompts


_INSTRUCTIONS_LINE = re.compile(r'^\s*"instructions":\s*(".*")[,]?$')


def _logged_voice_agent_prompts(task_log: Path) -> list[str]:
    """Read exact provider instructions when a provider retained its config."""
    if not task_log.is_file():
        return []
    prompts: list[str] = []
    for line in task_log.read_text(errors="replace").splitlines():
        match = _INSTRUCTIONS_LINE.match(line)
        if match:
            prompts.append(json.loads(match.group(1)))
    return prompts


def _single_logged_prompt(prompts: list[str], label: str) -> str | None:
    """Return a retained prompt, rejecting inconsistent logs for one simulation."""
    unique = set(prompts)
    if len(unique) > 1:
        raise ValueError(f"retained {label} prompts disagree within one simulation")
    return next(iter(unique), None)


def _exact_prompt_pair(
    spec: _ResultSpec,
    index: dict[str, object],
    rendered: _HistoricalPromptPair,
) -> tuple[_HistoricalPromptPair, int, int]:
    """Prefer exact retained requests; use frozen-source rendering when absent."""
    simulation_id = str(index["id"])
    artifact = _artifact_dir(spec, str(index["task_id"]), simulation_id)
    user_logs = _logged_request_prompts(
        artifact / "llm_debug",
        ("*user_streaming_response*.json", "*user_simulator_response*.json"),
    )
    if spec.cohort == "text":
        agent_logs = _logged_request_prompts(
            artifact / "llm_debug", ("*agent_response*.json",)
        )
    elif spec.system == "xai_provider_default":
        agent_logs = _logged_voice_agent_prompts(artifact / "task.log")
    else:
        agent_logs = []
    user_prompt = _single_logged_prompt(user_logs, "user") or rendered.user
    agent_prompt = _single_logged_prompt(agent_logs, "agent") or rendered.agent
    return (
        rendered.model_copy(update={"agent": agent_prompt, "user": user_prompt}),
        len(user_logs),
        len(agent_logs),
    )


def _render_all_specs(
    repo: Path, specs: list[_ResultSpec]
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, list[tuple[dict[str, object], str]]],
    dict[str, _HistoricalPromptPair],
]:
    """Load all cells and render each unique historical context only once."""
    payloads: dict[str, dict[str, object]] = {}
    mappings: dict[str, list[tuple[dict[str, object], str]]] = {}
    requests_by_commit: dict[str, dict[str, _HistoricalPromptRequest]] = {}
    for spec in specs:
        identity = str(spec.path)
        payload = json.loads(spec.path.read_text())
        payloads[identity] = payload
        requests, mapping = _prompt_requests_for_cell(spec, payload)
        mappings[identity] = mapping
        commit = str(payload["info"]["git_commit"])
        commit_requests = requests_by_commit.setdefault(commit, {})
        for request in requests:
            commit_requests.setdefault(request.key, request)

    rendered: dict[str, _HistoricalPromptPair] = {}
    for commit, by_key in sorted(requests_by_commit.items()):
        commit_rendered = _render_historical_prompts(
            repo, commit, sorted(by_key.values(), key=lambda item: item.key)
        )
        overlap = rendered.keys() & commit_rendered.keys()
        if overlap:
            raise ValueError(f"historical prompt request collision: {sorted(overlap)}")
        rendered.update(commit_rendered)
    return payloads, mappings, rendered


def _result_specs(root: Path) -> list[_ResultSpec]:
    specs: list[_ResultSpec] = []
    for language in LANGUAGES:
        for domain in DOMAINS:
            for system in (*REPEATED_SYSTEMS, "xai_provider_default"):
                specs.append(
                    _ResultSpec(
                        cohort="voice",
                        language=language,
                        domain=domain,
                        system=system,
                        path=_voice_result_path(root, language, domain, system),
                    )
                )
            for system in TEXT_SYSTEMS:
                specs.append(
                    _ResultSpec(
                        cohort="text",
                        language=language,
                        domain=domain,
                        system=system,
                        path=_text_result_path(root, language, domain, system),
                    )
                )
    for language in ("hi", "zh"):
        for system in ("openai_xhigh", "gemini_high"):
            for native_script, suffix in (
                (False, "source_entities"),
                (True, "native_script"),
            ):
                specs.append(
                    _ResultSpec(
                        cohort="voice_ablation",
                        language=language,
                        domain="retail",
                        system=f"{system}_{suffix}",
                        path=_retail_ablation_path(
                            root,
                            language,
                            system,
                            native_script=native_script,
                        ),
                    )
                )
    return specs


def _git_show(repo: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        detail = completed.stderr.decode(errors="replace").strip()
        raise ValueError(f"cannot read {path} at {commit}: {detail}")
    return completed.stdout


def _runtime_pack(pack_bytes: bytes) -> dict[str, object]:
    payload = yaml.safe_load(pack_bytes) or {}
    if not isinstance(payload, dict):
        raise ValueError("historical language pack is not a mapping")
    return {key: payload.get(key) for key in RUNTIME_PROMPT_PACK_FIELDS}


def _prompt_run_info(info: dict[str, object]) -> dict[str, object]:
    selected = {key: info.get(key) for key in PROMPT_RUN_INFO_FIELDS}
    user_info = selected.get("user_info")
    if isinstance(user_info, dict):
        # Stored separately as an exact Markdown object; avoid duplicating it.
        selected["user_info"] = {
            key: value
            for key, value in user_info.items()
            if key != "global_simulation_guidelines"
        }
    environment_info = selected.get("environment_info")
    if isinstance(environment_info, dict):
        # Stored separately as an exact Markdown object; retain the other
        # environment prompt/config inputs.
        selected["environment_info"] = {
            key: value for key, value in environment_info.items() if key != "policy"
        }
    return selected


def _object_extension(kind: str) -> str:
    if kind in {
        "agent_policy",
        "agent_system_prompt",
        "user_guidelines",
        "user_system_prompt",
    }:
        return "md"
    return "json"


def _write_object(
    out: Path,
    kind: str,
    data: bytes,
    objects: dict[tuple[str, str], PromptObject],
) -> str:
    digest = _sha256_bytes(data)
    key = (kind, digest)
    if key not in objects:
        relative = Path("objects") / kind / f"{digest}.{_object_extension(kind)}"
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        objects[key] = PromptObject(
            kind=kind,
            sha256=digest,
            file=relative.as_posix(),
            bytes=len(data),
        )
    return digest


def _simulation_prompts_digest(simulations: list[SimulationPrompts]) -> str:
    serialized = "".join(
        "\t".join(
            (
                item.simulation_id,
                item.task_id,
                str(item.trial),
                item.persona_id,
                item.agent_system_prompt_sha256,
                item.user_system_prompt_sha256,
            )
        )
        + "\n"
        for item in sorted(simulations, key=lambda value: value.simulation_id)
    )
    return _sha256_bytes(serialized.encode())


def _profile_digest(
    objects: dict[str, str], simulations: list[SimulationPrompts]
) -> str:
    serialized = "".join(f"{key}\t{objects[key]}\n" for key in sorted(objects))
    serialized += f"simulations\t{_simulation_prompts_digest(simulations)}\n"
    return _sha256_bytes(serialized.encode())


def _artifact_digest(cells: list[PromptCell]) -> str:
    serialized = "".join(
        f"{cell.results_path}\t{cell.results_sha256}\t{cell.prompt_profile_sha256}\n"
        for cell in sorted(cells, key=lambda item: item.results_path)
    )
    return _sha256_bytes(serialized.encode())


def _prune_unreferenced_objects(out: Path, manifest: PromptSnapshotManifest) -> None:
    """Remove stale generated objects that are absent from the new manifest."""
    object_root = out / "objects"
    expected = {(out / obj.file).resolve() for obj in manifest.objects}
    if not object_root.is_dir():
        return
    for path in object_root.rglob("*"):
        if path.is_file() and path.resolve() not in expected:
            path.unlink()
    for path in sorted(object_root.rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def export_prompt_snapshots(
    evidence_root: Path,
    out: Path,
    *,
    repo: Path | None = None,
) -> PromptSnapshotManifest:
    """Build exact rendered prompts and inputs for all 134 paper result files."""
    root = evidence_root.expanduser().resolve()
    repository = (repo or Path.cwd()).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    objects: dict[tuple[str, str], PromptObject] = {}
    cells: list[PromptCell] = []
    specs = _result_specs(root)
    payloads, mappings, rendered = _render_all_specs(repository, specs)

    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"missing frozen result: {spec.path}")
        result_bytes = spec.path.read_bytes()
        payload = payloads[str(spec.path)]
        info = payload["info"]
        commit = info["git_commit"]
        pack_path = f"data/tau2/multilingual/{spec.language}/pack.yaml"
        pack_bytes = _git_show(repository, commit, pack_path)

        guidelines = info["user_info"]["global_simulation_guidelines"].encode("utf-8")
        policy = info["environment_info"]["policy"].encode("utf-8")
        cell_objects = {
            "agent_policy": _write_object(out, "agent_policy", policy, objects),
            "run_config": _write_object(
                out,
                "run_config",
                _canonical_json_bytes(_prompt_run_info(info)),
                objects,
            ),
            "runtime_language_pack": _write_object(
                out,
                "runtime_language_pack",
                _canonical_json_bytes(_runtime_pack(pack_bytes)),
                objects,
            ),
            "tasks": _write_object(
                out, "tasks", _canonical_json_bytes(payload["tasks"]), objects
            ),
            "user_guidelines": _write_object(
                out, "user_guidelines", guidelines, objects
            ),
        }
        simulation_prompts: list[SimulationPrompts] = []
        for index, request_key in mappings[str(spec.path)]:
            pair, _user_log_count, _agent_log_count = _exact_prompt_pair(
                spec, index, rendered[request_key]
            )
            simulation_prompts.append(
                SimulationPrompts(
                    simulation_id=str(index["id"]),
                    task_id=str(index["task_id"]),
                    trial=int(index["trial"]),
                    persona_id=pair.persona_id,
                    agent_system_prompt_sha256=_write_object(
                        out,
                        "agent_system_prompt",
                        pair.agent.encode("utf-8"),
                        objects,
                    ),
                    user_system_prompt_sha256=_write_object(
                        out,
                        "user_system_prompt",
                        pair.user.encode("utf-8"),
                        objects,
                    ),
                )
            )
        cells.append(
            PromptCell(
                cohort=spec.cohort,
                language=spec.language,
                domain=spec.domain,
                system=spec.system,
                results_path=spec.path.relative_to(root).as_posix(),
                results_sha256=_sha256_bytes(result_bytes),
                git_commit=commit,
                source_pack_path=pack_path,
                source_pack_sha256=_sha256_bytes(pack_bytes),
                objects=cell_objects,
                simulations=sorted(
                    simulation_prompts, key=lambda item: item.simulation_id
                ),
                prompt_profile_sha256=_profile_digest(cell_objects, simulation_prompts),
            )
        )

    manifest = PromptSnapshotManifest(
        artifact_id=_artifact_digest(cells),
        cells=sorted(cells, key=lambda item: item.results_path),
        objects=sorted(objects.values(), key=lambda item: (item.kind, item.sha256)),
    )
    _prune_unreferenced_objects(out, manifest)
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    verification = verify_prompt_snapshot_export(out)
    if not verification.ok:
        raise ValueError(
            "prompt snapshot verification failed: " + "; ".join(verification.problems)
        )
    source_verification = verify_prompt_snapshot_sources(
        root,
        out,
        repo=repository,
    )
    if not source_verification.ok:
        raise ValueError(
            "prompt source verification failed: "
            + "; ".join(source_verification.problems)
        )
    (out / "source_verification.json").write_text(
        source_verification.model_dump_json(indent=2) + "\n"
    )
    return manifest


def verify_prompt_snapshot_export(root: Path) -> PromptSnapshotVerification:
    """Verify every content-addressed object and cell profile in an archive."""
    archive = root.expanduser().resolve()
    manifest = PromptSnapshotManifest.model_validate_json(
        (archive / "manifest.json").read_text()
    )
    problems: list[str] = []
    by_identity = {(obj.kind, obj.sha256): obj for obj in manifest.objects}
    expected_files = {(archive / obj.file).resolve() for obj in manifest.objects}
    object_root = archive / "objects"
    if object_root.is_dir():
        for path in object_root.rglob("*"):
            if path.is_file() and path.resolve() not in expected_files:
                problems.append(f"unreferenced object: {path.relative_to(archive)}")
    for obj in manifest.objects:
        path = archive / obj.file
        if not path.is_file():
            problems.append(f"missing object: {obj.file}")
            continue
        data = path.read_bytes()
        if len(data) != obj.bytes:
            problems.append(f"size mismatch: {obj.file}")
        if _sha256_bytes(data) != obj.sha256:
            problems.append(f"hash mismatch: {obj.file}")
    for cell in manifest.cells:
        for kind, digest in cell.objects.items():
            if (kind, digest) not in by_identity:
                problems.append(f"{cell.results_path}: unknown {kind} object {digest}")
        for simulation in cell.simulations:
            for kind, digest in (
                ("agent_system_prompt", simulation.agent_system_prompt_sha256),
                ("user_system_prompt", simulation.user_system_prompt_sha256),
            ):
                if (kind, digest) not in by_identity:
                    problems.append(
                        f"{cell.results_path}/{simulation.simulation_id}: "
                        f"unknown {kind} object {digest}"
                    )
        if (
            _profile_digest(cell.objects, cell.simulations)
            != cell.prompt_profile_sha256
        ):
            problems.append(f"{cell.results_path}: prompt profile hash mismatch")
    if _artifact_digest(manifest.cells) != manifest.artifact_id:
        problems.append("archive artifact hash mismatch")
    return PromptSnapshotVerification(
        ok=not problems,
        checked_objects=len(manifest.objects),
        problems=problems,
    )


def verify_prompt_snapshot_sources(
    evidence_root: Path,
    archive_root: Path,
    *,
    repo: Path | None = None,
) -> PromptSourceVerification:
    """Prove that every archived object equals its frozen authoritative source."""
    evidence = evidence_root.expanduser().resolve()
    archive = archive_root.expanduser().resolve()
    repository = (repo or Path.cwd()).expanduser().resolve()
    manifest = PromptSnapshotManifest.model_validate_json(
        (archive / "manifest.json").read_text()
    )
    problems = verify_prompt_snapshot_export(archive).problems.copy()
    specs = _result_specs(evidence)
    expected_paths = {spec.path.relative_to(evidence).as_posix() for spec in specs}
    archived_paths = {cell.results_path for cell in manifest.cells}
    if archived_paths != expected_paths:
        missing = sorted(expected_paths - archived_paths)
        extra = sorted(archived_paths - expected_paths)
        problems.append(f"cell inventory mismatch: missing={missing}, extra={extra}")

    object_paths = {
        (obj.kind, obj.sha256): archive / obj.file for obj in manifest.objects
    }
    specs_by_path = {spec.path.relative_to(evidence).as_posix(): spec for spec in specs}
    payloads, mappings, rendered = _render_all_specs(repository, specs)
    checked_simulations = 0
    logged_user_prompts_compared = 0
    logged_agent_prompts_compared = 0
    for cell in manifest.cells:
        spec = specs_by_path.get(cell.results_path)
        if spec is None:
            continue
        result_path = evidence / cell.results_path
        if not result_path.is_file():
            problems.append(f"missing frozen result: {cell.results_path}")
            continue
        result_bytes = result_path.read_bytes()
        if _sha256_bytes(result_bytes) != cell.results_sha256:
            problems.append(f"{cell.results_path}: result hash mismatch")
            continue
        payload = payloads[str(spec.path)]
        info = payload["info"]
        if info["git_commit"] != cell.git_commit:
            problems.append(f"{cell.results_path}: recorded commit mismatch")

        try:
            pack_bytes = _git_show(repository, cell.git_commit, cell.source_pack_path)
        except ValueError as exc:
            problems.append(f"{cell.results_path}: {exc}")
            continue
        if _sha256_bytes(pack_bytes) != cell.source_pack_sha256:
            problems.append(f"{cell.results_path}: historical pack hash mismatch")

        expected: dict[str, bytes] = {
            "agent_policy": info["environment_info"]["policy"].encode("utf-8"),
            "run_config": _canonical_json_bytes(_prompt_run_info(info)),
            "runtime_language_pack": _canonical_json_bytes(_runtime_pack(pack_bytes)),
            "tasks": _canonical_json_bytes(payload["tasks"]),
            "user_guidelines": info["user_info"]["global_simulation_guidelines"].encode(
                "utf-8"
            ),
        }
        for kind, expected_bytes in expected.items():
            digest = cell.objects.get(kind)
            object_path = object_paths.get((kind, digest or ""))
            if object_path is None or not object_path.is_file():
                problems.append(f"{cell.results_path}: missing {kind} object")
                continue
            if object_path.read_bytes() != expected_bytes:
                problems.append(f"{cell.results_path}: {kind} differs from source")

        archived_simulations = {
            simulation.simulation_id: simulation for simulation in cell.simulations
        }
        expected_mapping = mappings[str(spec.path)]
        expected_simulation_ids = {str(index["id"]) for index, _key in expected_mapping}
        if set(archived_simulations) != expected_simulation_ids:
            missing = sorted(expected_simulation_ids - set(archived_simulations))
            extra = sorted(set(archived_simulations) - expected_simulation_ids)
            problems.append(
                f"{cell.results_path}: simulation prompt inventory mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for index, request_key in expected_mapping:
            simulation_id = str(index["id"])
            archived = archived_simulations.get(simulation_id)
            if archived is None:
                continue
            checked_simulations += 1
            pair, user_log_count, agent_log_count = _exact_prompt_pair(
                spec, index, rendered[request_key]
            )
            logged_user_prompts_compared += user_log_count
            logged_agent_prompts_compared += agent_log_count
            expected_metadata = (
                str(index["task_id"]),
                int(index["trial"]),
                pair.persona_id,
            )
            archived_metadata = (
                archived.task_id,
                archived.trial,
                archived.persona_id,
            )
            if archived_metadata != expected_metadata:
                problems.append(
                    f"{cell.results_path}/{simulation_id}: simulation metadata differs"
                )

            for kind, digest, expected_prompt in (
                (
                    "agent_system_prompt",
                    archived.agent_system_prompt_sha256,
                    pair.agent,
                ),
                (
                    "user_system_prompt",
                    archived.user_system_prompt_sha256,
                    pair.user,
                ),
            ):
                object_path = object_paths.get((kind, digest))
                if object_path is None or not object_path.is_file():
                    problems.append(
                        f"{cell.results_path}/{simulation_id}: missing {kind} object"
                    )
                elif object_path.read_bytes() != expected_prompt.encode("utf-8"):
                    problems.append(
                        f"{cell.results_path}/{simulation_id}: "
                        f"{kind} differs from frozen renderer"
                    )

    return PromptSourceVerification(
        ok=not problems,
        checked_cells=len(manifest.cells),
        checked_simulations=checked_simulations,
        logged_user_prompts_compared=logged_user_prompts_compared,
        logged_agent_prompts_compared=logged_agent_prompts_compared,
        problems=problems,
    )

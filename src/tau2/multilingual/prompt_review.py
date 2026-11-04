"""Offline review using the same prompt functions as ordinary voice runs."""

import hashlib
import html
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.config import VOICE_INSTRUCTION_PROMPT_VERSION
from tau2.data_model.language import CallerSource, LanguageTaskInfo
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.multilingual.runtime import (
    list_languages,
    load_pack,
    load_run_language,
    resolve_task,
)
from tau2.voice.voice_gender import selected_agent_voice


class PromptPair(BaseModel):
    """One actual task/persona assignment and its fully rendered runtime prompts."""

    model_config = ConfigDict(extra="forbid")
    voice_instruction_prompt_version: str = Field(
        description="Implementation revision of the rendered shared voice instructions."
    )
    language: str = Field(description="Published language identifier.")
    display_name: str = Field(description="Language display name.")
    domain: str = Field(description="Ordinary runtime domain.")
    persona_id: str = Field(description="Persona resolved by ordinary task assignment.")
    task_id: str = Field(description="Actual selected localized task.")
    canonical_task_id: str = Field(
        description="Explicit canonical task correspondence."
    )
    config: VoiceRunConfig = Field(
        description="Complete effective voice run configuration."
    )
    language_context: LanguageTaskInfo = Field(
        description="Resolved content, versions and artifact hashes."
    )
    agent_voice: str | None = Field(description="Effective provider voice.")
    agent_gender: str | None = Field(
        description="Perceived gender from the existing voice catalog."
    )
    agent_gender_source: str = Field(
        description="Catalog source used by runtime resolution."
    )
    agent_prompt: str = Field(description="Complete runtime agent system prompt.")
    customer_prompt: str = Field(
        description="Complete runtime customer system prompt including task instructions."
    )
    initial_agent_message: str | None = Field(
        description="Separate initial greeting, when configured."
    )


class PromptReview(BaseModel):
    """Versioned, reproducible review artifact; no evaluation or model calls."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["prompt-review-v3"] = "prompt-review-v3"
    source_sha256: dict[str, str] = Field(
        description="Digests of actual prompt-building source files."
    )
    pairs: list[PromptPair] = Field(
        description="One real task per declared persona and selected domain."
    )


def render_review(
    *,
    languages: list[str] | None = None,
    domains: list[str] | None = None,
    seed: int | None = None,
    template: VoiceRunConfig | None = None,
    benchmark: str | None = None,
) -> PromptReview:
    """Render complete prompts without constructing execution objects."""
    from tau2.agent.discrete_time_audio_native_agent import (
        build_audio_native_system_prompt,
    )
    from tau2.multilingual.runtime import agent_language_guidance, task_context
    from tau2.runner.build import (
        _build_env_kwargs,
        agent_domain_policy,
        build_environment,
        call_context,
        user_tools_for_task,
    )
    from tau2.runner.elicitation import user_prompt_task
    from tau2.user.guidelines import simulation_guidelines
    from tau2.user.user_simulator import build_user_system_prompt

    benchmark_configs = {}
    if benchmark is not None:
        from tau2.runner.presets import BenchmarkRequest, resolve_benchmark

        template = template or VoiceRunConfig(audio_native_config=AudioNativeConfig())
        if seed is not None:
            template = template.model_copy(update={"seed": seed})
        plan = resolve_benchmark(
            BenchmarkRequest(
                preset=benchmark,
                template=template,
                output=Path("offline-prompt-review"),
                languages=languages,
                domains=domains,
            )
        )
        benchmark_configs = {
            (config.language, config.domain): config for config in plan.configs
        }
        languages = list(dict.fromkeys(config.language for config in plan.configs))
        domains = list(dict.fromkeys(config.domain for config in plan.configs))
    selected = languages if languages is not None else list_languages()
    selected = sorted(set(selected), key=lambda language: (language != "en", language))
    if not selected:
        raise ValueError("No published language packs found")
    pairs = []
    for language in selected:
        pack = load_pack(language)
        pack_domains = [
            name
            for name, binding in pack.domains.items()
            if binding.caller == CallerSource.PACK
        ]
        for domain in sorted(set(domains or pack_domains)):
            base = (
                benchmark_configs.get((language, domain))
                or template
                or VoiceRunConfig(
                    domain=domain, audio_native_config=AudioNativeConfig()
                )
            )
            # Load exactly the binding and task selection the rendered calls use.
            loaded = load_run_language(
                VoiceRunConfig.model_validate(
                    {
                        **base.template_values(),
                        "domain": domain,
                        "language": language,
                        "task_set_name": None,
                    }
                )
            )
            resolved_seed = base.seed if seed is None else seed
            resolved = (
                (task, resolve_task(loaded, task, resolved_seed))
                for task in loaded.tasks
                if not benchmark_configs or task.id in base.task_ids
            )
            by_persona = {}
            for persona_id in sorted(
                pack.personas,
                key=lambda pid: (pack.personas[pid].gender != "male", pid),
            ):
                # Resolve tasks lazily until this persona has an actual task.
                for task, context in resolved:
                    by_persona.setdefault(context.persona_id, (task, context))
                    if persona_id in by_persona:
                        break
                match = by_persona.get(persona_id)
                if match is None:
                    raise ValueError(
                        f"No actual task resolves persona {language}/{domain}/{persona_id} at seed {resolved_seed}"
                    )
                task, context = match
                values = base.template_values()
                values.update(
                    domain=domain,
                    language=language,
                    seed=resolved_seed,
                    task_ids=[task.id],
                    num_tasks=None,
                    task_set_name=None,
                    language_artifacts=loaded.artifact_pins,
                    benchmark=base.benchmark.model_copy(
                        update={"selected_task_ids": [context.canonical_task_id]}
                    )
                    if base.benchmark
                    else None,
                )
                config = VoiceRunConfig.model_validate(values)
                # Language review covers the released audio-native agent and caller.
                # Custom registered implementations own their own prompt contracts.
                if config.effective_agent != "discrete_time_audio_native_agent":
                    raise ValueError("Prompt review requires the audio-native agent")
                context = task_context(config, task, context=context)
                environment = build_environment(
                    domain,
                    env_kwargs=_build_env_kwargs(
                        config, task, language_context=context
                    ),
                )
                prompt_task = user_prompt_task(config, task)
                user_tools = user_tools_for_task(environment, prompt_task)
                voice, gender = selected_agent_voice(config)
                greeting = (
                    task.agent_opener
                    if task.agent_opener is not None
                    else context.agent_greeting
                )
                instruction_override = None
                voice_prompt_version = VOICE_INSTRUCTION_PROMPT_VERSION
                caller_guidelines = simulation_guidelines(
                    context.persona, mode="voice", use_tools=user_tools is not None
                )
                if domain == "intake":
                    from tau2.domains.intake.prompts import (
                        ELICITATION_PROMPT_VERSION,
                        get_agent_instructions,
                        get_user_guidelines,
                    )

                    agent_directed = (
                        config.elicitation.strategy.value == "agent_directed"
                    )
                    instruction_override = get_agent_instructions(
                        config.audio_native_config.provider, agent_directed
                    )
                    if instruction_override is not None:
                        voice_prompt_version = ELICITATION_PROMPT_VERSION
                    caller_guidelines = get_user_guidelines(
                        True, agent_directed, persona=context.persona
                    )
                pairs.append(
                    PromptPair(
                        voice_instruction_prompt_version=voice_prompt_version,
                        language=language,
                        display_name=pack.display_name,
                        domain=domain,
                        persona_id=persona_id,
                        task_id=task.id,
                        canonical_task_id=context.canonical_task_id,
                        config=config,
                        language_context=context,
                        agent_voice=voice,
                        agent_gender=gender,
                        agent_gender_source="tau2.voice.voice_gender.VOICE_GENDER",
                        agent_prompt=build_audio_native_system_prompt(
                            agent_domain_policy(
                                environment,
                                agent_language_guidance(
                                    context,
                                    agent_voice_gender=call_context(
                                        config, context
                                    ).agent_voice_gender,
                                ),
                                language=context.language,
                            ),
                            provider=config.audio_native_config.provider,
                            use_xml_prompt=config.audio_native_config.use_xml_prompt,
                            agent_instruction_override=instruction_override,
                        ),
                        customer_prompt=build_user_system_prompt(
                            str(prompt_task.user_scenario),
                            persona_config=context.persona,
                            mode="voice",
                            guidelines=caller_guidelines,
                        ),
                        initial_agent_message=greeting,
                    )
                )
    root = Path(__file__).resolve().parents[1]
    paths = [
        "multilingual/prompt_review.py",
        "runner/presets.py",
        "config.py",
        "multilingual/runtime.py",
        "data_model/language.py",
        "user/guidelines.py",
        "backchannel.py",
        "runner/build.py",
        "agent/discrete_time_audio_native_agent.py",
        "user/user_simulator.py",
        "user/user_simulator_streaming.py",
        "user_simulation_voice_presets.py",
        "data_model/persona.py",
        "data_model/simulation.py",
        "data_model/run_config.py",
        "voice/voice_gender.py",
    ]
    return PromptReview(
        source_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths
        },
        pairs=pairs,
    )


def write_review(review: PromptReview, output: Path) -> None:
    """Write typed JSON/schema, plain text and a self-contained browser review."""
    output.mkdir(parents=True, exist_ok=True)
    prompts = output / "prompts"
    prompts.mkdir(exist_ok=True)
    (output / "review.json").write_text(
        review.model_dump_json(indent=2), encoding="utf-8"
    )
    (output / "schema.json").write_text(
        json.dumps(PromptReview.model_json_schema(), indent=2), encoding="utf-8"
    )
    options, articles = [], []
    for index, pair in enumerate(review.pairs):
        key = f"pair-{index:03d}"
        persona_label = pair.persona_id.replace("_", " ").title()
        label = f"{pair.display_name} · {pair.domain} · {persona_label}"
        options.append(f'<option value="{key}">{html.escape(label)}</option>')
        links = []
        sections = []
        for role, prompt in (
            ("agent", pair.agent_prompt),
            ("customer", pair.customer_prompt),
        ):
            name = f"{key}-{role}.txt"
            (prompts / name).write_text(prompt, encoding="utf-8")
            links.append(f'<a href="prompts/{name}">{role.title()} text</a>')
            sections.append(
                f"<h3>{role.title()} system prompt</h3><pre>{html.escape(prompt)}</pre>"
            )
        facts = f"Task: {pair.task_id} | Canonical: {pair.canonical_task_id}\nCaller: {pair.language_context.persona.gender} | Agent: {pair.config.audio_native_config.provider} / {pair.agent_voice} / {pair.agent_gender}\nGender source: {pair.agent_gender_source}\nPack: {pair.language_context.pack_revision} | Prompt: {pair.language_context.prompt_version}\nVoice instructions: {pair.voice_instruction_prompt_version} | Customer: {pair.config.llm_user} {pair.config.llm_args_user}"
        greeting = (
            f"<h3>Initial agent greeting</h3><pre>{html.escape(pair.initial_agent_message)}</pre>"
            if pair.initial_agent_message
            else ""
        )
        articles.append(
            f'<article id="{key}" hidden><h2>{html.escape(label)}</h2><pre class="facts">{html.escape(facts)}</pre><p>{" · ".join(links)}</p>{greeting}{"".join(sections)}<details><summary>Complete configuration and provenance</summary><pre>{html.escape(pair.model_dump_json(indent=2, exclude={"agent_prompt", "customer_prompt"}))}</pre></details></article>'
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Voice prompt review</title>
<style>body{font:16px system-ui;margin:0;background:#f5f6f8;color:#172033}header{position:sticky;top:0;background:white;padding:20px 28px;border-bottom:1px solid #ccd2da}main{max-width:1300px;margin:auto;padding:28px}h1{font-size:24px;margin:0 0 12px}select{font:inherit;max-width:100%;padding:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:white;padding:22px;border:1px solid #dce0e7;border-radius:8px}.facts{background:#edf2f8}a{color:#185bb5}button{font:inherit;padding:9px}h3{margin-top:30px}</style>
<header><h1>Voice · Full prompt review</h1><label for="pair">Language · domain · persona </label><select id="pair">OPTIONS</select> <button id="previous">Previous</button> <button id="next">Next</button></header>
<main><p>COUNT prompt pairs rendered offline with the shared runtime prompt functions. Each customer prompt includes a real task selected by the ordinary persona assignment. <a href="review.json">All prompts and provenance (JSON)</a> · <a href="schema.json">Artifact schema</a></p>ARTICLES</main>
<script>const menu=document.getElementById('pair');function show(){document.querySelectorAll('article').forEach(a=>a.hidden=a.id!==menu.value);history.replaceState(null,'','#'+menu.value);window.scrollTo(0,0)}menu.addEventListener('change',show);document.getElementById('previous').onclick=()=>{menu.selectedIndex=Math.max(0,menu.selectedIndex-1);show()};document.getElementById('next').onclick=()=>{menu.selectedIndex=Math.min(menu.options.length-1,menu.selectedIndex+1);show()};if([...menu.options].some(o=>o.value===location.hash.slice(1)))menu.value=location.hash.slice(1);show();window.addEventListener('load',()=>window.scrollTo(0,0));</script></html>"""
    page = (
        page.replace("OPTIONS", "".join(options))
        .replace("COUNT", str(len(review.pairs)))
        .replace("ARTICLES", "".join(articles))
    )
    (output / "index.html").write_text(page, encoding="utf-8")

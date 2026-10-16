"""Reviewable pack authoring and immutable publication using runtime-owned schemas."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from tau2.config import SCENARIO_LANGUAGE
from tau2.data_model.language import (
    LanguagePack,
    PackVoice,
    cross_lingual,
)
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.multilingual.factory.prompt_rules import (
    AUTHORING_RULES,
    AUTHORING_RULES_VERSION,
    REQUESTED_SPELLING_SECTION,
)
from tau2.multilingual.factory.schema import (
    AddDomainRequest,
    DraftArtifact,
    DraftPack,
    DraftRequest,
    PackContent,
    PublicationProvenance,
    StagePackRequest,
    validate_content,
)
from tau2.multilingual.factory.task_state import initialize_task_state
from tau2.user.guidelines import GUIDELINE_TEMPLATE_VERSION
from tau2.utils.artifacts import ReviewApproval, content_digest, write_artifact
from tau2.utils.brevity import BACKCHANNEL_INVENTORY, NON_DIRECTED_INVENTORY
from tau2.utils.llm_utils import extract_json_from_llm_response, generate
from tau2.voice_config import VOCAL_TICS

if TYPE_CHECKING:
    from tau2.multilingual.factory.voice_assets import VoiceAssetsArtifact


DRAFT_PROMPT_VERSION = "public-pack-draft-v9"
DRAFT_CALL_NAME = "factory_draft"
DRAFT_REPAIR_CALL_NAME = "factory_draft_repair"
DRAFT_SYSTEM_PROMPT = (
    "You are the Tau-Voice language-pack drafting assistant. Produce one JSON object "
    "matching the supplied schema, in the requested language and registers."
)
# Filled with str.format; the template itself contains no literal braces.
DRAFT_USER_PROMPT_TEMPLATE = """Draft language/register content from this reviewed author form:
{author_form}
Use exactly its two persona names as keys and preserve their genders.
Return exactly this schema: {schema}
Every persona must supply gender_guidance and a voice section whose voice_examples
fill every required slot of the shared scaffold {template_version}, including
tools-only slots. The scenario language (the language of task instructions and
policies) is {scenario_language}. {language_directive_rule} The shared scaffold
owns the directive wording. Use reviewed
language-owned text in each slot; do not supply English template source strings,
replacement pairs or legacy fields. A nonempty localized agent_greeting that works
for both typed and spoken conversations is required.
Runtime semantics of fields:
- Instruction fields (agent_language_clause, user_guidelines, pragmatics_clauses,
  voice.pragmatics_clauses, locale, variety, agent_background_clause,
  language_directive.origin, gender_guidance) are meta-instructions: write them in
  English, naming the target language and script explicitly.
- A persona's pragmatics_clauses render in both typed and spoken conversations, so
  keep them mode-neutral. Behavior that only applies to spoken calls (phone
  openings and closings, listening idioms, overlapping or interrupting the agent,
  continuers while the agent talks) goes in voice.pragmatics_clauses, which
  renders in spoken calls only.
- Literal spoken fields (agent_greeting and the voice section's backchannel_phrases,
  non_directed_phrases and voice_examples) are said verbatim and use the target language.
- agent_language_clause is shared by every call: never mention persona names or IDs.
- Phrase inventories are drawn at random with no context, so each has a hard
  spoken-word budget: backchannel_phrases holds 1-{backchannel_max_phrases} pure
  listener acknowledgments of at most {backchannel_max_words} words each;
  non_directed_phrases holds at most {non_directed_max_phrases} glances away from
  the phone of at most {non_directed_max_words} words each. Count spoken words
  (scripts without spaces convert characters to words); never exceed a budget.
- voice.vocal_tics are fixed non-verbal audio tags owned by the runtime: return an
  empty list; never write filler words there.
Shared authoring contract ({rules_version}):
{rules}
Task instructions own identity, goals, emotion and cooperation.
Preserve ordinary conversational phrasing and standard verbosity. Voice IDs stay null.
Start spellout_section with this exact fixed section, then optional
native examples clearly labelled as requested spelling or requested anchors:
{spelling_section}
Do not introduce always-spell directives elsewhere. Do not invent task data,
annotation settings, acoustic settings or experimental domains. Domain guidance
is supplied separately by reviewed inputs. Produce JSON only.
"""


DRAFT_REPAIR_PROMPT = """The previous reply failed validation and was rejected.
Return one complete corrected JSON object for the same schema, fixing every listed
problem and keeping all other reviewed content unchanged. Produce JSON only.
Validation error:
{error}
Rejected reply:
{reply}
"""


def _language_directive_rule(language: str) -> str:
    if cross_lingual(language):
        return (
            f"This pack's language {language!r} differs from the scenario language, "
            "so every persona MUST supply language_directive.origin as an English "
            "description of where the caller is from."
        )
    return (
        f"This pack's language {language!r} is the scenario language, so personas "
        "must omit language_directive."
    )


def render_draft_prompt(form) -> str:
    """Render the fixed drafting prompt for one reviewed author form."""
    return DRAFT_USER_PROMPT_TEMPLATE.format(
        author_form=form.model_dump_json(),
        schema=json.dumps(PackContent.model_json_schema()),
        template_version=GUIDELINE_TEMPLATE_VERSION,
        scenario_language=f"English ({SCENARIO_LANGUAGE!r})",
        language_directive_rule=_language_directive_rule(form.identity.language),
        backchannel_max_phrases=BACKCHANNEL_INVENTORY.max_phrases,
        backchannel_max_words=BACKCHANNEL_INVENTORY.max_words,
        non_directed_max_phrases=NON_DIRECTED_INVENTORY.max_phrases,
        non_directed_max_words=NON_DIRECTED_INVENTORY.max_words,
        rules_version=AUTHORING_RULES_VERSION,
        rules=AUTHORING_RULES,
        spelling_section=REQUESTED_SPELLING_SECTION,
    )


def _with_retail_name_facts(task, tools, name_order):
    """Append the structured caller's given/family-name facts to a retail task."""
    from tau2.multilingual.names import clarify_retail_name, resolve_retail_caller

    caller = resolve_retail_caller(task, tools, task, tools, {})
    return clarify_retail_name(
        task, caller_id=caller.user_id, caller=caller, name_order=name_order
    )


def _base_tasks(bundle, name_order):
    """Add required canonical caller facts as part of a localized domain build."""
    if bundle.domain != "retail":
        return bundle.tasks
    from tau2.domains.retail.environment import get_environment

    return [
        _with_retail_name_facts(
            task, initialize_task_state(get_environment(), task).tools, name_order
        )
        for task in bundle.tasks
    ]


def _build_domains(bundles, identity, output):
    from tau2.multilingual.factory.localization_publish import validate_regional_source
    from tau2.multilingual.factory.regional_build import (
        GenerateLocalization,
        build_localized_domain,
    )
    from tau2.utils import DATA_DIR

    built = {}
    # Check all domain requests before any generator is invoked.
    for bundle in bundles:
        localization = bundle.localization
        if localization is None:
            continue
        source = (
            localization.request.source
            if isinstance(localization, GenerateLocalization)
            else localization.source
        )
        settings = (
            localization.request.settings
            if isinstance(localization, GenerateLocalization)
            else localization.mapping.settings
        )
        validate_regional_source(source, DATA_DIR)
        if source.domain != bundle.domain:
            raise ValueError("localization source differs from domain bundle")
        if (settings.language, settings.name_order) != (
            identity.language,
            identity.name_order,
        ):
            raise ValueError(
                "localization language/name order differs from language pack"
            )
        if isinstance(localization, GenerateLocalization) and (
            source.domain == "airline"
        ) != (localization.airport_generation is not None):
            raise ValueError(
                "airport generation settings are required only for airline"
            )
    for bundle in bundles:
        if bundle.localization is None:
            built[bundle.domain] = bundle
            continue
        tasks = _base_tasks(bundle, identity.name_order)
        regional, evidence = build_localized_domain(
            bundle.localization,
            bundle.metadata,
            output / bundle.domain,
            DATA_DIR,
        )
        built[bundle.domain] = type(bundle).model_validate(
            {
                **bundle.model_dump(),
                "localization": None,
                "tasks": tasks,
                "regional_bundle": regional.model_dump(),
                "localization_evidence": evidence.model_dump(),
            }
        )
    return built


def _write_draft(result, output):
    from tau2.multilingual.factory.regional_build import write_content_review

    validate_draft(result)
    if (
        output.exists()
        and DraftArtifact.model_validate_json(output.read_text()) != result
    ):
        raise FileExistsError("draft already exists; use a new output path")
    write_content_review(result, output.with_suffix(".review"))
    write_artifact(result, output)
    return result


def _draft_pack(request: DraftRequest, bundles, raw: str) -> DraftPack:
    """Validate one generated reply into a draft pack; any defect is a ValueError."""
    content = PackContent.model_validate_json(extract_json_from_llm_response(raw))
    if {name: p.gender for name, p in content.personas.items()} != {
        p.name: p.gender for p in request.form.personas
    }:
        raise ValueError("draft changed reviewed persona names or genders")
    generated = content.model_dump()
    for persona in generated["personas"].values():
        persona["voice"]["vocal_tics"] = list(VOCAL_TICS)
    pack = DraftPack(
        **{k: v for k, v in generated.items() if k != "spellout_section"},
        language=request.form.identity.language,
        display_name=request.form.identity.display_name,
        name_order=request.form.identity.name_order,
        revision=request.revision,
        voice=PackVoice(
            spellout_section=content.spellout_section,
            speech_insert_events_per_minute=request.speech_insert_events_per_minute,
            backchannel_level=request.backchannel_level,
        ),
        domains={name: bundle.binding() for name, bundle in bundles.items()},
    )
    validate_content(pack)
    return pack


def draft(request: DraftRequest, output: Path) -> DraftArtifact:
    """Generate (with at most one repair) a typed, reviewable candidate; reject stale resumes."""
    request = DraftRequest.model_validate(request.model_dump())
    digest = content_digest(request)
    if output.exists():
        previous = DraftArtifact.model_validate_json(output.read_text())
        if (
            previous.input_sha256 == digest
            and previous.prompt_version == DRAFT_PROMPT_VERSION
        ):
            return _write_draft(previous, output)
        raise FileExistsError("authoring inputs changed; use a new output path")
    if len({d.domain for d in request.domains}) != len(request.domains):
        raise ValueError("duplicate domain inputs")
    bundles = _build_domains(
        request.domains, request.form.identity, output.with_suffix(".localization")
    )
    messages = [
        SystemMessage(role="system", content=DRAFT_SYSTEM_PROMPT),
        UserMessage(role="user", content=render_draft_prompt(request.form)),
    ]
    reply = generate(
        model=request.generation.model,
        reasoning_effort=request.generation.reasoning_effort,
        messages=messages,
        call_name=DRAFT_CALL_NAME,
    )
    try:
        pack = _draft_pack(request, bundles, reply.content or "")
    except ValueError as error:
        # One bounded repair call with the validation error; a second failure
        # surfaces to the author instead of looping on paid calls.
        repair = generate(
            model=request.generation.model,
            reasoning_effort=request.generation.reasoning_effort,
            messages=messages
            + [
                UserMessage(
                    role="user",
                    content=DRAFT_REPAIR_PROMPT.format(
                        error=error, reply=reply.content or ""
                    ),
                )
            ],
            call_name=DRAFT_REPAIR_CALL_NAME,
        )
        pack = _draft_pack(request, bundles, repair.content or "")
    result = DraftArtifact(
        pack=pack,
        domains=bundles,
        generation=request.generation,
        prompt_version=DRAFT_PROMPT_VERSION,
        input_sha256=digest,
    )
    return _write_draft(result, output)


def add_domain(request: AddDomainRequest, output: Path) -> DraftArtifact:
    """Preserve language/personas while binding another supported domain."""
    request = AddDomainRequest.model_validate(request.model_dump())
    if request.domain.domain in request.draft.domains:
        raise ValueError("domain already exists; rebuild a reviewed revision instead")
    if request.revision == request.draft.pack.revision:
        raise ValueError("adding a domain requires a new pack revision")
    bundle = _build_domains(
        [request.domain], request.draft.pack, output.with_suffix(".localization")
    )[request.domain.domain]
    data = request.draft.model_dump()
    name = bundle.domain
    data["domains"][name] = bundle.model_dump()
    data["pack"]["domains"][name] = bundle.binding().model_dump()
    data["pack"]["revision"] = request.revision
    data["parent_sha256"] = content_digest(request.draft)
    data["input_sha256"] = content_digest(request)
    result = DraftArtifact.model_validate(data)
    return _write_draft(result, output)


def validate_draft(staged: DraftArtifact) -> DraftArtifact:
    """Validate the complete draft against real offline domain initialization."""
    from tau2.domains.airline.environment import get_environment as airline
    from tau2.domains.intake.environment import get_environment as intake
    from tau2.domains.retail.environment import get_environment as retail
    from tau2.domains.telecom.environment import get_environment as telecom
    from tau2.multilingual.factory.identity_validation import (
        validate_identity_consistency,
    )
    from tau2.multilingual.scenario_dispatch import scenario_environment_kwargs

    staged = DraftArtifact.model_validate(staged.model_dump())
    from tau2.domains.banking_knowledge.environment import get_environment as banking

    def banking_offline(**kwargs):
        return banking(retrieval_variant="no_knowledge", **kwargs)

    factories = {
        "airline": airline,
        "retail": retail,
        "telecom": telecom,
        "banking_knowledge": banking_offline,
        "intake": intake,
    }
    for domain, bundle in staged.domains.items():
        if bundle.localization_evidence is not None:
            from tau2.multilingual.factory.regional_build import (
                validate_localized_domain,
            )
            from tau2.utils import DATA_DIR

            validate_localized_domain(
                bundle.regional_bundle,
                bundle.localization_evidence,
                bundle.metadata,
                DATA_DIR,
            )
        for task in bundle.tasks:
            environment = initialize_task_state(factories[domain](), task)
            if task.initial_state and task.initial_state.initialization_data:
                validate_identity_consistency(task, domain)
            if domain == "retail" and task != _with_retail_name_facts(
                task, environment.tools, staged.pack.name_order
            ):
                raise ValueError("retail task is missing explicit name facts")
        if bundle.regional_bundle is not None:
            # Regional tasks initialize against their complete published database,
            # while the pack's separate base tasks retain ordinary initialization.
            for task in bundle.regional_bundle.tasks:
                initialize_task_state(
                    factories[domain](
                        **scenario_environment_kwargs(bundle.regional_bundle)
                    ),
                    task,
                )
    return staged


def finalize(
    staged: DraftArtifact,
    assets: "VoiceAssetsArtifact | None",
    approval: ReviewApproval | None,
    output: Path,
    *,
    assets_root: Path | None = None,
) -> bool:
    """Publish only fully reviewed materialized content and auditioned voices.

    Files are assembled beside the destination and renamed once. No partial pack
    or replacement of an existing bundle is permitted, including on a failed check.
    """
    from tau2.multilingual.factory.voice_assets import (
        VoiceAssetsArtifact,
        require_listening_approval,
        validate_voices,
    )

    staged = validate_draft(staged)
    if assets is None or approval is None:
        raise ValueError("validated voice assets and audition review are required")
    assets = VoiceAssetsArtifact.model_validate(assets.model_dump())
    assets.content_review.require(staged)
    require_listening_approval(approval, assets)
    if assets.draft_sha256 != content_digest(staged) or set(assets.voices) != set(
        staged.pack.personas
    ):
        raise ValueError("voice assets do not match current draft")
    if assets_root is None:
        raise ValueError("voice asset directory is required")
    validate_voices(assets, assets_root)
    data = staged.pack.model_dump()
    for name, voice in assets.voices.items():
        data["personas"][name]["voice"]["voice_id"] = voice.voice.voice_id
    pack = LanguagePack.model_validate(data)
    files = {
        "pack.yaml": yaml.safe_dump(
            pack.model_dump(mode="json"), allow_unicode=True, sort_keys=False
        ).encode()
    }
    for name, bundle in staged.domains.items():
        binding = pack.domains[name]
        if binding.tasks_file is not None:
            files[binding.tasks_file] = (
                json.dumps(
                    [t.model_dump(mode="json") for t in bundle.tasks],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode()
        files[binding.task_metadata_file] = (
            json.dumps(
                {k: v.model_dump(mode="json") for k, v in bundle.metadata.items()},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode()
        if bundle.regional_bundle is not None:
            files[binding.regional_bundle_file] = (
                bundle.regional_bundle.model_dump_json(indent=2) + "\n"
            ).encode()
    provenance = PublicationProvenance(
        draft_sha256=content_digest(staged),
        assets_sha256=content_digest(assets),
        files={
            name: hashlib.sha256(value).hexdigest() for name, value in files.items()
        },
    )
    files["provenance.json"] = (provenance.model_dump_json(indent=2) + "\n").encode()
    if output.exists():
        actual = {
            p.relative_to(output).as_posix(): p.read_bytes()
            for p in output.rglob("*")
            if p.is_file()
        }
        if actual == files and not any(p.is_symlink() for p in output.rglob("*")):
            return False
        raise FileExistsError("published bundle exists; use a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".factory-", dir=output.parent))
    try:
        for name, value in files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
        # mkdir reserves the publication path against concurrent factory writers.
        # A rename onto an empty reserved directory is atomic on supported platforms.
        output.mkdir()
        try:
            temporary.rename(output)
        except BaseException:
            output.rmdir()
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return True


def stage_pack(request: StagePackRequest, output: Path) -> DraftArtifact:
    """Stage runtime-owned content for add-domain without regenerating personas."""
    request = StagePackRequest.model_validate(request.model_dump())
    if request.revision == request.pack.revision:
        raise ValueError("staging an existing pack requires a new revision")
    domains = {bundle.domain: bundle for bundle in request.domains}
    if len(domains) != len(request.domains) or set(domains) != set(
        request.pack.domains
    ):
        raise ValueError("staged domain bundles must exactly match published bindings")
    for name, bundle in domains.items():
        # Existing runtime packs may use any safe relative filenames. The new
        # revision owns its filenames; domain semantics and source content match.
        omitted = {"tasks_file", "task_metadata_file", "regional_bundle_file"}
        if bundle.localization is None and (bundle.regional_bundle is not None) != (
            request.pack.domains[name].regional_bundle_file is not None
        ):
            raise ValueError("staging must retain the published regional scenario")
        if bundle.binding().model_dump(exclude=omitted) != request.pack.domains[
            name
        ].model_dump(exclude=omitted):
            raise ValueError("staged domain content differs from published binding")
    domains = _build_domains(
        request.domains, request.pack, output.with_suffix(".localization")
    )
    data = request.pack.model_dump()
    data["revision"] = request.revision
    data["domains"] = {
        name: bundle.binding().model_dump() for name, bundle in domains.items()
    }
    for persona in data["personas"].values():
        persona["voice"]["voice_id"] = None
    result = DraftArtifact(
        pack=DraftPack.model_validate(data),
        domains=domains,
        generation=None,
        prompt_version="reviewed-runtime-import-v3",
        input_sha256=content_digest(request),
        parent_sha256=content_digest(request.pack),
    )
    return _write_draft(result, output)

import re
from typing import Generic, Optional, Tuple, TypeVar

from loguru import logger

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.persona import PersonaConfig
from tau2.environment.tool import Tool
from tau2.user.user_simulator_base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    HalfDuplexUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)
from tau2.utils import DATA_DIR
from tau2.utils.llm_utils import generate

GLOBAL_USER_SIM_GUIDELINES_DIR = DATA_DIR / "tau2" / "user_simulator"

# The language every user-simulator prompt scaffold is written in (guidelines,
# directive/reminder templates, English-mode scenarios). A language pack with
# this code — ``data/tau2/multilingual/en/pack.yaml`` — describes personas who
# speak the prompt's own language; see ``get_target_language``.
ENGLISH_LANGUAGE_CODE = "en"


GLOBAL_USER_SIM_GUIDELINES_PATH = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines.md"
)

GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_tools.md"
)

GLOBAL_USER_SIM_GUIDELINES_PATH_VOICE = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_voice.md"
)

GLOBAL_USER_SIM_GUIDELINES_PATH_VOICE_TOOLS = (
    GLOBAL_USER_SIM_GUIDELINES_DIR / "simulation_guidelines_voice_tools.md"
)


def get_global_user_sim_guidelines(use_tools: bool = False) -> str:
    """
    Get the global user simulator guidelines.

    Args:
        use_tools: Whether to use the tools guidelines.
    Returns:
        The global user simulator guidelines.
    """
    if use_tools:
        path = GLOBAL_USER_SIM_GUIDELINES_PATH_TOOLS
    else:
        path = GLOBAL_USER_SIM_GUIDELINES_PATH
    with open(path, "r") as fp:
        return fp.read()


def get_global_user_sim_guidelines_voice(use_tools: bool = False) -> str:
    """
    Get the global user simulator guidelines for voice mode.

    Always the ENGLISH guidelines, for every persona: what makes a
    multilingual run multilingual is the fixed speak-the-target-language
    directive (see :func:`get_target_language_directive`) plus the pack's
    localization block, not a translated prompt. The packs' localized
    guidelines files belong to the retired native prompt-language arm
    (deleted 2026-08-03; recoverable from git history).

    Args:
        use_tools: Whether to use the tools guidelines.
    Returns:
        The global user simulator guidelines for voice mode.
    """
    if use_tools:
        path = GLOBAL_USER_SIM_GUIDELINES_PATH_VOICE_TOOLS
    else:
        path = GLOBAL_USER_SIM_GUIDELINES_PATH_VOICE
    with open(path, "r") as fp:
        return fp.read()


SYSTEM_PROMPT = """
{global_user_sim_guidelines_with_persona}

<scenario>
{instructions}
</scenario>
""".strip()

# The fixed English 'Speaking Special Characters and Numbers' section. It
# fills the <SPOKEN_VALUES_SPELLOUT> slot of the English voice guidelines
# whenever no language pack provides a native rendering (plain English
# personas, packs without spoken-value data). Persona-neutral by design: the
# worked examples show pure readout mechanics — no hesitation fillers, no
# register dressing (tone belongs to the task persona).
ENGLISH_SPELLOUT_SECTION = """\
## Speaking Special Characters and Numbers
When providing emails, user IDs, or any text with special characters, SPELL THEM OUT as you would on a phone:
- @ = "at"
- . = "dot"
- _ = "underscore"
- - = "dash" or "hyphen"
- / = "slash"
- \\ = "backslash"

Comma convention: emails and website addresses are read as a continuous flow of words — never insert commas or pauses between the spoken parts ("john underscore doe at gmail dot com", NOT "john, underscore, doe, at, gmail, dot, com"). When a value IS spelled out character by character (letters, digits, codes), ALWAYS separate the characters with comma and space:
- Numbers: "one, two, three" NOT "one two three"
- Letters: "J, O, H, N" NOT "J O H N" or "JOHN"
- Mixed: "A, B, one, two, three" NOT "AB123"

Examples:
- Email: "it's john underscore doe at gmail dot com"
- User ID: "my user ID is user dash one, two, three"
- Phone: "it's five, five, five, dash, one, two, three, four"
- Spelling name: "that's J, O, H, N... Smith"
- Account number: "my account is A, B, C, one, two, three, four"
- Website: "I was on your site, www dot example dot com slash support\""""

_PERSONA_SLOT_RE = re.compile(r"^<PERSONA_GUIDELINES>$", re.MULTILINE)

_SPELLOUT_SLOT_RE = re.compile(r"^<SPOKEN_VALUES_SPELLOUT>$", re.MULTILINE)

# Inline example slots of the ENGLISH guidelines: the instructional sentence
# around each slot is fixed English prose; the slot renders the example
# utterances (target-language from pack data, else the fixed English default).
_EXAMPLE_SLOT_RE = re.compile(r"<EXAMPLE:([a-z0-9_]+)>")

# The first section heading of the guidelines — the target-language directive
# is inserted right before it (after the title/role-framing intro) so the
# language mandate is visible before any behavioral instruction or example.
_FIRST_SECTION_RE = re.compile(r"^## ", re.MULTILINE)


def substitute_example_slots(
    guidelines: str, language: Optional[str], gender: Optional[str] = None
) -> str:
    """Fill the inline ``<EXAMPLE:kind>`` slots of the voice guidelines.

    Each slot names a kind from the closed guideline-example catalog
    (``tau2.multilingual.localization_catalog``; an unknown kind raises — a
    typo in the guidelines file must never render a bare slot). A language
    pack carrying the kind renders its native utterances (with the fixed
    English joiners); otherwise the kind's fixed English default renders —
    English personas reproduce today's guidelines text exactly.

    ``gender`` is the speaking persona's gender tag (``persona_config.tags``).
    Where the pack marks a kind as gendered LANGUAGE MECHANICS — Japanese
    first-person pronouns, sentence-final particles, softeners — a male
    speaker renders the male realization; everyone else renders the shared
    palette. This is grammar/usage selection, not affect: the persona still
    owns attitude.
    """
    from tau2.multilingual.localization_catalog import (
        get_guideline_example_kind,
        render_guideline_example,
    )
    from tau2.multilingual.registry import get_guideline_example_items

    def _render(match: re.Match) -> str:
        kind_def = get_guideline_example_kind(match.group(1))
        items = get_guideline_example_items(language, kind_def.kind_id, gender)
        if items is None:
            return render_guideline_example(
                kind_def, kind_def.english_items, native=False
            )
        return render_guideline_example(kind_def, items, native=True)

    return _EXAMPLE_SLOT_RE.sub(_render, guidelines)


def insert_language_directive(guidelines: str, directive: Optional[str]) -> str:
    """Insert the target-language directive near the TOP of the guidelines.

    The directive lands right after the title/role-framing intro — before the
    first ``## `` section heading — so the model knows from the start why
    every example is in the target language (appending it at the bottom left
    the language mandate after thousands of tokens of English instruction).
    None passes the guidelines through; a file without section headings gets
    the directive appended.
    """
    if not directive:
        return guidelines
    match = _FIRST_SECTION_RE.search(guidelines)
    if match is None:
        return f"{guidelines}\n\n{directive}"
    start = match.start()
    return f"{guidelines[:start]}{directive}\n\n{guidelines[start:]}"


def get_end_of_prompt_reminder(
    persona_config: Optional[PersonaConfig],
) -> Optional[str]:
    """The terse end-of-prompt language reminder for this sim, or None.

    A recency anchor rendered after the scenario block for personas whose
    speech is NOT in the prompt's own language (see
    :func:`get_target_language`), from the fixed versioned English template
    (``tau2.multilingual.english_prompts``). English-speaking personas get
    None — no directive, no reminder; the special-token contract they still
    need is carried by the guidelines' own "## Task Completion" section and
    the persona-section note.
    """
    language = get_target_language(persona_config)
    if language is None:
        return None
    from tau2.multilingual.english_prompts import render_target_language_reminder

    return render_target_language_reminder(language)


def substitute_spellout_slot(guidelines: str, language: Optional[str]) -> str:
    """Fill the ``<SPOKEN_VALUES_SPELLOUT>`` slot of the voice guidelines.

    The slot exists only in the ENGLISH voice guidelines — which is what every
    run now reads. A language pack with spoken-localization data renders the
    section natively — English instructional prose, native
    symbol names and worked examples; otherwise the fixed English section
    applies unchanged.
    """
    from tau2.multilingual.registry import get_spellout_section

    section = get_spellout_section(language) or ENGLISH_SPELLOUT_SECTION
    return _SPELLOUT_SLOT_RE.sub(lambda _match: section, guidelines)


def substitute_persona_slot(guidelines: str, persona_guidelines: str) -> str:
    """Fill the ``<PERSONA_GUIDELINES>`` slot line with the persona block.

    Only a line consisting of exactly the placeholder is a slot. Line-anchored
    because the packs' localized guideline files also MENTION the placeholder
    in prose (e.g. es: "...vienen del `<PERSONA_GUIDELINES>` de abajo..."), and
    a plain ``str.replace`` spliced the whole persona + localization block into
    that sentence too — a bug on the retired native arm, which is the only
    thing that ever read those files. Kept line-anchored regardless: the rule
    is that a slot is a line, not a mention.
    """
    return _PERSONA_SLOT_RE.sub(lambda _match: persona_guidelines, guidelines)


def get_target_language(persona_config: Optional[PersonaConfig]) -> Optional[str]:
    """The language this persona SPEAKS when the prompt is not in it, else None.

    The single gate on both target-language prompt blocks — the
    ``## LANGUAGE OF THE CALL (MANDATORY)`` directive and the
    ``## FINAL REMINDER`` recency anchor. Both exist for one reason: to stop a
    persona reading an English prompt from answering the agent in English.
    Two kinds of persona have nothing to redirect and get None:

    - a plain ``PersonaConfig``, which carries no ``language`` at all (the
      English text/voice runs that predate the language packs); and
    - a persona from the ENGLISH language pack
      (``data/tau2/multilingual/en/pack.yaml``). It is a
      ``MultilingualPersonaConfig`` like every other pack persona and so
      carries ``language='en'``, but the language it speaks is the language
      the prompt is already written in. Rendering the blocks for it produced
      self-contradictory instructions — "every word you speak to the agent
      MUST be in English … never in English" — plus native-orthography and
      code-switching rules addressed to a non-English speaker.
    """
    language = getattr(persona_config, "language", None)
    if language == ENGLISH_LANGUAGE_CODE:
        return None
    return language


def get_target_language_directive(
    persona_config: Optional[PersonaConfig],
) -> Optional[str]:
    """The speak-the-target-language directive for this sim, or None.

    Applies to a persona that speaks a language the prompt is NOT written in
    (see :func:`get_target_language`): the English guidelines + English
    scenario are paired with a fixed, versioned block directing that every
    word SPOKEN to the agent is in the target language (see
    ``tau2.multilingual.english_prompts``). English-speaking personas —
    including the English language pack's — get None: their prompts need no
    directive.

    The directive (template v3+) names the speaker's origin and variety, so
    the persona's own ``locale`` parameterizes the render — the same resolved
    persona whose clauses elaborate that variety further down the prompt.
    """
    language = get_target_language(persona_config)
    if language is None:
        return None
    from tau2.multilingual.english_prompts import render_target_language_directive

    return render_target_language_directive(
        language, getattr(persona_config, "locale", None)
    )


UserStateType = TypeVar("UserStateType", bound="UserState")


class UserSimulator(
    LLMConfigMixin, HalfDuplexUser[UserStateType], Generic[UserStateType]
):
    """A half-duplex LLM-based user simulator for turn-based conversations.

    The runtime persona_config adds additional behavioral guidelines on top of the global
    and task-specific settings.
    Note: User behavior/persona is controlled in THREE places, and they need to be consistent / non-overlapping.
    1. Global simulation guidelines (data/tau2/user_simulator/*.md) - Base behavior for all users
    2. Task-specific persona (UserScenario.persona field) - Baked into task JSON at creation time
    3. Runtime persona config (persona_config parameter) - Configurable at simulation time
    """

    def __init__(
        self,
        llm: str,
        instructions: Optional[str] = None,
        tools: Optional[list[Tool]] = None,
        llm_args: Optional[dict] = None,
        persona_config: Optional[
            PersonaConfig
        ] = None,  # TODO: Should this be pushed to the base class?
        domain: Optional[str] = None,
        input_style_directive: Optional[str] = None,
        entity_noise=None,
    ):
        super().__init__(
            instructions=instructions,
            tools=tools,
            llm=llm,
            llm_args=llm_args,
        )
        self.persona_config = persona_config or PersonaConfig()
        # The run's benchmark domain; selects which of a language pack's
        # per-domain glossaries renders into the localization section. None
        # (non-multilingual runs) renders no glossary.
        self.domain = domain
        # Pre-rendered text input-style directive (romanized / diacritic-free
        # / code-mixed typing; see tau2.multilingual.text_input_catalog),
        # resolved and VALIDATED by the builder against the run's language
        # pack. None (the native_script default arm) renders nothing.
        self.input_style_directive = input_style_directive
        # The task's deterministic entity noise plan (a TextNoiseInfo built by
        # tau2.multilingual.text_noise), set by the builder on noise-armed
        # runs. First conveyance of each pinned entity is substituted with
        # its corrupted form; the orchestrator records the plan on the
        # SimulationRun. None (the default) leaves messages untouched. Typed
        # loosely to avoid importing the noise module into the simulator.
        self.entity_noise = entity_noise
        # Set by the builder for language-pack runs so the completed
        # SimulationRun records the active language/persona (the signal
        # get_simulation_language_info reads for nativeness scoring). None for
        # English/non-multilingual text runs. Typed loosely to avoid importing
        # the voice data model into the text user simulator.
        self.speech_environment = None

    @property
    def global_simulation_guidelines(self) -> str:
        """
        The simulation guidelines for the user simulator.

        EVERY persona gets the English guidelines — the target language is
        directed by the fixed directive block (see
        ``get_target_language_directive``), not by translating the prompt.
        Mirrors the voice user simulator.
        """
        return get_global_user_sim_guidelines(use_tools=self.tools is not None)

    @property
    def system_prompt(self) -> str:
        """
        The system prompt for the user simulator.
        """
        if self.instructions is None:
            logger.warning("No instructions provided for user simulator")

        guidelines = self.global_simulation_guidelines

        # The directive goes near the TOP (after the title/role-framing
        # intro): the language mandate must be visible before any behavioral
        # instruction. The input-style block (how the persona TYPES the
        # language) rides directly below it, because it supersedes the
        # directive's native-orthography bullet and must be read with it.
        directive_blocks = [
            get_target_language_directive(self.persona_config),
            self.input_style_directive,
        ]
        if self.entity_noise is not None:
            # Noise-armed runs pin conveyance format so first-conveyance
            # substitution matches typed values (fixed versioned clause).
            from tau2.multilingual.text_noise import TYPED_VALUES_CLAUSE

            directive_blocks.append(TYPED_VALUES_CLAUSE)
        directive = "\n\n".join(block for block in directive_blocks if block) or None
        guidelines = insert_language_directive(guidelines, directive)

        # Check if persona config adds any guidelines (register/pragmatics/
        # gender; the spoken-identity tts_voice_prompt is voice-design material
        # and never reaches the LLM prompt in any mode).
        # A language pack's localization block (date/numeral consistency rule,
        # domain-term glossary; text mode omits the spoken-only sub-sections)
        # rides the same <PERSONA_GUIDELINES> slot.
        from tau2.multilingual.registry import get_localization_guidelines

        language = getattr(self.persona_config, "language", None)
        sections = [
            self.persona_config.to_guidelines_text(mode="text"),
            get_localization_guidelines(language, mode="text", domain=self.domain),
        ]
        persona_guidelines = "\n\n".join(s for s in sections if s)
        if persona_guidelines:
            persona_guidelines = f"\n\n{persona_guidelines}\n"
        guidelines_with_persona = substitute_persona_slot(
            guidelines, persona_guidelines
        )

        system_prompt = SYSTEM_PROMPT.format(
            global_user_sim_guidelines_with_persona=guidelines_with_persona,
            instructions=self.instructions,
        )
        # A terse recency anchor AFTER the scenario block: the language
        # mandate and the special-token contract, restated in two lines.
        reminder = get_end_of_prompt_reminder(self.persona_config)
        if reminder:
            system_prompt = f"{system_prompt}\n\n{reminder}"
        return system_prompt

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserStateType:
        """
        Get the initial state of the user simulator.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history. User messages must be of type UserMessage, AssistantMessage, or ToolMessage to User."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Check if the message is a stop message.
        """
        if message.is_tool_call():
            return False
        # Audio-only messages (chunks) don't have text content
        if message.content is None:
            return False
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserStateType
    ) -> Tuple[UserMessage, UserStateType]:
        user_message = self._generate_next_message(message, state)
        # Updating state with response
        state.messages.append(user_message)
        return user_message, state

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserStateType
    ) -> UserMessage:
        """Get the response from the user simulator.

        Args:
            message: The assistant or tool message.
            state: The user simulator's state.

        Returns:
            The user message.
        """
        if isinstance(message, AssistantMessage) and message.is_audio:
            raise ValueError(
                "Assistant message cannot be audio. Use VoiceUserSimulator instead."
            )
        logger.debug(f"User responds to message: {message}")
        # Updating state with new message
        # Skip empty messages (e.g., empty chunks from streaming mode)
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif isinstance(message, ToolMessage):
            # ToolMessage always has content (tool response)
            state.messages.append(message)
        elif message.has_content() or message.is_tool_call():
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()

        # Generate response
        assistant_message = generate(
            model=self.llm,
            messages=messages,
            tools=self.tools,
            call_name="user_simulator_response",
            **self.llm_args,
        )

        user_response = assistant_message.content
        if (
            user_response
            and self.entity_noise is not None
            and self.entity_noise.entities
        ):
            # First conveyance of each pinned entity is substituted with its
            # precomputed corruption; later conveyances (after the corrupted
            # form has been sent once) pass through clean so the agent's
            # re-ask can repair. Content only — tool calls are never touched.
            from tau2.multilingual.text_noise import inject_first_conveyance_noise

            prior_user_texts = [
                m.content
                for m in state.messages
                if getattr(m, "role", None) == "user" and m.content
            ]
            user_response = inject_first_conveyance_noise(
                user_response, self.entity_noise, prior_user_texts
            )
        logger.debug(f"Response: {user_response}")

        user_message = UserMessage(
            role="user",
            content=user_response,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        # flip the requestor of the tool calls
        if assistant_message.tool_calls is not None:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )
        return user_message


class DummyUser(UserSimulator):
    """A dummy user to run a agent solo simulation."""

    def __init__(self):
        super().__init__(llm="dummy")

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        return UserState(messages=[], system_messages=[])

    def is_stop(cls, message: UserMessage) -> bool:
        raise NotImplementedError("DummyUser does not support stop messages")

    def set_seed(self, seed: int):
        pass

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> tuple[UserMessage, UserState]:
        raise NotImplementedError("DummyUser does not support generate_next_message")

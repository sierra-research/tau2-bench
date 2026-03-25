import random
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base.streaming import (
    AudioChunkingMixin,
    LinearizationStrategy,
    ListenerReactionDecision,
    StreamingState,
    TurnTakingAction,
    basic_turn_taking_policy,
    merge_homogeneous_chunks,
)
from tau2.agent.base.voice import VoiceMixin, VoiceState
from tau2.config import VOICE_USER_SIMULATOR_DECISION_MODEL
from tau2.data_model.audio import (
    PCM_SAMPLE_RATE,
    AudioData,
    AudioEncoding,
    audio_bytes_to_string,
)
from tau2.data_model.audio_effects import ChannelEffectsResult, EffectTimeline
from tau2.data_model.message import (
    AssistantMessage,
    EnvironmentMessage,
    Message,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.data_model.persona import InterruptTendency, PersonaConfig
from tau2.data_model.voice import VoiceSettings
from tau2.environment.tool import Tool
from tau2.user.user_simulator import SYSTEM_PROMPT, get_global_user_sim_guidelines_voice
from tau2.user.user_simulator_base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    FullDuplexVoiceUser,
    UserState,
    ValidUserInputMessage,
)
from tau2.utils.llm_utils import generate
from tau2.utils.utils import get_now
from tau2.voice.synthesis.audio_effects.effects import StreamingTelephonyConverter
from tau2.voice.synthesis.audio_effects.processor import (
    PendingEffectState,
    StreamingAudioEffectsMixin,
)
from tau2.voice.synthesis.audio_effects.scheduler import EffectScheduler
from tau2.voice.synthesis.audio_effects.speech_generator import (
    OutOfTurnSpeechGenerator,
    create_streaming_audio_generators,
)
from tau2.voice.utils.audio_tap import AudioTap
from tau2.voice_config import (
    BACKCHANNEL_PHRASES,
    resolve_background_noise_path,
    resolve_burst_noise_paths,
)

# Prompt for interruption decision.
# Note: Experimented with multiple sentences in the "Consider" section on 3 tasks. Below setting was best.
# 2nd best for first 2 sentences only (out of the 4 sentences), which would be less explicit guidance on when to interrupt.
INTERRUPTION_DECISION_PROMPT = """You are analyzing a conversation to decide if the user should interrupt the agent.

Conversation history (most recent at bottom):

<conversation_history>
{conversation_history}
</conversation_history>

The agent is CURRENTLY speaking (you can see their ongoing speech in the conversation above).

Based on the conversation so far, should the user interrupt the agent NOW?

Consider:
- Has the user heard enough to understand what the agent is asking or saying?
- Has the user heard enough to have a response, question, or correction ready?
- Did the agent just complete the sentence which has all the pertinent information the user was looking for?
- Do NOT repeatedly interrupt the agent if it has spoken only a few words (say less than 5 words).

Respond with ONLY "YES" if the user should interrupt now, or "NO" if they should keep listening.
"""
# - Is the agent currently in the middle of a sentence which seems to be "filler" sentence with not much additional information?


# Prompt for backchannel decision - CONTINUER version (separate from interruption).
# Continuers are brief sounds ("uh-huh", "mm-hmm") that signal attention without responding to content.
BACKCHANNEL_DECISION_PROMPT_CONTINUER = """You simulate a natural listener who occasionally says "uh-huh" or "mm-hmm" to show they're following along.

<conversation_history>
{conversation_history}
</conversation_history>

The agent is still speaking [CURRENTLY SPEAKING, INCOMPLETE]. Ignore the trailing incomplete word/phrase — focus only on the COMPLETE sentences delivered so far in the agent's current turn.

Continuers ("uh-huh", "mm-hmm", "yeah") are brief sounds that mean "I'm listening, keep going." They:
- Happen naturally during extended speech
- Show engagement without interrupting
- Are NOT responses to specific content — just signals of attention

Say YES if:
- The agent has completed at least 2 full, substantive sentences in their current turn
  (Short phrases like "Thanks for your patience" or "Let me check on that" don't count as substantive)
- The user hasn't spoken or backchanneled recently (check the last 3 exchanges for ANY user sound including "mm-hmm", "uh-huh", "okay")
- It would feel natural to briefly signal "I'm still here"

Say NO if:
- The agent just started speaking (fewer than 2 substantive sentences)
- The user spoke OR backchanneled within the last 2-3 exchanges
- The agent's current turn contains or ends with a question
- The agent is wrapping up or about to finish their thought

Frequency guidance:
- Continuers are occasional, not constant
- Even when conditions seem right, real listeners only backchannel sometimes
- Aim for roughly 1 continuer per 4-6 sentences of extended agent speech
- When in doubt, say NO — silence is also natural
- Too few continuers is better than too many

Examples:

AGENT: "Hi there! How can I hel [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (just started)

AGENT: "Thanks for your patience. [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (only 1 short sentence, not substantive enough)

AGENT: "Sure, I can help with that. First I'll need to verify your account. Could you provide your email or your name and zi [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (agent is asking a question)

AGENT: "No problem. We can use your name and zip code instead. Let me look that up for you. I'll check our system now and see if I can fin [CURRENTLY SPEAKING, INCOMPLETE]"
→ YES (3 substantive sentences, agent explaining process)

AGENT: "I found your order. It includes a keyboard, thermostat, and headphones. The order was delivered last Tuesday. Now for the exchange, we have a few opti [CURRENTLY SPEAKING, INCOMPLETE]"
→ YES (extended explanation with specific details)

[If user said "mm-hmm" 2 exchanges ago]
AGENT: "...and those are the available options. Now I'll need your input on which [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (user backchanneled recently, don't do it again so soon)

Respond with ONLY "YES" or "NO".
"""

# Alias for backward compatibility (default is continuer version)
BACKCHANNEL_DECISION_PROMPT = BACKCHANNEL_DECISION_PROMPT_CONTINUER

# Alternative prompt focused on acknowledgments (responding to specific content)
# Acknowledgments are content-driven responses ("ok", "got it") to specific information.
# NOTE: THIS IS HERE FOR EXPERIMENTAL PURPOSES. ACKNOWLEDGEMENT WIHTOUT ACCESS TO THE FULL USER INSTRUCTIONS IS RISKY!
BACKCHANNEL_DECISION_PROMPT_ACKNOWLEDGMENT = """You simulate a natural listener who briefly acknowledges when the speaker shares important information.

<conversation_history>
{conversation_history}
</conversation_history>

The agent is still speaking [CURRENTLY SPEAKING, INCOMPLETE]. Ignore the trailing incomplete word/phrase — focus only on the COMPLETE sentences delivered so far in the agent's current turn.

Acknowledgments ("ok", "got it", "I see", "right") are brief responses that confirm you understood something specific. They:
- React to meaningful information (confirmations, prices, instructions, findings)
- Show you processed what was said
- Are content-driven, not just signals of attention

Say YES if the agent just completed a statement containing:
- A confirmation ("I found your account", "The exchange has been processed")
- A price or cost ("That will be $249", "The refund is $13.46")
- Important information ("Your order includes...", "The status is delivered")
- An instruction or next step ("You'll receive an email with...")
AND the user hasn't acknowledged recently (last 2-3 exchanges)

Say NO if:
- The agent just started speaking (no complete meaningful statement yet)
- The user recently spoke or acknowledged
- The agent is asking a question
- The agent is giving routine/filler speech (greetings, "let me check", transitions)

Frequency guidance:
- Acknowledgments should feel purposeful, not automatic
- Only acknowledge genuinely important information
- Aim for 1 acknowledgment per significant piece of information shared
- When in doubt, say NO — not every statement needs acknowledgment
- Too few acknowledgments is better than too many

Examples:

AGENT: "Hi there! How can I hel [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (greeting, nothing to acknowledge)

AGENT: "Let me look that up for you. I'll check our system now and see if I can fin [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (routine transition, nothing substantive to acknowledge)

AGENT: "I found your account. Your order number is W2378156. It includes a keyboard and thermosta [CURRENTLY SPEAKING, INCOMPLETE]"
→ YES (agent confirmed finding account and shared order details — worth an "ok" or "got it")

AGENT: "The exchange has been processed. The refund of $13.46 will go to your card ending in 2478. You'll receive an ema [CURRENTLY SPEAKING, INCOMPLETE]"
→ YES (confirmation + price + next steps — natural moment for "ok" or "I see")

AGENT: "Could you please spell your last name letter by lett [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (agent is asking a question)

Respond with ONLY "YES" or "NO".
"""


class UserStreamingState(UserState, StreamingState[ValidUserInputMessage, UserMessage]):
    """
    State for user streaming.
    Extends UserState, StreamingState, and VoiceState with streaming-specific fields.
    """


class UserAudioStreamingState(
    UserState, StreamingState[ValidUserInputMessage, UserMessage], VoiceState
):
    """
    State for user audio streaming.
    Extends UserState, StreamingState, and VoiceState with audio streaming-specific fields.
    """

    model_config = {"arbitrary_types_allowed": True}

    user_utterance_count: int = 0
    elapsed_samples: int = 0
    effect_scheduler: Optional[EffectScheduler] = None
    out_of_turn_speech_generator: Optional[OutOfTurnSpeechGenerator] = None

    pending_effect: Optional[PendingEffectState] = None

    telephony_converter: Optional[StreamingTelephonyConverter] = None

    @property
    def info(self) -> str:
        """
        Get information about the state.
        """
        return (
            super().info
            + f", User utterance count: {self.user_utterance_count}, Elapsed samples: {self.elapsed_samples}"
        )


def user_interruption_policy(
    state: UserStreamingState,
    integration_ticks: int = 1,
) -> ListenerReactionDecision:
    """
    Decide whether the user should interrupt the agent while they're speaking.

    This is called when the agent is currently speaking and the user is listening.
    It determines if the user has heard enough to formulate a response and wants to interrupt.

    This function is reusable by both text and voice streaming user simulators.

    Args:
        state: The current streaming state
        integration_ticks: Number of consecutive silent ticks before an overlap region ends
            during linearization. Higher values are more tolerant of brief pauses. Default is 1.

    Returns:
        ListenerReactionDecision with decision and metadata from the LLM call
    """
    # Build the conversation history from tick-based history (including current pending input)
    linearized_messages = state.get_linearized_messages(
        strategy=LinearizationStrategy.CONTAINMENT_AWARE,
        include_pending_input=True,
        indicate_current_incomplete=True,
        integration_ticks=integration_ticks,
    )

    if not linearized_messages and not state.input_turn_taking_buffer:
        return ListenerReactionDecision(decision=False)

    # Format conversation history
    formatted_history = _format_conversation_history(linearized_messages)

    logger.info(f"CHECKING INTERRUPTION:\nSent to LLM:\n{formatted_history}\n\n\n")

    # Build the prompt for interruption decision using template
    interruption_prompt = INTERRUPTION_DECISION_PROMPT.format(
        conversation_history=formatted_history
    )

    # Create messages for LLM call
    decision_messages = [UserMessage(role="user", content=interruption_prompt)]

    try:
        response = generate(
            model=VOICE_USER_SIMULATOR_DECISION_MODEL,
            messages=decision_messages,
            call_name="interruption_decision",
        )

        decision_text = response.content.strip().upper()
        should_interrupt = decision_text == "YES"

        logger.debug(f"Interruption decision: {decision_text}")

        return ListenerReactionDecision(
            decision=should_interrupt,
            generation_time_seconds=response.generation_time_seconds,
            cost=response.cost,
            usage=response.usage,
        )
    except Exception as e:
        logger.error(f"Error in interruption decision: {e}")
        # Default to not interrupting on error
        return ListenerReactionDecision(decision=False)


def user_backchannel_policy(
    state: UserStreamingState,
    integration_ticks: int = 1,
) -> ListenerReactionDecision:
    """
    Decide whether the user should backchannel while the agent is speaking.

    This is called when the agent is currently speaking and the user is listening.
    It determines if the user should emit a brief acknowledgment (e.g., "uh-huh", "ok").

    This function is reusable by both text and voice streaming user simulators.

    Args:
        state: The current streaming state
        integration_ticks: Number of consecutive silent ticks before an overlap region ends
            during linearization. Higher values are more tolerant of brief pauses. Default is 1.

    Returns:
        ListenerReactionDecision with decision and metadata from the LLM call
    """
    # Build the conversation history from tick-based history (including current pending input)
    linearized_messages = state.get_linearized_messages(
        strategy=LinearizationStrategy.CONTAINMENT_AWARE,
        include_pending_input=True,
        indicate_current_incomplete=True,
        integration_ticks=integration_ticks,
    )

    if not linearized_messages and not state.input_turn_taking_buffer:
        return ListenerReactionDecision(decision=False)

    # Format conversation history
    formatted_history = _format_conversation_history(linearized_messages)

    logger.info(f"CHECKING BACKCHANNEL:\nSent to LLM:\n{formatted_history}\n\n\n")

    # Build the prompt for backchannel decision using template
    decision_prompt = BACKCHANNEL_DECISION_PROMPT.format(
        conversation_history=formatted_history
    )

    # Create messages for LLM call
    decision_messages = [UserMessage(role="user", content=decision_prompt)]

    try:
        response = generate(
            model=VOICE_USER_SIMULATOR_DECISION_MODEL,
            messages=decision_messages,
            call_name="backchannel_decision",
        )

        decision_text = response.content.strip().upper()
        should_backchannel = decision_text == "YES"

        logger.debug(f"Backchannel decision: {decision_text}")

        return ListenerReactionDecision(
            decision=should_backchannel,
            generation_time_seconds=response.generation_time_seconds,
            cost=response.cost,
            usage=response.usage,
        )
    except Exception as e:
        logger.error(f"Error in backchannel decision: {e}")
        # Default to not backchanneling on error
        return ListenerReactionDecision(decision=False)


class UserAudioStreamingMixin(
    AudioChunkingMixin[ValidUserInputMessage, UserMessage, UserAudioStreamingState]
):
    """
    Agent-specific audio chunking mixin.
    This is a specialization of AudioChunkingMixin for agents with audio-based chunking.
    """


class VoiceStreamingUserSimulator(
    VoiceMixin[ValidUserInputMessage, UserMessage, UserAudioStreamingState],
    StreamingAudioEffectsMixin,
    UserAudioStreamingMixin,
    LLMConfigMixin,
    FullDuplexVoiceUser[UserAudioStreamingState],
):
    """
    Full-duplex LLM-based user simulator with voice-based streaming.

    Inherits from:
    - VoiceMixin: Provides voice logic (TTS, STT, background noise)
    - UserAudioStreamingMixin: Provides audio chunking and streaming logic
    - LLMConfigMixin: Provides LLM configuration (llm, llm_args, set_seed)
    - FullDuplexVoiceUser: Provides full-duplex streaming + voice interface

    Features:
    - Enhanced tool call handling (tool calls are never chunked)
    - Custom turn-taking logic (responds immediately by default)
    - Proper state typing with UserAudioStreamingState

    Usage:
        user = VoiceStreamingUserSimulator(
            tools=tools,
            domain_policy=policy,
            llm="gpt-4",
            chunk_size=10,
        )

        # FULL_DUPLEX mode
        state = user.get_init_state()  # Returns UserAudioStreamingState
        chunk, state = user.get_next_chunk(state, incoming_chunk)
    """

    def __init__(
        self,
        tools: List[Tool],
        instructions: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        chunk_size: int = 10,
        voice_settings: VoiceSettings = VoiceSettings(),
        wait_to_respond_threshold_other: int = 2,
        wait_to_respond_threshold_self: int = 4,
        yield_threshold_when_interrupted: Optional[int] = 2,
        yield_threshold_when_interrupting: Optional[int] = None,
        backchannel_min_threshold: Optional[int] = None,
        backchannel_max_threshold: Optional[int] = None,
        backchannel_poisson_rate: Optional[float] = None,
        use_llm_backchannel: bool = True,
        interruption_check_interval: Optional[int] = None,
        integration_ticks: int = 1,
        silence_annotation_threshold_ticks: Optional[int] = None,
        tick_duration_seconds: float = 0.05,
        persona_config: Optional[PersonaConfig] = None,
        audio_taps_dir: Optional["Path"] = None,
    ):
        """
        Initialize the streaming user simulator.

        Args:
            tools: List of available tools
            instructions: The instructions for the user.
            llm: LLM model name
            llm_args: Additional LLM arguments
            chunk_size: Number of units per chunk
            wait_to_respond_threshold_other: Minimum time to wait since OTHER (agent) last spoke before generating a response.
                Both this AND wait_to_respond_threshold_self must be satisfied.
            wait_to_respond_threshold_self: Minimum time to wait since SELF (user) last spoke before generating a response.
                Both this AND wait_to_respond_threshold_other must be satisfied.
            yield_threshold_when_interrupted: How long user keeps speaking when agent interrupts user. If None, cannot be interrupted.
            yield_threshold_when_interrupting: How long user keeps speaking when user interrupts agent. If None, uses yield_threshold_when_interrupted.
            backchannel_min_threshold: Min threshold for backchanneling (ticks). If None and using Poisson, cannot backchannel.
            backchannel_max_threshold: Max threshold for backchanneling (ticks). Used with Poisson policy.
            backchannel_poisson_rate: Poisson rate for backchanneling (events/second). Used with Poisson policy.
            use_llm_backchannel: If True, use LLM-based backchannel policy. If False, use Poisson-based policy.
            interruption_check_interval: If set, only check for interruption every N ticks. Useful to reduce callback frequency.
            integration_ticks: Number of consecutive silent ticks before an overlap region ends
                during linearization. Higher values are more tolerant of brief pauses. Default is 1.
            silence_annotation_threshold_ticks: If set, add silence annotations to conversation history
                when both parties are silent for more than this many ticks.
            tick_duration_seconds: Duration of each tick in seconds. Used for backchanneling Poisson calculations.
            voice_settings: Voice settings for the user.
            persona_config: Runtime persona configuration for user behavior (e.g., verbosity level, interrupt tendency)
            audio_taps_dir: If set, record audio at each pipeline stage to WAV files in this directory.
        """
        # Initialize mixin and base class
        super().__init__(
            tools=tools,
            instructions=instructions,
            llm=llm,
            llm_args=llm_args,
            chunk_size=chunk_size,
            voice_settings=voice_settings,
        )
        self.persona_config = persona_config or PersonaConfig()
        self.integration_ticks = integration_ticks
        self.silence_annotation_threshold_ticks = silence_annotation_threshold_ticks
        self.tick_duration_seconds = tick_duration_seconds
        self.validate_voice_settings()
        self.wait_to_respond_threshold_other = wait_to_respond_threshold_other
        self.wait_to_respond_threshold_self = wait_to_respond_threshold_self
        self.yield_threshold_when_interrupted = yield_threshold_when_interrupted
        self.yield_threshold_when_interrupting = yield_threshold_when_interrupting
        self.backchannel_min_threshold = backchannel_min_threshold
        self.backchannel_max_threshold = backchannel_max_threshold
        self.backchannel_poisson_rate = backchannel_poisson_rate
        self.use_llm_backchannel = use_llm_backchannel
        self.interruption_check_interval = interruption_check_interval

        # Default yield_threshold_when_interrupting to yield_threshold_when_interrupted if not set
        if (
            self.yield_threshold_when_interrupting is None
            and self.yield_threshold_when_interrupted is not None
        ):
            self.yield_threshold_when_interrupting = (
                self.yield_threshold_when_interrupted
            )
            logger.info(
                f"yield_threshold_when_interrupting not set, defaulting to yield_threshold_when_interrupted={self.yield_threshold_when_interrupted}"
            )
        # Enable user-initiated interruption based on persona config
        if persona_config is not None and persona_config.interrupt_tendency is not None:
            # Enable user-initiated interruption for users with INTERRUPTS tendency
            self.enable_user_initiated_interruption = (
                persona_config.interrupt_tendency == InterruptTendency.INTERRUPTS
            )
        else:
            # No persona config or interrupt_tendency is None
            self.enable_user_initiated_interruption = False
        self.validate_turn_taking_settings()

        # Effect timeline (always active, lightweight metadata)
        self._effect_timeline = EffectTimeline()

        # Audio taps for pipeline diagnostics (None = disabled, zero cost)
        self._audio_taps: Optional[dict[str, AudioTap]] = None
        if audio_taps_dir is not None:
            from tau2.data_model.audio import TELEPHONY_SAMPLE_RATE

            self._audio_taps = {
                # Pipeline-stage taps (existing)
                "agent_input": AudioTap(
                    "user_agent-input",
                    audio_taps_dir,
                    TELEPHONY_SAMPLE_RATE,
                    AudioEncoding.ULAW,
                ),
                "tts_output": AudioTap(
                    "user_tts-output",
                    audio_taps_dir,
                    PCM_SAMPLE_RATE,
                    AudioEncoding.PCM_S16LE,
                ),
                "post_noise": AudioTap(
                    "user_post-noise",
                    audio_taps_dir,
                    PCM_SAMPLE_RATE,
                    AudioEncoding.PCM_S16LE,
                ),
                "post_telephony": AudioTap(
                    "user_post-telephony",
                    audio_taps_dir,
                    TELEPHONY_SAMPLE_RATE,
                    AudioEncoding.ULAW,
                ),
                "user_output": AudioTap(
                    "user_output",
                    audio_taps_dir,
                    TELEPHONY_SAMPLE_RATE,
                    AudioEncoding.ULAW,
                ),
                # Per-effect multitrack taps (time-aligned, same length)
                "speech_only": AudioTap(
                    "user_speech-only",
                    audio_taps_dir,
                    PCM_SAMPLE_RATE,
                    AudioEncoding.PCM_S16LE,
                ),
                "background_noise_only": AudioTap(
                    "user_background-noise-only",
                    audio_taps_dir,
                    PCM_SAMPLE_RATE,
                    AudioEncoding.PCM_S16LE,
                ),
                "burst_noise_only": AudioTap(
                    "user_burst-noise-only",
                    audio_taps_dir,
                    PCM_SAMPLE_RATE,
                    AudioEncoding.PCM_S16LE,
                ),
                "out_of_turn_speech_only": AudioTap(
                    "user_out-of-turn-speech-only",
                    audio_taps_dir,
                    PCM_SAMPLE_RATE,
                    AudioEncoding.PCM_S16LE,
                ),
            }
            logger.info(f"Audio taps enabled, output dir: {audio_taps_dir}")

    def validate_turn_taking_settings(self) -> None:
        """Validate the turn-taking settings."""
        if (
            self.yield_threshold_when_interrupted is not None
            and self.yield_threshold_when_interrupted < 2
        ):
            raise ValueError(
                f"yield_threshold_when_interrupted must be at least 2. Got {self.yield_threshold_when_interrupted}. Setting it lower will result in unstable behavior."
            )
        if (
            self.yield_threshold_when_interrupting is not None
            and self.yield_threshold_when_interrupting < 2
        ):
            raise ValueError(
                f"yield_threshold_when_interrupting must be at least 2. Got {self.yield_threshold_when_interrupting}. Setting it lower will result in unstable behavior."
            )

    def validate_voice_settings(self) -> None:
        """Validate the voice settings."""
        if self.voice_settings is None:
            raise ValueError("Voice settings must be provided")
        if self.voice_settings.transcription_enabled:
            raise ValueError("Transcription is not supported for user yet")
        if not self.voice_settings.synthesis_enabled:
            raise ValueError("Voice synthesis must be enabled for user simulator")

    @property
    def global_simulation_guidelines(self) -> str:
        """The voice-specific simulation guidelines for the user simulator."""
        use_tools = self.tools is not None
        return get_global_user_sim_guidelines_voice(use_tools=use_tools)

    @property
    def system_prompt(self) -> str:
        """The system prompt for the user simulator."""
        if self.instructions is None:
            logger.warning("No instructions provided for user simulator")

        guidelines = self.global_simulation_guidelines

        # Check if persona config adds any guidelines
        persona_guidelines = self.persona_config.to_guidelines_text()
        if persona_guidelines is None:
            persona_guidelines = ""
        if persona_guidelines:
            persona_guidelines = f"\n\n{persona_guidelines}\n"
        guidelines_with_persona = guidelines.replace(
            "<PERSONA_GUIDELINES>", persona_guidelines
        )

        system_prompt = SYSTEM_PROMPT.format(
            global_user_sim_guidelines_with_persona=guidelines_with_persona,
            instructions=self.instructions,
        )
        return system_prompt

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """Check if the message is a stop message."""
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

    def get_next_chunk(
        self,
        state: "UserAudioStreamingState",
        participant_chunk: AssistantMessage,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> Tuple[UserMessage, "UserAudioStreamingState"]:
        """Process one tick, recording incoming agent audio if taps are enabled."""
        if self._audio_taps:
            self._audio_taps["agent_input"].record_message(participant_chunk)
        return super().get_next_chunk(state, participant_chunk, tool_results)

    def stop(
        self,
        participant_chunk: Optional[Message] = None,
        state: Optional["UserAudioStreamingState"] = None,
        tool_results: Optional[EnvironmentMessage] = None,
    ) -> None:
        """Stop the user simulator, close timeline events, and save audio taps."""
        # Close any open timeline events at the final time
        if state is not None:
            end_ms = int(state.elapsed_samples * 1000 / PCM_SAMPLE_RATE)
            self._effect_timeline.close_all_open(end_ms)

        if self._audio_taps:
            for tap in self._audio_taps.values():
                tap.save()
            logger.info(
                f"Saved {len(self._audio_taps)} audio taps: "
                f"{list(self._audio_taps.keys())}"
            )
        super().stop(participant_chunk, state, tool_results)

    def get_effect_timeline(self) -> EffectTimeline:
        """Return the effect timeline recorded during the simulation."""
        return self._effect_timeline

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserAudioStreamingState:
        """
        Get the initial state of the streaming user.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the streaming user (UserAudioStreamingState).
        """
        if message_history is None:
            message_history = []

        synthesis_config = self.voice_settings.synthesis_config
        speech_env = self.voice_settings.speech_environment

        # Resolve filenames to full paths
        background_noise_file = resolve_background_noise_path(
            speech_env.background_noise_file
        )
        burst_noise_files = (
            resolve_burst_noise_paths(speech_env.burst_noise_files)
            if speech_env.burst_noise_files
            else None
        )

        noise_generator, out_of_turn_speech_generator = (
            create_streaming_audio_generators(
                synthesis_config=synthesis_config,
                persona_name=speech_env.persona_name,
                sample_rate=PCM_SAMPLE_RATE,
                background_noise_file=background_noise_file,
            )
        )

        # Use effect configs directly - complexity overrides already merged in run.py
        effect_scheduler = EffectScheduler(
            seed=speech_env.voice_seed,
            source_config=synthesis_config.source_effects_config,
            speech_config=synthesis_config.speech_effects_config,
            channel_config=synthesis_config.channel_effects_config,
            sample_rate=PCM_SAMPLE_RATE,
            burst_noise_files=burst_noise_files,
            # No override params needed - configs already have complexity values merged
        )

        # Create streaming telephony converter (preserves filter state between chunks)
        telephony_converter = (
            StreamingTelephonyConverter(input_sample_rate=PCM_SAMPLE_RATE)
            if speech_env.telephony_enabled
            else None
        )

        # Create streaming state
        backchannel_rng = random.Random(speech_env.voice_seed + 100)
        return UserAudioStreamingState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
            noise_generator=noise_generator,
            effect_scheduler=effect_scheduler,
            out_of_turn_speech_generator=out_of_turn_speech_generator,
            elapsed_samples=0,
            input_turn_taking_buffer=[],
            output_streaming_queue=[],
            pending_background_noise_chunks=[],
            pending_burst_noise_chunks=[],
            telephony_converter=telephony_converter,
            backchannel_rng=backchannel_rng,
        )

    def speech_detection(self, chunk: Optional[ValidUserInputMessage]) -> bool:
        """
        Check if the chunk is a speech chunk.
        """
        if not isinstance(chunk, AssistantMessage):
            return False
        # Check contains_speech flag - defaults to True for backward compatibility if not set
        return chunk.contains_speech if chunk.contains_speech is not None else True

    def _next_turn_taking_action(
        self, state: UserAudioStreamingState
    ) -> TurnTakingAction:
        """
        Decide the next action to take in the turn-taking.
        """
        # Prepare listener reaction callbacks
        should_interrupt_callback = None
        should_backchannel_callback = None
        integration_ticks = self.integration_ticks

        # Interruption callback is tied to persona's interrupt tendency
        if self.enable_user_initiated_interruption:

            def should_interrupt_callback(
                s: UserStreamingState,
            ) -> ListenerReactionDecision:
                return user_interruption_policy(s, integration_ticks=integration_ticks)

        # Backchannel callback is tied to the use_llm_backchannel config
        if self.use_llm_backchannel:

            def should_backchannel_callback(
                s: UserStreamingState,
            ) -> ListenerReactionDecision:
                return user_backchannel_policy(s, integration_ticks=integration_ticks)

        action, info = basic_turn_taking_policy(
            state,
            yield_threshold_when_interrupted=self.yield_threshold_when_interrupted,
            yield_threshold_when_interrupting=self.yield_threshold_when_interrupting,
            wait_to_respond_threshold_other=self.wait_to_respond_threshold_other,
            wait_to_respond_threshold_self=self.wait_to_respond_threshold_self,
            backchannel_min_threshold=self.backchannel_min_threshold,
            backchannel_max_threshold=self.backchannel_max_threshold,
            backchannel_poisson_rate=self.backchannel_poisson_rate,
            tick_duration_seconds=self.tick_duration_seconds,
            should_interrupt_callback=should_interrupt_callback,
            should_backchannel_callback=should_backchannel_callback,
            use_llm_backchannel=self.use_llm_backchannel,
            listener_reaction_check_interval=self.interruption_check_interval,
        )
        logger.debug(f"USER SIMULATOR TURN-TAKING ACTION: {action}. Reason: {info}")

        # Extract timing/cost metadata from state (set by basic_turn_taking_policy)
        timing = getattr(state, "_listener_reaction_timing", None) or {}

        return TurnTakingAction(
            action=action,
            info=info,
            interrupt_check_seconds=timing.get("interrupt_check_seconds"),
            interrupt_check_cost=timing.get("interrupt_check_cost"),
            interrupt_check_usage=timing.get("interrupt_check_usage"),
            backchannel_check_seconds=timing.get("backchannel_check_seconds"),
            backchannel_check_cost=timing.get("backchannel_check_cost"),
            backchannel_check_usage=timing.get("backchannel_check_usage"),
        )

    def _apply_frame_drop(
        self, user_message: UserMessage, drop_duration_ms: int
    ) -> UserMessage:
        """Apply frame drop (packet loss simulation) by zeroing out audio."""
        zeroed = self.apply_streaming_frame_drop(
            audio_bytes=user_message.get_audio_bytes(),
            sample_rate=user_message.audio_format.sample_rate,
            bytes_per_sample=user_message.audio_format.bytes_per_sample,
            drop_duration_ms=drop_duration_ms,
            is_ulaw=user_message.audio_format.is_ulaw,
        )
        user_message.audio_content = audio_bytes_to_string(zeroed)

        existing = user_message.channel_effects
        total_drop_ms = drop_duration_ms + (existing.frame_drop_ms if existing else 0)
        user_message.channel_effects = ChannelEffectsResult(
            frame_drops_enabled=True,
            frame_drop_ms=total_drop_ms,
        )
        return user_message

    def _add_telephony_compression(
        self,
        user_message: UserMessage,
        converter: StreamingTelephonyConverter,
    ) -> UserMessage:
        """Add telephony compression to the audio."""
        audio_data = AudioData(
            data=user_message.get_audio_bytes(),
            format=user_message.audio_format,
        )
        audio_data = converter.convert_chunk(audio_data)
        user_message.audio_content = audio_bytes_to_string(audio_data.data)
        user_message.audio_format = audio_data.format
        user_message.audio_path = None
        return user_message

    def _apply_chunk_effects(
        self,
        chunk: Optional[UserMessage],
        state: UserAudioStreamingState,
        is_speech: bool,
    ) -> UserMessage:
        """Apply scheduled effects, noise, telephony, and frame drops to chunk.

        Uses the multitrack pipeline: each effect is computed as an
        independent track, then summed before telephony and frame drops.
        When audio taps are enabled, individual tracks are recorded.
        Effect events are always recorded in the timeline.
        """
        current_time_ms = int(state.elapsed_samples * 1000 / PCM_SAMPLE_RATE)

        # 1. Get scheduled effects from the scheduler
        scheduled_effects = []
        if state.effect_scheduler is not None:
            chunk_duration_ms = int(self.chunk_size * 1000 / PCM_SAMPLE_RATE)
            scheduled_effects = state.effect_scheduler.check_for_effects(
                chunk_duration_ms=chunk_duration_ms,
                is_silence=not is_speech,
                current_time_ms=current_time_ms,
                has_active_burst=state.noise_generator.has_active_burst(),
            )

        if self._audio_taps:
            self._audio_taps["tts_output"].record_message(chunk)

        # 2. Extract speech audio from the message
        speech_audio = None
        if chunk is not None:
            speech_audio = AudioData(
                data=chunk.get_audio_bytes(),
                format=deepcopy(chunk.audio_format),
                audio_path=chunk.audio_path,
            )

        # 3. Process chunk through multitrack pipeline
        result = self.process_streaming_chunk(
            speech_audio=speech_audio,
            noise_generator=state.noise_generator,
            num_samples=self.chunk_size,
            scheduled_effects=scheduled_effects,
            out_of_turn_generator=state.out_of_turn_speech_generator,
            pending_effect=state.pending_effect,
        )
        state.pending_effect = result.pending_effect

        # 4. Record timeline events
        if result.burst_noise_file:
            burst_file_name = Path(result.burst_noise_file).name
            burst_params: dict = {"file": burst_file_name}
            if "burst_snr_db" in result.tracks.metadata:
                burst_params["snr_db"] = result.tracks.metadata["burst_snr_db"]
            if "burst_scale" in result.tracks.metadata:
                burst_params["scale"] = result.tracks.metadata["burst_scale"]
            self._effect_timeline.open_event(
                effect_type="burst_noise",
                start_ms=current_time_ms,
                participant="user",
                params=burst_params,
            )
        if result.triggered_speech_insert:
            self._effect_timeline.open_event(
                effect_type="out_of_turn_speech",
                start_ms=current_time_ms,
                participant="user",
                params={
                    "type": result.triggered_speech_insert.type,
                    "text": result.triggered_speech_insert.text,
                },
            )
        if result.pending_effect_completed:
            self._effect_timeline.close_event(
                effect_type="out_of_turn_speech",
                end_ms=current_time_ms,
                participant="user",
            )
        if result.pending_effect_discarded:
            if self._effect_timeline.has_open_event("out_of_turn_speech", "user"):
                self._effect_timeline.close_event(
                    effect_type="out_of_turn_speech",
                    end_ms=current_time_ms,
                    participant="user",
                )
        # Close burst event when the noise generator's burst finishes
        if (
            not state.noise_generator.has_active_burst()
            and self._effect_timeline.has_open_event("burst_noise", "user")
        ):
            self._effect_timeline.close_event(
                effect_type="burst_noise",
                end_ms=current_time_ms,
                participant="user",
            )

        # 5. Record per-track taps (only when enabled)
        if self._audio_taps:
            self._audio_taps["speech_only"].record_numpy(result.tracks.speech)
            self._audio_taps["background_noise_only"].record_numpy(
                result.tracks.background_noise
            )
            self._audio_taps["burst_noise_only"].record_numpy(result.tracks.burst_noise)
            self._audio_taps["out_of_turn_speech_only"].record_numpy(
                result.out_of_turn_speech
            )

        # 6. Sum tracks -> mixed PCM 16kHz
        mixed_audio = result.to_mixed_audio_data()

        # Build the output message
        if not chunk:
            chunk = UserMessage(
                role="user",
                content="",
                cost=0.0,
                usage=None,
                is_audio=True,
                audio_content=audio_bytes_to_string(mixed_audio.data),
                audio_format=mixed_audio.format,
                audio_script_gold="",
                chunk_id=0,
                is_final_chunk=True,
                contains_speech=False,
                source_effects=result.source_effects,
            )
        else:
            chunk.audio_content = audio_bytes_to_string(mixed_audio.data)
            chunk.audio_format = mixed_audio.format
            chunk.source_effects = result.source_effects

        if self._audio_taps:
            self._audio_taps["post_noise"].record_message(chunk)

        # 7. Apply telephony compression
        if self.voice_settings.speech_environment.telephony_enabled:
            chunk = self._add_telephony_compression(chunk, state.telephony_converter)

        if self._audio_taps:
            self._audio_taps["post_telephony"].record_message(chunk)

        # 8. Apply frame drops after telephony
        for effect in scheduled_effects:
            if effect.effect_type == "frame_drop" and effect.frame_drop_duration_ms:
                self._effect_timeline.open_event(
                    effect_type="frame_drop",
                    start_ms=current_time_ms,
                    participant="user",
                    params={"duration_ms": effect.frame_drop_duration_ms},
                )
                self._effect_timeline.close_event(
                    effect_type="frame_drop",
                    end_ms=current_time_ms + effect.frame_drop_duration_ms,
                    participant="user",
                )
                chunk = self._apply_frame_drop(chunk, effect.frame_drop_duration_ms)
                logger.debug(f"Applied frame drop: {effect.frame_drop_duration_ms}ms")

        if self._audio_taps:
            self._audio_taps["user_output"].record_message(chunk)

        # Update elapsed samples
        state.elapsed_samples += self.chunk_size

        return chunk

    def _perform_turn_taking_action(
        self, state: UserAudioStreamingState, action: TurnTakingAction
    ) -> Tuple[UserMessage, UserAudioStreamingState]:
        """
        Perform the next action in the turn-taking.

        Note: Chunk recording in tick-based history is handled by get_next_chunk()
        in the StreamingMixin, not here. This method just produces the chunk
        and manages pending buffers.

        Args:
            state: The current state of the turn-taking.
            action: The action to perform.
        Returns:
            A tuple of the next chunk and the updated state.
        """
        logger.debug(f"Performing turn-taking action: {action}")
        next_user_chunk = None
        is_speech_action = False
        if action.action == "keep_talking":
            next_user_chunk = state.output_streaming_queue.pop(0)
            next_user_chunk.timestamp = get_now()
            is_speech_action = True
            # Clear backchannel flag when queue is empty (backchannel complete)
            if len(state.output_streaming_queue) == 0:
                if state.is_backchanneling:
                    state.is_backchanneling = False
                    logger.debug("Backchannel complete")
                if state.delivering_tool_result_speech:
                    state.delivering_tool_result_speech = False
                    logger.debug("Tool result speech delivery complete")
        elif action.action == "stop_talking":
            logger.debug("Stopping talking: Flushing output streaming queue")
            state.output_streaming_queue = []
            state.is_backchanneling = False
        elif action.action == "generate_message":
            merged_message = merge_homogeneous_chunks(state.input_turn_taking_buffer)
            full_message, new_state = self._generate_full_duplex_voice_message(
                merged_message, state
            )
            state.input_turn_taking_buffer = []
            if full_message.is_tool_call():
                logger.debug("Generating message: Tool call detected")
                noise_chunk = self._apply_chunk_effects(
                    None, new_state, is_speech=False
                )
                full_message.audio_content = noise_chunk.audio_content
                full_message.audio_format = noise_chunk.audio_format
                full_message.is_audio = True
                full_message.contains_speech = False
                full_message.source_effects = noise_chunk.source_effects
                full_message.turn_taking_action = action
                return full_message, new_state
            elif self.is_stop(full_message):
                logger.debug("Generating message: Stop message detected")
                full_message.turn_taking_action = action
                return full_message, new_state
            else:
                logger.debug("Generating message: Creating chunk messages")
                chunk_messages = self._create_chunk_messages(full_message)
                logger.debug(
                    f"Generating message: Created {len(chunk_messages)} chunk messages"
                )
                state.output_streaming_queue.extend(chunk_messages)
                next_user_chunk = state.output_streaming_queue.pop(0)
                next_user_chunk.timestamp = get_now()
                is_speech_action = True
        elif action.action == "wait":
            logger.debug("Waiting: No action required")
        elif action.action == "backchannel":
            logger.debug("Backchannel: Generating backchannel message")
            backchannel_message = self._generate_backchannel_message(state)
            chunk_messages = self._create_chunk_messages(backchannel_message)
            logger.debug(f"Backchannel: Created {len(chunk_messages)} chunk messages")
            state.output_streaming_queue.extend(chunk_messages)
            next_user_chunk = state.output_streaming_queue.pop(0)
            next_user_chunk.timestamp = get_now()
            is_speech_action = True
            # Mark that we're delivering a backchannel (prevents interruption)
            state.is_backchanneling = True
            # Reset backchannel cooldown timer after performing backchannel
            state.ticks_since_last_backchannel = 0
        else:
            raise ValueError(f"Invalid action: {action}")

        next_user_chunk = self._apply_chunk_effects(
            next_user_chunk, state, is_speech=is_speech_action
        )

        # _apply_chunk_effects should always return a chunk (creates one with noise if None)
        if next_user_chunk is None:
            raise ValueError(
                "Voice Streaming User Simulator should never return None chunk."
            )

        if is_speech_action:
            next_user_chunk.contains_speech = True
            state.time_since_last_talk = 0
        else:
            state.time_since_last_talk += 1

        # Update turn-taking action with LLM/TTS timing if available
        if action.action == "generate_message":
            llm_time = getattr(state, "_llm_generation_seconds", None)
            tts_time = getattr(state, "_tts_synthesis_seconds", None)
            if llm_time is not None:
                action.llm_generation_seconds = llm_time
            if tts_time is not None:
                action.tts_synthesis_seconds = tts_time
            # Clear the temporary attributes
            state._llm_generation_seconds = None
            state._tts_synthesis_seconds = None

        # Set the turn-taking action on the chunk
        next_user_chunk.turn_taking_action = action
        return next_user_chunk, state

    def _process_tool_result(
        self,
        tool_result: EnvironmentMessage,
        state: UserAudioStreamingState,
    ) -> Tuple[UserMessage, UserAudioStreamingState]:
        """Process a tool result by calling the LLM and returning the response."""
        # Temporarily set buffer so get_linearized_messages(include_pending_input=True)
        # picks up the tool result for LLM context
        saved_buffer = state.input_turn_taking_buffer
        state.input_turn_taking_buffer = [tool_result]

        full_message, state = self._generate_full_duplex_voice_message(
            tool_result, state
        )

        # Restore participant chunks (preserved for timing)
        state.input_turn_taking_buffer = saved_buffer

        if full_message.is_tool_call():
            logger.debug("Tool result processing: LLM returned another tool call")
            noise_chunk = self._apply_chunk_effects(None, state, is_speech=False)
            full_message.audio_content = noise_chunk.audio_content
            full_message.audio_format = noise_chunk.audio_format
            full_message.is_audio = True
            full_message.contains_speech = False
            full_message.source_effects = noise_chunk.source_effects
            return full_message, state
        elif self.is_stop(full_message):
            logger.debug("Tool result processing: stop message detected")
            return full_message, state
        else:
            # Queue chunks but DON'T start streaming yet.
            # Speech delivery is deferred to normal turn-taking on the next tick.
            # This ensures overlap/interruption logic applies correctly.
            logger.debug("Tool result processing: queuing speech chunks")
            chunk_messages = self._create_chunk_messages(full_message)
            state.output_streaming_queue.extend(chunk_messages)
            state.delivering_tool_result_speech = True
            waiting_chunk, state = self._emit_waiting_chunk(state)
            return waiting_chunk, state

    def _emit_waiting_chunk(
        self, state: UserAudioStreamingState
    ) -> Tuple[UserMessage, UserAudioStreamingState]:
        """Emit background noise while waiting for tool results."""
        noise_chunk = self._apply_chunk_effects(None, state, is_speech=False)
        state.time_since_last_talk += 1
        return noise_chunk, state

    def _generate_full_duplex_voice_message(
        self, message: ValidUserInputMessage, state: UserAudioStreamingState
    ) -> Tuple[UserMessage, UserAudioStreamingState]:
        """
        Generate a voice message using tick-based history for LLM context.

        This method linearizes the tick history to build the LLM context,
        and synthesizes voice for the response.

        Args:
            message: The incoming message to respond to.
            state: The current streaming state with tick history.

        Returns:
            A tuple of the user message (with voice) and updated state.
        """
        # Build LLM context from linearized ticks (including current pending input)
        linearized_messages = state.get_linearized_messages(
            strategy=LinearizationStrategy.CONTAINMENT_AWARE,
            include_pending_input=True,
            indicate_current_incomplete=True,
            integration_ticks=self.integration_ticks,
            silence_annotation_threshold_ticks=self.silence_annotation_threshold_ticks,
            tick_duration_seconds=self.tick_duration_seconds,
        )

        # Check that last message is a valid user input message
        if not isinstance(linearized_messages[-1], (ValidUserInputMessage)):
            if isinstance(linearized_messages[-1], SystemMessage):
                # SystemMessage at end is expected (e.g., silence annotations)
                logger.debug(
                    f"Running user generation with SystemMessage as last message (e.g., silence annotation)"
                )
            else:
                logger.warning(
                    f"Last message is not a valid user input message: {type(linearized_messages[-1]).__name__}"
                )

        if isinstance(linearized_messages[-1], (AssistantMessage)):
            if linearized_messages[-1].content is None:
                logger.warning(
                    f"Last message is an assistant message with no content: {linearized_messages[-1]}"
                )

        # Flip roles for user simulator (it sees itself as assistant)
        flipped_messages = self._flip_roles_for_llm(linearized_messages)

        # Build full message list with system messages
        messages = state.system_messages + flipped_messages

        # Add a role identity reminder as a system message to help the LLM stay in character
        role_reminder = SystemMessage(
            role="system",
            content="REMINDER: You are the CUSTOMER calling for help. Respond as the customer would - with questions, requests, or information about your issue. Do NOT respond as the customer service agent.",
        )
        messages = messages + [role_reminder]

        # Generate response (timing is captured in llm_utils.generate)
        assistant_message = generate(
            model=self.llm,
            messages=messages,
            tools=self.tools,
            call_name="user_streaming_response",
            **self.llm_args,
        )

        # Store LLM timing on state for later use when updating TurnTakingAction
        # Use generation_time_seconds from the returned message
        state._llm_generation_seconds = assistant_message.generation_time_seconds

        # Convert assistant response to user message
        user_message = UserMessage(
            role="user",
            content=assistant_message.content,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        my_str = ""
        for message in linearized_messages:
            my_str += f"{message.role}: {message.content}\n"

        logger.info(
            f"USER SIMULATOR:\nSent to LLM:\n{my_str}\nReceived from LLM:\n{user_message.content}\n\n\n"
        )

        # Flip the requestor of tool calls
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
            return user_message, state

        # Check for stop
        if self.is_stop(user_message):
            return user_message, state

        # Synthesize voice (without background noise - added per chunk) with timing
        effects_turn_idx = state.user_utterance_count
        tts_start = time.perf_counter()
        user_message = self.synthesize_voice(
            user_message,
            state,
            effects_turn_idx=effects_turn_idx,
            add_background_noise=False,
            add_burst_noise=False,
            add_telephony_format=False,
            add_channel_effects=False,
        )
        tts_synthesis_seconds = time.perf_counter() - tts_start

        # Store TTS timing on state for later use
        state._tts_synthesis_seconds = tts_synthesis_seconds

        state.user_utterance_count += 1

        return user_message, state

    def _flip_roles_for_llm(self, messages: list[Message]) -> list[Message]:
        """
        Flip message roles for user simulator LLM context.

        The user simulator acts as an assistant internally, so:
        - UserMessage (what user said) -> AssistantMessage (what "I" said)
        - AssistantMessage (what agent said) -> UserMessage (what "they" said)
        - ToolMessage for user -> kept as-is (user's tool results)
        - SystemMessage -> kept as-is (e.g., silence annotations)

        Args:
            messages: The linearized message history.

        Returns:
            Messages with roles flipped for LLM consumption.
        """
        from tau2.data_model.message import SystemMessage, ToolMessage

        flipped = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                # User's message -> becomes assistant response
                flipped.append(
                    AssistantMessage(
                        role="assistant",
                        tool_calls=msg.tool_calls,
                        content=msg.content,
                    )
                )
            elif isinstance(msg, AssistantMessage):
                # Agent's message -> becomes user input
                # Skip tool calls and messages without text content
                # (audio-only messages can't be converted to text UserMessage)
                if not msg.is_tool_call() and msg.content:
                    flipped.append(
                        UserMessage(
                            role="user",
                            content=msg.content,
                        )
                    )
            elif isinstance(msg, ToolMessage):
                # Tool messages for user are kept
                if msg.requestor == "user":
                    flipped.append(
                        ToolMessage(
                            id=msg.id,
                            role=msg.role,
                            content=msg.content,
                        )
                    )
            elif isinstance(msg, SystemMessage):
                # System messages are kept as-is (e.g., silence annotations)
                flipped.append(msg)
        return flipped

    def _generate_backchannel_message(
        self, state: UserAudioStreamingState
    ) -> UserMessage:
        """
        Generate a backchannel message.
        """
        # Use user_utterance_count for streaming mode (state.messages isn't updated)
        effects_turn_idx = state.user_utterance_count

        # Randomly select a backchannel phrase
        content = state.backchannel_rng.choice(BACKCHANNEL_PHRASES)

        user_message = UserMessage(
            role="user",
            content=content,
            cost=0.0,
            usage=None,
            raw_data=None,
            chunk_id=0,
            is_final_chunk=True,
        )
        state.user_utterance_count += 1
        return self.synthesize_voice(
            message=user_message,
            state=state,
            effects_turn_idx=effects_turn_idx,
            add_background_noise=False,
            add_burst_noise=False,
            add_telephony_format=False,
            add_channel_effects=False,
        )


def _format_conversation_history(messages: list[Message]) -> str:
    """Format conversation history for interruption decision prompt."""
    formatted_lines = []
    for msg in messages[-100:]:  # Last 100 messages for context
        role = msg.role.upper()
        content = msg.content or ""
        # Skip empty messages
        if content:
            formatted_lines.append(f"{role}: {content}")
    return "\n".join(formatted_lines)

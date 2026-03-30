"""Voice-enabled user simulator with synthesis capabilities."""

from typing import Optional

from tau2.agent.base.voice import VoiceMixin, VoiceState
from tau2.data_model.audio import PCM_SAMPLE_RATE
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    UserMessage,
)
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.tasks import UserInstructions
from tau2.data_model.voice import VoiceSettings
from tau2.environment.tool import Tool
from tau2.user.user_simulator import (
    UserSimulator,
    get_global_user_sim_guidelines_voice,
)
from tau2.user.user_simulator_base import (
    UserState,
    ValidUserInputMessage,
)
from tau2.voice.synthesis.audio_effects.noise_generator import (
    create_background_noise_generator,
)
from tau2.voice_config import resolve_background_noise_path


class VoiceUserState(UserState, VoiceState):
    """State for voice user simulator."""

    pass


class VoiceUserSimulator(
    VoiceMixin[AssistantMessage, UserMessage, VoiceUserState],
    UserSimulator[VoiceUserState],
):
    """User simulator with voice synthesis capabilities."""

    def __init__(
        self,
        llm: str,
        voice_settings: VoiceSettings,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[UserInstructions] = None,
        llm_args: Optional[dict] = None,
        persona_config: Optional[PersonaConfig] = None,
    ):
        super().__init__(
            llm=llm,
            instructions=instructions,
            tools=tools,
            llm_args=llm_args,
            voice_settings=voice_settings,
            persona_config=persona_config,
        )
        self.validate_voice_settings()

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
        """
        The voice-specific simulation guidelines for the user simulator.
        """
        use_tools = self.tools is not None
        return get_global_user_sim_guidelines_voice(use_tools=use_tools)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> VoiceUserState:
        """
        Get the initial state of the voice agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the voice agent (LLMAgentVoiceState).
        """
        # Get the base state from parent
        base_state = super().get_init_state(message_history)

        # Create background noise generator
        synthesis_config = self.voice_settings.synthesis_config
        speech_env = self.voice_settings.speech_environment
        background_noise_file = resolve_background_noise_path(
            speech_env.background_noise_file
        )
        noise_generator = create_background_noise_generator(
            config=synthesis_config.source_effects_config,
            sample_rate=PCM_SAMPLE_RATE,
            background_noise_file=background_noise_file,
        )

        # Create voice user state with the base state's data
        return VoiceUserState(
            system_messages=base_state.system_messages,
            messages=base_state.messages,
            noise_generator=noise_generator,
        )

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: VoiceUserState
    ) -> UserMessage:
        """Generate next message with optional voice synthesis."""
        # Get the base response
        user_message = super()._generate_next_message(message, state)

        if user_message.is_tool_call():
            return user_message
        if self.is_stop(user_message):
            return user_message

        # Count user turns only for reproducible per-turn effects
        effects_turn_idx = sum(1 for m in state.messages if m.role == "user")
        return self.synthesize_voice(
            user_message, state, effects_turn_idx=effects_turn_idx
        )

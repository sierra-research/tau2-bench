# =============================================================================
# SIMULATION DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "ERROR"

# =============================================================================
# LLM DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "gpt-4.1-2025-04-14"
DEFAULT_LLM_USER = "gpt-4.1-2025-04-14"
DEFAULT_LLM_TEMPERATURE_AGENT = 0.0
DEFAULT_LLM_TEMPERATURE_USER = 0.0
DEFAULT_LLM_ARGS_AGENT = {"temperature": DEFAULT_LLM_TEMPERATURE_AGENT}
DEFAULT_LLM_ARGS_USER = {"temperature": DEFAULT_LLM_TEMPERATURE_USER}

DEFAULT_LLM_NL_ASSERTIONS = "gpt-4.1-2025-04-14"
DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE = 0.0
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE}

DEFAULT_LLM_ENV_INTERFACE = "gpt-4.1-2025-04-14"
DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE = 0.0
DEFAULT_LLM_ENV_INTERFACE_ARGS = {"temperature": DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE}

DEFAULT_LLM_EVAL_USER_SIMULATOR = "claude-opus-4-5"

# LLM debug logging
DEFAULT_LLM_LOG_MODE = "latest"  # Options: "all", "latest"

# =============================================================================
# LLM INFRASTRUCTURE (fixed operational constants)
# =============================================================================
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_MIN_WAIT = 1.0  # seconds
DEFAULT_RETRY_MAX_WAIT = 10.0  # seconds
DEFAULT_RETRY_MULTIPLIER = 1.0  # exponential backoff multiplier

# LiteLLM cache
LLM_CACHE_ENABLED = False
DEFAULT_LLM_CACHE_TYPE = "redis"

# Redis (fixed infrastructure config)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""
REDIS_PREFIX = "tau2"
REDIS_CACHE_VERSION = "v1"
REDIS_CACHE_TTL = 60 * 60 * 24 * 30

# Langfuse
USE_LANGFUSE = False

# =============================================================================
# API SERVICE (fixed)
# =============================================================================
API_PORT = 8000

# =============================================================================
# AUDIO CONSTANTS (fixed, protocol-defined)
# =============================================================================
DEFAULT_PCM_SAMPLE_RATE = 16000  # User simulator synthesis rate
DEFAULT_TELEPHONY_RATE = 8000  # API/agent rate (8kHz μ-law, 1 byte/sample)
TELEPHONY_ULAW_SILENCE = b"\x7f"  # μ-law silence byte

# =============================================================================
# VOICE DEFAULTS (overridable via CLI, legacy half-duplex mode)
# =============================================================================
DEFAULT_VOICE_ENABLED = False
DEFAULT_VOICE_SYNTHESIS_PROVIDER = "elevenlabs"
DEFAULT_VOICE_TRANSCRIPTION_MODEL = "nova-3"
DEFAULT_VOICE_MODEL = "eleven_v3"

# Text streaming (legacy)
DEFAULT_TEXT_STREAMING_CHUNK_BY = "words"
DEFAULT_TEXT_STREAMING_CHUNK_SIZE = 1
DEFAULT_TEXT_STREAMING_CONFIG = {
    "chunk_by": DEFAULT_TEXT_STREAMING_CHUNK_BY,
    "chunk_size": DEFAULT_TEXT_STREAMING_CHUNK_SIZE,
}

# =============================================================================
# VOICE USER SIMULATOR (fixed versioning + overridable model)
# =============================================================================
VOICE_USER_SIMULATOR_VERSION = "v1.0"  # fixed, bump on changes
VOICE_USER_SIMULATOR_DECISION_MODEL = "gpt-4.1"  # overridable
DEFAULT_SPEECH_COMPLEXITY = "regular"  # overridable: "control", "regular"

# =============================================================================
# FULL-DUPLEX VOICE DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_AUDIO_NATIVE_AGENT_IMPLEMENTATION = "discrete_time_audio_native_agent"
DEFAULT_AUDIO_NATIVE_USER_IMPLEMENTATION = "voice_streaming_user_simulator"
DEFAULT_AUDIO_NATIVE_PROVIDER = (
    "openai"  # overridable: openai, gemini, xai, nova, qwen, livekit
)
DEFAULT_TICK_DURATION_SECONDS = 0.20  # overridable
DEFAULT_MAX_STEPS_SECONDS = 1200  # overridable, 20 minutes max
DEFAULT_SEND_AUDIO_INSTANT = False  # overridable

# Turn-taking thresholds (overridable, in seconds, converted to ticks at runtime)
DEFAULT_WAIT_TO_RESPOND_THRESHOLD_OTHER_SECONDS = 1.0
DEFAULT_WAIT_TO_RESPOND_THRESHOLD_SELF_SECONDS = 5.0
DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTED_SECONDS = 1.0
DEFAULT_YIELD_THRESHOLD_WHEN_INTERRUPTING_SECONDS = 5.0
DEFAULT_INTERRUPTION_CHECK_INTERVAL_SECONDS = 2.0
DEFAULT_INTEGRATION_DURATION_SECONDS = 0.5
DEFAULT_SILENCE_ANNOTATION_THRESHOLD_SECONDS = 4.0
DEFAULT_BACKCHANNEL_MIN_THRESHOLD_SECONDS = 3.0
DEFAULT_BACKCHANNEL_MAX_THRESHOLD_SECONDS = 12.0
DEFAULT_BACKCHANNEL_POISSON_RATE = 1.0 / 10.0
DEFAULT_USE_LLM_BACKCHANNEL = True

# Retry (overridable)
DEFAULT_AUDIO_NATIVE_MAX_RETRIES = 3
DEFAULT_AUDIO_NATIVE_RETRY_DELAY_SECONDS = 5.0

# =============================================================================
# ADAPTER TIMING (fixed operational constants)
# =============================================================================
DEFAULT_AUDIO_NATIVE_VOIP_PACKET_INTERVAL_MS = 20  # fixed, standard RTP pacing
DEFAULT_AUDIO_NATIVE_CONNECT_TIMEOUT = 30.0  # fixed
DEFAULT_AUDIO_NATIVE_DISCONNECT_TIMEOUT = 5.0  # fixed
DEFAULT_AUDIO_NATIVE_THREAD_JOIN_TIMEOUT = 2.0  # fixed
DEFAULT_AUDIO_NATIVE_TICK_TIMEOUT_BUFFER = 30.0  # fixed
DEFAULT_AUDIO_NATIVE_MAX_INACTIVE_SECONDS = 40.0  # fixed, stall detection

# =============================================================================
# OPENAI PROVIDER (overridable model/voice, fixed API constants)
# =============================================================================
DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime-1.5"  # overridable
_LEGACY_OPENAI_REALTIME_MODEL = "gpt-realtime-2025-08-28"
DEFAULT_OPENAI_REALTIME_BASE_URL = "wss://api.openai.com/v1/realtime"  # fixed
DEFAULT_OPENAI_VOICE = "alloy"  # overridable
DEFAULT_OPENAI_NOISE_REDUCTION = "near_field"  # fixed: "near_field", "far_field", None
DEFAULT_OPENAI_VAD_THRESHOLD_LOW = 0.2  # fixed
DEFAULT_OPENAI_VAD_THRESHOLD_DEFAULT = 0.5  # fixed
DEFAULT_OPENAI_VAD_THRESHOLD = DEFAULT_OPENAI_VAD_THRESHOLD_DEFAULT
DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE = 24000  # fixed, API-defined
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"  # overridable
DEFAULT_WHISPER_MODEL = "whisper-1"  # fixed

# =============================================================================
# GEMINI PROVIDER (overridable model/voice, fixed API constants)
# =============================================================================
# Auth mode selected from environment:
#   GEMINI_API_KEY -> AI Studio
#   GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_APPLICATION_CREDENTIALS -> Vertex AI
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-live-preview"  # overridable
_LEGACY_GEMINI_MODEL = "gemini-live-2.5-flash-native-audio"
DEFAULT_GEMINI_VOICE = "Zephyr"  # overridable
DEFAULT_GEMINI_THINKING_LEVEL = "high"  # overridable: minimal, low, medium, high
DEFAULT_GEMINI_PROACTIVE_AUDIO = True  # fixed
DEFAULT_GEMINI_LOCATION = "us-central1"  # fixed
DEFAULT_GEMINI_INPUT_SAMPLE_RATE = 16000  # fixed, API-defined
DEFAULT_GEMINI_OUTPUT_SAMPLE_RATE = 24000  # fixed, API-defined

# =============================================================================
# XAI PROVIDER (overridable voice, fixed API constants)
# =============================================================================
DEFAULT_XAI_REALTIME_BASE_URL = "wss://api.x.ai/v1/realtime"  # fixed
DEFAULT_XAI_VOICE = "Ara"  # overridable: Ara, Rex, Sal, Eve, Leo
DEFAULT_XAI_MODEL = "xai-realtime"  # fixed, determined by endpoint

# =============================================================================
# NOVA PROVIDER (overridable model/voice, fixed API constants)
# =============================================================================
DEFAULT_NOVA_MODEL = "amazon.nova-2-sonic-v1:0"  # overridable
DEFAULT_NOVA_VOICE = "tiffany"  # overridable: matthew, tiffany, amy
DEFAULT_NOVA_REGION = "us-east-1"  # fixed
DEFAULT_NOVA_INPUT_SAMPLE_RATE = 16000  # fixed, API-defined
DEFAULT_NOVA_OUTPUT_SAMPLE_RATE = 24000  # fixed, API-defined

# =============================================================================
# QWEN PROVIDER (overridable model/voice, fixed API constants)
# =============================================================================
DEFAULT_QWEN_REALTIME_URL = (
    "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"  # fixed
)
DEFAULT_QWEN_MODEL = "qwen3-omni-flash-realtime"  # overridable
DEFAULT_QWEN_VOICE = "Cherry"  # overridable
DEFAULT_QWEN_INPUT_SAMPLE_RATE = 16000  # fixed, API-defined
DEFAULT_QWEN_OUTPUT_SAMPLE_RATE = 24000  # fixed, API-defined

# =============================================================================
# PROVIDER REGISTRY (derived from above)
# =============================================================================
DEFAULT_AUDIO_NATIVE_MODELS = {
    "openai": DEFAULT_OPENAI_REALTIME_MODEL,
    "gemini": DEFAULT_GEMINI_MODEL,
    "xai": DEFAULT_XAI_MODEL,
    "nova": DEFAULT_NOVA_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
    "livekit": "dummy",
}

AUDIO_NATIVE_PROVIDER_TYPES = {
    "openai": "audio_native",
    "gemini": "audio_native",
    "xai": "audio_native",
    "nova": "audio_native",
    "qwen": "audio_native",
    "livekit": "cascaded",
}

# =============================================================================
# DISPLAY
# =============================================================================
TERM_DARK_MODE = True

"""Configuration for the OpenAI Live voice frontend and delegated backend."""

from pydantic import BaseModel, ConfigDict

DEFAULT_LIVE_FRONTEND_PROMPT_TEMPLATE = (
    "{system_prompt}\n\n"
    "You are the spoken interface for a live support conversation. You control "
    "how and when to speak, listen, stop, interrupt, and delegate. Listen to the "
    "user's complete turn. For every substantive request, clarification, answer, "
    "or reported action, delegate to the backend and remain silent until the "
    "backend result arrives. The backend owns task reasoning, policy decisions, "
    "tool execution, and the substance of your reply. Naturally rephrase only its "
    "concise customer-facing answer, question, or requested customer action. Do "
    "not independently diagnose, invent tool results, repeat the user, or narrate "
    "delegation. Stop speaking immediately when the user speaks."
)

DEFAULT_LIVE_BACKEND_PROMPT_TEMPLATE = (
    "{system_prompt}\n\n"
    "Normalize ten spoken phone-number digits as XXX-XXX-XXXX before calling a "
    "phone lookup."
)


class LiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_model: str
    voice: str = "marin"
    append_system_prompt: bool = True
    frontend_prompt: str | None = None
    backend_prompt: str | None = None

    def render_frontend_prompt(self, system_prompt: str) -> str:
        """Build frontend instructions, using an explicit override when provided."""
        if self.frontend_prompt is not None:
            return self.frontend_prompt
        return DEFAULT_LIVE_FRONTEND_PROMPT_TEMPLATE.format(system_prompt=system_prompt)

    def render_backend_prompt(self, system_prompt: str) -> str:
        """Build backend instructions, using an explicit override when provided."""
        if self.backend_prompt is None:
            return DEFAULT_LIVE_BACKEND_PROMPT_TEMPLATE.format(
                system_prompt=system_prompt
            )
        if self.append_system_prompt:
            return self.backend_prompt + "\n\n" + system_prompt
        return self.backend_prompt

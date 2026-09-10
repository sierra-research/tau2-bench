"""Configuration for the OpenAI Live voice frontend and delegated backend."""

from pydantic import BaseModel, ConfigDict


class LiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend_model: str
    voice: str = "marin"
    append_system_prompt: bool = True
    frontend_prompt: str = (
        "Listen and speak to the customer. Delegate support decisions and "
        "tool use to SpawnThinking. Speak the delegated answer to the customer."
    )
    backend_prompt: str = ""

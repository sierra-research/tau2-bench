from __future__ import annotations

from .seed.agent import create_agent as create_coral_agent


def create_promoted_agent(tools, domain_policy, **kwargs):
    return create_coral_agent(tools=tools, domain_policy=domain_policy, **kwargs)

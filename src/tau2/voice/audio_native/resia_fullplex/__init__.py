"""Resia Fullplex 1 provider (GPT-Live protocol over a WebSocket)."""

from .discrete_time_adapter import DiscreteTimeResiaFullplexAdapter
from .provider import ResiaFullplexProvider

__all__ = ["DiscreteTimeResiaFullplexAdapter", "ResiaFullplexProvider"]

"""Token usage and cost tracking for Anthropic API calls in agent examples."""

from __future__ import annotations

from dataclasses import dataclass, field

# USD per million tokens (standard rates, pre-cache/batch).
PRICING_USD_PER_MTOK = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}


def _usage_fields(response) -> dict | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_read_input_tokens": int(
            getattr(usage, "cache_read_input_tokens", 0) or 0
        ),
        "cache_creation_input_tokens": int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        ),
    }


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_USD_PER_MTOK.get(model, PRICING_USD_PER_MTOK["claude-sonnet-4-6"])
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


def anthropic_to_tau2_usage(input_tokens: int, output_tokens: int) -> dict:
    """Map Anthropic token fields to τ-bench message usage shape."""
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
    }


@dataclass
class UsageRecord:
    component: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


@dataclass
class UsageTracker:
    """Accumulates token usage across agent, supervisor, and user-simulator calls."""

    records: list[UsageRecord] = field(default_factory=list)

    def record(self, component: str, model: str, response) -> None:
        fields = _usage_fields(response)
        if fields is None:
            return
        self.records.append(UsageRecord(component=component, model=model, **fields))

    def totals_by_component(self) -> dict[str, dict]:
        by_comp: dict[str, dict] = {}
        for rec in self.records:
            bucket = by_comp.setdefault(
                rec.component,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += rec.input_tokens
            bucket["output_tokens"] += rec.output_tokens
            bucket["cost_usd"] += rec.cost_usd
        return by_comp

    def summary(self) -> dict:
        total_in = sum(r.input_tokens for r in self.records)
        total_out = sum(r.output_tokens for r in self.records)
        total_cost = sum(r.cost_usd for r in self.records)
        return {
            "api_calls": len(self.records),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": total_cost,
            "by_component": self.totals_by_component(),
        }

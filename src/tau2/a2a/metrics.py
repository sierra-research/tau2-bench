"""Protocol metrics for A2A protocol interactions."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ProtocolMetrics(BaseModel):
    """Performance measurements for A2A protocol interactions."""

    request_id: str
    endpoint: str
    method: str
    status_code: int | None = None
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    context_id: str | None = None
    error: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for export."""
        return self.model_dump(exclude_none=True)


class AggregatedMetrics(BaseModel):
    """Aggregated metrics computed post-run."""

    total_requests: int
    total_tokens: int
    total_latency_ms: float
    avg_latency_ms: float
    error_count: int
    estimated_cost_usd: float | None = None

    @classmethod
    def from_protocol_metrics(
        cls, metrics: list[ProtocolMetrics]
    ) -> "AggregatedMetrics":
        """Compute aggregated metrics from list of protocol metrics."""
        total_requests = len(metrics)
        total_tokens = sum(
            (m.input_tokens or 0) + (m.output_tokens or 0) for m in metrics
        )
        total_latency_ms = sum(m.latency_ms for m in metrics)
        avg_latency_ms = (
            total_latency_ms / total_requests if total_requests > 0 else 0.0
        )
        error_count = sum(1 for m in metrics if m.error is not None)

        return cls(
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            avg_latency_ms=avg_latency_ms,
            error_count=error_count,
        )


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Simple heuristic: ~4 characters per token for English text.
    This is a rough approximation and should be replaced with proper tokenization
    if accurate token counts are required.

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Simple heuristic: ~4 chars per token
    return len(text) // 4

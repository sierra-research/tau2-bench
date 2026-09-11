# Copyright Sierra
"""Runtime judging: decoupled scoring axes over simulations.

Three production judges, one harness shape (``tau2.judges.base``):

- **quality**    — universal deterministic + task-aware binary call rubric.
- **nativeness** — deterministic checkers + per-factor LLM judge over the
  agent's transcript (non-English runs; text or voice; re-judgeable post hoc).
- **delivery**   — multimodal audio judge (fidelity + intonation) over the
  agent's synthesized speech (voice-only; inline at run time, since audio is
  never serialized).

Both axes are siblings of reward — never folded into pass@1. Downstream
(annotation) consumers read typed verdict records from ``tau2.judges.export``.
"""

from tau2.judges.attach import attach_delivery, attach_nativeness, attach_quality
from tau2.judges.delivery.harness import evaluate_delivery
from tau2.judges.nativeness.harness import evaluate_nativeness
from tau2.judges.quality.harness import evaluate_quality

__all__ = [
    "attach_delivery",
    "attach_nativeness",
    "attach_quality",
    "evaluate_delivery",
    "evaluate_nativeness",
    "evaluate_quality",
]

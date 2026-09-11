# Copyright Sierra
"""Nativeness judging (deterministic checkers + LLM judge).

Scores the *agent's* language nativeness and deterministic interaction factors
on a per-simulation [0, 1] scale, as a decoupled axis that is never folded into
reward / pass@1. Language-specific LLM factors apply to multilingual packs;
reviewed deterministic factors may also apply to English.

Two kinds of factor:
- **Deterministic checkers** (``factors.DEFAULT_FACTORS``) — implemented in
  code and selected per language by its pack.
- **LLM-judge factors** (``type="judge"``) — OPT-IN (enable per run via
  ``NativenessJudgeSettings.llm_judge`` / ``--nativeness-llm-judge``); when off
  they are recorded DEFERRED and excluded from the score. Their
  per-language rubrics are authored in each ``pack.yaml`` (``nativeness:``
  block) and must reference the closed catalog in
  ``tau2.multilingual.nativeness_catalog``.
"""

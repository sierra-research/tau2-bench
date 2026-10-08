"""campus domain (Tau2-ZH) — the first native Chinese domain for tau2-bench.

Two version axes by design: the *engine* version lives in ``pyproject.toml``
(``tau2 == 1.0.1``, pinned for leaderboard comparability); the *benchmark*
version below tracks the domain release tags (``campus-vX.Y.Z``).
"""

__version__ = "2.3.0"  # campus benchmark version (tag campus-v2.3.0)
POLICY_VERSION = "1.4.2"
SCORING_PROTOCOL = "tau2-v1.0.1-compatible"

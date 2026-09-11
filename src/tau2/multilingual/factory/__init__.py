# Copyright Sierra
"""Language Factory: the automated pipeline for onboarding new languages.

The Language Factory (``tau2.multilingual``) made a language a pure-data folder;
the factory automates producing that folder — LLM-drafted packs, LLM
translation with verification, calibration — while keeping humans
on the genuine bottlenecks (voice ids, noise recordings, native calibration).

Layout:
- ``state``: persistent per-language factory project under
  ``data/tau2/multilingual/_factory/<lang>/``;
- ``llm``: the thin LLM seam every factory stage calls through (injectable, so
  tests never touch a real model);
- ``guardrails``: deterministic validation of pack drafts and final packs;
- ``cli``: the ``tau2 factory`` subcommand surface.
"""

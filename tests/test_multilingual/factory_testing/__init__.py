# Copyright Sierra
"""Shared test infrastructure for the multilingual Factory test suites.

Everything in this package is offline and deterministic; nothing here imports
``tau2.multilingual.factory`` (which is being built in parallel PRs). The
pieces:

- ``fake_llm``: ``FakeFactoryLLM`` — a scriptable stand-in for the FactoryLLM
  contract (``chat`` / ``json_call``), with call recording and assertions.
- ``toy_language``: a complete valid toy language pack ('tl', Latin script)
  plus a 4-task mini domain and a pre-localized task-set variant.
- ``conftest_helpers``: the ``isolated_pack_env`` fixture — a temporary
  DATA_DIR with the toy pack/domain installed and all loader/registry state
  reset and restored.
- ``results_factory``: on-disk synthetic run directories in the real
  ``Results`` layout, for parity/stop-rule/tag-pivot analysis tests.
"""

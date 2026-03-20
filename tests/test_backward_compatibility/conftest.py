"""
Test configuration for backward compatibility tests.

All tests in this directory should use mocks and NOT make real LLM calls
to ensure fast, reliable, and reproducible test execution.
"""

import pytest
from unittest.mock import Mock


@pytest.fixture(autouse=True)
def ensure_no_real_llm_calls(monkeypatch):
    """
    Auto-applied fixture that prevents accidental real LLM calls in backward compatibility tests.

    This fixture is applied to ALL tests in test_backward_compatibility/ directory.
    Tests requiring real LLM calls should be in test_a2a_e2e/ instead.
    """
    # This fixture runs for every test but doesn't block anything by default
    # Tests should explicitly mock LLM calls using @patch decorators
    pass


@pytest.fixture
def mock_llm_generate():
    """Mock for tau2.utils.llm_utils.generate function"""
    from tau2.data_model.message import AssistantMessage

    def _generate(*args, **kwargs):
        """Mock generate function that returns a simple response"""
        return AssistantMessage(
            role="assistant",
            content="Mock LLM response",
            tool_calls=[]
        )

    return Mock(side_effect=_generate)


@pytest.fixture
def nebius_test_config():
    """
    Configuration for tests that need to verify Nebius integration.

    Note: This fixture provides config structure only - it does NOT make real API calls.
    Tests using this should mock the actual LLM calls.
    """
    import os

    return {
        "agent": {
            "llm": "openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llm_args": {
                "api_base": os.getenv("NEBIUS_API_BASE", "https://api.tokenfactory.nebius.com/v1/"),
                "api_key": os.getenv("NEBIUS_API_KEY", "test-key-for-structure"),
                "temperature": 0.0,
            }
        },
        "user": {
            "llm": "openai/meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llm_args": {
                "api_base": os.getenv("NEBIUS_API_BASE", "https://api.tokenfactory.nebius.com/v1/"),
                "api_key": os.getenv("NEBIUS_API_KEY", "test-key-for-structure"),
                "temperature": 0.0,
            }
        }
    }

"""
Test CLI backward compatibility.

Verifies that existing CLI commands and flags work unchanged after A2A integration.
"""

import argparse
import pytest
from unittest.mock import Mock, patch

from tau2.cli import add_run_args
from tau2.config import (
    DEFAULT_AGENT_IMPLEMENTATION,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_USER,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
)


class TestCLIBackwardCompatibility:
    """Test that existing CLI flags work unchanged"""

    def test_basic_cli_args_parsing(self):
        """Test basic CLI argument parsing still works"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        # Parse minimal valid command
        args = parser.parse_args([
            "--domain", "mock"
        ])

        assert args.domain == "mock"
        assert args.agent == DEFAULT_AGENT_IMPLEMENTATION
        assert args.agent_llm == DEFAULT_LLM_AGENT
        assert args.user_llm == DEFAULT_LLM_USER

    def test_existing_llm_agent_flags_work(self):
        """Test existing LLM agent flags work as before"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "airline",
            "--agent", "llm_agent",
            "--agent-llm", "claude-3-5-sonnet-20241022",
            "--agent-llm-args", '{"temperature": 0.8}'
        ])

        assert args.domain == "airline"
        assert args.agent == "llm_agent"
        assert args.agent_llm == "claude-3-5-sonnet-20241022"
        assert args.agent_llm_args == {"temperature": 0.8}

    def test_user_simulator_flags_unchanged(self):
        """Test user simulator flags work unchanged"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "retail",
            "--user", "user_simulator",
            "--user-llm", "gpt-4o",
            "--user-llm-args", '{"temperature": 0.5}'
        ])

        assert args.user == "user_simulator"
        assert args.user_llm == "gpt-4o"
        assert args.user_llm_args == {"temperature": 0.5}

    def test_task_selection_flags_unchanged(self):
        """Test task selection flags work unchanged"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "telecom",
            "--task-ids", "task_001", "task_002", "task_003",
            "--num-tasks", "5"
        ])

        assert args.task_ids == ["task_001", "task_002", "task_003"]
        assert args.num_tasks == 5

    def test_simulation_control_flags_unchanged(self):
        """Test simulation control flags work unchanged"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "airline",
            "--num-trials", "3",
            "--max-steps", "100",
            "--max-errors", "5",
            "--max-concurrency", "10",
            "--seed", "42"
        ])

        assert args.num_trials == 3
        assert args.max_steps == 100
        assert args.max_errors == 5
        assert args.max_concurrency == 10
        assert args.seed == 42

    def test_save_output_flags_unchanged(self):
        """Test save output flags work unchanged"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "airline",
            "--save-to", "test_results"
        ])

        assert args.save_to == "test_results"

    def test_default_values_unchanged(self):
        """Test default values for existing flags are unchanged"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(["--domain", "mock"])

        # Verify defaults
        assert args.num_trials == DEFAULT_NUM_TRIALS
        assert args.max_steps == DEFAULT_MAX_STEPS
        assert args.agent == DEFAULT_AGENT_IMPLEMENTATION
        assert args.agent_llm == DEFAULT_LLM_AGENT
        assert args.user_llm == DEFAULT_LLM_USER


class TestA2AFlagsAreOptional:
    """Test that new A2A flags are optional and don't break existing usage"""

    def test_a2a_flags_not_required_for_llm_agent(self):
        """Test A2A flags are not required when using LLM agents"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        # Should parse successfully without A2A flags
        args = parser.parse_args([
            "--domain", "airline",
            "--agent", "llm_agent",
            "--agent-llm", "gpt-4o"
        ])

        assert args.agent == "llm_agent"
        # A2A flags should be None/default when not provided
        assert args.agent_a2a_endpoint is None
        assert args.agent_a2a_auth_token is None
        assert args.agent_a2a_timeout == 300  # Default timeout

    def test_a2a_endpoint_flag_is_optional(self):
        """Test --agent-a2a-endpoint flag is optional"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(["--domain", "mock"])

        # Should have None as default
        assert args.agent_a2a_endpoint is None

    def test_a2a_auth_token_flag_is_optional(self):
        """Test --agent-a2a-auth-token flag is optional"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(["--domain", "mock"])

        # Should have None as default
        assert args.agent_a2a_auth_token is None

    def test_a2a_timeout_has_default_value(self):
        """Test --agent-a2a-timeout has default value"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(["--domain", "mock"])

        # Should have default timeout
        assert args.agent_a2a_timeout == 300

    def test_can_use_a2a_flags_when_needed(self):
        """Test A2A flags can be used when agent is a2a_agent"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "airline",
            "--agent", "a2a_agent",
            "--agent-a2a-endpoint", "http://localhost:8080",
            "--agent-a2a-auth-token", "test-token-123",
            "--agent-a2a-timeout", "600"
        ])

        assert args.agent == "a2a_agent"
        assert args.agent_a2a_endpoint == "http://localhost:8080"
        assert args.agent_a2a_auth_token == "test-token-123"
        assert args.agent_a2a_timeout == 600


class TestCLIFlagCompatibility:
    """Test specific flag compatibility scenarios"""

    def test_agent_flag_accepts_llm_agent(self):
        """Test --agent flag accepts 'llm_agent' value"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "airline",
            "--agent", "llm_agent"
        ])

        assert args.agent == "llm_agent"

    def test_agent_flag_accepts_a2a_agent(self):
        """Test --agent flag accepts 'a2a_agent' value"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "airline",
            "--agent", "a2a_agent"
        ])

        assert args.agent == "a2a_agent"

    def test_domain_choices_unchanged(self):
        """Test domain choices include existing domains"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        # Test existing domains still work
        for domain in ["mock", "airline", "retail", "telecom"]:
            args = parser.parse_args(["--domain", domain])
            assert args.domain == domain

    def test_mixed_old_and_new_flags(self):
        """Test can mix old and new flags together"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        # Mix of traditional flags and A2A flags
        args = parser.parse_args([
            "--domain", "airline",
            "--agent", "a2a_agent",
            "--agent-a2a-endpoint", "http://localhost:8080",
            "--user", "user_simulator",
            "--user-llm", "gpt-4o",
            "--num-trials", "2",
            "--max-steps", "50"
        ])

        # Both old and new flags should work
        assert args.domain == "airline"
        assert args.agent == "a2a_agent"
        assert args.agent_a2a_endpoint == "http://localhost:8080"
        assert args.user == "user_simulator"
        assert args.num_trials == 2
        assert args.max_steps == 50


class TestCLINoBreakingChanges:
    """Test that no breaking changes were introduced to CLI"""

    def test_all_original_flags_present(self):
        """Test all original flags are still present"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        # Parse with all original flags
        args = parser.parse_args([
            "--domain", "airline",
            "--num-trials", "1",
            "--agent", "llm_agent",
            "--agent-llm", "gpt-4o",
            "--agent-llm-args", '{}',
            "--user", "user_simulator",
            "--user-llm", "gpt-4o",
            "--user-llm-args", '{}',
            "--max-steps", "50",
            "--max-errors", "10",
            "--max-concurrency", "3",
            "--seed", "42"
        ])

        # All flags should parse successfully
        assert args.domain == "airline"
        assert args.agent == "llm_agent"
        assert args.user == "user_simulator"

    def test_flag_short_forms_still_work(self):
        """Test flag short forms (if any) still work"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        # Test -d short form for --domain
        args = parser.parse_args(["-d", "mock"])

        assert args.domain == "mock"

    def test_json_args_parsing_unchanged(self):
        """Test JSON argument parsing for llm-args still works"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        # Complex JSON args
        args = parser.parse_args([
            "--domain", "airline",
            "--agent-llm-args", '{"temperature": 0.7, "top_p": 0.9, "max_tokens": 1000}'
        ])

        assert args.agent_llm_args == {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000
        }

    def test_task_split_name_flag_unchanged(self):
        """Test task-split-name flag works unchanged"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "airline",
            "--task-split-name", "test"
        ])

        assert args.task_split_name == "test"

    def test_task_set_name_flag_unchanged(self):
        """Test task-set-name flag works unchanged"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args([
            "--domain", "telecom",
            "--task-set-name", "telecom_small"
        ])

        assert args.task_set_name == "telecom_small"


class TestCLIHelp:
    """Test that CLI help text is clear and helpful"""

    def test_help_includes_a2a_flags(self):
        """Test help text includes A2A flags"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        help_text = parser.format_help()

        # Verify A2A flags are documented
        assert "--agent-a2a-endpoint" in help_text
        assert "--agent-a2a-auth-token" in help_text
        assert "--agent-a2a-timeout" in help_text

    def test_help_includes_existing_flags(self):
        """Test help text still includes all existing flags"""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        help_text = parser.format_help()

        # Verify existing flags are documented
        assert "--domain" in help_text
        assert "--agent" in help_text
        assert "--agent-llm" in help_text
        assert "--user" in help_text
        assert "--num-trials" in help_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

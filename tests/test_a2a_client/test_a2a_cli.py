"""Tests for A2A CLI flag integration.

Verifies that the new A2A flags (--agent-a2a-endpoint, --agent-a2a-auth-token,
--agent-a2a-timeout) are properly wired into the CLI, coexist with existing
flags, and flow through to TextRunConfig.
"""

import argparse

from tau2.cli import add_run_args


class TestA2AFlagsAreOptional:
    """A2A flags must not interfere with non-A2A usage."""

    def test_a2a_flags_absent_by_default(self):
        """Parsing without A2A flags yields None/default values."""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(["--domain", "airline", "--agent", "llm_agent"])

        assert args.agent_a2a_endpoint is None
        assert args.agent_a2a_auth_token is None
        assert args.agent_a2a_timeout == 300

    def test_a2a_flags_accepted_when_provided(self):
        """All three A2A flags parse correctly."""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(
            [
                "--domain",
                "airline",
                "--agent",
                "a2a_agent",
                "--agent-a2a-endpoint",
                "http://localhost:8080",
                "--agent-a2a-auth-token",
                "test-token",
                "--agent-a2a-timeout",
                "600",
            ]
        )

        assert args.agent == "a2a_agent"
        assert args.agent_a2a_endpoint == "http://localhost:8080"
        assert args.agent_a2a_auth_token == "test-token"
        assert args.agent_a2a_timeout == 600


class TestA2AAgentInChoices:
    """a2a_agent must appear in the --agent choices via registry."""

    def test_agent_flag_accepts_a2a_agent(self):
        """--agent a2a_agent parses without error."""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(["--domain", "airline", "--agent", "a2a_agent"])
        assert args.agent == "a2a_agent"


class TestA2ACliToConfig:
    """CLI args must flow through to the config objects unchanged."""

    def test_a2a_agent_args_dict_construction(self):
        """Parsed CLI args produce the dict shape create_a2a_agent expects."""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(
            [
                "--domain",
                "airline",
                "--agent",
                "a2a_agent",
                "--agent-a2a-endpoint",
                "http://localhost:8080",
                "--agent-a2a-auth-token",
                "tok-123",
                "--agent-a2a-timeout",
                "600",
            ]
        )

        a2a_agent_args = {
            "endpoint": args.agent_a2a_endpoint,
            "auth_token": args.agent_a2a_auth_token,
            "timeout": args.agent_a2a_timeout,
        }

        assert a2a_agent_args == {
            "endpoint": "http://localhost:8080",
            "auth_token": "tok-123",
            "timeout": 600,
        }

    def test_text_run_config_accepts_a2a_agent_args(self):
        """TextRunConfig stores a2a_agent_args without error."""
        from tau2.data_model.simulation import TextRunConfig

        config = TextRunConfig(
            domain="airline",
            a2a_agent_args={
                "endpoint": "http://localhost:9090",
                "auth_token": "secret",
                "timeout": 120,
            },
        )

        assert config.a2a_agent_args == {
            "endpoint": "http://localhost:9090",
            "auth_token": "secret",
            "timeout": 120,
        }

    def test_a2a_flags_coexist_with_existing_flags(self):
        """A2A flags parse correctly alongside traditional flags."""
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        args = parser.parse_args(
            [
                "--domain",
                "airline",
                "--agent",
                "a2a_agent",
                "--agent-a2a-endpoint",
                "http://localhost:8080",
                "--user",
                "user_simulator",
                "--user-llm",
                "gpt-4o",
                "--num-trials",
                "2",
                "--max-steps",
                "50",
            ]
        )

        assert args.agent == "a2a_agent"
        assert args.agent_a2a_endpoint == "http://localhost:8080"
        assert args.num_trials == 2
        assert args.max_steps == 50


class TestA2AHelpText:
    """A2A flags must appear in --help output."""

    def test_help_includes_a2a_flags(self):
        parser = argparse.ArgumentParser()
        add_run_args(parser)

        help_text = parser.format_help()

        assert "--agent-a2a-endpoint" in help_text
        assert "--agent-a2a-auth-token" in help_text
        assert "--agent-a2a-timeout" in help_text

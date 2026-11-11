"""Tests for the adversarial evaluation module."""

import pytest

from tau_adversarial import (
    AdversarialStrategy,
    AttackConfig,
    get_adversarial_instructions,
    get_domain_attack_goals,
    load_adversarial_tasks,
)
from tau_adversarial.strategies import (
    SOCIAL_ENGINEERING_INSTRUCTIONS,
    PROMPT_INJECTION_INSTRUCTIONS,
)
from tau_adversarial.evaluator_safety import (
    SafetyEvaluator,
    SafetyViolation,
    ViolationType,
)


class TestAdversarialStrategy:
    """Tests for AdversarialStrategy enum."""

    def test_all_strategies_defined(self):
        """All expected strategies should be defined."""
        strategies = [s.value for s in AdversarialStrategy]
        assert "social_engineering" in strategies
        assert "prompt_injection" in strategies
        assert "policy_exploitation" in strategies
        assert "identity_manipulation" in strategies
        assert "information_extraction" in strategies

    def test_strategy_from_string(self):
        """Should be able to create strategy from string."""
        strategy = AdversarialStrategy("social_engineering")
        assert strategy == AdversarialStrategy.SOCIAL_ENGINEERING


class TestAttackConfig:
    """Tests for AttackConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = AttackConfig(strategy=AdversarialStrategy.SOCIAL_ENGINEERING)
        assert config.sophistication == 0.5
        assert config.domain == "airline"
        assert config.persistence == 3
        assert config.escalation is True

    def test_sophistication_clamping(self):
        """Sophistication should be clamped to [0, 1]."""
        config_low = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=-0.5
        )
        assert config_low.sophistication == 0.0

        config_high = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=1.5
        )
        assert config_high.sophistication == 1.0

    def test_with_target_goal(self):
        """Should accept target goal."""
        config = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            target_goal="Get unauthorized refund"
        )
        assert config.target_goal == "Get unauthorized refund"


class TestGetAdversarialInstructions:
    """Tests for instruction generation."""

    def test_generates_instructions(self):
        """Should generate non-empty instructions."""
        config = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=0.5,
        )
        instructions = get_adversarial_instructions(config)
        assert instructions
        assert len(instructions) > 100

    def test_includes_persistence(self):
        """Instructions should include persistence value."""
        config = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=0.5,
            persistence=5,
        )
        instructions = get_adversarial_instructions(config)
        assert "5" in instructions

    def test_includes_target_goal(self):
        """Instructions should include target goal when specified."""
        config = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=0.5,
            target_goal="Get free upgrade",
        )
        instructions = get_adversarial_instructions(config)
        assert "Get free upgrade" in instructions

    def test_escalation_instructions(self):
        """Should include escalation instructions when enabled."""
        config = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=0.5,
            escalation=True,
        )
        instructions = get_adversarial_instructions(config)
        assert "ESCALATION" in instructions

    def test_all_strategies_have_instructions(self):
        """All strategies should generate valid instructions."""
        for strategy in AdversarialStrategy:
            config = AttackConfig(strategy=strategy, sophistication=0.5)
            instructions = get_adversarial_instructions(config)
            assert instructions
            assert len(instructions) > 50


class TestDomainAttackGoals:
    """Tests for domain-specific attack goals."""

    def test_airline_goals(self):
        """Airline domain should have relevant attack goals."""
        goals = get_domain_attack_goals("airline")
        assert len(goals) > 0
        # Check for some expected goals
        goal_text = " ".join(goals).lower()
        assert "refund" in goal_text or "cancel" in goal_text

    def test_retail_goals(self):
        """Retail domain should have relevant attack goals."""
        goals = get_domain_attack_goals("retail")
        assert len(goals) > 0

    def test_telecom_goals(self):
        """Telecom domain should have relevant attack goals."""
        goals = get_domain_attack_goals("telecom")
        assert len(goals) > 0

    def test_unknown_domain_fallback(self):
        """Unknown domain should fallback to airline goals."""
        goals = get_domain_attack_goals("unknown_domain")
        airline_goals = get_domain_attack_goals("airline")
        assert goals == airline_goals


class TestSafetyViolation:
    """Tests for SafetyViolation dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        violation = SafetyViolation(
            type=ViolationType.UNAUTHORIZED_ACTION,
            severity=0.8,
            description="Test violation",
            turn_index=5,
            evidence="Agent performed unauthorized action",
        )
        result = violation.to_dict()
        assert result["type"] == "unauthorized_action"
        assert result["severity"] == 0.8
        assert result["turn_index"] == 5

    def test_evidence_truncation(self):
        """Long evidence should be truncated."""
        long_evidence = "A" * 500
        violation = SafetyViolation(
            type=ViolationType.INFORMATION_DISCLOSURE,
            severity=0.5,
            description="Test",
            turn_index=1,
            evidence=long_evidence,
        )
        result = violation.to_dict()
        assert len(result["evidence"]) == 200


class TestSafetyEvaluator:
    """Tests for SafetyEvaluator."""

    def test_initialization(self):
        """Should initialize with domain."""
        evaluator = SafetyEvaluator(domain="airline")
        assert evaluator.domain == "airline"

    def test_dangerous_actions_by_domain(self):
        """Should have domain-specific dangerous actions."""
        airline_eval = SafetyEvaluator(domain="airline")
        assert "cancel_reservation" in airline_eval._dangerous_actions
        assert "send_certificate" in airline_eval._dangerous_actions

        retail_eval = SafetyEvaluator(domain="retail")
        assert "process_refund" in retail_eval._dangerous_actions

    def test_sensitive_keywords(self):
        """Should have sensitive keywords."""
        evaluator = SafetyEvaluator(domain="airline")
        assert len(evaluator._sensitive_keywords) > 0
        assert "internal policy" in evaluator._sensitive_keywords


class TestLoadAdversarialTasks:
    """Tests for loading adversarial tasks."""

    def test_load_airline_tasks(self):
        """Should load airline adversarial tasks."""
        tasks = load_adversarial_tasks("airline")
        assert len(tasks) > 0

    def test_task_structure(self):
        """Tasks should have required fields."""
        tasks = load_adversarial_tasks("airline")
        if tasks:
            task = tasks[0]
            assert task.id
            assert task.user_scenario

    def test_missing_domain_returns_empty(self):
        """Missing domain should return empty list."""
        tasks = load_adversarial_tasks("nonexistent_domain")
        assert tasks == []


class TestSophisticationLevels:
    """Tests for sophistication level variations."""

    def test_low_sophistication_is_direct(self):
        """Low sophistication should be more direct."""
        config = AttackConfig(
            strategy=AdversarialStrategy.PROMPT_INJECTION,
            sophistication=0.0,
        )
        instructions = get_adversarial_instructions(config)
        # Low sophistication uses direct phrases
        assert "ignore" in instructions.lower() or "override" in instructions.lower()

    def test_high_sophistication_is_subtle(self):
        """High sophistication should be more subtle."""
        config = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=1.0,
        )
        instructions = get_adversarial_instructions(config)
        # High sophistication mentions subtlety
        assert "confident" in instructions.lower() or "authority" in instructions.lower()

    def test_different_levels_different_instructions(self):
        """Different sophistication levels should produce different instructions."""
        config_low = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=0.0,
        )
        config_high = AttackConfig(
            strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
            sophistication=1.0,
        )
        instr_low = get_adversarial_instructions(config_low)
        instr_high = get_adversarial_instructions(config_high)
        # Should be meaningfully different
        assert instr_low != instr_high

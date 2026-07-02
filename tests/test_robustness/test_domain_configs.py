"""Tests for domain-specific injection YAML configurations."""

from pathlib import Path

import pytest

from tau_robustness.injection_config import InjectionConfig, InjectionType

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tau2" / "injections"


class TestRetailConfig:
    @pytest.fixture
    def config(self):
        return InjectionConfig.from_yaml(DATA_DIR / "retail_injections.yaml")

    def test_loads_successfully(self, config):
        assert config.domain == "retail"
        assert len(config.injections) >= 3

    def test_all_injections_have_required_fields(self, config):
        for inj in config.injections:
            assert inj.id, f"Injection missing id"
            assert inj.type, f"Injection {inj.id} missing type"
            assert inj.target_tool, f"Injection {inj.id} missing target_tool"
            assert inj.description, f"Injection {inj.id} missing description"
            assert len(inj.mutations) > 0, f"Injection {inj.id} has no mutations"
            assert 1 <= inj.difficulty <= 3, (
                f"Injection {inj.id} has invalid difficulty {inj.difficulty}"
            )

    def test_targets_real_tools(self, config):
        valid_tools = {
            "get_order_details",
            "get_user_details",
            "get_product_details",
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
            "list_all_product_types",
        }
        for inj in config.injections:
            assert inj.target_tool in valid_tools, (
                f"Injection {inj.id} targets unknown tool '{inj.target_tool}'"
            )

    def test_has_detection_signals(self, config):
        for inj in config.injections:
            assert len(inj.detection_signals) > 0, (
                f"Injection {inj.id} has no detection signals"
            )

    def test_has_recovery_signals(self, config):
        for inj in config.injections:
            assert len(inj.recovery_signals) > 0, (
                f"Injection {inj.id} has no recovery signals"
            )

    def test_covers_multiple_injection_types(self, config):
        types_covered = {inj.type for inj in config.injections}
        assert len(types_covered) >= 3, (
            f"Only covers {len(types_covered)} injection types: {types_covered}"
        )


class TestAirlineConfig:
    @pytest.fixture
    def config(self):
        return InjectionConfig.from_yaml(DATA_DIR / "airline_injections.yaml")

    def test_loads_successfully(self, config):
        assert config.domain == "airline"
        assert len(config.injections) >= 3

    def test_all_injections_have_required_fields(self, config):
        for inj in config.injections:
            assert inj.id
            assert inj.type
            assert inj.target_tool
            assert len(inj.mutations) > 0

    def test_targets_real_tools(self, config):
        valid_tools = {
            "get_user_details",
            "get_reservation_details",
            "search_direct_flight",
            "search_onestop_flight",
            "get_flight_details",
            "get_flight_status",
            "list_all_airports",
        }
        for inj in config.injections:
            assert inj.target_tool in valid_tools, (
                f"Injection {inj.id} targets unknown tool '{inj.target_tool}'"
            )


class TestTelecomConfig:
    @pytest.fixture
    def config(self):
        return InjectionConfig.from_yaml(DATA_DIR / "telecom_injections.yaml")

    def test_loads_successfully(self, config):
        assert config.domain == "telecom"
        assert len(config.injections) >= 7

    def test_all_injections_have_required_fields(self, config):
        for inj in config.injections:
            assert inj.id
            assert inj.type
            assert inj.target_tool
            assert len(inj.mutations) > 0

    def test_targets_real_tools(self, config):
        # Both agent tools and user tools are valid targets
        valid_tools = {
            # Agent tools
            "get_customer_by_phone",
            "get_customer_by_id",
            "get_customer_by_name",
            "get_details_by_id",
            "get_data_usage",
            "get_bills_for_customer",
            # User tools
            "check_network_status",
            "check_sim_status",
            "check_apn_settings",
            "run_speed_test",
            "check_status_bar",
        }
        for inj in config.injections:
            assert inj.target_tool in valid_tools, (
                f"Injection {inj.id} targets unknown tool '{inj.target_tool}'"
            )

    def test_has_detection_and_recovery_signals(self, config):
        for inj in config.injections:
            assert len(inj.detection_signals) > 0, (
                f"Injection {inj.id} has no detection signals"
            )
            assert len(inj.recovery_signals) > 0, (
                f"Injection {inj.id} has no recovery signals"
            )

    def test_targets_both_agent_and_user_tools(self, config):
        """Telecom is dual-control — injections should target both sides."""
        agent_tools = {"get_customer_by_phone", "get_data_usage", "get_details_by_id"}
        user_tools = {"check_network_status", "check_sim_status", "check_apn_settings"}
        targeted = {inj.target_tool for inj in config.injections}
        has_agent = bool(targeted & agent_tools)
        has_user = bool(targeted & user_tools)
        assert has_agent and has_user, (
            f"Expected both agent and user tools targeted, got: {targeted}"
        )


class TestFromDomain:
    def test_load_retail(self):
        config = InjectionConfig.from_domain("retail", data_dir=DATA_DIR)
        assert config.domain == "retail"

    def test_load_airline(self):
        config = InjectionConfig.from_domain("airline", data_dir=DATA_DIR)
        assert config.domain == "airline"

    def test_load_telecom(self):
        config = InjectionConfig.from_domain("telecom", data_dir=DATA_DIR)
        assert config.domain == "telecom"

    def test_load_unknown_domain(self):
        with pytest.raises(FileNotFoundError, match="No injection config found"):
            InjectionConfig.from_domain("nonexistent_domain", data_dir=DATA_DIR)

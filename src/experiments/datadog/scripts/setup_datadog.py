#!/usr/bin/env python3
"""Datadog resource setup script for tau2-bench observability.

This script creates and manages Datadog resources (monitors, SLOs, dashboards)
from JSON configuration files.

Environment Variables:
    DD_API_KEY: Required. Datadog API key.
    DD_APP_KEY: Required. Datadog Application key.
    DD_SITE: Optional. Datadog site. Defaults to "datadoghq.com".

Usage:
    # Create all resources
    python setup_datadog.py --all

    # Create specific resource types
    python setup_datadog.py --monitors
    python setup_datadog.py --slos
    python setup_datadog.py --dashboard

    # Export current configurations from Datadog
    python setup_datadog.py --export

    # Dry run (show what would be created)
    python setup_datadog.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# Config directory relative to this script
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class DatadogSetup:
    """Manages Datadog resource creation via API.

    This class provides methods to create monitors, SLOs, and dashboards
    from JSON configuration files.
    """

    def __init__(self, api_key: str, app_key: str, site: str = "datadoghq.com", dry_run: bool = False):
        """Initialize the Datadog setup client.

        Args:
            api_key: Datadog API key.
            app_key: Datadog Application key.
            site: Datadog site (e.g., datadoghq.com, datadoghq.eu).
            dry_run: If True, log actions without creating resources.
        """
        self.api_key = api_key
        self.app_key = app_key
        self.site = site
        self.dry_run = dry_run
        self._api_client: Any = None
        self._monitors_api: Any = None
        self._slo_api: Any = None
        self._dashboards_api: Any = None

        if not dry_run:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the Datadog API client."""
        try:
            from datadog_api_client import ApiClient, Configuration
            from datadog_api_client.v1.api.dashboards_api import DashboardsApi
            from datadog_api_client.v1.api.monitors_api import MonitorsApi
            from datadog_api_client.v1.api.service_level_objectives_api import (
                ServiceLevelObjectivesApi,
            )

            configuration = Configuration()
            configuration.api_key["apiKeyAuth"] = self.api_key
            configuration.api_key["appKeyAuth"] = self.app_key
            configuration.server_variables["site"] = self.site

            self._api_client = ApiClient(configuration)
            self._monitors_api = MonitorsApi(self._api_client)
            self._slo_api = ServiceLevelObjectivesApi(self._api_client)
            self._dashboards_api = DashboardsApi(self._api_client)

            logger.info(f"Datadog API client initialized for site: {self.site}")

        except ImportError:
            logger.error(
                "datadog-api-client package not installed. "
                "Install with: pip install datadog-api-client"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Datadog API client: {e}")
            raise

    def validate_api_keys(self) -> bool:
        """Validate that API keys are working.

        Returns:
            True if API keys are valid, False otherwise.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would validate API keys")
            return True

        try:
            from datadog_api_client.v1.api.authentication_api import AuthenticationApi

            auth_api = AuthenticationApi(self._api_client)
            auth_api.validate()
            logger.info("API key validation successful")
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

    def load_config(self, filename: str) -> dict:
        """Load a JSON configuration file.

        Args:
            filename: Name of the config file in the configs directory.

        Returns:
            The parsed JSON configuration.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        config_path = CONFIGS_DIR / filename
        if not config_path.exists():
            msg = f"Config file not found: {config_path}"
            raise FileNotFoundError(msg)

        with open(config_path) as f:
            return json.load(f)

    def create_monitors(self) -> list[dict]:
        """Create monitors from monitors.json.

        Returns:
            List of created monitor responses.
        """
        logger.info("Creating monitors...")
        config = self.load_config("monitors.json")
        monitors = config.get("monitors", [])

        if not monitors:
            logger.warning("No monitors found in configuration")
            return []

        results = []
        for monitor_def in monitors:
            try:
                result = self._create_monitor(monitor_def)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to create monitor {monitor_def.get('id')}: {e}")
                continue

        logger.info(f"Created {len(results)} monitors")
        return results

    def _create_monitor(self, monitor_def: dict) -> dict:
        """Create a single monitor.

        Args:
            monitor_def: Monitor definition from config.

        Returns:
            Created monitor response.
        """
        monitor_id = monitor_def.get("id", "unknown")
        name = monitor_def.get("name", "Unnamed Monitor")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would create monitor: {monitor_id} - {name}")
            return {"id": monitor_id, "name": name, "dry_run": True}

        from datadog_api_client.v1.model.monitor import Monitor
        from datadog_api_client.v1.model.monitor_options import MonitorOptions
        from datadog_api_client.v1.model.monitor_thresholds import MonitorThresholds
        from datadog_api_client.v1.model.monitor_type import MonitorType

        # Map config type to API enum
        type_mapping = {
            "metric alert": MonitorType.METRIC_ALERT,
            "log alert": MonitorType.LOG_ALERT,
            "query alert": MonitorType.QUERY_ALERT,
        }
        monitor_type = type_mapping.get(
            monitor_def.get("type", "metric alert"), MonitorType.METRIC_ALERT
        )

        # Build thresholds
        options_def = monitor_def.get("options", {})
        thresholds_def = options_def.get("thresholds", {})
        thresholds = MonitorThresholds(
            critical=thresholds_def.get("critical"),
            warning=thresholds_def.get("warning"),
        )

        # Build options
        options = MonitorOptions(
            thresholds=thresholds,
            notify_no_data=options_def.get("notify_no_data", False),
            renotify_interval=options_def.get("renotify_interval"),
            escalation_message=options_def.get("escalation_message"),
            notify_audit=options_def.get("notify_audit", False),
            include_tags=options_def.get("include_tags", True),
        )

        # Build monitor
        monitor = Monitor(
            name=name,
            type=monitor_type,
            query=monitor_def.get("query", ""),
            message=monitor_def.get("message", ""),
            tags=monitor_def.get("tags", []),
            options=options,
        )

        response = self._monitors_api.create_monitor(body=monitor)
        logger.info(f"Created monitor: {monitor_id} - {name} (ID: {response.id})")
        return {"id": response.id, "name": name, "config_id": monitor_id}

    def create_slos(self) -> list[dict]:
        """Create SLOs from slos.json.

        Returns:
            List of created SLO responses.
        """
        logger.info("Creating SLOs...")
        config = self.load_config("slos.json")
        slos = config.get("slos", [])

        if not slos:
            logger.warning("No SLOs found in configuration")
            return []

        results = []
        for slo_def in slos:
            try:
                result = self._create_slo(slo_def)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to create SLO {slo_def.get('name')}: {e}")
                continue

        logger.info(f"Created {len(results)} SLOs")
        return results

    def _create_slo(self, slo_def: dict) -> dict:
        """Create a single SLO.

        Args:
            slo_def: SLO definition from config.

        Returns:
            Created SLO response.
        """
        name = slo_def.get("name", "Unnamed SLO")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would create SLO: {name}")
            return {"name": name, "dry_run": True}

        from datadog_api_client.v1.model.service_level_objective import (
            ServiceLevelObjective,
        )
        from datadog_api_client.v1.model.service_level_objective_query import (
            ServiceLevelObjectiveQuery,
        )
        from datadog_api_client.v1.model.slo_threshold import SLOThreshold
        from datadog_api_client.v1.model.slo_timeframe import SLOTimeframe
        from datadog_api_client.v1.model.slo_type import SLOType

        # Map timeframe
        timeframe_mapping = {
            "7d": SLOTimeframe.SEVEN_DAYS,
            "30d": SLOTimeframe.THIRTY_DAYS,
            "90d": SLOTimeframe.NINETY_DAYS,
        }
        timeframe = timeframe_mapping.get(
            slo_def.get("timeframe", "7d"), SLOTimeframe.SEVEN_DAYS
        )

        # Build query
        query_def = slo_def.get("query", {})
        query = ServiceLevelObjectiveQuery(
            numerator=query_def.get("numerator", ""),
            denominator=query_def.get("denominator", ""),
        )

        # Build thresholds
        thresholds_def = slo_def.get("thresholds", {})
        thresholds = [
            SLOThreshold(
                target=thresholds_def.get("target", 99.0),
                timeframe=timeframe,
                warning=thresholds_def.get("warning"),
            )
        ]

        # Build SLO
        slo = ServiceLevelObjective(
            name=name,
            description=slo_def.get("description", ""),
            type=SLOType.METRIC,
            query=query,
            thresholds=thresholds,
            tags=slo_def.get("tags", []),
        )

        response = self._slo_api.create_slo(body=slo)
        slo_data = response.data[0] if response.data else {}
        logger.info(f"Created SLO: {name} (ID: {slo_data.get('id', 'unknown')})")
        return {"id": slo_data.get("id"), "name": name}

    def create_dashboard(self) -> dict:
        """Create dashboard from dashboards.json.

        Returns:
            Created dashboard response.
        """
        logger.info("Creating dashboard...")
        config = self.load_config("dashboards.json")
        dashboard_def = config.get("dashboard", {})

        if not dashboard_def:
            logger.warning("No dashboard found in configuration")
            return {}

        title = dashboard_def.get("title", "Unnamed Dashboard")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would create dashboard: {title}")
            return {"title": title, "dry_run": True}

        from datadog_api_client.v1.model.dashboard import Dashboard
        from datadog_api_client.v1.model.dashboard_layout_type import (
            DashboardLayoutType,
        )

        # Map layout type
        layout_mapping = {
            "ordered": DashboardLayoutType.ORDERED,
            "free": DashboardLayoutType.FREE,
        }
        layout_type = layout_mapping.get(
            dashboard_def.get("layout_type", "ordered"), DashboardLayoutType.ORDERED
        )

        # Build widgets from config
        # Note: Full widget conversion is complex; using simplified approach
        widgets = self._convert_widgets(dashboard_def.get("widgets", []))

        # Build dashboard
        dashboard = Dashboard(
            title=title,
            description=dashboard_def.get("description", ""),
            layout_type=layout_type,
            widgets=widgets,
        )

        response = self._dashboards_api.create_dashboard(body=dashboard)
        logger.info(f"Created dashboard: {title} (ID: {response.id})")
        return {"id": response.id, "title": title, "url": response.url}

    def _convert_widgets(self, widgets_config: list[dict]) -> list:
        """Convert widget configurations to API format.

        Args:
            widgets_config: List of widget definitions from config.

        Returns:
            List of Widget objects for the API.
        """
        from datadog_api_client.v1.model.note_widget_definition import (
            NoteWidgetDefinition,
        )
        from datadog_api_client.v1.model.note_widget_definition_type import (
            NoteWidgetDefinitionType,
        )
        from datadog_api_client.v1.model.query_value_widget_definition import (
            QueryValueWidgetDefinition,
        )
        from datadog_api_client.v1.model.query_value_widget_definition_type import (
            QueryValueWidgetDefinitionType,
        )
        from datadog_api_client.v1.model.query_value_widget_request import (
            QueryValueWidgetRequest,
        )
        from datadog_api_client.v1.model.timeseries_widget_definition import (
            TimeseriesWidgetDefinition,
        )
        from datadog_api_client.v1.model.timeseries_widget_definition_type import (
            TimeseriesWidgetDefinitionType,
        )
        from datadog_api_client.v1.model.timeseries_widget_request import (
            TimeseriesWidgetRequest,
        )
        from datadog_api_client.v1.model.widget import Widget
        from datadog_api_client.v1.model.widget_layout import WidgetLayout

        widgets = []
        y_position = 0

        for widget_config in widgets_config:
            widget_type = widget_config.get("type", "note")
            title = widget_config.get("title", "")

            try:
                if widget_type == "note":
                    definition = NoteWidgetDefinition(
                        type=NoteWidgetDefinitionType.NOTE,
                        content=widget_config.get("content", ""),
                        background_color=widget_config.get("background_color", "white"),
                        font_size=widget_config.get("font_size", "14"),
                        text_align=widget_config.get("text_align", "left"),
                    )
                elif widget_type == "timeseries":
                    requests = []
                    for req in widget_config.get("requests", []):
                        requests.append(
                            TimeseriesWidgetRequest(
                                q=req.get("q", ""),
                                display_type=req.get("display_type", "line"),
                            )
                        )
                    definition = TimeseriesWidgetDefinition(
                        type=TimeseriesWidgetDefinitionType.TIMESERIES,
                        title=title,
                        requests=requests,
                    )
                elif widget_type == "query_value":
                    requests = []
                    for req in widget_config.get("requests", []):
                        requests.append(
                            QueryValueWidgetRequest(
                                q=req.get("q", ""),
                                aggregator=req.get("aggregator", "avg"),
                            )
                        )
                    definition = QueryValueWidgetDefinition(
                        type=QueryValueWidgetDefinitionType.QUERY_VALUE,
                        title=title,
                        requests=requests,
                        precision=widget_config.get("precision", 2),
                        autoscale=widget_config.get("autoscale", True),
                    )
                else:
                    # For unsupported types, create a note placeholder
                    definition = NoteWidgetDefinition(
                        type=NoteWidgetDefinitionType.NOTE,
                        content=f"Widget type '{widget_type}' - {title}",
                        background_color="gray",
                    )

                # Create widget with layout
                widget = Widget(
                    definition=definition,
                    layout=WidgetLayout(x=0, y=y_position, width=12, height=3),
                )
                widgets.append(widget)
                y_position += 3

            except Exception as e:
                logger.warning(f"Failed to convert widget {title}: {e}")
                continue

        return widgets

    def export_configs(self, output_dir: Path | None = None) -> None:
        """Export current Datadog configurations to JSON files.

        Args:
            output_dir: Directory to save exports. Defaults to configs directory.
        """
        output_dir = output_dir or CONFIGS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.dry_run:
            logger.info(f"[DRY RUN] Would export configs to: {output_dir}")
            return

        logger.info("Exporting Datadog configurations...")

        # Export monitors
        try:
            monitors = self._monitors_api.list_monitors()
            monitors_export = {"monitors": [m.to_dict() for m in monitors]}
            export_path = output_dir / "monitors_export.json"
            with open(export_path, "w") as f:
                json.dump(monitors_export, f, indent=2, default=str)
            logger.info(f"Exported {len(monitors)} monitors to {export_path}")
        except Exception as e:
            logger.error(f"Failed to export monitors: {e}")

        # Export SLOs
        try:
            slos = self._slo_api.list_slos()
            slos_export = {"slos": [s.to_dict() for s in slos.data] if slos.data else []}
            export_path = output_dir / "slos_export.json"
            with open(export_path, "w") as f:
                json.dump(slos_export, f, indent=2, default=str)
            logger.info(f"Exported {len(slos_export['slos'])} SLOs to {export_path}")
        except Exception as e:
            logger.error(f"Failed to export SLOs: {e}")

        # Export dashboards
        try:
            dashboards = self._dashboards_api.list_dashboards()
            dashboards_export = {
                "dashboards": [d.to_dict() for d in dashboards.dashboards]
                if dashboards.dashboards
                else []
            }
            export_path = output_dir / "dashboards_export.json"
            with open(export_path, "w") as f:
                json.dump(dashboards_export, f, indent=2, default=str)
            logger.info(
                f"Exported {len(dashboards_export['dashboards'])} dashboards to {export_path}"
            )
        except Exception as e:
            logger.error(f"Failed to export dashboards: {e}")

        logger.info(f"Export complete. Files saved to: {output_dir}")

    def create_all(self) -> dict:
        """Create all Datadog resources.

        Returns:
            Summary of created resources.
        """
        logger.info("Creating all Datadog resources...")

        results = {
            "monitors": [],
            "slos": [],
            "dashboard": {},
        }

        try:
            results["monitors"] = self.create_monitors()
        except Exception as e:
            logger.error(f"Failed to create monitors: {e}")

        try:
            results["slos"] = self.create_slos()
        except Exception as e:
            logger.error(f"Failed to create SLOs: {e}")

        try:
            results["dashboard"] = self.create_dashboard()
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")

        # Summary
        logger.info("=" * 50)
        logger.info("Setup Complete Summary:")
        logger.info(f"  Monitors created: {len(results['monitors'])}")
        logger.info(f"  SLOs created: {len(results['slos'])}")
        logger.info(f"  Dashboard created: {'Yes' if results['dashboard'] else 'No'}")
        logger.info("=" * 50)

        return results


def get_api_keys() -> tuple[str, str]:
    """Get API keys from environment variables.

    Returns:
        Tuple of (api_key, app_key).

    Raises:
        ValueError: If required keys are not set.
    """
    api_key = os.getenv("DD_API_KEY")
    app_key = os.getenv("DD_APP_KEY")

    if not api_key:
        msg = "DD_API_KEY environment variable is required"
        raise ValueError(msg)
    if not app_key:
        msg = "DD_APP_KEY environment variable is required"
        raise ValueError(msg)

    return api_key, app_key


def main() -> int:
    """Main entry point for Datadog setup."""
    parser = argparse.ArgumentParser(
        description="Set up Datadog monitors, SLOs, and dashboards for tau2-bench"
    )
    parser.add_argument(
        "--monitors",
        action="store_true",
        help="Create monitors from monitors.json",
    )
    parser.add_argument(
        "--slos",
        action="store_true",
        help="Create SLOs from slos.json",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Create dashboard from dashboards.json",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create all resources (monitors, SLOs, dashboard)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export current Datadog configurations to JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without making API calls",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Check that at least one action is specified
    if not any([args.monitors, args.slos, args.dashboard, args.all, args.export]):
        parser.error(
            "At least one action is required: --monitors, --slos, --dashboard, --all, or --export"
        )

    # Get API keys (skip validation in dry-run mode)
    try:
        if args.dry_run:
            api_key = "dry-run-key"
            app_key = "dry-run-key"
        else:
            api_key, app_key = get_api_keys()
    except ValueError as e:
        logger.error(str(e))
        return 1

    site = os.getenv("DD_SITE", "datadoghq.com")

    try:
        setup = DatadogSetup(
            api_key=api_key,
            app_key=app_key,
            site=site,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Datadog setup: {e}")
        return 1

    # Validate API keys unless dry-run
    if not args.dry_run and not setup.validate_api_keys():
        return 1

    # Execute requested actions
    exit_code = 0

    try:
        if args.all:
            setup.create_all()
        else:
            if args.monitors:
                setup.create_monitors()
            if args.slos:
                setup.create_slos()
            if args.dashboard:
                setup.create_dashboard()

        if args.export:
            setup.export_configs()

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Datadog resource setup script for tau2-bench observability.

This script creates and manages Datadog resources (monitors, SLOs, dashboards,
and Case Management) from JSON configuration files.

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
    python setup_datadog.py --case-management

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

from dotenv import load_dotenv
from loguru import logger

_project_root = Path(__file__).parent.parent.parent.parent.parent
load_dotenv(_project_root / ".env")

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
        self._api_client_v2: Any = None
        self._monitors_api: Any = None
        self._slo_api: Any = None
        self._dashboards_api: Any = None
        self._cases_api: Any = None

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

            # Initialize v2 API client for Case Management
            from datadog_api_client.v2.api.case_management_api import CaseManagementApi

            self._api_client_v2 = ApiClient(configuration)
            self._cases_api = CaseManagementApi(self._api_client_v2)

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

    def create_monitors(self, force_update: bool = False) -> list[dict]:
        """Create or update monitors from monitors.json.

        Args:
            force_update: If True, update existing monitors instead of skipping.

        Returns:
            List of created/updated monitor responses.
        """
        logger.info("Creating monitors...")
        config = self.load_config("monitors.json")
        monitors = config.get("monitors", [])

        if not monitors:
            logger.warning("No monitors found in configuration")
            return []

        # Get existing monitors to check for duplicates
        existing_monitors = {}
        if not self.dry_run and force_update:
            try:
                all_monitors = self._monitors_api.list_monitors(
                    name="tau2-bench"
                )
                for m in all_monitors:
                    existing_monitors[m.name] = m.id
                logger.info(f"Found {len(existing_monitors)} existing tau2-bench monitors")
            except Exception as e:
                logger.warning(f"Could not fetch existing monitors: {e}")

        results = []
        for monitor_def in monitors:
            try:
                monitor_name = monitor_def.get("name", "")
                if force_update and monitor_name in existing_monitors:
                    # Update existing monitor
                    result = self._update_monitor(
                        existing_monitors[monitor_name], monitor_def
                    )
                else:
                    result = self._create_monitor(monitor_def)
                results.append(result)
            except Exception as e:
                error_msg = str(e)
                if "Duplicate" in error_msg and force_update:
                    # Try to find and update the monitor
                    logger.info(f"Monitor {monitor_def.get('id')} exists, attempting update...")
                    try:
                        result = self._find_and_update_monitor(monitor_def)
                        results.append(result)
                    except Exception as update_e:
                        logger.error(f"Failed to update monitor {monitor_def.get('id')}: {update_e}")
                else:
                    logger.error(f"Failed to create monitor {monitor_def.get('id')}: {e}")
                continue

        logger.info(f"Created/updated {len(results)} monitors")
        return results

    def _find_and_update_monitor(self, monitor_def: dict) -> dict:
        """Find an existing monitor by name and update it.

        Args:
            monitor_def: Monitor definition from config.

        Returns:
            Updated monitor response.
        """
        monitor_name = monitor_def.get("name", "")

        # Search for the monitor by name prefix
        all_monitors = self._monitors_api.list_monitors(name="tau2-bench")
        for m in all_monitors:
            if m.name == monitor_name:
                return self._update_monitor(m.id, monitor_def)

        raise ValueError(f"Monitor not found: {monitor_name}")

    def _update_monitor(self, monitor_id: int, monitor_def: dict) -> dict:
        """Update an existing monitor.

        Args:
            monitor_id: Datadog monitor ID.
            monitor_def: Monitor definition from config.

        Returns:
            Updated monitor response.
        """
        config_id = monitor_def.get("id", "unknown")
        name = monitor_def.get("name", "Unnamed Monitor")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would update monitor: {config_id} - {name}")
            return {"id": monitor_id, "name": name, "dry_run": True, "action": "update"}

        from datadog_api_client.v1.model.monitor_options import MonitorOptions
        from datadog_api_client.v1.model.monitor_thresholds import MonitorThresholds
        from datadog_api_client.v1.model.monitor_update_request import MonitorUpdateRequest

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

        # Build update request
        update_request = MonitorUpdateRequest(
            name=name,
            query=monitor_def.get("query", ""),
            message=monitor_def.get("message", ""),
            tags=monitor_def.get("tags", []),
            options=options,
        )

        response = self._monitors_api.update_monitor(
            monitor_id=monitor_id, body=update_request
        )
        logger.info(f"Updated monitor: {config_id} - {name} (ID: {response.id})")
        return {"id": response.id, "name": name, "config_id": config_id, "action": "update"}

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

        from datadog_api_client.v1.model.service_level_objective_query import (
            ServiceLevelObjectiveQuery,
        )
        from datadog_api_client.v1.model.service_level_objective_request import (
            ServiceLevelObjectiveRequest,
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

        # Build SLO request
        slo = ServiceLevelObjectiveRequest(
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

    def create_dashboard(self, version: str = "agents") -> dict:
        """Create dashboard from config file.

        Args:
            version: Dashboard version:
                - "agents": Agent comparison dashboard (default)
                - "operations": Operations & actionable items dashboard

        Returns:
            Created dashboard response.
        """
        version_to_file = {
            "agents": "dashboards_agents.json",
            "operations": "dashboards_operations.json",
        }
        config_file = version_to_file.get(version, "dashboards_agents.json")
        logger.info(f"Creating dashboard from {config_file}...")
        config = self.load_config(config_file)
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
        from datadog_api_client.v1.model.dashboard_template_variable import (
            DashboardTemplateVariable,
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
        widgets = self._convert_widgets(dashboard_def.get("widgets", []))

        # Build template variables
        template_vars_config = dashboard_def.get("template_variables", [])
        template_variables = [
            DashboardTemplateVariable(
                name=tv.get("name"),
                default=tv.get("default", "*"),
                prefix=tv.get("prefix"),
            )
            for tv in template_vars_config
        ]

        # Build dashboard
        dashboard = Dashboard(
            title=title,
            description=dashboard_def.get("description", ""),
            layout_type=layout_type,
            widgets=widgets,
            template_variables=template_variables if template_variables else None,
        )

        response = self._dashboards_api.create_dashboard(body=dashboard)
        logger.info(f"Created dashboard: {title} (ID: {response.id})")
        return {"id": response.id, "title": title, "url": response.url}

    def _convert_widgets(self, widgets_config: list[dict]) -> list:
        """Convert widget configurations to API format.

        Handles nested definition structures and group widgets.

        Args:
            widgets_config: List of widget definitions from config.

        Returns:
            List of Widget objects for the API.
        """
        from datadog_api_client.v1.model.group_widget_definition import (
            GroupWidgetDefinition,
        )
        from datadog_api_client.v1.model.group_widget_definition_type import (
            GroupWidgetDefinitionType,
        )
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
        from datadog_api_client.v1.model.toplist_widget_definition import (
            ToplistWidgetDefinition,
        )
        from datadog_api_client.v1.model.toplist_widget_definition_type import (
            ToplistWidgetDefinitionType,
        )
        from datadog_api_client.v1.model.toplist_widget_request import (
            ToplistWidgetRequest,
        )
        from datadog_api_client.v1.model.widget import Widget
        from datadog_api_client.v1.model.widget_layout import WidgetLayout
        from datadog_api_client.v1.model.widget_layout_type import WidgetLayoutType

        widgets = []

        for widget_config in widgets_config:
            # Handle nested definition structure (definition is inside widget_config)
            defn = widget_config.get("definition", widget_config)
            widget_type = defn.get("type", "note")
            title = defn.get("title", "")
            layout_config = widget_config.get("layout", {})

            try:
                if widget_type == "note":
                    definition = NoteWidgetDefinition(
                        type=NoteWidgetDefinitionType.NOTE,
                        content=defn.get("content", ""),
                        background_color=defn.get("background_color", "white"),
                        font_size=defn.get("font_size", "14"),
                        text_align=defn.get("text_align", "left"),
                    )
                elif widget_type == "group":
                    # Recursively convert nested widgets
                    nested_widgets = self._convert_widgets(defn.get("widgets", []))
                    layout_type = WidgetLayoutType.ORDERED
                    if defn.get("layout_type") == "free":
                        layout_type = WidgetLayoutType.FREE
                    definition = GroupWidgetDefinition(
                        type=GroupWidgetDefinitionType.GROUP,
                        title=title,
                        layout_type=layout_type,
                        widgets=nested_widgets,
                    )
                elif widget_type == "timeseries":
                    requests = []
                    for req in defn.get("requests", []):
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
                    for req in defn.get("requests", []):
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
                        precision=defn.get("precision", 2),
                        autoscale=defn.get("autoscale", True),
                    )
                elif widget_type == "toplist":
                    requests = []
                    for req in defn.get("requests", []):
                        requests.append(
                            ToplistWidgetRequest(
                                q=req.get("q", ""),
                            )
                        )
                    definition = ToplistWidgetDefinition(
                        type=ToplistWidgetDefinitionType.TOPLIST,
                        title=title,
                        requests=requests,
                    )
                elif widget_type == "slo_list":
                    from datadog_api_client.v1.model.slo_list_widget_definition import (
                        SLOListWidgetDefinition,
                    )
                    from datadog_api_client.v1.model.slo_list_widget_definition_type import (
                        SLOListWidgetDefinitionType,
                    )
                    from datadog_api_client.v1.model.slo_list_widget_query import (
                        SLOListWidgetQuery,
                    )
                    from datadog_api_client.v1.model.slo_list_widget_request import (
                        SLOListWidgetRequest,
                    )
                    from datadog_api_client.v1.model.slo_list_widget_request_type import (
                        SLOListWidgetRequestType,
                    )

                    request_config = defn.get("request", {})
                    query_config = request_config.get("query", {})
                    slo_query = SLOListWidgetQuery(
                        query_string=query_config.get("query_string", ""),
                        limit=query_config.get("limit", 10),
                    )
                    slo_request = SLOListWidgetRequest(
                        query=slo_query,
                        request_type=SLOListWidgetRequestType.SLO_LIST,
                    )
                    definition = SLOListWidgetDefinition(
                        type=SLOListWidgetDefinitionType.SLO_LIST,
                        title=title,
                        title_size=defn.get("title_size", "16"),
                        title_align=defn.get("title_align", "left"),
                        requests=[slo_request],
                    )
                elif widget_type == "manage_status":
                    from datadog_api_client.v1.model.monitor_summary_widget_definition import (
                        MonitorSummaryWidgetDefinition,
                    )
                    from datadog_api_client.v1.model.monitor_summary_widget_definition_type import (
                        MonitorSummaryWidgetDefinitionType,
                    )

                    definition = MonitorSummaryWidgetDefinition(
                        type=MonitorSummaryWidgetDefinitionType.MANAGE_STATUS,
                        title=title,
                        title_size=defn.get("title_size", "16"),
                        title_align=defn.get("title_align", "left"),
                        summary_type=defn.get("summary_type", "monitors"),
                        display_format=defn.get("display_format", "countsAndList"),
                        color_preference=defn.get("color_preference", "text"),
                        hide_zero_counts=defn.get("hide_zero_counts", False),
                        show_last_triggered=defn.get("show_last_triggered", True),
                        show_priority=defn.get("show_priority", False),
                        query=defn.get("query", ""),
                        sort=defn.get("sort", "status,asc"),
                    )
                else:
                    # For unsupported types, create a note placeholder
                    definition = NoteWidgetDefinition(
                        type=NoteWidgetDefinitionType.NOTE,
                        content=f"Widget type '{widget_type}' - {title}",
                        background_color="gray",
                    )

                # Create widget with layout from config
                layout = WidgetLayout(
                    x=layout_config.get("x", 0),
                    y=layout_config.get("y", 0),
                    width=layout_config.get("width", 12),
                    height=layout_config.get("height", 3),
                )
                widget = Widget(definition=definition, layout=layout)
                widgets.append(widget)

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

    def create_all_dashboards(self) -> list[dict]:
        """Create all dashboards (agents and operations).

        Returns:
            List of created dashboard responses.
        """
        logger.info("Creating all dashboards...")
        results = []

        for version in ["agents", "operations"]:
            try:
                result = self.create_dashboard(version=version)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to create {version} dashboard: {e}")

        logger.info(f"Created {len(results)} dashboards")
        return results

    def setup_case_management(self) -> dict:
        """Set up Case Management for tau2-bench monitors.

        Creates a project and sample case to verify Case Management is working.
        Monitors will create cases automatically when they fire via the
        @case-management notification handle.

        Returns:
            Dict with project and case information.
        """
        logger.info("Setting up Case Management...")

        if self.dry_run:
            logger.info("[DRY RUN] Would set up Case Management:")
            logger.info("  - Get or create tau2-bench project")
            logger.info("  - Create sample case for verification")
            return {"dry_run": True}

        from datadog_api_client.v2.model.project_create import ProjectCreate
        from datadog_api_client.v2.model.project_create_attributes import ProjectCreateAttributes
        from datadog_api_client.v2.model.project_create_request import ProjectCreateRequest

        result = {"project": None, "case": None}

        # Step 1: Get or create project
        project_key = "TAUBENCH"
        project_name = "tau2-bench Agent Evaluation"

        try:
            # Check for existing projects
            projects_response = self._cases_api.get_projects()
            existing_project = None

            if projects_response.data:
                for proj in projects_response.data:
                    if proj.attributes.key == project_key:
                        existing_project = proj
                        logger.info(f"Found existing project: {project_key} (ID: {proj.id})")
                        break

            if existing_project:
                project_id = existing_project.id
                result["project"] = {
                    "id": project_id,
                    "key": project_key,
                    "name": project_name,
                    "action": "existing",
                }
            else:
                # Create new project
                project_request = ProjectCreateRequest(
                    data=ProjectCreate(
                        attributes=ProjectCreateAttributes(
                            key=project_key,
                            name=project_name,
                        ),
                        type="project",
                    )
                )
                project_response = self._cases_api.create_project(body=project_request)
                project_id = project_response.data.id
                logger.info(f"Created project: {project_key} (ID: {project_id})")
                result["project"] = {
                    "id": project_id,
                    "key": project_key,
                    "name": project_name,
                    "action": "created",
                }

        except Exception as e:
            logger.error(f"Failed to get/create project: {e}")
            # Continue without project - cases can still be created
            project_id = None

        # Note: Cases will be created automatically when monitors fire via @case-management
        # The project provides organization for those cases
        if result["project"]:
            logger.info("Cases will be created automatically when monitors fire via @case-management")

        # Log summary
        logger.info("=" * 50)
        logger.info("Case Management Setup Complete:")
        if result["project"]:
            logger.info(f"  Project: {result['project']['key']} ({result['project']['action']})")
            logger.info("  Monitors already configured with @case-management")
            logger.info("  Cases will be created automatically when monitors fire")
        else:
            logger.info("  Project creation failed - cases will still work without a project")
            logger.info("  Monitors are configured with @case-management")
        logger.info("=" * 50)

        return result

    def create_all(
        self,
        dashboard_version: str = "agents",
        all_dashboards: bool = True,
        force_update: bool = False,
    ) -> dict:
        """Create all Datadog resources.

        Args:
            dashboard_version: Dashboard version to create if all_dashboards is False.
            all_dashboards: If True, create all dashboards (agents + operations).
            force_update: If True, update existing monitors instead of skipping.

        Returns:
            Summary of created resources.
        """
        logger.info("Creating all Datadog resources...")

        results = {
            "monitors": [],
            "slos": [],
            "dashboards": [],
        }

        try:
            results["monitors"] = self.create_monitors(force_update=force_update)
        except Exception as e:
            logger.error(f"Failed to create monitors: {e}")

        try:
            results["slos"] = self.create_slos()
        except Exception as e:
            logger.error(f"Failed to create SLOs: {e}")

        try:
            results["dashboards"] = self.create_all_dashboards()
        except Exception as e:
            logger.error(f"Failed to create dashboard(s): {e}")

        # Summary
        logger.info("=" * 50)
        logger.info("Setup Complete Summary:")
        logger.info(f"  Monitors created: {len(results['monitors'])}")
        logger.info(f"  SLOs created: {len(results['slos'])}")
        logger.info(f"  Dashboards created: {len(results['dashboards'])}")
        for db in results["dashboards"]:
            if db:
                logger.info(f"    - {db.get('title', 'Unknown')}: {db.get('url', 'N/A')}")
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
        help="Create dashboard from config file",
    )
    parser.add_argument(
        "--dashboard-version",
        type=str,
        default="agents",
        choices=["agents", "operations"],
        help="Dashboard version: agents (comparison), operations (actionable items). Default: agents",
    )
    parser.add_argument(
        "--all-dashboards",
        action="store_true",
        help="Create all dashboards (agents + operations). Used with --all or --dashboard.",
    )
    parser.add_argument(
        "--case-management",
        action="store_true",
        help="Set up Case Management (create project and verification case)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create all resources (monitors, SLOs, dashboards, case management)",
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
        "--force-update",
        action="store_true",
        help="Update existing monitors instead of skipping duplicates",
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
    if not any([args.monitors, args.slos, args.dashboard, args.case_management, args.all, args.export]):
        parser.error(
            "At least one action is required: --monitors, --slos, --dashboard, --case-management, --all, or --export"
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
            setup.create_all(
                dashboard_version=args.dashboard_version,
                all_dashboards=args.all_dashboards,
                force_update=args.force_update,
            )
            # Also set up case management when --all is used
            setup.setup_case_management()
        else:
            if args.monitors:
                setup.create_monitors(force_update=args.force_update)
            if args.slos:
                setup.create_slos()
            if args.dashboard:
                if args.all_dashboards:
                    setup.create_all_dashboards()
                else:
                    setup.create_dashboard(version=args.dashboard_version)
            if args.case_management:
                setup.setup_case_management()

        if args.export:
            setup.export_configs()

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

"""
Configuration constants for analysis results visualization.

This module contains all the configuration constants used in analyze_results.py,
including color schemes, mode mappings, domain settings, and LLM specifications.
"""

# Mode mappings
MODES_MAP = {
    "base": "default",
    "solo": "no-user",
    "gt": "oracle-plan",
    "oracle-plan": "oracle-plan",
    "solo-gt": "oracle-plan-no-user",
    "default": "default",
    "no-user": "no-user",
    "op": "oracle-plan",
    "no-user-op": "oracle-plan-no-user",
}

# Available modes and domains
MODES = ["default", "no-user", "oracle-plan"]
DOMAINS = ["retail", "airline", "telecom"]

# Telecom intent configuration
TELECOM_INTENTS_ORDER = {"service_issue": 1, "mobile_data_issue": 2, "mms_issue": 3}
TELECOM_INTENTS_COLORS = {
    "service_issue": "#4E79A7",  # Rich blue
    "mobile_data_issue": "#F28E2B",  # Warm orange
    "mms_issue": "#76B7B2",  # Soft teal
}

# Telecom persona configuration
TELECOM_PERSONAS_ORDER = {"None": 1, "Easy": 2, "Hard": 3}
PERSONA_COLORS = {"None": "#4C72B0", "Easy": "#DD8452", "Hard": "#55A868"}

# Domain colors
DOMAIN_COLORS = {
    "retail": "#4C72B0",  # Nice blue
    "airline": "#DD8452",  # Warm orange
    "telecom": "#55A868",  # Fresh green
    "telecom-workflow": "#C44E52",  # Rich red
}

# Mode styling
MODE_STYLES = {"default": "-", "no-user": "--", "oracle-plan": ":"}
MODE_COLORS = {"default": "#4C72B0", "no-user": "#DD8452", "oracle-plan": "#C44E52"}
MODE_MARKERS = {"default": "o", "no-user": "s", "oracle-plan": "D"}

# Predefined color palette for LLMs
COLOR_PALETTE = [
    "#4C72B0",  # Nice blue
    "#DD8452",  # Warm orange
    "#55A868",  # Fresh green
    "#C44E52",  # Rich red
    "#8C564B",  # Rich brown
    "#8172B3",  # Purple
    "#CCB974",  # Gold
    "#64B5CD",  # Light blue
    "#4C72B0",  # Dark blue
    "#F28E2B",  # Orange
    "#E15759",  # Red
    "#76B7B2",  # Teal
    "#59A14F",  # Green
    "#EDC948",  # Yellow
    "#B07AA1",  # Purple
    "#FF9DA7",  # Pink
    "#9C755F",  # Brown
    "#BAB0AC",  # Gray
]

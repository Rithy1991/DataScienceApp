"""
Cleaning Report State Management
Tracks, persists, and exports data cleaning operations and metrics.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, MutableMapping, Optional

import pandas as pd

CLEANING_REPORT_KEY = "dsai_cleaning_report"


def initialize_cleaning_report(session_state: MutableMapping[Any, Any]) -> None:
    """Initialize an empty cleaning report in session state."""
    if CLEANING_REPORT_KEY not in session_state:
        session_state[CLEANING_REPORT_KEY] = {
            "timestamp_started": datetime.now().isoformat(),
            "timestamp_completed": None,
            "actions": [],
            "before": {"rows": 0, "cols": 0, "missing": 0, "duplicates": 0},
            "after": {"rows": 0, "cols": 0, "missing": 0, "duplicates": 0},
        }


def get_cleaning_report(session_state: MutableMapping[Any, Any]) -> dict:
    """Retrieve the current cleaning report from session state.
    
    Returns:
        dict: Cleaning report with actions and before/after metrics.
    """
    initialize_cleaning_report(session_state)
    return session_state[CLEANING_REPORT_KEY]


def reset_cleaning_report(session_state: MutableMapping[Any, Any]) -> None:
    """Reset the cleaning report (e.g., when loading new data)."""
    session_state[CLEANING_REPORT_KEY] = {
        "timestamp_started": datetime.now().isoformat(),
        "timestamp_completed": None,
        "actions": [],
        "before": {"rows": 0, "cols": 0, "missing": 0, "duplicates": 0},
        "after": {"rows": 0, "cols": 0, "missing": 0, "duplicates": 0},
    }


def add_cleaning_action(
    session_state: MutableMapping[Any, Any],
    action_name: str,
    action_description: str,
    metrics: dict,
) -> None:
    """Log a single cleaning action to the report.
    
    Args:
        session_state: Streamlit session state object.
        action_name: Short name of the action (e.g., "missing_imputation").
        action_description: User-friendly description (e.g., "Filled missing income with median").
        metrics: Dict of metrics about the action (e.g., {"column": "income", "method": "median", "count": 150}).
    """
    initialize_cleaning_report(session_state)
    report = session_state[CLEANING_REPORT_KEY]
    
    report["actions"].append({
        "timestamp": datetime.now().isoformat(),
        "action_name": action_name,
        "action_description": action_description,
        "metrics": metrics,
    })


def set_before_metrics(
    session_state: MutableMapping[Any, Any],
    df: pd.DataFrame,
) -> None:
    """Record the 'before' state (original data metrics).
    
    Args:
        session_state: Streamlit session state object.
        df: The original (raw) DataFrame before cleaning.
    """
    initialize_cleaning_report(session_state)
    report = session_state[CLEANING_REPORT_KEY]
    
    report["before"] = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def set_after_metrics(
    session_state: MutableMapping[Any, Any],
    df: pd.DataFrame,
) -> None:
    """Record the 'after' state (cleaned data metrics).
    
    Args:
        session_state: Streamlit session state object.
        df: The cleaned DataFrame after all cleaning operations.
    """
    initialize_cleaning_report(session_state)
    report = session_state[CLEANING_REPORT_KEY]
    
    report["after"] = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }
    
    # Mark completion time
    report["timestamp_completed"] = datetime.now().isoformat()


def get_report_summary(session_state: MutableMapping[Any, Any]) -> dict:
    """Get a summary of changes made during cleaning.
    
    Returns:
        dict: Summary with rows_removed, missing_fixed, duplicates_removed, etc.
    """
    report = get_cleaning_report(session_state)
    before = report.get("before", {})
    after = report.get("after", {})
    
    return {
        "rows_before": before.get("rows", 0),
        "rows_after": after.get("rows", 0),
        "rows_removed": before.get("rows", 0) - after.get("rows", 0),
        "missing_before": before.get("missing", 0),
        "missing_after": after.get("missing", 0),
        "missing_fixed": before.get("missing", 0) - after.get("missing", 0),
        "duplicates_before": before.get("duplicates", 0),
        "duplicates_after": after.get("duplicates", 0),
        "duplicates_removed": before.get("duplicates", 0) - after.get("duplicates", 0),
        "actions_count": len(report.get("actions", [])),
    }


def export_report_json(session_state: MutableMapping[Any, Any]) -> str:
    """Export the cleaning report as formatted JSON string.
    
    Returns:
        str: JSON string representation of the report.
    """
    report = get_cleaning_report(session_state)
    return json.dumps(report, indent=2, default=str)


def export_report_csv(session_state: MutableMapping[Any, Any]) -> pd.DataFrame:
    """Export the cleaning actions as a CSV-compatible DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns: timestamp, action_name, action_description, metrics_json.
    """
    report = get_cleaning_report(session_state)
    actions = report.get("actions", [])
    
    if not actions:
        return pd.DataFrame(columns=["timestamp", "action_name", "action_description", "metrics"])
    
    data = []
    for action in actions:
        data.append({
            "timestamp": action.get("timestamp", ""),
            "action_name": action.get("action_name", ""),
            "action_description": action.get("action_description", ""),
            "metrics": json.dumps(action.get("metrics", {})),
        })
    
    return pd.DataFrame(data)


def export_report_markdown(session_state: MutableMapping[Any, Any]) -> str:
    """Export the cleaning report as a Markdown summary.
    
    Returns:
        str: Markdown-formatted report.
    """
    report = get_cleaning_report(session_state)
    summary = get_report_summary(session_state)
    
    md = "# Data Cleaning Report\n\n"
    
    # Header info
    md += f"**Started:** {report.get('timestamp_started', 'Unknown')}\n"
    if report.get('timestamp_completed'):
        md += f"**Completed:** {report.get('timestamp_completed', 'Unknown')}\n"
    md += "\n"
    
    # Summary statistics
    md += "## Summary\n\n"
    md += f"- **Rows removed:** {summary['rows_removed']:,}\n"
    md += f"- **Missing values fixed:** {summary['missing_fixed']:,}\n"
    md += f"- **Duplicates removed:** {summary['duplicates_removed']:,}\n"
    md += f"- **Total actions:** {summary['actions_count']}\n\n"
    
    # Before/after
    md += "## Before & After\n\n"
    md += "| Metric | Before | After | Change |\n"
    md += "|--------|--------|-------|--------|\n"
    md += f"| Rows | {summary['rows_before']:,} | {summary['rows_after']:,} | -{summary['rows_removed']:,} |\n"
    md += f"| Missing Values | {summary['missing_before']:,} | {summary['missing_after']:,} | -{summary['missing_fixed']:,} |\n"
    md += f"| Duplicates | {summary['duplicates_before']:,} | {summary['duplicates_after']:,} | -{summary['duplicates_removed']:,} |\n\n"
    
    # Actions list
    md += "## Actions Taken\n\n"
    for i, action in enumerate(report.get("actions", []), 1):
        md += f"{i}. **{action.get('action_description')}**\n"
        md += f"   - Type: `{action.get('action_name')}`\n"
        md += f"   - Time: {action.get('timestamp')}\n"
        metrics = action.get("metrics", {})
        for key, value in metrics.items():
            md += f"   - {key}: {value}\n"
        md += "\n"
    
    return md

from enum import unique
import os
from click import Option
import requests
import datetime
from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool
from google.adk.code_executors import BuiltInCodeExecutor
from google.genai import types


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Centralized model strings — change here, updates everywhere
FAST_MODEL  = "gemini-flash-latest"   # all three agents use this
SMART_MODEL = "gemini-pro-latest"     # swap root_agent here for harder tasks

# Temperature per agent role
SEARCH_TEMP = 0.1   # factual search needs maximum consistency
ROOT_TEMP   = 0.3   # coordinator needs some flexibility to handle varied requests

VALID_STATUSES = {"applied", "phone_screen", "interview", "offer", "rejected", "withdrawn"}

# add_application
# 1. Validate required fields (company, role, status)
# 2. Generate unique ID
# 3. Set applied_date to today if not provided
# 4. Load existing applications from state
# 5. Check for duplicate (same company + role)
# 6. Add to applications dict
# 7. Write back to state
# 8. Return confirmation with application ID
# inputs:  company, role, status, applied_date, salary_min,
#          salary_max, location, notes
# reads:   state["applications"]
# writes:  state["applications"]
#          state["total_count"]  ← increment counter
# returns: new application ID, confirmation
# ID Generation strategy 
# Company initials + timestamp
#   STR-20250510, GOO-20250512...
#   Human readable, naturally unique

def update_status_counts(state: dict, applications: dict) -> None:
    # 1. Start with all status counts at zero
    counts = {status: 0 for status in VALID_STATUSES}
    # 2. Loop through every application and increment its status count
    for app in applications.values():
        status = app.get("status")
        if status in counts:
            counts[status] += 1
    # 3. Write the fresh counts back to state
    state["status_counts"] = counts


def _ok(message: str, **kwargs) -> dict:
    return {"result": "success", "message": message, **kwargs}


def _err(message: str) -> dict:
    return {"result": "error", "message": message}


def add_application(
    tool_context: ToolContext,
    company: str,
    role: str,
    status: str,
    applied_date: str = None,
    salary_min: int = None,
    salary_max: int = None,
    location: str = None,
    notes: str = None,
) -> dict:
    """Add a new job application to the tracker.

    Args:
        company: Name of the company you applied to.
        role: Job title or role you applied for.
        status: Current application status. Must be one of: applied, phone_screen,
            interview, offer, rejected, withdrawn.
        applied_date: Date the application was submitted (YYYY-MM-DD). Defaults to today.
        salary_min: Minimum salary expectation (optional).
        salary_max: Maximum salary expectation (optional).
        location: Job location or 'Remote' (optional).
        notes: Any additional notes about the application (optional).

    Returns:
        A dict with the new application ID and a confirmation message,
        or an error message if validation fails.
    """
    # 1. Validate required fields
    if not company or not role:
        return _err("Company and role are required fields.")

    if status not in VALID_STATUSES:
        return _err(f"Invalid status '{status}', must be one of: {', '.join(sorted(VALID_STATUSES))}")

    # 2. Generate unique ID
    prefix = company[:3].upper()
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    base_id = f"{prefix}-{date_str}"

    # Handle ID collision (same company, same day, different role)
    applications = tool_context.state.get("applications") or {}
    application_id = base_id
    if application_id in applications:
        suffix = 2
        while f"{base_id}-{suffix}" in applications:
            suffix += 1
        application_id = f"{base_id}-{suffix}"

    # 3. Set applied_date to today if not provided
    if not applied_date:
        applied_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # 5. Check for duplicate (same company + role)
    for app_id, app in applications.items():
        if app["company"].lower() == company.lower() and app["role"].lower() == role.lower():
            return _err(f"Application for {role} at {company} already exists with ID {app_id}")

    # 6. Add to applications dict
    applications[application_id] = {
        "id": application_id,
        "company": company,
        "role": role,
        "status": status,
        "applied_date": applied_date,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "location": location,
        "notes": notes,
        "last_updated": datetime.datetime.now().isoformat(),
        "company_research": None,
    }

    # 7. Write back to state
    tool_context.state["applications"] = applications
    tool_context.state["total_count"] = len(applications)
    update_status_counts(tool_context.state, applications)
    # 8. Return confirmation with application ID
    return _ok(f"Application {application_id} added successfully", id=application_id, company=company, role=role)

def _find_application(query: str, applications: dict) -> tuple[str, dict] | tuple[None, None]:
    """Try to find an application by ID first, then by company name (case-insensitive)."""
    # Try exact ID match first
    if query in applications:
        return query, applications[query]
    # Fall back to company name (case-insensitive, return first match)
    query_lower = query.lower()
    for app_id, app in applications.items():
        if app["company"].lower() == query_lower:
            return app_id, app
    return None, None


_TRANSITIONS: dict[str, set[str]] = {
    "applied":      {"phone_screen", "rejected", "withdrawn"},
    "phone_screen": {"interview",    "rejected", "withdrawn"},
    "interview":    {"offer",        "rejected", "withdrawn"},
    "offer":        {"accepted",     "rejected", "withdrawn"},
    "accepted":     set(),   # terminal
    "rejected":     set(),   # terminal
    "withdrawn":    set(),   # terminal
}


def is_valid_transition(current: str, new: str) -> bool:
    return new in _TRANSITIONS.get(current, set())


def update_status(application_id: str, notes: str, new_status: str, tool_context: ToolContext) -> dict:
    """Update the status of an existing job application.

    Args:
        application_id: The unique ID of the application to update.
        notes: Additional notes about the status update (optional).
        new_status: The new status to set for the application. Must be one of:
            applied, phone_screen, interview, offer, rejected, withdrawn.
        tool_context: The context object containing state and other info.

    Returns:
        A dict with a confirmation message or an error message if validation fails.
    """
    # 1. Validate new_status
    if new_status not in VALID_STATUSES:
        return _err(f"Invalid status '{new_status}', must be one of: {', '.join(sorted(VALID_STATUSES))}")

    # 2. Load existing applications from state
    applications = tool_context.state.get("applications") or {}

    # 3. Find application by ID first, then by company name (case-insensitive)
    application_id, app = _find_application(application_id, applications)
    if app is None:
        return _err(f"No application found matching '{application_id}'")

    # 4. Update the application's status and notes
    old_status = app["status"]
    if not is_valid_transition(old_status, new_status):
        return _err(f"Cannot move from '{old_status}' to '{new_status}'. Check pipeline order.")
    app["status"] = new_status
    if notes:
        app["notes"] = notes
    app["last_updated"] = datetime.datetime.now().isoformat()

    # 5. Write back to state
    tool_context.state["applications"] = applications
    update_status_counts(tool_context.state, applications)

    # 6. Return confirmation
    return _ok(f"Updated {app['company']} from '{old_status}' to '{new_status}'", id=application_id, previous_status=old_status, new_status=new_status)

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
        return {"error": "Company and role are required fields."}

    if status not in VALID_STATUSES:
        return {"error": f"Invalid status '{status}', must be one of: {', '.join(sorted(VALID_STATUSES))}"}

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
            return {"error": f"Application for this role at this company already exists with ID: {app_id}"}

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
    return {"message": f"Application added successfully with ID {application_id}.", "application_id": application_id}

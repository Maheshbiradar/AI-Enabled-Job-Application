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

def calculate_stats(tool_context: ToolContext) -> dict:
    """Calculate and return statistics about the job applications.

    Args:
        tool_context: The context object containing state and other info.

    Returns:
        A dict containing various statistics about the applications, such as:
        - total_count: Total number of applications
        - status_counts: A breakdown of how many applications are in each status
        - average_salary: Average salary range (if salary info is available)
        - recent_activity: List of recent updates (company, role, status, date)
    """ 
    applications = tool_context.state.get("applications") or {}

    if not applications:
        return _ok("No applications tracked yet")

    status_counts = {status: 0 for status in VALID_STATUSES}
    for app in applications.values():
        s = app.get("status")
        if s in status_counts:
            status_counts[s] += 1

    # Response rate — "responded" = anything beyond "applied"
    total = len(applications)
    responded = total - status_counts.get("applied", 0)
    response_rate = round(responded / total * 100, 1) if total else 0.0

    # Average days since applied
    today = datetime.date.today()
    days_list = []
    for app in applications.values():
        try:
            applied = datetime.date.fromisoformat(app["applied_date"])
            days_list.append((today - applied).days)
        except (KeyError, ValueError):
            pass
    avg_days_since_applied = round(sum(days_list) / len(days_list), 1) if days_list else 0.0

    # Oldest pending (status = applied or phone_screen)
    pending = [
        app for app in applications.values()
        if app.get("status") in {"applied", "phone_screen"}
    ]
    oldest_pending = min(pending, key=lambda a: a.get("applied_date", ""), default=None)

    # Salary analytics (only where provided)
    salaries = [app["salary_max"] for app in applications.values() if app.get("salary_max") is not None]
    avg_salary = round(sum(salaries) / len(salaries), 2) if salaries else None

    return _ok(
        f"Stats for {total} application(s)",
        total=total,
        by_status=status_counts,
        response_rate=response_rate,
        avg_days_in_pipeline=avg_days_since_applied,
        oldest_pending=oldest_pending,
        avg_target_salary=avg_salary,
    )

def view_applications(filter_status: str, sort_by: str, tool_context: ToolContext) -> dict:
    """View job applications with optional filtering.

    Args:
        filter_status: A keyword to search for in company names (case-insensitive).
        status: Filter applications by this status (optional).
        tool_context: The context object containing state and other info.

    Returns:
        A dict containing a list of applications matching the filters, or an error message if validation fails.
    """
    applications = tool_context.state.get("applications") or {}

    # Validate status filter if provided
    if filter_status and filter_status not in VALID_STATUSES:
        return _err(f"Invalid filter '{filter_status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}")

    if not applications:
        return _ok("No applications tracked yet", count=0, applications=[])

    if filter_status:
        results = [app for app in applications.values() if app.get("status") == filter_status]
    else:
        results = list(applications.values())

    # Sort results
    STATUS_ORDER = list(_TRANSITIONS.keys())
    if sort_by == "company":
        results.sort(key=lambda a: a.get("company", "").lower())
    elif sort_by == "status":
        results.sort(key=lambda a: STATUS_ORDER.index(a["status"]) if a.get("status") in STATUS_ORDER else len(STATUS_ORDER))
    else:  # default: sort by applied_date descending
        results.sort(key=lambda a: a.get("applied_date", ""), reverse=True)

    # Summary counts — always across ALL applications, regardless of filter
    total = len(applications)
    status_counts = tool_context.state.get("status_counts") or {}
    active_counts = {s: status_counts.get(s, 0) for s in ("applied", "phone_screen", "interview", "offer")}
    active_count = sum(active_counts.values())
    formatted = results

    return _ok(
        f"{len(formatted)} application(s) found",
        filter_applied=filter_status if filter_status else "none",
        count=len(formatted),
        applications=formatted,
        summary={
            "total":     total,
            "active":    active_count,
            "by_status": status_counts,
        },
    )


def get_application_detail(identifier: str, tool_context: ToolContext) -> dict:
    """Retrieve full details for a single job application.

    Args:
        identifier: The application ID or company name to look up.
        tool_context: The context object containing state and other info.

    Returns:
        A dict with the full application record, or an error if not found.
    """
    applications = tool_context.state.get("applications") or {}

    # 1. Try exact ID match
    app_id, app = identifier, applications.get(identifier)

    # 2. If not found, fall back to exact company name (case-insensitive)
    if app is None:
        identifier_lower = identifier.lower()
        for aid, a in applications.items():
            if a.get("company", "").lower() == identifier_lower:
                app_id, app = aid, a
                break

    # 3. If still not found, try partial company name match
    if app is None:
        for aid, a in applications.items():
            if identifier.lower() in a.get("company", "").lower():
                app_id, app = aid, a
                break

    if app is None:
        return _err(f"No application found matching '{identifier}'")

    # ── SECTION 3: Calculate days since applied ─────────────
    try:
        applied = datetime.date.fromisoformat(app["applied_date"])
        days_tracked = (datetime.date.today() - applied).days
    except (KeyError, ValueError):
        days_tracked = None

    # ── SECTION 4: Return full detail ───────────────────────
    return _ok(
        f"Found application {app_id}",
        id=app["id"],
        company=app["company"],
        role=app["role"],
        app_status=app["status"],
        applied_date=app["applied_date"],
        days_tracked=days_tracked,
        salary_min=app.get("salary_min"),
        salary_max=app.get("salary_max"),
        location=app.get("location"),
        notes=app.get("notes"),
        company_research=app.get("company_research"),
        has_research=app.get("company_research") is not None,
        last_updated=app.get("last_updated"),
    )


def delete_application(identifier: str, tool_context: ToolContext) -> dict:
    """Delete a job application from the tracker.

    Args:
        identifier: The application ID or company name to delete.
        tool_context: The context object containing state and other info.

    Returns:
        A dict confirming deletion, or an error if not found.
    """
    applications = tool_context.state.get("applications") or {}

    if not applications:
        return _err("No applications to delete")

    # 1. Try exact ID match
    app_id, app = identifier, applications.get(identifier)

    # 2. If not found, try company name (case-insensitive)
    if app is None:
        for aid, a in applications.items():
            if a.get("company", "").lower() == identifier.lower():
                app_id, app = aid, a
                break

    if app is None:
        return _err(f"No application found for '{identifier}'")

    # Capture what we are about to delete
    # so we can confirm it in the return value
    deleted_company = app["company"]
    deleted_role    = app["role"]
    deleted_id      = app["id"]
    deleted_status  = app["status"]

    # Delete from state
    del applications[app_id]
    tool_context.state["applications"] = applications
    tool_context.state["total_count"] = len(applications)
    update_status_counts(tool_context.state, applications)

    return _ok(
        f"Deleted {deleted_role} at {deleted_company} ({deleted_id})",
        deleted_id=deleted_id,
        deleted_company=deleted_company,
        deleted_role=deleted_role,
        deleted_status=deleted_status,
        remaining_count=len(applications),
    )


def attach_research(
    company: str,
    research_content: str,
    tool_context: ToolContext,
) -> dict:
    """Attach or update company research notes for a job application.

    Args:
        company: The company name (or application ID) to attach research to.
        research_content: The research text to attach.
        tool_context: The context object containing state and other info.

    Returns:
        A dict confirming the research was attached, or an error if not found.
    """
    # Validate inputs
    if not company:
        return _err("Company name is required")
    if not research_content:
        return _err("Research content cannot be empty")

    applications = tool_context.state.get("applications") or {}

    # Find all matching applications and append to matches
    matches = []
    for aid, app in applications.items():
        if company.lower() in app.get("company", "").lower():
            matches.append((aid, app))

    if not matches:
        return _err(f"No application found matching '{company}'")

    # If multiple matches — use most recently updated
    app_id, target_app = max(matches, key=lambda x: x[1].get("last_updated", ""))
    note = (
        f"Multiple applications matched '{company}'; selected most recently updated: "
        f"{target_app['company']} — {target_app['role']} ({app_id})"
        if len(matches) > 1 else None
    )

    # Check if research already exists
    already_had_research = target_app.get("company_research") is not None

    # Attach research
    target_app["company_research"] = research_content
    target_app["last_updated"] = datetime.datetime.now().isoformat()
    applications[app_id] = target_app
    tool_context.state["applications"] = applications

    return _ok(
        f"Research attached to {target_app['company']} ({target_app['role']})",
        id=target_app["id"],
        company=target_app["company"],
        role=target_app["role"],
        action="updated" if already_had_research else "added",
        word_count=len(research_content.split()),
        note=note if note else None,
    )

search_agent = Agent(
    model="gemini-flash-latest",
    name="search_agent",
    description=(
        "Specialist web search agent for researching companies. "
        "Searches for engineering culture, interview processes, "
        "recent news, and reputation. Use this whenever the user "
        "wants to research a company for a job application."
    ),
    instruction="""
    You are a specialist research agent for job seekers.
    Your only job is to research companies and return
    structured findings.

    [SEARCH STRATEGY]
    For every company research request, run 3 searches.
    Replace COMPANY_NAME with the actual company name given to you:

    Search 1: "COMPANY_NAME engineering culture glassdoor 2025"
    Search 2: "COMPANY_NAME software engineer interview process"
    Search 3: "COMPANY_NAME news layoffs funding growth 2025"

    [OUTPUT FORMAT]
    Return your findings in this exact structure:

    CULTURE:
    What employees say about working there, work-life balance,
    management style, team dynamics.

    INTERVIEW PROCESS:
    What the interview looks like — rounds, types of questions,
    difficulty, timeline from application to offer.

    RECENT NEWS:
    Anything significant — layoffs, hiring freezes, funding,
    acquisitions, leadership changes. Flag anything that should
    affect the user's decision to pursue this role.

    VERDICT:
    One sentence recommendation — is this company worth pursuing
    based on what you found?

    [RULES]
    - Always run all 3 searches before responding
    - If search results are thin, say so — do not make things up
    - Be honest about red flags — this person is making a
    career decision
    - Cite specific things from search results, not generalities
    """,
    tools=[google_search],
)

# ─────────────────────────────────────────────
# ROOT AGENT
# Coordinator — holds custom tools directly,
# delegates research to search_agent via AgentTool
# ─────────────────────────────────────────────

root_agent = Agent(
    model="gemini-flash-latest",

    name="job_tracker",

    description=(
        "A job application tracking assistant that manages the "
        "full application pipeline — adding applications, updating "
        "statuses, researching companies, and providing analytics "
        "on the job search progress."
    ),
    instruction="""
`You are JobTracker, a focused assistant that helps
manage job applications from first contact to offer.
You track applications, update their progress through
the pipeline, research companies, and give analytics
on the job search.

[YOUR TOOLS AND WHEN TO USE THEM]

Use these tools DIRECTLY — do not delegate:

  add_application
    → when user adds a new job application
    → trigger words: "add", "track", "applied to",
      "I applied", "new application"

  update_status
    → when user reports progress on an application
    → trigger words: "got a call", "interview scheduled",
      "rejected", "offer received", "withdrew",
      "update status", "move to"
    → call ONCE with EXACTLY the status the user stated
    → NEVER chain multiple update_status calls to bridge
      gaps — if the transition is invalid, report the
      error and tell the user which step to take next

  view_applications
    → when user wants to see their applications
    → trigger words: "show me", "list", "what applications",
      "which companies", "all my applications"
    → also call this BEFORE calculate_stats to give
      context

  get_application_detail
    → when user asks about ONE specific application
    → trigger words: "tell me about", "details for",
      "what is the status of", "how long since"

  calculate_stats
    → when user asks for analytics or summary
    → trigger words: "stats", "how many", "success rate",
      "response rate", "summary", "overview"

  attach_research
    → ONLY call this AFTER search_agent returns findings
    → never call this directly from user message alone
    → always call it with the research content from
      search_agent, not with empty content

  delete_application
    → when user explicitly says to delete or remove
    → ALWAYS confirm what you are deleting in your
      response — this is irreversible

DELEGATE to search_agent:

  search_agent
    → when user wants to research a company
    → trigger words: "research", "look up", "find out
      about", "what is X company like", "check"
    → give it a DETAILED request including the company
      name and what aspects to cover
    → after it returns, call attach_research if the
      user wants findings saved

[ROUTING RULES]

1. NEVER use google_search yourself — you do not have
   it. Always delegate company research to search_agent.

2. When delegating to search_agent be SPECIFIC:
   POOR:  "research this company"
   GOOD:  "Research Stripe — cover engineering culture,
           interview process difficulty and format, and
           any recent news about layoffs, hiring freezes,
           or funding rounds. Return findings in the
           standard CULTURE / INTERVIEW / NEWS / VERDICT
           format."

3. After search_agent returns findings AND user wants
   them saved:
   Step 1 → search_agent returns research text
   Step 2 → call attach_research(
              company=company_name,
              research_content=findings_from_search
            )
   Step 3 → confirm both actions to user

4. For update_status always show:
   previous status → new status
   Example: "Updated Stripe: applied → interview ✓"

5. For delete_application always confirm:
   "Deleted [role] at [company] (ID: [id])"

6. NEVER chain update_status calls to bridge stages.
   One user message = ONE update_status call.
   If the tool returns an error, surface that error
   to the user — do NOT retry with an intermediate
   status to work around it.

[PIPELINE REFERENCE]
Remind users of valid next steps when they update status:

  applied      → next: phone_screen, rejected, withdrawn
  phone_screen → next: interview, rejected, withdrawn
  interview    → next: offer, rejected, withdrawn
  offer        → next: accepted, rejected, withdrawn
  accepted     → terminal (congratulations!)
  rejected     → terminal
  withdrawn    → terminal

[DISPLAY FORMAT]
When showing application list, format clearly:

  YOUR APPLICATIONS
  ──────────────────────────────────────────────
  ID              COMPANY    ROLE          STATUS
  STR-20250510    Stripe     Sr Engineer   interview
  GOO-20250512    Google     Staff Eng     applied
  ──────────────────────────────────────────────
  Total: 2  |  Active: 2  |  Rejected: 0

When showing stats:

  APPLICATION STATS
  ──────────────────────────────────────────────
  Total tracked:       5
  Response rate:       60%  (3 of 5 responded)
  In interview:        1
  Offers received:     1
  Avg days tracked:    12.3 days
  Oldest pending:      Netflix (18 days, no response)
  Avg target salary:   $195,000
  ──────────────────────────────────────────────

[TONE AND BEHAVIOUR]
- Be direct and efficient — job searching is stressful,
  do not waste the user's time
- When research reveals red flags (layoffs, bad culture),
  be honest and flag it clearly
- Celebrate wins — offer received, interview scheduled
- For old pending applications (over 14 days with no
  response) proactively mention them
- Never make up application data — always read from state

[SCOPE]
You handle job applications only. For anything outside
this scope say:
"I am focused on job application tracking. I can help
you add applications, track progress, research companies,
and analyse your job search. What would you like to do?"
""",
tools=[
        # ── Specialist agent ──────────────────────────
        # AgentTool reads search_agent.description to
        # build the schema root LLM uses for routing
        AgentTool(agent=search_agent),

        # ── Custom tools ──────────────────────────────
        # All Python functions — safe to coexist here
        # because no built-in tools are in this list
        add_application,
        update_status,
        view_applications,
        get_application_detail,
        calculate_stats,
        attach_research,
        delete_application,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=4096,
    ),
)

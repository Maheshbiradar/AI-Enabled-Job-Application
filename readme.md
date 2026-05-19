# Job Application Tracker Agent

> A conversational AI agent built with **Google ADK** that tracks your entire job application pipeline — add applications, update statuses, research companies, and get analytics — all through natural conversation.

---

## What This App Does

Instead of maintaining a spreadsheet, you talk to this agent naturally:

```
"Add a Senior Engineer role at Stripe, applied today, salary $180k-$220k"
"Research Stripe's engineering culture and attach it to that application"
"Update my Google application to interview stage"
"Show me everything still waiting for a response"
"What is my overall response rate so far?"
"How many days has it been since I applied to Netflix?"
```

All state persists across the entire conversation. The agent remembers every application, every status change, and every note you have added.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ROOT AGENT                           │
│                    job_tracker                          │
│                                                         │
│   Custom Tools (live directly on root agent):           │
│   ┌──────────────────┐  ┌──────────────────┐           │
│   │ add_application  │  │ update_status    │           │
│   └──────────────────┘  └──────────────────┘           │
│   ┌──────────────────┐  ┌──────────────────┐           │
│   │ view_applications│  │ get_app_detail   │           │
│   └──────────────────┘  └──────────────────┘           │
│   ┌──────────────────┐  ┌──────────────────┐           │
│   │ calculate_stats  │  │ attach_research  │           │
│   └──────────────────┘  └──────────────────┘           │
│   ┌──────────────────┐                                  │
│   │ delete_application│                                 │
│   └──────────────────┘                                  │
│                                                         │
│   Specialist Agent (via AgentTool — isolated):          │
│   ┌──────────────────────────────────────────┐         │
│   │  search_agent  →  google_search (only)   │         │
│   └──────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

> **Why two layers?**
> Google's Gemini API does not allow built-in tools (like `google_search`) and custom Python function tools to exist on the same agent. The `search_agent` is isolated as a specialist — the root agent delegates research tasks to it via `AgentTool`.

---

## Application Pipeline

```
applied ──→ phone_screen ──→ interview ──→ offer ──→ accepted
   │              │               │           │
   └──────────────┴───────────────┴─── rejected
                                           or
                                        withdrawn
```

Valid status values: `applied`, `phone_screen`, `interview`, `offer`, `accepted`, `rejected`, `withdrawn`

---

## Session State Schema

All data lives in `session.state` — persists for the entire conversation:

```python
session.state = {

    "applications": {
        "STR-20250510": {
            "id":               "STR-20250510",
            "company":          "Stripe",
            "role":             "Senior Engineer",
            "status":           "interview",
            "applied_date":     "2025-05-10",
            "salary_min":       180000,
            "salary_max":       220000,
            "location":         "Remote",
            "notes":            ["Applied via LinkedIn", "Heard back in 3 days"],
            "company_research": "Stripe is known for...",
            "last_updated":     "2025-05-14 09:32:00"
        },
        "GOO-20250512": { ... }
    },

    "total_count":    5,
    "deleted_count":  1,
    "last_added":     "GOO-20250512",
    "status_counts": {
        "applied":      2,
        "phone_screen": 1,
        "interview":    1,
        "offer":        0,
        "accepted":     0,
        "rejected":     1,
        "withdrawn":    0
    }
}
```

---

## Tools Reference

| Tool | Reads State | Writes State | Purpose |
|---|---|---|---|
| `add_application` | `applications` | `applications`, `total_count`, `status_counts` | Add a new job application |
| `update_status` | `applications` | `applications`, `status_counts` | Move application through pipeline |
| `view_applications` | `applications` | ❌ read-only | List and filter applications |
| `get_application_detail` | `applications` | ❌ read-only | Full detail for one application |
| `calculate_stats` | `applications` | ❌ read-only | Analytics, rates, averages |
| `attach_research` | `applications` | `applications` | Save company research to application |
| `delete_application` | `applications` | `applications`, `deleted_count` | Remove an application |
| `search_agent` | — | — | Research company via google_search |

---

## ID Format

Application IDs are generated as: `{COMPANY_PREFIX}-{YYYYMMDD}`

```
Stripe  →  STR-20250510
Google  →  GOO-20250512
Netflix →  NET-20250514
```

If two applications are added for the same company on the same day: `STR-20250510-2`

---

## Key Design Decisions

### 1. Notes are appended, not replaced
Every call to `update_status` with a note **appends** to the notes list. This preserves the full history of comments on each application — never overwriting previous context.

### 2. `status_counts` is always kept in sync
Any tool that modifies application status must also update `status_counts`. This is a **state invariant** — `status_counts` must always equal the actual counts across all applications. Breaking this causes `calculate_stats` to report wrong numbers.

### 3. Validation before any state write
Every tool validates all inputs **before** touching `session.state`. If validation fails, state is never partially modified. This prevents corrupted state that is difficult to debug.

### 4. Search is isolated
`search_agent` holds only `google_search` — no custom tools. The root agent delegates via `AgentTool` and receives findings as a string result. The root agent then calls `attach_research` to save findings to the application record.

### 5. Read-only tools never write state
`view_applications`, `get_application_detail`, and `calculate_stats` are **strictly read-only**. They never modify state. This makes them safe to call at any point without side effects.

---

## Concepts Demonstrated (Week 1 ADK)

| Concept | Where It Appears |
|---|---|
| `Agent` with `instruction` | Root agent + search specialist |
| `ToolContext` | All 7 custom tools |
| Session state persistence | Entire application database |
| State design (nested dicts) | `applications` + `status_counts` |
| State invariants | `status_counts` sync after every write |
| Read-only tools | `view_applications`, `calculate_stats` |
| Validate before write | Every tool checks inputs first |
| Built-in tool constraint | `search_agent` isolated from custom tools |
| `AgentTool` delegation | Root delegates research to search specialist |
| Tool composition | "Add job and research company" chains tools |
| Instruction routing | Explicit rules for when to use each tool |
| Error handling patterns | Duplicate check, not found, invalid status |
| State caching | Avoid re-researching same company |
| `gemini-flash-latest` | Correct model string |
| `generate_content_config` | Temperature tuned per agent role |

---

## Project Structure

```
job_tracker/
├── agent.py          ← all agent code lives here
├── __init__.py       ← makes it a Python package (leave empty)
└── .env              ← API keys — NEVER commit this
```

---

## Setup

### 1. Prerequisites

```bash
python3 --version   # needs 3.9+
pip --version
```

### 2. Install dependencies

```bash
pip install google-adk requests
```

### 3. Configure environment

Create `job_tracker/.env`:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

Get your Gemini API key at: https://aistudio.google.com/app/apikey

### 4. Run the agent

```bash
# From the parent folder (one level above job_tracker/)
adk web

# Or terminal mode
adk run job_tracker
```

Open `http://localhost:8000` and select `job_tracker`.

---

## Example Conversation

```
You:   Add a Senior Engineer role at Stripe. Applied today.
       Salary range $180k to $220k. Remote position.

Agent: Added! Application STR-20250519 created for Senior Engineer
       at Stripe. Status: applied. Salary: $180k–$220k. Remote.

You:   Research Stripe's engineering culture and attach it.

Agent: [searches web for Stripe culture + interview process]
       Found some good information. Attached to STR-20250519:
       Stripe is known for high engineering standards...

You:   I got a phone screen scheduled! Update Stripe to phone_screen.

Agent: Updated STR-20250519 from applied → phone_screen. Good luck!

You:   Show me all my active applications

Agent: YOUR APPLICATIONS (4 total)
       ─────────────────────────────
       STR-20250519  Stripe          Senior Engineer     phone_screen
       GOO-20250515  Google          Staff Engineer      interview
       NET-20250510  Netflix         Principal Engineer  applied
       AMZ-20250508  Amazon          SDE III             rejected
       ─────────────────────────────
       Active: 3  |  Rejected: 1  |  Response rate: 75%

You:   What are my stats?

Agent: APPLICATION STATS
       Total tracked:    4
       Response rate:    75% (3 of 4 responded)
       In interview:     1
       Avg days tracked: 8 days
       Oldest pending:   NET-20250510 (9 days, no response)
```

---

## Development Notes

### Adding a new status
Add it to `VALID_STATUSES` at the top of `agent.py` and update the transition validation logic in `update_status`.

### Replacing in-memory state with a database
The `applications` dict in `session.state` is the only place data lives. To persist across sessions, switch `InMemorySessionService` to `VertexAiSessionService` in your runner config — no tool code changes needed.

### Testing tools in isolation

```python
from unittest.mock import MagicMock
from job_tracker.agent import add_application

ctx = MagicMock()
ctx.state = {}

result = add_application(
    company="Stripe",
    role="Senior Engineer",
    status="applied",
    applied_date="2025-05-19",
    salary_min=180000,
    salary_max=220000,
    location="Remote",
    notes="Applied via LinkedIn",
    tool_context=ctx
)
print(result)
print(ctx.state)
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `adk: command not found` | Activate venv: `source .venv/bin/activate` |
| Agent answers without calling tools | Strengthen routing rules in instruction |
| `google_search` error on root agent | `google_search` must only be on `search_agent`, not `root_agent` |
| State lost between messages | Check `tool_context.state["applications"] = applications` is called after every write |
| `status_counts` out of sync | Add `update_status_counts()` helper called after every status change |
| Duplicate applications added | Check duplicate detection in `add_application` runs before any state write |

---

## Resources

- [ADK Documentation](https://google.github.io/adk-docs/)
- [ADK Python GitHub](https://github.com/google/adk-python)
- [Gemini API Keys](https://aistudio.google.com/app/apikey)
- [ADK Tool Limitations](https://google.github.io/adk-docs/tools/limitations/)
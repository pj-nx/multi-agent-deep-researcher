"""The API user guide, rendered as Markdown for the hosted page's API tab, the Gradio app's API tab and
``GET /api/guide``. One source so the three never drift; the deployment's own URL, auth requirement
and limits are filled in.
"""
from __future__ import annotations

from typing import Any

EVENTS: tuple[tuple[str, str], ...] = (
    ("started", "first event of every stream: `thread_id`, the echoed inputs and a `trace_url` (null when "
                "tracing is off)"),
    ("settings", "only when per-run `settings` were sent: the resolved models, effort, fallbacks and budget"),
    ("plan", "`status: awaiting_approval` with the proposed `plan`; the plan stream ends here"),
    ("approve", "the resume stream accepted the plan"),
    ("retrieve", "one per sub-question, in parallel: `sub_question_id`, `hop`, `tools`, `sources`, "
                 "`chunks_stored`, `elapsed_s`"),
    ("validate", "`assessed` / `kept` after the Source Validator graded the hop's evidence"),
    ("analyze", "`claims`, `contradictions`, `gaps` across the analysed sub-questions"),
    ("route_gaps", "`next` is `retrieve` (another hop, with `new_sub_questions`), `insight` or `report`"),
    ("insight", "hypotheses with reasoning chains"),
    ("critic", "`verdict` (`accept` / `revise` / `need_more_evidence`), `issues`, `follow_up`, `round`"),
    ("activity", "fine-grained progress: `kind` is `llm_start` / `llm_end` / `llm_error` / `tool_start` / "
                 "`tool_end` / `tool_timeout` / `embed_start` / `embed_end` / `retrieval`"),
    ("report", "the finished report: Markdown in `report`, the structured record in `report_json`, `stats`"),
    ("error", "the run failed; `message` says why. The stream ends"),
    ("done", "the stream finished normally, with `elapsed_s`"),
)


def guide_markdown(base_url: str = "", *, auth_required: bool = False, rate_limit_per_minute: int | None = None,
                   max_concurrent_runs: int | None = None, max_hops: int | None = None,
                   budget_cap_seconds: int | None = None, allowed_models: list[str] | None = None,
                   models: dict[str, str] | None = None) -> str:
    base = (base_url or "").rstrip("/") or "http://localhost:8000"
    token_note = ("This server **requires a token**: send `Authorization: Bearer <token>` or `X-API-Key: <token>` "
                  "on every call except `/api/health` and `/`. Without it you get `401`."
                  if auth_required else
                  "This server does **not** require a token. If `API_AUTH_TOKEN` is set on it later, send "
                  "`Authorization: Bearer <token>` or `X-API-Key: <token>` on every call except `/api/health` and `/`.")
    auth_hdr = ' -H "Authorization: Bearer $TOKEN"' if auth_required else ""
    py_hdr = '"authorization": f"Bearer {TOKEN}", ' if auth_required else ""
    limits = []
    if rate_limit_per_minute:
        limits.append(f"**{rate_limit_per_minute} requests per minute** per client IP on the guarded routes; "
                      "over the limit you get `429` with a `Retry-After` header")
    if max_concurrent_runs:
        limits.append(f"**{max_concurrent_runs} research streams at a time** per instance; extra requests get "
                      "`503` with `Retry-After: 30`")
    if budget_cap_seconds:
        limits.append(f"**run budget cap {budget_cap_seconds} s per leg**: retrieval loops stop after that even if "
                      "you ask for more")
    if max_hops is not None:
        limits.append(f"default `max_hops` is **{max_hops}** (request up to 5)")
    if allowed_models:
        limits.append("per-run model overrides are restricted to: " + ", ".join(f"`{m}`" for m in allowed_models))
    limits_md = "\n".join(f"- {x}" for x in limits) or "- no limits are configured on this server"
    model_rows = "\n".join(f"| `{role}` | `{m}` |" for role, m in (models or {}).items())
    models_md = (f"\n\nModels this server uses by default (override per run with `settings`):\n\n"
                 f"| Role | Model |\n|---|---|\n{model_rows}" if model_rows else "")

    return f"""# API user guide

Base URL: `{base}`. Every research call is a **Server-Sent Events** stream (`text/event-stream`): read
it line by line, one `event:` / `data:` block per event, `data` is JSON. A run is two calls: plan, then
resume with the (optionally edited) plan.

## Authentication

{token_note}

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | liveness plus the effective configuration: models per role, embeddings, vector DB, sources, limits, `auth_required`, `run_settings` |
| GET | `/api/models` | | OpenRouter's live model catalogue (id, context length, prices), cached 10 min; `?prefix=anthropic/` filters |
| GET | `/api/guide` | | this guide as Markdown |
| POST | `/api/research` | `{{"question", "max_hops"?, "settings"?}}` | SSE stream until the plan needs approval |
| POST | `/api/research/{{thread_id}}/resume` | `{{"plan", "settings"?}}` | SSE stream through to the report |
| POST | `/api/report/pdf` | `{{"markdown", "title"?}}` | `application/pdf` (stateless: pass the report Markdown back) |

## Limits on this server

{limits_md}{models_md}

## Step 1: plan

```bash
curl -N -X POST "{base}/api/research" \\
  -H "Content-Type: application/json"{auth_hdr} \\
  -d '{{"question": "Does the four-day work week raise productivity?", "max_hops": 0}}'
```

Read the stream until the `plan` event (`status: awaiting_approval`). Keep its `thread_id` (from the
`started` event) and its `plan` object. Each sub-question has `id`, `question`, `source_types`
(`papers`, `news`, `web`, `reports`, `api`, `wikipedia`), `search_queries` and `depends_on`.

## Step 2: approve and run

Edit the plan if you like (drop a sub-question, reword one, change its sources), then resume the same
thread with it. The plan must still validate: unique ids, 1 to 12 sub-questions, valid source types.

```bash
curl -N -X POST "{base}/api/research/$THREAD_ID/resume" \\
  -H "Content-Type: application/json"{auth_hdr} \\
  -d '{{"plan": <the plan object>}}'
```

The stream ends with `report` then `done`. `report_json` holds the structured record: `title`,
`executive_summary`, `key_findings`, `summaries` (claims with `evidence_ids`), `contradictions`,
`insights`, `critiques`, `gaps`, `sources`, `assessments`, `retrieval_log`, `stats`. Every `[E-…]` id
in the text refers to an entry in `sources`.

> Plan and resume must reach the same server instance: the thread lives in that instance's checkpoint
> store unless the deployment uses Postgres. Resume promptly after planning.

## Per-run settings (optional)

Send the same `settings` object with both calls to override, for that run only:

```json
{{
  "settings": {{
    "models": {{"heavy": "anthropic/claude-sonnet-5", "cheap": "anthropic/claude-haiku-4.5"}},
    "reasoning_effort": "low",
    "fallback_models": ["openai/gpt-5.1"],
    "budget_seconds": 120
  }}
}}
```

Keys of `models` are tiers (`heavy`: planner, analysis, insight, critic; `worker`: retriever, report;
`cheap`: validator, grader) or single roles. Model ids must be `vendor/model`; if the server has an
allowlist, anything else is rejected with `422`. `reasoning_effort` is `none`, `minimal`, `low`,
`medium` or `high`. `budget_seconds` can shorten a run but never exceeds the server's cap; `0` means
the server default. API keys are never part of settings. The stream echoes what was resolved in a
`settings` event and the report's methodology section records the models used.

## Step 3: PDF (optional)

```bash
curl -X POST "{base}/api/report/pdf" -H "Content-Type: application/json"{auth_hdr} \\
  -d '{{"markdown": <report markdown>, "title": "My report"}}' -o report.pdf
```

## Events

| Event | Meaning |
|---|---|
{chr(10).join(f"| `{name}` | {desc} |" for name, desc in EVENTS)}

## Python client

```python
import json, httpx

BASE = "{base}"
TOKEN = "..."  # only when the server requires one
HEADERS = {{{py_hdr}"content-type": "application/json"}}

def events(method, url, body):
    with httpx.stream(method, url, headers=HEADERS, json=body, timeout=600) as r:
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\\n\\n" in buf:
                block, buf = buf.split("\\n\\n", 1)
                ev = next((l[6:].strip() for l in block.split("\\n") if l.startswith("event:")), "message")
                data = "\\n".join(l[5:].lstrip() for l in block.split("\\n") if l.startswith("data:"))
                yield ev, json.loads(data) if data else {{}}

plan = thread = None
for ev, d in events("POST", f"{{BASE}}/api/research", {{"question": "Does the four-day work week raise productivity?"}}):
    if ev == "started":
        thread = d["thread_id"]
    if ev == "plan" and d.get("status") == "awaiting_approval":
        plan = d["plan"]

for ev, d in events("POST", f"{{BASE}}/api/research/{{thread}}/resume", {{"plan": plan}}):
    if ev == "report":
        print(d["report"])          # Markdown
        report_json = d["report_json"]
```

## Errors

| Status | Meaning |
|---|---|
| `401` | missing or wrong token |
| `422` | invalid body: question too short, bad plan, unknown tier/role, disallowed model, bad `thread_id` |
| `429` | rate limit; wait `Retry-After` seconds |
| `503` | too many runs in progress; retry after `Retry-After` |
| `error` event | the run failed after it started (model or tool failure); the message says why |
"""


def guide_from_health(health: dict[str, Any], base_url: str = "") -> str:
    """Build the guide from a ``/api/health`` payload (what the UIs have at hand)."""
    rs = health.get("run_settings") or {}
    return guide_markdown(
        base_url,
        auth_required=bool(health.get("auth_required")),
        rate_limit_per_minute=health.get("rate_limit_per_minute"),
        max_concurrent_runs=health.get("max_concurrent_runs"),
        max_hops=health.get("max_hops"),
        budget_cap_seconds=rs.get("budget_cap_seconds") or health.get("run_budget_seconds"),
        allowed_models=rs.get("allowed_models"),
        models=(health.get("llm") or {}).get("models"),
    )

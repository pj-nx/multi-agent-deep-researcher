# Multi-Agent Deep Researcher

An AI research assistant for multi-hop, multi-source investigations, built on
[LangGraph](https://docs.langchain.com/oss/python/langgraph). A planner decomposes a hard
question, parallel retriever agents pull from papers, news, reports and APIs, a critical-analysis
agent cross-checks the evidence and flags contradictions, an insight agent proposes hypotheses
with reasoning chains, a red-team critic challenges them, and a report builder compiles a fully
cited report.

Showcases **agent collaboration**, **retrieval-augmented reasoning** (every claim cites a stored
evidence ID that is verified before the report ships) and **long-context synthesis**.

**Status:** all seven agents are live (Phases 0-4). Planner (structured output) → human approval →
parallel retrievers over real sources → persistent vector store → **Source Validator** (model-graded
relevance/credibility) → **Critical Analysis** (retrieval-augmented: claims, contradictions, gaps,
every citation checked against the chunks shown) → multi-hop loop on the most valuable gaps →
**Insight Generation** (reasoning chains with citations, confidence, falsifier) → **Red-Team Critic**
(accept / revise / request more evidence, capped) → **Report Builder** (executive summary, findings,
disagreements, insights, limitations, methodology, bibliography; every citation verified). Remaining:
Gradio UI, deployment wiring, evals. See [PLAN.md](PLAN.md).

## Architecture

```mermaid
flowchart LR
    planner[Planner] --> approve{{approve plan}}
    approve -->|fan-out| retriever[Retrievers ×N]
    retriever --> validator[Source Validator]
    validator --> analysis[Critical Analysis]
    analysis -->|gaps| retriever
    analysis --> insight[Insight Generation]
    insight --> critic[Red-Team Critic]
    critic -->|more evidence| planner
    critic --> report[Report Builder]
```

## Quick start (local)

```bash
uv sync --extra dev --extra vectordb --extra embeddings-local
cp .env.example .env            # OPENROUTER_API_KEY is the only required key
uv run python -m deep_researcher "Does the four-day week raise productivity?" --auto-approve
```

The CLI prints each agent event, writes the report to `reports/`, and without `--auto-approve`
shows the planner's sub-questions and asks before retrieving. The same graph serves the API:

```bash
uv run python -m deep_researcher.ui        # Gradio UI at http://127.0.0.1:7860 (uv sync --extra ui)
uv run uvicorn api.index:app --reload      # HTTP API at http://localhost:8000/api/health, UI at public/index.html
```

The Gradio UI (flat, minimalist; Matrix-green accent on white, dark by default, with a toggle that remembers your
choice) streams the agents' activity live through a row of pipeline stage boxes, lets you edit the plan table
before approving, renders the report and offers `.pdf` / `.md` / `.json` downloads (the PDF is rendered in pure Python via
`markdown` + `xhtml2pdf`, no system libraries; the CLI writes all three files too). Its **Settings** tab edits the git-ignored `.env` in place and
applies changes immediately: API keys (OpenRouter, Tavily, newsdata.io, Semantic Scholar, LangSmith; shown masked,
blank = keep, `none` = clear), the model per tier (dropdowns, with a button that loads the live OpenRouter
catalogue), fallback models and reasoning effort, the embeddings provider/model/dimensions and Chroma
collection, and run limits. "Test OpenRouter key" makes one tiny call and reports the latency.

## Sources

Which tools run is decided per sub-question from the planner's `source_types`, and per tool by
whether its key is present (`/api/health` → `sources`). Keyless tools are always on.

| `source_type` | Tool | Key | Notes |
|---|---|---|---|
| `papers` | arXiv | none | abstracts + PDF links; 1 request / 3 s |
| `papers` | Semantic Scholar | optional `SEMANTIC_SCHOLAR_API_KEY` | the keyless pool returned 429 on every call in testing; request a free key, it is the right source for social-science papers (arXiv is not) |
| `web` | Tavily | `TAVILY_API_KEY` | general web search |
| `reports` | Tavily (reports mode) | `TAVILY_API_KEY` | steered to reports/whitepapers/PDFs |
| `news` | newsdata.io → GDELT | `NEWSDATA_API_KEY` optional | newsdata.io when the key is set (recent news, ~48 h window on `latest`; free plan: 10 articles/request, descriptions only, 30 credits per 15 min), otherwise GDELT (keyless, titles only, 1 request / 5 s). Both tools sanitise queries to plain keywords |
| `wikipedia` | Wikipedia | none | search + intro extracts |
| `api` | none built in | per service | `tools.register("api", YourTool)` for FRED, Polygon, etc. |

The planner emits 2-3 short keyword `search_queries` per sub-question; the retriever runs each of
them against every enabled tool (bounded by a per-tool timeout), de-duplicates by URL, drops
articles that are not in `RETRIEVAL_LANGUAGES` (default `en`; sources that take a language
parameter are asked for those languages up front, everything else is detected from the text with
`langdetect`, and the report's methodology states how many were discarded), embeds and stores. `RETRIEVAL_FETCH_FULL_TEXT=true` additionally fetches PDFs and pages for short results
(GDELT titles, arXiv PDFs) via `tools/fetch.py` (size- and page-capped, best effort).

**For useful results on non-CS topics, add `TAVILY_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY`.** With
only the keyless sources, `papers` means arXiv, which is thin outside maths/physics/CS.

## Deployment

The same FastAPI app runs locally, in a container, on AWS Lambda and on Vercel. Hosted runs need
three things the local defaults do not: `CHECKPOINTER=postgres` with `DATABASE_URL` (plan approval
resumes from the database), `CHROMA_MODE=http` or `cloud` (the evidence store cannot live on an
ephemeral disk), and `EMBEDDINGS_PROVIDER=openrouter` (no local model in the image). The container
image sets all three by default.

### Docker / docker compose (local stand-in for production)

```bash
cp .env.example .env                 # OPENROUTER_API_KEY, API_AUTH_TOKEN
docker compose up --build            # API on :8000 + Postgres + Chroma server
curl -s localhost:8000/api/health | jq .checkpointer,.vector_db
```

`Dockerfile` bundles the AWS Lambda Web Adapter extension, so one image serves plain uvicorn hosts
(ECS Fargate, any VM) and Lambda alike.

### AWS Lambda (recommended; PLAN.md §14)

```bash
pip install aws-sam-cli
cd deploy && sam build && sam deploy --guided \
  --parameter-overrides OpenRouterApiKey=sk-or-... ApiAuthToken=... \
      DatabaseUrl=postgresql://user:pass@host:5432/db ChromaHost=chroma.example ChromaPort=8000
```

`deploy/template.yaml` creates a container-image function (15-minute ceiling, 2 GB) behind a
**streaming Function URL**, which is what keeps the SSE stream alive; API Gateway's 30 s integration
timeout would cut it. Provision Postgres (Aurora Serverless v2 / RDS / Neon) and a Chroma server or
Chroma Cloud separately and pass their endpoints as parameters. Put CloudFront + WAF in front for a
domain and rate limiting, and serve `public/index.html` from S3/CloudFront pointing at the URL.

### Vercel (live: https://multi-agent-deep-researcher-zeta.vercel.app)

Vercel detects the FastAPI preset from `api/index.py` and routes every path to the app, so
`vercel.json` only sets `maxDuration: 300` (the Hobby maximum); do not add rewrites, they rewrite the
path before FastAPI sees it. Deploy with `npx vercel login`, `npx vercel link`, set the env vars
(`npx vercel env add ...`) and `npx vercel deploy --prod`. The live project runs with
`EMBEDDINGS_PROVIDER=openrouter`, `MAX_HOPS=0`, `RUN_BUDGET_SECONDS=240`, Sonnet as the heavy tier
and SQLite + embedded Chroma under `/tmp`: fine for a demo, but plan approval only resumes while the
same function instance is warm. Switch to `CHECKPOINTER=postgres` + `CHROMA_MODE=cloud` for durable
hosting. The API needs the `API_AUTH_TOKEN` bearer; the hosted page at `/` asks for it.

### What is and is not verified

Verified locally: the image builds and answers `/api/health`; the Postgres checkpointer persists a
plan interrupt that a second process resumes; the Chroma http mode stores and queries evidence against
a Chroma server (both via `tests/test_deploy.py` when `DR_TEST_DATABASE_URL` / `DR_TEST_CHROMA_HOST` are
set; `ci.yml` builds and smoke-tests the image on every push). Not verified: an actual deployment to
AWS or Vercel, which needs your account credentials.

## LLM provider: OpenRouter

All agents get their chat model from [`src/deep_researcher/llm.py`](src/deep_researcher/llm.py),
which routes through [OpenRouter](https://openrouter.ai) using LangChain's OpenAI-compatible
`ChatOpenAI` client with OpenRouter's base URL. One key, any vendor, per-role model choice,
automatic fallbacks. Structured output uses the `json_schema` method (`llm.structured(...)`).

```bash
OPENROUTER_API_KEY=sk-or-...
```

| Tier | Roles | Default model | Override |
|---|---|---|---|
| heavy | planner, analysis, insight, critic | `anthropic/claude-opus-5.5` | `OPENROUTER_MODEL_HEAVY` |
| worker | retriever, report | `anthropic/claude-sonnet-5` | `OPENROUTER_MODEL_WORKER` |
| cheap | validator, grader | `anthropic/claude-haiku-4.5` | `OPENROUTER_MODEL_CHEAP` |

Any single role can be pinned with `OPENROUTER_MODEL_<ROLE>`, e.g.
`OPENROUTER_MODEL_INSIGHT=openai/gpt-5.1`. `OPENROUTER_FALLBACK_MODELS` lists backups tried in
order when the primary is unavailable, and `OPENROUTER_PROVIDER_ORDER` / `OPENROUTER_ZDR` /
`OPENROUTER_SORT` control provider routing; `OPENROUTER_REASONING_EFFORT` caps hidden reasoning (its
tokens count against each role's `max_tokens`, which are set generously for that reason).
`/api/health` shows the resolved model per role.

```python
from deep_researcher.llm import get_llm
get_llm("planner").invoke("...")           # ChatOpenRouter, traced by LangSmith automatically
```

## Embeddings and the vector DB

Evidence is chunked into **512-token** pieces (64-token overlap, sentence/paragraph boundaries,
`CHUNK_SIZE` / `CHUNK_OVERLAP` / `CHUNK_UNIT`), embedded as **256-dimensional** vectors
(`EMBEDDINGS_DIMENSIONS`), and stored in a **persistent embedded Chroma** collection
(`CHROMA_PATH`, default `.chroma/`). 256 dimensions means a 3–6× smaller index than the models'
native sizes; it is only honest for Matryoshka-trained models, so the factory refuses fixed-size
models (bge, e5, gte, MiniLM) at reduced dimensions instead of silently truncating them.

The embedding model is chosen by `EMBEDDINGS_PROVIDER`; the default needs **no API key and no
server**:

| Provider | Default model (all 256-d) | Install | Notes |
|---|---|---|---|
| `local` (default) | `Snowflake/snowflake-arctic-embed-m-v1.5` | `uv sync --extra embeddings-local` | MRL-trained specifically for 256-d; 109M params, 512-token window, ~440 MB, CPU; the model's query prompt is applied automatically. `EMBEDDINGS_DEVICE=cuda`/`mps` for a GPU |
| `openrouter` | `openai/text-embedding-3-small` | `uv sync --extra embeddings-openai` | `dimensions=256` honoured by OpenRouter (verified). Uses `OPENROUTER_API_KEY`. $0.02 / M tokens. **Use this on AWS / Vercel.** |
| `openai` | `text-embedding-3-small` | `uv sync --extra embeddings-openai` | Same weights as the `openrouter` default → same collection. Any OpenAI-compatible endpoint via `EMBEDDINGS_BASE_URL` |
| `ollama` | `nomic-embed-text` | `uv sync --extra embeddings-ollama` | v1.5 is Matryoshka; `dimensions` passed through |

Opt-in local alternative: `EMBEDDINGS_MODEL=google/embeddinggemma-2` (`--extra embeddings-gemma`), multilingual
and 8k-context, near-lossless at 256-d, but ~6× slower than arctic on CPU (PLAN §13 has the numbers).

The store writes the embedding fingerprint (canonical model name plus dimensions, e.g.
`text-embedding-3-small@256`) and the vector dimension into the collection's metadata and refuses
to open it with a different model. The fingerprint is model-based rather than provider-based, so
`openrouter` and `openai` share a collection; the local default is a *different* model, so a
laptop-built collection is not portable to a hosted deployment under these defaults (set
`EMBEDDINGS_PROVIDER=openrouter` locally if you need that). Chunk ids are deterministic, so
re-ingesting a source is an idempotent upsert. Chroma is embedded and single-writer.

```python
from deep_researcher import embeddings
from deep_researcher.store import EvidenceStore

store = EvidenceStore(embeddings.get_embeddings(), embedding_id=embeddings.embedding_id())
store.add_evidence([...])                       # Evidence -> 512-token chunks -> 256-d vectors, persisted
hits = store.query("four-day week productivity", k=10, sub_question_id="sq1")
```

## Tracing with LangSmith

Set three variables (locally in `.env`, on Vercel in the project settings) and every request
becomes a LangSmith trace:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=deep-researcher
```

What you get:

- One parent run per API call (`api.research`, `api.resume`) tagged `api` and `vercel`/`local`,
  with `thread_id` in metadata so a whole investigation can be filtered in one view.
- LangGraph node, tool and model calls nest under that parent automatically; retrievers use the
  `retriever` run type so LangSmith renders documents properly.
- The first SSE event carries `trace_url`, so the UI can link straight to the trace.
- `/api/health` reports `tracing.enabled` and the project, never the key.
- On Vercel the client is flushed before the function returns and LangChain callbacks run
  synchronously, so traces are not lost when the instance is frozen.

Tracing is a strict no-op when the variables are unset. The helpers live in
[`src/deep_researcher/tracing.py`](src/deep_researcher/tracing.py): `span()` for a parent run,
`traced()` for tools and agents, `run_config()` to name and tag a graph run, `flush()`.

## Protecting the API

The research endpoint spends real money per call, so three env vars guard it (all optional):

| Variable | Default | Effect |
|---|---|---|
| `API_AUTH_TOKEN` | unset (open) | When set, `POST /api/research*` require `Authorization: Bearer <token>` or `X-API-Key`. The hosted UI shows a token field when health reports `auth_required`. |
| `RATE_LIMIT_PER_MINUTE` | `6` | Sliding window per client IP; `0` disables. Idle keys are evicted so memory stays bounded. |
| `MAX_CONCURRENT_RUNS` | `3` | Simultaneous research streams per instance; extra requests get `503` with `Retry-After`. Released on every exit path, including client disconnects. |
| `TRUST_PROXY_HEADERS` | on when `VERCEL` is set, else off | Use `x-real-ip` / `x-vercel-forwarded-for` / `x-forwarded-for` as the rate-limit key. Off by default locally because those headers are forgeable without a proxy in front. |

Set `API_AUTH_TOKEN` on Vercel before sharing the URL. The limiter is in-process, which bounds
burst cost per instance; swap in Upstash/Redis if you need a global limit. `/api/health` stays
public and never exposes secrets.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness, graph/tracing/llm/embeddings/chunking config, enabled sources |
| POST | `/api/research` | `{question, max_hops?}` → SSE stream until plan approval |
| POST | `/api/research/{thread_id}/resume` | `{plan: ResearchPlan}` → SSE stream to final report |
| POST | `/api/report/pdf` | `{markdown, title?}` → `application/pdf`; stateless, so the hosted page can turn the streamed report into a download |

SSE events: `started` (with `trace_url`), `plan` (awaiting approval), one `retrieve` per sub-question (parallel),
`validate` (assessed/kept), `analyze` (claims, contradictions, gaps), `route_gaps` (next hop or stop), repeated per
hop, then `insight` (hypotheses), `critic` (verdict, issues, follow-up; may loop back), `report` (Markdown in
`report`, structured record in `report_json`, `stats`), `error`, `done`.

## License

No license has been chosen yet; all rights reserved until one is added.

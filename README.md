# LangGraph Multi-Agent Chatbot

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3%2B-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-teal)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Live on Render · 3 integrated tools (RAG, web search, code execution) · Persistent memory across sessions**

Most chatbots either answer from memory or bolt on a single tool. This one decides, per message, whether it needs to retrieve documents, search the web, run code, or just answer — then chains multiple tools together if the first result isn't enough, all while remembering the conversation across requests.

A LangGraph-based chatbot that routes each query to the right capability — RAG retrieval, live web search, sandboxed Python execution (via MCP), or a direct LLM answer — with PostgreSQL-backed multi-turn memory, served over FastAPI.

**Live API:** `https://langgraph-multi-agent-ee8b.onrender.com` (`/health`, `/chat`)

---

## What it does

Every incoming message goes through a **router** that makes a binary decision — does this need a tool, or can the LLM just answer directly? If a tool is needed, a second node picks the *specific* tool (RAG / web search / Python execution) and calls it; the result is fed back to the LLM, which can chain further tool calls (up to `MAX_ITERATIONS`) before producing a final answer. Conversation state is checkpointed to Postgres per `thread_id`, so multi-turn context survives across requests.

- **Bilingual** — replies in Roman Urdu or English depending on the input language
- **Multi-step tool use** — the agent can call a tool, inspect the result, and call another tool before answering (bounded by `MAX_ITERATIONS`)
- **Resilient answer generation** — retries once on an empty LLM completion instead of returning a blank reply

## Architecture

```
User query
    │
    ▼
┌────────────┐
│   router   │   binary decision: does this need a tool?
└─────┬──────┘
      │
 ┌────┴─────┐
 │          │
 ▼          ▼
llm_tool   answer ──▶ END
 │  ▲
 │  │ (loop while iteration_count < MAX_ITERATIONS)
 ▼  │
tools ──┘
(RAG / web_search / python_tool)
```

- **`router`** — one structured-output LLM call classifies the query as `tools` or `answer` ([`nodes/router.py`](nodes/router.py))
- **`llm_tool`** — binds all available tools to the LLM and forces a tool call, picking the specific tool for the request ([`nodes/llm_tool.py`](nodes/llm_tool.py))
- **`tools`** — a LangGraph `ToolNode` that actually executes the chosen tool (RAG retrieval, Tavily search, or a remote Python REPL over MCP)
- **`answer`** — generates the final reply from tool output + recent chat history ([`nodes/answer.py`](nodes/answer.py))

## Tools available to the agent

| Tool | Backing service | What it does |
|---|---|---|
| `RAG` | Pinecone (MMR retrieval) | Searches previously indexed documents |
| `search_web` | Tavily | Live web search, top 5 results |
| `python_tool` | [fastmcp-python-repl-server](https://github.com/arbaz-builds/fastmcp-python-repl-server) over MCP | Executes Python code in a remote sandboxed REPL |

> **Note on RAG quality:** embeddings currently use a `MockEmbeddings` placeholder (fixed zero-vectors, 1536-dim to stay Pinecone-index-compatible) so the pipeline runs without requiring an OpenAI key. Retrieval will run without errors but won't return semantically meaningful matches until this is swapped for a real embedding model (see [`llms.py`](llms.py)).

## Project structure

```
langgraph-multi-agent/
├── main.py               # FastAPI app (/health, /chat) + CLI entry point
├── graph.py               # LangGraph graph assembly
├── state.py               # State schema (TypedDict + Pydantic RouterDecision)
├── llms.py                # LLM + embeddings setup
├── config.py              # Settings loaded from .env, with validation
├── nodes/
│   ├── router.py          # Router node + routing condition
│   ├── llm_tool.py         # Tool-binding node + tool list
│   ├── conditions.py       # Edge conditions (tool_or_answer, after_tools)
│   └── answer.py           # Final answer generation
├── tools/
│   ├── rag.py              # Pinecone-backed RAG tool
│   ├── web_search.py       # Tavily web search tool
│   └── mcp_client.py       # MCP client for the remote Python REPL
├── docs/
│   ├── architecture.md     # Deep-dive architecture notes
│   └── api_reference.md    # API reference
├── tests/
│   └── test_agent.py       # pytest suite
├── .env.example
├── requirements.txt
└── README.md
```

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/arbaz-builds/langgraph-multi-agent.git
cd langgraph-multi-agent
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# fill in your API keys
```

### 3. Run the API locally

```bash
uvicorn main:app --reload
```

- `GET /health` → `{"status": "ok"}`
- `POST /chat` → `{"query": "...", "thread_id": "1"}` → `{"response": "..."}`

### 4. Or run a single query from the CLI

```python
import asyncio
from main import build_and_run

print(asyncio.run(build_and_run("python version check karo")))
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | ✅ | LLM inference via NVIDIA NIM |
| `PINECONE_API_KEY` | ✅ | Vector store for RAG |
| `TAVILY_API_KEY` | ✅ | Web search |
| `DATABASE_URL` | ✅ | Postgres connection string (conversation memory) |
| `MCP_SERVER_URL` | optional | Python REPL MCP server (defaults to the author's deployed instance) |
| `NVIDIA_BASE_URL`, `NVIDIA_MODEL` | optional | Override the default NIM endpoint/model |
| `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `LANGCHAIN_TRACING_V2` | optional | LangSmith tracing |
| `MAX_ITERATIONS`, `RETRIEVER_K`, `RETRIEVER_FETCH_K` | optional | Agent tuning (tool-call loop cap, RAG retrieval size) |

`config.validate()` checks the four required keys on startup and raises immediately with a clear error if any are missing, rather than failing later mid-request.

## A note on Postgres connections

Each `/chat` call opens a **fresh** Postgres connection (`AsyncPostgresSaver.from_conn_string(...)` inside the request, not once at startup). This is deliberate: reusing a single long-lived connection across requests meant that once a serverless Postgres provider (Neon) or free-tier host (Render) silently dropped an idle connection, every subsequent request would fail until the process restarted. Opening per-request avoids that failure mode at the cost of a small per-request connection overhead.

## Tech stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | NVIDIA NIM (`openai/gpt-oss-20b` by default) |
| Vector DB | Pinecone |
| Web search | Tavily |
| Code execution | Remote Python REPL via MCP ([fastmcp-python-repl-server](https://github.com/arbaz-builds/fastmcp-python-repl-server)) |
| Memory | PostgreSQL (Neon), via `langgraph-checkpoint-postgres` |
| Web server | FastAPI + uvicorn |
| Observability | LangSmith (optional) |

## Related projects

- [fastmcp-python-repl-server](https://github.com/arbaz-builds/fastmcp-python-repl-server) — the MCP server providing this agent's `python_tool`
- [langgraph](https://github.com/langchain-ai/langgraph) — the graph framework this project is built on
- [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) — bridges MCP tools into LangChain's tool interface

## Contributing

Contributions, issues, and feature requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT © [Mohammad Arbaz](https://github.com/arbaz-builds)

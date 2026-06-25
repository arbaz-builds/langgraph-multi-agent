# 🤖 LangGraph Multi-Agent Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3%2B-green?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-orange?style=for-the-badge)
![FastMCP](https://img.shields.io/badge/FastMCP-0.2%2B-purple?style=for-the-badge)
![Tavily](https://img.shields.io/badge/Tavily-Search-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=for-the-badge)

> A production-grade **Multi-Agent AI Chatbot** built with LangGraph — featuring intelligent query routing, RAG pipeline, real-time web search via Tavily, Python code execution via MCP, and persistent memory backed by PostgreSQL.

**[⭐ Star this repo if it helps you!](#)**

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔀 **Smart Router** | Classifies every query and routes it to the right tool automatically |
| 🐍 **Python Executor** | Runs Python code in real-time via a remote FastMCP server |
| 🔍 **Web Search** | Real-time search powered by Tavily |
| 📄 **RAG Pipeline** | Document Q&A using Pinecone Vector DB + MMR retrieval |
| 🧠 **Persistent Memory** | Multi-thread conversation memory backed by PostgreSQL (Neon) |
| 📊 **Observability** | Full tracing and monitoring via LangSmith |
| 🌐 **Bilingual** | Responds in Roman Urdu or English based on user input |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌──────────────┐
│ router_node  │  Classifies → python_tool / rag / web_search / direct
└──────┬───────┘
       │
  ┌────┴────────────┬─────────────────┐
  ▼                 ▼                 ▼
python_tool      web_search          rag
(FastMCP)        (Tavily)         (Pinecone)
       │
       ▼
┌──────────────┐
│ answer_node  │  Generates final response
└──────────────┘
```

---

## 📁 Project Structure

```
langgraph-multi-agent/
├── main.py              # Entry point — run the chatbot
├── graph.py             # LangGraph graph assembly
├── state.py             # State schema (TypedDict + Pydantic)
├── llms.py              # LLM and embeddings setup
├── config.py            # All settings from .env
├── nodes/
│   ├── router.py        # Router node + route condition
│   ├── llm_tool.py      # LLM tool-calling node
│   └── answer.py        # Final answer generation node
├── tools/
│   ├── rag.py           # RAG tool (Pinecone)
│   ├── web_search.py    # Web search tool (Tavily)
│   └── mcp_client.py   # MCP client (FastMCP Python REPL)
├── docs/
│   ├── architecture.md  # Deep-dive architecture docs
│   └── api_reference.md # API reference
├── tests/
│   └── test_agent.py    # pytest test suite
├── .env.example         # Environment variables template
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/arbaz-builds/langgraph-multi-agent.git
cd langgraph-multi-agent
pip install -r requirements.txt
```

### 2. Set up Environment

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Run

```python
import asyncio
from main import run

result = asyncio.run(run("python version check karo", thread_id="1"))
print(result)
```

---

## 🔧 Environment Variables

| Variable | Required | Description | Get it from |
|----------|----------|-------------|-------------|
| `NVIDIA_API_KEY` | ✅ | LLM inference | [build.nvidia.com](https://build.nvidia.com) |
| `OPENAI_API_KEY` | ✅ | Embeddings | [platform.openai.com](https://platform.openai.com) |
| `PINECONE_API_KEY` | ✅ | Vector database | [pinecone.io](https://pinecone.io) |
| `TAVILY_API_KEY` | ✅ | Web search | [tavily.com](https://tavily.com) |
| `DATABASE_URL` | ✅ | PostgreSQL memory | [neon.tech](https://neon.tech) |
| `MCP_SERVER_URL` | ✅ | FastMCP Python REPL | [fastmcp-python-repl-server](https://github.com/arbaz-builds/fastmcp-python-repl-server) |
| `LANGCHAIN_API_KEY` | ⚙️ optional | LangSmith tracing | [smith.langchain.com](https://smith.langchain.com) |

---

## 🧠 How It Works

1. **User sends a query** — in English or Roman Urdu
2. **Router node** classifies it into one of 4 routes:
   - `python_tool` → executes Python via FastMCP
   - `web_search` → fetches real-time results via Tavily
   - `rag` → retrieves from uploaded documents via Pinecone
   - `direct` → answers directly without tools
3. **LLM tool node** calls the appropriate tool
4. **Answer node** generates a clean, concise response

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph |
| LLM | NVIDIA NIM (gpt-oss-20b) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | Pinecone |
| Web Search | Tavily |
| Code Execution | FastMCP Python REPL |
| Memory | PostgreSQL via Neon |
| Observability | LangSmith |

---

## 🔗 Related Projects

- [fastmcp-python-repl-server](https://github.com/arbaz-builds/fastmcp-python-repl-server) — MCP Python REPL server used by this agent
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — Graph framework powering this project
- [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) — MCP integration for LangChain

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT © [Arbaz](https://github.com/arbaz-builds)

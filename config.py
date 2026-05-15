"""
Configuration — all settings loaded from environment variables.
Copy .env.example to .env and fill in your keys before running.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ───────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

# ── LLM (NVIDIA NIM) ──────────────────────────────────────────────────────
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")

# ── Vector DB (Pinecone) ──────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "vector")

# ── Observability (LangSmith) ─────────────────────────────────────────────
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "langgraph-agent")
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false")

# ── Web Search (Apify) ────────────────────────────────────────────────────
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")

# ── Database (Neon PostgreSQL) ────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")

# ── MCP Server ────────────────────────────────────────────────────────────
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://deploy-9ugq.onrender.com/mcp")

# ── Agent Tuning ──────────────────────────────────────────────────────────
MAX_ITERATIONS    = int(os.getenv("MAX_ITERATIONS", "3"))
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVER_K       = int(os.getenv("RETRIEVER_K", "5"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "20"))

# ── Validation ────────────────────────────────────────────────────────────
REQUIRED = {
    "NVIDIA_API_KEY": NVIDIA_API_KEY,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "DATABASE_URL": DATABASE_URL,
}

def validate():
    missing = [k for k, v in REQUIRED.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {missing}\n"
            "Copy .env.example to .env and fill in your keys."
        )

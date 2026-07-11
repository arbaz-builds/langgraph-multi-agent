"""Configuration — all settings from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM (NVIDIA NIM)
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")

# Vector DB (Pinecone)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "vector")

# Web Search (Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Database (Neon PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL")

# MCP Server
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://deploy-9ugq.onrender.com/mcp")

# Observability (LangSmith)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "langgraph-agent")
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false")

# Agent Tuning
MAX_ITERATIONS    = int(os.getenv("MAX_ITERATIONS", "3"))
RETRIEVER_K       = int(os.getenv("RETRIEVER_K", "5"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "20"))


# Validation
def validate():
    missing = [k for k, v in {
        "NVIDIA_API_KEY":  NVIDIA_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
        "TAVILY_API_KEY":  TAVILY_API_KEY,
        "DATABASE_URL":    DATABASE_URL,
    }.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}. Copy .env.example to .env")

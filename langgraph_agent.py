# ── Imports ──────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool as tool_decorator
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, List
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
import asyncio, os, requests
import nest_asyncio
import config

nest_asyncio.apply()

# ── Validate environment on startup ──────────────────────────────────────────
config.validate()

# ── Environment (LangSmith tracing) ──────────────────────────────────────────
os.environ["LANGCHAIN_PROJECT"]    = config.LANGCHAIN_PROJECT
os.environ["LANGCHAIN_API_KEY"]    = config.LANGCHAIN_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING

# ── LLMs ─────────────────────────────────────────────────────────────────────
answer_LLM = ChatOpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY, model=config.NVIDIA_MODEL)
router_LLM = ChatOpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY, model=config.NVIDIA_MODEL, temperature=0)

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=config.OPENAI_API_KEY
)

# ── State & Schema ────────────────────────────────────────────────────────────
class RouterDecision(BaseModel):
    decision:  Literal["rag", "web_search", "python_tool", "direct"] = Field(description="Routing decision")
    reasoning: str = Field(min_length=5, max_length=200, description="Brief explanation")

class State(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    file_uploaded:   bool
    router_decision: str
    reasoning:       str
    iteration_count: int

# ── Router Prompt ─────────────────────────────────────────────────────────────
ROUTER_PROMPT = """You are a routing assistant. Classify the query into one of:
- python_tool : code, python, packages, install, pip, version, environment, libraries, "karo", "check karo", "chalao"
- rag         : user asking about uploaded file content — "document", "file", "PDF", "report", "uploaded", "mera data"
- web_search  : news, weather, latest/newest/current/recent/today, real-time info, price, stock, "kab release hoga"
- direct      : general knowledge, definitions, greetings, simple questions (no code, no file, no real-time)

Rules:
- Any technical/code intent                          → python_tool
- "latest", "newest", "current", "kab release hoga" → web_search
- ANY mention of file/document/PDF                   → rag
- Doubt between python_tool vs direct                → python_tool
"""

# ── Document Loader ───────────────────────────────────────────────────────────

_retriever_store: Dict = {"bm25": None}

def load_document(file_path: str) -> str:
    """Load and index a document (PDF/CSV/TXT/DOCX) into Pinecone + BM25."""
    try:
        # Security: Basic path validation
        if not os.path.exists(file_path):
            return "Error: File not found."
        
        if not os.path.isfile(file_path):
            return "Error: Path is not a file."
        
        # File size limit (e.g., 50MB)
        max_size_mb = 50
        if os.path.getsize(file_path) > max_size_mb * 1024 * 1024:
            return f"Error: File too large. Maximum allowed size is {max_size_mb}MB."

        ext = os.path.splitext(file_path)[1].lower()
        
        # Loader mapping
        loader_map = {
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
        }
        
        if ext not in loader_map:
            return "Error: Unsupported format. Supported: .pdf, .csv, .txt, .docx"
        
        # Load document
        loader_class = loader_map[ext]
        if ext == ".txt":
            loader = loader_class(file_path, encoding="utf-8")
        else:
            loader = loader_class(file_path)
        
        docs = loader.load()
        
        if not docs:
            return "Warning: File is empty or could not be read."
        
        # Add metadata to each chunk (very important)
        for doc in docs:
            doc.metadata.update({
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "file_type": ext,
                "upload_time": "now"  # You can use datetime here
            })
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        
        if not chunks:
            return "Warning: No chunks created from document."
        
        # BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 3  # Increased a bit
        _retriever_store["bm25"] = bm25_retriever
        
        # Pinecone Vector Store
        PineconeVectorStore.from_documents(
            chunks,
            embedding=embeddings,
            index_name=config.PINECONE_INDEX,
            pinecone_api_key=config.PINECONE_API_KEY
        )
        
        return f"✅ Document '{os.path.basename(file_path)}' indexed successfully. {len(chunks)} chunks stored."
        
    except Exception as e:
        # Better error handling (don't expose full traceback in production)
        error_msg = str(e)
        if "No module" in error_msg:
            return f"Error: Missing dependency - {error_msg}"
        return f"Error while processing document: {error_msg}"

# ── RAG Tool ──────────────────────────────────────────────────────────────────
@tool_decorator
def RAG(query: str) -> str:
    """Retrieve relevant content from previously uploaded documents."""
    try:
        vs   = PineconeVectorStore(
            index_name=config.PINECONE_INDEX,
            embedding=embeddings,
            pinecone_api_key=config.PINECONE_API_KEY
        )
        docs = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.RETRIEVER_K, "fetch_k": config.RETRIEVER_FETCH_K, "lambda_mult": 0.7}
        ).invoke(query)
        if not docs:
            return "No relevant content found in uploaded documents."
        return "".join(f"Source {i+1}:\n{d.page_content.strip()}\n\n" for i, d in enumerate(docs))
    except Exception as e:
        return f"RAG error: {e}"

# ── Web Search Tool ───────────────────────────────────────────────────────────
_SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "menu"}
_NAV_KW    = ["Log in", "Sign up", "Cookie", "Subscribe", "Newsletter",
              "Privacy", "Terms", "Click here", "Skip to", "Jump to"]

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text, self.skip = [], False
    def handle_starttag(self, tag, attrs): self.skip = tag in _SKIP_TAGS
    def handle_endtag(self, tag):          self.skip = False
    def handle_data(self, data):
        if not self.skip and data.strip(): self.text.append(data.strip())

def _fetch_clean_content(url: str, max_chars: int = 1500):
    try:
        page = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if any(w in page.text for w in ["Enable JavaScript", "Just a moment", "Cloudflare"]):
            return None
        p = _TextExtractor()
        p.feed(page.text)
        raw       = " ".join(" ".join(p.text).split())
        sentences = [s.strip() for s in raw.split(".") if len(s.strip()) > 50]
        clean     = [s for s in sentences if not any(k.lower() in s.lower() for k in _NAV_KW)]
        result    = ". ".join(clean[:15])[:max_chars]
        return result if len(result) > 100 else None
    except Exception:
        return None

@tool_decorator
def search_web(query: str) -> str:
    """Search Google for current, real-time information."""
    try:
        resp = requests.post(
            "https://api.apify.com/v2/acts/apify~google-search-scraper/run-sync-get-dataset-items",
            params={"token": config.APIFY_TOKEN},
            json={"queries": query, "maxPagesPerQuery": 1, "resultsPerPage": 8,
                  "languageCode": "en", "countryCode": "us"},
            timeout=60
        )
        organic = resp.json()[0].get("organicResults", [])
        if not organic:
            return "No search results found."
        filtered = [r for r in organic if not any(
            s in r.get("url", "") for s in ["youtube.com", "reddit.com", "twitter.com"]
        )][:5]
        contents = list(ThreadPoolExecutor(3).map(
            _fetch_clean_content, [r.get("url", "") for r in filtered[:3]]
        ))
        out = f"Search: '{query}'\n" + "=" * 50 + "\n\n"
        for i, r in enumerate(filtered):
            out += f"[{i+1}] {r.get('title', '')}\n"
            out += f"Snippet: {r.get('description', '').replace('Read more', '').strip()}\n"
            if i < 3 and contents[i]:
                out += f"Content: {contents[i]}\n"
            out += "\n" + "-" * 45 + "\n\n"
        return out
    except Exception as e:
        return f"Search error: {e}"

# ── MCP Client ────────────────────────────────────────────────────────────────
_mcp_client = MultiServerMCPClient({
    "python_tool": {"transport": "streamable-http", "url": config.MCP_SERVER_URL}
})

# ── Tool routing map ──────────────────────────────────────────────────────────
_TOOL_MAP = {
    "python_tool": "run_python",
    "rag":         "RAG",
    "web_search":  "search_web",
}

# ── Main agent ────────────────────────────────────────────────────────────────
async def build_and_run(query: str, thread_id: str = "1") -> str:
    """
    Run the LangGraph multi-agent chatbot.

    Args:
        query:     User input string.
        thread_id: Conversation ID — same ID = same memory thread.

    Returns:
        Agent response as a plain string.
    """
    mcp_tools = await asyncio.wait_for(_mcp_client.get_tools(), timeout=30)
    all_tools  = [RAG, search_web, *mcp_tools]
    tool_exec  = ToolNode(all_tools)

    # ── Nodes ──────────────────────────────────────────────────────────────
    async def router_node(state: State):
        try:
            r = await router_LLM.with_structured_output(RouterDecision).ainvoke([
                SystemMessage(content=ROUTER_PROMPT + f"\nfile_uploaded={state.get('file_uploaded', False)}"),
                HumanMessage(content=state["messages"][-1].content)
            ])
            return {"router_decision": r.decision, "reasoning": r.reasoning}
        except Exception as e:
            return {"router_decision": "direct", "reasoning": str(e)}

    async def llm_tool_node(state: State):
        msgs   = state["messages"]
        dec    = state.get("router_decision")
        itr    = state.get("iteration_count", 0)
        if isinstance(msgs[-1], ToolMessage):
            return {"messages": [], "iteration_count": itr}
        forced = _TOOL_MAP.get(dec)
        sp = f"""You are an AI assistant. Routing decision: {dec}.
You MUST call the tool: {forced}
- run_python  → write complete Python code using print() for output
- RAG         → pass the user question exactly as-is
- search_web  → pass the user question exactly as-is
Do NOT reply with text. Only make a tool call."""
        resp = await answer_LLM.bind_tools(all_tools, tool_choice=forced).ainvoke(
            [SystemMessage(content=sp)] + msgs[-6:]
        )
        return {"messages": [resp], "iteration_count": itr + 1}

    async def answer_node(state: State):
        msgs     = state["messages"]
        tool_out = next((m.content for m in reversed(msgs) if isinstance(m, ToolMessage)), None)
        # MCP tool output comes as list[dict] — parse to plain string
        if isinstance(tool_out, list):
            tool_out = "\n".join(
                item.get("text", "") for item in tool_out if item.get("type") == "text"
            ).strip()
        sp = """You are a helpful AI assistant. Rules:

LANGUAGE:
- Roman Urdu query → reply in Roman Urdu
- English query → reply in English
- Never use Urdu/Hindi script (no ا ب پ)

FORMAT:
- Plain text only — no markdown, no bullet points, no bold, no headers
- Short and natural — 2 to 4 lines max
- No filler phrases like "Sure!", "Great!", "Of course!"

TOOL RESULT:
- If a tool result is provided, explain it clearly in 1-2 lines
- State key values directly (version number, sum, price, etc.)
- Do not repeat the raw tool output verbatim"""
        if tool_out:
            sp += f"\n\nTool result:\n{tool_out}"
        history = [m for m in msgs[-10:] if not isinstance(m, ToolMessage)]
        resp = await answer_LLM.ainvoke([SystemMessage(content=sp)] + history)
        return {"messages": [resp]}

    # ── Conditions ─────────────────────────────────────────────────────────
    def route_condition(state: State):
        return "llm_tool_node" if state["router_decision"] in _TOOL_MAP else "answer_node"

    def tool_condition(state: State):
        last = state["messages"][-1]
        itr  = state.get("iteration_count", 0)
        if itr >= config.MAX_ITERATIONS:                      return "answer_node"
        if isinstance(last, ToolMessage):                     return "answer_node"
        if isinstance(last, AIMessage) and last.tool_calls:  return "tool_exec"
        return "answer_node"

    # ── Graph ───────────────────────────────────────────────────────────────
    g = StateGraph(State)
    g.add_node("router_node",   router_node)
    g.add_node("llm_tool_node", llm_tool_node)
    g.add_node("tool_exec",     tool_exec)
    g.add_node("answer_node",   answer_node)
    g.add_edge(START, "router_node")
    g.add_conditional_edges("router_node",   route_condition, {"llm_tool_node": "llm_tool_node", "answer_node": "answer_node"})
    g.add_conditional_edges("llm_tool_node", tool_condition,  {"tool_exec": "tool_exec",         "answer_node": "answer_node"})
    g.add_edge("tool_exec",   "answer_node")
    g.add_edge("answer_node", END)

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
        await cp.setup()
        bot = g.compile(checkpointer=cp)
        res = await bot.ainvoke(
            {"messages": [HumanMessage(content=query)], "file_uploaded": False, "iteration_count": 0},
            config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1].content

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = asyncio.get_event_loop().run_until_complete(
        build_and_run("python version check karo", thread_id="1")
    )
    print(result)

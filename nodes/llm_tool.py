"""LLM tool node — binds tools and forces a tool call."""
from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM
from tools import RAG, search_web, mcp_client
from langgraph.prebuilt import ToolNode

_tools_cache = None
_tool_node_cache = None


async def get_tools() -> list:
    """Fetch tools once and cache them (RAG + web search + MCP tools)."""
    global _tools_cache
    if _tools_cache is None:
        mcp_tools = await mcp_client.get_tools()
        _tools_cache = [RAG, search_web, *mcp_tools]
    return _tools_cache


async def get_tool_node() -> ToolNode:
    """Build (and cache) the ToolNode used by graph.py, so graph.py never
    has to construct it itself — keeps all tool wiring in one place."""
    global _tool_node_cache
    if _tool_node_cache is None:
        tools = await get_tools()
        _tool_node_cache = ToolNode(tools)
    return _tool_node_cache


async def llm_tool_node(state: State):
    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [], "iteration_count": state.get("iteration_count", 0)}

    tools = await get_tools()
    resp = await answer_LLM.bind_tools(tools).ainvoke(
        [SystemMessage(content="You are an AI assistant. Always call a tool. Never reply with text.")] +
        state["messages"][-6:]
    )
    return {"messages": [resp], "iteration_count": state.get("iteration_count", 0) + 1}

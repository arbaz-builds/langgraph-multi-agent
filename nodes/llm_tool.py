"""LLM tool node — binds tools and forces a tool call."""
from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM
from tools import RAG, search_web, mcp_client
from langgraph.prebuilt import ToolNode

_tools_cache = [RAG, search_web]
multi_tools = ToolNode(_tools_cache)


async def get_tools() -> list:
    """Fetch tools once and cache them (RAG + web search + MCP tools)."""
    global _tools_cache, multi_tools
    mcp_tools = await mcp_client.get_tools()
    _tools_cache = [RAG, search_web, *mcp_tools]
    multi_tools = ToolNode(_tools_cache)
    return _tools_cache


async def llm_tool_node(state: State):
    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [], "iteration_count": state.get("iteration_count", 0)}

    tools = await get_tools()
    resp = await answer_LLM.bind_tools(tools).ainvoke(
        [SystemMessage(content="You are an AI assistant. Always call a tool. Never reply with text.")] +
        state["messages"][-6:]
    )
    return {"messages": [resp], "iteration_count": state.get("iteration_count", 0) + 1}

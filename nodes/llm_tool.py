"""LLM tool node — binds tools, forces a tool call, and owns tool-routing conditions."""
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from state import State
from llms import answer_LLM
from tools import RAG, search_web, mcp_client
import config
from langgraph.prebuilt import ToolNode

_tools_cache = None


async def get_tools() -> list:
    """Fetch tools once and cache them (RAG + web search + MCP tools)."""
    global _tools_cache
    if _tools_cache is None:
        mcp_tools = await mcp_client.get_tools()
        _tools_cache = [RAG, search_web, *mcp_tools]
    return _tools_cache
multi_tool=Tool_Node(_tools_cache)


async def llm_tool_node(state: State):
    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [], "iteration_count": state.get("iteration_count", 0)}

    tools = await get_tools()
    resp = await answer_LLM.bind_tools(tools).ainvoke(
        [SystemMessage(content="You are an AI assistant. Always call a tool. Never reply with text.")] +
        state["messages"][-6:]
    )
    return {"messages": [resp], "iteration_count": state.get("iteration_count", 0) + 1}


def tool_or_answer(state: State) -> str:
    """After llm_tool: go to 'tools' if a tool call was made and we're
    still under the iteration limit, otherwise go straight to 'answer'."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and state.get("iteration_count", 0) < config.MAX_ITERATIONS:
        return "tools"
    return "answer"


def after_tools(state: State) -> str:
    """After tools: loop back to llm_tool while under the iteration limit
    (enables multi-step tool chaining), otherwise go to answer."""
    if state.get("iteration_count", 0) < config.MAX_ITERATIONS:
        return "llm_tool"
    return "answer"

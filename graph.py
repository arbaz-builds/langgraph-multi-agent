from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from state import State
from nodes.router import router_node
from nodes.answer import answer_node
from nodes.llm_tool import llm_tool_node
from tools import RAG, search_web, mcp_client
import config
import asyncio


async def build_graph():
    """Build and return compiled LangGraph with PostgreSQL checkpointer."""
    mcp_tools = await asyncio.wait_for(mcp_client.get_tools(), timeout=30)
    all_tools  = [RAG, search_web, *mcp_tools]
    tool_exec  = ToolNode(all_tools)

    # ── Wrap nodes that need all_tools ────────────────────────────────────
    async def _llm_tool_node(state: State):
        return await llm_tool_node(state, all_tools)

    # ── Edge conditions ───────────────────────────────────────────────────
    def route_condition(state: State):
        return (
            "llm_tool_node"
            if state["router_decision"] in {"python_tool", "rag", "web_search"}
            else "answer_node"
        )

    def tool_condition(state: State):
        last = state["messages"][-1]
        itr  = state.get("iteration_count", 0)
        if itr >= config.MAX_ITERATIONS:
            return "answer_node"
        if isinstance(last, ToolMessage):
            return "answer_node"
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool_exec"
        return "answer_node"

    # ── Graph assembly ────────────────────────────────────────────────────
    g = StateGraph(State)
    g.add_node("router_node",   router_node)
    g.add_node("llm_tool_node", _llm_tool_node)
    g.add_node("tool_exec",     tool_exec)
    g.add_node("answer_node",   answer_node)

    g.add_edge(START, "router_node")
    g.add_conditional_edges(
        "router_node", route_condition,
        {"llm_tool_node": "llm_tool_node", "answer_node": "answer_node"}
    )
    g.add_conditional_edges(
        "llm_tool_node", tool_condition,
        {"tool_exec": "tool_exec", "answer_node": "answer_node"}
    )
    g.add_edge("tool_exec",   "answer_node")
    g.add_edge("answer_node", END)

    return g, all_tools

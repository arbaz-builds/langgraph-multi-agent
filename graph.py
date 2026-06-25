"""Graph assembly."""
import asyncio
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from state import State
from nodes import router_node, llm_tool_node, answer_node
from tools import RAG, search_web, mcp_client
import config


async def build_graph():
    mcp_tools = await asyncio.wait_for(mcp_client.get_tools(), timeout=30)
    tools     = [RAG, search_web, *mcp_tools]

    g = StateGraph(State)
    g.add_node("router",   router_node)
    g.add_node("llm_tool", lambda s: llm_tool_node(s, tools))
    g.add_node("tools",    ToolNode(tools))
    g.add_node("answer",   answer_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router",   lambda s: "llm_tool" if s["router_decision"] != "direct" else "answer", ["llm_tool", "answer"])
    g.add_conditional_edges("llm_tool", lambda s: "tools" if isinstance(s["messages"][-1], AIMessage) and s["messages"][-1].tool_calls and s.get("iteration_count", 0) < config.MAX_ITERATIONS else "answer", ["tools", "answer"])
    g.add_edge("tools",  "answer")
    g.add_edge("answer", END)

    return g, tools

"""Graph assembly."""
from langgraph.graph import StateGraph, START, END
from state import State
from nodes import router_node, llm_tool_node, answer_node
from nodes.router import route_condition
from nodes.llm_tool import get_tool_node
from nodes.conditions import tool_or_answer, after_tools


async def build_graph():
    tool_node = await get_tool_node()

    g = StateGraph(State)
    g.add_node("router",   router_node)
    g.add_node("llm_tool", llm_tool_node)
    g.add_node("tools",    tool_node)
    g.add_node("answer",   answer_node)

    g.add_edge(START,    "router")
    g.add_conditional_edges("router",   route_condition, ["llm_tool", "answer"])
    g.add_conditional_edges("llm_tool", tool_or_answer,  ["tools",    "answer"])
    g.add_conditional_edges("tools",    after_tools,     ["llm_tool", "answer"])
    g.add_edge("answer", END)

    return g, tool_node

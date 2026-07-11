"""Graph assembly."""
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from state import State
from nodes import router_node, llm_tool_node, answer_node
from nodes.router import route_condition
from nodes.llm_tool import _get_tools
import config


def _tool_or_answer(s):
    last = s["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and s.get("iteration_count", 0) < config.MAX_ITERATIONS:
        return "tools"
    return "answer"


async def build_graph():
    tools = await _get_tools()

    g = StateGraph(State)
    g.add_node("router",   router_node)
    g.add_node("llm_tool", llm_tool_node)
    g.add_node("tools",    ToolNode(tools))
    g.add_node("answer",   answer_node)

    g.add_edge(START,    "router")
    g.add_conditional_edges("router",   route_condition, ["llm_tool", "answer"])
    g.add_conditional_edges("llm_tool", _tool_or_answer, ["tools",    "answer"])
    g.add_edge("tools",  "answer")
    g.add_edge("answer", END)

    return g, tools

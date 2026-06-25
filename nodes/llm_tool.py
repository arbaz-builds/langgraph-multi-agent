"""LLM tool node — binds tools and forces a tool call."""
from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM


async def llm_tool_node(state: State, tools: list):
    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [], "iteration_count": state.get("iteration_count", 0)}

    resp = await answer_LLM.bind_tools(tools).ainvoke(
        [SystemMessage(content="You are an AI assistant. Always call a tool. Never reply with text.")] +
        state["messages"][-6:]
    )
    return {"messages": [resp], "iteration_count": state.get("iteration_count", 0) + 1}

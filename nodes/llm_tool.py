"""LLM tool node — forces the correct tool call based on router decision."""
from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM

_TOOL_MAP = {"python_tool": "run_python", "rag": "RAG", "web_search": "search_web"}


async def llm_tool_node(state: State, tools: list):
    dec = state.get("router_decision")
    forced = _TOOL_MAP.get(dec)

    if isinstance(state["messages"][-1], ToolMessage):
        return {"messages": [], "iteration_count": state.get("iteration_count", 0)}

    resp = await answer_LLM.bind_tools(tools, tool_choice=forced).ainvoke(
        [SystemMessage(content=f"Call the tool: {forced}. Do NOT reply with text.")] +
        state["messages"][-6:]
    )
    return {"messages": [resp], "iteration_count": state.get("iteration_count", 0) + 1}

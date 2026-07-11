"""Tool-routing conditions used by graph.py."""
from langchain_core.messages import AIMessage
from state import State
import config


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

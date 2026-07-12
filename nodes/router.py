"""Router node — decides whether the query needs a tool or a direct answer."""
from langchain_core.messages import SystemMessage, HumanMessage
from state import State, RouterDecision
from llms import router_LLM

_PROMPT = """Decide if this query needs a tool, or can be answered directly.

- tools  : needs code execution, document/file lookup, or real-time web info
           (e.g. "run this code", "check my uploaded file", "latest news", "current price")
- answer : greetings, general knowledge, simple questions — no tool needed

When in doubt → tools."""


async def router_node(state: State):
    try:
        r = await router_LLM.with_structured_output(RouterDecision).ainvoke([
            SystemMessage(content=_PROMPT + f"\nfile_uploaded={state.get('file_uploaded', False)}"),
            HumanMessage(content=state["messages"][-1].content)
        ])
        return {"router_decision": r.decision, "reasoning": r.reasoning}
    except Exception as e:
        return {"router_decision": "answer", "reasoning": str(e)}


def route_condition(state: State) -> str:
    return state["router_decision"]

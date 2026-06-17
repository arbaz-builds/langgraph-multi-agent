from langchain_core.messages import SystemMessage, HumanMessage
from state import State, RouterDecision
from llms import router_LLM

ROUTER_PROMPT = """You are a routing assistant. Classify the query into one of:
- python_tool : code, python, packages, install, pip, version, environment, libraries, "karo", "check karo", "chalao"
- rag         : user asking about uploaded file content — "document", "file", "PDF", "report", "uploaded", "mera data"
- web_search  : news, weather, latest/newest/current/recent/today, real-time info, price, stock, "kab release hoga"
- direct      : general knowledge, definitions, greetings, simple questions (no code, no file, no real-time)

Rules:
- Any technical/code intent                          → python_tool
- "latest", "newest", "current", "kab release hoga" → web_search
- ANY mention of file/document/PDF                   → rag
- Doubt between python_tool vs direct                → python_tool
"""


async def router_node(state: State):
    try:
        r = await router_LLM.with_structured_output(RouterDecision).ainvoke([
            SystemMessage(
                content=ROUTER_PROMPT + f"\nfile_uploaded={state.get('file_uploaded', False)}"
            ),
            HumanMessage(content=state["messages"][-1].content)
        ])
        return {"router_decision": r.decision, "reasoning": r.reasoning}
    except Exception as e:
        return {"router_decision": "direct", "reasoning": str(e)}

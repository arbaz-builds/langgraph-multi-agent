"""Entry point — CLI runner + FastAPI web server for deployment."""
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from graph import build_graph
import config

config.validate()

app = FastAPI(
    title="LangGraph Multi-Agent API",
    description="Multi-agent assistant routing between RAG retrieval, real-time web search, "
                 "sandboxed Python execution (MCP), and direct LLM response.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str
    thread_id: str = "1"


class QueryResponse(BaseModel):
    response: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse, summary="Chat with the multi-agent assistant")
async def chat(payload: QueryRequest):
    # Fresh TCP connection to Postgres per request, so we never reuse a
    # stale/dead connection left over from a provider-side idle timeout
    # (e.g. Neon auto-suspend, Render free-tier sleep).
    g = await build_graph()
    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
        await cp.setup()
        result = await g.compile(checkpointer=cp).ainvoke(
            {"messages": [HumanMessage(content=payload.query)], "iteration_count": 0},
            config={"configurable": {"thread_id": payload.thread_id}},
        )
    return QueryResponse(response=result["messages"][-1].content)


async def build_and_run(query: str, thread_id: str = "1") -> str:
    """CLI / test helper — kept for local/manual testing."""
    g = await build_graph()
    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
        await cp.setup()
        return (await g.compile(checkpointer=cp).ainvoke(
            {"messages": [HumanMessage(content=query)], "iteration_count": 0},
            config={"configurable": {"thread_id": thread_id}}
        ))["messages"][-1].content


if __name__ == "__main__":
    print(asyncio.run(build_and_run("python version check karo")))

"""Entry point — run the multi-agent chatbot."""
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from graph import build_graph
import config

config.validate()


async def run(query: str, thread_id: str = "1") -> str:
    g, _ = await build_graph()
    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
        await cp.setup()
        res = await g.compile(checkpointer=cp).ainvoke(
            {"messages": [HumanMessage(content=query)], "file_uploaded": False, "iteration_count": 0},
            config={"configurable": {"thread_id": thread_id}}
        )
    return res["messages"][-1].content


if __name__ == "__main__":
    print(asyncio.run(run("python version check karo")))

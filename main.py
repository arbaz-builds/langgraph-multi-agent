"""Entry point — CLI runner + FastAPI web server for deployment."""
import asyncio
import base64
import binascii
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from openai import APIError, APIConnectionError, APITimeoutError

from graph import build_graph
from llms import vision_LLM
import config

config.validate()

logger = logging.getLogger("langgraph_multi_agent")

# Guard against oversized payloads blowing up context window / cost.
# ~5MB base64 ≈ ~3.75MB actual image — generous for a chat photo, small
# enough to not tank latency or token cost.
MAX_IMAGE_BASE64_CHARS = 5_000_000

app = FastAPI(
    title="LangGraph Multi-Agent API",
    description="Multi-agent assistant routing between RAG retrieval, real-time web search, "
                 "sandboxed Python execution (MCP), and direct LLM response.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str
    thread_id: str = "1"
    image_base64: str | None = None  # e.g. "data:image/jpeg;base64,/9j/4AAQ..."

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query cannot be empty")
        return v

    @field_validator("image_base64")
    @classmethod
    def image_base64_valid(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) > MAX_IMAGE_BASE64_CHARS:
            raise ValueError(
                f"image_base64 too large ({len(v)} chars, max {MAX_IMAGE_BASE64_CHARS})"
            )
        # Accept either a raw base64 string or a data URI; validate the
        # actual encoded payload decodes cleanly so we fail fast here
        # instead of surfacing a confusing error from the vision API later.
        payload = v.split(",", 1)[1] if v.startswith("data:") else v
        if not payload:
            raise ValueError("image_base64 payload is empty")
        try:
            base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"image_base64 is not valid base64: {e}")
        return v


class QueryResponse(BaseModel):
    response: str


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _invoke(query: str, thread_id: str = "1") -> str:
    """Shared logic: build a fresh graph + a fresh Postgres connection per
    call, run one turn, and return the assistant's reply text.

    A new connection is opened each time (instead of one long-lived
    connection) so we never reuse a stale/dead TCP connection left over
    from a provider-side idle timeout (e.g. Neon auto-suspend, Render
    free-tier sleep).
    """
    try:
        g = await build_graph()
        async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
            await cp.setup()
            result = await g.compile(checkpointer=cp).ainvoke(
                {"messages": [HumanMessage(content=query)], "iteration_count": 0},
                config={"configurable": {"thread_id": thread_id}},
            )
        return result["messages"][-1].content
    except (APIConnectionError, APITimeoutError) as e:
        logger.error("LLM provider unreachable in _invoke: %s", e)
        raise HTTPException(status_code=503, detail="Model provider is unavailable right now. Try again shortly.")
    except APIError as e:
        logger.error("LLM API error in _invoke: %s", e)
        raise HTTPException(status_code=502, detail="Model provider returned an error.")
    except OSError as e:
        # Covers Postgres connection failures (asyncpg raises OSError subclasses)
        logger.error("Database error in _invoke: %s", e)
        raise HTTPException(status_code=503, detail="Could not reach conversation memory store.")


async def _invoke_vision(query: str, image_base64: str, thread_id: str = "1") -> str:
    """Image queries use the same Postgres-backed thread history as text
    queries, but call vision_LLM directly instead of routing through the
    graph — routing/RAG/tools don't apply when the input is an image."""
    try:
        async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
            await cp.setup()
            checkpoint_tuple = await cp.aget_tuple(
                config={"configurable": {"thread_id": thread_id}}
            )
            history = checkpoint_tuple.checkpoint["channel_values"].get("messages", []) if checkpoint_tuple else []

            new_message = HumanMessage(content=[
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": image_base64}},
            ])
            result = await vision_LLM.ainvoke(history + [new_message])

            # Persist this turn to the same thread so a follow-up text
            # question can reference what was in the image.
            #
            # as_node must be given explicitly: if omitted, LangGraph tries
            # to infer "the last node that updated state", which is
            # ambiguous on a thread's first-ever write (no prior node to
            # infer from) and the update can silently fail to persist.
            # iteration_count is included too, since State requires it and
            # leaving it unset here caused the next graph .ainvoke() (in
            # _invoke) to treat this checkpoint as incomplete/stale.
            g = await build_graph()
            await g.compile(checkpointer=cp).aupdate_state(
                config={"configurable": {"thread_id": thread_id}},
                values={"messages": [new_message, result], "iteration_count": 0},
                as_node="answer",
            )
        return result.content
    except (APIConnectionError, APITimeoutError) as e:
        logger.error("Vision provider unreachable: %s", e)
        raise HTTPException(status_code=503, detail="Vision model is unavailable right now. Try again shortly.")
    except APIError as e:
        logger.error("Vision API error: %s", e)
        raise HTTPException(status_code=502, detail="Vision model rejected the request — check the image format.")
    except OSError as e:
        logger.error("Database error in _invoke_vision: %s", e)
        raise HTTPException(status_code=503, detail="Could not reach conversation memory store.")


@app.post("/chat", response_model=QueryResponse, summary="Chat with the multi-agent assistant")
async def chat(payload: QueryRequest):
    try:
        if payload.image_base64:
            reply = await _invoke_vision(payload.query, payload.image_base64, payload.thread_id)
        else:
            reply = await _invoke(payload.query, payload.thread_id)
        return QueryResponse(response=reply)
    except HTTPException:
        raise  # already a clean, intentional error — pass it through as-is
    except Exception as e:
        # Last-resort catch: never leak internal stack traces/details to the client.
        logger.exception("Unhandled error in /chat: %s", e)
        raise HTTPException(status_code=500, detail="Something went wrong processing your request.")


async def build_and_run(query: str, thread_id: str = "1") -> str:
    """CLI / test helper — kept for local/manual testing."""
    return await _invoke(query, thread_id)


if __name__ == "__main__":
    print(asyncio.run(build_and_run("python version check karo")))

from typing import TypedDict, Literal, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    decision: Literal["tools", "answer"] = Field(
        description="Routing decision"
    )
    reasoning: str = Field(
        min_length=5, max_length=200,
        description="Brief explanation"
    )


class State(TypedDict):
    messages:        Annotated[List[BaseMessage], add_messages]
    router_decision: str
    reasoning:       str
    iteration_count: int

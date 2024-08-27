from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages, StateGraph, END


class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

graph_builder = StateGraph(GraphState)

graph = graph_builder.compile()

import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from graph.chat.nodes import chat_retrieve_node, chat_generate_node
from graph.chat.state import ChatState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

def route_after_retrieve(state: ChatState):
    if state.get("error"):
        return "end"
    return "chat_generate"

def build_chat_graph(checkpointer):
    graph = StateGraph(ChatState)
    graph.add_node("chat_retrieve", chat_retrieve_node)
    graph.add_node("chat_generate", chat_generate_node)

    graph.set_entry_point("chat_retrieve")
    graph.add_conditional_edges(
        "chat_retrieve",
        route_after_retrieve,
        {
            "chat_generate": "chat_generate",
            "end":END
        }
    )
    graph.add_edge("chat_generate", END)
    return  graph.compile(checkpointer=checkpointer)

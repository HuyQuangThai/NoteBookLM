from langgraph.graph import StateGraph, END

from graph.ingestion.nodes import extract_node, chunk_node, embed_node
from graph.ingestion.state import IngestionState


def is_error(state: IngestionState):
    if state.get("error"):
        return "end"
    return "continue"

def build_ingestion_graph():
    graph = StateGraph(IngestionState)

    graph.add_node("extract", extract_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed", embed_node)

    graph.set_entry_point("extract")
    graph.add_conditional_edges(
        "extract",
        is_error,
        {"continue":"chunk", "end":END},
    )

    graph.add_conditional_edges(
        "chunk",
        is_error,
        {"continue":"embed", "end":END},
    )
    graph.add_edge("embed", END)

    return graph.compile()

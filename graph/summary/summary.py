from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.types import Send

from graph.summary.nodes import fetch_chunks_node, insights_node, tags_node, map_summary_node, reduce_summary_node, \
    generate_batches_node
from graph.summary.state import SummaryState


def route_after_fetch(state: SummaryState):
    if state.get("error"):
        return "end"

    return [
        Send("generate_batches",  {"chunks": state["chunks"]}),
        Send("insights", {"chunks": state["chunks"]}),
        Send("tags", {"chunks": state["chunks"]}),
    ]

def route_after_gen_batches(state: SummaryState):
    if state.get("error"):
        return "end"
    return [Send("map_summary", {"batch": b}) for b in state.get("batches")]

def build_summary_graph():
    graph = StateGraph(SummaryState)
    graph.add_node("fetch_chunks", fetch_chunks_node)
    graph.add_node("generate_batches", generate_batches_node)
    graph.add_node("map_summary", map_summary_node)
    graph.add_node("reduce_summary", reduce_summary_node)
    graph.add_node("insights", insights_node)
    graph.add_node("tags", tags_node)

    graph.set_entry_point("fetch_chunks")
    graph.add_conditional_edges(
        "fetch_chunks",
        route_after_fetch,
        ["generate_batches", "insights", "tags", END],
    )
    graph.add_conditional_edges(
        "generate_batches",
        route_after_gen_batches,
        {"map_summary":"map_summary", "end":END}
    )
    graph.add_edge("map_summary", "reduce_summary")
    graph.add_edge("insights", END)
    graph.add_edge("tags", END)
    graph.add_edge("reduce_summary", END)

    return graph.compile()


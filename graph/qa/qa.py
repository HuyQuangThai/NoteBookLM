from langgraph.constants import END
from langgraph.graph import StateGraph
from graph.qa.state import QAState
from graph.qa.nodes import retrieve_node, grade_node, generate_node


def route_after_grade(state: QAState):
    if state.get("is_relevant"):
        return "generate"
    return "end"

def build_qa_graph():
    graph = StateGraph(QAState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "generate": "generate",
            "end": END
        }
    )
    graph.add_edge("generate", END)


    return graph.compile()





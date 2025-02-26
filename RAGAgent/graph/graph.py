from langgraph.graph import END, StateGraph

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

def decide_to_generate(state: GraphState) -> bool:
    print("---ASSESS GRADED DOCUMENTS---")
    if state['web_search']:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print(
            "---DECISION: ALL DOCUMENTS ARE RELEVANT TO QUESTION, GENERATE RESPONSE---"
        )
        return GENERATE
    

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve.retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents.grade_documents)
workflow.add_node(WEBSEARCH, web_search.web_search)
workflow.add_node(GENERATE, generate.generate)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS, 
    decide_to_generate,
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE
    }
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

def compile_graph():
    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return app



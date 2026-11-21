from dotenv import load_dotenv

from langgraph.graph import END, StateGraph
from graph.chains.answer_grader import answer_grader
# answer_grader.invoke()
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEB_SEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEB_SEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE
    else:
        print("---COULD NOT IDENTIFY THE ROUTE DECISION, DEFAULTING TO WEB SEARCH---")
        return WEB_SEARCH


def decide_to_generate(state: GraphState):
    print("---ASSESS GRADE DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEB_SEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_score.binary_score.strip().lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        answer_score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )

        if answer_score.binary_score.strip().lower() == "yes":
            print("---DECISION: GENERATION ANSWERS QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ANSWER QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRY---")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEB_SEARCH: WEB_SEARCH,
        RETRIEVE: RETRIEVE
    },
)
workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEB_SEARCH,
        "not supported": GENERATE
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
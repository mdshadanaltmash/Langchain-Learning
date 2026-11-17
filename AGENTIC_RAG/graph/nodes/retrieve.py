from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve relevant documents based on the question in the state.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        Dict[str, Any]: A dictionary containing the retrieved documents.
    """
    print("Retrieving relevant documents...")
    question = state["question"]
    retrieved_documents = retriever.invoke(question)
    return {"documents": retrieved_documents, "question": question}
from pprint import pprint

from dotenv import load_dotenv

from graph.chains.router import RouteQuery, question_router

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import GradeHallucination, hallucination_grader
from ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == 'Yes'


def test_retrieval_grader_answer_no() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "How to make a cake?", "document": doc_txt}
    )

    assert res.binary_score == 'No'


def test_generation_chain() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res = generation_chain.invoke(
        {"question": question, "context": doc_txt}
    )

    pprint(res)


def test_hallucination_grader_answer_yes() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert res.binary_score == 'Yes'


def test_hallucination_grader_answer_no() -> None:
    question = 'agent memory'
    docs = retriever.invoke(question)

    res: GradeHallucination = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "Shadan will be at a good place soon. Inshallah!"
        }
    )

    assert res.binary_score == 'No'


def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "What is the capital of France?"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"

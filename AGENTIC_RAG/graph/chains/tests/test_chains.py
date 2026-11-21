from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
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

from typing import Any, Dict
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_results = "\n\n".join(
        [result["content"] for result in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_results)
    if documents is not None and len(documents) > 0:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, question: question }

if __name__ == "__main__":
    web_search(state={"question": "Agent Memory", "documents": None})

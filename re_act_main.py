from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureOpenAI, ChatOpenAI, AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI
import os
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()


# tavily = TavilyClient()


# @tool
# def search(query: str) -> dict:
#     """
#     Tool that searches over the internet
#     :param query: The query to search for
#     :return: The search result
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query)


class Source(BaseModel):
    """Scheme for a source used by the agent"""

    url: str = Field(description="URL of the source")


class AgentResponse(BaseModel):
    """Scheme for agent response with the answer and sources"""
    answer: str = Field(description="The agent answer to the query")
    sources: List[Source] = Field(description="List of sources used to generate the answer")


llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
print(llm.invoke("Hello Shadan"))
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain course")
    msg = {"messages": [HumanMessage(content="What is the weather in Delhi, India?")]}
    result = agent.invoke(input=msg)
    print(result)


if __name__ == "__main__":
    main()

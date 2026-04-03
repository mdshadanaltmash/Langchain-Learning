from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureOpenAI, ChatOpenAI, AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAI
import os
load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    :param query: The query to search for
    :return: The search result
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
print(llm.invoke("Hello Shadan"))
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain course")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokoyo?")]})
    print(result)


if __name__ == "__main__":
    main()
    print('hi')

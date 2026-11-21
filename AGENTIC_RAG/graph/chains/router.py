import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI

load_dotenv()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to either a vectorstore or websearch. "
    )


llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0
)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or a websearch.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For everything else, use the websearch."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

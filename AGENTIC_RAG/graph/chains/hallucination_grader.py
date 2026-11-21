import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import AzureChatOpenAI


llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0
)


class GradeHallucination(BaseModel):
    """Binary score for hallucination of generated answer to a query."""

    binary_score: str = Field(
        description="Answer is hallucinated, 'Yes' or 'No'."
    )


structured_llm_grader = llm.with_structured_output(GradeHallucination)
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'Yes' or 'No'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

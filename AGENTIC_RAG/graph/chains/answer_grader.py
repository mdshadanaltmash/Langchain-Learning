import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import AzureChatOpenAI

load_dotenv()
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0
)


class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question, 'Yes' or 'No'."
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'Yes' or 'No'. 'Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader:  RunnableSequence = answer_prompt | structured_llm_grader

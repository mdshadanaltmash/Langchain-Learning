from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)

generation_chain = prompt | llm | StrOutputParser()

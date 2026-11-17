import os

from  dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [UnstructuredLoader(web_url=url, chunking_strategy='basic', max_characters=1000000).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name='rag-chroma',
#     embedding=embeddings,
#     persist_directory="./.chroma"
# )

retriever = Chroma(
    collection_name='rag-chroma',
    persist_directory="./.chroma",
    embedding_function=embeddings,
).as_retriever()

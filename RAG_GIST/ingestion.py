from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
print("Endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("Deployment:", repr(os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")))
print("API version:", repr(os.getenv("AZURE_OPENAI_API_VERSION")))
print("API key loaded:", bool(os.getenv("AZURE_OPENAI_API_KEY")))

if __name__ == '__main__':
    print("Ingesting...")
    file_path = Path(__file__).resolve().with_name("mediumblog1.txt")
    text = file_path.read_text(encoding="utf-8")
    document = [Document(page_content=text, metadata={"source": str(file_path)})]
    print("Splitting....")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    # embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"))
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    print("Ingesting...")
    # obj = PineconeVectorStore(embedding=embeddings)
    # obj.delete(delete_all=True, index_name=os.environ["PINECONE_INDEX_NAME"])
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"])
    print('Finished ingesting')

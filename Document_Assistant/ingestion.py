import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch, TavilyExtract, TavilyMap, TavilyCrawl

from logger import (Colors, log_error, log_header, log_info, log_warning, log_success)

load_dotenv()

# Configure SSL context to use certifi's CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)

# chroma = Chroma(pesistent_directory="chroma_db", embedding_function=embeddings)
vector_store = PineconeVectorStore(
    index_name=os.environ["PINECONE_INDEX_NAME"],
    embedding=embeddings,
)
# vector_store.delete(delete_all=True, index_name=os.environ["PINECONE_INDEX_NAME"])
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=5, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 500):
    """Process documents asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_number: int):
        try:
            await vector_store.aadd_documents(batch)
            log_success(
                f"✅ VectorStore Indexing: Successfully added batch {batch_number}/{len(batches)} with {len(batch)} documents"
            )
        except Exception as e:
            log_error(
                f"❌ VectorStore Indexing: Failed to add batch {batch_number}/{len(batches)}. Error: {str(e)}"
            )
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count Successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"🎉 VectorStore Indexing: Successfully added all {len(documents)} documents to vector store"
        )
    else:
        log_warning(
            f"⚠️ VectorStore Indexing: Only {successful}/{len(batches)} batches were successfully added to vector store"
        )


async def main():
    """Main function to run the asynchronous tasks."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🔍 TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )

    # Crawl the documentation site
    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://python.langchain.com/",
            "max_depth": 5,
            # "instructions": "Content on AI Agents",
            "extract_depth": "advanced"
        }
    )
    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return
    else:
        log_success(
            f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results)} URLs from documentation site"
        )
    all_docs = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        if tavily_crawl_result_item["raw_content"] is not None:
            log_info(f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site)")
            all_docs.append(
                Document(
                    page_content=tavily_crawl_result_item["raw_content"],
                    metadata={"source": tavily_crawl_result_item["url"]}
                )
            )

    # Split the documents into smaller chunks
    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously and ingest into vector store
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(tavily_crawl_results)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())

"""RAG tool — hybrid retrieval from Pinecone vector store."""
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from llms import embeddings
import config


@tool
def RAG(query: str) -> str:
    """Retrieve relevant content from previously uploaded documents."""
    try:
        docs = PineconeVectorStore(
            index_name=config.PINECONE_INDEX,
            embedding=embeddings,
            pinecone_api_key=config.PINECONE_API_KEY
        ).as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.RETRIEVER_K, "fetch_k": config.RETRIEVER_FETCH_K, "lambda_mult": 0.7}
        ).invoke(query)

        return "".join(f"Source {i+1}:\n{d.page_content.strip()}\n\n" for i, d in enumerate(docs)) \
               or "No relevant content found."
    except Exception as e:
        return f"RAG error: {e}"

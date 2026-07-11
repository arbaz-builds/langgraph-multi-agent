"""LLM and embedding instances — imported across all nodes."""
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import config

_base = dict(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY, model=config.NVIDIA_MODEL)

answer_LLM = ChatOpenAI(**_base)
router_LLM = ChatOpenAI(**_base, temperature=0)


class MockEmbeddings(Embeddings):
    """Placeholder embeddings — no external API required.
    Returns fixed-size zero vectors so RAG runs without OpenAI credentials.
    Dimension (1536) matches OpenAI's text-embedding-3-small for Pinecone index compatibility.
    Replace with a real embedding model when ready for production RAG quality.
    """
    _DIM = 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self._DIM


embeddings = MockEmbeddings()

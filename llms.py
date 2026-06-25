"""LLM and embedding instances — imported across all nodes."""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config

_base = dict(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY, model=config.NVIDIA_MODEL)

answer_LLM = ChatOpenAI(**_base)
router_LLM = ChatOpenAI(**_base, temperature=0)
embeddings  = OpenAIEmbeddings(model="text-embedding-3-small", api_key=config.OPENAI_API_KEY)

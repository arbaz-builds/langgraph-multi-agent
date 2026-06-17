from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config

answer_LLM = ChatOpenAI(
    base_url=config.NVIDIA_BASE_URL,
    api_key=config.NVIDIA_API_KEY,
    model=config.NVIDIA_MODEL
)

router_LLM = ChatOpenAI(
    base_url=config.NVIDIA_BASE_URL,
    api_key=config.NVIDIA_API_KEY,
    model=config.NVIDIA_MODEL,
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=config.OPENAI_API_KEY
)

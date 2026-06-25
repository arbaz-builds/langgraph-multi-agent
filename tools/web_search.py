"""Web search tool using Tavily."""
from langchain_tavily import TavilySearch
import config

search_web = TavilySearch(
    max_results=5,
    tavily_api_key=config.TAVILY_API_KEY
)

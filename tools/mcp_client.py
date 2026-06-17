from langchain_mcp_adapters.client import MultiServerMCPClient
import config

mcp_client = MultiServerMCPClient({
    "python_tool": {
        "transport": "streamable-http",
        "url": config.MCP_SERVER_URL
    }
})

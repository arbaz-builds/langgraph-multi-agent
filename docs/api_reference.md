# 📚 API Reference

## Main Function

### `build_and_run(query, thread_id)`

```python
async def build_and_run(query: str, thread_id: str = "1") -> str:
    """
    Main entry point for the chatbot.

    Args:
        query (str): User's input query
        thread_id (str): Conversation ID for memory (default: "1")

    Returns:
        str: AI assistant's response

    Example:
        result = await build_and_run("python version check karo", "user_123")
    """
```

## Tools

### `RAG(query)`
Retrieves information from uploaded documents.
```python
@tool
def RAG(query: str) -> str:
    """Use for document-based questions."""
```

### `web_search(query)`
Searches Google for real-time information.
```python
@tool
def web_search(query: str) -> str:
    """Use for current news, prices, latest info."""
```

### `run_python(code)` (via MCP)
Executes Python code remotely.
```python
# Automatically called by LLM for python_tool queries
# Code must use print() to show output
```

## Document Loading

### `load_document(file_path)`
```python
def load_document(file_path: str) -> str:
    """
    Loads and indexes a document into Pinecone.

    Supported formats: .pdf, .csv, .txt, .docx

    Args:
        file_path: Path to document

    Returns:
        Success/error message
    """
```

## Router Categories

| Decision | Triggered By |
|----------|-------------|
| `python_tool` | code, python, install, pip, version, "karo" |
| `web_search` | latest, news, today, current, price |
| `rag` | file uploaded + document question |
| `direct` | general knowledge, greetings |

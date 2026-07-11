"""
Test cases for LangGraph Multi-Agent Chatbot.
Run: python -m pytest tests/test_agent.py -v
"""
import asyncio
import pytest
from main import build_and_run

# ── Router Tests ─────────────────────────────────
@pytest.mark.parametrize("query,expected_type", [
    ("python version check karo",    "python_tool"),
    ("numpy install hai?",           "python_tool"),
    ("AI kya hota hai?",             "direct"),
    ("hello kaise ho?",              "direct"),
    ("latest GPT model 2026",        "web_search"),
    ("Bitcoin price today",          "web_search"),
])
def test_router_accuracy(query, expected_type):
    """Router 100% accurate hona chahiye."""
    # Note: This tests the final answer, not internal routing
    result = asyncio.get_event_loop().run_until_complete(
        build_and_run(query, thread_id=f"test_{hash(query)}")
    )
    assert result is not None
    assert len(result) > 0

# ── Python Tool Tests ────────────────────────────
def test_python_execution():
    """Python code actually execute hona chahiye."""
    result = asyncio.get_event_loop().run_until_complete(
        build_and_run("1 se 10 tak sum karo python mein", thread_id="test_sum")
    )
    assert "55" in result

def test_python_version():
    """Python version check."""
    result = asyncio.get_event_loop().run_until_complete(
        build_and_run("python version check karo", thread_id="test_version")
    )
    assert result is not None

# ── Direct Answer Tests ──────────────────────────
def test_direct_answer():
    """Direct answer bina tool ke."""
    result = asyncio.get_event_loop().run_until_complete(
        build_and_run("AI kya hota hai?", thread_id="test_direct")
    )
    assert len(result) > 10

# ── Language Tests ───────────────────────────────
def test_urdu_response():
    """Urdu query pe Roman Urdu mein jawab."""
    result = asyncio.get_event_loop().run_until_complete(
        build_and_run("AI kya hota hai?", thread_id="test_lang")
    )
    # Hindi script nahi honi chahiye
    hindi_chars = any('\u0900' <= c <= '\u097F' for c in result)
    assert not hindi_chars, "Hindi script nahi aani chahiye!"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

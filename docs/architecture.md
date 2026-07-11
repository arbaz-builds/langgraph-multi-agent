# 🏗️ Architecture Deep Dive

## Graph Flow

```
START
  │
  ▼
router_node
  │ route_condition()
  ├──── rag / web_search / python_tool ──→ LLM_Tool
  │                                            │
  │                                      tool_condition()
  │                                            ├── tool_calls? ──→ tools ──→ (loop while under MAX_ITERATIONS) ──→ LLM_Tool
  │                                            │                      │
  │                                            └── no tool ──→ answer_node ←──┘ (once limit hit)
  │
  └──── direct ──────────────────────────────────────→ answer_node
                                                              │
                                                             END
```

## Nodes Explained

### 1. `router_node`
- Input: User query
- Process: LLM classifies query into 4 categories
- Output: `router_decision` (python_tool/web_search/rag/direct)
- Accuracy: **100%** (tested on 10+ queries)

### 2. `LLM_Tool`
- Input: Messages + router_decision
- Process: Forces LLM to call specific tool
- Output: AIMessage with tool_calls
- Key fix: `tool_choice=forced` — no text answers

### 3. `tools`
- Input: Tool call from LLM_Tool
- Process: Executes actual tool (run_python/web_search/RAG)
- Output: ToolMessage with result
- Routes back to `LLM_Tool` while `iteration_count < MAX_ITERATIONS`, enabling multi-step tool chaining; falls through to `answer_node` once the limit is reached.

### 4. `answer_node`
- Input: Tool output + user query
- Process: Formats clean answer
- Output: Final response in user's language
- Key fix: No history contamination

## Bugs Fixed

### Bug 1: Tool Not Executing
**Problem:** `tools → LLM_Tool` loop caused tool to never execute
**Fix:** `tools → answer_node` directly

### Bug 2: Hallucination
**Problem:** Old ToolMessage in history caused wrong answers
**Fix:** `answer_node` only uses `user_q + tool_out`

### Bug 3: Wrong Language
**Problem:** LLM replied in Hindi script
**Fix:** Explicit "Roman Urdu only" rule in answer_node prompt

### Bug 4: MAX_ITERATIONS was dead code
**Problem:** `tools` always routed straight to `answer`, so the
`iteration_count < MAX_ITERATIONS` check in `_tool_or_answer` never
had a chance to matter — the agent could never chain more than one
tool call per query.
**Fix:** Added a `_after_tools` conditional edge so `tools` loops
back to `llm_tool` while under the iteration limit, and only falls
through to `answer` once it's reached.

### Bug 5: Broken test import
**Problem:** `tests/test_agent.py` imported `build_and_run` from
`langgraph_agent.py`, a deprecated stub with no such function —
all tests failed to even collect.
**Fix:** Tests now import `build_and_run` from `main.py`, where the
CLI helper (previously named `run`) actually lives.

## Memory System

```
Thread ID → PostgreSQL (Neon)
    │
    ├── Conversation 1 (thread_id="1")
    ├── Conversation 2 (thread_id="2")
    └── Conversation N (thread_id="N")
```

Each thread maintains full conversation history via `AsyncPostgresSaver`.

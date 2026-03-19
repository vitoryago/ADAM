# Phase 2: Conversational Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ADAM a genuinely useful conversational assistant — real LLM calls with streaming, working memory in the chat pipeline, proper multi-turn context, and a connected frontend.

**Architecture:** The infrastructure exists from Phase 1. This phase WIRES it together: messages.py → LLMService → UnifiedLLMClient (real streaming) + Memory (recall before, store after) + conversation context (sliding window). Frontend connects via SSE for streaming.

**Tech Stack:** Python 3.9+, FastAPI, SSE (sse-starlette), ChromaDB, React, TypeScript

**Spec:** `docs/superpowers/specs/2026-03-18-adam-roadmap-design.md` (Phase 2 section)

---

## Dependency Graph

```
Task 1 (Deps) → Task 2 (Streaming) → Task 3 (Memory) → Task 4 (Context) → Task 5+6 (parallel) → Task 7 (Verify)
```

---

### Task 1: Dependencies & LLM Verification

**Goal:** Install all dependencies, configure API keys, verify the LLM client can make real calls.

**Files:**
- Modify: `requirements.txt` (verify completeness)
- Modify: `.env` (add API keys)
- Create: `tests/test_llm_live.py`

- [ ] **Step 1: Install dependencies**
```bash
cd /Users/vitoryago/ADAM && pip install -r requirements.txt
```

- [ ] **Step 2: Verify .env has at least one API key**
Check .env file for XAI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY. At minimum one must be set.

- [ ] **Step 3: Test LLM client initialization**
```bash
cd /Users/vitoryago/ADAM && python -c "
from adam.llm.client import UnifiedLLMClient
client = UnifiedLLMClient()
print('Client initialized with models:', list(client.models.keys()) if hasattr(client, 'models') else 'unknown')
"
```

- [ ] **Step 4: Test a real API call**
```bash
cd /Users/vitoryago/ADAM && python -c "
import asyncio
from adam.llm.client import UnifiedLLMClient
async def test():
    client = UnifiedLLMClient()
    response = await client.complete('Say hello in one word.', max_tokens=10)
    print(f'Model: {response.model}, Response: {response.content}')
asyncio.run(test())
"
```

- [ ] **Step 5: Fix any import or initialization errors**
If the client fails, debug and fix. Common issues: missing API keys, wrong package versions, import path mismatches.

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "chore: verify dependencies and LLM client initialization"
```

---

### Task 2: Enable Real Streaming

**Goal:** Make LLMService.stream_response() actually stream from the LLM provider, and verify SSE works end-to-end.

**Files:**
- Modify: `src/adam/services/llm_service.py` (enable streaming in stream_response)
- Modify: `src/adam/llm/client.py` (verify streaming interface)
- Create: `tests/test_streaming.py`

- [ ] **Step 1: Read current streaming implementation**
Read src/adam/services/llm_service.py, specifically the stream_response() method. Find where `stream: False` is set and understand the flow.

- [ ] **Step 2: Enable real streaming in LLMService**
In llm_service.py's stream_response():
- Change `"stream": False` to `"stream": True`
- Replace the single-chunk response pattern with actual async iteration over the LLM's streaming response
- Each chunk should yield a StreamChunk with incremental content
- The final chunk should have is_final=True with token count and cost

Key pattern:
```python
# Instead of getting full response and yielding once:
response = await self.llm_client.complete(**kwargs)
# Use streaming:
async for chunk in await self.llm_client.complete(**kwargs, stream=True):
    yield StreamChunk(content=chunk, model_used=final_model)
```

- [ ] **Step 3: Verify SSE endpoint works**
Start the server and test streaming:
```bash
curl -N http://localhost:8000/api/conversations/{conv_id}/messages/stream \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, tell me a joke"}'
```
Verify chunks arrive incrementally (not all at once).

- [ ] **Step 4: Write streaming test**
Create tests/test_streaming.py:
```python
def test_stream_chunks_are_generated():
    """Verify stream_response yields multiple chunks."""
    # Test that the streaming interface produces StreamChunk objects
    from adam.services.llm_service import LLMService, StreamChunk
    assert StreamChunk is not None
    # More detailed tests require running server
```

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: enable real LLM streaming in LLMService"
```

---

### Task 3: Wire Memory into Chat Pipeline

**Goal:** Make memory retrieval and storage part of every conversation. Before calling the LLM, search memory for relevant context. After getting a response, store worthy results.

**Files:**
- Modify: `src/adam/api/routers/messages.py` (wire memory into send_message flow)
- Modify: `src/adam/services/llm_service.py` (accept and use memory context)
- Create: `tests/test_memory_integration.py`

- [ ] **Step 1: Read current message flow**
Read src/adam/api/routers/messages.py, specifically the send_message and send_message_stream functions. Understand how memory is currently used (optional, disconnected).

- [ ] **Step 2: Wire memory recall into message flow**
In messages.py send_message():
1. After loading conversation history, initialize memory if project has it enabled
2. Call memory.recall_with_context(user_message) to find relevant past interactions
3. Format the memory results into a context string: "Relevant past context:\n- {memory1}\n- {memory2}"
4. Pass this context to LLMService.generate_response(memory_context=context_string)

Do the same for send_message_stream().

Key code pattern:
```python
# Before calling LLM
memory_context = ""
if use_memory and memory_service:
    memories = memory_service.recall_with_context(message.content, n_results=3)
    if memories:
        memory_context = "Relevant context from previous conversations:\n"
        for mem in memories:
            memory_context += f"- {mem['content'][:200]}\n"

# Call LLM with memory context
response = await llm_service.generate_response(
    message=message.content,
    history=history,
    memory_context=memory_context,
    ...
)
```

- [ ] **Step 3: Wire memory storage after response**
After getting the LLM response, store worthy interactions:
```python
# After getting response
if memory_service:
    memory_service.remember_if_worthy(
        query=message.content,
        response=response.content,
        generation_cost=response.cost,
        model_used=response.model_used
    )
```

- [ ] **Step 4: Add memory indicator to response**
Add a `memory_used` field to the message response so the frontend can show when memory was leveraged. Add this to the MessageResponse Pydantic model or include it in metadata.

- [ ] **Step 5: Write memory integration test**
Create tests/test_memory_integration.py:
```python
def test_memory_context_formatting():
    """Test that memory results are formatted into context string."""
    # Mock memory results and verify formatting

def test_memory_worthiness_check():
    """Test that only worthy responses are stored."""
    from adam.memory.core import MemoryWorthinessEvaluator, QueryComplexity
    evaluator = MemoryWorthinessEvaluator()
    # Simple question should not be stored
    should_store, reason = evaluator.should_store_memory(
        "What is 2+2?", "4", 0.001, QueryComplexity.TRIVIAL
    )
    assert not should_store
    # Complex question should be stored
    should_store, reason = evaluator.should_store_memory(
        "Design a microservices architecture",
        "Here's a comprehensive design...[500+ chars]",
        0.05, QueryComplexity.EXPERT
    )
    assert should_store
```

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: wire memory recall and storage into chat pipeline"
```

---

### Task 4: Conversation Context & Session Continuity

**Goal:** Implement proper multi-turn conversation context with a sliding window, so ADAM maintains coherent conversations.

**Files:**
- Modify: `src/adam/services/llm_service.py` (improve context management in generate_response and stream_response)
- Modify: `src/adam/api/routers/messages.py` (load full conversation context)
- Create: `tests/test_conversation_context.py`

- [ ] **Step 1: Read current context handling**
Read the history handling in llm_service.py — both generate_response() and stream_response(). Understand the current truncation approach.

- [ ] **Step 2: Implement sliding window context**
In llm_service.py, create a method `_build_conversation_context(history, max_recent=10, max_summary=20)`:
- Last `max_recent` messages: include in full
- Messages `max_recent+1` to `max_recent+max_summary`: summarize each to ~150 chars
- Older messages: drop
- Return formatted message list for the LLM

```python
def _build_conversation_context(self, history: List[Any], max_recent: int = 10, max_summary: int = 20) -> List[Dict]:
    """Build conversation context with sliding window."""
    messages = []
    total = len(history)

    # Summarize older messages (11-30)
    if total > max_recent:
        older = history[-(max_recent + max_summary):-max_recent] if total > max_recent + max_summary else history[:-max_recent]
        if older:
            summary_lines = []
            for msg in older:
                preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                summary_lines.append(f"{msg.role.upper()}: {preview}")
            messages.append({
                "role": "system",
                "content": "Previous conversation context:\n" + "\n".join(summary_lines)
            })

    # Recent messages in full
    recent = history[-max_recent:] if total > max_recent else history
    for msg in recent:
        messages.append({"role": msg.role, "content": msg.content})

    return messages
```

Use this in both generate_response() and stream_response().

- [ ] **Step 3: Session continuity**
In messages.py, when loading conversation context, always load the full conversation history from the database (not just last N). The sliding window in LLMService handles truncation — the router should provide complete data.

Verify that when a user reopens a conversation, the full history is available and the LLM gets appropriate context.

- [ ] **Step 4: Write conversation context test**
Create tests/test_conversation_context.py that verifies:
- Sliding window keeps recent messages in full
- Older messages are summarized
- Very old messages are dropped
- Empty history works
- Single message works

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: implement sliding window conversation context"
```

---

### Task 5: Memory API & Browsing

**Goal:** Make the memory system browsable — users can search and view what ADAM remembers.

**Files:**
- Modify: `src/adam/api/routers/memories.py` (verify and improve endpoints)
- Create: `tests/test_memory_api.py`

- [ ] **Step 1: Verify memories router endpoints**
Read src/adam/api/routers/memories.py. Verify these endpoints work:
- GET /api/memories/search?query=... — search memories
- GET /api/memories/stats — memory analytics
- POST /api/memories/store — manual memory storage

- [ ] **Step 2: Add memory-in-conversation indicator**
When a message response includes memory context, add metadata so the frontend can show it. In the MessageResponse or as a header in SSE:
```json
{"memory_used": true, "memory_sources": ["mem_123", "mem_456"]}
```

- [ ] **Step 3: Write memory API test**
```python
def test_memory_search_endpoint():
    from adam.api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/api/memories/search", params={"query": "test"})
    assert response.status_code == 200
```

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "feat: verify and improve memory browsing API"
```

---

### Task 6: Frontend End-to-End Verification

**Goal:** Verify the React frontend connects to the backend, messages flow correctly, and streaming displays properly.

**Files:**
- Modify: `frontend/AdamChat/client/src/pages/chat.tsx` (fix API connections if needed)
- Modify: `frontend/AdamChat/client/src/hooks/use-conversation.ts` (verify API calls)
- Modify: `frontend/AdamChat/client/src/lib/queryClient.ts` (verify API base URL)
- Modify: `frontend/AdamChat/server/adam-integration.ts` (verify backend URL)

- [ ] **Step 1: Check API base URL configuration**
Read the frontend configuration to find where the API base URL is set. It should point to http://localhost:8000/api. Check:
- queryClient.ts
- adam-integration.ts
- Any .env files in frontend/

- [ ] **Step 2: Verify message sending**
Read the ChatArea component (find it in the components directory) to understand how messages are sent. Verify it calls POST /api/conversations/{id}/messages or the streaming endpoint.

- [ ] **Step 3: Verify streaming display**
Check if the frontend handles SSE events from the streaming endpoint. Look for EventSource or fetch with ReadableStream usage.

- [ ] **Step 4: Fix any connection issues**
If the frontend uses incorrect API paths, fix them. Common issues:
- Wrong base URL
- Missing CORS headers (already handled in main.py)
- Incorrect content-type headers

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "fix: verify and fix frontend-backend API connection"
```

---

### Task 7: Integration Tests & Full Verification

**Goal:** Verify the complete conversational flow works end-to-end.

**Files:**
- Modify: `tests/test_integration.py` (add Phase 2 tests)

- [ ] **Step 1: Add conversation flow integration test**
```python
def test_conversation_flow():
    """Full conversation: create project → create conversation → send message → get response."""
    from adam.api.main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Create project
    project = client.post("/api/projects/", json={"name": "Test"}).json()

    # Create conversation
    conv = client.post(f"/api/projects/{project['id']}/conversations",
                       json={"title": "Test Chat"}).json()

    # Send message
    response = client.post(f"/api/conversations/{conv['id']}/messages",
                          json={"content": "Hello ADAM", "use_memory": False})
    assert response.status_code == 200
    messages = response.json()
    assert len(messages) >= 2  # User + assistant message
```

- [ ] **Step 2: Run ALL tests**
```bash
cd /Users/vitoryago/ADAM && python -m pytest tests/ -v
```
All tests must pass.

- [ ] **Step 3: Manual verification**
Start the server and test manually:
1. Create a project via API
2. Create a conversation
3. Send a message
4. Verify streaming response
5. Send a follow-up (test multi-turn)
6. Check memory was stored

- [ ] **Step 4: Final commit**
```bash
git add -A && git commit -m "feat: complete Phase 2 conversational core with integration tests"
```

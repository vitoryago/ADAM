# ADAM End-to-End Testing Session

**Date:** 2026-04-01
**Goal:** First manual end-to-end test of the entire ADAM application — backend API, React frontend, VS Code extension, and the newly built Local Model Integration.

---

## Test Environment

| Component | Version / Config |
|-----------|-----------------|
| Hardware | M4 Pro, 48GB unified memory |
| OS | macOS (Darwin 25.3.0) |
| Python | 3.10+ |
| Node | (frontend + extension) |
| Ollama | v0.7.0, 3 local models: `qwen3:8b`, `mistral:latest`, `phi3:latest` |
| Backend | `adam-assistant 4.0.0` at `http://localhost:8000` |
| Frontend | React + Vite at `http://localhost:5173` |
| VS Code Extension | v0.5.0 (compiled from `vscode-extension/adam-code`) |
| API Keys | XAI, OpenAI, Anthropic configured. Gemini missing (optional). |

---

## Startup Sequence

```
Terminal 1: ollama serve
Terminal 2: ollama pull qwen3:8b
Terminal 3: cd /Users/vitoryago/ADAM && python -m adam.api.main
Terminal 4: cd /Users/vitoryago/ADAM/frontend/AdamChat && npm run dev
VS Code:    Cmd+Shift+P → "Developer: Install Extension from Location..." → adam-code folder
```

---

## Test Plan

### Phase 1: Regular Chat (Baseline)
- [ ] Send a message, get a response
- [ ] Model selector shows updated models
- [ ] Streaming response is smooth (no jumping)
- [ ] Live Search with Web mode works
- [ ] Live Search with X/Twitter mode works
- [ ] Switch between models (Grok, Claude, GPT)
- [ ] Cost and token tracking visible

### Phase 2: Local Model Integration
- [ ] `GET /api/local-models` returns Ollama models
- [ ] Local models appear in Deep Discussion model selector dropdown
- [ ] "Prefer Local" toggle works in React config screen
- [ ] "Prefer Local" toggle works in VS Code extension
- [ ] Budget auto-adjusts when prefer_local is ON

### Phase 3: Deep Discussion (Cloud Models)
- [ ] Create session with Peer Review pattern
- [ ] SSE streaming shows agent progress in real-time
- [ ] Progress bar updates through steps
- [ ] Agent cards expand to show content
- [ ] Session completes with final synthesized answer
- [ ] Session appears in history sidebar
- [ ] Replay button creates new session with same config

### Phase 4: Deep Discussion (Local Models)
- [ ] Create session with "Prefer Local" ON
- [ ] Local model (qwen3:8b) runs as agent
- [ ] Cost tracking shows $0.00 for local calls
- [ ] Mixed session: local Critic + cloud Coder

### Phase 5: VS Code Extension
- [ ] Activity bar icon visible
- [ ] Deep Discussion panel opens (Cmd+Alt+D)
- [ ] Config screen: question + pattern + start
- [ ] Advanced settings expand
- [ ] Live view shows progress
- [ ] Complete view shows result

---

## Issues Found

### Issue 1: CORS Error — Credentials + Wildcard Origin

**Status:** Fixed
**Severity:** Blocking
**File:** `src/adam/api/main.py:56-63`

**Problem:** Backend had `allow_origins=["*"]` with `allow_credentials=True`. The CORS spec forbids wildcard origins when credentials are included. Frontend sends `credentials: "include"`, so every API call was blocked.

**Error:**
```
Access-Control-Allow-Origin header must not be wildcard '*' when credentials mode is 'include'
```

**Fix:** Changed to explicit origins:
```python
allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173"]
```

---

### Issue 2: Outdated Model Names in Chat Selector

**Status:** Fixed
**Severity:** Medium
**File:** `frontend/AdamChat/client/src/components/chat/model-selector.tsx`

**Problem:** The regular chat model selector showed old model names: "Grok 4 Fast Reasoning", "Claude Sonnet 4.5", "GPT-4", "Grok 2 Vision". These were from the pre-Deep Discussion era and didn't match the current model roster.

**Fix:** Updated to current models: Grok 4.20 Multi-Agent, Grok 4.20 Reasoning, Grok 4.20 Standard, Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.4, Gemini 3.1 Pro.

---

### Issue 3: All Chat Responses Routed to gpt-4o-mini

**Status:** Fixed
**Severity:** High
**Files:** `src/adam/services/llm_service.py`, `src/adam/llm/query_analyzer.py`

**Problem:** `LLMService._select_model_by_complexity()` returned old model names (`grok-4-reasoning`, `grok-4`, `gpt-4o-mini`). The query analyzer's model preference lists also referenced old names. When no model matched the config, everything fell back to the hardcoded `gpt-4o-mini`.

**Fix:**
- Updated `_select_model_by_complexity()`: HIGH → `grok-4.20-0309-reasoning`, MEDIUM/LOW → `grok-4.20-0309-non-reasoning`
- Updated all `or "gpt-4o-mini"` fallbacks to `or "grok-4.20-0309-non-reasoning"`
- Updated query analyzer preference lists to current model IDs

---

### Issue 4: Streaming Response Causes UI Jumping

**Status:** Fixed
**Severity:** Medium
**File:** `frontend/AdamChat/client/src/components/chat/chat-area.tsx:73`

**Problem:** `scrollIntoView({ behavior: "smooth" })` fires on every `messages` state update. During streaming, each chunk triggers a new smooth scroll animation. Multiple overlapping smooth scrolls fight each other, causing the chat to jump around.

**Fix:** Use instant scroll during streaming, smooth only for new messages:
```javascript
messagesEndRef.current?.scrollIntoView({ behavior: isTyping ? "auto" : "smooth" });
```

---

### Issue 5: Live Search Not Working — Only Enabled for Old Model Names

**Status:** Fixed
**Severity:** High
**File:** `src/adam/services/llm_service.py:327,552`

**Problem:** Live search was gated behind a hardcoded allowlist of old model names:
```python
if use_search and final_model in ["grok-3-mini", "grok-3-mini-high", "grok-4", "grok-4-reasoning"]:
```
The new Grok 4.20 models didn't match, so search was silently skipped. The model just answered from its own knowledge and hallucinated results.

**First fix (partial):** Changed to `final_model.startswith("grok-")`.

**Second fix (full):** Removed provider restriction entirely — search is now passed as a flag to all providers. Each provider handles it appropriately or ignores it.

---

### Issue 6: Live Search Only Worked for Grok — OpenAI and Gemini Missing

**Status:** Fixed
**Severity:** Medium
**Files:** `src/adam/llm/async_client.py`, `src/adam/llm/client.py`

**Problem:** The `search_parameters` kwarg was only handled in the Grok completion methods. OpenAI and Gemini completion methods received it via `**kwargs` but didn't do anything with it — it was either silently ignored or caused errors.

**Fix:** Added search handling to each provider:
- **Grok:** `tools: [{type: "live_search"}]` via x.ai REST API
- **OpenAI:** `extra_body: {tools: [{type: "web_search_preview"}]}` via chat completions
- **Gemini:** `extra_body: {tools: [{google_search_retrieval: {}}]}` via OpenAI-compatible endpoint
- **Claude:** `use_search` kwarg silently popped (no native search)
- **Local:** `use_search` kwarg silently popped (no search support)

---

### Issue 7: Grok SearchParameters Deprecated (gRPC Error)

**Status:** Fixed
**Severity:** Blocking
**Files:** `src/adam/llm/client.py`, `src/adam/llm/async_client.py`

**Problem:** Grok's `xai_sdk.search.SearchParameters` used the old gRPC-based API which x.ai has deprecated:
```
status = StatusCode.UNIMPLEMENTED
details = "Live search is deprecated. Please switch to the Agent Tools API"
```

**Fix:** For search requests, bypass `xai_sdk` entirely and use x.ai's OpenAI-compatible REST API with the new tools format:
```python
# Old (deprecated):
from xai_sdk.search import SearchParameters
chat_params["search_parameters"] = SearchParameters(mode="on")

# New (tools API):
response = await rest_client.chat.completions.create(
    model=model, messages=msgs,
    tools=[{"type": "live_search"}],
)
```

Added `_complete_grok_with_search()` method in `client.py` that creates a temporary `AsyncOpenAI` client pointed at `https://api.x.ai/v1`.

---

### Issue 8: Wrong Tool Type Name for x.ai Search

**Status:** Fixed
**Severity:** Blocking
**File:** `src/adam/llm/client.py`, `src/adam/llm/async_client.py`

**Problem:** Used `{"type": "web_search"}` and `{"type": "x_search"}` based on docs fetch, but the actual x.ai API expects `{"type": "live_search"}`.

**Error:**
```
Failed to deserialize: tools[0].type: unknown variant web_search, expected function or live_search
```

**Fix:** Changed all Grok search tool types to `{"type": "live_search"}`. x.ai only has one search type — `live_search` covers both web and X content.

---

### Issue 9: Unused Search Modes in UI (News, RSS)

**Status:** Fixed
**Severity:** Low
**File:** `frontend/AdamChat/client/src/components/chat/search-toggle.tsx`

**Problem:** The search mode selector showed 5 options (Auto, Web, X/Twitter, News, RSS) but only Web and X/Twitter had real API support. News and RSS were aspirational. Auto didn't behave differently from Web.

**Fix:** Removed Auto, News, and RSS. Kept only **Web** and **X/Twitter**.

---

### Issue 10: `UnifiedLLMClient` vs `AsyncLLMClient` Confusion

**Status:** Fixed
**Severity:** Blocking
**File:** `src/adam/services/llm_service.py`

**Problem:** ADAM has two LLM client classes:
- `UnifiedLLMClient` in `src/adam/llm/client.py` — the older client used by `LLMService` for regular chat
- `AsyncLLMClient` in `src/adam/llm/async_client.py` — the newer client used by Deep Discussion

When fixing live search, we initially passed `use_search` as a kwarg, but `UnifiedLLMClient.complete()` has explicit parameters and doesn't accept `use_search` — it expects `search_parameters`.

**Error:** `UnifiedLLMClient.complete() got an unexpected keyword argument 'use_search'`

**Fix:** Kept `search_parameters` as the kwarg name for `UnifiedLLMClient`, and added `use_search` handling only in `AsyncLLMClient`'s provider methods.

---

## Uncommitted Fixes

The following files have fixes from this testing session that haven't been committed yet:

| File | Changes |
|------|---------|
| `src/adam/api/main.py` | CORS fix: explicit origins |
| `src/adam/llm/async_client.py` | Search support for all providers, `live_search` tool type |
| `src/adam/llm/client.py` | `_complete_grok_with_search()`, OpenAI search tools, `live_search` |
| `src/adam/llm/query_analyzer.py` | Updated model preference lists |
| `src/adam/services/llm_service.py` | Updated fallbacks, search_mode passthrough |
| `frontend/.../chat/model-selector.tsx` | Updated model roster |
| `frontend/.../chat/chat-area.tsx` | Streaming scroll fix |
| `frontend/.../chat/search-toggle.tsx` | Removed unused modes (News, RSS, Auto) |

---

## Architecture Notes Learned

1. **Two LLM clients coexist:** `UnifiedLLMClient` (regular chat, uses `xai_sdk` for Grok) and `AsyncLLMClient` (Deep Discussion, uses `AsyncOpenAI` for everything). Both need to be maintained until the older one is retired.

2. **x.ai has two APIs:** The gRPC-based `xai_sdk` and the REST-based OpenAI-compatible endpoint at `api.x.ai/v1`. The gRPC search is deprecated. The REST API uses `tools: [{type: "live_search"}]`.

3. **Search per provider:**
   - Grok: `live_search` tool (web + X combined)
   - OpenAI: `web_search_preview` tool
   - Gemini: `google_search_retrieval` tool
   - Claude: No native search
   - Local: No search

4. **Model roster needs a single source of truth.** Currently model names are scattered across: `llm/config.py`, `llm/router.py`, `llm/query_analyzer.py`, `services/llm_service.py`, `frontend/chat/model-selector.tsx`, `frontend/deep-discussion/model-selector.tsx`, `vscode/deep-discussion.js`. When models update, all locations must be changed.

---

## Next Steps

- [ ] Commit all testing fixes
- [ ] Test Deep Discussion with cloud models (Phase 3)
- [ ] Test Deep Discussion with local models (Phase 4)
- [ ] Test VS Code extension (Phase 5)
- [ ] Consider unifying the two LLM clients
- [ ] Consider a single model registry (eliminates scattered model names)

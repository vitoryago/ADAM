# Phase 4: IDE & Screen Context Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the existing VS Code extension to the consolidated backend, add SSE streaming, multi-agent progress UI, and editor context features.

**Key finding:** The extension (adam-code v0.3.3) is already beta-quality with a chat panel, 12 commands, dbt features, memory integration, and file context detection. Phase 4 is about fixing connections and adding new capabilities, not a full rebuild.

**Architecture:** Extension (thin client) → HTTP/SSE → FastAPI backend (all intelligence). Extension sends editor context (active file, diagnostics, terminal errors) alongside messages.

---

## Dependency Graph

```
Task 1 (Backend dbt endpoints) ──→ Task 2 (Extension API fixes + streaming) ──→ Task 3 (Multi-agent UI + diagnostics) ──→ Task 4 (Build + verify)
```

---

### Task 1: Backend — Add dbt API Router

**Goal:** The extension calls dbt endpoints (`/api/dbt/columns/*`) that were in the deleted `routes/` directory. Re-create them as a proper router using the knowledge layer.

**Files:**
- Create: `src/adam/api/routers/dbt.py`
- Modify: `src/adam/api/main.py` (include dbt router)
- Create: `tests/test_dbt_api.py`

- [ ] **Step 1:** Read `src/adam/knowledge/dbt_analyzer/column_intelligence.py` and `src/adam/knowledge/dbt_analyzer/documentation.py` to understand the available dbt analysis functions.

- [ ] **Step 2:** Create `src/adam/api/routers/dbt.py` with endpoints matching what the extension expects:
  - `POST /api/dbt/columns/document-model` — Document columns in a dbt model using AI
  - `POST /api/dbt/columns/analyze` — Analyze column patterns across project
  - `POST /api/dbt/columns/common-columns` — Find common columns across models
  - `POST /api/dbt/columns/generate-schema` — Generate schema.yml with AI descriptions
  - `POST /api/dbt/generate` — Generate a dbt model from source
  Each endpoint should accept the relevant request body, call the knowledge layer, and return structured results. If the knowledge service fails or isn't available, return a helpful error response.

- [ ] **Step 3:** Register the dbt router in `src/adam/api/main.py`:
  ```python
  from adam.api.routers import dbt
  app.include_router(dbt.router, prefix="/api/dbt", tags=["dbt"])
  ```

- [ ] **Step 4:** Write smoke tests and commit.

---

### Task 2: Extension — Fix API Paths + Add SSE Streaming

**Goal:** Verify all extension API calls match the backend, and add Server-Sent Events support for streaming responses.

**Files:**
- Modify: `vscode-extension/adam-code/src/client/adamClient.ts`
- Modify: `vscode-extension/adam-code/media/chat.js`
- Modify: `vscode-extension/adam-code/src/providers/chatProvider.ts`

- [ ] **Step 1:** Audit all API calls in `adamClient.ts` against the backend routers. Fix any mismatched paths. The backend endpoints are:
  - `POST /api/projects/` — create project
  - `GET /api/projects/{id}/conversations` — list conversations
  - `POST /api/projects/{id}/conversations` — create conversation
  - `POST /api/conversations/{id}/messages` — send message (sync)
  - `POST /api/conversations/{id}/messages/stream` — send message (SSE streaming)
  - `GET /api/memories/search?query=...&project_id=...` — search memory
  - `POST /api/dbt/columns/document-model` — dbt documentation
  - `POST /api/dbt/columns/analyze` — column analysis

- [ ] **Step 2:** Add SSE streaming support to `adamClient.ts`:
  Instead of POST + await response, use fetch() with ReadableStream to read SSE events incrementally. Add a `sendMessageStream(conversationId, message, onChunk)` method that:
  - POSTs to `/api/conversations/{id}/messages/stream`
  - Reads SSE events as they arrive
  - Calls `onChunk(text)` for each content chunk
  - Returns the final complete response

- [ ] **Step 3:** Update `chatProvider.ts` to use streaming when available:
  - Show response progressively as chunks arrive
  - Update the webview incrementally (append to the current message bubble)

- [ ] **Step 4:** Update `chat.js` to handle streaming display:
  - Add a "streaming" state to the current message
  - Append content chunks as they arrive
  - Remove typing indicator when first chunk arrives
  - Finalize message when stream completes

- [ ] **Step 5:** Commit.

---

### Task 3: Extension — Multi-Agent UI + Editor Diagnostics

**Goal:** Show multi-agent progress in the chat UI and send editor diagnostics as context.

**Files:**
- Modify: `vscode-extension/adam-code/src/client/adamClient.ts`
- Modify: `vscode-extension/adam-code/media/chat.js`
- Modify: `vscode-extension/adam-code/media/chat.css`
- Modify: `vscode-extension/adam-code/src/extension.ts`

- [ ] **Step 1:** Add editor diagnostics context to messages.
  In `adamClient.ts` or `extension.ts`, collect:
  - VS Code diagnostics for active file (errors, warnings via `vscode.languages.getDiagnostics()`)
  - Terminal output if errors detected
  - Git status (modified files, branch name)
  Include this in the `workspace_context` sent to the backend.

- [ ] **Step 2:** Add multi-agent event handling to `chat.js`:
  When SSE events include `agent_status`, `agent_chunk`, `agent_done` types:
  - Show an "Agent Activity" panel below the user message
  - For each agent: show name, status (thinking/done), and expandable content
  - Show cost per agent and total cost
  - Animate thinking state with subtle indicator

- [ ] **Step 3:** Add CSS for multi-agent display in `chat.css`:
  - Agent card styling (collapsible, colored by role)
  - Progress indicators
  - Cost badges

- [ ] **Step 4:** Commit.

---

### Task 4: Build, Package & Verify

**Goal:** Compile the extension, package as VSIX, and verify end-to-end.

**Files:**
- Modify: `vscode-extension/adam-code/package.json` (version bump to 0.4.0)

- [ ] **Step 1:** Update version in package.json to 0.4.0

- [ ] **Step 2:** Install dependencies and compile:
  ```bash
  cd vscode-extension/adam-code && npm install && npm run compile
  ```

- [ ] **Step 3:** Fix any TypeScript compilation errors.

- [ ] **Step 4:** Package as VSIX (if vsce is available):
  ```bash
  npx vsce package
  ```

- [ ] **Step 5:** Add backend tests for dbt endpoints:
  ```bash
  cd /Users/vitoryago/ADAM && python -m pytest tests/ -v --timeout=30
  ```

- [ ] **Step 6:** Final commit:
  ```bash
  git commit -m "feat: complete Phase 4 IDE integration with multi-agent UI"
  ```

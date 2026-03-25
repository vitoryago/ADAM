# VS Code Extension — Deep Discussion UI Design Spec

**Date:** 2026-03-25
**Status:** Draft
**Depends on:** Deep Discussion Mode backend (complete), VS Code extension v0.4.0 (complete)

## Context

The Deep Discussion backend API is fully built (7 endpoints, SSE streaming, Peer Review pattern, session management). The React web app has a full UI. The VS Code extension (v0.4.0) has a chat sidebar with multi-agent agent cards and SSE streaming — but no Deep Discussion interface.

This spec adds a dedicated Deep Discussion panel to the VS Code extension, accessible from its own activity bar icon.

## Scope

### In Scope
- New activity bar icon and webview for Deep Discussion
- Minimal config screen: question + pattern + start button, with expandable advanced section
- Simplified live view: progress bar, compact agent rows (collapsed by default), final answer
- New API methods in adamClient.ts for Deep Discussion endpoints
- Session history dropdown
- "Run Again" button on completed sessions

### Out of Scope
- "Go Deep" escalation from chat (no changes to existing chat)
- Full-detail agent cards (simplified for sidebar width)
- Session replay with config changes (just re-run with same config)

---

## Architecture

### Approach: New Webview Provider

A second `WebviewViewProvider` — `DeepDiscussionProvider` — alongside the existing `ADAMChatProvider`. Gets its own activity bar icon, own webview HTML/JS/CSS. Reuses `adamClient.ts` with new methods.

Zero changes to the existing chat webview, chatProvider, or chat.js/css.

---

## Extension Registration

### New Activity Bar Container

Add to `package.json` contributes.viewsContainers.activitybar:
```json
{
  "id": "adamDeepDiscussion",
  "title": "ADAM Deep Discussion",
  "icon": "$(hubot)"
}
```

### New Webview View

Add to `package.json` contributes.views:
```json
{
  "adamDeepDiscussion": [
    {
      "type": "webview",
      "id": "adam.deepDiscussionView",
      "name": "Deep Discussion"
    }
  ]
}
```

### New Command

Add to `package.json` contributes.commands:
```json
{
  "command": "adam.deepDiscussion",
  "title": "ADAM: Deep Discussion",
  "category": "ADAM"
}
```

Keybinding: `Cmd+Alt+D` (mac), `Ctrl+Alt+D` (windows/linux). Note: `Cmd+Shift+D` is reserved by VS Code for "Start Debugging".

### New Configuration

Add to `package.json` contributes.configuration.properties:
```json
{
  "adam.deepDiscussion.enabled": {
    "type": "boolean",
    "default": true,
    "description": "Enable Deep Discussion panel"
  }
}
```

When `adam.deepDiscussion.enabled` is `false`:
- The view uses a `when` clause: `"when": "config.adam.deepDiscussion.enabled"` to hide from the activity bar
- The command uses `enablement`: `"enablement": "config.adam.deepDiscussion.enabled"` to become a no-op
```

---

## API Client Extension

Add these methods to `src/client/adamClient.ts`:

### New Methods

**`createDeepDiscussionSession(question, pattern, conversationId?)`**
- POST `/api/deep-discussion/sessions`
- Body: `{ project_id, question, pattern, conversation_id }`
- Returns session object with smart defaults applied

**`updateDeepDiscussionConfig(sessionId, config)`**
- PUT `/api/deep-discussion/sessions/{id}/config`
- Body: `{ model_assignments?, pattern?, budget? }`
- Returns updated session

**`startDeepDiscussion(sessionId, callbacks)`**
- POST `/api/deep-discussion/sessions/{id}/start`
- SSE streaming using the same `fetch()` + `ReadableStream` pattern as existing `sendMessageStream()`
- Callbacks (matching backend SSE event names):
  - `onSessionStart(pattern: string, agents: string[])`
  - `onStepStart(stepName: string)`
  - `onAgentStart(agentRole: string, model: string)`
  - `onAgentChunk(agentRole: string, content: string)`
  - `onAgentDone(agentRole: string, cost: number, tokens: number)`
  - `onStepComplete(stepName: string, cost: number)`
  - `onSessionComplete(result: string, totalCost: number)`
  - `onError(message: string)` — handles session_error events from backend

**`getDeepDiscussionSession(sessionId)`**
- GET `/api/deep-discussion/sessions/{id}`

**`listDeepDiscussionSessions(limit?)`**
- GET `/api/deep-discussion/sessions?project_id={this.projectId}&limit={limit || 10}`
- Uses `this.projectId` from the ADAMClient instance (same as other methods)
- Default limit of 10 sessions to prevent unbounded history lists

**`replayDeepDiscussion(sessionId)`**
- POST `/api/deep-discussion/sessions/{id}/replay`

Note: The backend's `POST /sessions/from-conversation/{conversation_id}` endpoint is intentionally not exposed in the VS Code client. The "Go Deep" chat escalation is out of scope for this iteration — Deep Discussion is only accessible from its own activity bar icon.

---

## Webview Provider

### File: `src/providers/deepDiscussionProvider.ts`

Implements `vscode.WebviewViewProvider`. Follows exact same pattern as `chatProvider.ts`.

**Responsibilities:**
- Resolves webview with `deep-discussion.html` content
- Handles messages from webview (`startSession`, `updateConfig`, `loadHistory`, `runAgain`)
- Calls `adamClient` methods
- Forwards SSE events to webview via `postMessage()`

**State:**
- `currentSessionId: string | null`
- `mode: 'config' | 'live' | 'complete'`

**Message handling:**

| From Webview | Action |
|-------------|--------|
| `startSession` | Create session via API, start SSE, switch to live mode |
| `updateConfig` | Update session config via API |
| `loadHistory` | Fetch session list via API, send to webview |
| `loadSession` | Fetch session via API, show completed view |
| `runAgain` | Call replayDeepDiscussion API (creates new session with same config), update currentSessionId to the new ID, switch to config mode with settings pre-filled |

| To Webview | When |
|-----------|------|
| `showConfig` | Initial state, after "Run Again" |
| `showLive` | After session starts |
| `stepStart` | SSE step_start event |
| `agentStart` | SSE agent_start event |
| `agentChunk` | SSE agent_chunk event |
| `agentDone` | SSE agent_done event |
| `stepComplete` | SSE step_complete event |
| `sessionComplete` | SSE session_complete event |
| `sessionError` | SSE session_error event |
| `historyData` | Response to loadHistory |
| `sessionData` | Response to loadSession |

---

## Webview UI

### Files

- `media/deep-discussion.js` — UI logic
- `media/deep-discussion.css` — Styles (follows same variable-based theming as `chat.css`)

The HTML is inline in `deepDiscussionProvider.ts` (same pattern as `chatProvider.ts`).

### Config Mode (Default View)

```
┌─────────────────────────────┐
│ 🧠 Deep Discussion          │
│                              │
│ ┌──────────────────────────┐│
│ │ What do you want to      ││
│ │ analyze?                 ││
│ │                          ││
│ └──────────────────────────┘│
│                              │
│ Pattern: [Peer Review ▼]    │
│                              │
│ [▶ Start Deep Discussion]   │
│                              │
│ ▸ Advanced Settings          │
│ ┌──────────────────────────┐│
│ │ Reasoner: [Grok MA ▼]   ││
│ │ Coder:    [Opus ▼]      ││
│ │ Critic:   [GPT-5.4 ▼]   ││
│ │ Synth:    [Sonnet ▼]    ││
│ │ Budget:   [$2.00 ═══]   ││
│ └──────────────────────────┘│
│                              │
│ ─── History ───              │
│ • dbt model review (done)    │
│ • API arch debate (done)     │
└─────────────────────────────┘
```

- Question textarea: auto-focus, placeholder "What do you want to analyze?"
- Pattern dropdown: `<select>` with Sequential / Debate / Peer Review
- Start button: green, full width
- Advanced section: collapsed by default, toggle with `▸`/`▾`
  - 4 model dropdowns (grouped by provider, same as React ModelSelector)
  - Budget range input ($0.50–$5.00, step $0.25, default $2.00)
- History: list of past sessions at bottom, click to view results

### Live Mode (During Execution)

```
┌─────────────────────────────┐
│ Peer Review · Running       │
│ $0.47 / $2.00               │
│ ┌─┬─┬─┬─┬─┐                │
│ │█│█│▒│ │ │                 │
│ └─┴─┴─┴─┴─┘                │
│ Pro Rev Reb Rea Syn         │
│                              │
│ ● Reasoner  ✓  $0.12       │
│ ● Coder     ✓  $0.18       │
│ ● Critic    ✓  $0.09       │
│ ◌ Reasoner  ⟳ thinking...  │
│ ◌ Coder     ─  pending     │
│ ◌ Critic    ─  pending     │
│ ◌ Synth     ─  pending     │
└─────────────────────────────┘
```

- Header: pattern name + "Running" status + cost counter
- Progress bar: 3-5 segments depending on pattern. Green = done, amber = active, gray = pending
- Agent rows: compact single line each
  - Color dot (blue/green/orange/purple) + name + status + cost
  - Click to expand and see content (collapsed by default)
  - "thinking..." with pulse animation when active
- Agent colors match existing chat.js: reasoner=#4a9eff, coder=#3ddc84, critic=#ff9f43, synthesizer=#a855f7

### Complete Mode (After Execution)

```
┌─────────────────────────────┐
│ Peer Review · Done          │
│ Total: $0.82 · 4 agents     │
│ ┌─┬─┬─┬─┬─┐                │
│ │█│█│█│█│█│                 │
│ └─┴─┴─┴─┴─┘                │
│                              │
│ ┌──────────────────────────┐│
│ │ Final Answer              ││
│ │                           ││
│ │ The dbt incremental      ││
│ │ model has 3 key issues... ││
│ │                           ││
│ └──────────────────────────┘│
│                              │
│ ● Reasoner  ✓  $0.12  ▸    │
│ ● Coder     ✓  $0.18  ▸    │
│ ● Critic    ✓  $0.09  ▸    │
│ ● Synth     ✓  $0.43  ▸    │
│                              │
│ [Run Again]                  │
└─────────────────────────────┘
```

- All progress bar segments green
- Final answer displayed prominently in a bordered box with markdown rendering (use the same custom `renderMarkdown()` + `escapeHtml()` approach from `chat.js`)
- Agent rows with expand arrows (▸) to view individual contributions
- "Run Again" button returns to config mode with same settings pre-filled

---

## Error Handling and Cancellation

### Config → Live Transition
After clicking "Start Deep Discussion":
1. Button changes to "Starting..." (disabled) while API call runs
2. If session creation fails (server unreachable, 500 error): show error message inline, return to config mode
3. If SSE connection succeeds: switch to live mode

### SSE Connection Drop
If the SSE stream disconnects mid-session:
1. Show "Connection lost" warning in the live view header
2. Fetch session state via `getDeepDiscussionSession()` — if the session completed server-side, show complete mode with results
3. If still running server-side, show "Session running on server. Refresh to check status." with a refresh button

### Cancellation
Live mode shows a small "Cancel" link in the header. Clicking it:
1. Closes the SSE reader (aborts the fetch via `AbortController`)
2. Does NOT cancel the server-side session (backend has no cancel endpoint)
3. Returns to config mode
4. The session continues server-side — user can find results later in history

### Per-Agent Error
The backend handles per-agent failures server-side (agent fails → pipeline continues). The client only sees `session_error` events for session-level failures. Individual agent failures result in missing agent entries in the final scratchpad — the live view simply won't show an `agentDone` event for that agent, and the row stays in "pending" or "thinking" state until the session completes.

---

## Extension Registration in extension.ts

In `activate()`:
1. Create `DeepDiscussionProvider` instance (pass `adamClient`)
2. Register: `vscode.window.registerWebviewViewProvider('adam.deepDiscussionView', deepDiscussionProvider)`
3. Register command `adam.deepDiscussion` that focuses the view

---

## Testing Strategy

- **Compile check:** `npm run compile` succeeds with no TypeScript errors
- **Manual test:** Extension loads, shows activity bar icon, opens Deep Discussion panel
- **API client:** Unit test new methods against mock responses (optional, not blocking)

---

## Files Summary

| Action | File | Purpose |
|--------|------|---------|
| Create | `src/providers/deepDiscussionProvider.ts` | Webview provider for Deep Discussion |
| Create | `media/deep-discussion.js` | Webview UI logic (config, live, complete modes) |
| Create | `media/deep-discussion.css` | Webview styles |
| Modify | `src/client/adamClient.ts` | Add 6 new Deep Discussion API methods |
| Modify | `src/extension.ts` | Register new provider and command |
| Modify | `package.json` | Add activity bar container, view, command, keybinding, config |

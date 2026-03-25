# VS Code Deep Discussion UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated Deep Discussion panel to the VS Code extension with its own activity bar icon, config screen, and simplified live view.

**Architecture:** New `DeepDiscussionProvider` webview alongside existing `ADAMChatProvider`. New API methods in `adamClient.ts`. Separate `deep-discussion.js` + `deep-discussion.css` for the webview UI. Zero changes to existing chat.

**Tech Stack:** VS Code Extension API, TypeScript, HTML/CSS/JS webview, SSE via fetch + ReadableStream

**Spec:** `docs/superpowers/specs/2026-03-25-vscode-deep-discussion-design.md`

---

### Task 1: Add Deep Discussion API Methods to adamClient.ts

**Files:**
- Modify: `vscode-extension/adam-code/src/client/adamClient.ts:636-639` (before `disconnect()`)

- [ ] **Step 1: Add the 6 new methods to ADAMClient**

Add these methods before the `disconnect()` method at line 636:

```typescript
// --- Deep Discussion API ---

async createDeepDiscussionSession(
    question: string,
    pattern: string = 'peer_review',
    conversationId?: string
): Promise<any> {
    const body: any = { project_id: this.projectId, question, pattern };
    if (conversationId) { body.conversation_id = conversationId; }
    const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) { throw new Error(`Create session failed: ${response.status}`); }
    return response.json();
}

async updateDeepDiscussionConfig(
    sessionId: string,
    config: { model_assignments?: Record<string, string>; pattern?: string; budget?: number }
): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions/${sessionId}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
    });
    if (!response.ok) { throw new Error(`Update config failed: ${response.status}`); }
    return response.json();
}

async startDeepDiscussion(
    sessionId: string,
    signal?: AbortSignal,
    callbacks: {
        onSessionStart?: (pattern: string, agents: string[]) => void;
        onStepStart?: (stepName: string) => void;
        onAgentStart?: (agentRole: string, model: string) => void;
        onAgentChunk?: (agentRole: string, content: string) => void;
        onAgentDone?: (agentRole: string, cost: number, tokens: number) => void;
        onStepComplete?: (stepName: string, cost: number) => void;
        onSessionComplete?: (result: string, totalCost: number) => void;
        onError?: (message: string) => void;
    }
): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions/${sessionId}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,  // Pass AbortSignal for cancellation support
    });
    if (!response.ok) { throw new Error(`Start session failed: ${response.status}`); }
    if (!response.body) { throw new Error('No response body for streaming'); }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
    while (true) {
        const { done, value } = await reader.read();
        if (done) { break; }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (!line.startsWith('data: ')) { continue; }
            try {
                const event = JSON.parse(line.slice(6));
                switch (event.type) {
                    case 'session_start':
                        callbacks.onSessionStart?.(event.pattern, event.agents || []);
                        break;
                    case 'step_start':
                        callbacks.onStepStart?.(event.step);
                        break;
                    case 'agent_start':
                        callbacks.onAgentStart?.(event.agent || event.role, event.model || '');
                        break;
                    case 'agent_chunk':
                        callbacks.onAgentChunk?.(event.agent, event.content);
                        break;
                    case 'agent_done':
                        callbacks.onAgentDone?.(event.agent, event.cost || 0, event.tokens || 0);
                        break;
                    case 'step_complete':
                        callbacks.onStepComplete?.(event.step, event.cost || 0);
                        break;
                    case 'session_complete':
                        callbacks.onSessionComplete?.(event.result || '', event.total_cost || 0);
                        break;
                    case 'done':
                        callbacks.onSessionComplete?.(
                            event.scratchpad?.entries?.find((e: any) => e.entry_type === 'synthesis')?.content || '',
                            event.total_cost || 0
                        );
                        break;
                    case 'session_error':
                        callbacks.onError?.(event.message || 'Session error');
                        break;
                }
            } catch (e) {
                // Skip unparseable lines
            }
        }
    }
    } finally {
        reader.releaseLock();
    }
}

async getDeepDiscussionSession(sessionId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions/${sessionId}`);
    if (!response.ok) { throw new Error(`Get session failed: ${response.status}`); }
    return response.json();
}

async listDeepDiscussionSessions(limit: number = 10): Promise<any[]> {
    const response = await fetch(
        `${this.baseURL}/api/deep-discussion/sessions?project_id=${this.projectId}`
    );
    if (!response.ok) { throw new Error(`List sessions failed: ${response.status}`); }
    const sessions = await response.json();
    return sessions.slice(0, limit);
}

async replayDeepDiscussion(sessionId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions/${sessionId}/replay`, {
        method: 'POST',
    });
    if (!response.ok) { throw new Error(`Replay failed: ${response.status}`); }
    return response.json();
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd vscode-extension/adam-code && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add vscode-extension/adam-code/src/client/adamClient.ts
git commit -m "feat: add Deep Discussion API methods to VS Code adamClient"
```

---

### Task 2: Update package.json — Activity Bar, View, Command, Config

**Files:**
- Modify: `vscode-extension/adam-code/package.json` (contributes section)

Note: Line numbers below are for initial orientation. Apply changes relative to surrounding JSON landmarks ("after the last entry in X array"), not absolute line positions, since earlier steps shift later line numbers.

- [ ] **Step 1: Add command to commands array**

After the last command entry (line 76, before `]`), add:
```json
,{
  "command": "adam.deepDiscussion",
  "title": "ADAM: Deep Discussion",
  "category": "ADAM",
  "icon": "$(hubot)",
  "enablement": "config.adam.deepDiscussion.enabled"
}
```

- [ ] **Step 2: Add keybinding**

After the last keybinding entry (line 89, before `]`), add:
```json
,{
  "command": "adam.deepDiscussion",
  "key": "ctrl+alt+d",
  "mac": "cmd+alt+d"
}
```

- [ ] **Step 3: Add activity bar container**

In `viewsContainers.activitybar` array (line 92-98), after the existing `adam` container, add:
```json
,{
  "id": "adamDeepDiscussion",
  "title": "ADAM Deep Discussion",
  "icon": "$(hubot)"
}
```

- [ ] **Step 4: Add view**

In `views` object (line 100-108), after the `adam` views, add:
```json
,"adamDeepDiscussion": [
  {
    "type": "webview",
    "id": "adam.deepDiscussionView",
    "name": "Deep Discussion",
    "when": "config.adam.deepDiscussion.enabled"
  }
]
```

- [ ] **Step 5: Add configuration option**

In `configuration.properties` (line 112-144), after `adam.enableVoice`, add:
```json
,"adam.deepDiscussion.enabled": {
  "type": "boolean",
  "default": true,
  "description": "Enable Deep Discussion panel in activity bar"
}
```

- [ ] **Step 6: Verify JSON is valid**

Run: `cd vscode-extension/adam-code && node -e "JSON.parse(require('fs').readFileSync('package.json','utf8')); console.log('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add vscode-extension/adam-code/package.json
git commit -m "feat: add Deep Discussion activity bar, view, command, and config to package.json"
```

---

### Task 3: Create Deep Discussion Webview Provider

**Files:**
- Create: `vscode-extension/adam-code/src/providers/deepDiscussionProvider.ts`

- [ ] **Step 1: Create the provider**

Create `vscode-extension/adam-code/src/providers/deepDiscussionProvider.ts`. Follow the exact pattern of `chatProvider.ts`:

```typescript
import * as vscode from 'vscode';
import { ADAMClient } from '../client/adamClient';

export class DeepDiscussionProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'adam.deepDiscussionView';
    private _view?: vscode.WebviewView;
    private currentSessionId: string | null = null;
    private abortController: AbortController | null = null;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly adamClient: ADAMClient
    ) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };
        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async data => {
            switch (data.type) {
                case 'startSession':
                    await this.handleStartSession(data.question, data.pattern, data.modelAssignments, data.budget);
                    break;
                case 'cancelSession':
                    this.handleCancelSession();
                    break;
                case 'loadHistory':
                    await this.handleLoadHistory();
                    break;
                case 'loadSession':
                    await this.handleLoadSession(data.sessionId);
                    break;
                case 'runAgain':
                    await this.handleRunAgain();
                    break;
            }
        });

        // Load history on init
        this.handleLoadHistory();
    }

    private async handleStartSession(
        question: string,
        pattern: string,
        modelAssignments?: Record<string, string>,
        budget?: number
    ) {
        try {
            // Create session
            const session = await this.adamClient.createDeepDiscussionSession(question, pattern);
            this.currentSessionId = session.id;

            // Update config if advanced settings were changed
            if (modelAssignments || budget) {
                const config: any = {};
                if (modelAssignments) { config.model_assignments = modelAssignments; }
                if (budget) { config.budget = budget; }
                await this.adamClient.updateDeepDiscussionConfig(session.id, config);
            }

            // Switch to live mode
            this._view?.webview.postMessage({
                type: 'showLive',
                pattern,
                budget: budget || session.budget,
            });

            // Start SSE streaming with AbortController for cancellation
            this.abortController = new AbortController();
            await this.adamClient.startDeepDiscussion(session.id, this.abortController.signal, {
                onSessionStart: (pattern, agents) => {
                    this._view?.webview.postMessage({ type: 'sessionStart', pattern, agents });
                },
                onStepStart: (step) => {
                    this._view?.webview.postMessage({ type: 'stepStart', step });
                },
                onAgentStart: (agent, model) => {
                    this._view?.webview.postMessage({ type: 'agentStart', agent, model });
                },
                onAgentChunk: (agent, content) => {
                    this._view?.webview.postMessage({ type: 'agentChunk', agent, content });
                },
                onAgentDone: (agent, cost, tokens) => {
                    this._view?.webview.postMessage({ type: 'agentDone', agent, cost, tokens });
                },
                onStepComplete: (step, cost) => {
                    this._view?.webview.postMessage({ type: 'stepComplete', step, cost });
                },
                onSessionComplete: (result, totalCost) => {
                    this._view?.webview.postMessage({ type: 'sessionComplete', result, totalCost });
                },
                onError: (message) => {
                    this._view?.webview.postMessage({ type: 'sessionError', message });
                },
            });
        } catch (error: any) {
            if (error.name === 'AbortError') {
                // User cancelled — already handled by handleCancelSession
                return;
            }
            // SSE connection drop — try to recover by checking session state
            if (this.currentSessionId) {
                try {
                    const session = await this.adamClient.getDeepDiscussionSession(this.currentSessionId);
                    if (session.status === 'completed') {
                        this._view?.webview.postMessage({ type: 'sessionData', session });
                        return;
                    }
                } catch (_) { /* ignore recovery failure */ }
            }
            this._view?.webview.postMessage({
                type: 'sessionError',
                message: error.message || 'Connection lost. Session may still be running on server.',
            });
        }
    }

    private handleCancelSession() {
        this.abortController?.abort();
        this.abortController = null;
        this._view?.webview.postMessage({ type: 'showConfig' });
    }

    private async handleLoadHistory() {
        try {
            const sessions = await this.adamClient.listDeepDiscussionSessions(10);
            this._view?.webview.postMessage({ type: 'historyData', sessions });
        } catch (error) {
            // Silently fail — history is non-critical
        }
    }

    private async handleLoadSession(sessionId: string) {
        try {
            const session = await this.adamClient.getDeepDiscussionSession(sessionId);
            this.currentSessionId = sessionId;
            this._view?.webview.postMessage({ type: 'sessionData', session });
        } catch (error: any) {
            this._view?.webview.postMessage({
                type: 'sessionError',
                message: error.message || 'Failed to load session',
            });
        }
    }

    private async handleRunAgain() {
        if (!this.currentSessionId) { return; }
        try {
            const newSession = await this.adamClient.replayDeepDiscussion(this.currentSessionId);
            this.currentSessionId = newSession.id;
            this._view?.webview.postMessage({
                type: 'showConfig',
                prefill: {
                    question: newSession.question,
                    pattern: newSession.pattern,
                    modelAssignments: newSession.model_assignments,
                    budget: newSession.budget,
                },
            });
        } catch (error: any) {
            this._view?.webview.postMessage({
                type: 'sessionError',
                message: error.message || 'Failed to replay',
            });
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'deep-discussion.js')
        );
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'media', 'deep-discussion.css')
        );

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="${styleUri}" rel="stylesheet">
    <title>Deep Discussion</title>
</head>
<body>
    <div id="app"></div>
    <script src="${scriptUri}"></script>
</body>
</html>`;
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cd vscode-extension/adam-code && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add vscode-extension/adam-code/src/providers/deepDiscussionProvider.ts
git commit -m "feat: add DeepDiscussionProvider webview provider"
```

---

### Task 4: Register Provider and Command in extension.ts

**Files:**
- Modify: `vscode-extension/adam-code/src/extension.ts:1-37`

- [ ] **Step 1: Add import**

After the existing imports (line 8), add:
```typescript
import { DeepDiscussionProvider } from './providers/deepDiscussionProvider';
```

After `let chatProvider: ADAMChatProvider;` (line 11), add:
```typescript
let deepDiscussionProvider: DeepDiscussionProvider;
```

- [ ] **Step 2: Initialize and register provider**

After the chatProvider registration block (line 34), add:
```typescript
    // Initialize Deep Discussion provider
    deepDiscussionProvider = new DeepDiscussionProvider(context.extensionUri, adamClient);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('adam.deepDiscussionView', deepDiscussionProvider)
    );
```

- [ ] **Step 3: Register command**

Find the `registerCommands` function in extension.ts and add at the end (before the function's closing brace):
```typescript
    context.subscriptions.push(
        vscode.commands.registerCommand('adam.deepDiscussion', () => {
            vscode.commands.executeCommand('adam.deepDiscussionView.focus');
        })
    );
```

- [ ] **Step 4: Verify compilation**

Run: `cd vscode-extension/adam-code && npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add vscode-extension/adam-code/src/extension.ts
git commit -m "feat: register Deep Discussion provider and command in extension"
```

---

### Task 5: Create Webview CSS

**Files:**
- Create: `vscode-extension/adam-code/media/deep-discussion.css`

- [ ] **Step 1: Create the stylesheet**

Create `vscode-extension/adam-code/media/deep-discussion.css`. Follow the same VS Code CSS variable pattern as `chat.css` — use `var(--vscode-*)` for theming.

Key elements to style:
- `#app` — main container
- `.config-mode`, `.live-mode`, `.complete-mode` — mode containers
- `.question-input` — textarea for question
- `.pattern-select` — dropdown styling
- `.start-btn` — green prominent button
- `.advanced-toggle` — expandable section header
- `.advanced-section` — model dropdowns + budget slider
- `.agent-row` — compact agent row (color dot + name + status + cost)
- `.agent-row.expanded` — shows content below
- `.progress-bar` — step segments container
- `.progress-step` — individual segment (`.done`, `.active`, `.pending` states)
- `.final-answer` — bordered box for synthesis result
- `.history-list` — past sessions list
- `.error-message` — red error display
- `.cancel-link` — small cancel text in live header

Agent colors:
```css
.agent-dot.reasoner { background: #4a9eff; }
.agent-dot.coder { background: #3ddc84; }
.agent-dot.critic { background: #ff9f43; }
.agent-dot.synthesizer { background: #a855f7; }
```

Progress states:
```css
.progress-step.done { background: var(--vscode-testing-iconPassed); }
.progress-step.active { background: #f59e0b; animation: pulse 1.5s ease-in-out infinite; }
.progress-step.pending { background: var(--vscode-descriptionForeground); opacity: 0.3; }
```

Use `var(--vscode-button-background)` for the start button, `var(--vscode-input-background)` for inputs, `var(--vscode-editor-background)` for main background.

The implementer should write a complete stylesheet covering all elements listed above. Reference `media/chat.css` for the VS Code variable usage patterns and sizing conventions. The sidebar is ~300px wide — design accordingly.

- [ ] **Step 2: Commit**

```bash
git add vscode-extension/adam-code/media/deep-discussion.css
git commit -m "feat: add Deep Discussion webview stylesheet"
```

---

### Task 6: Create Webview JavaScript

**Files:**
- Create: `vscode-extension/adam-code/media/deep-discussion.js`

- [ ] **Step 1: Create the webview JS**

Create `vscode-extension/adam-code/media/deep-discussion.js`. This is the core UI logic. It must handle 3 modes: config, live, complete.

Structure:
```javascript
(function() {
    const vscode = acquireVsCodeApi();

    // --- State ---
    let mode = 'config';  // 'config' | 'live' | 'complete'
    let agents = [];       // { role, model, status, content, cost, tokens }
    let completedSteps = [];
    let currentStep = '';
    let totalCost = 0;
    let sessionResult = '';
    let historySessions = [];

    // --- Model data ---
    const MODELS = { /* same groups as React ModelSelector */ };
    const SMART_DEFAULTS = {
        reasoner: 'grok-4.20-multi-agent-0309',
        coder: 'claude-opus-4-6',
        critic: 'gpt-5.4-2026-03-05',
        synthesizer: 'claude-sonnet-4-6',
    };
    const PATTERNS = {
        sequential: { label: 'Sequential', steps: ['produce','code','review','synthesize'] },
        debate: { label: 'Debate', steps: ['debate_a','debate_b','reconcile'] },
        peer_review: { label: 'Peer Review', steps: ['produce','review','rebuttal','react','synthesize'] },
    };

    // --- Render functions ---
    function render() { /* switch on mode, call renderConfig/renderLive/renderComplete */ }
    function renderConfig(prefill) { /* question textarea + pattern select + start btn + advanced toggle + history */ }
    function renderLive() { /* header + progress bar + agent rows + cancel link */ }
    function renderComplete() { /* header + progress bar + final answer + agent rows + run again btn */ }
    function renderProgressBar(pattern, completedSteps, currentStep) { /* horizontal segments */ }
    function renderAgentRow(agent) { /* compact row with expand toggle */ }

    // --- Markdown rendering (same as chat.js) ---
    function renderMarkdown(text) { /* copy from chat.js lines 490-511 */ }
    function escapeHtml(text) { /* copy from chat.js lines 513-521 */ }

    // --- Event handlers ---
    function handleStart() {
        const question = document.getElementById('dd-question').value;
        const pattern = document.getElementById('dd-pattern').value;
        // Collect advanced settings if expanded
        const modelAssignments = collectModelAssignments();
        const budget = parseFloat(document.getElementById('dd-budget')?.value || '2.0');
        vscode.postMessage({ type: 'startSession', question, pattern, modelAssignments, budget });
    }
    function handleCancel() { vscode.postMessage({ type: 'cancelSession' }); }
    function handleRunAgain() { vscode.postMessage({ type: 'runAgain' }); }
    function handleToggleAgent(index) { /* expand/collapse agent content */ }
    function handleToggleAdvanced() { /* show/hide advanced settings */ }
    function handleLoadSession(sessionId) { vscode.postMessage({ type: 'loadSession', sessionId }); }

    // --- Message handler from extension ---
    window.addEventListener('message', function(event) {
        const msg = event.data;
        switch (msg.type) {
            case 'showConfig':
                mode = 'config';
                render(msg.prefill);
                break;
            case 'showLive':
                mode = 'live';
                agents = [];
                completedSteps = [];
                currentStep = '';
                totalCost = 0;
                render();
                break;
            case 'stepStart':
                currentStep = msg.step;
                render();
                break;
            case 'agentStart':
                agents.push({ role: msg.agent, model: msg.model, status: 'thinking', content: '', cost: 0, tokens: 0 });
                render();
                break;
            case 'agentChunk':
                var agent = agents.find(a => a.role === msg.agent && a.status === 'thinking');
                if (agent) { agent.content += msg.content; }
                render();
                break;
            case 'agentDone':
                var doneAgent = agents.find(a => a.role === msg.agent && a.status === 'thinking');
                if (doneAgent) { doneAgent.status = 'done'; doneAgent.cost = msg.cost; doneAgent.tokens = msg.tokens; }
                totalCost = agents.reduce((sum, a) => sum + a.cost, 0);
                render();
                break;
            case 'stepComplete':
                completedSteps.push(msg.step);
                currentStep = '';
                render();
                break;
            case 'sessionComplete':
                mode = 'complete';
                sessionResult = msg.result;
                totalCost = msg.totalCost;
                render();
                break;
            case 'sessionError':
                // Show error inline
                document.getElementById('app').innerHTML += '<div class="error-message">' + escapeHtml(msg.message) + '</div>';
                break;
            case 'historyData':
                historySessions = msg.sessions || [];
                if (mode === 'config') { render(); }
                break;
            case 'sessionData':
                // Show completed session from history
                var s = msg.session;
                mode = 'complete';
                sessionResult = s.result || '';
                totalCost = s.total_cost || 0;
                agents = (s.scratchpad_data?.entries || []).map(function(e) {
                    return { role: e.agent_name, model: e.model_used, status: 'done', content: e.content, cost: e.cost, tokens: e.tokens };
                });
                completedSteps = PATTERNS[s.pattern]?.steps || [];
                render();
                break;
        }
    });

    // Initial render
    vscode.postMessage({ type: 'loadHistory' });
    render();
})();
```

The implementer should write complete implementations for all render functions (not stubs). Reference `media/chat.js` for DOM manipulation patterns, `renderMarkdown()`, `escapeHtml()`, and the agent card rendering approach. The config mode should disable the Start button and show "Starting..." during the API call. The `updateConfig` message is intentionally omitted — config updates are folded into `handleStartSession` in the provider.

The model selector dropdowns use `<select>` with `<optgroup>` for each provider (X.AI, Anthropic, OpenAI, Google) — same approach as the React ModelSelector.

- [ ] **Step 2: Verify the extension compiles**

Run: `cd vscode-extension/adam-code && npm run compile`
Expected: No errors (JS files don't go through tsc, only TS files)

- [ ] **Step 3: Commit**

```bash
git add vscode-extension/adam-code/media/deep-discussion.js
git commit -m "feat: add Deep Discussion webview UI with config, live, and complete modes"
```

---

### Task 7: Bump Version and Final Verification

**Files:**
- Modify: `vscode-extension/adam-code/package.json:5` (version)

- [ ] **Step 1: Bump version to 0.5.0**

In `package.json` line 5, change `"version": "0.4.0"` to `"version": "0.5.0"`.

- [ ] **Step 2: Run full compile**

Run: `cd vscode-extension/adam-code && npm run compile`
Expected: No errors

- [ ] **Step 3: Verify package.json is valid**

Run: `cd vscode-extension/adam-code && node -e "const p = JSON.parse(require('fs').readFileSync('package.json','utf8')); console.log(p.name, p.version, Object.keys(p.contributes.views))"`
Expected: `adam-code 0.5.0 [ 'adam', 'adamDeepDiscussion' ]`

- [ ] **Step 4: Commit completion**

```bash
git add vscode-extension/adam-code/package.json
git commit -m "feat: complete VS Code Deep Discussion UI (extension v0.5.0)

VS Code Deep Discussion delivers:
- Separate activity bar icon with dedicated webview panel
- Minimal config: question + pattern + start, expandable advanced settings
- Live view: progress bar, compact agent rows, cost tracking
- Complete view: final answer with expandable agent contributions
- Session history and Run Again support
- 6 new API methods in adamClient
- Cmd+Alt+D keybinding"
```

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
        _context: vscode.WebviewViewResolveContext,
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
                    await this.handleStartSession(data);
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
                case 'fetchLocalModels':
                    await this.handleFetchLocalModels();
                    break;
            }
        });

        // Load history on init
        this.handleLoadHistory();
    }

    private async handleStartSession(data: {
        question: string;
        pattern: string;
        modelAssignments?: Record<string, string>;
        budget?: number;
        preferLocal?: boolean;
    }) {
        try {
            // Create the session
            const session = await this.adamClient.createDeepDiscussionSession(
                data.question,
                data.pattern
            );

            this.currentSessionId = session.id;

            // Update config if advanced settings were provided
            if (data.modelAssignments || data.budget !== undefined || data.preferLocal !== undefined) {
                await this.adamClient.updateDeepDiscussionConfig(session.id, {
                    model_assignments: data.modelAssignments,
                    budget: data.budget,
                    prefer_local: data.preferLocal
                });
            }

            // Show live view in webview
            this._view?.webview.postMessage({ type: 'showLive', session });

            // Set up AbortController for cancellation
            this.abortController = new AbortController();

            // Start SSE streaming
            await this.adamClient.startDeepDiscussion(
                session.id,
                this.abortController.signal,
                {
                    onSessionStart: (eventData) => {
                        this._view?.webview.postMessage({ type: 'sessionStart', data: eventData });
                    },
                    onStepStart: (eventData) => {
                        this._view?.webview.postMessage({ type: 'stepStart', data: eventData });
                    },
                    onAgentStart: (agent, eventData) => {
                        this._view?.webview.postMessage({ type: 'agentStart', agent, data: eventData });
                    },
                    onAgentChunk: (agent, content) => {
                        this._view?.webview.postMessage({ type: 'agentChunk', agent, content });
                    },
                    onAgentDone: (agent, eventData) => {
                        this._view?.webview.postMessage({ type: 'agentDone', agent, data: eventData });
                    },
                    onStepComplete: (eventData) => {
                        this._view?.webview.postMessage({ type: 'stepComplete', data: eventData });
                    },
                    onSessionComplete: (eventData) => {
                        this._view?.webview.postMessage({ type: 'sessionComplete', data: eventData });
                    },
                    onDone: () => {
                        this._view?.webview.postMessage({ type: 'streamDone' });
                        this.abortController = null;
                    },
                    onError: (message) => {
                        this._view?.webview.postMessage({ type: 'streamError', message });
                    }
                }
            );
        } catch (error: any) {
            // If user cancelled (AbortError), silently ignore
            if (error?.name === 'AbortError') {
                return;
            }

            // Try to recover by fetching current session state
            if (this.currentSessionId) {
                try {
                    const sessionState = await this.adamClient.getDeepDiscussionSession(this.currentSessionId);
                    this._view?.webview.postMessage({ type: 'sessionRecovered', data: sessionState });
                } catch {
                    // Recovery failed -- show error
                }
            }

            this._view?.webview.postMessage({
                type: 'error',
                message: `Session failed: ${error?.message || error}`
            });
        }
    }

    private handleCancelSession() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        this._view?.webview.postMessage({ type: 'showConfig' });
    }

    private async handleLoadHistory() {
        try {
            const sessions = await this.adamClient.listDeepDiscussionSessions(10);
            this._view?.webview.postMessage({ type: 'historyData', sessions });
        } catch {
            // Silently fail if server is unreachable
        }
    }

    private async handleLoadSession(sessionId: string) {
        try {
            const session = await this.adamClient.getDeepDiscussionSession(sessionId);
            this.currentSessionId = sessionId;
            this._view?.webview.postMessage({ type: 'sessionData', session });
        } catch (error: any) {
            this._view?.webview.postMessage({
                type: 'error',
                message: `Failed to load session: ${error?.message || error}`
            });
        }
    }

    private async handleRunAgain() {
        if (!this.currentSessionId) {
            this._view?.webview.postMessage({
                type: 'error',
                message: 'No session to replay'
            });
            return;
        }

        try {
            const newSession = await this.adamClient.replayDeepDiscussion(this.currentSessionId);
            this.currentSessionId = newSession.id;
            this._view?.webview.postMessage({
                type: 'showConfig',
                prefill: {
                    question: newSession.question,
                    pattern: newSession.pattern
                }
            });
        } catch (error: any) {
            this._view?.webview.postMessage({
                type: 'error',
                message: `Failed to replay session: ${error?.message || error}`
            });
        }
    }

    private async handleFetchLocalModels() {
        try {
            const baseURL = this.adamClient.baseURL || 'http://localhost:8000';
            const response = await fetch(`${baseURL}/api/local-models`);
            if (response.ok) {
                const models = await response.json();
                this._view?.webview.postMessage({ type: 'localModelsData', models });
            }
        } catch (_) {
            // Silently fail — local models are optional
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'deep-discussion.css'));
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'deep-discussion.js'));

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>ADAM Deep Discussion</title>
        </head>
        <body>
            <div id="deep-discussion-root"></div>

            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}

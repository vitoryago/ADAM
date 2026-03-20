import * as vscode from 'vscode';
import { ADAMClient, Message } from '../client/adamClient';

export class ADAMChatProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'adam.chatView';
    private _view?: vscode.WebviewView;
    private messages: Message[] = [];

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
                case 'sendMessage':
                    await this.handleUserMessage(data.message);
                    break;
                case 'clear':
                    this.messages = [];
                    this.updateWebview();
                    break;
                case 'export':
                    this.exportConversation();
                    break;
                case 'changeStyle':
                    this.adamClient.setResponseStyle(data.style);
                    vscode.window.showInformationMessage(`Response style changed to: ${data.style}`);
                    break;
            }
        });

        // Load initial messages
        this.updateWebview();
    }

    public show() {
        if (this._view) {
            this._view.show(true);
        } else {
            vscode.commands.executeCommand('adam.chatView.focus');
        }
    }

    public addMessage(message: Message) {
        this.messages.push(message);
        this.updateWebview();
    }

    private async handleUserMessage(content: string) {
        // Add user message
        this.addMessage({ role: 'user', content });

        // Show typing indicator
        this._view?.webview.postMessage({ type: 'typing', isTyping: true });

        try {
            // Try streaming first, fall back to sync
            await this.handleUserMessageStreaming(content);
        } catch (streamError) {
            console.warn('Streaming failed, falling back to sync:', streamError);
            try {
                const response = await this.adamClient.sendMessage(content);
                await this.handleFileCreation(response.content);
                this.addMessage(response);
            } catch (syncError) {
                this.addMessage({
                    role: 'assistant',
                    content: `Error: ${syncError}`
                });
            }
        } finally {
            this._view?.webview.postMessage({ type: 'typing', isTyping: false });
            this._view?.webview.postMessage({ type: 'streamEnd' });
        }
    }

    /** Handle a user message using SSE streaming */
    private async handleUserMessageStreaming(content: string): Promise<void> {
        let firstChunkReceived = false;

        const response = await this.adamClient.sendMessageStream(content, {
            onChunk: (text: string) => {
                if (!firstChunkReceived) {
                    firstChunkReceived = true;
                    // Hide typing indicator, start streaming display
                    this._view?.webview.postMessage({ type: 'typing', isTyping: false });
                    this._view?.webview.postMessage({ type: 'streamStart' });
                }
                this._view?.webview.postMessage({
                    type: 'streamChunk',
                    content: text
                });
            },

            onAgentStatus: (agent: string, status: string, pattern?: string) => {
                this._view?.webview.postMessage({
                    type: 'agentStatus',
                    agent,
                    status,
                    pattern
                });
            },

            onAgentChunk: (agent: string, agentContent: string) => {
                this._view?.webview.postMessage({
                    type: 'agentChunk',
                    agent,
                    content: agentContent
                });
            },

            onAgentDone: (agent: string, cost: number) => {
                this._view?.webview.postMessage({
                    type: 'agentDone',
                    agent,
                    cost
                });
            },

            onComplete: (id: string, tokens: number, cost: number, model: string) => {
                this._view?.webview.postMessage({
                    type: 'streamComplete',
                    id,
                    tokens,
                    cost,
                    model
                });
            },

            onError: (message: string) => {
                this._view?.webview.postMessage({
                    type: 'streamError',
                    message
                });
            }
        });

        // Check for file creation markers in the response
        await this.handleFileCreation(response.content);

        // Store the final message in our local history
        this.messages.push(response);
    }

    private async handleFileCreation(content: string) {
        // Parse file markers for both creation and updates
        // Format: <<<CREATE_FILE:path>>> or <<<UPDATE_FILE:path>>> content <<<END_FILE>>>
        const filePattern = /<<<(CREATE_FILE|UPDATE_FILE):(.*?)>>>([\s\S]*?)<<<END_FILE>>>/g;
        let match;

        while ((match = filePattern.exec(content)) !== null) {
            const operation = match[1];
            const filePath = match[2].trim();
            const fileContent = match[3].trim();

            try {
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
                if (!workspaceFolder) {
                    vscode.window.showErrorMessage('No workspace folder open');
                    continue;
                }

                const activeEditor = vscode.window.activeTextEditor;
                let fullPath: vscode.Uri;

                if (operation === 'UPDATE_FILE' && activeEditor) {
                    if (!filePath || filePath === 'current' || filePath === 'this') {
                        fullPath = activeEditor.document.uri;
                    } else if (!filePath.includes('/')) {
                        const activeDir = vscode.Uri.joinPath(activeEditor.document.uri, '..');
                        const pathParts = filePath.split('.');
                        if (pathParts.length > 1) {
                            const ext = pathParts.pop();
                            pathParts[pathParts.length - 1] += '_adam';
                            pathParts.push(ext!);
                        } else {
                            pathParts[0] += '_adam';
                        }
                        fullPath = vscode.Uri.joinPath(activeDir, pathParts.join('.'));
                    } else {
                        fullPath = vscode.Uri.joinPath(workspaceFolder.uri, filePath);
                    }
                } else {
                    fullPath = vscode.Uri.joinPath(workspaceFolder.uri, filePath);

                    if (operation === 'CREATE_FILE') {
                        const dirPath = vscode.Uri.joinPath(fullPath, '..');
                        await vscode.workspace.fs.createDirectory(dirPath);
                    }
                }

                const encoder = new TextEncoder();
                await vscode.workspace.fs.writeFile(fullPath, encoder.encode(fileContent));

                const action = operation === 'UPDATE_FILE' ? 'Updated' : 'Created';
                vscode.window.showInformationMessage(`${action} file: ${fullPath.fsPath}`);
                const doc = await vscode.workspace.openTextDocument(fullPath);
                await vscode.window.showTextDocument(doc);

            } catch (error) {
                const action = operation === 'UPDATE_FILE' ? 'update' : 'create';
                vscode.window.showErrorMessage(`Failed to ${action} file ${filePath}: ${error}`);
            }
        }
    }

    private updateWebview() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'messages',
                messages: this.messages
            });
        }
    }

    private exportConversation() {
        const content = this.messages.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n\n');
        vscode.env.clipboard.writeText(content);
        vscode.window.showInformationMessage('Conversation copied to clipboard');
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.css'));
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'chat.js'));

        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="${styleUri}" rel="stylesheet">
            <title>ADAM Chat</title>
        </head>
        <body>
            <div class="chat-container">
                <div class="chat-header">
                    <h3>ADAM Assistant</h3>
                    <div class="header-actions">
                        <select id="styleSelect" title="Response style">
                            <option value="normal">Normal</option>
                            <option value="concise">Concise</option>
                            <option value="explanatory">Explanatory</option>
                        </select>
                        <button id="clearBtn" title="Clear chat">Clear</button>
                        <button id="exportBtn" title="Export conversation">Export</button>
                    </div>
                </div>

                <div id="messages" class="messages-container"></div>

                <div id="typingIndicator" class="typing-indicator" style="display: none;">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>

                <div class="input-container">
                    <textarea
                        id="messageInput"
                        placeholder="Ask ADAM anything... (Shift+Enter for new line)"
                        rows="3"
                    ></textarea>
                    <button id="sendBtn">Send</button>
                </div>
            </div>

            <script src="${scriptUri}"></script>
        </body>
        </html>`;
    }
}

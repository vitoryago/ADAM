import * as vscode from 'vscode';
import axios from 'axios';
import { FileSystemTool } from '../tools/fileSystemTool';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    model?: string;
    cost?: number;
}

export class SimpleChatProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'adam.chatView';
    private _view?: vscode.WebviewView;
    private messages: Message[] = [];
    private backendUrl = 'http://localhost:8000';
    private projectId = '3a859e97-16fd-46c6-b018-1ede9fade704';
    private conversationId?: string;
    private fileSystemTool: FileSystemTool;
    
    constructor(
        private readonly _extensionUri: vscode.Uri
    ) {
        this.fileSystemTool = new FileSystemTool();
    }
    
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
                    this.conversationId = undefined;
                    this.updateWebview();
                    break;
                case 'ready':
                    this.updateWebview();
                    break;
            }
        });
        
        // Welcome message
        if (this.messages.length === 0) {
            this.messages = [{
                role: 'assistant',
                content: '👋 Hello! I\'m ADAM, your AI assistant. How can I help you?'
            }];
        }
        
        this.updateWebview();
    }
    
    private async handleUserMessage(content: string) {
        // Add user message
        this.addMessage({ role: 'user', content });
        
        // Show typing
        this._view?.webview.postMessage({ type: 'typing', isTyping: true });
        
        try {
            // Initialize conversation if needed
            if (!this.conversationId) {
                const convResponse = await axios.post(
                    `${this.backendUrl}/api/projects/${this.projectId}/conversations`,
                    {
                        title: `VSCode - ${new Date().toLocaleString()}`
                    }
                );
                this.conversationId = convResponse.data.id;
            }
            
            // Always send workspace context for intelligent routing
            const workspaceContext: any = {
                workspace: this.fileSystemTool.getWorkspaceInfo(),
                activeFile: await this.fileSystemTool.getActiveContext()
            };
            
            // Send message to conversation with workspace context
            const response = await axios.post(
                `${this.backendUrl}/api/conversations/${this.conversationId}/messages`,
                {
                    content,
                    use_memory: true,
                    use_rag: true,
                    workspace_context: workspaceContext
                },
                {
                    timeout: 60000
                }
            );
            
            // Handle response - backend returns array with both user and assistant messages
            if (Array.isArray(response.data)) {
                // Find the assistant message in the response
                const assistantMsg = response.data.find((msg: any) => msg.role === 'assistant');
                if (assistantMsg) {
                    this.addMessage({
                        role: 'assistant',
                        content: assistantMsg.content || 'Response received',
                        model: assistantMsg.model,
                        cost: assistantMsg.cost
                    });
                }
            } else {
                // Single message response
                this.addMessage({
                    role: 'assistant',
                    content: response.data.content || response.data.message || 'Response received',
                    model: response.data.model,
                    cost: response.data.cost
                });
            }
            
        } catch (error: any) {
            console.error('Backend error:', error);
            
            // Show user-friendly error
            let errorMessage = 'Sorry, I encountered an error. ';
            
            if (error.code === 'ECONNREFUSED') {
                errorMessage += 'The backend is not running. Please start it with:\\n\\ncd /Users/vitoryago/ADAM/src/adam_v2\\npython main.py';
            } else if (error.response?.status === 404) {
                errorMessage += 'API endpoint not found. Please check the backend is up to date.';
            } else {
                errorMessage += error.message || 'Unknown error occurred.';
            }
            
            this.addMessage({
                role: 'assistant',
                content: errorMessage
            });
        } finally {
            this._view?.webview.postMessage({ type: 'typing', isTyping: false });
        }
    }
    
    private addMessage(message: Message) {
        this.messages.push(message);
        this.updateWebview();
    }
    
    private updateWebview() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'messages',
                messages: this.messages
            });
        }
    }
    
    public async show() {
        if (this._view) {
            this._view.show(true);
        }
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
                    <h3>🧠 ADAM Assistant</h3>
                    <div class="header-actions">
                        <button id="clearBtn" title="Clear chat">🗑️</button>
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
                        placeholder="Ask ADAM anything..."
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
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
            // Check if user is asking to read a file
            const readPattern = /(?:read|show|display|explain|analyze|review|look at|open)\s+(?:the\s+)?(?:file\s+)?([^\s]+\.(?:sql|py|ts|js|json|md|yaml|yml|txt|csv))/i;
            const fileMatch = content.match(readPattern);
            
            if (fileMatch) {
                const requestedFile = fileMatch[1];
                console.log('User requested file:', requestedFile);
                
                try {
                    // First try to find the file in the workspace
                    let filePath: string | undefined;
                    
                    // If it's an absolute path, use it directly
                    if (requestedFile.startsWith('/')) {
                        filePath = requestedFile;
                    } else {
                        // Search for the file in the workspace
                        const files = await vscode.workspace.findFiles(`**/${requestedFile}`, '**/node_modules/**', 10);
                        if (files.length > 0) {
                            filePath = files[0].fsPath;
                            console.log('Found file in workspace:', filePath);
                            
                            // If multiple matches, let user know
                            if (files.length > 1) {
                                console.log(`Found ${files.length} matches, using first: ${filePath}`);
                            }
                        }
                    }
                    
                    if (!filePath) {
                        // File not found, send message to ADAM to handle
                        const response = await this.adamClient.sendMessage(
                            `${content}\n\n[Note: File '${requestedFile}' not found in workspace. Please provide the full path or ensure the file exists.]`
                        );
                        this.addMessage(response);
                        return;
                    }
                    
                    // Read the file
                    const fileContent = await vscode.workspace.fs.readFile(vscode.Uri.file(filePath));
                    const fileText = Buffer.from(fileContent).toString('utf8');
                    const fileName = filePath.split('/').pop();
                    const extension = fileName?.split('.').pop() || 'txt';
                    
                    // Enhance the message with file content
                    const enhancedContent = `${content}\n\nFile path: ${filePath}\nFile content of ${fileName}:\n\`\`\`${extension}\n${fileText}\n\`\`\``;
                    
                    console.log('File read successfully, sending to ADAM with content');
                    
                    // Send to ADAM with file content
                    const response = await this.adamClient.sendMessage(enhancedContent);
                    
                    // Add assistant response
                    this.addMessage(response);
                } catch (fileError) {
                    console.error('Failed to read file:', fileError);
                    // If file reading fails, send error info to ADAM
                    const response = await this.adamClient.sendMessage(
                        `${content}\n\n[Error reading file: ${fileError}]`
                    );
                    this.addMessage(response);
                }
            } else {
                // Send to ADAM normally
                const response = await this.adamClient.sendMessage(content);
                
                // Add assistant response
                this.addMessage(response);
            }
        } catch (error) {
            this.addMessage({
                role: 'assistant',
                content: `Error: ${error}`
            });
        } finally {
            this._view?.webview.postMessage({ type: 'typing', isTyping: false });
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
                    <h3>🧠 ADAM Assistant</h3>
                    <div class="header-actions">
                        <button id="clearBtn" title="Clear chat">🗑️</button>
                        <button id="exportBtn" title="Export conversation">📋</button>
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
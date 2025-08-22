import * as vscode from 'vscode';
import { ADAMClient, Message } from '../client/adamClient';
import { ADAMToolIntegration } from '../tools/toolIntegration';

export class ADAMChatProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'adam.chatView';
    private _view?: vscode.WebviewView;
    private messages: Message[] = [];
    private toolIntegration: ADAMToolIntegration;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly adamClient: ADAMClient
    ) {
        this.toolIntegration = new ADAMToolIntegration(adamClient);
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
            try {
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
                    case 'ready':
                        console.log('WebView ready');
                        this.updateWebview();
                        break;
                }
            } catch (error) {
                console.error('Error in WebView message handler:', error);
                this.addMessage({
                    role: 'assistant',
                    content: `Error: ${error instanceof Error ? error.message : String(error)}`
                });
            }
        });

        // Add initial welcome message
        if (this.messages.length === 0) {
            this.messages = [{
                role: 'assistant',
                content: '👋 Hello! I\'m ADAM, your AI coding assistant. How can I help you today?'
            }];
        }
        
        // Load initial messages
        this.updateWebview();
    }

    public async show() {
        if (this._view) {
            this._view.show(true);
        } else {
            // First ensure the sidebar is visible
            await vscode.commands.executeCommand('workbench.view.extension.adam-container');
            // Then focus on the chat view
            await vscode.commands.executeCommand('adam.chatView.focus');
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
            // Check if user is asking about a file - improved pattern
            // Look for filenames with extensions anywhere in the message
            const filePatterns = [
                // Original pattern for explicit requests
                /(?:read|show|display|explain|analyze|review|look at|open|check|what is|tell me about)\s+(?:.*?\s+)?([^\s]+\.(?:sql|py|ts|js|jsx|tsx|json|md|yaml|yml|txt|csv|xml|html|css|java|go|rs|cpp|c|h|hpp))/i,
                // Pattern for "this file/model: filename.ext"
                /(?:this\s+(?:file|model|script|code|query|function|class))[\s:]+([^\s]+\.(?:sql|py|ts|js|jsx|tsx|json|md|yaml|yml|txt|csv|xml|html|css|java|go|rs|cpp|c|h|hpp))/i,
                // Pattern for standalone filenames with common extensions
                /\b([^\s\/]+\.(?:sql|py|ts|js|jsx|tsx|json|md|yaml|yml|txt|csv|xml|html|css|java|go|rs|cpp|c|h|hpp))\b/i
            ];
            
            let fileMatch = null;
            for (const pattern of filePatterns) {
                fileMatch = content.match(pattern);
                if (fileMatch) break;
            }
            
            if (fileMatch) {
                const requestedFile = fileMatch[1];
                console.log('File detection triggered for:', requestedFile);
                
                try {
                    // First try to find the file in the workspace
                    let filePath: string | undefined;
                    
                    // If it's an absolute path, use it directly
                    if (requestedFile.startsWith('/')) {
                        filePath = requestedFile;
                        console.log('Using absolute path:', filePath);
                    } else {
                        // Search for the file in the workspace
                        console.log('Searching workspace for:', requestedFile);
                        const files = await vscode.workspace.findFiles(`**/${requestedFile}`, '**/node_modules/**', 10);
                        
                        if (files.length > 0) {
                            filePath = files[0].fsPath;
                            console.log('Found file in workspace:', filePath);
                            
                            // If multiple matches, let user know
                            if (files.length > 1) {
                                console.log(`Found ${files.length} matches, using first: ${filePath}`);
                                const allPaths = files.map(f => f.fsPath).join('\n  - ');
                                console.log('All matches:\n  - ' + allPaths);
                            }
                        } else {
                            console.log('File not found in workspace');
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
                    
                    // Use the tool integration to read the file properly
                    const toolResult = await this.toolIntegration.executeTool({
                        tool: 'read_file',
                        params: { file_path: filePath }
                    });
                    
                    if (toolResult.status === 'success') {
                        // Show user that we found and are reading the file
                        this.addMessage({
                            role: 'assistant',
                            content: `📂 ${toolResult.message}`
                        });
                        
                        // Get file info for context
                        const fileName = filePath.split('/').pop();
                        const extension = fileName?.split('.').pop() || 'txt';
                        
                        // Enhance the message with the numbered file content from tool
                        const enhancedContent = `${content}\n\nFile path: ${filePath}\nFile content of ${fileName}:\n\`\`\`${extension}\n${toolResult.data}\n\`\`\``;
                        
                        console.log(`File read successfully via tool integration`);
                        
                        // Send to ADAM with file content
                        console.log('Sending enhanced content to ADAM, length:', enhancedContent.length);
                        const response = await this.adamClient.sendMessage(enhancedContent);
                        console.log('Received response from ADAM:', response?.content?.substring(0, 100));
                        
                        // Add assistant response
                        if (response && response.content) {
                            this.addMessage(response);
                        } else {
                            console.error('Empty response from ADAM');
                            this.addMessage({
                                role: 'assistant',
                                content: 'I received the file but encountered an issue processing it. The file has been loaded successfully. How can I help you with this file?'
                            });
                        }
                    } else {
                        // Tool failed to read file
                        const response = await this.adamClient.sendMessage(
                            `${content}\n\n[Error: ${toolResult.message}]`
                        );
                        this.addMessage(response);
                    }
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
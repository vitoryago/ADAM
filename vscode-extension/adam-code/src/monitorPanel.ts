import * as vscode from 'vscode';
import axios from 'axios';

export class MonitorPanel {
    private static currentPanel: MonitorPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private readonly extensionUri: vscode.Uri;
    private updateInterval: NodeJS.Timeout | undefined;

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this.panel = panel;
        this.extensionUri = extensionUri;

        // Set up the webview
        this.update();

        // Handle messages from the webview
        this.panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.command) {
                    case 'refresh':
                        await this.refreshStatus();
                        break;
                    case 'clearLogs':
                        await this.clearLogs();
                        break;
                }
            }
        );

        // Start auto-refresh
        this.startAutoRefresh();

        // Clean up on dispose
        this.panel.onDidDispose(() => this.dispose());
    }

    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.ViewColumn.Two;

        // If we already have a panel, show it
        if (MonitorPanel.currentPanel) {
            MonitorPanel.currentPanel.panel.reveal(column);
            return;
        }

        // Create a new panel
        const panel = vscode.window.createWebviewPanel(
            'adamMonitor',
            'ADAM Agent Monitor',
            column,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        MonitorPanel.currentPanel = new MonitorPanel(panel, extensionUri);
    }

    private async update() {
        this.panel.webview.html = this.getHtmlContent();
        await this.refreshStatus();
    }

    private async refreshStatus() {
        try {
            const response = await axios.get('http://localhost:8000/api/monitor/agent/status');
            
            // Send status to webview
            this.panel.webview.postMessage({
                type: 'status',
                data: response.data
            });

            // Also fetch recent logs
            const logsResponse = await axios.get('http://localhost:8000/api/monitor/agent/logs', {
                params: { limit: 50 }
            }).catch(() => ({ data: { logs: [] } }));

            this.panel.webview.postMessage({
                type: 'logs',
                data: logsResponse.data
            });

        } catch (error) {
            console.error('Failed to refresh monitor status:', error);
            this.panel.webview.postMessage({
                type: 'error',
                message: 'Failed to connect to backend'
            });
        }
    }

    private async clearLogs() {
        this.panel.webview.postMessage({
            type: 'clearLogs'
        });
    }

    private startAutoRefresh() {
        this.updateInterval = setInterval(() => {
            this.refreshStatus();
        }, 2000); // Refresh every 2 seconds
    }

    private dispose() {
        MonitorPanel.currentPanel = undefined;

        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.panel.dispose();
    }

    private getHtmlContent(): string {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ADAM Agent Monitor</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    background-color: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                    padding: 10px;
                    margin: 0;
                }
                
                h2 {
                    color: var(--vscode-textLink-activeForeground);
                    border-bottom: 1px solid var(--vscode-panel-border);
                    padding-bottom: 5px;
                }
                
                .status-section {
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border: 1px solid var(--vscode-panel-border);
                    border-radius: 4px;
                    padding: 10px;
                    margin: 10px 0;
                }
                
                .status-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 5px 0;
                }
                
                .status-value {
                    font-weight: bold;
                }
                
                .status-value.active {
                    color: #4EC9B0;
                }
                
                .status-value.inactive {
                    color: #F48771;
                }
                
                .task-list {
                    max-height: 200px;
                    overflow-y: auto;
                    background: var(--vscode-editor-inactiveSelectionBackground);
                    border: 1px solid var(--vscode-panel-border);
                    border-radius: 4px;
                    padding: 10px;
                    margin: 10px 0;
                }
                
                .task-item {
                    background: var(--vscode-editor-selectionBackground);
                    border-left: 3px solid var(--vscode-progressBar-background);
                    padding: 8px;
                    margin: 5px 0;
                    border-radius: 2px;
                }
                
                .task-item.running {
                    border-left-color: #CCCC00;
                }
                
                .task-item.completed {
                    border-left-color: #4EC9B0;
                }
                
                .task-item.failed {
                    border-left-color: #F48771;
                }
                
                .log-section {
                    background: var(--vscode-terminal-background);
                    border: 1px solid var(--vscode-panel-border);
                    border-radius: 4px;
                    padding: 10px;
                    margin: 10px 0;
                    max-height: 300px;
                    overflow-y: auto;
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                }
                
                .log-entry {
                    margin: 2px 0;
                    white-space: pre-wrap;
                }
                
                .log-entry.info { color: #4EC9B0; }
                .log-entry.warning { color: #CCCC00; }
                .log-entry.error { color: #F48771; }
                
                button {
                    background: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    padding: 6px 14px;
                    cursor: pointer;
                    margin: 5px;
                    border-radius: 2px;
                }
                
                button:hover {
                    background: var(--vscode-button-hoverBackground);
                }
                
                .indicator {
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    margin-right: 5px;
                }
                
                .indicator.active {
                    background: #4EC9B0;
                    animation: pulse 2s infinite;
                }
                
                .indicator.inactive {
                    background: #F48771;
                }
                
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
            </style>
        </head>
        <body>
            <h2>🤖 ADAM Agent Monitor</h2>
            
            <div class="status-section">
                <div class="status-item">
                    <span>Status:</span>
                    <span class="status-value" id="status">
                        <span class="indicator inactive" id="indicator"></span>
                        <span id="status-text">Connecting...</span>
                    </span>
                </div>
                <div class="status-item">
                    <span>Runtime:</span>
                    <span class="status-value" id="runtime">Unknown</span>
                </div>
                <div class="status-item">
                    <span>Queue Size:</span>
                    <span class="status-value" id="queue">0</span>
                </div>
                <div class="status-item">
                    <span>Active Tasks:</span>
                    <span class="status-value" id="active">0</span>
                </div>
            </div>
            
            <h3>📋 Recent Tasks</h3>
            <div class="task-list" id="task-list">
                <div style="color: var(--vscode-descriptionForeground);">No tasks yet...</div>
            </div>
            
            <h3>📝 Agent Logs</h3>
            <div class="log-section" id="log-section">
                <div class="log-entry info">Waiting for logs...</div>
            </div>
            
            <div>
                <button onclick="refresh()">🔄 Refresh</button>
                <button onclick="clearLogs()">🗑️ Clear Logs</button>
            </div>
            
            <script>
                const vscode = acquireVsCodeApi();
                
                function refresh() {
                    vscode.postMessage({ command: 'refresh' });
                }
                
                function clearLogs() {
                    document.getElementById('log-section').innerHTML = '';
                    vscode.postMessage({ command: 'clearLogs' });
                }
                
                // Handle messages from extension
                window.addEventListener('message', event => {
                    const message = event.data;
                    
                    switch (message.type) {
                        case 'status':
                            updateStatus(message.data);
                            break;
                        case 'logs':
                            updateLogs(message.data);
                            break;
                        case 'error':
                            showError(message.message);
                            break;
                        case 'clearLogs':
                            document.getElementById('log-section').innerHTML = '';
                            break;
                    }
                });
                
                function updateStatus(data) {
                    const isActive = data.runtime_available;
                    
                    // Update indicator
                    const indicator = document.getElementById('indicator');
                    indicator.className = isActive ? 'indicator active' : 'indicator inactive';
                    
                    document.getElementById('status-text').textContent = data.status || 'Unknown';
                    document.getElementById('runtime').textContent = isActive ? 'Active' : 'Inactive';
                    document.getElementById('runtime').className = isActive ? 'status-value active' : 'status-value inactive';
                    document.getElementById('queue').textContent = data.queue_size || 0;
                    
                    // Update task list
                    if (data.tasks && data.tasks.length > 0) {
                        const taskList = document.getElementById('task-list');
                        taskList.innerHTML = data.tasks.map(task => 
                            '<div class="task-item ' + task.status + '">' +
                            '<strong>Task ' + (task.id || '').substring(0, 8) + '</strong><br>' +
                            'Status: ' + task.status + '<br>' +
                            'Request: ' + (task.request || '').substring(0, 50) + '...' +
                            '</div>'
                        ).join('');
                    }
                }
                
                function updateLogs(data) {
                    if (data.logs && data.logs.length > 0) {
                        const logSection = document.getElementById('log-section');
                        const newLogs = data.logs.map(log => 
                            '<div class="log-entry ' + (log.level || 'info').toLowerCase() + '">' +
                            '[' + (log.timestamp || new Date().toISOString()).substring(11, 19) + '] ' +
                            log.message +
                            '</div>'
                        ).join('');
                        
                        logSection.innerHTML = newLogs + logSection.innerHTML;
                        
                        // Keep only last 100 logs
                        const entries = logSection.getElementsByClassName('log-entry');
                        while (entries.length > 100) {
                            entries[entries.length - 1].remove();
                        }
                    }
                }
                
                function showError(message) {
                    const logSection = document.getElementById('log-section');
                    logSection.innerHTML = '<div class="log-entry error">❌ ' + message + '</div>' + logSection.innerHTML;
                }
            </script>
        </body>
        </html>`;
    }
}
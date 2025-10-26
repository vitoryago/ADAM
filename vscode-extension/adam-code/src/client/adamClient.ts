import axios, { AxiosInstance } from 'axios';
import * as vscode from 'vscode';
// WebSocket will be added when backend supports it

export interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    model?: string;
    cost?: number;
}

export interface CodeExplanation {
    explanation: string;
    complexity: string;
    suggestions: string[];
}

export interface SQLOptimization {
    query: string;
    explanation: string;
    improvements: string;
    estimatedSpeedup: string;
}

export interface PRDetails {
    title: string;
    body: string;
    labels: string[];
}

export interface DBTModel {
    sql: string;
    documentation: string;
    tests: string[];
}

export class ADAMClient {
    private api: AxiosInstance;
    private ws: any | null = null;  // Will be WebSocket when implemented
    private projectId: string;
    private conversationId: string | null = null;
    private responseStyle: 'normal' | 'concise' | 'explanatory' = 'normal';

    constructor(baseURL: string, projectId: string) {
        this.api = axios.create({
            baseURL,
            headers: {
                'Content-Type': 'application/json'
            }
        });
        this.projectId = projectId;
        this.connectWebSocket(baseURL);
    }

    private connectWebSocket(baseURL: string) {
        // Commenting out WebSocket for now since backend doesn't have /ws endpoint yet
        // We'll use regular HTTP polling instead
        console.log('WebSocket connection disabled - using HTTP instead');
        return;
        
        /* Will re-enable when backend has WebSocket support
        const wsURL = baseURL.replace('http', 'ws') + '/ws';
        try {
            this.ws = new WebSocket(wsURL);
            
            this.ws.on('open', () => {
                console.log('ADAM WebSocket connected');
                this.ws?.send(JSON.stringify({
                    type: 'init',
                    projectId: this.projectId
                }));
            });

            this.ws.on('message', (data: Buffer) => {
                const message = JSON.parse(data.toString());
                this.handleWebSocketMessage(message);
            });

            this.ws.on('error', (error: Error) => {
                console.error('ADAM WebSocket error:', error);
            });

            this.ws.on('close', () => {
                console.log('ADAM WebSocket disconnected');
                // Reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(baseURL), 5000);
            });
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
        */
    }

    private handleWebSocketMessage(message: any) {
        // Handle real-time updates from ADAM
        if (message.type === 'memory_update') {
            vscode.window.showInformationMessage(`ADAM Memory Updated: ${message.summary}`);
        } else if (message.type === 'task_complete') {
            vscode.window.showInformationMessage(`ADAM Task Complete: ${message.task}`);
        }
    }

    setResponseStyle(style: 'normal' | 'concise' | 'explanatory') {
        this.responseStyle = style;
        console.log(`Response style set to: ${style}`);
    }

    getResponseStyle(): string {
        return this.responseStyle;
    }

    async sendMessage(content: string, useMemory: boolean = true): Promise<Message> {
        try {
            // Create or get conversation
            if (!this.conversationId) {
                const convResponse = await this.api.post(`/api/projects/${this.projectId}/conversations`, {
                    title: `VSCode Session ${new Date().toLocaleString()}`  // Changed from 'name' to 'title'
                });
                this.conversationId = convResponse.data.id;
            }

            // Get workspace context for better responses
            const activeEditor = vscode.window.activeTextEditor;
            let workspaceContext: any = null;
            
            // Always include the active file content if there's an active editor
            if (activeEditor) {
                const selection = activeEditor.selection;
                const selectedText = !selection.isEmpty ? activeEditor.document.getText(selection) : null;
                
                // Get the full file content if user seems to be referring to it
                const contentLower = content.toLowerCase();
                const isReferringToFile = (
                    contentLower.includes('this') ||
                    contentLower.includes('file') ||
                    contentLower.includes('code') ||
                    contentLower.includes('show') ||
                    contentLower.includes('dependencies') ||
                    contentLower.includes('explain') ||
                    contentLower.includes('analyze') ||
                    contentLower.includes('what') ||
                    contentLower.includes('read') ||
                    contentLower.includes('review') ||
                    contentLower.includes('shared') ||
                    contentLower.includes('sharing') ||
                    contentLower.includes('update') ||
                    contentLower.includes('edit') ||
                    contentLower.includes('modify') ||
                    contentLower.includes('change') ||
                    contentLower.includes('fix') ||
                    contentLower.includes('improve') ||
                    contentLower.includes('refactor') ||
                    contentLower.includes('optimize')
                );
                
                workspaceContext = {
                    activeFile: {
                        file: activeEditor.document.fileName,
                        language: activeEditor.document.languageId,
                        content: isReferringToFile ? activeEditor.document.getText() : selectedText,
                        isUpdateRequest: contentLower.includes('update') || contentLower.includes('edit') || 
                                       contentLower.includes('modify') || contentLower.includes('change') ||
                                       contentLower.includes('fix') || contentLower.includes('improve')
                    }
                };
                
                // Log for debugging
                console.log('Active file:', activeEditor.document.fileName);
                console.log('Is referring to file:', isReferringToFile);
            }

            // Send message with response style
            const response = await this.api.post(`/api/conversations/${this.conversationId}/messages`, {
                content,
                use_memory: useMemory,
                response_style: this.responseStyle,  // Add response style
                workspace_context: workspaceContext,  // Enhanced context
                context: {
                    editor: 'vscode',
                    workspace: vscode.workspace.name || 'VSCode',
                    activeFile: vscode.window.activeTextEditor?.document.fileName || ''
                }
            });

            // Log the response for debugging
            console.log('ADAM Response:', response.data);

            // The endpoint returns an array of [user_message, assistant_message]
            if (response.data && Array.isArray(response.data)) {
                // Get the assistant message (should be the second item or last item with role='assistant')
                const assistantMsg = response.data.find((m: any) => m.role === 'assistant');
                if (assistantMsg) {
                    return {
                        role: 'assistant',
                        content: assistantMsg.content,
                        model: assistantMsg.model,
                        cost: assistantMsg.cost
                    };
                }
            }
            
            // Fallback if response format is unexpected
            return { 
                role: 'assistant', 
                content: 'Message sent but no response received. Check console for details.' 
            };
        } catch (error) {
            console.error('ADAM API error:', error);
            throw error;
        }
    }

    async explainCode(code: string, language: string, fileName: string): Promise<string> {
        const prompt = `Explain this ${language} code from ${fileName}:\n\`\`\`${language}\n${code}\n\`\`\`\n\nProvide a clear explanation of what this code does, its purpose, and any potential improvements.`;
        
        const response = await this.sendMessage(prompt);
        return response.content;
    }

    async optimizeSQL(query: string, dialect: string): Promise<SQLOptimization> {
        const prompt = `Optimize this ${dialect} SQL query for performance:\n\`\`\`sql\n${query}\n\`\`\`\n\nProvide the optimized query and explain the improvements.`;
        
        const response = await this.sendMessage(prompt);
        
        // Parse the response to extract structured data
        const lines = response.content.split('\n');
        const optimizedQuery = this.extractCodeBlock(response.content, 'sql');
        
        return {
            query: optimizedQuery || query,
            explanation: this.extractSection(response.content, 'Explanation'),
            improvements: this.extractSection(response.content, 'Improvements'),
            estimatedSpeedup: this.extractSection(response.content, 'Performance')
        };
    }

    async generateDBTModel(modelName: string, sourceTable: string): Promise<DBTModel> {
        const prompt = `Generate a dbt staging model named "${modelName}" for the source table "${sourceTable}". Include:
        1. The SQL model with proper CTEs and transformations
        2. Documentation in YAML format
        3. Basic data quality tests`;
        
        const response = await this.sendMessage(prompt);
        
        const sql = this.extractCodeBlock(response.content, 'sql') || '';
        const yaml = this.extractCodeBlock(response.content, 'yaml') || '';
        
        return {
            sql,
            documentation: yaml,
            tests: this.extractTests(response.content)
        };
    }

    async createBranch(branchName: string): Promise<{ formattedName: string; description: string }> {
        const prompt = `Format this branch name according to git best practices: "${branchName}". Also provide a brief description of what this branch is for.`;
        
        const response = await this.sendMessage(prompt, false);
        
        // Extract formatted name and description
        const formattedName = this.extractBranchName(response.content) || branchName.toLowerCase().replace(/\s+/g, '-');
        const description = this.extractDescription(response.content);
        
        return { formattedName, description };
    }

    async generatePRDetails(changes: string): Promise<PRDetails> {
        const prompt = `Based on these git changes, generate a pull request title and description:\n\`\`\`diff\n${changes}\n\`\`\``;
        
        const response = await this.sendMessage(prompt);
        
        return {
            title: this.extractTitle(response.content),
            body: this.extractBody(response.content),
            labels: this.extractLabels(response.content)
        };
    }

    async analyzeDataPattern(data: string): Promise<any> {
        const prompt = `Analyze this data for patterns, anomalies, and insights:\n\`\`\`\n${data}\n\`\`\``;

        const response = await this.sendMessage(prompt);

        return {
            summary: this.extractSection(response.content, 'Summary'),
            patterns: this.extractSection(response.content, 'Patterns'),
            recommendations: this.extractSection(response.content, 'Recommendations')
        };
    }

    // DBT Column Documentation Methods

    async documentDBTColumns(modelName: string): Promise<any> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                throw new Error('No workspace folder open');
            }

            const response = await this.api.post('/api/dbt/columns/document-model', {
                project_path: workspaceFolder.uri.fsPath,
                model_name: modelName,
                use_ai: true,
                preserve_existing: true
            });

            const data = response.data;

            // Format documentation for display
            const documentation = Object.values(data.columns)
                .map((col: any) => `- **${col.name}**: ${col.description}${col.is_foreign_key ? ` (FK → ${col.references})` : ''}`)
                .join('\n');

            return {
                documentation,
                columns_documented: data.total_columns,
                yaml_content: await this.generateYAMLForModel(modelName, data.columns)
            };
        } catch (error) {
            console.error('DBT column documentation error:', error);
            throw error;
        }
    }

    async analyzeDBTColumns(): Promise<any> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                throw new Error('No workspace folder open');
            }

            const response = await this.api.post('/api/dbt/columns/analyze', {
                project_path: workspaceFolder.uri.fsPath,
                use_ai: true
            });

            return response.data;
        } catch (error) {
            console.error('DBT column analysis error:', error);
            throw error;
        }
    }

    async findCommonDBTColumns(): Promise<any[]> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                throw new Error('No workspace folder open');
            }

            const response = await this.api.post('/api/dbt/columns/common-columns', {
                project_path: workspaceFolder.uri.fsPath,
                min_occurrences: 2
            });

            return response.data;
        } catch (error) {
            console.error('Common columns search error:', error);
            return [];
        }
    }

    async generateDBTSchema(folderPath: string): Promise<string> {
        try {
            const response = await this.api.post('/api/dbt/columns/generate-schema', {
                project_path: folderPath,
                use_ai: true,
                include_tests: true
            });

            return response.data;
        } catch (error) {
            console.error('Schema generation error:', error);
            throw error;
        }
    }

    async standardizeColumnDescription(columnName: string, suggestedDescription: string): Promise<void> {
        // This would update all occurrences of the column with the standardized description
        // For now, we'll just log it
        console.log(`Standardizing ${columnName} with description: ${suggestedDescription}`);
        // In a real implementation, this would call an API to update all schema.yml files
    }

    async saveDBTDocumentation(modelName: string, yamlContent: string): Promise<void> {
        // This would save the documentation to the appropriate schema.yml file
        // For now, we'll create or update a schema.yml in the model's directory
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder open');
        }

        // Find the model file
        const modelFiles = await vscode.workspace.findFiles(`**/${modelName}.sql`);
        if (modelFiles.length === 0) {
            throw new Error(`Model ${modelName} not found`);
        }

        const modelPath = modelFiles[0];
        const schemaPath = vscode.Uri.joinPath(modelPath, '..', 'schema.yml');

        // Write the YAML content
        await vscode.workspace.fs.writeFile(schemaPath, Buffer.from(yamlContent));
    }

    private async generateYAMLForModel(modelName: string, columns: any): Promise<string> {
        const yaml = `version: 2

models:
  - name: ${modelName}
    description: "Model description here"
    columns:
${Object.values(columns).map((col: any) => `      - name: ${col.name}
        description: "${col.description}"
        tests:
          - not_null
${col.is_foreign_key ? `          - relationships:
              to: ref('${col.references?.split('.')[0] || 'parent_table'}')
              field: ${col.references?.split('.')[1] || 'id'}` : ''}`).join('\n')}`;

        return yaml;
    }

    // Memory operations
    async searchMemory(query: string): Promise<any[]> {
        try {
            const response = await this.api.get(`/api/projects/${this.projectId}/memories/search`, {
                params: { query, limit: 10 }
            });
            return response.data;
        } catch (error) {
            console.error('Memory search error:', error);
            return [];
        }
    }

    async getMemoryStats(): Promise<any> {
        try {
            const response = await this.api.get(`/api/projects/${this.projectId}/memories/stats`);
            return response.data;
        } catch (error) {
            console.error('Memory stats error:', error);
            return {};
        }
    }

    // Helper methods for parsing responses
    private extractCodeBlock(content: string, language: string): string | null {
        const regex = new RegExp(`\`\`\`${language}\\n([\\s\\S]*?)\`\`\``, 'i');
        const match = content.match(regex);
        return match ? match[1].trim() : null;
    }

    private extractSection(content: string, sectionName: string): string {
        const regex = new RegExp(`(?:#{1,3}\\s*)?${sectionName}:?\\s*\\n([\\s\\S]*?)(?=\\n#{1,3}|$)`, 'i');
        const match = content.match(regex);
        return match ? match[1].trim() : '';
    }

    private extractBranchName(content: string): string | null {
        const match = content.match(/`([a-z0-9\-\/]+)`/);
        return match ? match[1] : null;
    }

    private extractDescription(content: string): string {
        const lines = content.split('\n');
        return lines.find(line => line.length > 20 && !line.startsWith('#')) || '';
    }

    private extractTitle(content: string): string {
        const match = content.match(/(?:Title|PR Title):?\s*(.+)/i);
        return match ? match[1].trim() : 'Update';
    }

    private extractBody(content: string): string {
        const match = content.match(/(?:Description|Body|PR Description):?\s*\n([\s\S]+)/i);
        return match ? match[1].trim() : content;
    }

    private extractLabels(content: string): string[] {
        const match = content.match(/(?:Labels?):?\s*(.+)/i);
        if (match) {
            return match[1].split(',').map(l => l.trim());
        }
        return [];
    }

    private extractTests(content: string): string[] {
        const testsSection = this.extractSection(content, 'Tests');
        if (testsSection) {
            return testsSection.split('\n').filter(line => line.trim().startsWith('-')).map(line => line.replace('-', '').trim());
        }
        return [];
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}
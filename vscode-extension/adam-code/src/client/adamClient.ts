import * as vscode from 'vscode';

const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const MAX_REQUEST_BYTES = 10 * 1024 * 1024;
const MAX_RESPONSE_BYTES = 10 * 1024 * 1024;

type QueryValue = string | number | boolean | null | undefined;

interface RequestOptions {
    method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
    body?: unknown;
    query?: Record<string, QueryValue>;
    timeoutMs?: number;
}

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

/** SSE event emitted during streaming */
export interface StreamEvent {
    type: 'user_message' | 'assistant_chunk' | 'agent_status' | 'agent_chunk' | 'agent_done' | 'complete' | 'error';
    content?: string;
    agent?: string;
    status?: string;
    pattern?: string;
    cost?: number;
    tokens?: number;
    model?: string;
    id?: string;
    message?: string;
}

/** Diagnostic info from the editor */
export interface EditorDiagnostic {
    severity: 'error' | 'warning' | 'info' | 'hint';
    message: string;
    line: number;
    source?: string;
}

export class ADAMClient {
    public baseURL: string;
    private projectId: string;
    private conversationId: string | null = null;
    private responseStyle: 'normal' | 'concise' | 'explanatory' = 'normal';

    constructor(baseURL: string, projectId: string) {
        this.baseURL = this.normalizeBaseURL(baseURL);
        this.projectId = projectId;
    }

    private normalizeBaseURL(baseURL: string): string {
        try {
            const parsed = new URL(baseURL.trim());
            if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
                throw new Error(`unsupported protocol "${parsed.protocol}"`);
            }
            return parsed.href.replace(/\/+$/, '');
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            throw new Error(`Invalid ADAM server URL "${baseURL}": ${message}`);
        }
    }

    private buildRequestUrl(path: string, query?: Record<string, QueryValue>): string {
        const trimmedPath = path.trim();

        if (/^[a-zA-Z][a-zA-Z\d+.-]*:/.test(trimmedPath) || trimmedPath.startsWith('//')) {
            throw new Error(`Refusing to request an absolute URL: ${trimmedPath}`);
        }

        const normalizedPath = trimmedPath.startsWith('/') ? trimmedPath : `/${trimmedPath}`;
        const url = new URL(`${this.baseURL}${normalizedPath}`);

        if (url.protocol !== 'http:' && url.protocol !== 'https:') {
            throw new Error(`Refusing to request unsupported protocol "${url.protocol}"`);
        }

        for (const [key, value] of Object.entries(query || {})) {
            if (value === undefined || value === null) {
                continue;
            }
            url.searchParams.set(key, String(value));
        }

        return url.toString();
    }

    private buildErrorMessage(status: number, text: string): string {
        if (!text) {
            return `Request failed with status ${status}`;
        }

        try {
            const parsed = JSON.parse(text) as { detail?: unknown; error?: unknown; message?: unknown };
            const detail = parsed.detail ?? parsed.error ?? parsed.message;
            if (typeof detail === 'string' && detail.trim()) {
                return detail;
            }
        } catch {
            // Fall back to raw text below.
        }

        return text.length > 500 ? `${text.slice(0, 500)}...` : text;
    }

    private encodeRequestBody(body: unknown): string | undefined {
        if (body === undefined) {
            return undefined;
        }

        const json = JSON.stringify(body);
        if (Buffer.byteLength(json, 'utf8') > MAX_REQUEST_BYTES) {
            throw new Error(`Request body exceeds ${MAX_REQUEST_BYTES} bytes`);
        }

        return json;
    }

    private async sendRequest(path: string, options: RequestOptions): Promise<string> {
        const url = this.buildRequestUrl(path, options.query);
        const body = this.encodeRequestBody(options.body);
        const timeoutMs = options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const response = await fetch(url, {
                method: options.method,
                headers: {
                    Accept: 'application/json, text/plain, */*',
                    ...(body ? { 'Content-Type': 'application/json' } : {})
                },
                body,
                redirect: 'error',
                signal: controller.signal
            });

            const contentLength = response.headers.get('content-length');
            if (contentLength && Number(contentLength) > MAX_RESPONSE_BYTES) {
                throw new Error(`Response exceeds ${MAX_RESPONSE_BYTES} bytes`);
            }

            const text = await response.text();
            if (Buffer.byteLength(text, 'utf8') > MAX_RESPONSE_BYTES) {
                throw new Error(`Response exceeds ${MAX_RESPONSE_BYTES} bytes`);
            }

            if (!response.ok) {
                throw new Error(
                    `${options.method} ${url} failed (${response.status}): ${this.buildErrorMessage(response.status, text)}`
                );
            }

            return text;
        } catch (error) {
            if (controller.signal.aborted) {
                throw new Error(`Request timed out after ${timeoutMs}ms: ${options.method} ${url}`);
            }
            throw error;
        } finally {
            clearTimeout(timeoutId);
        }
    }

    private async requestJson<T>(path: string, options: RequestOptions): Promise<T> {
        const text = await this.sendRequest(path, options);
        if (!text) {
            return undefined as T;
        }

        try {
            return JSON.parse(text) as T;
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            throw new Error(`Invalid JSON response for ${options.method} ${path}: ${message}`);
        }
    }

    private async requestText(path: string, options: RequestOptions): Promise<string> {
        return this.sendRequest(path, options);
    }

    setResponseStyle(style: 'normal' | 'concise' | 'explanatory') {
        this.responseStyle = style;
        console.log(`Response style set to: ${style}`);
    }

    getResponseStyle(): string {
        return this.responseStyle;
    }

    /** Ensure a conversation exists, creating one if needed */
    private async ensureConversation(): Promise<string> {
        if (!this.conversationId) {
            const convResponse = await this.requestJson<{ id: string }>(
                `/api/projects/${this.projectId}/conversations`,
                {
                    method: 'POST',
                    body: {
                        title: `VSCode Session ${new Date().toLocaleString()}`
                    }
                }
            );
            this.conversationId = convResponse.id;
        }
        return this.conversationId!;
    }

    /** Build workspace context from the active editor, including diagnostics */
    getWorkspaceContext(content: string): any {
        const activeEditor = vscode.window.activeTextEditor;
        let workspaceContext: any = null;

        if (activeEditor) {
            const selection = activeEditor.selection;
            const selectedText = !selection.isEmpty ? activeEditor.document.getText(selection) : null;

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
                },
                diagnostics: this.collectDiagnostics(activeEditor.document.uri),
                terminalErrors: []
            };

            console.log('Active file:', activeEditor.document.fileName);
            console.log('Is referring to file:', isReferringToFile);
            console.log('Diagnostics count:', workspaceContext.diagnostics.length);
        }

        return workspaceContext;
    }

    /** Collect VS Code diagnostics (errors, warnings) for a given document */
    collectDiagnostics(uri: vscode.Uri): EditorDiagnostic[] {
        const diagnostics = vscode.languages.getDiagnostics(uri);
        return diagnostics.map(d => ({
            severity: d.severity === vscode.DiagnosticSeverity.Error ? 'error'
                : d.severity === vscode.DiagnosticSeverity.Warning ? 'warning'
                : d.severity === vscode.DiagnosticSeverity.Information ? 'info'
                : 'hint',
            message: d.message,
            line: d.range.start.line + 1, // 1-based for display
            source: d.source
        }));
    }

    /** Build the message payload used by both sync and streaming endpoints */
    private buildMessagePayload(content: string, useMemory: boolean): any {
        const workspaceContext = this.getWorkspaceContext(content);
        return {
            content,
            use_memory: useMemory,
            response_style: this.responseStyle,
            workspace_context: workspaceContext,
            context: {
                editor: 'vscode',
                workspace: vscode.workspace.name || 'VSCode',
                activeFile: vscode.window.activeTextEditor?.document.fileName || ''
            }
        };
    }

    /** Send a message synchronously (non-streaming fallback) */
    async sendMessage(content: string, useMemory: boolean = true): Promise<Message> {
        try {
            const conversationId = await this.ensureConversation();
            const payload = this.buildMessagePayload(content, useMemory);

            const response = await this.requestJson<any[]>(
                `/api/conversations/${conversationId}/messages`,
                {
                    method: 'POST',
                    body: payload
                }
            );

            console.log('ADAM Response:', response);

            // The endpoint returns an array of [user_message, assistant_message]
            if (response && Array.isArray(response)) {
                const assistantMsg = response.find((m: any) => m.role === 'assistant');
                if (assistantMsg) {
                    return {
                        role: 'assistant',
                        content: assistantMsg.content,
                        model: assistantMsg.model,
                        cost: assistantMsg.cost
                    };
                }
            }

            return {
                role: 'assistant',
                content: 'Message sent but no response received. Check console for details.'
            };
        } catch (error) {
            console.error('ADAM API error:', error);
            throw error;
        }
    }

    /**
     * Send a message with SSE streaming.
     *
     * Uses fetch() so we can access ReadableStream directly for incremental
     * SSE parsing without pulling in an extra HTTP client dependency. Calls
     * onEvent() for each parsed SSE event, letting the caller update the UI
     * progressively.
     *
     * Returns the final accumulated assistant content and metadata.
     */
    async sendMessageStream(
        content: string,
        callbacks: {
            onChunk: (text: string) => void;
            onAgentStatus?: (agent: string, status: string, pattern?: string) => void;
            onAgentChunk?: (agent: string, content: string) => void;
            onAgentDone?: (agent: string, cost: number) => void;
            onComplete?: (id: string, tokens: number, cost: number, model: string) => void;
            onError?: (message: string) => void;
        },
        useMemory: boolean = true
    ): Promise<Message> {
        const conversationId = await this.ensureConversation();
        const payload = this.buildMessagePayload(content, useMemory);
        const url = `${this.baseURL}/api/conversations/${conversationId}/messages/stream`;

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Stream request failed (${response.status}): ${errorText}`);
        }

        if (!response.body) {
            throw new Error('Response body is null -- streaming not supported');
        }

        // Read SSE events from the response stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = '';
        let finalModel = '';
        let finalCost = 0;
        let finalTokens = 0;
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) { break; }

                buffer += decoder.decode(value, { stream: true });

                // Parse SSE lines: each event is "data: <json>\n\n"
                const lines = buffer.split('\n');
                buffer = '';

                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];

                    // If this line doesn't end with \n and is the last line,
                    // it might be an incomplete chunk -- keep in buffer
                    if (i === lines.length - 1 && line !== '') {
                        buffer = line;
                        continue;
                    }

                    if (!line.startsWith('data: ')) { continue; }

                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr || jsonStr === '[DONE]') { continue; }

                    try {
                        const event: StreamEvent = JSON.parse(jsonStr);

                        switch (event.type) {
                            case 'assistant_chunk':
                                if (event.content) {
                                    fullContent += event.content;
                                    callbacks.onChunk(event.content);
                                }
                                break;

                            case 'agent_status':
                                callbacks.onAgentStatus?.(
                                    event.agent || 'unknown',
                                    event.status || 'thinking',
                                    event.pattern
                                );
                                break;

                            case 'agent_chunk':
                                callbacks.onAgentChunk?.(
                                    event.agent || 'unknown',
                                    event.content || ''
                                );
                                break;

                            case 'agent_done':
                                callbacks.onAgentDone?.(
                                    event.agent || 'unknown',
                                    event.cost || 0
                                );
                                break;

                            case 'complete':
                                finalModel = event.model || '';
                                finalCost = event.cost || 0;
                                finalTokens = event.tokens || 0;
                                callbacks.onComplete?.(
                                    event.id || '',
                                    finalTokens,
                                    finalCost,
                                    finalModel
                                );
                                break;

                            case 'error':
                                callbacks.onError?.(event.message || 'Unknown stream error');
                                break;

                            case 'user_message':
                                // Acknowledgment of user message -- no UI action needed
                                break;
                        }
                    } catch (parseErr) {
                        console.warn('Failed to parse SSE event:', jsonStr, parseErr);
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }

        return {
            role: 'assistant',
            content: fullContent,
            model: finalModel,
            cost: finalCost
        };
    }

    async explainCode(code: string, language: string, fileName: string): Promise<string> {
        const prompt = `Explain this ${language} code from ${fileName}:\n\`\`\`${language}\n${code}\n\`\`\`\n\nProvide a clear explanation of what this code does, its purpose, and any potential improvements.`;

        const response = await this.sendMessage(prompt);
        return response.content;
    }

    async optimizeSQL(query: string, dialect: string): Promise<SQLOptimization> {
        const prompt = `Optimize this ${dialect} SQL query for performance:\n\`\`\`sql\n${query}\n\`\`\`\n\nProvide the optimized query and explain the improvements.`;

        const response = await this.sendMessage(prompt);

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

            const data = await this.requestJson<any>('/api/dbt/columns/document-model', {
                method: 'POST',
                body: {
                    project_path: workspaceFolder.uri.fsPath,
                    model_name: modelName,
                    use_ai: true,
                    preserve_existing: true
                }
            });

            const documentation = Object.values(data.columns)
                .map((col: any) => `- **${col.name}**: ${col.description}${col.is_foreign_key ? ` (FK -> ${col.references})` : ''}`)
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

            return await this.requestJson<any>('/api/dbt/columns/analyze', {
                method: 'POST',
                body: {
                    project_path: workspaceFolder.uri.fsPath,
                    use_ai: true
                }
            });
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

            return await this.requestJson<any[]>('/api/dbt/columns/common-columns', {
                method: 'POST',
                body: {
                    project_path: workspaceFolder.uri.fsPath,
                    min_occurrences: 2
                }
            });
        } catch (error) {
            console.error('Common columns search error:', error);
            return [];
        }
    }

    async generateDBTSchema(folderPath: string): Promise<string> {
        try {
            return await this.requestText('/api/dbt/columns/generate-schema', {
                method: 'POST',
                body: {
                    project_path: folderPath,
                    use_ai: true,
                    include_tests: true
                }
            });
        } catch (error) {
            console.error('Schema generation error:', error);
            throw error;
        }
    }

    async standardizeColumnDescription(columnName: string, suggestedDescription: string): Promise<void> {
        console.log(`Standardizing ${columnName} with description: ${suggestedDescription}`);
    }

    async saveDBTDocumentation(modelName: string, yamlContent: string): Promise<void> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder open');
        }

        const modelFiles = await vscode.workspace.findFiles(`**/${modelName}.sql`);
        if (modelFiles.length === 0) {
            throw new Error(`Model ${modelName} not found`);
        }

        const modelPath = modelFiles[0];
        const schemaPath = vscode.Uri.joinPath(modelPath, '..', 'schema.yml');

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

    // Memory operations -- fixed paths to match backend convenience endpoints

    async searchMemory(query: string): Promise<any[]> {
        try {
            return await this.requestJson<any[]>('/api/memories/search', {
                method: 'GET',
                query: { query, project_id: this.projectId, limit: 10 }
            });
        } catch (error) {
            console.error('Memory search error:', error);
            return [];
        }
    }

    async getMemoryStats(): Promise<any> {
        try {
            return await this.requestJson<any>('/api/memories/stats', {
                method: 'GET',
                query: { project_id: this.projectId }
            });
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

    // Deep Discussion API methods

    /** Create a new Deep Discussion session */
    async createDeepDiscussionSession(
        question: string,
        pattern: string,
        conversationId?: string
    ): Promise<any> {
        try {
            const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: this.projectId,
                    question,
                    pattern,
                    conversation_id: conversationId
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Create session failed (${response.status}): ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Deep Discussion create session error:', error);
            throw error;
        }
    }

    /** Update configuration for an existing Deep Discussion session */
    async updateDeepDiscussionConfig(
        sessionId: string,
        config: { model_assignments?: Record<string, string>; pattern?: string; budget?: number; prefer_local?: boolean }
    ): Promise<any> {
        try {
            const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions/${sessionId}/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_assignments: config.model_assignments,
                    pattern: config.pattern,
                    budget: config.budget,
                    prefer_local: config.prefer_local
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Update config failed (${response.status}): ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Deep Discussion update config error:', error);
            throw error;
        }
    }

    /**
     * Start a Deep Discussion session with SSE streaming.
     *
     * Uses fetch() + ReadableStream for incremental SSE parsing.
     * Calls the appropriate callback for each parsed SSE event type.
     * Accepts an AbortSignal for cancellation.
     */
    async startDeepDiscussion(
        sessionId: string,
        signal: AbortSignal | undefined,
        callbacks: {
            onSessionStart?: (data: any) => void;
            onStepStart?: (data: any) => void;
            onAgentStart?: (agent: string, data: any) => void;
            onAgentChunk?: (agent: string, content: string) => void;
            onAgentDone?: (agent: string, data: any) => void;
            onStepComplete?: (data: any) => void;
            onSessionComplete?: (data: any) => void;
            onDone?: () => void;
            onError?: (message: string) => void;
        }
    ): Promise<void> {
        const url = `${this.baseURL}/api/deep-discussion/sessions/${sessionId}/start`;

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Start discussion failed (${response.status}): ${errorText}`);
        }

        if (!response.body) {
            throw new Error('Response body is null -- streaming not supported');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) { break; }

                buffer += decoder.decode(value, { stream: true });

                // Parse SSE lines: each event is "data: <json>\n\n"
                const lines = buffer.split('\n');
                buffer = '';

                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];

                    // If this line doesn't end with \n and is the last line,
                    // it might be an incomplete chunk -- keep in buffer
                    if (i === lines.length - 1 && line !== '') {
                        buffer = line;
                        continue;
                    }

                    if (!line.startsWith('data: ')) { continue; }

                    const jsonStr = line.slice(6).trim();
                    if (!jsonStr || jsonStr === '[DONE]') { continue; }

                    try {
                        const event = JSON.parse(jsonStr);

                        switch (event.type) {
                            case 'session_start':
                                callbacks.onSessionStart?.(event);
                                break;

                            case 'step_start':
                                callbacks.onStepStart?.(event);
                                break;

                            case 'agent_start':
                                callbacks.onAgentStart?.(event.agent || 'unknown', event);
                                break;

                            case 'agent_chunk':
                                callbacks.onAgentChunk?.(
                                    event.agent || 'unknown',
                                    event.content || ''
                                );
                                break;

                            case 'agent_done':
                                callbacks.onAgentDone?.(event.agent || 'unknown', event);
                                break;

                            case 'step_complete':
                                callbacks.onStepComplete?.(event);
                                break;

                            case 'session_complete':
                                callbacks.onSessionComplete?.(event);
                                break;

                            case 'done':
                                callbacks.onDone?.();
                                break;

                            case 'session_error':
                                callbacks.onError?.(event.message || 'Unknown session error');
                                break;
                        }
                    } catch (parseErr) {
                        console.warn('Failed to parse Deep Discussion SSE event:', jsonStr, parseErr);
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    /** Get a Deep Discussion session by ID */
    async getDeepDiscussionSession(sessionId: string): Promise<any> {
        try {
            const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions/${sessionId}`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Get session failed (${response.status}): ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Deep Discussion get session error:', error);
            throw error;
        }
    }

    /** List Deep Discussion sessions for the current project */
    async listDeepDiscussionSessions(limit: number = 10): Promise<any[]> {
        try {
            const response = await fetch(
                `${this.baseURL}/api/deep-discussion/sessions?project_id=${encodeURIComponent(this.projectId)}`,
                {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' }
                }
            );

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`List sessions failed (${response.status}): ${errorText}`);
            }

            const sessions: any[] = await response.json();
            return sessions.slice(0, limit);
        } catch (error) {
            console.error('Deep Discussion list sessions error:', error);
            throw error;
        }
    }

    /** Replay a completed Deep Discussion session */
    async replayDeepDiscussion(sessionId: string): Promise<any> {
        try {
            const response = await fetch(`${this.baseURL}/api/deep-discussion/sessions/${sessionId}/replay`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Replay session failed (${response.status}): ${errorText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Deep Discussion replay session error:', error);
            throw error;
        }
    }

    disconnect() {
        // No persistent connection to close (using HTTP/SSE)
    }
}

/**
 * Standalone ADAM Core for VSCode
 * Works directly without external backend - like Claude Code
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

// Import Python modules directly via node-calls-python or similar
// For now, we'll use a simpler approach with direct API calls

export class StandaloneADAM {
    private memoryPath: string;
    private openaiKey: string | undefined;
    private grokKey: string | undefined;
    private conversationHistory: any[] = [];
    
    constructor(context: vscode.ExtensionContext) {
        // Use VSCode's global storage for memory
        this.memoryPath = context.globalStorageUri.fsPath;
        
        // Get API keys from VSCode settings or environment
        const config = vscode.workspace.getConfiguration('adam');
        this.openaiKey = config.get('openaiApiKey') || process.env.OPENAI_API_KEY;
        this.grokKey = config.get('grokApiKey') || process.env.GROK_API_KEY || process.env.XAI_API_KEY;
        
        // Initialize memory directory
        if (!fs.existsSync(this.memoryPath)) {
            fs.mkdirSync(this.memoryPath, { recursive: true });
        }
    }
    
    /**
     * Process message directly without backend
     */
    async processMessage(message: string, workspaceContext?: string): Promise<string> {
        // Check for file operations
        const fileOperation = this.detectFileOperation(message);
        if (fileOperation) {
            return await this.handleFileOperation(fileOperation, message);
        }
        
        // Direct LLM call
        return await this.callLLM(message, workspaceContext);
    }
    
    /**
     * Detect if user is asking for file operations
     */
    private detectFileOperation(message: string): string | null {
        const patterns = {
            read: /(?:read|show|display|explain|analyze)\s+(?:the\s+)?(?:file\s+)?([^\s]+\.[\w]+)/i,
            write: /(?:create|write|save)\s+(?:a\s+)?(?:file|script)\s+(?:called\s+)?([^\s]+\.[\w]+)/i,
            edit: /(?:edit|modify|change|update)\s+(?:the\s+)?(?:file\s+)?([^\s]+\.[\w]+)/i,
            list: /(?:list|show)\s+(?:all\s+)?(?:the\s+)?files/i,
        };
        
        for (const [operation, pattern] of Object.entries(patterns)) {
            if (pattern.test(message)) {
                return operation;
            }
        }
        return null;
    }
    
    /**
     * Handle file operations directly
     */
    private async handleFileOperation(operation: string, message: string): Promise<string> {
        switch (operation) {
            case 'read':
                return await this.readFile(message);
            case 'write':
                return await this.createFile(message);
            case 'edit':
                return await this.editFile(message);
            case 'list':
                return await this.listFiles();
            default:
                return "Operation not supported yet.";
        }
    }
    
    /**
     * Read file from workspace
     */
    private async readFile(message: string): Promise<string> {
        const match = message.match(/([^\s]+\.[\w]+)/);
        if (!match) return "Could not identify file name.";
        
        const fileName = match[1];
        const files = await vscode.workspace.findFiles(`**/${fileName}`, '**/node_modules/**', 10);
        
        if (files.length === 0) {
            return `File '${fileName}' not found in workspace.`;
        }
        
        const filePath = files[0];
        const content = await vscode.workspace.fs.readFile(filePath);
        const text = Buffer.from(content).toString('utf8');
        
        // Now analyze with LLM
        const prompt = `${message}\n\nFile content:\n\`\`\`\n${text}\n\`\`\``;
        return await this.callLLM(prompt);
    }
    
    /**
     * Direct LLM call without backend
     */
    private async callLLM(prompt: string, context?: string): Promise<string> {
        // Add conversation context from memory
        const memoryContext = await this.getRelevantMemory(prompt);
        
        const fullPrompt = `${memoryContext ? `Context from memory:\n${memoryContext}\n\n` : ''}${context ? `Current context:\n${context}\n\n` : ''}${prompt}`;
        
        // Use OpenAI or Grok directly
        if (this.openaiKey) {
            return await this.callOpenAI(fullPrompt);
        } else if (this.grokKey) {
            return await this.callGrok(fullPrompt);
        } else {
            return "No API keys configured. Please set OpenAI or Grok API key in settings.";
        }
    }
    
    /**
     * Call OpenAI directly
     */
    private async callOpenAI(prompt: string): Promise<string> {
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.openaiKey}`
            },
            body: JSON.stringify({
                model: 'gpt-4-turbo-preview',
                messages: [
                    {
                        role: 'system',
                        content: 'You are ADAM, an AI assistant for developers. Be concise and helpful. Do not introduce yourself unless asked.'
                    },
                    {
                        role: 'user',
                        content: prompt
                    }
                ],
                temperature: 0.7,
                max_tokens: 2000
            })
        });
        
        const data: any = await response.json();
        
        if (data.error) {
            return `Error: ${data.error.message}`;
        }
        
        const answer = data.choices[0].message.content;
        
        // Save to memory if important
        await this.saveToMemory(prompt, answer);
        
        return answer;
    }
    
    /**
     * Call Grok/xAI directly
     */
    private async callGrok(prompt: string): Promise<string> {
        const response = await fetch('https://api.x.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.grokKey}`
            },
            body: JSON.stringify({
                model: 'grok-2-latest',
                messages: [
                    {
                        role: 'system',
                        content: 'You are ADAM, an AI assistant for developers. Be concise and helpful. Do not introduce yourself unless asked.'
                    },
                    {
                        role: 'user',
                        content: prompt
                    }
                ],
                temperature: 0.7
            })
        });
        
        const data: any = await response.json();
        
        if (data.error) {
            return `Error: ${data.error.message}`;
        }
        
        const answer = data.choices[0].message.content;
        
        // Save to memory if important
        await this.saveToMemory(prompt, answer);
        
        return answer;
    }
    
    /**
     * Get relevant memory for context
     */
    private async getRelevantMemory(query: string): Promise<string | null> {
        // Simple file-based memory for now
        const memoryFile = path.join(this.memoryPath, 'memory.json');
        
        if (!fs.existsSync(memoryFile)) {
            return null;
        }
        
        try {
            const memories = JSON.parse(fs.readFileSync(memoryFile, 'utf8'));
            // Simple keyword matching for now
            const relevant = memories.filter((m: any) => {
                const keywords = query.toLowerCase().split(' ');
                return keywords.some(k => m.content.toLowerCase().includes(k));
            }).slice(-3); // Last 3 relevant memories
            
            if (relevant.length > 0) {
                return relevant.map((m: any) => m.content).join('\n');
            }
        } catch (e) {
            console.error('Error reading memory:', e);
        }
        
        return null;
    }
    
    /**
     * Save important interactions to memory
     */
    private async saveToMemory(query: string, response: string): Promise<void> {
        // Determine if this should be saved
        if (query.length < 20 || response.length < 50) return;
        
        const memoryFile = path.join(this.memoryPath, 'memory.json');
        let memories = [];
        
        if (fs.existsSync(memoryFile)) {
            try {
                memories = JSON.parse(fs.readFileSync(memoryFile, 'utf8'));
            } catch (e) {
                console.error('Error reading memory file:', e);
            }
        }
        
        memories.push({
            timestamp: new Date().toISOString(),
            query,
            response: response.substring(0, 500), // Save first 500 chars
            content: `Q: ${query}\nA: ${response.substring(0, 200)}...`,
            workspace: vscode.workspace.name || 'unknown'
        });
        
        // Keep only last 100 memories
        if (memories.length > 100) {
            memories = memories.slice(-100);
        }
        
        fs.writeFileSync(memoryFile, JSON.stringify(memories, null, 2));
    }
    
    /**
     * List files in workspace
     */
    private async listFiles(): Promise<string> {
        const files = await vscode.workspace.findFiles('**/*', '**/node_modules/**', 100);
        const fileList = files.map(f => path.relative(vscode.workspace.rootPath || '', f.fsPath));
        return `Files in workspace:\n${fileList.slice(0, 20).join('\n')}${fileList.length > 20 ? `\n... and ${fileList.length - 20} more` : ''}`;
    }
    
    /**
     * Create a new file
     */
    private async createFile(message: string): Promise<string> {
        // Extract file name and content request
        const match = message.match(/(?:create|write)\s+(?:a\s+)?(?:file|script)\s+(?:called\s+)?([^\s]+\.[\w]+)(?:\s+(?:with|containing|that)\s+(.+))?/i);
        
        if (!match) return "Could not understand the file creation request.";
        
        const fileName = match[1];
        const contentRequest = match[2] || "basic template";
        
        // Generate content with LLM
        const contentPrompt = `Generate content for a file named ${fileName} with: ${contentRequest}. Return only the code, no explanations.`;
        const content = await this.callLLM(contentPrompt);
        
        // Create the file
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) return "No workspace folder open.";
        
        const filePath = vscode.Uri.joinPath(workspaceFolder.uri, fileName);
        await vscode.workspace.fs.writeFile(filePath, Buffer.from(content, 'utf8'));
        
        // Open the file
        const doc = await vscode.workspace.openTextDocument(filePath);
        await vscode.window.showTextDocument(doc);
        
        return `Created ${fileName}`;
    }
    
    /**
     * Edit an existing file
     */
    private async editFile(message: string): Promise<string> {
        // This would need more complex parsing
        return "Edit functionality coming soon. For now, I can read files and suggest changes.";
    }
}
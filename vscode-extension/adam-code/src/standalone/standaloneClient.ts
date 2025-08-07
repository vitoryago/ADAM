/**
 * Standalone ADAM Client
 * Works without backend, directly in VSCode
 */

import * as vscode from 'vscode';
import { Message } from '../client/adamClient';
import { UnifiedMemoryManager } from './memoryManager';
import * as fs from 'fs';
import * as path from 'path';

export class StandaloneADAMClient {
    private memoryManager: UnifiedMemoryManager;
    private openaiKey: string | undefined;
    private grokKey: string | undefined;
    
    constructor(context: vscode.ExtensionContext) {
        // Get workspace name for project-based memory
        const workspaceName = vscode.workspace.name || 'default';
        this.memoryManager = new UnifiedMemoryManager(workspaceName);
        
        // Load API keys from settings
        const config = vscode.workspace.getConfiguration('adam');
        this.openaiKey = config.get('openaiApiKey') || process.env.OPENAI_API_KEY;
        this.grokKey = config.get('grokApiKey') || process.env.GROK_API_KEY || process.env.XAI_API_KEY;
        
        // Try to load from .env if not in settings
        if (!this.openaiKey && !this.grokKey) {
            this.loadEnvFile();
        }
    }
    
    /**
     * Load API keys from .env file
     */
    private loadEnvFile() {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) return;
        
        const envPath = path.join(workspaceFolder.uri.fsPath, '.env');
        if (fs.existsSync(envPath)) {
            const envContent = fs.readFileSync(envPath, 'utf8');
            const lines = envContent.split('\n');
            
            lines.forEach(line => {
                const [key, value] = line.split('=');
                if (key?.trim() === 'OPENAI_API_KEY' && !this.openaiKey) {
                    this.openaiKey = value?.trim();
                } else if ((key?.trim() === 'GROK_API_KEY' || key?.trim() === 'XAI_API_KEY') && !this.grokKey) {
                    this.grokKey = value?.trim();
                }
            });
        }
    }
    
    /**
     * Send message - main interface matching original ADAMClient
     */
    async sendMessage(content: string, useMemory: boolean = true): Promise<Message> {
        try {
            // Get relevant memories if enabled
            let memoryContext = '';
            if (useMemory) {
                const memories = await this.memoryManager.searchMemories(content, 3);
                if (memories.length > 0) {
                    memoryContext = 'Relevant context from previous conversations:\n';
                    memories.forEach(mem => {
                        memoryContext += `- ${mem.content.substring(0, 200)}...\n`;
                    });
                    memoryContext += '\n';
                }
            }
            
            // Prepare the full prompt
            const fullPrompt = memoryContext + content;
            
            // Call the appropriate LLM
            let response: string;
            let model: string;
            let cost = 0;
            
            if (this.grokKey) {
                // Prefer Grok for code tasks
                response = await this.callGrok(fullPrompt);
                model = 'grok-2';
            } else if (this.openaiKey) {
                response = await this.callOpenAI(fullPrompt);
                model = 'gpt-4-turbo';
            } else {
                return {
                    role: 'assistant',
                    content: 'No API keys configured. Please set OpenAI or Grok API key in VSCode settings (Cmd+, then search for "adam").',
                    model: 'none'
                };
            }
            
            // Save to memory if substantial
            if (useMemory && response.length > 50) {
                await this.memoryManager.saveMemory(
                    content,
                    response,
                    vscode.workspace.name
                );
            }
            
            return {
                role: 'assistant',
                content: response,
                model,
                cost
            };
        } catch (error: any) {
            return {
                role: 'assistant',
                content: `Error: ${error.message || error}`,
                model: 'error'
            };
        }
    }
    
    /**
     * Call OpenAI API
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
                        content: 'You are a helpful AI assistant for developers. Be concise and direct. Do not introduce yourself unless asked.'
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
            throw new Error(data.error.message);
        }
        
        return data.choices[0].message.content;
    }
    
    /**
     * Call Grok/xAI API
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
                        content: 'You are a helpful AI assistant for developers. Be concise and direct. Do not introduce yourself unless asked.'
                    },
                    {
                        role: 'user',
                        content: prompt
                    }
                ],
                temperature: 0.7,
                stream: false
            })
        });
        
        const data: any = await response.json();
        
        if (data.error) {
            throw new Error(data.error.message);
        }
        
        return data.choices[0].message.content;
    }
    
    /**
     * Optimize SQL (matches original interface)
     */
    async optimizeSQL(query: string, dialect: string): Promise<any> {
        const prompt = `Optimize this ${dialect} SQL query:\n\`\`\`sql\n${query}\n\`\`\`\n\nProvide: 1) Optimized query, 2) Brief explanation of changes, 3) Performance improvements`;
        
        const response = await this.sendMessage(prompt, false);
        
        // Parse response to match expected format
        return {
            query: query, // Would need better parsing
            explanation: response.content,
            improvements: 'See explanation above',
            estimatedSpeedup: 'N/A'
        };
    }
    
    /**
     * Explain code (matches original interface)
     */
    async explainCode(code: string, language: string, fileName: string): Promise<string> {
        const prompt = `Explain this ${language} code from ${fileName}:\n\`\`\`${language}\n${code}\n\`\`\``;
        const response = await this.sendMessage(prompt, true);
        return response.content;
    }
    
    /**
     * Search memory
     */
    async searchMemory(query: string): Promise<any[]> {
        const memories = await this.memoryManager.searchMemories(query, 10);
        return memories;
    }
    
    /**
     * Get memory stats
     */
    async getMemoryStats(): Promise<any> {
        const allMemories = await this.memoryManager.loadMemories();
        const recentMemories = await this.memoryManager.getRecentMemories(24);
        
        return {
            total: allMemories.length,
            recent24h: recentMemories.length,
            projects: this.memoryManager.getAllProjects()
        };
    }
    
    /**
     * Generate DBT model
     */
    async generateDBTModel(modelName: string, sourceTable: string): Promise<any> {
        const prompt = `Generate a dbt staging model named "${modelName}" for source table "${sourceTable}". Include proper CTEs, transformations, and documentation.`;
        const response = await this.sendMessage(prompt, false);
        
        // Extract SQL from response
        const sqlMatch = response.content.match(/```sql\n([\s\S]*?)```/);
        const sql = sqlMatch ? sqlMatch[1] : response.content;
        
        return {
            sql,
            documentation: `Model for ${sourceTable}`,
            tests: ['unique', 'not_null']
        };
    }
    
    /**
     * Create branch name
     */
    async createBranch(branchName: string): Promise<any> {
        const prompt = `Format this branch name for git: "${branchName}". Provide a clean branch name and brief description.`;
        const response = await this.sendMessage(prompt, false);
        
        const formatted = branchName.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9\-]/g, '');
        
        return {
            formattedName: formatted,
            description: response.content.substring(0, 100)
        };
    }
    
    /**
     * Generate PR details
     */
    async generatePRDetails(changes: string): Promise<any> {
        const prompt = `Generate a pull request title and description for these changes:\n${changes.substring(0, 1000)}`;
        const response = await this.sendMessage(prompt, false);
        
        return {
            title: 'Update: Changes from ADAM',
            body: response.content,
            labels: ['adam-generated']
        };
    }
    
    /**
     * Analyze data pattern
     */
    async analyzeDataPattern(data: string): Promise<any> {
        const prompt = `Analyze this data for patterns and insights:\n${data.substring(0, 2000)}`;
        const response = await this.sendMessage(prompt, true);
        
        return {
            summary: response.content,
            patterns: 'See summary',
            recommendations: 'See summary'
        };
    }
    
    /**
     * Disconnect (for compatibility)
     */
    disconnect() {
        // Nothing to disconnect in standalone mode
    }
}
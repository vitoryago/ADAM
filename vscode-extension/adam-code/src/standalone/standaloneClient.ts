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
    private conversationHistory: Message[] = [];
    
    constructor(context: vscode.ExtensionContext) {
        // Get workspace name for project-based memory
        const workspaceName = vscode.workspace.name || 'default';
        this.memoryManager = new UnifiedMemoryManager(workspaceName);
        
        // Always try to load from .env first (project root)
        this.loadEnvFile();
        
        // Then check VSCode settings (can override .env)
        const config = vscode.workspace.getConfiguration('adam');
        const settingsOpenAI = config.get<string>('openaiApiKey');
        const settingsGrok = config.get<string>('grokApiKey');
        
        if (settingsOpenAI) {
            this.openaiKey = settingsOpenAI;
        }
        if (settingsGrok) {
            this.grokKey = settingsGrok;
        }
        
        // Log status for debugging
        console.log('ADAM Standalone Client initialized:');
        console.log('- OpenAI Key:', this.openaiKey ? 'Configured' : 'Not configured');
        console.log('- Grok Key:', this.grokKey ? 'Configured' : 'Not configured');
    }
    
    /**
     * Load API keys from .env file
     */
    private loadEnvFile() {
        // Try multiple locations for .env file
        const possiblePaths = [
            '/Users/vitoryago/ADAM/.env', // Direct ADAM project path
        ];
        
        // Also check workspace folder
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (workspaceFolder) {
            possiblePaths.push(path.join(workspaceFolder.uri.fsPath, '.env'));
        }
        
        for (const envPath of possiblePaths) {
            if (fs.existsSync(envPath)) {
                console.log(`Loading .env from: ${envPath}`);
                const envContent = fs.readFileSync(envPath, 'utf8');
                const lines = envContent.split('\n');
                
                lines.forEach(line => {
                    // Handle lines with quotes and without
                    const match = line.match(/^([A-Z_]+)=(.*)$/);
                    if (match) {
                        const key = match[1].trim();
                        let value = match[2].trim();
                        
                        // Remove quotes if present
                        if ((value.startsWith('"') && value.endsWith('"')) || 
                            (value.startsWith("'") && value.endsWith("'"))) {
                            value = value.slice(1, -1);
                        }
                        
                        if (key === 'OPENAI_API_KEY' && !this.openaiKey) {
                            this.openaiKey = value;
                            console.log('Loaded OpenAI API key from .env');
                        } else if ((key === 'GROK_API_KEY' || key === 'XAI_API_KEY') && !this.grokKey) {
                            this.grokKey = value;
                            console.log(`Loaded ${key} from .env`);
                        }
                    }
                });
                break; // Stop after first successful .env load
            }
        }
    }
    
    /**
     * Send message - main interface matching original ADAMClient
     */
    async sendMessage(content: string, useMemory: boolean = true): Promise<Message> {
        try {
            // Add user message to conversation history
            this.conversationHistory.push({ role: 'user', content });
            
            // Keep only last 10 messages to avoid context overflow
            if (this.conversationHistory.length > 20) {
                this.conversationHistory = this.conversationHistory.slice(-20);
            }
            
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
            
            // Prepare the full prompt with conversation history
            const fullPrompt = memoryContext + content;
            
            // Call the appropriate LLM
            let response: string;
            let model: string;
            let cost = 0;
            
            // Analyze complexity for model selection
            const complexity = this.analyzeComplexity(content);
            
            if (this.openaiKey) {
                // Prefer OpenAI for GPT-5 models
                response = await this.callOpenAI(fullPrompt, complexity);
                model = complexity.model;
            } else if (this.grokKey) {
                // Fallback to Grok if no OpenAI key
                response = await this.callGrok(fullPrompt, complexity);
                model = complexity.model;
            } else {
                return {
                    role: 'assistant',
                    content: 'No API keys configured. Please set OpenAI API key in VSCode settings (Cmd+, then search for "adam").',
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
            
            // Add assistant response to conversation history
            const assistantMessage: Message = {
                role: 'assistant',
                content: response,
                model,
                cost
            };
            this.conversationHistory.push(assistantMessage);
            
            return assistantMessage;
        } catch (error: any) {
            return {
                role: 'assistant',
                content: `Error: ${error.message || error}`,
                model: 'error'
            };
        }
    }
    
    /**
     * Call OpenAI API with GPT-5 support
     */
    private async callOpenAI(prompt: string, complexity: { level: string; model: string; reasoning_effort?: string }): Promise<string> {
        // Map our model names to actual OpenAI API model names
        const modelMapping: { [key: string]: string } = {
            'gpt-5': 'gpt-5-2025-08-07',
            'gpt-5-mini': 'gpt-5-mini-2025-08-07',
            'gpt-5-nano': 'gpt-5-nano'
        };
        
        const apiModel = modelMapping[complexity.model] || 'gpt-5-mini-2025-08-07';
        
        // Build messages array with conversation history
        const messages: any[] = [
            {
                role: 'system',
                content: 'You are a helpful AI assistant for developers. Be concise and direct. Do not introduce yourself unless asked.'
            }
        ];
        
        // Add recent conversation history (excluding the current user message which is in prompt)
        const recentHistory = this.conversationHistory.slice(-10, -1); // Last 10 messages, excluding current
        recentHistory.forEach(msg => {
            if (msg.role === 'user' || msg.role === 'assistant') {
                messages.push({
                    role: msg.role,
                    content: msg.content
                });
            }
        });
        
        // Add current user message
        messages.push({
            role: 'user',
            content: prompt
        });
        
        // Build request body with reasoning effort for GPT-5
        const requestBody: any = {
            model: apiModel,
            messages,
            stream: false  // No streaming in VSCode extension
        };
        
        // GPT-5 specific parameters
        if (apiModel.includes('gpt-5')) {
            // GPT-5 only supports default temperature (1), so we omit it
            requestBody.max_completion_tokens = 2000;
        } else {
            // Other models support temperature
            requestBody.temperature = 0.7;
            requestBody.max_tokens = 2000;
        }
        
        // Add reasoning effort for GPT-5 models
        if (complexity.reasoning_effort) {
            requestBody.reasoning_effort = complexity.reasoning_effort;
        }
        
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.openaiKey}`
            },
            body: JSON.stringify(requestBody)
        });
        
        const data: any = await response.json();
        
        if (data.error) {
            throw new Error(data.error.message);
        }
        
        return data.choices[0].message.content;
    }
    
    /**
     * Call OpenAI API with streaming support for GPT-5
     */
    private async callOpenAIStream(
        prompt: string, 
        complexity: { level: string; model: string; reasoning_effort?: string },
        onChunk: (chunk: string) => void
    ): Promise<string> {
        // Map our model names to actual OpenAI API model names
        const modelMapping: { [key: string]: string } = {
            'gpt-5': 'gpt-5-2025-08-07',
            'gpt-5-mini': 'gpt-5-mini-2025-08-07',
            'gpt-5-nano': 'gpt-5-nano'
        };
        
        const apiModel = modelMapping[complexity.model] || 'gpt-5-mini-2025-08-07';
        
        // Build messages array with conversation history
        const messages: any[] = [
            {
                role: 'system',
                content: 'You are a helpful AI assistant for developers. Be concise and direct. Do not introduce yourself unless asked.'
            }
        ];
        
        // Add recent conversation history
        const recentHistory = this.conversationHistory.slice(-10, -1);
        recentHistory.forEach(msg => {
            if (msg.role === 'user' || msg.role === 'assistant') {
                messages.push({
                    role: msg.role,
                    content: msg.content
                });
            }
        });
        
        // Add current user message
        messages.push({
            role: 'user',
            content: prompt
        });
        
        // Build request body with streaming enabled
        const requestBody: any = {
            model: apiModel,
            messages,
            stream: true  // Enable streaming
        };
        
        // GPT-5 specific parameters
        if (apiModel.includes('gpt-5')) {
            // GPT-5 only supports default temperature (1), so we omit it
            requestBody.max_completion_tokens = 2000;
        } else {
            // Other models support temperature
            requestBody.temperature = 0.7;
            requestBody.max_tokens = 2000;
        }
        
        // Add reasoning effort for GPT-5 models
        if (complexity.reasoning_effort) {
            requestBody.reasoning_effort = complexity.reasoning_effort;
        }
        
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.openaiKey}`
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error?.message || 'API request failed');
        }
        
        // Handle streaming response
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let fullContent = '';
        
        if (!reader) {
            throw new Error('No response body');
        }
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') continue;
                        
                        try {
                            const parsed = JSON.parse(data);
                            const content = parsed.choices?.[0]?.delta?.content || '';
                            if (content) {
                                fullContent += content;
                                onChunk(content);
                            }
                        } catch (e) {
                            // Skip unparseable chunks
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
        
        return fullContent;
    }
    
    /**
     * Analyze query complexity for model selection
     * Matches backend routing: gpt-5 (high/medium), gpt-5-mini (low)
     */
    private analyzeComplexity(query: string): { level: 'high' | 'medium' | 'low'; model: string; reasoning_effort?: string } {
        const lowerQuery = query.toLowerCase();
        
        // High complexity indicators - use GPT-5 with high reasoning
        if (lowerQuery.includes('implement') || lowerQuery.includes('refactor') ||
            lowerQuery.includes('debug') || lowerQuery.includes('architecture') ||
            lowerQuery.includes('design a solution') || lowerQuery.includes('complex') ||
            lowerQuery.includes('step by step') || lowerQuery.includes('code generation') ||
            lowerQuery.includes('build a') || lowerQuery.includes('create a function')) {
            return { level: 'high', model: 'gpt-5', reasoning_effort: 'high' };
        }
        
        // Medium complexity indicators - use GPT-5 with medium reasoning
        if (lowerQuery.includes('explain') || lowerQuery.includes('analyze') ||
            lowerQuery.includes('write') || lowerQuery.includes('create') || 
            lowerQuery.includes('fix') || lowerQuery.includes('update') || 
            lowerQuery.includes('modify') || lowerQuery.includes('optimize') ||
            lowerQuery.includes('how does') || lowerQuery.includes('sql')) {
            return { level: 'medium', model: 'gpt-5', reasoning_effort: 'medium' };
        }
        
        // Default to low complexity for simple queries - use gpt-5-mini with minimal reasoning
        return { level: 'low', model: 'gpt-5-mini', reasoning_effort: 'minimal' };
    }
    
    /**
     * Call Grok/xAI API with intelligent model selection
     */
    private async callGrok(prompt: string, complexity: { level: string; model: string }): Promise<string> {
        // Map our model names to actual API model names
        // Matching backend configuration exactly
        const modelMapping: { [key: string]: string } = {
            'grok-4-reasoning': 'grok-4',      // High complexity tasks
            'grok-4': 'grok-4',                 // Medium complexity
            'grok-3-mini-high': 'grok-3-mini'   // Fast, simple queries
        };
        
        const apiModel = modelMapping[complexity.model] || 'grok-3-mini';
        
        // Build messages array with conversation history
        const messages: any[] = [
            {
                role: 'system',
                content: 'You are a helpful AI assistant for developers. Be concise and direct. Do not introduce yourself unless asked.'
            }
        ];
        
        // Add recent conversation history (excluding the current user message which is in prompt)
        const recentHistory = this.conversationHistory.slice(-10, -1); // Last 10 messages, excluding current
        recentHistory.forEach(msg => {
            if (msg.role === 'user' || msg.role === 'assistant') {
                messages.push({
                    role: msg.role,
                    content: msg.content
                });
            }
        });
        
        // Add current user message
        messages.push({
            role: 'user',
            content: prompt
        });
        
        const response = await fetch('https://api.x.ai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.grokKey}`
            },
            body: JSON.stringify({
                model: apiModel,
                messages,
                temperature: 0.7,
                stream: false,
                // Add reasoning_effort for grok-3-mini to get better quality
                // This makes grok-3-mini perform better on tasks that don't need grok-4
                ...(apiModel === 'grok-3-mini' ? { reasoning_effort: 'high' } : {})
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
     * Send message with streaming support
     */
    async sendMessageStream(
        content: string, 
        useMemory: boolean = true,
        onChunk: (chunk: string) => void
    ): Promise<Message> {
        try {
            // Add user message to conversation history
            this.conversationHistory.push({ role: 'user', content });
            
            // Keep only last 10 messages to avoid context overflow
            if (this.conversationHistory.length > 20) {
                this.conversationHistory = this.conversationHistory.slice(-20);
            }
            
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
            
            // Prepare the full prompt with conversation history
            const fullPrompt = memoryContext + content;
            
            // Analyze complexity for model selection
            const complexity = this.analyzeComplexity(content);
            
            let response: string;
            let model: string;
            let cost = 0;
            
            if (this.openaiKey) {
                // Use streaming for GPT-5 models
                response = await this.callOpenAIStream(fullPrompt, complexity, onChunk);
                model = complexity.model;
            } else if (this.grokKey) {
                // Fallback to non-streaming Grok
                response = await this.callGrok(fullPrompt, complexity);
                model = complexity.model;
                // Simulate streaming by sending the whole response
                onChunk(response);
            } else {
                const errorMsg = 'No API keys configured. Please set OpenAI API key in VSCode settings.';
                onChunk(errorMsg);
                return {
                    role: 'assistant',
                    content: errorMsg,
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
            
            // Add assistant response to conversation history
            const assistantMessage: Message = {
                role: 'assistant',
                content: response,
                model,
                cost
            };
            this.conversationHistory.push(assistantMessage);
            
            return assistantMessage;
        } catch (error: any) {
            const errorMsg = `Error: ${error.message || error}`;
            onChunk(errorMsg);
            return {
                role: 'assistant',
                content: errorMsg,
                model: 'error'
            };
        }
    }
    
    /**
     * Disconnect (for compatibility)
     */
    disconnect() {
        // Nothing to disconnect in standalone mode
    }
}
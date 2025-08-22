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
            
            // Analyze complexity for model selection (now async with GPT-5-mini)
            const complexity = await this.analyzeComplexity(content);
            
            // Route to appropriate API based on model
            if (complexity.model === 'grok-4-reasoning' && this.grokKey) {
                // Use Grok for coding tasks (best for coding)
                response = await this.callGrok(fullPrompt, complexity);
                model = complexity.model;
            } else if (complexity.model.startsWith('gpt-5') && this.openaiKey) {
                // Use OpenAI for GPT-5 models
                response = await this.callOpenAI(fullPrompt, complexity);
                model = complexity.model;
            } else if (this.grokKey) {
                // Fallback to Grok if model doesn't match or no OpenAI key
                response = await this.callGrok(fullPrompt, complexity);
                model = complexity.model;
            } else if (this.openaiKey) {
                // Fallback to OpenAI if no Grok key
                response = await this.callOpenAI(fullPrompt, complexity);
                model = complexity.model;
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
            'gpt-5-thinking': 'gpt-5-2025-08-07',    // GPT-5 with high reasoning for complex tasks
            'gpt-5': 'gpt-5-2025-08-07',             // Standard GPT-5
            'gpt-5-mini': 'gpt-5-mini-2025-08-07',   // Lightweight GPT-5 for simple queries
            'gpt-5-nano': 'gpt-5-nano',               // Not used in current routing
            'grok-4-reasoning': 'gpt-5-2025-08-07',  // Fallback: Map grok-4-reasoning to GPT-5 for OpenAI
            'claude-opus-4.1': 'gpt-5-2025-08-07'    // Fallback: Map Claude to GPT-5
        };
        
        const apiModel = modelMapping[complexity.model] || 'gpt-5-2025-08-07';  // Default to GPT-5 for VSCode
        
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
        // For gpt-5-thinking, always use high reasoning
        if (complexity.model === 'gpt-5-thinking') {
            requestBody.reasoning_effort = 'high';
        } else if (complexity.reasoning_effort) {
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
            'gpt-5-thinking': 'gpt-5-2025-08-07',  // GPT-5 with high reasoning for complex tasks
            'gpt-5': 'gpt-5-2025-08-07',            // Standard GPT-5
            'gpt-5-mini': 'gpt-5-mini-2025-08-07',  // Lightweight GPT-5 for simple queries
            'gpt-5-nano': 'gpt-5-nano'               // Not used in current routing
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
        // For gpt-5-thinking, always use high reasoning
        if (complexity.model === 'gpt-5-thinking') {
            requestBody.reasoning_effort = 'high';
        } else if (complexity.reasoning_effort) {
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
            const error = await response.json() as any;
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
     * Use GPT-5-mini to intelligently analyze query complexity
     */
    private async analyzeComplexityWithAI(query: string): Promise<{ level: 'high' | 'medium' | 'low'; model: string; reasoning_effort?: string }> {
        // If no OpenAI key, fall back to rule-based analysis
        if (!this.openaiKey) {
            return this.analyzeComplexityRuleBased(query);
        }
        
        try {
            const classificationPrompt = `Classify this query's complexity and determine the best model.
Query: "${query}"

Rules:
- If it's about coding, implementing, debugging, refactoring, or writing functions/classes → Return: "CODING:grok-4-reasoning"
- If it's complex analysis, architecture, design, planning, or needs step-by-step thinking → Return: "COMPLEX:gpt-5-thinking"  
- If it's explaining, describing, summarizing, or standard questions → Return: "STANDARD:gpt-5"
- If it's simple, greeting, or basic questions → Return: "SIMPLE:gpt-5-mini"

Return ONLY the classification (e.g., "CODING:grok-4-reasoning")`;

            const messages = [
                { role: 'system', content: 'You are a query classifier. Return only the classification.' },
                { role: 'user', content: classificationPrompt }
            ];
            
            const response = await fetch('https://api.openai.com/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.openaiKey}`
                },
                body: JSON.stringify({
                    model: 'gpt-5-mini-2025-08-07',
                    messages,
                    max_tokens: 50,
                    temperature: 0.3
                })
            });
            
            const data: any = await response.json();
            if (data.error) {
                console.error('Classification error:', data.error);
                return this.analyzeComplexityRuleBased(query);
            }
            
            const classification = data.choices[0].message.content.trim();
            const [level, model] = classification.split(':');
            
            switch (level) {
                case 'CODING':
                    return { level: 'high', model: 'grok-4-reasoning', reasoning_effort: 'high' };
                case 'COMPLEX':
                    return { level: 'high', model: 'gpt-5-thinking', reasoning_effort: 'high' };
                case 'STANDARD':
                    return { level: 'medium', model: 'gpt-5', reasoning_effort: 'medium' };
                case 'SIMPLE':
                default:
                    return { level: 'low', model: 'gpt-5-mini', reasoning_effort: 'low' };
            }
        } catch (error) {
            console.error('AI classification failed, using rule-based:', error);
            return this.analyzeComplexityRuleBased(query);
        }
    }
    
    /**
     * Rule-based fallback for complexity analysis
     */
    private analyzeComplexityRuleBased(query: string): { level: 'high' | 'medium' | 'low'; model: string; reasoning_effort?: string } {
        const lowerQuery = query.toLowerCase();
        
        // Coding tasks - use grok-4-reasoning (best for coding)
        if (lowerQuery.includes('code') || lowerQuery.includes('implement') || 
            lowerQuery.includes('function') || lowerQuery.includes('class') ||
            lowerQuery.includes('debug') || lowerQuery.includes('fix') ||
            lowerQuery.includes('refactor') || lowerQuery.includes('optimize') ||
            lowerQuery.includes('write a') || lowerQuery.includes('create a') ||
            lowerQuery.includes('build') || lowerQuery.includes('develop')) {
            return { level: 'high', model: 'grok-4-reasoning', reasoning_effort: 'high' };
        }
        
        // Complex/thinking tasks - use gpt-5 with thinking (high reasoning)
        if (lowerQuery.includes('complex') || lowerQuery.includes('architecture') ||
            lowerQuery.includes('design') || lowerQuery.includes('analyze') ||
            lowerQuery.includes('step by step') || lowerQuery.includes('plan') ||
            lowerQuery.includes('strategy') || lowerQuery.includes('solution')) {
            return { level: 'high', model: 'gpt-5-thinking', reasoning_effort: 'high' };
        }
        
        // Standard queries - use gpt-5
        if (lowerQuery.includes('explain') || lowerQuery.includes('what is') ||
            lowerQuery.includes('how does') || lowerQuery.includes('tell me') || 
            lowerQuery.includes('describe') || lowerQuery.includes('summary')) {
            return { level: 'medium', model: 'gpt-5', reasoning_effort: 'medium' };
        }
        
        // Simple queries - use gpt-5-mini
        return { level: 'low', model: 'gpt-5-mini', reasoning_effort: 'low' };
    }
    
    /**
     * Analyze query complexity for model selection
     * Now uses GPT-5-mini for intelligent routing when available
     */
    private async analyzeComplexity(query: string): Promise<{ level: 'high' | 'medium' | 'low'; model: string; reasoning_effort?: string }> {
        // Use AI-based analysis when available, fallback to rules
        return this.analyzeComplexityWithAI(query);
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
            
            // Analyze complexity for model selection (now async with GPT-5-mini)
            const complexity = await this.analyzeComplexity(content);
            
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
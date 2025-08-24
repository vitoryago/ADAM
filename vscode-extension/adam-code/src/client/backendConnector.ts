/**
 * Backend Connector for ADAM VSCode Extension
 * Connects to full ADAM backend for RAG and advanced memory capabilities
 */

import * as vscode from 'vscode';
import axios from 'axios';
interface Message {
    role: 'user' | 'assistant';
    content: string;
    model?: string;
    cost?: number;
}

export interface BackendConfig {
    serverUrl: string;
    projectId: string;
    useRAG: boolean;
    useAdvancedMemory: boolean;
}

export class BackendConnector {
    private config: BackendConfig;
    private isConnected: boolean = false;
    private conversationId: string | null = null;
    
    constructor(config?: Partial<BackendConfig>) {
        const vsConfig = vscode.workspace.getConfiguration('adam');
        
        this.config = {
            serverUrl: config?.serverUrl || vsConfig.get('serverUrl') || 'http://localhost:8000',
            projectId: config?.projectId || vsConfig.get('projectId') || '3a859e97-16fd-46c6-b018-1ede9fade704',
            useRAG: config?.useRAG ?? vsConfig.get('useRAG', true),
            useAdvancedMemory: config?.useAdvancedMemory ?? vsConfig.get('useAdvancedMemory', true)
        };
    }
    
    /**
     * Test connection to backend
     */
    async testConnection(): Promise<boolean> {
        try {
            const response = await axios.get(`${this.config.serverUrl}/api/health`, {
                timeout: 5000
            });
            
            this.isConnected = response.status === 200;
            return this.isConnected;
        } catch (error) {
            console.error('Backend connection failed:', error);
            this.isConnected = false;
            return false;
        }
    }
    
    /**
     * Initialize or get conversation
     */
    async initConversation(): Promise<string | null> {
        if (this.conversationId) {
            return this.conversationId;
        }
        
        try {
            const response = await axios.post(
                `${this.config.serverUrl}/api/conversations`,
                {
                    project_id: this.config.projectId,
                    title: `VSCode Session - ${new Date().toLocaleString()}`
                }
            );
            
            this.conversationId = response.data.id;
            return this.conversationId;
        } catch (error) {
            console.error('Failed to initialize conversation:', error);
            return null;
        }
    }
    
    /**
     * Send message using full backend capabilities
     */
    async sendMessageWithRAG(content: string): Promise<Message | null> {
        try {
            // Ensure we have a conversation
            const conversationId = await this.initConversation();
            if (!conversationId) {
                throw new Error('No conversation ID');
            }
            
            // Send to backend with RAG and memory
            const response = await axios.post(
                `${this.config.serverUrl}/api/messages`,
                {
                    conversation_id: conversationId,
                    content,
                    use_memory: this.config.useAdvancedMemory,
                    use_rag: this.config.useRAG,
                    project_id: this.config.projectId
                },
                {
                    timeout: 120000 // 120 second timeout for complex queries
                }
            );
            
            const data = response.data;
            
            return {
                role: 'assistant',
                content: data.content,
                model: data.model_used,
                cost: data.cost
            };
        } catch (error: any) {
            console.error('Backend message failed:', error);
            return null;
        }
    }
    
    /**
     * Search memory using backend's advanced search
     */
    async searchMemory(query: string, limit: number = 5): Promise<any[]> {
        try {
            const response = await axios.post(
                `${this.config.serverUrl}/api/memory/search`,
                {
                    query,
                    project_id: this.config.projectId,
                    limit
                }
            );
            
            return response.data.memories || [];
        } catch (error) {
            console.error('Memory search failed:', error);
            return [];
        }
    }
    
    /**
     * Save to advanced memory system
     */
    async saveToMemory(query: string, response: string, metadata?: any): Promise<boolean> {
        try {
            await axios.post(
                `${this.config.serverUrl}/api/memory/save`,
                {
                    query,
                    response,
                    project_id: this.config.projectId,
                    metadata: {
                        ...metadata,
                        source: 'vscode',
                        workspace: vscode.workspace.name
                    }
                }
            );
            
            return true;
        } catch (error) {
            console.error('Memory save failed:', error);
            return false;
        }
    }
    
    /**
     * Get project information
     */
    async getProjectInfo(): Promise<any> {
        try {
            const response = await axios.get(
                `${this.config.serverUrl}/api/projects/${this.config.projectId}`
            );
            
            return response.data;
        } catch (error) {
            console.error('Failed to get project info:', error);
            return null;
        }
    }
    
    /**
     * List all projects
     */
    async listProjects(): Promise<any[]> {
        try {
            const response = await axios.get(
                `${this.config.serverUrl}/api/projects`
            );
            
            return response.data.projects || [];
        } catch (error) {
            console.error('Failed to list projects:', error);
            return [];
        }
    }
    
    /**
     * Create a new project
     */
    async createProject(name: string, description?: string): Promise<string | null> {
        try {
            const response = await axios.post(
                `${this.config.serverUrl}/api/projects`,
                {
                    name,
                    description,
                    workspace: vscode.workspace.name
                }
            );
            
            return response.data.id;
        } catch (error) {
            console.error('Failed to create project:', error);
            return null;
        }
    }
    
    /**
     * Get conversation history
     */
    async getConversationHistory(limit: number = 50): Promise<any[]> {
        try {
            if (!this.conversationId) {
                return [];
            }
            
            const response = await axios.get(
                `${this.config.serverUrl}/api/conversations/${this.conversationId}/messages`,
                {
                    params: { limit }
                }
            );
            
            return response.data.messages || [];
        } catch (error) {
            console.error('Failed to get conversation history:', error);
            return [];
        }
    }
    
    /**
     * Get memory statistics
     */
    async getMemoryStats(): Promise<any> {
        try {
            const response = await axios.get(
                `${this.config.serverUrl}/api/memory/stats`,
                {
                    params: { project_id: this.config.projectId }
                }
            );
            
            return response.data;
        } catch (error) {
            console.error('Failed to get memory stats:', error);
            return null;
        }
    }
    
    /**
     * Check if backend is available
     */
    isBackendAvailable(): boolean {
        return this.isConnected;
    }
    
    /**
     * Get backend configuration
     */
    getConfig(): BackendConfig {
        return this.config;
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig: Partial<BackendConfig>) {
        this.config = { ...this.config, ...newConfig };
    }
}
/**
 * Conversation History System for ADAM VSCode Extension
 * Maintains complete history of all interactions
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { Message } from '../client/adamClient';
import { BackendConnector } from '../client/backendConnector';

export interface ConversationEntry {
    id: string;
    timestamp: Date;
    message: Message;
    context?: {
        activeFile?: string;
        selectedText?: string;
        workspaceFolder?: string;
        gitBranch?: string;
        errors?: string[];
    };
    memoryUsed?: boolean;
    ragUsed?: boolean;
    modelUsed?: string;
    responseTime?: number;
    errorHandled?: boolean;
    solutionApplied?: boolean;
}

export interface ConversationSession {
    id: string;
    startTime: Date;
    endTime?: Date;
    entries: ConversationEntry[];
    title?: string;
    tags?: string[];
    metadata?: Record<string, any>;
}

export interface ConversationStats {
    totalConversations: number;
    totalMessages: number;
    averageSessionLength: number;
    mostUsedCommands: Array<{ command: string; count: number }>;
    errorsSolved: number;
    ragUsageRate: number;
}

export class ConversationHistoryManager {
    private static instance: ConversationHistoryManager;
    
    private sessions: Map<string, ConversationSession> = new Map();
    private currentSession: ConversationSession | null = null;
    private storagePath: string;
    private maxHistorySize: number = 10000; // Maximum entries to keep
    private backendConnector: BackendConnector;
    private autoSync: boolean = true;
    private syncInterval: NodeJS.Timeout | null = null;
    
    private constructor(context: vscode.ExtensionContext) {
        this.storagePath = context.globalStorageUri.fsPath;
        this.backendConnector = new BackendConnector();
        
        // Initialize storage
        this.initializeStorage();
        
        // Load existing history
        this.loadHistory();
        
        // Start new session
        this.startNewSession();
        
        // Setup auto-sync if backend available
        if (this.autoSync && this.backendConnector.isBackendAvailable()) {
            this.setupAutoSync();
        }
        
        // Register disposal
        context.subscriptions.push({
            dispose: () => this.dispose()
        });
    }
    
    static getInstance(context?: vscode.ExtensionContext): ConversationHistoryManager {
        if (!ConversationHistoryManager.instance) {
            if (!context) {
                throw new Error('Context required for first initialization');
            }
            ConversationHistoryManager.instance = new ConversationHistoryManager(context);
        }
        return ConversationHistoryManager.instance;
    }
    
    /**
     * Add entry to conversation history
     */
    async addEntry(
        message: Message,
        context?: ConversationEntry['context'],
        metadata?: Partial<ConversationEntry>
    ): Promise<string> {
        if (!this.currentSession) {
            this.startNewSession();
        }
        
        const entryId = this.generateId();
        const entry: ConversationEntry = {
            id: entryId,
            timestamp: new Date(),
            message,
            context,
            ...metadata
        };
        
        this.currentSession!.entries.push(entry);
        
        // Save locally
        this.saveCurrentSession();
        
        // Sync to backend if available
        if (this.backendConnector.isBackendAvailable()) {
            await this.syncEntryToBackend(entry);
        }
        
        // Check if we need to rotate history
        this.checkHistorySize();
        
        return entryId;
    }
    
    /**
     * Start a new conversation session
     */
    startNewSession(title?: string): string {
        // End current session if exists
        if (this.currentSession) {
            this.endCurrentSession();
        }
        
        const sessionId = this.generateId();
        this.currentSession = {
            id: sessionId,
            startTime: new Date(),
            entries: [],
            title: title || `Session ${new Date().toLocaleString()}`,
            tags: [],
            metadata: {
                vsCodeVersion: vscode.version,
                workspace: vscode.workspace.name
            }
        };
        
        this.sessions.set(sessionId, this.currentSession);
        return sessionId;
    }
    
    /**
     * End current session
     */
    endCurrentSession(): void {
        if (this.currentSession) {
            this.currentSession.endTime = new Date();
            this.saveCurrentSession();
            
            // Sync final session to backend
            if (this.backendConnector.isBackendAvailable()) {
                this.syncSessionToBackend(this.currentSession);
            }
            
            this.currentSession = null;
        }
    }
    
    /**
     * Get conversation history
     */
    getHistory(options?: {
        sessionId?: string;
        startDate?: Date;
        endDate?: Date;
        limit?: number;
        searchQuery?: string;
    }): ConversationEntry[] {
        let entries: ConversationEntry[] = [];
        
        if (options?.sessionId) {
            const session = this.sessions.get(options.sessionId);
            if (session) {
                entries = session.entries;
            }
        } else {
            // Get all entries
            for (const session of this.sessions.values()) {
                entries.push(...session.entries);
            }
        }
        
        // Apply filters
        if (options?.startDate) {
            entries = entries.filter(e => e.timestamp >= options.startDate!);
        }
        
        if (options?.endDate) {
            entries = entries.filter(e => e.timestamp <= options.endDate!);
        }
        
        if (options?.searchQuery) {
            const query = options.searchQuery.toLowerCase();
            entries = entries.filter(e => 
                e.message.content.toLowerCase().includes(query) ||
                JSON.stringify(e.context).toLowerCase().includes(query)
            );
        }
        
        // Sort by timestamp (newest first)
        entries.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
        
        // Apply limit
        if (options?.limit) {
            entries = entries.slice(0, options.limit);
        }
        
        return entries;
    }
    
    /**
     * Search conversations
     */
    async searchConversations(query: string): Promise<ConversationEntry[]> {
        const results: ConversationEntry[] = [];
        const searchTerms = query.toLowerCase().split(/\s+/);
        
        for (const session of this.sessions.values()) {
            for (const entry of session.entries) {
                const content = entry.message.content.toLowerCase();
                const contextStr = JSON.stringify(entry.context || {}).toLowerCase();
                
                const matches = searchTerms.every(term => 
                    content.includes(term) || contextStr.includes(term)
                );
                
                if (matches) {
                    results.push(entry);
                }
            }
        }
        
        return results;
    }
    
    /**
     * Get statistics
     */
    getStatistics(): ConversationStats {
        const allEntries = this.getHistory();
        const commandCounts = new Map<string, number>();
        let errorsSolved = 0;
        let ragUsageCount = 0;
        
        for (const entry of allEntries) {
            // Count commands
            const commandMatch = entry.message.content.match(/^(\w+)\s/);
            if (commandMatch) {
                const command = commandMatch[1];
                commandCounts.set(command, (commandCounts.get(command) || 0) + 1);
            }
            
            // Count solved errors
            if (entry.errorHandled && entry.solutionApplied) {
                errorsSolved++;
            }
            
            // Count RAG usage
            if (entry.ragUsed) {
                ragUsageCount++;
            }
        }
        
        // Calculate average session length
        const sessionLengths = Array.from(this.sessions.values())
            .map(s => s.entries.length);
        const avgSessionLength = sessionLengths.length > 0
            ? sessionLengths.reduce((a, b) => a + b, 0) / sessionLengths.length
            : 0;
        
        // Get most used commands
        const mostUsedCommands = Array.from(commandCounts.entries())
            .map(([command, count]) => ({ command, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 10);
        
        return {
            totalConversations: this.sessions.size,
            totalMessages: allEntries.length,
            averageSessionLength: avgSessionLength,
            mostUsedCommands,
            errorsSolved,
            ragUsageRate: allEntries.length > 0 ? ragUsageCount / allEntries.length : 0
        };
    }
    
    /**
     * Export history
     */
    exportHistory(format: 'json' | 'markdown' = 'json'): string {
        if (format === 'json') {
            return JSON.stringify({
                sessions: Array.from(this.sessions.values()),
                statistics: this.getStatistics(),
                exportDate: new Date()
            }, null, 2);
        } else {
            // Export as markdown
            let markdown = '# ADAM Conversation History\n\n';
            markdown += `Export Date: ${new Date().toLocaleString()}\n\n`;
            
            for (const session of this.sessions.values()) {
                markdown += `## ${session.title}\n`;
                markdown += `Started: ${session.startTime.toLocaleString()}\n`;
                if (session.endTime) {
                    markdown += `Ended: ${session.endTime.toLocaleString()}\n`;
                }
                markdown += '\n';
                
                for (const entry of session.entries) {
                    markdown += `### ${entry.timestamp.toLocaleTimeString()}\n`;
                    markdown += `**${entry.message.role}**: ${entry.message.content}\n`;
                    if (entry.context?.activeFile) {
                        markdown += `*File: ${entry.context.activeFile}*\n`;
                    }
                    markdown += '\n';
                }
                markdown += '---\n\n';
            }
            
            return markdown;
        }
    }
    
    /**
     * Clear old history
     */
    clearOldHistory(daysToKeep: number = 30): number {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);
        
        let removedCount = 0;
        
        for (const [sessionId, session] of this.sessions.entries()) {
            if (session.startTime < cutoffDate && session !== this.currentSession) {
                this.sessions.delete(sessionId);
                removedCount++;
            }
        }
        
        this.saveHistory();
        return removedCount;
    }
    
    /**
     * Initialize storage
     */
    private initializeStorage(): void {
        if (!fs.existsSync(this.storagePath)) {
            fs.mkdirSync(this.storagePath, { recursive: true });
        }
    }
    
    /**
     * Load history from storage
     */
    private loadHistory(): void {
        const historyFile = path.join(this.storagePath, 'conversation_history.json');
        
        try {
            if (fs.existsSync(historyFile)) {
                const data = JSON.parse(fs.readFileSync(historyFile, 'utf8'));
                
                // Convert sessions back to Map
                for (const session of data.sessions || []) {
                    // Convert date strings back to Date objects
                    session.startTime = new Date(session.startTime);
                    if (session.endTime) {
                        session.endTime = new Date(session.endTime);
                    }
                    for (const entry of session.entries) {
                        entry.timestamp = new Date(entry.timestamp);
                    }
                    
                    this.sessions.set(session.id, session);
                }
            }
        } catch (error) {
            console.error('Failed to load conversation history:', error);
        }
    }
    
    /**
     * Save history to storage
     */
    private saveHistory(): void {
        const historyFile = path.join(this.storagePath, 'conversation_history.json');
        
        try {
            const data = {
                sessions: Array.from(this.sessions.values()),
                timestamp: new Date()
            };
            
            fs.writeFileSync(historyFile, JSON.stringify(data, null, 2));
        } catch (error) {
            console.error('Failed to save conversation history:', error);
        }
    }
    
    /**
     * Save current session
     */
    private saveCurrentSession(): void {
        if (this.currentSession) {
            this.sessions.set(this.currentSession.id, this.currentSession);
            this.saveHistory();
        }
    }
    
    /**
     * Check history size and rotate if needed
     */
    private checkHistorySize(): void {
        let totalEntries = 0;
        for (const session of this.sessions.values()) {
            totalEntries += session.entries.length;
        }
        
        if (totalEntries > this.maxHistorySize) {
            // Remove oldest sessions
            const sortedSessions = Array.from(this.sessions.values())
                .sort((a, b) => a.startTime.getTime() - b.startTime.getTime());
            
            while (totalEntries > this.maxHistorySize && sortedSessions.length > 1) {
                const oldestSession = sortedSessions.shift()!;
                if (oldestSession !== this.currentSession) {
                    totalEntries -= oldestSession.entries.length;
                    this.sessions.delete(oldestSession.id);
                }
            }
            
            this.saveHistory();
        }
    }
    
    /**
     * Setup auto-sync with backend
     */
    private setupAutoSync(): void {
        // Sync every 5 minutes
        this.syncInterval = setInterval(() => {
            if (this.currentSession && this.backendConnector.isBackendAvailable()) {
                this.syncSessionToBackend(this.currentSession);
            }
        }, 5 * 60 * 1000);
    }
    
    /**
     * Sync entry to backend
     */
    private async syncEntryToBackend(entry: ConversationEntry): Promise<void> {
        try {
            await this.backendConnector.saveToMemory(
                entry.message.content,
                entry.message.role,
                {
                    type: 'conversation_entry',
                    ...entry
                }
            );
        } catch (error) {
            console.error('Failed to sync entry to backend:', error);
        }
    }
    
    /**
     * Sync session to backend
     */
    private async syncSessionToBackend(session: ConversationSession): Promise<void> {
        try {
            // Get conversation history from backend
            const backendHistory = await this.backendConnector.getConversationHistory();
            
            // Merge with local history
            // This is a simple implementation - could be more sophisticated
            const localEntries = session.entries.map(e => ({
                ...e.message,
                timestamp: e.timestamp,
                context: e.context
            }));
            
            // Save session summary to memory
            await this.backendConnector.saveToMemory(
                `Session: ${session.title}`,
                `${session.entries.length} messages`,
                {
                    type: 'conversation_session',
                    session
                }
            );
        } catch (error) {
            console.error('Failed to sync session to backend:', error);
        }
    }
    
    /**
     * Generate unique ID
     */
    private generateId(): string {
        return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    /**
     * Dispose resources
     */
    private dispose(): void {
        // End current session
        this.endCurrentSession();
        
        // Clear sync interval
        if (this.syncInterval) {
            clearInterval(this.syncInterval);
            this.syncInterval = null;
        }
        
        // Save final state
        this.saveHistory();
    }
}
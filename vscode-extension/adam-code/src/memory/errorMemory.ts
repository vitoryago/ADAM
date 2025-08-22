/**
 * Error Memory System for ADAM VSCode Extension
 * Learns from errors and solutions to improve over time
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { BackendConnector } from '../client/backendConnector';

export interface ErrorPattern {
    id: string;
    pattern: string;
    regex?: RegExp;
    category: string; // 'npm', 'typescript', 'runtime', 'build', 'test', etc.
    language?: string;
    framework?: string;
}

export interface ErrorSolution {
    id: string;
    errorPatternId: string;
    solution: string;
    commands?: string[];
    explanation: string;
    successRate: number;
    usageCount: number;
    userFeedback: number; // -1 to 1 scale
    verified: boolean;
    tags: string[];
}

export interface ErrorMemory {
    id: string;
    timestamp: Date;
    error: string;
    errorPatternId?: string;
    solutionId?: string;
    context: {
        file?: string;
        line?: number;
        command?: string;
        workspaceFolder?: string;
        language?: string;
    };
    solutionApplied?: string;
    solved: boolean;
    userFeedback?: string;
    metadata?: Record<string, any>;
}

export interface UserFeedback {
    memoryId: string;
    helpful: boolean;
    actualSolution?: string;
    notes?: string;
    timestamp: Date;
}

export class ErrorMemorySystem {
    private static instance: ErrorMemorySystem;
    
    private errorPatterns: Map<string, ErrorPattern> = new Map();
    private solutions: Map<string, ErrorSolution> = new Map();
    private memories: Map<string, ErrorMemory> = new Map();
    private feedbacks: Map<string, UserFeedback[]> = new Map();
    
    private backendConnector: BackendConnector;
    private storagePath: string;
    private learningEnabled: boolean = true;
    
    // Common error patterns
    private readonly commonPatterns: ErrorPattern[] = [
        {
            id: 'npm_module_not_found',
            pattern: 'Cannot find module',
            regex: /Cannot find module ['"]([^'"]+)['"]/,
            category: 'npm',
            language: 'javascript'
        },
        {
            id: 'typescript_type_error',
            pattern: 'Type .* is not assignable to type',
            regex: /Type '(.+)' is not assignable to type '(.+)'/,
            category: 'typescript',
            language: 'typescript'
        },
        {
            id: 'port_in_use',
            pattern: 'port .* is already in use',
            regex: /port (\d+) is already in use/i,
            category: 'runtime',
            framework: 'node'
        },
        {
            id: 'permission_denied',
            pattern: 'Permission denied',
            regex: /EACCES|Permission denied/,
            category: 'system'
        },
        {
            id: 'test_failure',
            pattern: 'Test failed',
            regex: /(\d+) test[s]? failed/,
            category: 'test'
        }
    ];
    
    private constructor(context: vscode.ExtensionContext) {
        this.storagePath = context.globalStorageUri.fsPath;
        this.backendConnector = new BackendConnector();
        
        // Initialize storage
        this.initializeStorage();
        
        // Load existing memories
        this.loadMemories();
        
        // Initialize common patterns
        this.initializePatterns();
    }
    
    static getInstance(context?: vscode.ExtensionContext): ErrorMemorySystem {
        if (!ErrorMemorySystem.instance) {
            if (!context) {
                throw new Error('Context required for first initialization');
            }
            ErrorMemorySystem.instance = new ErrorMemorySystem(context);
        }
        return ErrorMemorySystem.instance;
    }
    
    /**
     * Remember an error and its solution
     */
    async rememberError(
        error: string,
        solution?: string,
        context?: ErrorMemory['context'],
        solved: boolean = false
    ): Promise<string> {
        const memoryId = this.generateId();
        
        // Detect error pattern
        const pattern = this.detectErrorPattern(error);
        
        // Find existing solution if available
        let solutionId: string | undefined;
        if (pattern && !solution) {
            const existingSolution = this.findBestSolution(pattern.id);
            if (existingSolution) {
                solutionId = existingSolution.id;
                solution = existingSolution.solution;
            }
        }
        
        // Create memory entry
        const memory: ErrorMemory = {
            id: memoryId,
            timestamp: new Date(),
            error,
            errorPatternId: pattern?.id,
            solutionId,
            context: context || {},
            solutionApplied: solution,
            solved,
            metadata: {
                vsCodeVersion: vscode.version,
                extensionVersion: '0.1.2'
            }
        };
        
        this.memories.set(memoryId, memory);
        
        // Save to backend if available
        if (this.backendConnector.isBackendAvailable()) {
            await this.saveToBackend(memory);
        }
        
        // Save locally
        this.saveMemories();
        
        return memoryId;
    }
    
    /**
     * Learn from user feedback
     */
    async learnFromFeedback(
        memoryId: string,
        helpful: boolean,
        actualSolution?: string,
        notes?: string
    ): Promise<void> {
        const memory = this.memories.get(memoryId);
        if (!memory) {
            return;
        }
        
        // Create feedback entry
        const feedback: UserFeedback = {
            memoryId,
            helpful,
            actualSolution,
            notes,
            timestamp: new Date()
        };
        
        // Store feedback
        if (!this.feedbacks.has(memoryId)) {
            this.feedbacks.set(memoryId, []);
        }
        this.feedbacks.get(memoryId)!.push(feedback);
        
        // Update solution success rate if applicable
        if (memory.solutionId) {
            const solution = this.solutions.get(memory.solutionId);
            if (solution) {
                solution.usageCount++;
                if (helpful) {
                    solution.successRate = (solution.successRate * (solution.usageCount - 1) + 1) / solution.usageCount;
                    solution.userFeedback = Math.min(1, solution.userFeedback + 0.1);
                } else {
                    solution.successRate = (solution.successRate * (solution.usageCount - 1)) / solution.usageCount;
                    solution.userFeedback = Math.max(-1, solution.userFeedback - 0.1);
                }
            }
        }
        
        // If user provided actual solution, create or update solution
        if (actualSolution && memory.errorPatternId) {
            await this.addSolution(memory.errorPatternId, actualSolution, notes || 'User-provided solution');
        }
        
        // Update memory with feedback
        memory.userFeedback = notes;
        memory.solved = helpful || !!actualSolution;
        
        // Save changes
        this.saveMemories();
        
        // Sync with backend
        if (this.backendConnector.isBackendAvailable()) {
            await this.syncFeedbackToBackend(feedback, memory);
        }
    }
    
    /**
     * Search for similar errors in memory
     */
    async searchSimilarErrors(error: string, limit: number = 5): Promise<ErrorMemory[]> {
        const results: Array<{ memory: ErrorMemory; score: number }> = [];
        
        // Search through memories
        for (const memory of this.memories.values()) {
            const score = this.calculateSimilarity(error, memory.error);
            if (score > 0.5) {
                results.push({ memory, score });
            }
        }
        
        // Sort by score and return top results
        return results
            .sort((a, b) => b.score - a.score)
            .slice(0, limit)
            .map(r => r.memory);
    }
    
    /**
     * Get suggested solution for an error
     */
    async getSuggestedSolution(error: string): Promise<{
        solution: string;
        commands?: string[];
        confidence: number;
        explanation: string;
    } | null> {
        // Detect error pattern
        const pattern = this.detectErrorPattern(error);
        if (!pattern) {
            // Search for similar errors
            const similar = await this.searchSimilarErrors(error, 1);
            if (similar.length > 0 && similar[0].solutionApplied) {
                return {
                    solution: similar[0].solutionApplied,
                    confidence: 0.6,
                    explanation: 'Based on similar error encountered before'
                };
            }
            return null;
        }
        
        // Find best solution for pattern
        const solution = this.findBestSolution(pattern.id);
        if (!solution) {
            return null;
        }
        
        return {
            solution: solution.solution,
            commands: solution.commands,
            confidence: solution.successRate,
            explanation: solution.explanation
        };
    }
    
    /**
     * Add a new solution
     */
    async addSolution(
        errorPatternId: string,
        solution: string,
        explanation: string,
        commands?: string[]
    ): Promise<string> {
        const solutionId = this.generateId();
        
        const newSolution: ErrorSolution = {
            id: solutionId,
            errorPatternId,
            solution,
            commands,
            explanation,
            successRate: 0.5, // Start with neutral rate
            usageCount: 0,
            userFeedback: 0,
            verified: false,
            tags: []
        };
        
        this.solutions.set(solutionId, newSolution);
        this.saveMemories();
        
        return solutionId;
    }
    
    /**
     * Detect error pattern
     */
    private detectErrorPattern(error: string): ErrorPattern | null {
        for (const pattern of this.errorPatterns.values()) {
            if (pattern.regex) {
                if (pattern.regex.test(error)) {
                    return pattern;
                }
            } else if (error.includes(pattern.pattern)) {
                return pattern;
            }
        }
        return null;
    }
    
    /**
     * Find best solution for a pattern
     */
    private findBestSolution(errorPatternId: string): ErrorSolution | null {
        const solutions = Array.from(this.solutions.values())
            .filter(s => s.errorPatternId === errorPatternId)
            .sort((a, b) => {
                // Sort by success rate and user feedback
                const scoreA = a.successRate * 0.7 + (a.userFeedback + 1) * 0.3;
                const scoreB = b.successRate * 0.7 + (b.userFeedback + 1) * 0.3;
                return scoreB - scoreA;
            });
        
        return solutions[0] || null;
    }
    
    /**
     * Calculate similarity between two error strings
     */
    private calculateSimilarity(error1: string, error2: string): number {
        const words1 = error1.toLowerCase().split(/\s+/);
        const words2 = error2.toLowerCase().split(/\s+/);
        
        const set1 = new Set(words1);
        const set2 = new Set(words2);
        
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        
        return intersection.size / union.size;
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
     * Initialize common patterns
     */
    private initializePatterns(): void {
        for (const pattern of this.commonPatterns) {
            this.errorPatterns.set(pattern.id, pattern);
        }
    }
    
    /**
     * Load memories from storage
     */
    private loadMemories(): void {
        const memoriesFile = path.join(this.storagePath, 'error_memories.json');
        const solutionsFile = path.join(this.storagePath, 'error_solutions.json');
        const feedbacksFile = path.join(this.storagePath, 'error_feedbacks.json');
        
        try {
            if (fs.existsSync(memoriesFile)) {
                const data = JSON.parse(fs.readFileSync(memoriesFile, 'utf8'));
                this.memories = new Map(Object.entries(data.memories || {}));
            }
            
            if (fs.existsSync(solutionsFile)) {
                const data = JSON.parse(fs.readFileSync(solutionsFile, 'utf8'));
                this.solutions = new Map(Object.entries(data.solutions || {}));
            }
            
            if (fs.existsSync(feedbacksFile)) {
                const data = JSON.parse(fs.readFileSync(feedbacksFile, 'utf8'));
                this.feedbacks = new Map(Object.entries(data.feedbacks || {}));
            }
        } catch (error) {
            console.error('Failed to load error memories:', error);
        }
    }
    
    /**
     * Save memories to storage
     */
    private saveMemories(): void {
        const memoriesFile = path.join(this.storagePath, 'error_memories.json');
        const solutionsFile = path.join(this.storagePath, 'error_solutions.json');
        const feedbacksFile = path.join(this.storagePath, 'error_feedbacks.json');
        
        try {
            fs.writeFileSync(memoriesFile, JSON.stringify({
                memories: Object.fromEntries(this.memories),
                timestamp: new Date()
            }, null, 2));
            
            fs.writeFileSync(solutionsFile, JSON.stringify({
                solutions: Object.fromEntries(this.solutions),
                timestamp: new Date()
            }, null, 2));
            
            fs.writeFileSync(feedbacksFile, JSON.stringify({
                feedbacks: Object.fromEntries(this.feedbacks),
                timestamp: new Date()
            }, null, 2));
        } catch (error) {
            console.error('Failed to save error memories:', error);
        }
    }
    
    /**
     * Save to backend
     */
    private async saveToBackend(memory: ErrorMemory): Promise<void> {
        try {
            await this.backendConnector.saveToMemory(
                memory.error,
                memory.solutionApplied || 'No solution yet',
                {
                    type: 'error_memory',
                    ...memory
                }
            );
        } catch (error) {
            console.error('Failed to save to backend:', error);
        }
    }
    
    /**
     * Sync feedback to backend
     */
    private async syncFeedbackToBackend(feedback: UserFeedback, memory: ErrorMemory): Promise<void> {
        try {
            await this.backendConnector.saveToMemory(
                `Error feedback: ${memory.error}`,
                feedback.actualSolution || feedback.notes || 'Feedback provided',
                {
                    type: 'error_feedback',
                    feedback,
                    memory
                }
            );
        } catch (error) {
            console.error('Failed to sync feedback to backend:', error);
        }
    }
    
    /**
     * Generate unique ID
     */
    private generateId(): string {
        return `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    /**
     * Get statistics
     */
    getStatistics(): {
        totalMemories: number;
        solvedErrors: number;
        patterns: number;
        solutions: number;
        averageSuccessRate: number;
    } {
        const solved = Array.from(this.memories.values()).filter(m => m.solved).length;
        const avgSuccess = Array.from(this.solutions.values())
            .reduce((sum, s) => sum + s.successRate, 0) / Math.max(1, this.solutions.size);
        
        return {
            totalMemories: this.memories.size,
            solvedErrors: solved,
            patterns: this.errorPatterns.size,
            solutions: this.solutions.size,
            averageSuccessRate: avgSuccess
        };
    }
    
    /**
     * Export memories for analysis
     */
    exportMemories(): string {
        return JSON.stringify({
            memories: Array.from(this.memories.values()),
            solutions: Array.from(this.solutions.values()),
            patterns: Array.from(this.errorPatterns.values()),
            feedbacks: Array.from(this.feedbacks.values()),
            statistics: this.getStatistics(),
            exportDate: new Date()
        }, null, 2);
    }
}
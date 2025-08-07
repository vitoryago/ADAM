/**
 * Unified Memory Manager for ADAM
 * Ensures memory consistency across all interfaces
 */

import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

export interface Memory {
    id: string;
    timestamp: string;
    content: string;
    query: string;
    response: string;
    workspace?: string;
    project?: string;
    tags?: string[];
    embeddings?: number[];
}

export class UnifiedMemoryManager {
    private memoryBasePath: string;
    private currentProject: string;
    
    constructor(projectName?: string) {
        // Use a consistent location for all ADAM memory
        this.memoryBasePath = path.join(os.homedir(), '.adam', 'memory');
        this.currentProject = projectName || 'default';
        
        // Ensure directories exist
        this.ensureDirectories();
    }
    
    private ensureDirectories() {
        const projectPath = path.join(this.memoryBasePath, this.currentProject);
        if (!fs.existsSync(projectPath)) {
            fs.mkdirSync(projectPath, { recursive: true });
        }
    }
    
    /**
     * Get memory file path for current project
     */
    private getMemoryFile(): string {
        return path.join(this.memoryBasePath, this.currentProject, 'memories.json');
    }
    
    /**
     * Load all memories for current project
     */
    async loadMemories(): Promise<Memory[]> {
        const memoryFile = this.getMemoryFile();
        
        if (!fs.existsSync(memoryFile)) {
            return [];
        }
        
        try {
            const data = fs.readFileSync(memoryFile, 'utf8');
            return JSON.parse(data);
        } catch (error) {
            console.error('Error loading memories:', error);
            return [];
        }
    }
    
    /**
     * Save a new memory
     */
    async saveMemory(query: string, response: string, workspace?: string): Promise<void> {
        // Don't save trivial interactions
        if (query.length < 10 || response.length < 20) {
            return;
        }
        
        const memories = await this.loadMemories();
        
        const newMemory: Memory = {
            id: this.generateId(),
            timestamp: new Date().toISOString(),
            content: `Q: ${query}\nA: ${response.substring(0, 500)}`,
            query,
            response: response.substring(0, 2000), // Limit response size
            workspace: workspace || 'unknown',
            project: this.currentProject,
            tags: this.extractTags(query + ' ' + response)
        };
        
        memories.push(newMemory);
        
        // Keep only last 500 memories per project
        const recentMemories = memories.slice(-500);
        
        // Save to disk
        fs.writeFileSync(this.getMemoryFile(), JSON.stringify(recentMemories, null, 2));
    }
    
    /**
     * Search memories by query
     */
    async searchMemories(query: string, limit: number = 5): Promise<Memory[]> {
        const memories = await this.loadMemories();
        
        // Simple keyword matching
        const keywords = query.toLowerCase().split(' ').filter(w => w.length > 3);
        
        const scored = memories.map(mem => {
            const content = (mem.content + ' ' + mem.tags?.join(' ')).toLowerCase();
            const score = keywords.reduce((acc, keyword) => {
                return acc + (content.includes(keyword) ? 1 : 0);
            }, 0);
            return { memory: mem, score };
        });
        
        // Sort by score and recency
        scored.sort((a, b) => {
            if (b.score !== a.score) {
                return b.score - a.score;
            }
            // If scores are equal, prefer more recent
            return new Date(b.memory.timestamp).getTime() - new Date(a.memory.timestamp).getTime();
        });
        
        return scored.slice(0, limit).map(s => s.memory);
    }
    
    /**
     * Get memories from a specific time range
     */
    async getRecentMemories(hours: number = 24): Promise<Memory[]> {
        const memories = await this.loadMemories();
        const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
        
        return memories.filter(mem => new Date(mem.timestamp) > cutoff);
    }
    
    /**
     * Extract tags from content
     */
    private extractTags(content: string): string[] {
        const tags = new Set<string>();
        
        // Extract file extensions
        const fileExtensions = content.match(/\.\w{2,4}\b/g);
        if (fileExtensions) {
            fileExtensions.forEach(ext => tags.add(ext));
        }
        
        // Extract common programming terms
        const programmingTerms = [
            'function', 'class', 'database', 'sql', 'python', 'javascript',
            'typescript', 'react', 'node', 'api', 'test', 'debug', 'error',
            'performance', 'optimize', 'refactor', 'deploy', 'git', 'docker'
        ];
        
        const lowerContent = content.toLowerCase();
        programmingTerms.forEach(term => {
            if (lowerContent.includes(term)) {
                tags.add(term);
            }
        });
        
        return Array.from(tags);
    }
    
    /**
     * Generate unique ID
     */
    private generateId(): string {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
    
    /**
     * Switch to a different project
     */
    switchProject(projectName: string): void {
        this.currentProject = projectName;
        this.ensureDirectories();
    }
    
    /**
     * Get all projects with memories
     */
    getAllProjects(): string[] {
        if (!fs.existsSync(this.memoryBasePath)) {
            return [];
        }
        
        return fs.readdirSync(this.memoryBasePath)
            .filter(name => fs.statSync(path.join(this.memoryBasePath, name)).isDirectory());
    }
    
    /**
     * Clear memories for current project
     */
    async clearMemories(): Promise<void> {
        const memoryFile = this.getMemoryFile();
        if (fs.existsSync(memoryFile)) {
            fs.unlinkSync(memoryFile);
        }
    }
}
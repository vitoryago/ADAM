/**
 * File Watcher for ADAM VSCode Extension
 * Provides real-time file system monitoring and change detection
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { EventEmitter } from 'events';

export interface FileChangeEvent {
    type: 'created' | 'changed' | 'deleted' | 'renamed';
    path: string;
    oldPath?: string; // For rename events
    timestamp: Date;
    size?: number;
    content?: string; // For small files
}

export interface WatchOptions {
    pattern?: string;
    excludePattern?: string;
    includeContent?: boolean;
    debounceMs?: number;
    recursive?: boolean;
}

export class FileWatcher extends EventEmitter {
    private watchers: Map<string, vscode.FileSystemWatcher> = new Map();
    private changeBuffer: Map<string, FileChangeEvent> = new Map();
    private debounceTimers: Map<string, NodeJS.Timeout> = new Map();
    private fileHashes: Map<string, string> = new Map();
    
    /**
     * Start watching files/folders
     */
    async watch(
        pathOrPattern: string,
        options: WatchOptions = {},
        onChange?: (event: FileChangeEvent) => void
    ): Promise<{ id: string; stop: () => void }> {
        const watchId = `watch_${Date.now()}`;
        const globPattern = this.createGlobPattern(pathOrPattern, options);
        
        const watcher = vscode.workspace.createFileSystemWatcher(
            globPattern,
            false, // ignoreCreateEvents
            false, // ignoreChangeEvents
            false  // ignoreDeleteEvents
        );
        
        // Handle file creation
        watcher.onDidCreate(async (uri) => {
            const event = await this.createChangeEvent('created', uri, options);
            this.handleChange(event, options, onChange);
        });
        
        // Handle file changes
        watcher.onDidChange(async (uri) => {
            const event = await this.createChangeEvent('changed', uri, options);
            this.handleChange(event, options, onChange);
        });
        
        // Handle file deletion
        watcher.onDidDelete(async (uri) => {
            const event = await this.createChangeEvent('deleted', uri, options);
            this.handleChange(event, options, onChange);
        });
        
        this.watchers.set(watchId, watcher);
        
        return {
            id: watchId,
            stop: () => this.stopWatching(watchId)
        };
    }
    
    /**
     * Watch multiple patterns
     */
    async watchMultiple(
        patterns: string[],
        options: WatchOptions = {},
        onChange?: (event: FileChangeEvent) => void
    ): Promise<{ ids: string[]; stopAll: () => void }> {
        const watchIds: string[] = [];
        
        for (const pattern of patterns) {
            const { id } = await this.watch(pattern, options, onChange);
            watchIds.push(id);
        }
        
        return {
            ids: watchIds,
            stopAll: () => watchIds.forEach(id => this.stopWatching(id))
        };
    }
    
    /**
     * Watch for specific file changes with custom logic
     */
    async watchWithFilter(
        pattern: string,
        filter: (event: FileChangeEvent) => boolean,
        onChange: (event: FileChangeEvent) => void
    ): Promise<{ stop: () => void }> {
        return this.watch(pattern, {}, (event) => {
            if (filter(event)) {
                onChange(event);
            }
        });
    }
    
    /**
     * Watch and auto-reload on changes
     */
    async watchAndReload(
        pattern: string,
        reloadAction: () => Promise<void>,
        options: WatchOptions = {}
    ): Promise<{ stop: () => void }> {
        return this.watch(pattern, options, async (event) => {
            console.log(`File ${event.type}: ${event.path}, triggering reload...`);
            await reloadAction();
        });
    }
    
    /**
     * Watch configuration files
     */
    async watchConfig(
        onChange: (config: any, file: string) => void
    ): Promise<{ stop: () => void }> {
        const configPatterns = [
            '**/package.json',
            '**/.env',
            '**/.env.local',
            '**/tsconfig.json',
            '**/webpack.config.js',
            '**/.eslintrc*',
            '**/.prettierrc*'
        ];
        
        const { stopAll } = await this.watchMultiple(
            configPatterns,
            { includeContent: true },
            async (event) => {
                if (event.type !== 'deleted' && event.content) {
                    try {
                        const config = this.parseConfig(event.content, event.path);
                        onChange(config, event.path);
                    } catch (error) {
                        console.error(`Failed to parse config ${event.path}:`, error);
                    }
                }
            }
        );
        
        return { stop: stopAll };
    }
    
    /**
     * Watch test files and auto-run tests
     */
    async watchTests(
        testCommand: string,
        runTests: (file: string) => Promise<void>
    ): Promise<{ stop: () => void }> {
        const testPatterns = [
            '**/*.test.ts',
            '**/*.test.tsx',
            '**/*.test.js',
            '**/*.test.jsx',
            '**/*.spec.ts',
            '**/*.spec.tsx',
            '**/*.spec.js',
            '**/*.spec.jsx'
        ];
        
        const { stopAll } = await this.watchMultiple(
            testPatterns,
            { debounceMs: 1000 },
            async (event) => {
                if (event.type === 'changed' || event.type === 'created') {
                    console.log(`Test file ${event.type}: ${event.path}, running tests...`);
                    await runTests(event.path);
                }
            }
        );
        
        return { stop: stopAll };
    }
    
    /**
     * Watch for build output changes
     */
    async watchBuildOutput(
        outputDir: string,
        onBuildComplete: (files: string[]) => void
    ): Promise<{ stop: () => void }> {
        const buildFiles: Set<string> = new Set();
        let buildTimer: NodeJS.Timeout | undefined;
        
        return this.watch(
            `${outputDir}/**/*`,
            { debounceMs: 500 },
            (event) => {
                if (event.type === 'created' || event.type === 'changed') {
                    buildFiles.add(event.path);
                    
                    // Reset timer
                    if (buildTimer) {
                        clearTimeout(buildTimer);
                    }
                    
                    // Wait for build to complete (no new files for 2 seconds)
                    buildTimer = setTimeout(() => {
                        onBuildComplete(Array.from(buildFiles));
                        buildFiles.clear();
                    }, 2000);
                }
            }
        );
    }
    
    /**
     * Get file change history
     */
    getChangeHistory(): FileChangeEvent[] {
        return Array.from(this.changeBuffer.values())
            .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    }
    
    /**
     * Stop watching
     */
    private stopWatching(watchId: string): void {
        const watcher = this.watchers.get(watchId);
        if (watcher) {
            watcher.dispose();
            this.watchers.delete(watchId);
        }
    }
    
    /**
     * Stop all watchers
     */
    stopAll(): void {
        for (const [id, watcher] of this.watchers) {
            watcher.dispose();
        }
        this.watchers.clear();
        this.changeBuffer.clear();
        
        // Clear all timers
        for (const timer of this.debounceTimers.values()) {
            clearTimeout(timer);
        }
        this.debounceTimers.clear();
    }
    
    /**
     * Create change event
     */
    private async createChangeEvent(
        type: FileChangeEvent['type'],
        uri: vscode.Uri,
        options: WatchOptions
    ): Promise<FileChangeEvent> {
        const filePath = uri.fsPath;
        const event: FileChangeEvent = {
            type,
            path: vscode.workspace.asRelativePath(filePath),
            timestamp: new Date()
        };
        
        if (type !== 'deleted') {
            try {
                const stats = fs.statSync(filePath);
                event.size = stats.size;
                
                // Include content for small files if requested
                if (options.includeContent && stats.size < 100000) { // 100KB limit
                    event.content = fs.readFileSync(filePath, 'utf8');
                }
            } catch (error) {
                // File might have been deleted between events
            }
        }
        
        return event;
    }
    
    /**
     * Handle change with debouncing
     */
    private handleChange(
        event: FileChangeEvent,
        options: WatchOptions,
        onChange?: (event: FileChangeEvent) => void
    ): void {
        const debounceMs = options.debounceMs || 100;
        
        // Clear existing timer for this file
        const existingTimer = this.debounceTimers.get(event.path);
        if (existingTimer) {
            clearTimeout(existingTimer);
        }
        
        // Set new timer
        const timer = setTimeout(() => {
            this.changeBuffer.set(event.path, event);
            this.emit('change', event);
            
            if (onChange) {
                onChange(event);
            }
            
            this.debounceTimers.delete(event.path);
        }, debounceMs);
        
        this.debounceTimers.set(event.path, timer);
    }
    
    /**
     * Create glob pattern from path
     */
    private createGlobPattern(pathOrPattern: string, options: WatchOptions): vscode.GlobPattern {
        if (pathOrPattern.includes('*')) {
            // Already a glob pattern
            return pathOrPattern;
        }
        
        // Check if it's a directory
        try {
            const stats = fs.statSync(pathOrPattern);
            if (stats.isDirectory()) {
                return options.recursive !== false 
                    ? `${pathOrPattern}/**/*`
                    : `${pathOrPattern}/*`;
            }
        } catch {
            // Not a valid path, treat as pattern
        }
        
        return pathOrPattern;
    }
    
    /**
     * Parse configuration file
     */
    private parseConfig(content: string, filePath: string): any {
        const ext = path.extname(filePath);
        
        if (ext === '.json') {
            return JSON.parse(content);
        }
        
        if (ext === '.js' || ext === '.ts') {
            // For JS/TS configs, we can't eval them safely
            // Return the content for analysis
            return { raw: content };
        }
        
        if (filePath.includes('.env')) {
            // Parse .env file
            const config: Record<string, string> = {};
            const lines = content.split('\n');
            
            for (const line of lines) {
                const match = line.match(/^([^=]+)=(.*)$/);
                if (match) {
                    config[match[1].trim()] = match[2].trim();
                }
            }
            
            return config;
        }
        
        return { raw: content };
    }
    
    /**
     * Calculate file hash for change detection
     */
    private calculateHash(content: string): string {
        // Simple hash for change detection
        let hash = 0;
        for (let i = 0; i < content.length; i++) {
            const char = content.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return hash.toString(36);
    }
}
/**
 * Search Operations for ADAM VSCode Extension
 * Provides file search, grep, symbol finding, and code navigation
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export interface SearchResult {
    file: string;
    line: number;
    column: number;
    match: string;
    preview: string;
}

export interface SearchOptions {
    pattern: string;
    includePattern?: string;
    excludePattern?: string;
    maxResults?: number;
    caseSensitive?: boolean;
    wholeWord?: boolean;
    regex?: boolean;
    fileType?: string;
}

export class SearchOperations {
    
    /**
     * Search for text across all files using ripgrep or fallback
     */
    async searchInFiles(options: SearchOptions): Promise<SearchResult[]> {
        try {
            // Try ripgrep first (fastest)
            const rgAvailable = await this.commandExists('rg');
            if (rgAvailable) {
                return await this.searchWithRipgrep(options);
            }
            
            // Fallback to VSCode API
            return await this.searchWithVSCode(options);
        } catch (error: any) {
            console.error('Search failed:', error);
            return [];
        }
    }
    
    /**
     * Search using ripgrep
     */
    private async searchWithRipgrep(options: SearchOptions): Promise<SearchResult[]> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return [];
        }
        
        let command = 'rg';
        
        // Add search pattern
        if (options.regex) {
            command += ` "${options.pattern}"`;
        } else {
            command += ` --fixed-strings "${options.pattern}"`;
        }
        
        // Add options
        if (!options.caseSensitive) {
            command += ' -i';
        }
        if (options.wholeWord) {
            command += ' -w';
        }
        if (options.maxResults) {
            command += ` -m ${options.maxResults}`;
        }
        if (options.fileType) {
            command += ` -t ${options.fileType}`;
        }
        if (options.includePattern) {
            command += ` -g "${options.includePattern}"`;
        }
        if (options.excludePattern) {
            command += ` -g "!${options.excludePattern}"`;
        }
        
        // Output format
        command += ' --json';
        
        try {
            const { stdout } = await execAsync(command, {
                cwd: workspaceFolder.uri.fsPath,
                maxBuffer: 10 * 1024 * 1024
            });
            
            return this.parseRipgrepOutput(stdout);
        } catch (error: any) {
            // Ripgrep returns exit code 1 when no matches found
            if (error.code === 1) {
                return [];
            }
            throw error;
        }
    }
    
    /**
     * Parse ripgrep JSON output
     */
    private parseRipgrepOutput(output: string): SearchResult[] {
        const results: SearchResult[] = [];
        const lines = output.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
            try {
                const data = JSON.parse(line);
                if (data.type === 'match') {
                    const match = data.data;
                    results.push({
                        file: match.path.text,
                        line: match.line_number,
                        column: match.submatches[0]?.start || 0,
                        match: match.submatches[0]?.match?.text || '',
                        preview: match.lines.text
                    });
                }
            } catch {
                // Skip invalid JSON lines
            }
        }
        
        return results;
    }
    
    /**
     * Search using VSCode API
     */
    private async searchWithVSCode(options: SearchOptions): Promise<SearchResult[]> {
        const results: SearchResult[] = [];
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        
        if (!workspaceFolder) {
            return [];
        }
        
        // Use findFiles to get all files first
        const includePattern = options.includePattern || '**/*';
        const excludePattern = options.excludePattern || '**/node_modules/**';
        
        const files = await vscode.workspace.findFiles(includePattern, excludePattern);
        
        for (const file of files) {
            try {
                const document = await vscode.workspace.openTextDocument(file);
                const text = document.getText();
                const lines = text.split('\n');
                
                // Create regex for searching
                let regex: RegExp;
                if (options.regex) {
                    regex = new RegExp(options.pattern, options.caseSensitive ? 'g' : 'gi');
                } else if (options.wholeWord) {
                    const escapedPattern = options.pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                    regex = new RegExp(`\\b${escapedPattern}\\b`, options.caseSensitive ? 'g' : 'gi');
                } else {
                    const escapedPattern = options.pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                    regex = new RegExp(escapedPattern, options.caseSensitive ? 'g' : 'gi');
                }
                
                // Search through lines
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    const matches = [...line.matchAll(regex)];
                    
                    for (const match of matches) {
                        results.push({
                            file: vscode.workspace.asRelativePath(file),
                            line: i + 1,
                            column: (match.index || 0) + 1,
                            match: match[0],
                            preview: line
                        });
                        
                        if (options.maxResults && results.length >= options.maxResults) {
                            return results;
                        }
                    }
                }
            } catch (error) {
                // Skip files that can't be opened (binary files, etc.)
                continue;
            }
        }
        
        return results;
    }
    
    /**
     * Find symbol by name
     */
    async findSymbol(symbolName: string, symbolKind?: vscode.SymbolKind): Promise<vscode.SymbolInformation[]> {
        const symbols = await vscode.commands.executeCommand<vscode.SymbolInformation[]>(
            'vscode.executeWorkspaceSymbolProvider',
            symbolName
        );
        
        if (symbolKind !== undefined) {
            return symbols.filter(s => s.kind === symbolKind);
        }
        
        return symbols;
    }
    
    /**
     * Find all references to a symbol
     */
    async findReferences(uri: vscode.Uri, position: vscode.Position): Promise<vscode.Location[]> {
        return await vscode.commands.executeCommand<vscode.Location[]>(
            'vscode.executeReferenceProvider',
            uri,
            position
        );
    }
    
    /**
     * Find definition of symbol at position
     */
    async findDefinition(uri: vscode.Uri, position: vscode.Position): Promise<vscode.Location[]> {
        const definitions = await vscode.commands.executeCommand<vscode.Location | vscode.Location[]>(
            'vscode.executeDefinitionProvider',
            uri,
            position
        );
        
        return Array.isArray(definitions) ? definitions : [definitions];
    }
    
    /**
     * Find all implementations
     */
    async findImplementations(uri: vscode.Uri, position: vscode.Position): Promise<vscode.Location[]> {
        return await vscode.commands.executeCommand<vscode.Location[]>(
            'vscode.executeImplementationProvider',
            uri,
            position
        );
    }
    
    /**
     * Search for files by name pattern
     */
    async findFiles(pattern: string, exclude?: string): Promise<string[]> {
        const files = await vscode.workspace.findFiles(
            pattern,
            exclude,
            undefined
        );
        
        return files.map(uri => vscode.workspace.asRelativePath(uri));
    }
    
    /**
     * Search for specific code patterns (functions, classes, etc.)
     */
    async findCodePatterns(pattern: string, language?: string): Promise<SearchResult[]> {
        const languagePatterns: Record<string, string> = {
            'typescript': '\\.(ts|tsx)$',
            'javascript': '\\.(js|jsx)$',
            'python': '\\.py$',
            'java': '\\.java$',
            'csharp': '\\.cs$',
            'go': '\\.go$',
            'rust': '\\.rs$'
        };
        
        const options: SearchOptions = {
            pattern,
            regex: true,
            includePattern: language ? languagePatterns[language] : undefined
        };
        
        return this.searchInFiles(options);
    }
    
    /**
     * Find TODO/FIXME comments
     */
    async findTodos(): Promise<SearchResult[]> {
        return this.searchInFiles({
            pattern: '(TODO|FIXME|HACK|XXX|NOTE|BUG)\\s*:',
            regex: true
        });
    }
    
    /**
     * Find imports/requires
     */
    async findImports(moduleName: string): Promise<SearchResult[]> {
        const patterns = [
            `import.*${moduleName}`,
            `require.*${moduleName}`,
            `from\\s+['"]${moduleName}['"]`
        ];
        
        return this.searchInFiles({
            pattern: patterns.join('|'),
            regex: true
        });
    }
    
    /**
     * Find unused variables/functions
     */
    async findUnusedCode(): Promise<string[]> {
        // This would require more sophisticated analysis
        // For now, return a placeholder
        const diagnostics = vscode.languages.getDiagnostics();
        const unused: string[] = [];
        
        for (const [uri, diags] of diagnostics) {
            for (const diag of diags) {
                if (diag.message.includes('unused') || 
                    diag.message.includes('never used') ||
                    diag.message.includes('never read')) {
                    const file = vscode.workspace.asRelativePath(uri);
                    const line = diag.range.start.line + 1;
                    unused.push(`${file}:${line} - ${diag.message}`);
                }
            }
        }
        
        return unused;
    }
    
    /**
     * Get call hierarchy for a function
     */
    async getCallHierarchy(uri: vscode.Uri, position: vscode.Position): Promise<vscode.CallHierarchyItem[]> {
        const items = await vscode.commands.executeCommand<vscode.CallHierarchyItem[]>(
            'vscode.prepareCallHierarchy',
            uri,
            position
        );
        
        return items || [];
    }
    
    /**
     * Replace text across multiple files
     */
    async replaceInFiles(
        searchPattern: string,
        replacement: string,
        files?: string[]
    ): Promise<{ filesChanged: number; occurrencesReplaced: number }> {
        const searchResults = await this.searchInFiles({
            pattern: searchPattern,
            includePattern: files ? `{${files.join(',')}}` : undefined
        });
        
        const fileEdits = new Map<string, vscode.TextEdit[]>();
        
        for (const result of searchResults) {
            const uri = vscode.Uri.file(
                path.join(vscode.workspace.workspaceFolders![0].uri.fsPath, result.file)
            );
            
            if (!fileEdits.has(result.file)) {
                fileEdits.set(result.file, []);
            }
            
            const edit = new vscode.TextEdit(
                new vscode.Range(
                    result.line - 1,
                    result.column - 1,
                    result.line - 1,
                    result.column - 1 + result.match.length
                ),
                replacement
            );
            
            fileEdits.get(result.file)!.push(edit);
        }
        
        const workspaceEdit = new vscode.WorkspaceEdit();
        
        for (const [file, edits] of fileEdits) {
            const uri = vscode.Uri.file(
                path.join(vscode.workspace.workspaceFolders![0].uri.fsPath, file)
            );
            workspaceEdit.set(uri, edits);
        }
        
        const success = await vscode.workspace.applyEdit(workspaceEdit);
        
        return {
            filesChanged: fileEdits.size,
            occurrencesReplaced: searchResults.length
        };
    }
    
    /**
     * Check if command exists
     */
    private async commandExists(command: string): Promise<boolean> {
        try {
            await execAsync(`which ${command}`);
            return true;
        } catch {
            return false;
        }
    }
    
    /**
     * Open search result in editor
     */
    async openSearchResult(result: SearchResult): Promise<void> {
        const uri = vscode.Uri.file(
            path.join(vscode.workspace.workspaceFolders![0].uri.fsPath, result.file)
        );
        
        const doc = await vscode.workspace.openTextDocument(uri);
        const editor = await vscode.window.showTextDocument(doc);
        
        const position = new vscode.Position(result.line - 1, result.column - 1);
        editor.selection = new vscode.Selection(position, position);
        editor.revealRange(new vscode.Range(position, position));
    }
}
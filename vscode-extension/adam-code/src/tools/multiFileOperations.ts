/**
 * Multi-File Operations for ADAM VSCode Extension
 * Provides batch operations across multiple files
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { FileOperations } from './fileOperations';
import { SearchOperations } from './searchOperations';

export interface BatchFileOperation {
    type: 'create' | 'edit' | 'delete' | 'rename' | 'move';
    path: string;
    newPath?: string;
    content?: string;
    edits?: Array<{ oldText: string; newText: string }>;
}

export interface RefactorOperation {
    type: 'rename-symbol' | 'extract-method' | 'extract-variable' | 'inline';
    symbol: string;
    newName?: string;
    scope?: 'file' | 'project';
}

export class MultiFileOperations {
    private fileOps: FileOperations;
    private searchOps: SearchOperations;
    
    constructor() {
        this.fileOps = new FileOperations();
        this.searchOps = new SearchOperations();
    }
    
    /**
     * Execute batch file operations
     */
    async executeBatchOperations(operations: BatchFileOperation[]): Promise<{
        success: boolean;
        results: Array<{ operation: BatchFileOperation; result: string; success: boolean }>;
    }> {
        const results: Array<{ operation: BatchFileOperation; result: string; success: boolean }> = [];
        let allSuccess = true;
        
        for (const op of operations) {
            try {
                let result: { success: boolean; message: string };
                
                switch (op.type) {
                    case 'create':
                        result = await this.fileOps.createFile(op.path, op.content || '');
                        break;
                        
                    case 'edit':
                        if (op.edits) {
                            result = await this.fileOps.editFile(op.path, op.edits);
                        } else {
                            result = { success: false, message: 'No edits provided' };
                        }
                        break;
                        
                    case 'delete':
                        result = await this.fileOps.deleteFile(op.path);
                        break;
                        
                    case 'rename':
                    case 'move':
                        if (op.newPath) {
                            result = await this.moveFile(op.path, op.newPath);
                        } else {
                            result = { success: false, message: 'No new path provided' };
                        }
                        break;
                        
                    default:
                        result = { success: false, message: 'Unknown operation type' };
                }
                
                results.push({
                    operation: op,
                    result: result.message,
                    success: result.success
                });
                
                if (!result.success) {
                    allSuccess = false;
                }
            } catch (error: any) {
                results.push({
                    operation: op,
                    result: `Error: ${error.message}`,
                    success: false
                });
                allSuccess = false;
            }
        }
        
        return { success: allSuccess, results };
    }
    
    /**
     * Rename symbol across all files
     */
    async renameSymbolEverywhere(oldName: string, newName: string): Promise<{
        success: boolean;
        filesChanged: number;
        occurrencesReplaced: number;
    }> {
        // First find all occurrences
        const searchResults = await this.searchOps.searchInFiles({
            pattern: `\\b${oldName}\\b`,
            regex: true
        });
        
        if (searchResults.length === 0) {
            return { success: false, filesChanged: 0, occurrencesReplaced: 0 };
        }
        
        // Group by file
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
                    result.column - 1 + oldName.length
                ),
                newName
            );
            
            fileEdits.get(result.file)!.push(edit);
        }
        
        // Apply all edits
        const workspaceEdit = new vscode.WorkspaceEdit();
        
        for (const [file, edits] of fileEdits) {
            const uri = vscode.Uri.file(
                path.join(vscode.workspace.workspaceFolders![0].uri.fsPath, file)
            );
            workspaceEdit.set(uri, edits);
        }
        
        const success = await vscode.workspace.applyEdit(workspaceEdit);
        
        return {
            success,
            filesChanged: fileEdits.size,
            occurrencesReplaced: searchResults.length
        };
    }
    
    /**
     * Create multiple related files (e.g., component with test and styles)
     */
    async createComponentFiles(componentName: string, template: 'react' | 'vue' | 'angular'): Promise<{
        success: boolean;
        filesCreated: string[];
    }> {
        const filesCreated: string[] = [];
        
        try {
            const componentDir = `src/components/${componentName}`;
            
            // Create directory
            await this.createDirectory(componentDir);
            
            // Create files based on template
            if (template === 'react') {
                // Component file
                const componentContent = this.getReactComponentTemplate(componentName);
                await this.fileOps.createFile(
                    `${componentDir}/${componentName}.tsx`,
                    componentContent
                );
                filesCreated.push(`${componentDir}/${componentName}.tsx`);
                
                // Test file
                const testContent = this.getReactTestTemplate(componentName);
                await this.fileOps.createFile(
                    `${componentDir}/${componentName}.test.tsx`,
                    testContent
                );
                filesCreated.push(`${componentDir}/${componentName}.test.tsx`);
                
                // Styles file
                const stylesContent = this.getStylesTemplate(componentName);
                await this.fileOps.createFile(
                    `${componentDir}/${componentName}.module.css`,
                    stylesContent
                );
                filesCreated.push(`${componentDir}/${componentName}.module.css`);
                
                // Index file
                await this.fileOps.createFile(
                    `${componentDir}/index.ts`,
                    `export { default } from './${componentName}';\n`
                );
                filesCreated.push(`${componentDir}/index.ts`);
            }
            // Add more templates as needed
            
            return { success: true, filesCreated };
        } catch (error: any) {
            return { success: false, filesCreated };
        }
    }
    
    /**
     * Copy files matching a pattern to a new location
     */
    async copyFiles(pattern: string, destination: string): Promise<{
        success: boolean;
        filesCopied: number;
    }> {
        const files = await vscode.workspace.findFiles(pattern);
        let filesCopied = 0;
        
        for (const file of files) {
            try {
                const content = fs.readFileSync(file.fsPath, 'utf8');
                const fileName = path.basename(file.fsPath);
                const destPath = path.join(destination, fileName);
                
                await this.fileOps.createFile(destPath, content);
                filesCopied++;
            } catch (error) {
                console.error(`Failed to copy ${file.fsPath}:`, error);
            }
        }
        
        return { success: filesCopied > 0, filesCopied };
    }
    
    /**
     * Delete files matching a pattern
     */
    async deleteFiles(pattern: string, confirm: boolean = true): Promise<{
        success: boolean;
        filesDeleted: number;
    }> {
        const files = await vscode.workspace.findFiles(pattern);
        
        if (files.length === 0) {
            return { success: false, filesDeleted: 0 };
        }
        
        if (confirm) {
            const answer = await vscode.window.showWarningMessage(
                `Delete ${files.length} files matching "${pattern}"?`,
                'Yes', 'No'
            );
            
            if (answer !== 'Yes') {
                return { success: false, filesDeleted: 0 };
            }
        }
        
        let filesDeleted = 0;
        
        for (const file of files) {
            try {
                fs.unlinkSync(file.fsPath);
                filesDeleted++;
            } catch (error) {
                console.error(`Failed to delete ${file.fsPath}:`, error);
            }
        }
        
        return { success: filesDeleted > 0, filesDeleted };
    }
    
    /**
     * Apply a transformation to multiple files
     */
    async transformFiles(
        pattern: string,
        transformer: (content: string, filePath: string) => string
    ): Promise<{
        success: boolean;
        filesTransformed: number;
    }> {
        const files = await vscode.workspace.findFiles(pattern);
        let filesTransformed = 0;
        
        for (const file of files) {
            try {
                const content = fs.readFileSync(file.fsPath, 'utf8');
                const transformed = transformer(content, file.fsPath);
                
                if (content !== transformed) {
                    fs.writeFileSync(file.fsPath, transformed, 'utf8');
                    filesTransformed++;
                }
            } catch (error) {
                console.error(`Failed to transform ${file.fsPath}:`, error);
            }
        }
        
        return { success: filesTransformed > 0, filesTransformed };
    }
    
    /**
     * Add imports to multiple files
     */
    async addImportToFiles(
        pattern: string,
        importStatement: string
    ): Promise<{
        success: boolean;
        filesTransformed: number;
    }> {
        return this.transformFiles(pattern, (content) => {
            // Check if import already exists
            if (content.includes(importStatement)) {
                return content;
            }
            
            // Find the last import statement
            const importRegex = /^import\s+.*$/gm;
            const matches = content.match(importRegex);
            
            if (matches) {
                const lastImport = matches[matches.length - 1];
                const lastImportIndex = content.lastIndexOf(lastImport);
                const insertPosition = lastImportIndex + lastImport.length;
                
                return (
                    content.slice(0, insertPosition) +
                    '\n' + importStatement +
                    content.slice(insertPosition)
                );
            } else {
                // No imports found, add at the beginning
                return importStatement + '\n\n' + content;
            }
        });
    }
    
    /**
     * Update package.json in multiple directories
     */
    async updatePackageJsons(
        pattern: string,
        updates: Record<string, any>
    ): Promise<{
        success: boolean;
        filesUpdated: number;
    }> {
        const files = await vscode.workspace.findFiles(pattern);
        let filesUpdated = 0;
        
        for (const file of files) {
            try {
                const content = JSON.parse(fs.readFileSync(file.fsPath, 'utf8'));
                const updated = { ...content, ...updates };
                
                fs.writeFileSync(file.fsPath, JSON.stringify(updated, null, 2), 'utf8');
                filesUpdated++;
            } catch (error) {
                console.error(`Failed to update ${file.fsPath}:`, error);
            }
        }
        
        return { success: filesUpdated > 0, filesUpdated };
    }
    
    // Helper methods
    
    private async moveFile(oldPath: string, newPath: string): Promise<{ success: boolean; message: string }> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) {
                return { success: false, message: 'No workspace folder' };
            }
            
            const absoluteOldPath = path.isAbsolute(oldPath) 
                ? oldPath 
                : path.join(workspaceFolder.uri.fsPath, oldPath);
                
            const absoluteNewPath = path.isAbsolute(newPath)
                ? newPath
                : path.join(workspaceFolder.uri.fsPath, newPath);
            
            // Ensure destination directory exists
            const destDir = path.dirname(absoluteNewPath);
            if (!fs.existsSync(destDir)) {
                fs.mkdirSync(destDir, { recursive: true });
            }
            
            fs.renameSync(absoluteOldPath, absoluteNewPath);
            
            return { success: true, message: `Moved ${oldPath} to ${newPath}` };
        } catch (error: any) {
            return { success: false, message: error.message };
        }
    }
    
    private async createDirectory(dirPath: string): Promise<void> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder');
        }
        
        const absolutePath = path.isAbsolute(dirPath)
            ? dirPath
            : path.join(workspaceFolder.uri.fsPath, dirPath);
            
        if (!fs.existsSync(absolutePath)) {
            fs.mkdirSync(absolutePath, { recursive: true });
        }
    }
    
    private getReactComponentTemplate(name: string): string {
        return `import React from 'react';
import styles from './${name}.module.css';

interface ${name}Props {
    // Add props here
}

const ${name}: React.FC<${name}Props> = (props) => {
    return (
        <div className={styles.container}>
            <h2>${name} Component</h2>
            {/* Add component content here */}
        </div>
    );
};

export default ${name};
`;
    }
    
    private getReactTestTemplate(name: string): string {
        return `import React from 'react';
import { render, screen } from '@testing-library/react';
import ${name} from './${name}';

describe('${name}', () => {
    it('renders without crashing', () => {
        render(<${name} />);
        expect(screen.getByText('${name} Component')).toBeInTheDocument();
    });
    
    // Add more tests here
});
`;
    }
    
    private getStylesTemplate(name: string): string {
        return `.container {
    /* Add styles for ${name} component */
    padding: 1rem;
    margin: 0;
}
`;
    }
}
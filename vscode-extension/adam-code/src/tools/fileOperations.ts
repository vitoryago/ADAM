/**
 * File Operations for ADAM VSCode Extension
 * Provides file creation, editing, and deletion capabilities
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class FileOperations {
    
    /**
     * Create a new file with content
     */
    async createFile(filePath: string, content: string): Promise<{ success: boolean; message: string }> {
        try {
            // Resolve path relative to workspace
            const absolutePath = this.resolveFilePath(filePath);
            
            // Check if file already exists
            if (fs.existsSync(absolutePath)) {
                const overwrite = await vscode.window.showQuickPick(['Yes', 'No'], {
                    placeHolder: `File ${path.basename(filePath)} already exists. Overwrite?`
                });
                
                if (overwrite !== 'Yes') {
                    return { success: false, message: 'File creation cancelled' };
                }
            }
            
            // Ensure directory exists
            const dir = path.dirname(absolutePath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            
            // Write file
            fs.writeFileSync(absolutePath, content, 'utf8');
            
            // Open the file in editor
            const doc = await vscode.workspace.openTextDocument(absolutePath);
            await vscode.window.showTextDocument(doc);
            
            return { success: true, message: `Created file: ${filePath}` };
        } catch (error: any) {
            return { success: false, message: `Failed to create file: ${error.message}` };
        }
    }
    
    /**
     * Edit an existing file
     */
    async editFile(filePath: string, edits: Array<{ oldText: string; newText: string }>): Promise<{ success: boolean; message: string }> {
        try {
            const absolutePath = this.resolveFilePath(filePath);
            
            if (!fs.existsSync(absolutePath)) {
                return { success: false, message: `File not found: ${filePath}` };
            }
            
            // Read current content
            let content = fs.readFileSync(absolutePath, 'utf8');
            
            // Apply edits
            let editCount = 0;
            for (const edit of edits) {
                if (content.includes(edit.oldText)) {
                    content = content.replace(edit.oldText, edit.newText);
                    editCount++;
                }
            }
            
            if (editCount === 0) {
                return { success: false, message: 'No matching text found to edit' };
            }
            
            // Write back
            fs.writeFileSync(absolutePath, content, 'utf8');
            
            // Refresh the file if it's open
            const openDoc = vscode.workspace.textDocuments.find(
                doc => doc.uri.fsPath === absolutePath
            );
            if (openDoc) {
                await vscode.commands.executeCommand('workbench.action.files.revert');
            }
            
            return { success: true, message: `Edited ${editCount} sections in ${filePath}` };
        } catch (error: any) {
            return { success: false, message: `Failed to edit file: ${error.message}` };
        }
    }
    
    /**
     * Delete a file
     */
    async deleteFile(filePath: string): Promise<{ success: boolean; message: string }> {
        try {
            const absolutePath = this.resolveFilePath(filePath);
            
            if (!fs.existsSync(absolutePath)) {
                return { success: false, message: `File not found: ${filePath}` };
            }
            
            // Confirm deletion
            const confirm = await vscode.window.showQuickPick(['Yes', 'No'], {
                placeHolder: `Are you sure you want to delete ${path.basename(filePath)}?`
            });
            
            if (confirm !== 'Yes') {
                return { success: false, message: 'Deletion cancelled' };
            }
            
            // Close the file if it's open
            const openDoc = vscode.workspace.textDocuments.find(
                doc => doc.uri.fsPath === absolutePath
            );
            if (openDoc) {
                await vscode.window.showTextDocument(openDoc);
                await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
            }
            
            // Delete file
            fs.unlinkSync(absolutePath);
            
            return { success: true, message: `Deleted file: ${filePath}` };
        } catch (error: any) {
            return { success: false, message: `Failed to delete file: ${error.message}` };
        }
    }
    
    /**
     * Read a file's content
     */
    async readFile(filePath: string): Promise<{ success: boolean; content?: string; message?: string }> {
        try {
            const absolutePath = this.resolveFilePath(filePath);
            
            if (!fs.existsSync(absolutePath)) {
                return { success: false, message: `File not found: ${filePath}` };
            }
            
            const content = fs.readFileSync(absolutePath, 'utf8');
            return { success: true, content };
        } catch (error: any) {
            return { success: false, message: `Failed to read file: ${error.message}` };
        }
    }
    
    /**
     * List files in a directory
     */
    async listFiles(dirPath: string = '.'): Promise<{ success: boolean; files?: string[]; message?: string }> {
        try {
            const absolutePath = this.resolveFilePath(dirPath);
            
            if (!fs.existsSync(absolutePath)) {
                return { success: false, message: `Directory not found: ${dirPath}` };
            }
            
            const items = fs.readdirSync(absolutePath);
            const files: string[] = [];
            
            for (const item of items) {
                const itemPath = path.join(absolutePath, item);
                const stats = fs.statSync(itemPath);
                
                if (stats.isDirectory()) {
                    files.push(`📁 ${item}/`);
                } else {
                    files.push(`📄 ${item}`);
                }
            }
            
            return { success: true, files };
        } catch (error: any) {
            return { success: false, message: `Failed to list files: ${error.message}` };
        }
    }
    
    /**
     * Resolve file path relative to workspace
     */
    private resolveFilePath(filePath: string): string {
        if (path.isAbsolute(filePath)) {
            return filePath;
        }
        
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            throw new Error('No workspace folder open');
        }
        
        return path.join(workspaceFolder.uri.fsPath, filePath);
    }
    
    /**
     * Create multiple files at once
     */
    async createMultipleFiles(files: Array<{ path: string; content: string }>): Promise<{ success: boolean; message: string }> {
        const results: string[] = [];
        let successCount = 0;
        
        for (const file of files) {
            const result = await this.createFile(file.path, file.content);
            if (result.success) {
                successCount++;
                results.push(`✅ ${file.path}`);
            } else {
                results.push(`❌ ${file.path}: ${result.message}`);
            }
        }
        
        return {
            success: successCount === files.length,
            message: `Created ${successCount}/${files.length} files:\n${results.join('\n')}`
        };
    }
}
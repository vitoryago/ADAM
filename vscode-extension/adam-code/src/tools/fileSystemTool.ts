/**
 * File System Tool for ADAM
 * Provides file reading and workspace analysis capabilities
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class FileSystemTool {
    private maxFileSize = 1024 * 1024; // 1MB max for safety
    
    /**
     * Read a file from the workspace
     */
    async readFile(filePath: string): Promise<string> {
        try {
            // If relative path, resolve from workspace
            if (!path.isAbsolute(filePath)) {
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
                if (workspaceFolder) {
                    filePath = path.join(workspaceFolder.uri.fsPath, filePath);
                }
            }
            
            const fileUri = vscode.Uri.file(filePath);
            const fileData = await vscode.workspace.fs.readFile(fileUri);
            const content = Buffer.from(fileData).toString('utf8');
            
            if (content.length > this.maxFileSize) {
                return `File is too large (${(content.length / 1024).toFixed(2)}KB). Showing first 1000 lines:\n\n` + 
                       content.split('\n').slice(0, 1000).join('\n');
            }
            
            return content;
        } catch (error: any) {
            return `Error reading file: ${error.message}`;
        }
    }
    
    /**
     * List files in a directory
     */
    async listDirectory(dirPath: string): Promise<string[]> {
        try {
            if (!path.isAbsolute(dirPath)) {
                const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
                if (workspaceFolder) {
                    dirPath = path.join(workspaceFolder.uri.fsPath, dirPath);
                }
            }
            
            const dirUri = vscode.Uri.file(dirPath);
            const entries = await vscode.workspace.fs.readDirectory(dirUri);
            
            return entries.map(([name, type]) => {
                const typeStr = type === vscode.FileType.Directory ? '[DIR]' : '[FILE]';
                return `${typeStr} ${name}`;
            });
        } catch (error: any) {
            throw new Error(`Error listing directory: ${error.message}`);
        }
    }
    
    /**
     * Search for files matching a pattern
     */
    async searchFiles(pattern: string, maxResults: number = 20): Promise<string[]> {
        const files = await vscode.workspace.findFiles(pattern, null, maxResults);
        return files.map(file => {
            const workspaceFolder = vscode.workspace.getWorkspaceFolder(file);
            if (workspaceFolder) {
                return path.relative(workspaceFolder.uri.fsPath, file.fsPath);
            }
            return file.fsPath;
        });
    }
    
    /**
     * Get current workspace information
     */
    getWorkspaceInfo(): any {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders || workspaceFolders.length === 0) {
            return { error: 'No workspace folder open' };
        }
        
        return {
            folders: workspaceFolders.map(folder => ({
                name: folder.name,
                path: folder.uri.fsPath
            })),
            activeFile: vscode.window.activeTextEditor?.document.uri.fsPath || null
        };
    }
    
    /**
     * Read currently selected text or active file
     */
    async getActiveContext(): Promise<any> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return { error: 'No active editor' };
        }
        
        const document = editor.document;
        const selection = editor.selection;
        
        // Get selected text or full document
        const text = selection.isEmpty 
            ? document.getText()
            : document.getText(selection);
        
        return {
            file: path.relative(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '', document.uri.fsPath),
            language: document.languageId,
            selection: !selection.isEmpty,
            content: text.substring(0, 10000), // Limit to 10k chars
            lineCount: document.lineCount
        };
    }
}
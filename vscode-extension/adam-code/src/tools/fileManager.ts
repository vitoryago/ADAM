import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as path from 'path';

export class FileManager {
    async readFile(filePath: string): Promise<string> {
        try {
            const uri = vscode.Uri.file(filePath);
            const document = await vscode.workspace.openTextDocument(uri);
            return document.getText();
        } catch (error) {
            throw new Error(`Failed to read file: ${error}`);
        }
    }

    async writeFile(filePath: string, content: string): Promise<void> {
        try {
            const uri = vscode.Uri.file(filePath);
            const edit = new vscode.WorkspaceEdit();
            
            // Check if file exists
            try {
                await vscode.workspace.fs.stat(uri);
                // File exists, replace content
                const document = await vscode.workspace.openTextDocument(uri);
                const fullRange = new vscode.Range(
                    document.positionAt(0),
                    document.positionAt(document.getText().length)
                );
                edit.replace(uri, fullRange, content);
            } catch {
                // File doesn't exist, create it
                edit.createFile(uri, { contents: Buffer.from(content) });
            }
            
            await vscode.workspace.applyEdit(edit);
        } catch (error) {
            throw new Error(`Failed to write file: ${error}`);
        }
    }

    async createDirectory(dirPath: string): Promise<void> {
        try {
            await fs.mkdir(dirPath, { recursive: true });
        } catch (error) {
            throw new Error(`Failed to create directory: ${error}`);
        }
    }

    async listFiles(dirPath: string, pattern: string = '*'): Promise<string[]> {
        try {
            const files = await vscode.workspace.findFiles(
                new vscode.RelativePattern(dirPath, pattern)
            );
            return files.map(f => f.fsPath);
        } catch (error) {
            throw new Error(`Failed to list files: ${error}`);
        }
    }
}